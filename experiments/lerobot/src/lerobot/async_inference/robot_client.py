# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import csv
import math
import os
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            policy_kwargs=config.policy_kwargs,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        # 3 threads: action receiver, action playback loop, observation sender.
        self.start_barrier = threading.Barrier(3)
        self.robot_io_lock = threading.RLock()
        self.observation_request_lock = threading.Lock()
        self._observation_request_pending = False

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self._so101_molmoact2_frame_transform = (
            os.environ.get("ROBOT_LAB_SO101_MOLMOACT2_FRAME_TRANSFORM", "1").lower()
            not in {"0", "false", "no"}
            and config.policy_type == "molmoact2"
            and getattr(config.robot, "type", "") in {"so100_follower", "so101_follower"}
        )
        self._so101_motor_order = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self._so101_joint_offsets = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 90.0,
            "elbow_flex": 90.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self._so101_joint_signs = {
            "shoulder_pan": 1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_roll": 1.0,
            "gripper": 1.0,
        }
        if self._so101_molmoact2_frame_transform:
            self.logger.info(
                "SO100/101 MolmoAct2 frame transform enabled: "
                "state_model=sign*arm_state+offset, action_arm=(model_action-offset)*sign."
            )
        self.action_log_path = os.environ.get("ROBOT_LAB_SO101_ACTION_LOG")
        self.action_log_telemetry = os.environ.get("ROBOT_LAB_SO101_ACTION_TELEMETRY", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._so101_watch_motor = os.environ.get("ROBOT_LAB_SO101_WATCH_MOTOR", "elbow_flex")
        self._so101_watch_enabled = (
            os.environ.get("ROBOT_LAB_SO101_WATCHDOG", "1").lower() not in {"0", "false", "no"}
            and getattr(config.robot, "type", "") in {"so100_follower", "so101_follower"}
        )
        self._so101_watch_max_error_ticks = float(os.environ.get("ROBOT_LAB_SO101_WATCH_MAX_ERROR_TICKS", "45"))
        self._so101_watch_max_current = float(os.environ.get("ROBOT_LAB_SO101_WATCH_MAX_CURRENT", "130"))
        self._so101_watch_max_abs_load = float(os.environ.get("ROBOT_LAB_SO101_WATCH_MAX_ABS_LOAD", "300"))
        self._so101_watch_frames = int(os.environ.get("ROBOT_LAB_SO101_WATCH_FRAMES", "4"))
        self._so101_watch_count = 0
        if self._so101_watch_enabled:
            self.logger.info(
                "SO101 watchdog enabled: motor=%s max_error_ticks=%.1f max_current=%.1f "
                "max_abs_load=%.1f frames=%d",
                self._so101_watch_motor,
                self._so101_watch_max_error_ticks,
                self._so101_watch_max_current,
                self._so101_watch_max_abs_load,
                self._so101_watch_frames,
            )
        self._so101_action_interpolation_enabled = (
            self._env_flag("ROBOT_LAB_SO101_ACTION_INTERPOLATION", True)
            and getattr(config.robot, "type", "") in {"so100_follower", "so101_follower"}
        )
        self._so101_action_interpolation_hz = max(
            1.0, self._env_float("ROBOT_LAB_SO101_ACTION_INTERPOLATION_HZ", 60.0)
        )
        default_interpolation_window_sec = 1.0 / max(float(config.fps), 1.0)
        self._so101_action_interpolation_window_sec = max(
            1.0 / self._so101_action_interpolation_hz,
            self._env_float("ROBOT_LAB_SO101_ACTION_INTERPOLATION_WINDOW_SEC", default_interpolation_window_sec),
        )
        self._so101_default_speed_limit = self._env_float("ROBOT_LAB_SO101_MAX_SPEED_DEG_PER_SEC", 45.0)
        self._so101_speed_limits = {}
        for motor in self._so101_motor_order:
            if motor == "gripper":
                env_key = "ROBOT_LAB_SO101_GRIPPER_MAX_SPEED_PER_SEC"
                default_speed = 80.0
            else:
                env_key = f"ROBOT_LAB_SO101_{motor.upper()}_MAX_SPEED_DEG_PER_SEC"
                default_speed = 18.0 if motor == "elbow_flex" else self._so101_default_speed_limit
            self._so101_speed_limits[motor] = max(0.1, self._env_float(env_key, default_speed))
        if self._so101_action_interpolation_enabled:
            self.logger.info(
                "SO101 action interpolation enabled: hz=%.1f window_sec=%.4f speed_limits=%s",
                self._so101_action_interpolation_hz,
                self._so101_action_interpolation_window_sec,
                self._so101_speed_limits,
            )
        self._action_log_lock = threading.Lock()
        if self.action_log_path:
            os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
            with open(self.action_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "wall_time",
                        "timestep",
                        "motor",
                        "raw_policy_goal",
                        "sent_goal",
                        "present_before",
                        "present_after",
                        "goal_after",
                        "load_after",
                        "current_after",
                        "moving_after",
                        "status_after",
                        "torque_after",
                    ],
                )
                writer.writeheader()
            self.logger.info(f"Robot action log enabled: {self.action_log_path}")

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        with self.robot_io_lock:
            self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!
                with self.observation_request_lock:
                    self._observation_request_pending = False

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _env_flag(self, name: str, default: bool) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        return raw_value.lower() not in {"0", "false", "no", "off"}

    def _env_float(self, name: str, default: float) -> float:
        raw_value = os.environ.get(name)
        if raw_value is None or raw_value == "":
            return default
        try:
            return float(raw_value)
        except ValueError:
            self.logger.warning("Ignoring invalid float env %s=%r; using %.3f", name, raw_value, default)
            return default

    def _so101_arm_to_model_value(self, motor: str, value: float) -> float:
        return self._so101_joint_signs[motor] * value + self._so101_joint_offsets[motor]

    def _so101_model_to_arm_value(self, motor: str, value: float) -> float:
        return (value - self._so101_joint_offsets[motor]) * self._so101_joint_signs[motor]

    def _transform_so101_observation_to_model_frame(self, observation: RawObservation) -> RawObservation:
        if not self._so101_molmoact2_frame_transform:
            return observation
        transformed = dict(observation)
        for motor in self._so101_motor_order:
            key = f"{motor}.pos"
            if key in transformed:
                transformed[key] = self._so101_arm_to_model_value(motor, float(transformed[key]))
        return transformed

    def _transform_so101_action_to_arm_frame(self, action: dict[str, float]) -> dict[str, float]:
        if not self._so101_molmoact2_frame_transform:
            return action
        transformed = dict(action)
        for motor in self._so101_motor_order:
            key = f"{motor}.pos"
            if key in transformed:
                transformed[key] = self._so101_model_to_arm_value(motor, float(transformed[key]))
        return transformed

    def _read_so101_present_action(self) -> dict[str, float] | None:
        bus = getattr(self.robot, "bus", None)
        if bus is None:
            return None
        try:
            present_positions = bus.sync_read("Present_Position")
        except Exception as exc:
            self.logger.warning("SO101 interpolation disabled for this action: present read failed (%s)", exc)
            return None
        return {f"{motor}.pos": float(value) for motor, value in present_positions.items()}

    def _limit_so101_target_by_speed(
        self,
        present_action: dict[str, float],
        target_action: dict[str, float],
    ) -> tuple[dict[str, float], float, list[str]]:
        scale = 1.0
        constrained_motors = []
        deltas = {}
        for motor in self._so101_motor_order:
            key = f"{motor}.pos"
            if key not in target_action or key not in present_action:
                continue
            present = present_action[key]
            target = float(target_action[key])
            delta = target - present
            deltas[motor] = delta
            speed_limit = self._so101_speed_limits[motor]
            max_delta = speed_limit * self._so101_action_interpolation_window_sec
            if abs(delta) > max_delta:
                scale = min(scale, max_delta / abs(delta))
                constrained_motors.append(motor)

        limited_action = dict(target_action)
        required_duration_sec = 0.0
        for motor, delta in deltas.items():
            key = f"{motor}.pos"
            limited_delta = delta * scale
            limited_action[key] = present_action[key] + limited_delta
            required_duration_sec = max(required_duration_sec, abs(limited_delta) / self._so101_speed_limits[motor])
        return limited_action, required_duration_sec, constrained_motors

    def _send_so101_interpolated_action(self, target_action: dict[str, float]) -> dict[str, float]:
        if not self._so101_action_interpolation_enabled:
            with self.robot_io_lock:
                return self.robot.send_action(target_action)

        with self.robot_io_lock:
            present_action = self._read_so101_present_action()
        if present_action is None:
            with self.robot_io_lock:
                return self.robot.send_action(target_action)

        limited_action, duration_sec, limited_motors = self._limit_so101_target_by_speed(
            present_action, target_action
        )
        if limited_motors:
            self.logger.debug(
                "SO101 speed-limited action over %.4fs: %s",
                self._so101_action_interpolation_window_sec,
                ", ".join(limited_motors),
            )

        if duration_sec <= 0:
            return self.robot.send_action(limited_action)

        steps = max(1, math.ceil(duration_sec * self._so101_action_interpolation_hz))
        if steps <= 1:
            with self.robot_io_lock:
                return self.robot.send_action(limited_action)

        sleep_sec = 1.0 / self._so101_action_interpolation_hz
        performed_action = limited_action
        for step in range(1, steps + 1):
            alpha = step / steps
            substep_action = dict(limited_action)
            for motor in self._so101_motor_order:
                key = f"{motor}.pos"
                if key in limited_action and key in present_action:
                    substep_action[key] = present_action[key] + (float(limited_action[key]) - present_action[key]) * alpha
            with self.robot_io_lock:
                performed_action = self.robot.send_action(substep_action)
            if step < steps:
                time.sleep(sleep_sec)
        return performed_action

    def _read_action_telemetry(self) -> dict[str, dict[str, Any]]:
        if not self.action_log_path or not self.action_log_telemetry:
            return {}
        bus = getattr(self.robot, "bus", None)
        if bus is None:
            return {}

        telemetry: dict[str, dict[str, Any]] = {}
        with self.robot_io_lock:
            for column, register in {
                "present": "Present_Position",
                "goal": "Goal_Position",
                "load": "Present_Load",
                "current": "Present_Current",
                "moving": "Moving",
                "status": "Status",
                "torque": "Torque_Enable",
            }.items():
                try:
                    values = bus.sync_read(register, normalize=False, num_retry=1)
                except Exception as exc:
                    values = {motor: f"err:{type(exc).__name__}" for motor in bus.motors}
                for motor, value in values.items():
                    telemetry.setdefault(motor, {})[column] = value
        return telemetry

    def _log_action(
        self,
        timestep: int,
        raw_action: dict[str, float],
        sent_action: dict[str, float],
        telemetry_before: dict[str, dict[str, Any]] | None = None,
        telemetry_after: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not self.action_log_path:
            return
        rows = []
        now = time.time()
        telemetry_before = telemetry_before or {}
        telemetry_after = telemetry_after or {}
        for key in raw_action:
            motor = key.removesuffix(".pos")
            before = telemetry_before.get(motor, {})
            after = telemetry_after.get(motor, {})
            rows.append(
                {
                    "wall_time": f"{now:.6f}",
                    "timestep": timestep,
                    "motor": motor,
                    "raw_policy_goal": raw_action.get(key),
                    "sent_goal": sent_action.get(key),
                    "present_before": before.get("present", ""),
                    "present_after": after.get("present", ""),
                    "goal_after": after.get("goal", ""),
                    "load_after": after.get("load", ""),
                    "current_after": after.get("current", ""),
                    "moving_after": after.get("moving", ""),
                    "status_after": after.get("status", ""),
                    "torque_after": after.get("torque", ""),
                }
            )
        with self._action_log_lock:
            with open(self.action_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "wall_time",
                        "timestep",
                        "motor",
                        "raw_policy_goal",
                        "sent_goal",
                        "present_before",
                        "present_after",
                        "goal_after",
                        "load_after",
                        "current_after",
                        "moving_after",
                        "status_after",
                        "torque_after",
                    ],
                )
                writer.writerows(rows)

    def _as_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and (value == "" or value.startswith("err:")):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _check_so101_watchdog(self, telemetry_after: dict[str, dict[str, Any]]) -> None:
        if not self._so101_watch_enabled:
            return
        after = telemetry_after.get(self._so101_watch_motor, {})
        present = self._as_float(after.get("present"))
        goal = self._as_float(after.get("goal"))
        current = self._as_float(after.get("current"))
        load = self._as_float(after.get("load"))

        reasons = []
        if present is not None and goal is not None:
            error = abs(goal - present)
            if error >= self._so101_watch_max_error_ticks:
                reasons.append(f"goal-present error {error:.0f} ticks")
        if current is not None and abs(current) >= self._so101_watch_max_current:
            reasons.append(f"current {current:.0f}")
        if load is not None and abs(load) >= self._so101_watch_max_abs_load:
            reasons.append(f"load {load:.0f}")

        if reasons:
            self._so101_watch_count += 1
        else:
            self._so101_watch_count = 0

        if reasons:
            self.logger.warning(
                "SO101 watchdog sample %d/%d on %s: %s",
                self._so101_watch_count,
                self._so101_watch_frames,
                self._so101_watch_motor,
                ", ".join(reasons),
            )

        if self._so101_watch_count >= self._so101_watch_frames:
            raise RuntimeError(
                f"SO101 watchdog abort on {self._so101_watch_motor}: {', '.join(reasons)} "
                f"for {self._so101_watch_count} consecutive samples"
            )

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        raw_action = self._action_tensor_to_action_dict(timed_action.get_action())
        telemetry_before = self._read_action_telemetry()
        action_to_robot = self._transform_so101_action_to_arm_frame(raw_action)
        _performed_action = self._send_so101_interpolated_action(action_to_robot)
        telemetry_after = self._read_action_telemetry()
        self._log_action(timed_action.get_timestep(), raw_action, _performed_action, telemetry_before, telemetry_after)
        self._check_so101_watchdog(telemetry_after)
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        if self.action_log_path:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.info(f"Applied action #{timed_action.get_timestep()} | Queue size: {current_queue_size}")

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.observation_request_lock:
            if self._observation_request_pending:
                return False
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def _claim_observation_request(self) -> bool:
        """Atomically reserve the next observation request slot."""
        with self.observation_request_lock:
            if self._observation_request_pending:
                return False
            self._observation_request_pending = True
            return True

    def _release_observation_request(self) -> None:
        with self.observation_request_lock:
            self._observation_request_pending = False

    def _get_observation_without_blocking_actions(self) -> RawObservation:
        """Read robot observation while keeping the action loop blocked briefly.

        SO follower observations combine a motor bus read with camera reads. The
        bus read must not overlap action writes, but camera reads and network
        upload can proceed without holding the robot I/O lock.
        """
        bus = getattr(self.robot, "bus", None)
        cameras = getattr(self.robot, "cameras", None)
        if bus is None or cameras is None:
            with self.robot_io_lock:
                return self.robot.get_observation()

        start = time.perf_counter()
        with self.robot_io_lock:
            obs_dict = bus.sync_read("Present_Position")
        raw_observation: RawObservation = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        self.logger.debug("%s read state: %.1fms", self.robot, (time.perf_counter() - start) * 1e3)

        for cam_key, cam in cameras.items():
            start = time.perf_counter()
            raw_observation[cam_key] = cam.read_latest()
            self.logger.debug("%s read %s: %.1fms", self.robot, cam_key, (time.perf_counter() - start) * 1e3)

        return raw_observation

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        if not self._claim_observation_request():
            return None

        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            self.logger.info("Capturing observation")
            raw_observation = self._get_observation_without_blocking_actions()
            raw_observation = self._transform_so101_observation_to_model_frame(raw_observation)
            raw_observation["task"] = task
            self.logger.info("Captured observation")

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            self.logger.info(f"Sending observation #{observation.get_timestep()} (must_go={observation.must_go})")
            sent = self.send_observation(observation)
            self.logger.info(f"Sent observation #{observation.get_timestep()}")
            if not sent:
                self._release_observation_request()

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            self._release_observation_request()

    def observation_loop(self, task: str, verbose: bool = False) -> RawObservation:
        """Capture and upload observations without blocking action playback."""
        self.start_barrier.wait()
        self.logger.info("Observation loop thread starting")

        _captured_observation = None
        while self.running:
            loop_start = time.perf_counter()
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - loop_start)))

        return _captured_observation

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Fixed-rate action playback loop."""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return None, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver and observation threads...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        observation_thread = threading.Thread(target=client.observation_loop, args=(cfg.task,), daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()
        observation_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            observation_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
