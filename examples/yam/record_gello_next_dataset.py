"""Record NEXT training data during YAM GELLO leader/follower tele-op.

Unlike ``record_teleop_force_log.py``, this script owns the leader loop and
therefore records the actual follower target command sent by the leader arm.
It is motion-capable: it commands the follower exactly like
``examples/minimum_gello/minimum_gello.py --mode leader``.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import h5py
import numpy as np
import tyro


DEFAULT_ROBOT_PORT = 11333


@dataclass
class Args:
    output_dir: str = "./yam_next_data_logs"
    """Directory where the timestamped HDF5 log will be written."""

    server_host: str = "127.0.0.1"
    """Host running the i2rt follower portal server."""

    server_port: int = DEFAULT_ROBOT_PORT
    """Portal port from minimum_gello.py follower mode."""

    leader_can_channel: str = "can_leader_l"
    """CAN channel for the leader arm."""

    arm: str = "yam"
    """Arm type passed to i2rt get_yam_robot."""

    leader_gripper: str = "yam_teaching_handle"
    """Leader gripper type."""

    bilateral_kp: float = 0.1
    """Leader feedback gain, matching minimum_gello.py."""

    leader_feedback_mode: Literal["mirror", "off"] = "mirror"
    """Leader haptic feedback mode. Use off to leave the leader passive/read-only."""

    hz: float = 100.0
    """Target logging and command rate."""

    duration_sec: float = 0.0
    """Synchronized recording duration in seconds. Use 0 to record until Ctrl-C."""

    ee_mass: Optional[float] = None
    """Optional i2rt end-effector mass override."""

    yes: bool = False
    """Start without an interactive confirmation prompt."""

    next_lite_checkpoint: str = ""
    """Optional NEXT-lite model.pt checkpoint for live command-aware residuals."""

    next_lite_device: str = "cpu"
    """Device for live NEXT-lite inference: cpu, cuda, mps, or auto."""

    force_stream_jsonl: str = ""
    """Optional JSONL path for live dashboard force samples."""

    force_stream_hz: float = 30.0
    """Maximum live force JSONL publish rate."""

    arm_label: str = "left"
    """Arm label written into live force samples."""


class PortalFollowerClient:
    def __init__(self, host: str, port: int) -> None:
        import portal

        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(), dtype=np.float32)

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(np.asarray(joint_pos, dtype=np.float32))

    def get_observations(self) -> Dict[str, Any]:
        return self._client.get_observations().result()


class YAMLeaderRobot:
    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> tuple[np.ndarray, list[bool]]:
        qpos = np.asarray(self._robot.get_observations()["joint_pos"], dtype=np.float32)
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1.0 - float(encoder_obs[0].position)
        qpos_with_gripper = np.concatenate([qpos, np.array([gripper_cmd], dtype=np.float32)])
        return qpos_with_gripper.astype(np.float32), encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


class NextLiteLiveEstimator:
    def __init__(self, checkpoint_path: str, device: str) -> None:
        from gello_min.next_lite import load_next_lite_checkpoint

        self._checkpoint = load_next_lite_checkpoint(Path(checkpoint_path), map_location="cpu")
        self._device = self._choose_device(device)
        self._history: deque[np.ndarray] = deque(maxlen=self._checkpoint.history)

    @property
    def history(self) -> int:
        return self._checkpoint.history

    @staticmethod
    def _choose_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def update(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        command: np.ndarray,
        effort: np.ndarray,
        predict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        from gello_min.next_lite import make_next_features

        features = make_next_features(
            np.asarray(q, dtype=np.float32).reshape(1, -1),
            np.asarray(qdot, dtype=np.float32).reshape(1, -1),
            np.asarray(command, dtype=np.float32).reshape(1, -1),
        )[0]
        self._history.append(features)
        if not predict:
            return None
        if len(self._history) < self._checkpoint.history:
            return {
                "ready": False,
                "expected_effort": None,
                "residual": None,
                "score_l1": None,
                "score_l2": None,
            }
        expected = self._checkpoint.predict_effort(
            np.stack(self._history).astype(np.float32),
            device=self._device,
        )
        residual = np.asarray(effort, dtype=np.float32) - expected
        return {
            "ready": True,
            "expected_effort": expected,
            "residual": residual,
            "score_l1": float(np.linalg.norm(residual, ord=1)),
            "score_l2": float(np.linalg.norm(residual, ord=2)),
        }


class ForceJsonlPublisher:
    def __init__(self, path: str, arm_label: str, hz: float) -> None:
        if hz <= 0:
            raise ValueError("--force-stream-hz must be positive")
        self.path = Path(path).expanduser()
        self.arm_label = arm_label
        self.period = 1.0 / float(hz)
        self._next_publish_time = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def maybe_publish(self, sample: Dict[str, Any], force: Optional[Dict[str, Any]]) -> None:
        now = time.monotonic()
        if now < self._next_publish_time:
            return
        self._next_publish_time = now + self.period
        record = {
            "arm": self.arm_label,
            "timestamp": float(sample["timestamp"]),
            "ready": bool(force and force.get("ready")),
            "source": "command_aware",
            "command_proxy": False,
            "phase_code": int(sample["phase_code"]),
            "joint_positions": np.asarray(sample["joint_positions"], dtype=np.float32).tolist(),
            "joint_velocities": np.asarray(sample["joint_velocities"], dtype=np.float32).tolist(),
            "joint_efforts": np.asarray(sample["joint_efforts"], dtype=np.float32).tolist(),
            "commanded_joint_positions": np.asarray(
                sample["commanded_joint_positions"], dtype=np.float32
            ).tolist(),
            "command_error_l2": float(np.linalg.norm(sample["command_error"])),
        }
        if force is not None:
            record["score_l1"] = force.get("score_l1")
            record["score_l2"] = force.get("score_l2")
            if force.get("expected_effort") is not None:
                record["expected_effort"] = np.asarray(
                    force["expected_effort"], dtype=np.float32
                ).tolist()
            if force.get("residual") is not None:
                record["residual"] = np.asarray(force["residual"], dtype=np.float32).tolist()
        with self.path.open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def should_publish(self) -> bool:
        return time.monotonic() >= self._next_publish_time


def _as_vector(value: Any, n: int, fill: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(n, fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full(n, fill, dtype=np.float32)
    out[: min(n, len(arr))] = arr[:n]
    return out


def _combine_arm_gripper(
    obs: Dict[str, Any],
    arm_names: tuple[str, ...],
    gripper_names: tuple[str, ...],
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    arm = None
    for name in arm_names:
        if name in obs:
            arm = np.asarray(obs[name], dtype=np.float32).reshape(-1)
            break
    if arm is None:
        if fallback is None:
            return np.full(7, np.nan, dtype=np.float32)
        return _as_vector(fallback, 7)
    if len(arm) >= 7:
        return _as_vector(arm, 7)
    gripper = None
    for name in gripper_names:
        if name in obs:
            g = np.asarray(obs[name], dtype=np.float32).reshape(-1)
            gripper = float(g[0]) if len(g) else 0.0
            break
    if gripper is None:
        gripper = 0.0
    return _as_vector(np.concatenate([arm, [gripper]]), 7)


def _read_follower_sample(
    follower: PortalFollowerClient,
    commanded: np.ndarray,
    commanded_velocity: np.ndarray,
    leader_q: np.ndarray,
    leader_buttons: list[bool],
    phase_code: int,
    loop_dt_sec: float,
    loop_lag_sec: float,
) -> Dict[str, Any]:
    read_start = time.monotonic()
    obs = follower.get_observations()
    follower_q_fallback = follower.get_joint_pos()
    read_latency_sec = time.monotonic() - read_start
    q = _combine_arm_gripper(
        obs,
        ("joint_positions", "joint_pos"),
        ("gripper_position", "gripper_pos"),
        fallback=follower_q_fallback,
    )
    qdot = _combine_arm_gripper(
        obs,
        ("joint_velocities", "joint_vel"),
        ("gripper_velocity", "gripper_vel"),
    )
    effort = _combine_arm_gripper(
        obs,
        ("joint_efforts", "joint_eff"),
        ("gripper_effort", "gripper_eff"),
    )
    temp_mos = _combine_arm_gripper(
        obs,
        ("temp_mos",),
        ("gripper_temp_mos",),
    )
    temp_rotor = _combine_arm_gripper(
        obs,
        ("temp_rotor",),
        ("gripper_temp_rotor",),
    )
    commanded = np.asarray(commanded, dtype=np.float32).copy()
    commanded_velocity = np.asarray(commanded_velocity, dtype=np.float32).copy()
    leader_q = np.asarray(leader_q, dtype=np.float32).copy()
    return {
        "timestamp": time.time(),
        "monotonic_time": time.monotonic(),
        "joint_positions": q,
        "joint_velocities": qdot,
        "joint_efforts": effort,
        "commanded_joint_positions": commanded,
        "commanded_joint_velocities": commanded_velocity,
        "command_error": commanded - q,
        "leader_joint_positions": leader_q,
        "leader_buttons": _as_vector(leader_buttons, 2, fill=0.0),
        "temp_mos": temp_mos,
        "temp_rotor": temp_rotor,
        "loop_dt_sec": float(loop_dt_sec),
        "loop_lag_sec": float(loop_lag_sec),
        "read_latency_sec": float(read_latency_sec),
        "phase_code": int(phase_code),
    }


def _empty_samples() -> Dict[str, List[Any]]:
    return {
        "timestamp": [],
        "monotonic_time": [],
        "joint_positions": [],
        "joint_velocities": [],
        "joint_efforts": [],
        "commanded_joint_positions": [],
        "commanded_joint_velocities": [],
        "command_error": [],
        "leader_joint_positions": [],
        "leader_buttons": [],
        "temp_mos": [],
        "temp_rotor": [],
        "loop_dt_sec": [],
        "loop_lag_sec": [],
        "read_latency_sec": [],
        "phase_code": [],
    }


def _append_sample(samples: Dict[str, List[Any]], sample: Dict[str, Any]) -> None:
    for key in samples:
        samples[key].append(sample[key])


def _write_h5(path: Path, samples: Dict[str, List[Any]], attrs: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, values in samples.items():
            if key in ("timestamp", "monotonic_time"):
                f.create_dataset(key, data=np.asarray(values, dtype=np.float64))
            elif key in ("loop_dt_sec", "loop_lag_sec", "read_latency_sec"):
                f.create_dataset(key, data=np.asarray(values, dtype=np.float32))
            elif key == "phase_code":
                f.create_dataset(key, data=np.asarray(values, dtype=np.int8))
            else:
                f.create_dataset(
                    key,
                    data=np.stack(values).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
        for key, value in attrs.items():
            f.attrs[key] = value


def _sleep_until(next_tick: float) -> float:
    while True:
        now = time.monotonic()
        if now >= next_tick:
            return now
        time.sleep(min(0.001, next_tick - now))


def main() -> None:
    args = tyro.cli(Args)
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if args.duration_sec < 0:
        raise SystemExit("--duration-sec must be >= 0")
    print("\nYAM GELLO NEXT data collector")
    print("  MOTION-CAPABLE: this script commands the follower from the leader arm.")
    print(f"  follower server: {args.server_host}:{args.server_port}")
    print(f"  leader CAN: {args.leader_can_channel}")
    print(f"  target hz: {args.hz}")
    print("  records true commanded_joint_positions from the leader loop")
    print("  records joint_efforts from DM driver torque feedback")
    if args.next_lite_checkpoint:
        print(f"  live NEXT-lite checkpoint: {args.next_lite_checkpoint}")
    if args.force_stream_jsonl:
        print(f"  live force JSONL: {args.force_stream_jsonl}")
    if not args.yes:
        input("Confirm workspace is clear and follower server is running. Press Enter to start, Ctrl-C to cancel.")

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.leader_gripper)
    leader_raw = get_yam_robot(
        channel=args.leader_can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=False,
    )
    leader = YAMLeaderRobot(leader_raw)
    leader_kp = leader_raw._kp
    if args.leader_feedback_mode == "off":
        leader.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
    follower = PortalFollowerClient(args.server_host, args.server_port)
    force_estimator = (
        NextLiteLiveEstimator(args.next_lite_checkpoint, args.next_lite_device)
        if args.next_lite_checkpoint
        else None
    )
    force_publisher = (
        ForceJsonlPublisher(args.force_stream_jsonl, args.arm_label, args.force_stream_hz)
        if args.force_stream_jsonl
        else None
    )

    current_leader_q, current_button = leader.get_info()
    current_follower_q = follower.get_joint_pos()
    print(f"Current leader joint pos: {np.array2string(current_leader_q, precision=4)}")
    print(f"Current follower joint pos: {np.array2string(current_follower_q, precision=4)}")
    print("Press the leader top button to synchronize/start; press again to stop tele-op.")

    samples = _empty_samples()
    period = 1.0 / float(args.hz)
    synchronized = False
    record_start: Optional[float] = None
    interrupted = False
    last_commanded_q: Optional[np.ndarray] = None
    last_record_time: Optional[float] = None

    def record_command(
        commanded: np.ndarray,
        leader_q: np.ndarray,
        leader_buttons: list[bool],
        phase_code: int,
        loop_dt_sec: float,
        loop_lag_sec: float,
    ) -> None:
        nonlocal last_commanded_q, last_record_time
        now_mono = time.monotonic()
        if last_commanded_q is None or last_record_time is None:
            commanded_velocity = np.zeros_like(commanded, dtype=np.float32)
        else:
            dt = max(1e-6, now_mono - last_record_time)
            commanded_velocity = (np.asarray(commanded, dtype=np.float32) - last_commanded_q) / dt
        follower.command_joint_pos(commanded)
        sample = _read_follower_sample(
            follower,
            commanded,
            commanded_velocity,
            leader_q,
            leader_buttons,
            phase_code,
            loop_dt_sec,
            loop_lag_sec,
        )
        _append_sample(samples, sample)
        force = None
        if force_estimator is not None:
            force = force_estimator.update(
                sample["joint_positions"],
                sample["joint_velocities"],
                sample["commanded_joint_positions"],
                sample["joint_efforts"],
                predict=force_publisher is not None and force_publisher.should_publish(),
            )
        if force_publisher is not None and force is not None:
            force_publisher.maybe_publish(sample, force)
        last_commanded_q = np.asarray(commanded, dtype=np.float32).copy()
        last_record_time = now_mono

    def slow_move(target_q: np.ndarray, duration: float = 1.0) -> None:
        nonlocal current_follower_q
        steps = max(1, int(round(duration * args.hz)))
        for i in range(steps):
            alpha = float(i + 1) / float(steps)
            command = target_q * alpha + current_follower_q * (1.0 - alpha)
            record_command(
                command,
                target_q,
                current_button,
                phase_code=1,
                loop_dt_sec=period,
                loop_lag_sec=0.0,
            )
            time.sleep(period)
        current_follower_q = follower.get_joint_pos()

    try:
        next_tick = time.monotonic()
        previous_tick = next_tick
        while True:
            stop_requested = False
            now = _sleep_until(next_tick)
            loop_dt_sec = now - previous_tick
            loop_lag_sec = max(0.0, now - next_tick)
            previous_tick = now
            next_tick = now + period

            current_leader_q, current_button = leader.get_info()
            button0 = bool(current_button[0])
            if button0:
                if not synchronized:
                    if args.leader_feedback_mode == "mirror":
                        leader.update_kp_kd(kp=leader_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                        leader.command_joint_pos(current_leader_q[:6])
                    else:
                        leader.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    slow_move(current_leader_q)
                    record_start = time.monotonic()
                    print("synchronized; recording tele-op")
                else:
                    print("stopping synchronized tele-op")
                    leader.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    if args.leader_feedback_mode == "mirror":
                        leader.command_joint_pos(current_follower_q[:6])
                    stop_requested = True
                synchronized = not synchronized
                while bool(current_button[0]):
                    time.sleep(0.01)
                    current_leader_q, current_button = leader.get_info()
                if stop_requested:
                    break
                next_tick = time.monotonic() + period
                continue

            current_follower_q = follower.get_joint_pos()
            if synchronized:
                commanded = current_leader_q.copy()
                record_command(
                    commanded,
                    current_leader_q,
                    current_button,
                    phase_code=2,
                    loop_dt_sec=loop_dt_sec,
                    loop_lag_sec=loop_lag_sec,
                )
                if args.leader_feedback_mode == "mirror":
                    leader.command_joint_pos(current_follower_q[:6])
                if (
                    args.duration_sec > 0
                    and record_start is not None
                    and time.monotonic() - record_start >= args.duration_sec
                ):
                    print("duration reached; stopping")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; writing partial log.")
    finally:
        leader.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)

    if len(samples["timestamp"]) == 0:
        raise SystemExit("No synchronized samples recorded")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / run_ts / "gello_next_log.h5"
    elapsed = float(samples["monotonic_time"][-1] - samples["monotonic_time"][0])
    actual_hz = float((len(samples["timestamp"]) - 1) / elapsed) if elapsed > 0 else 0.0
    attrs = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "yam_gello_next_log_v1",
        "server_host": args.server_host,
        "server_port": int(args.server_port),
        "leader_can_channel": args.leader_can_channel,
        "target_hz": float(args.hz),
        "actual_hz": actual_hz,
        "duration_sec": float(args.duration_sec),
        "interrupted": bool(interrupted),
        "commanded_source": "leader_joint_positions_sent_to_follower",
        "effort_source": "DM driver torque feedback from follower joint_eff",
        "effort_units": "Nm per DM motor torque frame scaling; verify absolute calibration per motor model",
        "phase_code_1": "initial sync interpolation",
        "phase_code_2": "synchronized leader/follower tele-op",
        "next_lite_checkpoint": str(args.next_lite_checkpoint),
        "force_stream_jsonl": str(args.force_stream_jsonl),
    }
    _write_h5(out_path, samples, attrs)
    print(f"Saved {len(samples['timestamp'])} samples over {elapsed:.2f}s ({actual_hz:.2f} Hz) to {out_path}")


if __name__ == "__main__":
    main()
