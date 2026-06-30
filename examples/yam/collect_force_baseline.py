"""Collect free-space YAM force/effort logs for safety calibration.

This script does not run policy inference or cameras. It commands small,
operator-supervised joint-space sweeps and writes per-control-tick HDF5 logs
for raw threshold tuning and NEXT-style free-space effort modeling.

Example:

    python examples/yam/collect_force_baseline.py \
        --left_config_path examples/yam/configs/yam_left.yaml \
        --output_dir ./yam_force_baselines \
        --skip_move_to_start \
        --joints 0 \
        --amplitude_rad 0.02 \
        --dry_run
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import tyro
from omegaconf import OmegaConf

from gello_min.env import RobotCommandResult, RobotEnv
from gello_min.launch_utils import instantiate_from_dict, move_to_start_position
from gello_min.robot import BimanualRobot


@dataclass
class Args:
    left_config_path: str = "examples/yam/configs/yam_left.yaml"
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Optional right arm configuration YAML file for bimanual collection."""

    output_dir: str = "./yam_force_baselines"
    """Directory where the timestamped baseline HDF5 run will be written."""

    joints: str = ""
    """Comma-separated joint indices to sweep, e.g. '0,1,2'. Use 'all' only in a fully clear workspace."""

    amplitude_rad: float = 0.04
    """Peak joint sweep amplitude around the current/start pose."""

    joint_amplitudes: str = ""
    """Optional per-joint amplitudes, e.g. '0:0.02,1:0.015'. Overrides --amplitude_rad for listed joints."""

    gripper_amplitude: float = 0.02
    """Peak gripper sweep amplitude when --include_gripper is enabled."""

    cycle_sec: float = 8.0
    """Seconds per sinusoidal cycle."""

    cycles_per_joint: int = 1
    """Number of slow/medium cycles to run for each joint."""

    hold_sec: float = 1.0
    """Seconds to hold the center pose before and after sweeps."""

    include_gripper: bool = False
    """Also sweep gripper joints. Leave false for first arm-effort baselines."""

    skip_move_to_start: bool = False
    """Use current pose as center without first moving to config start_joints."""

    dry_run: bool = False
    """Print the planned sweeps and exit before commanding motion."""

    yes: bool = False
    """Start without an interactive confirmation prompt."""


def _load_cfg(path: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _resolve_robot_cfg(robot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(robot_cfg)
    if isinstance(cfg.get("config"), str):
        cfg["config"] = _load_cfg(cfg["config"])
    return cfg


def _build_env(args: Args) -> Tuple[RobotEnv, Dict[str, Any], Optional[Dict[str, Any]], bool]:
    left_cfg = _load_cfg(args.left_config_path)
    right_cfg = _load_cfg(args.right_config_path) if args.right_config_path else None
    bimanual = right_cfg is not None

    left_robot = instantiate_from_dict(_resolve_robot_cfg(left_cfg["robot"]))
    if bimanual:
        right_robot = instantiate_from_dict(_resolve_robot_cfg(right_cfg["robot"]))
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    env = RobotEnv(
        robot,
        control_rate_hz=float(left_cfg.get("hz", 30)),
        force_safety=left_cfg.get("force_safety"),
    )
    return env, left_cfg, right_cfg, bimanual


def _joint_indices(num_dofs: int, include_gripper: bool, spec: str) -> List[int]:
    spec = spec.strip().lower()
    if not spec:
        raise SystemExit(
            "Refusing to run without explicit --joints. "
            "Start with a single known-safe joint, e.g. --joints 0."
        )
    if spec == "all":
        candidates = list(range(num_dofs))
    else:
        try:
            candidates = [int(part.strip()) for part in spec.split(",") if part.strip()]
        except ValueError as exc:
            raise SystemExit(f"Invalid --joints value: {spec}") from exc

    if not candidates:
        raise SystemExit("--joints did not contain any joint indices")

    result = []
    for idx in candidates:
        if idx < 0 or idx >= num_dofs:
            raise SystemExit(f"Joint index {idx} is outside 0..{num_dofs - 1}")
        if not include_gripper and num_dofs % 7 == 0 and idx % 7 == 6:
            raise SystemExit(
                f"Joint {idx} is a gripper joint. Pass --include_gripper to sweep it."
            )
        result.append(idx)
    return result


def _joint_amplitudes(num_dofs: int, args: Args) -> np.ndarray:
    amplitudes = np.full(num_dofs, float(args.amplitude_rad), dtype=float)
    if num_dofs % 7 == 0:
        amplitudes[6::7] = float(args.gripper_amplitude)
    if not args.joint_amplitudes.strip():
        return amplitudes

    for item in args.joint_amplitudes.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx_s, amp_s = item.split(":", 1)
            idx = int(idx_s)
            amp = float(amp_s)
        except ValueError as exc:
            raise SystemExit(
                "--joint_amplitudes entries must look like '0:0.02,1:0.015'"
            ) from exc
        if idx < 0 or idx >= num_dofs:
            raise SystemExit(f"Joint amplitude index {idx} is outside 0..{num_dofs - 1}")
        if amp < 0.0:
            raise SystemExit("Joint amplitudes must be non-negative")
        amplitudes[idx] = amp
    return amplitudes


def _target_for_joint(
    center: np.ndarray,
    joint_idx: int,
    phase: float,
    amplitudes: np.ndarray,
) -> np.ndarray:
    target = center.copy()
    amp = float(amplitudes[joint_idx])
    target[joint_idx] = center[joint_idx] + amp * np.sin(phase)
    return target


def _empty_samples() -> Dict[str, List[Any]]:
    return {
        "timestamp": [],
        "phase_index": [],
        "joint_index": [],
        "target_joint_positions": [],
        "requested_joint_positions": [],
        "commanded_joint_positions": [],
        "command_delta": [],
        "joint_positions": [],
        "joint_velocities": [],
        "joint_efforts": [],
        "next_joint_positions": [],
        "next_joint_velocities": [],
        "next_joint_efforts": [],
    }


def _obs_array(obs: Dict[str, Any], key: str, num_dofs: int) -> np.ndarray:
    if key not in obs:
        return np.full(num_dofs, np.nan, dtype=np.float32)
    arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if arr.shape == (num_dofs,):
        return arr.copy()
    out = np.full(num_dofs, np.nan, dtype=np.float32)
    out[: min(num_dofs, len(arr))] = arr[:num_dofs]
    return out


def _record_sample(
    samples: Dict[str, List[Any]],
    target: np.ndarray,
    result: RobotCommandResult,
    obs_post: Dict[str, Any],
    phase_index: int,
    joint_index: int,
) -> None:
    num_dofs = len(target)
    before = np.asarray(result.observed_joint_positions, dtype=np.float32)
    before_vel = np.asarray(result.observed_joint_velocities, dtype=np.float32)
    before_eff = (
        np.asarray(result.observed_joint_efforts, dtype=np.float32)
        if result.observed_joint_efforts is not None
        else np.full(num_dofs, np.nan, dtype=np.float32)
    )
    sent = np.asarray(result.sent_command, dtype=np.float32)

    samples["timestamp"].append(float(result.timestamp))
    samples["phase_index"].append(int(phase_index))
    samples["joint_index"].append(int(joint_index))
    samples["target_joint_positions"].append(np.asarray(target, dtype=np.float32).copy())
    samples["requested_joint_positions"].append(
        np.asarray(result.requested_command, dtype=np.float32).copy()
    )
    samples["commanded_joint_positions"].append(sent.copy())
    samples["command_delta"].append((sent - before).astype(np.float32))
    samples["joint_positions"].append(before.copy())
    samples["joint_velocities"].append(before_vel.copy())
    samples["joint_efforts"].append(before_eff.copy())
    samples["next_joint_positions"].append(_obs_array(obs_post, "joint_positions", num_dofs))
    samples["next_joint_velocities"].append(_obs_array(obs_post, "joint_velocities", num_dofs))
    samples["next_joint_efforts"].append(_obs_array(obs_post, "joint_efforts", num_dofs))


def _hold_pose(
    env: RobotEnv,
    samples: Dict[str, List[Any]],
    target: np.ndarray,
    ticks: int,
    phase_index: int,
) -> None:
    for _ in range(ticks):
        result = env.step_command_only(target)
        obs_post = env.get_robot_state()
        _record_sample(samples, target, result, obs_post, phase_index, joint_index=-1)


def _write_h5(
    path: Path,
    samples: Dict[str, List[Any]],
    attrs: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, values in samples.items():
            if key in ("timestamp",):
                data = np.asarray(values, dtype=np.float64)
            elif key in ("phase_index", "joint_index"):
                data = np.asarray(values, dtype=np.int32)
            else:
                data = np.stack(values).astype(np.float32)
            compression = "gzip" if data.ndim > 1 else None
            if compression is None:
                f.create_dataset(key, data=data)
            else:
                f.create_dataset(key, data=data, compression=compression, compression_opts=4)
        for key, value in attrs.items():
            f.attrs[key] = value


def main() -> None:
    args = tyro.cli(Args)
    if not args.joints.strip():
        raise SystemExit(
            "Refusing to run without explicit --joints. "
            "Start with a single known-safe joint, e.g. --joints 0."
        )

    env, left_cfg, right_cfg, bimanual = _build_env(args)
    hz = float(left_cfg.get("hz", 30))
    num_dofs = env.robot().num_dofs()
    joint_indices = _joint_indices(num_dofs, args.include_gripper, args.joints)
    amplitudes = _joint_amplitudes(num_dofs, args)

    if not args.skip_move_to_start and not args.dry_run:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)

    center = np.asarray(env.get_robot_state()["joint_positions"], dtype=float)
    print("\nFree-space force baseline collection")
    print(f"  dofs: {num_dofs}")
    print(f"  joints swept: {joint_indices}")
    print(f"  center: {np.array2string(center, precision=4)}")
    print(
        "  amplitudes: "
        + ", ".join(f"j{idx}={amplitudes[idx]:.4f}" for idx in joint_indices)
    )
    print(f"  cycle_sec: {args.cycle_sec}")
    print(f"  cycles_per_joint: {args.cycles_per_joint}")
    if not args.skip_move_to_start:
        print("  pre-run move: configured start_joints")
    print("\nThis is autonomous scripted motion, not tele-op.")
    print("Keep the planned swept volumes clear. This run must remain contact-free.")
    if args.dry_run:
        print("\nDry run only; no robot commands were sent.")
        return
    if not args.yes:
        input("Press Enter to begin, or Ctrl-C to cancel.")

    samples = _empty_samples()
    hold_ticks = max(1, int(round(args.hold_sec * hz)))
    cycle_ticks = max(4, int(round(args.cycle_sec * hz)))

    _hold_pose(env, samples, center, hold_ticks, phase_index=0)
    phase_index = 1
    for joint_idx in joint_indices:
        print(f"[baseline] sweeping joint {joint_idx}")
        for cycle in range(args.cycles_per_joint):
            for tick in range(cycle_ticks):
                phase = 2.0 * np.pi * (cycle + tick / cycle_ticks)
                target = _target_for_joint(
                    center,
                    joint_idx,
                    phase,
                    amplitudes,
                )
                result = env.step_command_only(target)
                obs_post = env.get_robot_state()
                _record_sample(samples, target, result, obs_post, phase_index, joint_idx)
        phase_index += 1
        _hold_pose(env, samples, center, hold_ticks, phase_index=phase_index)
        phase_index += 1

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / run_ts / "free_space_baseline.h5"
    attrs = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "yam_force_baseline_v1",
        "hz": hz,
        "num_dofs": num_dofs,
        "amplitude_rad": float(args.amplitude_rad),
        "gripper_amplitude": float(args.gripper_amplitude),
        "joints": args.joints,
        "joint_amplitudes": args.joint_amplitudes,
        "cycle_sec": float(args.cycle_sec),
        "cycles_per_joint": int(args.cycles_per_joint),
        "include_gripper": bool(args.include_gripper),
        "left_config": json.dumps(left_cfg),
        "right_config": json.dumps(right_cfg) if right_cfg is not None else "",
    }
    _write_h5(out_path, samples, attrs)
    print(f"\nSaved {len(samples['timestamp'])} samples to {out_path}")


if __name__ == "__main__":
    main()
