"""Passively record YAM force/proprioception during leader/follower tele-op.

This logger connects to the i2rt ``minimum_gello`` follower server over portal
and only calls read methods. It does not open the CAN bus and does not command
the robot. Use it while a human tele-operates through contact-free motions to
build NEXT/FACTR2-style free-space training data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import tyro


DEFAULT_ROBOT_PORT = 11333


@dataclass
class Args:
    output_dir: str = "./yam_teleop_force_logs"
    """Directory where the timestamped HDF5 log will be written."""

    host: str = "127.0.0.1"
    """Host running the i2rt follower portal server."""

    port: int = DEFAULT_ROBOT_PORT
    """Portal port from minimum_gello.py follower mode."""

    hz: float = 50.0
    """Polling rate."""

    duration_sec: float = 120.0
    """Recording duration in seconds. Use 0 to record until Ctrl-C."""

    commanded_source: str = "observed_joint_positions_proxy"
    """Label for commanded_joint_positions. Passive mode uses observed q as proxy."""

    yes: bool = False
    """Start without an interactive confirmation prompt."""


class PortalFollowerClient:
    def __init__(self, host: str, port: int) -> None:
        import portal

        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(), dtype=np.float32)

    def get_observations(self) -> Dict[str, Any]:
        return self._client.get_observations().result()


def _as_vector(value: Any, n: int, fill: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(n, fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full(n, fill, dtype=np.float32)
    out[: min(n, len(arr))] = arr[:n]
    return out


def _combine_arm_gripper(obs: Dict[str, Any], arm_names: tuple[str, ...], gripper_names: tuple[str, ...], fallback: Optional[np.ndarray] = None) -> np.ndarray:
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


def _read_sample(client: PortalFollowerClient) -> Dict[str, np.ndarray]:
    obs = client.get_observations()
    joint_pos_fallback = client.get_joint_pos()
    q = _combine_arm_gripper(
        obs,
        ("joint_positions", "joint_pos"),
        ("gripper_position", "gripper_pos"),
        fallback=joint_pos_fallback,
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
    return {
        "joint_positions": q,
        "joint_velocities": qdot,
        "joint_efforts": effort,
        # Passive portal logging cannot see the leader's exact target command.
        # Use q as an explicit proxy so train_next_lite.py can consume the log.
        "commanded_joint_positions": q.copy(),
    }


def _empty_samples() -> Dict[str, List[Any]]:
    return {
        "timestamp": [],
        "joint_positions": [],
        "joint_velocities": [],
        "joint_efforts": [],
        "commanded_joint_positions": [],
    }


def _write_h5(path: Path, samples: Dict[str, List[Any]], attrs: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, values in samples.items():
            if key == "timestamp":
                f.create_dataset(key, data=np.asarray(values, dtype=np.float64))
            else:
                f.create_dataset(
                    key,
                    data=np.stack(values).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
        for key, value in attrs.items():
            f.attrs[key] = value


def main() -> None:
    args = tyro.cli(Args)
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if args.duration_sec < 0:
        raise SystemExit("--duration_sec must be >= 0")

    print("\nPassive tele-op force logger")
    print(f"  follower server: {args.host}:{args.port}")
    print(f"  hz: {args.hz}")
    print(f"  duration_sec: {args.duration_sec or 'until Ctrl-C'}")
    print("  commands sent by this logger: none")
    print("  required operator behavior: contact-free leader/follower motion")
    print("  commanded_joint_positions: observed joint positions proxy")
    if not args.yes:
        input("Start follower server first. Press Enter to connect and record, or Ctrl-C to cancel.")

    client = PortalFollowerClient(args.host, args.port)
    # Fail early if the server is not responding.
    first = _read_sample(client)
    print(f"Connected. Initial q: {np.array2string(first['joint_positions'], precision=4)}")

    samples = _empty_samples()
    period = 1.0 / float(args.hz)
    start = time.monotonic()
    next_tick = start
    interrupted = False
    try:
        while True:
            now = time.monotonic()
            if args.duration_sec > 0 and now - start >= args.duration_sec:
                break
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            sample = _read_sample(client)
            samples["timestamp"].append(time.time())
            for key in ("joint_positions", "joint_velocities", "joint_efforts", "commanded_joint_positions"):
                samples[key].append(sample[key])
            next_tick += period
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; writing partial log.")

    if len(samples["timestamp"]) == 0:
        raise SystemExit("No samples recorded")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / run_ts / "teleop_force_log.h5"
    attrs = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "yam_teleop_force_log_v1",
        "host": args.host,
        "port": int(args.port),
        "hz": float(args.hz),
        "duration_sec": float(args.duration_sec),
        "interrupted": bool(interrupted),
        "commanded_source": args.commanded_source,
        "notes": (
            "Passive portal log. commanded_joint_positions uses observed joint "
            "positions as a proxy; integrate with teleop command path for exact targets."
        ),
    }
    _write_h5(out_path, samples, attrs)
    elapsed = samples["timestamp"][-1] - samples["timestamp"][0]
    print(f"Saved {len(samples['timestamp'])} samples over {elapsed:.2f}s to {out_path}")


if __name__ == "__main__":
    main()
