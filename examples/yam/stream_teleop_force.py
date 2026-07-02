"""Stream read-only YAM proprioception samples as JSON lines.

This script is intended to run on the robot PC where ``portal`` and the i2rt
runtime are installed. It connects to the existing ``minimum_gello`` follower
server and only calls read methods.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import tyro


DEFAULT_ROBOT_PORT = 11333


@dataclass
class Args:
    host: str = "127.0.0.1"
    """Host running the i2rt follower portal server."""

    port: int = DEFAULT_ROBOT_PORT
    """Portal port from minimum_gello.py follower mode."""

    hz: float = 50.0
    """Polling rate."""

    duration_sec: float = 0.0
    """Streaming duration in seconds. Use 0 to stream until Ctrl-C."""


class PortalFollowerClient:
    def __init__(self, host: str, port: int) -> None:
        import portal

        with redirect_stdout(sys.stderr):
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


def _sample(client: PortalFollowerClient) -> Dict[str, Any]:
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
        "timestamp": time.time(),
        "joint_positions": q.tolist(),
        "joint_velocities": qdot.tolist(),
        "joint_efforts": effort.tolist(),
        "commanded_joint_positions": q.tolist(),
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if args.duration_sec < 0:
        raise SystemExit("--duration-sec must be >= 0")

    client = PortalFollowerClient(args.host, args.port)
    period = 1.0 / float(args.hz)
    start = time.monotonic()
    next_tick = start
    while True:
        now = time.monotonic()
        if args.duration_sec > 0 and now - start >= args.duration_sec:
            break
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
            continue
        sys.stdout.write(json.dumps(_sample(client), separators=(",", ":")) + "\n")
        sys.stdout.flush()
        next_tick += period


if __name__ == "__main__":
    main()
