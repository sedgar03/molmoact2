"""Record a fixed-pose follower hold probe for NEXT/FACTR-style contact checks.

This script connects to an existing i2rt ``minimum_gello.py --mode follower``
portal server, captures the follower's current pose, and repeatedly commands
that same pose while logging measured q/qdot/effort. It is motion-capable only
in the sense that it keeps the follower in position hold at the current pose.
Use it for short, supervised, light-contact validation runs.
"""

from __future__ import annotations

import time
import subprocess
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
    output_dir: str = "./yam_next_data_logs/hold_contact_probe_left"
    """Directory where the timestamped HDF5 log will be written."""

    host: str = "127.0.0.1"
    """Host running the i2rt follower portal server."""

    port: int = DEFAULT_ROBOT_PORT
    """Portal port from minimum_gello.py follower mode."""

    hz: float = 100.0
    """Target command/logging rate."""

    duration_sec: float = 60.0
    """Probe duration in seconds."""

    start_delay_sec: float = 5.0
    """Delay before hold logging starts, so the operator can get ready."""

    cue_first_sec: float = 0.0
    """First cue time after recording starts. Use 0 to disable scheduled cues."""

    cue_period_sec: float = 0.0
    """Seconds between cues. Use 0 to disable scheduled cues."""

    cue_hold_sec: float = 2.0
    """Expected contact window duration after each cue."""

    cue_audio_command: str = ""
    """Optional shell command run at each cue, e.g. paplay a short sound."""

    yes: bool = False
    """Start without an interactive confirmation prompt."""


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


def _read_sample(
    follower: PortalFollowerClient,
    commanded: np.ndarray,
    loop_dt_sec: float,
    loop_lag_sec: float,
    cue_active: bool,
    cue_index: int,
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
    commanded = np.asarray(commanded, dtype=np.float32).copy()
    return {
        "timestamp": time.time(),
        "monotonic_time": time.monotonic(),
        "joint_positions": q,
        "joint_velocities": qdot,
        "joint_efforts": effort,
        "commanded_joint_positions": commanded,
        "commanded_joint_velocities": np.zeros_like(commanded, dtype=np.float32),
        "command_error": commanded - q,
        "loop_dt_sec": float(loop_dt_sec),
        "loop_lag_sec": float(loop_lag_sec),
        "read_latency_sec": float(read_latency_sec),
        "phase_code": 3,
        "cue_active": int(cue_active),
        "cue_index": int(cue_index),
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
        "loop_dt_sec": [],
        "loop_lag_sec": [],
        "read_latency_sec": [],
        "phase_code": [],
        "cue_active": [],
        "cue_index": [],
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
            elif key in ("phase_code", "cue_active"):
                f.create_dataset(key, data=np.asarray(values, dtype=np.int8))
            elif key == "cue_index":
                f.create_dataset(key, data=np.asarray(values, dtype=np.int16))
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


def _cue_state(elapsed: float, first: float, period: float, hold: float) -> tuple[bool, int]:
    if first <= 0.0 or period <= 0.0 or hold <= 0.0 or elapsed < first:
        return False, -1
    index = int((elapsed - first) // period)
    cue_start = first + index * period
    return elapsed < cue_start + hold, index


def _maybe_play_cue(command: str) -> None:
    print("\a", end="", flush=True)
    if not command:
        return
    try:
        subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"cue audio command failed: {exc}", flush=True)


def main() -> None:
    args = tyro.cli(Args)
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    if args.start_delay_sec < 0:
        raise SystemExit("--start-delay-sec must be >= 0")
    if args.cue_hold_sec < 0:
        raise SystemExit("--cue-hold-sec must be >= 0")

    print("\nYAM follower fixed-hold contact probe")
    print("  MOTION-CAPABLE: commands the follower to hold its current pose.")
    print(f"  follower server: {args.host}:{args.port}")
    print(f"  target hz: {args.hz}")
    print(f"  duration_sec: {args.duration_sec}")
    print(f"  start_delay_sec: {args.start_delay_sec}")
    if args.cue_first_sec > 0 and args.cue_period_sec > 0:
        print(
            "  cues: "
            f"first={args.cue_first_sec:.1f}s period={args.cue_period_sec:.1f}s "
            f"hold={args.cue_hold_sec:.1f}s"
        )
        if args.cue_audio_command:
            print(f"  cue_audio_command: {args.cue_audio_command}")
    print("  operator behavior: light pushes only; stop if anything moves unexpectedly.")
    if not args.yes:
        input("Confirm follower pose is safe. Press Enter to capture hold pose, Ctrl-C to cancel.")

    follower = PortalFollowerClient(args.host, args.port)
    hold_q = follower.get_joint_pos().astype(np.float32)
    follower.command_joint_pos(hold_q)
    print(f"Hold target q: {np.array2string(hold_q, precision=4)}")
    if args.start_delay_sec:
        print(f"Starting in {args.start_delay_sec:.1f}s...")
        time.sleep(args.start_delay_sec)

    samples = _empty_samples()
    period = 1.0 / float(args.hz)
    interrupted = False
    start = time.monotonic()
    next_tick = start
    previous_tick = start
    last_cue_index = -1
    try:
        while True:
            now = _sleep_until(next_tick)
            elapsed = now - start
            if elapsed >= args.duration_sec:
                print("duration reached; stopping")
                break
            cue_active, cue_index = _cue_state(
                elapsed,
                args.cue_first_sec,
                args.cue_period_sec,
                args.cue_hold_sec,
            )
            if cue_active and cue_index != last_cue_index:
                print(f"CUE {cue_index + 1}: contact for {args.cue_hold_sec:.1f}s", flush=True)
                _maybe_play_cue(args.cue_audio_command)
                last_cue_index = cue_index
            loop_dt_sec = now - previous_tick
            loop_lag_sec = max(0.0, now - next_tick)
            previous_tick = now
            next_tick = now + period
            follower.command_joint_pos(hold_q)
            _append_sample(
                samples,
                _read_sample(
                    follower,
                    hold_q,
                    loop_dt_sec,
                    loop_lag_sec,
                    cue_active,
                    cue_index if cue_active else -1,
                ),
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; writing partial log.")

    if len(samples["timestamp"]) == 0:
        raise SystemExit("No samples recorded")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / run_ts / "gello_next_log.h5"
    elapsed = float(samples["monotonic_time"][-1] - samples["monotonic_time"][0])
    actual_hz = float((len(samples["timestamp"]) - 1) / elapsed) if elapsed > 0 else 0.0
    attrs = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "yam_follower_hold_contact_probe_v1",
        "host": args.host,
        "port": int(args.port),
        "target_hz": float(args.hz),
        "actual_hz": actual_hz,
        "duration_sec": float(args.duration_sec),
        "interrupted": bool(interrupted),
        "commanded_source": "fixed_follower_hold_pose",
        "effort_source": "DM driver torque feedback from follower joint_eff",
        "phase_code_3": "fixed follower hold contact probe",
        "cue_first_sec": float(args.cue_first_sec),
        "cue_period_sec": float(args.cue_period_sec),
        "cue_hold_sec": float(args.cue_hold_sec),
        "cue_audio_command": str(args.cue_audio_command),
    }
    _write_h5(out_path, samples, attrs)
    print(f"Saved {len(samples['timestamp'])} samples over {elapsed:.2f}s ({actual_hz:.2f} Hz) to {out_path}")


if __name__ == "__main__":
    main()
