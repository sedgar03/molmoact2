"""Compute NEXT-lite residuals from command-aware JSONL samples.

This is intended for the dashboard path where the robot PC owns motion and
writes raw command-aware samples, while a nearby workstation with torch/model
dependencies computes the learned residual.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, Optional

import numpy as np
import torch
import tyro

from gello_min.next_lite import load_next_lite_checkpoint, make_next_features


@dataclass
class Args:
    checkpoint_path: Annotated[str, tyro.conf.Positional]
    """NEXT-lite model.pt checkpoint."""

    jsonl_path: str = "/tmp/yam_left_force.jsonl"
    """Local or robot-side command-aware JSONL path."""

    arm: str = "left"
    """Arm label included in each emitted sample."""

    ssh_host: str = ""
    """Optional SSH target. When set, tails --jsonl-path on that host."""

    device: str = "cpu"
    """Inference device: cpu, cuda, mps, or auto."""

    from_start: bool = False
    """Read the existing file from the start instead of tailing only new lines."""


def _choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _jsonl_lines(args: Args) -> Iterator[str]:
    tail_start = "+1" if args.from_start else "0"
    if args.ssh_host:
        cmd = ["ssh", args.ssh_host, f"tail -n {tail_start} -F {args.jsonl_path}"]
    else:
        cmd = ["tail", "-n", tail_start, "-F", args.jsonl_path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def _predict(checkpoint: Any, history: deque[np.ndarray], device: torch.device) -> Optional[np.ndarray]:
    if len(history) < checkpoint.history:
        return None
    x = np.stack(list(history), axis=0).astype(np.float32)
    x_norm = (x - checkpoint.x_mean) / checkpoint.x_std
    checkpoint.model.to(device)
    checkpoint.model.eval()
    with torch.no_grad():
        pred_norm = checkpoint.model(
            torch.as_tensor(x_norm[None, ...], dtype=torch.float32, device=device)
        )
    return (pred_norm.cpu().numpy()[0] * checkpoint.y_std + checkpoint.y_mean).astype(np.float32)


def _emit(args: Args, sample: Dict[str, Any], expected: Optional[np.ndarray]) -> None:
    q = np.asarray(sample["joint_positions"], dtype=np.float32).reshape(1, -1)
    qdot = np.asarray(sample["joint_velocities"], dtype=np.float32).reshape(1, -1)
    command = np.asarray(sample["commanded_joint_positions"], dtype=np.float32).reshape(1, -1)
    effort = np.asarray(sample["joint_efforts"], dtype=np.float32).reshape(-1)
    residual = None if expected is None else effort - expected
    record: Dict[str, Any] = {
        "arm": str(sample.get("arm", args.arm)),
        "timestamp": float(sample.get("timestamp", time.time())),
        "ready": residual is not None,
        "source": "command_aware_mac_next_lite",
        "command_proxy": False,
        "joint_efforts": effort.tolist(),
        "joint_velocities": qdot.reshape(-1).tolist(),
        "command_error_l2": float(np.linalg.norm(command.reshape(-1) - q.reshape(-1))),
        "qdot_l2": float(np.linalg.norm(qdot.reshape(-1))),
    }
    if "phase_code" in sample:
        record["phase_code"] = int(sample["phase_code"])
    if residual is not None:
        record["expected_effort"] = expected.tolist()
        record["residual"] = residual.tolist()
        record["score_l1"] = float(np.linalg.norm(residual, ord=1))
        record["score_l2"] = float(np.linalg.norm(residual, ord=2))
    else:
        record["score_l1"] = None
        record["score_l2"] = None
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    args = tyro.cli(Args)
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_next_lite_checkpoint(checkpoint_path, map_location="cpu")
    device = _choose_device(args.device)
    history: deque[np.ndarray] = deque(maxlen=checkpoint.history)
    for line in _jsonl_lines(args):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            q = np.asarray(sample["joint_positions"], dtype=np.float32).reshape(1, -1)
            qdot = np.asarray(sample["joint_velocities"], dtype=np.float32).reshape(1, -1)
            command = np.asarray(sample["commanded_joint_positions"], dtype=np.float32).reshape(1, -1)
        except KeyError as exc:
            print(f"missing key in JSONL sample: {exc}", file=sys.stderr)
            continue
        history.append(make_next_features(q, qdot, command)[0])
        _emit(args, sample, _predict(checkpoint, history, device))


if __name__ == "__main__":
    main()
