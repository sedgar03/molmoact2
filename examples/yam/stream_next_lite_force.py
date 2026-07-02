"""Stream live NEXT-lite residual samples as JSON lines.

This is read-only with respect to the robot. In SSH mode it spawns
stream_teleop_force.py on the robot host, consumes q/qdot/effort samples, and
computes the NEXT-lite expected free-space effort locally.
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
from stream_teleop_force import PortalFollowerClient, _sample


@dataclass
class Args:
    checkpoint_path: Annotated[str, tyro.conf.Positional]
    """NEXT-lite model.pt checkpoint."""

    arm: str = "left"
    """Arm label included in each JSON sample."""

    host: str = "127.0.0.1"
    """Portal host for direct mode, or robot-side portal host for SSH mode."""

    port: int = 11333
    """Portal port."""

    hz: float = 30.0
    """Polling rate."""

    ssh_host: str = ""
    """Optional SSH target, e.g. steven@100.99.120.65."""

    remote_repo: str = "/home/steven/code/molmoact2"
    """Remote molmoact2 checkout when --ssh-host is used."""

    remote_python: str = "/home/steven/code/i2rt-official/.venv/bin/python"
    """Remote Python with portal installed when --ssh-host is used."""

    device: str = "cpu"
    """Inference device: cpu, cuda, mps, or auto."""


def _choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _direct_samples(host: str, port: int, hz: float) -> Iterator[Dict[str, Any]]:
    client = PortalFollowerClient(host, port)
    period = 1.0 / float(hz)
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
            continue
        yield _sample(client)
        next_tick += period


def _ssh_samples(args: Args) -> Iterator[Dict[str, Any]]:
    cmd = [
        "ssh",
        args.ssh_host,
        (
            f"cd {args.remote_repo} && "
            f"{args.remote_python} -u examples/yam/stream_teleop_force.py "
            f"--host {args.host} --port {args.port} --hz {args.hz}"
        ),
    ]
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
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
                continue
            yield json.loads(line)
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
    effort = np.asarray(sample["joint_efforts"], dtype=np.float32).reshape(-1)
    command = np.asarray(
        sample.get("commanded_joint_positions", q.reshape(-1)),
        dtype=np.float32,
    ).reshape(1, -1)
    residual = None if expected is None else effort - expected
    record: Dict[str, Any] = {
        "arm": args.arm,
        "timestamp": float(sample.get("timestamp", time.time())),
        "ready": expected is not None,
        "joint_efforts": effort.tolist(),
        "joint_velocities": qdot.reshape(-1).tolist(),
        "command_error_l2": float(np.linalg.norm(command.reshape(-1) - q.reshape(-1))),
        "qdot_l2": float(np.linalg.norm(qdot.reshape(-1))),
        "command_proxy": bool(np.allclose(command.reshape(-1), q.reshape(-1), equal_nan=True)),
    }
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
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")

    checkpoint = load_next_lite_checkpoint(checkpoint_path, map_location="cpu")
    device = _choose_device(args.device)
    history: deque[np.ndarray] = deque(maxlen=checkpoint.history)
    samples = _ssh_samples(args) if args.ssh_host else _direct_samples(args.host, args.port, args.hz)
    for sample in samples:
        q = np.asarray(sample["joint_positions"], dtype=np.float32).reshape(1, -1)
        qdot = np.asarray(sample["joint_velocities"], dtype=np.float32).reshape(1, -1)
        command = np.asarray(
            sample.get("commanded_joint_positions", q.reshape(-1)),
            dtype=np.float32,
        ).reshape(1, -1)
        history.append(make_next_features(q, qdot, command)[0])
        _emit(args, sample, _predict(checkpoint, history, device))


if __name__ == "__main__":
    main()
