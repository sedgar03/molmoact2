"""Live NEXT-lite external torque monitor for YAM tele-op.

The monitor is read-only. It either connects directly to a portal follower
server, or spawns ``stream_teleop_force.py`` over SSH and consumes JSON lines.
It does not command the robot.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np
import torch
import tyro

from gello_min.next_lite import load_next_lite_checkpoint, make_next_features
from record_teleop_force_log import PortalFollowerClient, _read_sample


@dataclass
class Args:
    checkpoint_path: str
    """NEXT-lite model.pt checkpoint."""

    host: str = "127.0.0.1"
    """Portal host for direct mode, or robot-side portal host for SSH mode."""

    port: int = 11333
    """Portal port."""

    hz: float = 30.0
    """Display/update rate."""

    ssh_host: str = ""
    """Optional SSH target for remote JSON streaming, e.g. steven@100.99.120.65."""

    remote_repo: str = "/home/steven/code/molmoact2"
    """Remote molmoact2 checkout when --ssh-host is used."""

    remote_python: str = "/home/steven/code/i2rt-official/.venv/bin/python"
    """Remote Python with portal installed when --ssh-host is used."""

    device: str = "auto"
    """Model device: auto, cpu, cuda, or mps."""

    window_sec: float = 20.0
    """Rolling plot window in seconds."""

    score_scale: float = 8.0
    """Score value mapped to full-scale display."""

    warn_score: float = 4.5
    """Residual L1 score shown as warning threshold."""

    contact_score: float = 7.5
    """Residual L1 score shown as contact threshold."""

    title: str = "YAM Live NEXT-lite External Torque"
    """cv2 window title."""


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
        sample = _read_sample(client)
        sample["timestamp"] = time.time()
        yield sample
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


def _sample_arrays(sample: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    q = np.asarray(sample["joint_positions"], dtype=np.float32).reshape(1, -1)
    qdot = np.asarray(sample["joint_velocities"], dtype=np.float32).reshape(1, -1)
    effort = np.asarray(sample["joint_efforts"], dtype=np.float32).reshape(-1)
    command = np.asarray(sample.get("commanded_joint_positions", q.reshape(-1)), dtype=np.float32).reshape(1, -1)
    ts = float(sample.get("timestamp", time.time()))
    return q, qdot, effort, command, ts


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


def _draw_plot(
    canvas: np.ndarray,
    values: deque[tuple[float, float]],
    now: float,
    window_sec: float,
    scale: float,
    warn: float,
    contact: float,
) -> None:
    h, w = canvas.shape[:2]
    left, top, right, bottom = 70, 90, w - 30, h - 170
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
    for score, color in ((warn, (0, 190, 255)), (contact, (80, 80, 255))):
        y = int(bottom - np.clip(score / scale, 0.0, 1.0) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), color, 1, cv2.LINE_AA)
    pts = []
    for ts, value in values:
        age = now - ts
        if age > window_sec:
            continue
        x = int(right - (age / window_sec) * (right - left))
        y = int(bottom - np.clip(value / scale, 0.0, 1.0) * (bottom - top))
        pts.append((x, y))
    if len(pts) > 1:
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"last {window_sec:.0f}s", (right - 120, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"scale {scale:.1f}", (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_bars(canvas: np.ndarray, residual: Optional[np.ndarray], scale: float) -> None:
    h, w = canvas.shape[:2]
    y0 = h - 115
    x0 = 70
    bar_w = max(48, (w - 120) // 7 - 10)
    for i in range(7):
        x = x0 + i * (bar_w + 10)
        cv2.rectangle(canvas, (x, y0), (x + bar_w, y0 + 70), (70, 70, 70), 1)
        value = 0.0 if residual is None else float(residual[i])
        fill = int(np.clip(abs(value) / max(scale / 2.0, 1e-6), 0.0, 1.0) * 68)
        color = (120, 230, 160) if abs(value) < scale * 0.25 else (0, 190, 255)
        if abs(value) >= scale * 0.45:
            color = (80, 80, 255)
        cv2.rectangle(canvas, (x + 1, y0 + 69 - fill), (x + bar_w - 1, y0 + 69), color, -1)
        cv2.putText(canvas, f"J{i}", (x + 6, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:+.2f}", (x + 2, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def _render(
    title: str,
    score: Optional[float],
    residual: Optional[np.ndarray],
    expected: Optional[np.ndarray],
    effort: np.ndarray,
    values: deque[tuple[float, float]],
    now: float,
    args: Args,
    ready: bool,
) -> np.ndarray:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = "WARMUP" if not ready else "FREE"
    color = (0, 200, 255) if not ready else (120, 230, 160)
    score_value = float(score) if score is not None and np.isfinite(score) else 0.0
    if ready and score_value >= args.warn_score:
        state, color = "PRE-CONTACT", (0, 190, 255)
    if ready and score_value >= args.contact_score:
        state, color = "CONTACT", (80, 80, 255)

    cv2.putText(canvas, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{state}   residual L1={score_value:.3f}   warn={args.warn_score:.2f} contact={args.contact_score:.2f}",
        (30, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    _draw_plot(canvas, values, now, args.window_sec, args.score_scale, args.warn_score, args.contact_score)
    _draw_bars(canvas, residual, args.score_scale)
    cv2.putText(canvas, "Per-joint external torque residual (measured effort - NEXT-lite expected effort)", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)
    raw = np.array2string(effort, precision=2, suppress_small=True)
    exp = "warming up" if expected is None else np.array2string(expected, precision=2, suppress_small=True)
    cv2.putText(canvas, f"measured effort: {raw}", (30, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"expected effort: {exp}", (30, 652), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "q to quit", (1160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = tyro.cli(Args)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")

    device = _choose_device(args.device)
    checkpoint = load_next_lite_checkpoint(checkpoint_path, map_location="cpu")
    history: deque[np.ndarray] = deque(maxlen=checkpoint.history)
    values: deque[tuple[float, float]] = deque(maxlen=max(1, int(args.window_sec * args.hz * 2)))
    samples = _ssh_samples(args) if args.ssh_host else _direct_samples(args.host, args.port, args.hz)

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, 1280, 720)
    try:
        for sample in samples:
            q, qdot, effort, command, ts = _sample_arrays(sample)
            feature = make_next_features(q, qdot, command)[0]
            history.append(feature)
            expected = _predict(checkpoint, history, device)
            residual = None if expected is None else effort - expected
            score = None if residual is None else float(np.linalg.norm(residual, ord=1))
            if score is not None:
                values.append((ts, score))
            canvas = _render(
                args.title,
                score,
                residual,
                expected,
                effort,
                values,
                ts,
                args,
                ready=expected is not None,
            )
            cv2.imshow(args.title, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyWindow(args.title)


if __name__ == "__main__":
    main()
