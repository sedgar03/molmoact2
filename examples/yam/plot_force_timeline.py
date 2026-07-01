"""Plot a FACTR2-style scalar force/contact timeline from a YAM HDF5 log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    h5_path: Annotated[str, tyro.conf.Positional]
    """Path to episode.h5 or free_space_baseline.h5."""

    output_path: str = ""
    """Output PNG path. Defaults to force_timeline.png next to the HDF5 file."""

    fps: float = 30.0
    """Fallback sample rate when the HDF5 log has no timestamp dataset."""

    high_threshold: float = 0.45
    """Hysteresis high threshold on the plotted normalized score."""

    low_threshold: float = 0.20
    """Hysteresis low threshold on the plotted normalized score."""

    pre_contact_sec: float = 1.0
    """Seconds before contact onset to shade as pre-contact."""

    normalize: bool = True
    """Normalize the scalar score by its max for FACTR2-style display."""

    title: str = "YAM NEXT-lite Contact Score"
    """Plot title."""


def _load_time_and_score(path: Path, fps: float) -> Tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as f:
        if "timestamp" in f:
            ts = np.asarray(f["timestamp"][:], dtype=np.float64)
            t = ts - ts[0]
        else:
            n = _infer_length(f)
            t = np.arange(n, dtype=np.float64) / float(fps)

        if "force_contact_score" in f:
            score = np.asarray(f["force_contact_score"][:], dtype=np.float64)
            source = "force_contact_score"
        elif "force_residual" in f:
            residual = np.asarray(f["force_residual"][:], dtype=np.float64)
            score = np.linalg.norm(residual, ord=1, axis=1)
            source = "L1(force_residual)"
        elif "joint_efforts" in f:
            effort = np.asarray(f["joint_efforts"][:], dtype=np.float64)
            score = np.linalg.norm(effort, ord=1, axis=1)
            source = "L1(joint_efforts)"
        else:
            raise SystemExit(
                f"{path} has no force_contact_score, force_residual, or joint_efforts"
            )

    n = min(len(t), len(score))
    return t[:n], score[:n], source


def _infer_length(f: h5py.File) -> int:
    for key in (
        "force_contact_score",
        "force_residual",
        "joint_efforts",
        "state",
        "joint_positions",
    ):
        if key in f:
            return len(f[key])
    raise SystemExit("Could not infer log length")


def _hysteresis(score: np.ndarray, high: float, low: float) -> np.ndarray:
    state = np.zeros(len(score), dtype=bool)
    active = False
    for i, value in enumerate(score):
        if active:
            active = value > low
        else:
            active = value >= high
        state[i] = active
    return state


def _segments(mask: np.ndarray) -> list[Tuple[int, int]]:
    segments = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _shade(ax, t: np.ndarray, contact: np.ndarray, pre_contact_sec: float) -> None:
    ax.axvspan(t[0], t[-1], color="#b7ead9", alpha=0.35, label="Free Motion")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    pre_ticks = max(1, int(round(pre_contact_sec / max(dt, 1e-6))))
    pre = np.zeros_like(contact)
    for start, _end in _segments(contact):
        pre_start = max(0, start - pre_ticks)
        pre[pre_start:start] = True

    for start, end in _segments(pre):
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="#f3d9ad", alpha=0.45)
    for start, end in _segments(contact):
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="#f3a5a9", alpha=0.45)

    ax.plot([], [], color="#f3d9ad", linewidth=8, alpha=0.75, label="Pre-contact")
    ax.plot([], [], color="#f3a5a9", linewidth=8, alpha=0.75, label="Contact")


def main() -> None:
    args = tyro.cli(Args)
    path = Path(args.h5_path)
    if not path.exists():
        raise SystemExit(f"Missing HDF5 file: {path}")
    if args.high_threshold <= args.low_threshold:
        raise SystemExit("--high_threshold must be > --low_threshold")

    t, score, source = _load_time_and_score(path, args.fps)
    finite = np.isfinite(score)
    score = np.where(finite, score, 0.0)
    if args.normalize:
        denom = float(np.max(score))
        if denom > 1e-9:
            score = score / denom

    contact = _hysteresis(score, args.high_threshold, args.low_threshold)
    out = Path(args.output_path) if args.output_path else path.with_name("force_timeline.png")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    _shade(ax, t, contact, args.pre_contact_sec)
    ax.plot(t, score, color="#333333", linewidth=1.8, label=source)
    ax.axhline(args.high_threshold, color="#b54545", linestyle="--", linewidth=1.0)
    ax.axhline(args.low_threshold, color="#b9822f", linestyle=":", linewidth=1.0)
    ax.set_title(args.title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized force/contact score" if args.normalize else "Force/contact score")
    ax.set_xlim(float(t[0]), float(t[-1]) if len(t) else 1.0)
    ax.set_ylim(bottom=0.0)
    if args.normalize:
        ax.set_ylim(top=max(1.05, float(np.max(score)) * 1.05))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
