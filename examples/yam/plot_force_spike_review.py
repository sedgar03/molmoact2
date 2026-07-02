"""Generate a review pack for NEXT-lite force/contact spikes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    h5_path: Annotated[str, tyro.conf.Positional]
    """HDF5 log containing force_contact_score and force_residual."""

    output_dir: str = ""
    """Output directory. Defaults to spike_review next to the input HDF5."""

    top_n: int = 12
    """Number of ranked spikes to review."""

    threshold_quantile: float = 0.995
    """Score quantile used as the spike threshold."""

    min_score: float = 0.0
    """Optional absolute minimum score threshold."""

    min_separation_sec: float = 2.0
    """Minimum time separation between selected spike peaks."""

    window_sec: float = 6.0
    """Seconds shown around each spike in the zoom grid."""

    marked_times_sec: str = ""
    """Comma-separated relative times to draw as operator/event marks."""

    title: str = "YAM NEXT-lite Spike Review"
    """Plot title prefix."""


def _load(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        data = {key: np.asarray(f[key][:]) for key in f.keys()}
    if "timestamp" in data:
        t = data["timestamp"].astype(np.float64)
        data["_time_sec"] = t - t[0]
    elif "monotonic_time" in data:
        t = data["monotonic_time"].astype(np.float64)
        data["_time_sec"] = t - t[0]
    else:
        n = len(data.get("force_contact_score", data.get("force_residual")))
        data["_time_sec"] = np.arange(n, dtype=np.float64)
    if "force_contact_score" not in data:
        if "force_residual" not in data:
            raise SystemExit(f"{path} has no force_contact_score or force_residual")
        data["force_contact_score"] = np.linalg.norm(data["force_residual"], ord=1, axis=1)
    if "force_residual" not in data:
        raise SystemExit(f"{path} has no force_residual dataset")
    return data


def _parse_marks(spec: str) -> list[float]:
    marks = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            marks.append(float(part))
    return marks


def _local_maxima(score: np.ndarray) -> np.ndarray:
    finite_score = np.where(np.isfinite(score), score, -np.inf)
    if len(finite_score) < 3:
        return np.arange(len(finite_score))
    mids = finite_score[1:-1]
    mask = (mids >= finite_score[:-2]) & (mids > finite_score[2:])
    return np.flatnonzero(mask) + 1


def _select_peaks(
    t: np.ndarray,
    score: np.ndarray,
    threshold: float,
    top_n: int,
    min_separation_sec: float,
) -> list[int]:
    candidates = _local_maxima(score)
    candidates = [int(i) for i in candidates if np.isfinite(score[i]) and score[i] >= threshold]
    candidates.sort(key=lambda i: float(score[i]), reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(float(t[idx] - t[prev])) >= min_separation_sec for prev in selected):
            selected.append(idx)
            if len(selected) >= top_n:
                break
    return selected


def _duration_above_threshold(t: np.ndarray, score: np.ndarray, idx: int, threshold: float) -> float:
    lo = idx
    hi = idx
    while lo > 0 and np.isfinite(score[lo - 1]) and score[lo - 1] >= threshold:
        lo -= 1
    while hi + 1 < len(score) and np.isfinite(score[hi + 1]) and score[hi + 1] >= threshold:
        hi += 1
    return float(t[hi] - t[lo])


def _write_csv(
    out_path: Path,
    t: np.ndarray,
    score: np.ndarray,
    residual: np.ndarray,
    data: dict[str, np.ndarray],
    peaks: list[int],
    threshold: float,
    marked_times: list[float],
) -> None:
    score_l2 = data.get("force_residual_l2")
    command_error = data.get("command_error")
    qdot = data.get("joint_velocities")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "time_sec",
                "score_l1",
                "score_l2",
                "dominant_joint",
                "dominant_joint_abs_residual",
                "duration_above_threshold_sec",
                "nearest_mark_sec",
                "distance_to_nearest_mark_sec",
                "command_error_l2",
                "joint_velocity_l2",
                "residual_per_joint",
            ]
        )
        for rank, idx in enumerate(peaks, start=1):
            abs_res = np.abs(residual[idx])
            dominant = int(np.nanargmax(abs_res))
            nearest = ""
            distance = ""
            if marked_times:
                nearest_value = min(marked_times, key=lambda m: abs(m - float(t[idx])))
                nearest = f"{nearest_value:.6f}"
                distance = f"{abs(nearest_value - float(t[idx])):.6f}"
            writer.writerow(
                [
                    rank,
                    f"{float(t[idx]):.6f}",
                    f"{float(score[idx]):.6f}",
                    f"{float(score_l2[idx]) if score_l2 is not None else np.nan:.6f}",
                    dominant,
                    f"{float(abs_res[dominant]):.6f}",
                    f"{_duration_above_threshold(t, score, idx, threshold):.6f}",
                    nearest,
                    distance,
                    f"{float(np.linalg.norm(command_error[idx])) if command_error is not None else np.nan:.6f}",
                    f"{float(np.linalg.norm(qdot[idx])) if qdot is not None else np.nan:.6f}",
                    ";".join(f"{float(v):.6f}" for v in residual[idx]),
                ]
            )


def _plot_overview(
    out_path: Path,
    title: str,
    t: np.ndarray,
    score: np.ndarray,
    threshold: float,
    peaks: list[int],
    marked_times: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=160)
    ax.plot(t, score, color="#2f3333", linewidth=1.0)
    ax.axhline(threshold, color="#b54545", linestyle="--", linewidth=1.1, label="spike threshold")
    for rank, idx in enumerate(peaks, start=1):
        ax.scatter([t[idx]], [score[idx]], s=28, color="#d04f3a", zorder=5)
        ax.annotate(str(rank), (t[idx], score[idx]), xytext=(4, 6), textcoords="offset points", fontsize=8)
    for mark in marked_times:
        ax.axvline(mark, color="#3f6db5", linestyle=":", linewidth=1.2, alpha=0.9)
    if marked_times:
        ax.plot([], [], color="#3f6db5", linestyle=":", label="marked event")
    ax.set_title(f"{title}: ranked spikes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L1 residual/contact score")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_zoom_grid(
    out_path: Path,
    title: str,
    t: np.ndarray,
    score: np.ndarray,
    threshold: float,
    peaks: list[int],
    marked_times: list[float],
    window_sec: float,
) -> None:
    if not peaks:
        return
    cols = 3
    rows = int(np.ceil(len(peaks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.0, rows * 2.7)), dpi=160, squeeze=False)
    half = window_sec / 2.0
    for ax in axes.ravel():
        ax.axis("off")
    for rank, idx in enumerate(peaks, start=1):
        ax = axes.ravel()[rank - 1]
        ax.axis("on")
        lo = float(t[idx] - half)
        hi = float(t[idx] + half)
        mask = (t >= lo) & (t <= hi)
        ax.plot(t[mask], score[mask], color="#2f3333", linewidth=1.2)
        ax.axhline(threshold, color="#b54545", linestyle="--", linewidth=0.9)
        ax.axvline(float(t[idx]), color="#d04f3a", linewidth=1.2)
        for mark in marked_times:
            if lo <= mark <= hi:
                ax.axvline(mark, color="#3f6db5", linestyle=":", linewidth=1.2)
        ax.set_title(f"#{rank}  t={float(t[idx]):.2f}s  score={float(score[idx]):.2f}", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{title}: spike zooms", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_joint_heatmap(
    out_path: Path,
    title: str,
    residual: np.ndarray,
    peaks: list[int],
) -> None:
    if not peaks:
        return
    values = residual[peaks]
    scale = float(np.nanmax(np.abs(values))) if values.size else 1.0
    scale = max(scale, 1e-6)
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(peaks))), dpi=160)
    im = ax.imshow(values, cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_title(f"{title}: residual per joint at spike peak")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Spike rank")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(len(peaks)))
    ax.set_yticklabels([str(i) for i in range(1, len(peaks) + 1)])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Residual effort")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = tyro.cli(Args)
    path = Path(args.h5_path)
    if not path.exists():
        raise SystemExit(f"Missing HDF5 file: {path}")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise SystemExit("--threshold-quantile must be in (0, 1)")

    data = _load(path)
    t = data["_time_sec"].astype(np.float64)
    score = data["force_contact_score"].astype(np.float64)
    residual = data["force_residual"].astype(np.float64)
    finite_score = score[np.isfinite(score)]
    if len(finite_score) == 0:
        raise SystemExit("No finite force_contact_score samples")

    threshold = max(float(np.quantile(finite_score, args.threshold_quantile)), float(args.min_score))
    peaks = _select_peaks(t, score, threshold, args.top_n, args.min_separation_sec)
    marked_times = _parse_marks(args.marked_times_sec)
    out_dir = Path(args.output_dir) if args.output_dir else path.with_name("spike_review")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "spikes.csv", t, score, residual, data, peaks, threshold, marked_times)
    _plot_overview(out_dir / "spike_overview.png", args.title, t, score, threshold, peaks, marked_times)
    _plot_zoom_grid(
        out_dir / "spike_zoom_grid.png",
        args.title,
        t,
        score,
        threshold,
        peaks,
        marked_times,
        args.window_sec,
    )
    _plot_joint_heatmap(out_dir / "spike_joint_heatmap.png", args.title, residual, peaks)

    print(f"threshold={threshold:.6f} selected_peaks={len(peaks)}")
    print(f"Wrote {out_dir / 'spikes.csv'}")
    print(f"Wrote {out_dir / 'spike_overview.png'}")
    print(f"Wrote {out_dir / 'spike_zoom_grid.png'}")
    print(f"Wrote {out_dir / 'spike_joint_heatmap.png'}")


if __name__ == "__main__":
    main()
