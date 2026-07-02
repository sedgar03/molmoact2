"""Apply a NEXT-lite effort model to a YAM HDF5 log.

The output log keeps the original datasets and adds:

    force_expected_effort
    force_residual
    force_contact_score
    force_residual_l1
    force_residual_l2

Use this offline to inspect FACTR2-style external torque residuals before
enabling the same model in the live force_safety config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Tuple

import h5py
import numpy as np
import torch
import tyro

from gello_min.next_lite import (
    load_next_lite_checkpoint,
    make_next_features,
    residual_stats,
)


@dataclass
class Args:
    h5_path: Annotated[str, tyro.conf.Positional]
    """Input YAM HDF5 log."""

    checkpoint_path: Annotated[str, tyro.conf.Positional]
    """NEXT-lite model.pt checkpoint from train_next_lite.py."""

    output_path: str = ""
    """Output HDF5 path. Defaults next to input as *_with_next_lite_residual.h5."""

    batch_size: int = 1024
    """Inference batch size for history windows."""

    device: str = "auto"
    """Inference device: auto, cpu, cuda, or mps."""


def _dataset(f: h5py.File, names: Tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in f:
            return np.asarray(f[name][:], dtype=np.float32)
    return None


def _choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(path, "r") as f:
        datasets = {key: f[key][:] for key in f.keys()}
        attrs = dict(f.attrs.items())
    return datasets, attrs


def _required_arrays(datasets: dict[str, np.ndarray], path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = datasets.get("joint_positions")
    if q is None:
        q = datasets.get("state")
    qdot = datasets.get("joint_velocities")
    effort = datasets.get("joint_efforts")
    command = datasets.get("commanded_joint_positions")
    if command is None:
        command = datasets.get("target_joint_positions")
    if command is None:
        command = datasets.get("requested_joint_positions")
    if command is None:
        command = datasets.get("policy_action")
    if command is None:
        command = datasets.get("next_state")

    missing = [
        name
        for name, value in (
            ("joint_positions/state", q),
            ("joint_velocities", qdot),
            ("joint_efforts", effort),
            ("commanded/target/requested command", command),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"{path} missing required datasets: {', '.join(missing)}")

    length = min(len(q), len(qdot), len(effort), len(command))
    q = np.asarray(q[:length], dtype=np.float32)
    qdot = np.asarray(qdot[:length], dtype=np.float32)
    effort = np.asarray(effort[:length], dtype=np.float32)
    command = np.asarray(command[:length], dtype=np.float32)
    if q.ndim != 2 or qdot.shape != q.shape or effort.shape != q.shape or command.shape != q.shape:
        raise SystemExit(
            f"{path} dataset shape mismatch: q={q.shape}, qdot={qdot.shape}, "
            f"effort={effort.shape}, command={command.shape}"
        )
    return q, qdot, effort, command


def _predict_expected_effort(
    features: np.ndarray,
    effort_shape: tuple[int, int],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    checkpoint = load_next_lite_checkpoint(checkpoint_path, map_location="cpu")
    history = checkpoint.history
    expected = np.full(effort_shape, np.nan, dtype=np.float32)
    if len(features) < history:
        return expected, history

    windows = np.stack(
        [features[end - history + 1 : end + 1] for end in range(history - 1, len(features))]
    ).astype(np.float32)
    x_norm = (windows - checkpoint.x_mean) / checkpoint.x_std

    model = checkpoint.model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_norm), batch_size):
            xb = torch.as_tensor(
                x_norm[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            pred_norm = model(xb).cpu().numpy()
            preds.append(pred_norm * checkpoint.y_std + checkpoint.y_mean)

    expected[history - 1 :] = np.concatenate(preds, axis=0).astype(np.float32)
    return expected, history


def _write_h5(
    path: Path,
    datasets: dict[str, np.ndarray],
    attrs: dict[str, object],
    expected: np.ndarray,
    residual: np.ndarray,
    score_l1: np.ndarray,
    score_l2: np.ndarray,
    checkpoint_path: Path,
    history: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in datasets.items():
            kwargs = {}
            if key != "timestamp":
                kwargs = {"compression": "gzip", "compression_opts": 4}
            f.create_dataset(key, data=value, **kwargs)
        f.create_dataset("force_expected_effort", data=expected, compression="gzip", compression_opts=4)
        f.create_dataset("force_residual", data=residual, compression="gzip", compression_opts=4)
        f.create_dataset("force_contact_score", data=score_l1, compression="gzip", compression_opts=4)
        f.create_dataset("force_residual_l1", data=score_l1, compression="gzip", compression_opts=4)
        f.create_dataset("force_residual_l2", data=score_l2, compression="gzip", compression_opts=4)
        for key, value in attrs.items():
            f.attrs[key] = value
        f.attrs["schema"] = "yam_next_lite_residual_log_v1"
        f.attrs["next_lite_checkpoint"] = str(checkpoint_path)
        f.attrs["next_lite_history"] = int(history)


def main() -> None:
    args = tyro.cli(Args)
    h5_path = Path(args.h5_path)
    checkpoint_path = Path(args.checkpoint_path)
    if not h5_path.exists():
        raise SystemExit(f"Missing HDF5 file: {h5_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    datasets, attrs = _load_arrays(h5_path)
    q, qdot, effort, command = _required_arrays(datasets, h5_path)
    features = make_next_features(q, qdot, command)
    device = _choose_device(args.device)
    expected, history = _predict_expected_effort(
        features,
        effort.shape,
        checkpoint_path,
        device,
        args.batch_size,
    )
    residual = effort - expected
    score_l1 = np.nansum(np.abs(residual), axis=1).astype(np.float32)
    score_l2 = np.sqrt(np.nansum(residual * residual, axis=1)).astype(np.float32)
    score_l1[: history - 1] = np.nan
    score_l2[: history - 1] = np.nan

    output_path = (
        Path(args.output_path)
        if args.output_path
        else h5_path.with_name(h5_path.stem + "_with_next_lite_residual.h5")
    )
    _write_h5(
        output_path,
        datasets,
        attrs,
        expected,
        residual,
        score_l1,
        score_l2,
        checkpoint_path,
        history,
    )

    finite = residual[np.isfinite(residual).all(axis=1)]
    print(f"Wrote {output_path}")
    print(f"finite residual samples: {len(finite)} / {len(residual)}")
    if len(finite):
        stats = residual_stats(finite)
        print(f"residual abs_q99: {stats['abs_q99']}")
        print(f"residual norm_q99: {stats['norm_q99']:.6g}")
        print(f"residual norm_q995: {stats['norm_q995']:.6g}")
        print(
            "force_contact_score L1 p50/p95/p99/max: "
            f"{np.nanpercentile(score_l1, [50, 95, 99, 100]).tolist()}"
        )


if __name__ == "__main__":
    main()
