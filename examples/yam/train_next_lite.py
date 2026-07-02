"""Train a NEXT-lite free-space effort model from YAM HDF5 logs.

The model learns expected free-space effort from histories of:

    [q, qdot, commanded_q - q]

At runtime, external load is:

    measured_effort - predicted_free_space_effort

This script is offline-only. Use it after collecting contact-free baseline logs.
"""

from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, TensorDataset

from gello_min.next_lite import NEXTLiteModel, make_next_features, residual_stats


@dataclass
class Args:
    input_glob: str
    """Glob or comma-separated globs for contact-free HDF5 files."""

    output_dir: str = "./yam_next_lite_runs"
    """Directory where model.pt and metrics.json will be written."""

    history: int = 50
    """Number of control ticks in each input history window."""

    epochs: int = 50
    """Training epochs."""

    batch_size: int = 256
    """Batch size."""

    learning_rate: float = 1e-3
    """AdamW learning rate."""

    weight_decay: float = 1e-6
    """AdamW weight decay."""

    hidden_size: int = 128
    """LSTM hidden size."""

    num_layers: int = 2
    """LSTM layer count."""

    mlp_hidden: int = 256
    """Prediction head hidden size."""

    dropout: float = 0.1
    """Dropout probability."""

    val_fraction: float = 0.2
    """Fraction of windows used for validation."""

    max_windows: int = 0
    """Optional cap on training windows; 0 means use all."""

    seed: int = 0
    """Random seed."""

    device: str = "auto"
    """Training device: auto, cpu, cuda, or mps."""


def _resolve_paths(spec: str) -> List[Path]:
    paths: List[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        matches = sorted(glob.glob(part))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(part))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise SystemExit(f"No input files matched: {spec}")
    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))
    return unique


def _dataset(f: h5py.File, names: Tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in f:
            return np.asarray(f[name][:], dtype=np.float32)
    return None


def _load_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        q = _dataset(f, ("joint_positions", "state"))
        qdot = _dataset(f, ("joint_velocities",))
        effort = _dataset(f, ("joint_efforts",))
        command = _dataset(
            f,
            (
                "commanded_joint_positions",
                "target_joint_positions",
                "requested_joint_positions",
                "policy_action",
                "next_state",
            ),
        )

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
    q = q[:length]
    qdot = qdot[:length]
    effort = effort[:length]
    command = command[:length]
    if q.ndim != 2 or qdot.shape != q.shape or effort.shape != q.shape or command.shape != q.shape:
        raise SystemExit(
            f"{path} dataset shape mismatch: q={q.shape}, qdot={qdot.shape}, "
            f"effort={effort.shape}, command={command.shape}"
        )

    features = make_next_features(q, qdot, command)
    finite = np.isfinite(features).all(axis=1) & np.isfinite(effort).all(axis=1)
    return features[finite], effort[finite]


def _make_windows(
    feature_logs: List[np.ndarray],
    target_logs: List[np.ndarray],
    history: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for features, targets in zip(feature_logs, target_logs):
        if len(features) < history:
            continue
        for end in range(history - 1, len(features)):
            start = end - history + 1
            xs.append(features[start : end + 1])
            ys.append(targets[end])
    if not xs:
        raise SystemExit(
            f"No windows created. Need at least {history} finite samples per log."
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def _choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    x_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    x_norm = (x_all - x_mean) / x_std
    y_norm = (y_all - y_mean) / y_std
    return x_norm, y_norm, x_mean, x_std, y_mean, y_std


def _eval_model(
    model: NEXTLiteModel,
    x: np.ndarray,
    y: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    losses = []
    preds = []
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0)
    pred = pred_norm * y_std + y_mean
    residual = y - pred
    loss = float(np.mean(((pred - y) / y_std) ** 2))
    losses.append(loss)
    return float(np.mean(losses)), residual_stats(residual)


def main() -> None:
    args = tyro.cli(Args)
    if args.history < 2:
        raise SystemExit("--history must be >= 2")
    if not 0.0 < args.val_fraction < 0.9:
        raise SystemExit("--val_fraction must be in (0, 0.9)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = _resolve_paths(args.input_glob)
    feature_logs = []
    target_logs = []
    for path in paths:
        features, targets = _load_log(path)
        print(f"loaded {path}: {len(features)} finite samples")
        feature_logs.append(features)
        target_logs.append(targets)

    x, y = _make_windows(feature_logs, target_logs, args.history)
    indices = np.arange(len(x))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    if args.max_windows and args.max_windows > 0 and args.max_windows < len(indices):
        indices = indices[: args.max_windows]
        x = x[indices]
        y = y[indices]
        indices = np.arange(len(x))
        rng.shuffle(indices)

    val_count = max(1, int(round(len(indices) * args.val_fraction)))
    train_idx = indices[val_count:]
    val_idx = indices[:val_count]
    if len(train_idx) == 0:
        raise SystemExit("Not enough windows for train/val split")

    x_norm, y_norm, x_mean, x_std, y_mean, y_std = _normalize(
        x[train_idx],
        y[train_idx],
        x,
        y,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = _choose_device(args.device)
    model = NEXTLiteModel(
        input_dim=x.shape[-1],
        output_dim=y.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history_rows = []
    print(
        f"training windows={len(train_idx)} val={len(val_idx)} "
        f"input_dim={x.shape[-1]} output_dim={y.shape[-1]} device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_resid_stats = _eval_model(
            model,
            x_norm[val_idx],
            y[val_idx],
            y_mean,
            y_std,
            device,
            args.batch_size,
        )
        train_loss = float(np.mean(train_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_norm_q99={val_resid_stats['norm_q99']:.6g}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_stats = _eval_model(
        model,
        x_norm[train_idx],
        y[train_idx],
        y_mean,
        y_std,
        device,
        args.batch_size,
    )
    val_loss, val_stats = _eval_model(
        model,
        x_norm[val_idx],
        y[val_idx],
        y_mean,
        y_std,
        device,
        args.batch_size,
    )

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "yam_next_lite_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_paths": [str(p) for p in paths],
        "history": args.history,
        "input_dim": int(x.shape[-1]),
        "output_dim": int(y.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "mlp_hidden": args.mlp_hidden,
        "dropout": args.dropout,
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
    }
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "metadata": metadata,
    }
    torch.save(checkpoint, run_dir / "model.pt")

    metrics = {
        "metadata": metadata,
        "history": history_rows,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_residual": train_stats,
        "val_residual": val_stats,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nSaved NEXT-lite model: {run_dir / 'model.pt'}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print("Validation residual summary:")
    print(f"  abs_q99: {val_stats['abs_q99']}")
    print(f"  norm_q99: {val_stats['norm_q99']:.6g}")
    print(f"  norm_q995: {val_stats['norm_q995']:.6g}")


if __name__ == "__main__":
    main()
