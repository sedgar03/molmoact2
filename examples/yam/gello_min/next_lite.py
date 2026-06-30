from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn


class NEXTLiteModel(nn.Module):
    """Small recurrent free-space effort predictor.

    The model predicts measured free-space joint effort from a short history of
    proprioception and command deltas. External load is the residual between
    measured effort and this prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _hidden = self.lstm(x)
        return self.head(out[:, -1])


@dataclass
class NEXTLiteCheckpoint:
    model: NEXTLiteModel
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    metadata: Dict[str, Any]

    @property
    def history(self) -> int:
        return int(self.metadata["history"])

    def predict_effort(self, history_features: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict effort from one unnormalized history window."""
        x = np.asarray(history_features, dtype=np.float32)
        if x.shape != (self.history, len(self.x_mean)):
            raise ValueError(
                f"Expected history shape {(self.history, len(self.x_mean))}, got {x.shape}"
            )
        x_norm = (x - self.x_mean) / self.x_std
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(
                torch.as_tensor(x_norm[None, ...], dtype=torch.float32, device=device)
            )
        pred = pred_norm.cpu().numpy()[0] * self.y_std + self.y_mean
        return pred.astype(np.float32)


def load_next_lite_checkpoint(path: str | Path, map_location: str = "cpu") -> NEXTLiteCheckpoint:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    metadata = checkpoint["metadata"]
    model = NEXTLiteModel(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        mlp_hidden=int(metadata["mlp_hidden"]),
        dropout=float(metadata["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return NEXTLiteCheckpoint(
        model=model,
        x_mean=np.asarray(checkpoint["x_mean"], dtype=np.float32),
        x_std=np.asarray(checkpoint["x_std"], dtype=np.float32),
        y_mean=np.asarray(checkpoint["y_mean"], dtype=np.float32),
        y_std=np.asarray(checkpoint["y_std"], dtype=np.float32),
        metadata=metadata,
    )


def make_next_features(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    commanded_joint_positions: np.ndarray,
) -> np.ndarray:
    q = np.asarray(joint_positions, dtype=np.float32)
    qdot = np.asarray(joint_velocities, dtype=np.float32)
    command = np.asarray(commanded_joint_positions, dtype=np.float32)
    return np.concatenate([q, qdot, command - q], axis=-1)


def residual_stats(residual: np.ndarray) -> Dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64)
    abs_residual = np.abs(residual)
    norm = np.linalg.norm(residual, axis=1)
    return {
        "mae": abs_residual.mean(axis=0).tolist(),
        "rmse": np.sqrt((residual ** 2).mean(axis=0)).tolist(),
        "abs_q95": np.quantile(abs_residual, 0.95, axis=0).tolist(),
        "abs_q99": np.quantile(abs_residual, 0.99, axis=0).tolist(),
        "abs_q995": np.quantile(abs_residual, 0.995, axis=0).tolist(),
        "norm_q95": float(np.quantile(norm, 0.95)),
        "norm_q99": float(np.quantile(norm, 0.99)),
        "norm_q995": float(np.quantile(norm, 0.995)),
    }


def split_q_qdot_command(feature_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q, qdot, command_delta from a NEXT feature window."""
    feat = np.asarray(feature_window)
    if feat.shape[-1] % 3 != 0:
        raise ValueError(f"Feature dimension must be divisible by 3, got {feat.shape[-1]}")
    n = feat.shape[-1] // 3
    return feat[..., :n], feat[..., n : 2 * n], feat[..., 2 * n :]
