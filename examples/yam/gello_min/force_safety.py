from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np


class ForceSafetyError(RuntimeError):
    """Raised when the force safety monitor aborts a robot command."""


@dataclass
class ForceSafetyDecision:
    command: np.ndarray
    abort: bool
    reason: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None


@dataclass
class ForceSafetyTelemetry:
    effort: Optional[np.ndarray] = None
    expected_effort: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    residual_l1: Optional[float] = None
    residual_l2: Optional[float] = None
    contact_score: Optional[float] = None
    contact_score_normalized: bool = False
    contact_score_source: str = "unavailable"
    contact_state_code: int = -1
    next_lite_ready: bool = False

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "force_contact_score_normalized": bool(self.contact_score_normalized),
            "force_contact_state_code": int(self.contact_state_code),
            "force_next_lite_ready": bool(self.next_lite_ready),
            "force_contact_score_source": self.contact_score_source,
        }
        if self.effort is not None:
            record["force_effort"] = np.asarray(self.effort, dtype=np.float32).copy()
        if self.expected_effort is not None:
            record["force_expected_effort"] = np.asarray(
                self.expected_effort, dtype=np.float32
            ).copy()
        if self.residual is not None:
            record["force_residual"] = np.asarray(self.residual, dtype=np.float32).copy()
        if self.residual_l1 is not None:
            record["force_residual_l1"] = float(self.residual_l1)
        if self.residual_l2 is not None:
            record["force_residual_l2"] = float(self.residual_l2)
        if self.contact_score is not None:
            record["force_contact_score"] = float(self.contact_score)
        return record


def _as_limit_array(value: Any, n: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or length {n}, got shape {arr.shape}")
    return arr


class ForceSafetyMonitor:
    """Small deterministic guard for YAM commands.

    This is intentionally a raw-effort guard with an optional synchronized
    command limiter. The next step is replacing raw effort thresholds with
    residual thresholds once free-space effort logs exist.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]], num_dofs: int):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.num_dofs = int(num_dofs)
        self.filter_alpha = float(cfg.get("filter_alpha", 0.35))
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("force_safety.filter_alpha must be in (0, 1]")
        self.min_trigger_ticks = max(1, int(cfg.get("min_trigger_ticks", 2)))

        self.warning_abs_effort = _as_limit_array(
            cfg.get("warning_abs_effort"), self.num_dofs, "warning_abs_effort"
        )
        self.hard_abs_effort = _as_limit_array(
            cfg.get("hard_abs_effort"), self.num_dofs, "hard_abs_effort"
        )
        self.command_limit_mode = str(cfg.get("command_limit_mode", "off"))
        if self.command_limit_mode not in ("off", "scale"):
            raise ValueError("force_safety.command_limit_mode must be 'off' or 'scale'")
        self.max_command_delta = _as_limit_array(
            cfg.get("max_command_delta"), self.num_dofs, "max_command_delta"
        )
        self.hard_effort_norm = cfg.get("hard_effort_norm")
        self.hard_effort_norm = (
            float(self.hard_effort_norm) if self.hard_effort_norm is not None else None
        )
        self.warning_abs_residual = _as_limit_array(
            cfg.get("warning_abs_residual"), self.num_dofs, "warning_abs_residual"
        )
        self.hard_abs_residual = _as_limit_array(
            cfg.get("hard_abs_residual"), self.num_dofs, "hard_abs_residual"
        )
        self.hard_residual_norm = cfg.get("hard_residual_norm")
        self.hard_residual_norm = (
            float(self.hard_residual_norm) if self.hard_residual_norm is not None else None
        )

        gripper_indices = cfg.get("gripper_indices")
        if gripper_indices is None and self.num_dofs % 7 == 0:
            gripper_indices = list(range(6, self.num_dofs, 7))
        self.gripper_indices = np.asarray(gripper_indices or [], dtype=int)
        max_gripper_delta = cfg.get("max_gripper_delta")
        if max_gripper_delta is not None and self.max_command_delta is not None:
            for idx in self.gripper_indices:
                if idx < 0 or idx >= self.num_dofs:
                    raise ValueError(f"Invalid gripper index {idx} for {self.num_dofs} DOFs")
                self.max_command_delta[idx] = float(max_gripper_delta)

        self._filtered_effort: Optional[np.ndarray] = None
        self._next_lite = None
        self._next_lite_device = "cpu"
        self._next_feature_history: Optional[Deque[np.ndarray]] = None
        self._load_next_lite(cfg.get("next_lite") or {})
        self._hard_ticks = 0
        self._warn_ticks = 0
        self._last_reason: Optional[str] = None

    def check(self, target: np.ndarray, obs: Dict[str, Any]) -> ForceSafetyDecision:
        target = np.asarray(target, dtype=float).copy()
        if target.shape != (self.num_dofs,):
            raise ValueError(f"Command shape {target.shape} != ({self.num_dofs},)")
        if not self.enabled:
            return ForceSafetyDecision(command=target, abort=False)

        current = np.asarray(obs["joint_positions"], dtype=float).reshape(-1)
        if current.shape != (self.num_dofs,):
            raise ValueError(f"Current joint shape {current.shape} != ({self.num_dofs},)")

        command = self._limit_command_delta(target, current)
        effort = self._read_effort(obs)
        if effort is None:
            telemetry = ForceSafetyTelemetry()
            return ForceSafetyDecision(
                command=command,
                abort=False,
                telemetry=telemetry.as_record(),
            )
        residual, expected_effort, next_ready = self._next_lite_residual(
            effort, current, command, obs
        )
        telemetry = self._build_telemetry(effort, expected_effort, residual, next_ready)

        hard_reasons = []
        warn_reasons = []
        abs_effort = np.abs(effort)
        if self.hard_abs_effort is not None:
            mask = abs_effort > self.hard_abs_effort
            if np.any(mask):
                hard_reasons.append(self._format_joint_reason("hard effort", abs_effort, mask))
        if self.warning_abs_effort is not None:
            mask = abs_effort > self.warning_abs_effort
            if np.any(mask):
                warn_reasons.append(self._format_joint_reason("warning effort", abs_effort, mask))
        if self.hard_effort_norm is not None:
            effort_norm = float(np.linalg.norm(effort))
            if effort_norm > self.hard_effort_norm:
                hard_reasons.append(
                    f"hard effort norm {effort_norm:.3f} > {self.hard_effort_norm:.3f}"
                )
        if residual is not None:
            abs_residual = np.abs(residual)
            if self.hard_abs_residual is not None:
                mask = abs_residual > self.hard_abs_residual
                if np.any(mask):
                    hard_reasons.append(
                        self._format_joint_reason("hard residual", abs_residual, mask)
                    )
            if self.warning_abs_residual is not None:
                mask = abs_residual > self.warning_abs_residual
                if np.any(mask):
                    warn_reasons.append(
                        self._format_joint_reason("warning residual", abs_residual, mask)
                    )
            if self.hard_residual_norm is not None:
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm > self.hard_residual_norm:
                    hard_reasons.append(
                        f"hard residual norm {residual_norm:.3f} > "
                        f"{self.hard_residual_norm:.3f}"
                    )

        if hard_reasons:
            self._hard_ticks += 1
        else:
            self._hard_ticks = 0
        if warn_reasons:
            self._warn_ticks += 1
        else:
            self._warn_ticks = 0

        if self._hard_ticks >= self.min_trigger_ticks:
            telemetry.contact_state_code = 2
        elif self._warn_ticks >= self.min_trigger_ticks:
            telemetry.contact_state_code = 1
        elif self._next_lite is not None and not next_ready:
            telemetry.contact_state_code = -2
        else:
            telemetry.contact_state_code = 0
        telemetry_record = telemetry.as_record()

        if self._hard_ticks >= self.min_trigger_ticks:
            reason = "; ".join(hard_reasons)
            return ForceSafetyDecision(
                command=current,
                abort=True,
                reason=reason,
                telemetry=telemetry_record,
            )
        if self._warn_ticks >= self.min_trigger_ticks:
            reason = "; ".join(warn_reasons)
            return ForceSafetyDecision(
                command=current,
                abort=False,
                reason=reason,
                telemetry=telemetry_record,
            )
        return ForceSafetyDecision(command=command, abort=False, telemetry=telemetry_record)

    def _load_next_lite(self, cfg: Dict[str, Any]) -> None:
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path is None:
            if (
                self.warning_abs_residual is not None
                or self.hard_abs_residual is not None
                or self.hard_residual_norm is not None
            ):
                raise ValueError(
                    "NEXT-lite residual thresholds require "
                    "force_safety.next_lite.checkpoint_path"
                )
            return

        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"NEXT-lite checkpoint not found: {path}")

        try:
            from gello_min.next_lite import load_next_lite_checkpoint
        except ImportError as exc:
            raise ImportError(
                "force_safety.next_lite requires torch and gello_min.next_lite"
            ) from exc

        self._next_lite_device = str(cfg.get("device", "cpu"))
        self._next_lite = load_next_lite_checkpoint(path, map_location=self._next_lite_device)
        output_dim = int(self._next_lite.metadata["output_dim"])
        input_dim = int(self._next_lite.metadata["input_dim"])
        expected_input_dim = self.num_dofs * 3
        if output_dim != self.num_dofs:
            raise ValueError(
                f"NEXT-lite output_dim {output_dim} != robot DOFs {self.num_dofs}"
            )
        if input_dim != expected_input_dim:
            raise ValueError(
                f"NEXT-lite input_dim {input_dim} != expected {expected_input_dim}"
            )
        self._next_feature_history = deque(maxlen=self._next_lite.history)

    def _next_lite_residual(
        self,
        effort: np.ndarray,
        current: np.ndarray,
        command: np.ndarray,
        obs: Dict[str, Any],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if self._next_lite is None or self._next_feature_history is None:
            return None, None, False
        velocity = np.asarray(obs["joint_velocities"], dtype=float).reshape(-1)
        if velocity.shape != (self.num_dofs,):
            raise ValueError(f"Velocity shape {velocity.shape} != ({self.num_dofs},)")

        try:
            from gello_min.next_lite import make_next_features
        except ImportError as exc:
            raise ImportError(
                "force_safety.next_lite requires torch and gello_min.next_lite"
            ) from exc

        features = make_next_features(current, velocity, command)
        self._next_feature_history.append(features)
        if len(self._next_feature_history) < self._next_lite.history:
            return None, None, False

        expected_effort = self._next_lite.predict_effort(
            np.stack(self._next_feature_history),
            device=self._next_lite_device,
        )
        return effort - expected_effort, expected_effort, True

    def _build_telemetry(
        self,
        effort: np.ndarray,
        expected_effort: Optional[np.ndarray],
        residual: Optional[np.ndarray],
        next_ready: bool,
    ) -> ForceSafetyTelemetry:
        telemetry = ForceSafetyTelemetry(
            effort=effort,
            expected_effort=expected_effort,
            residual=residual,
            next_lite_ready=next_ready,
        )
        signal = residual if residual is not None else effort
        telemetry.contact_score_source = "residual" if residual is not None else "effort"
        abs_signal = np.abs(signal)

        if residual is not None:
            telemetry.residual_l1 = float(np.sum(abs_signal))
            telemetry.residual_l2 = float(np.linalg.norm(residual))
            if self.hard_abs_residual is not None:
                ratios = abs_signal / self.hard_abs_residual
                telemetry.contact_score = float(np.max(ratios))
                telemetry.contact_score_normalized = True
            elif self.hard_residual_norm is not None:
                telemetry.contact_score = float(telemetry.residual_l2 / self.hard_residual_norm)
                telemetry.contact_score_normalized = True
            else:
                telemetry.contact_score = telemetry.residual_l1
            return telemetry

        effort_l1 = float(np.sum(abs_signal))
        effort_l2 = float(np.linalg.norm(effort))
        if self.hard_abs_effort is not None:
            ratios = abs_signal / self.hard_abs_effort
            telemetry.contact_score = float(np.max(ratios))
            telemetry.contact_score_normalized = True
        elif self.hard_effort_norm is not None:
            telemetry.contact_score = float(effort_l2 / self.hard_effort_norm)
            telemetry.contact_score_normalized = True
        else:
            telemetry.contact_score = effort_l1
        return telemetry

    def _limit_command_delta(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        if self.command_limit_mode == "off" or self.max_command_delta is None:
            return target

        limit = self.max_command_delta
        if np.any(limit <= 0.0):
            raise ValueError("force_safety.max_command_delta values must be positive")

        delta = target - current
        ratios = np.divide(
            np.abs(delta),
            limit,
            out=np.zeros_like(delta, dtype=float),
            where=limit > 0.0,
        )
        max_ratio = float(np.max(ratios))
        if max_ratio <= 1.0:
            return target

        # Scale the full command vector together. This preserves the requested
        # joint-space path instead of independently clipping individual joints.
        return current + delta / max_ratio

    def _read_effort(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        raw = obs.get("joint_efforts")
        if raw is None:
            raw = obs.get("joint_eff")
        if raw is None:
            if (
                self.warning_abs_effort is not None
                or self.hard_abs_effort is not None
                or self.hard_effort_norm is not None
                or self.warning_abs_residual is not None
                or self.hard_abs_residual is not None
                or self.hard_residual_norm is not None
            ):
                raise ForceSafetyError("force_safety effort thresholds enabled but obs has no effort")
            return None

        effort = np.asarray(raw, dtype=float).reshape(-1)
        if effort.shape != (self.num_dofs,):
            raise ValueError(f"Effort shape {effort.shape} != ({self.num_dofs},)")
        if self._filtered_effort is None:
            self._filtered_effort = effort.copy()
        else:
            a = self.filter_alpha
            self._filtered_effort = a * effort + (1.0 - a) * self._filtered_effort
        return self._filtered_effort

    def _format_joint_reason(self, label: str, values: np.ndarray, mask: np.ndarray) -> str:
        parts = [f"j{i}={values[i]:.3f}" for i in np.flatnonzero(mask)]
        return f"{label}: " + ", ".join(parts)
