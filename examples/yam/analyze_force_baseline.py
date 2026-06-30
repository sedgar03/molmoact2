"""Summarize a free-space YAM force baseline and propose raw effort thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numpy as np
import tyro


@dataclass
class Args:
    baseline_path: Annotated[str, tyro.conf.Positional]
    """Path to free_space_baseline.h5 from collect_force_baseline.py."""

    quantile: float = 0.995
    """Free-space absolute-effort quantile used as the baseline envelope."""

    warning_scale: float = 1.5
    """Multiplier from baseline envelope to warning threshold."""

    hard_scale: float = 2.5
    """Multiplier from baseline envelope to hard-abort threshold."""

    min_warning_abs_effort: float = 0.05
    """Minimum per-joint warning threshold."""

    min_hard_abs_effort: float = 0.10
    """Minimum per-joint hard threshold."""

    output_path: Optional[str] = None
    """Optional path to write the recommended YAML snippet."""


def _fmt_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.6g}" for v in values) + "]"


def _build_yaml(
    warning: np.ndarray,
    hard: np.ndarray,
    hard_norm: float,
    args: Args,
) -> str:
    lines = [
        "# Baseline-derived raw effort thresholds.",
        "# Validate on a soft surrogate before glass handling.",
        "force_safety:",
        "  enabled: true",
        "  filter_alpha: 0.35",
        "  min_trigger_ticks: 2",
        "  command_limit_mode: off",
        "  max_command_delta: null",
        "  max_gripper_delta: null",
        f"  warning_abs_effort: {_fmt_array(warning)}",
        f"  hard_abs_effort: {_fmt_array(hard)}",
        f"  hard_effort_norm: {hard_norm:.6g}",
        "",
        f"# quantile: {args.quantile}",
        f"# warning_scale: {args.warning_scale}",
        f"# hard_scale: {args.hard_scale}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = tyro.cli(Args)
    if not 0.0 < args.quantile < 1.0:
        raise SystemExit("--quantile must be between 0 and 1")

    with h5py.File(args.baseline_path, "r") as f:
        if "joint_efforts" not in f:
            raise SystemExit(f"{args.baseline_path} has no joint_efforts dataset")
        efforts = np.asarray(f["joint_efforts"][:], dtype=np.float64)

    if efforts.ndim != 2:
        raise SystemExit(f"joint_efforts must be rank 2, got shape {efforts.shape}")

    finite_rows = np.isfinite(efforts).all(axis=1)
    efforts = efforts[finite_rows]
    if len(efforts) == 0:
        raise SystemExit("No finite effort rows found")

    abs_effort = np.abs(efforts)
    envelope = np.quantile(abs_effort, args.quantile, axis=0)
    warning = np.maximum(envelope * args.warning_scale, args.min_warning_abs_effort)
    hard = np.maximum(envelope * args.hard_scale, args.min_hard_abs_effort)

    effort_norm = np.linalg.norm(efforts, axis=1)
    norm_envelope = float(np.quantile(effort_norm, args.quantile))
    hard_norm = max(norm_envelope * args.hard_scale, args.min_hard_abs_effort)

    print("Free-space effort summary")
    print(f"  samples: {len(efforts)}")
    print(f"  dofs: {efforts.shape[1]}")
    print(f"  abs effort q{args.quantile:g}: {_fmt_array(envelope)}")
    print(f"  warning_abs_effort: {_fmt_array(warning)}")
    print(f"  hard_abs_effort: {_fmt_array(hard)}")
    print(f"  hard_effort_norm: {hard_norm:.6g}")
    print()

    yaml = _build_yaml(warning, hard, hard_norm, args)
    print(yaml)
    if args.output_path:
        Path(args.output_path).write_text(yaml + "\n")
        print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
