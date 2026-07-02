"""Copy an HDF5 robot log while excluding relative time windows.

The input must contain either ``timestamp`` or ``monotonic_time``. Windows are
specified in seconds relative to the first sample, for example:

    --exclude 120:126 --exclude 209:211
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import tyro


@dataclass
class Args:
    input_path: Annotated[str, tyro.conf.Positional]
    """Input HDF5 path."""

    output_path: Annotated[str, tyro.conf.Positional]
    """Output HDF5 path."""

    exclude: list[str] = field(default_factory=list)
    """Relative time windows START:END to exclude, in seconds."""

    time_key: str = ""
    """Time dataset to use. Defaults to monotonic_time, then timestamp."""


def _parse_window(spec: str) -> tuple[float, float]:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid window {spec!r}; expected START:END")
    start = float(parts[0])
    end = float(parts[1])
    if end <= start:
        raise ValueError(f"Invalid window {spec!r}; END must be greater than START")
    return start, end


def _time_vector(f: h5py.File, key: str) -> np.ndarray:
    if key:
        if key not in f:
            raise KeyError(f"Missing time dataset {key!r}")
        return np.asarray(f[key][:], dtype=np.float64)
    if "monotonic_time" in f:
        return np.asarray(f["monotonic_time"][:], dtype=np.float64)
    if "timestamp" in f:
        return np.asarray(f["timestamp"][:], dtype=np.float64)
    raise KeyError("Input must contain monotonic_time or timestamp")


def main() -> None:
    args = tyro.cli(Args)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    windows = [_parse_window(spec) for spec in args.exclude]
    with h5py.File(input_path, "r") as src:
        t = _time_vector(src, args.time_key)
        rel = t - t[0]
        keep = np.ones(len(t), dtype=bool)
        for start, end in windows:
            keep &= ~((rel >= start) & (rel <= end))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["trim_source_path"] = str(input_path)
            dst.attrs["trim_exclude_windows_sec"] = ",".join(args.exclude)
            dst.attrs["trim_kept_samples"] = int(keep.sum())
            dst.attrs["trim_dropped_samples"] = int((~keep).sum())

            for key, dataset in src.items():
                data = dataset[:]
                if len(data) == len(keep):
                    data = data[keep]
                kwargs = {}
                if data.ndim > 0 and key not in ("timestamp", "monotonic_time"):
                    kwargs = {"compression": "gzip", "compression_opts": 4}
                dst.create_dataset(key, data=data, **kwargs)

    print(
        f"Wrote {output_path}: kept {int(keep.sum())} / {len(keep)} samples, "
        f"dropped {int((~keep).sum())}"
    )


if __name__ == "__main__":
    main()
