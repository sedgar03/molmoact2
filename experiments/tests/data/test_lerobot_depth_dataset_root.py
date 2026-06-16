from pathlib import Path

from olmo.data.lerobot_wrapper import _resolve_depth_dataset_root


def test_resolve_depth_dataset_root_prefers_existing_plain_repo(tmp_path: Path):
    root = tmp_path / "depth"
    plain = root / "owner" / "dataset"
    plain.mkdir(parents=True)
    suffixed = root / "owner" / "dataset_depth"
    suffixed.mkdir(parents=True)

    assert _resolve_depth_dataset_root(str(root), "owner/dataset") == plain


def test_resolve_depth_dataset_root_accepts_generator_default_suffix(tmp_path: Path):
    root = tmp_path / "depth"
    suffixed = root / "owner" / "dataset_depth"
    suffixed.mkdir(parents=True)

    assert _resolve_depth_dataset_root(str(root), "owner/dataset") == suffixed


def test_resolve_depth_dataset_root_keeps_plain_path_when_missing(tmp_path: Path):
    root = tmp_path / "depth"

    assert _resolve_depth_dataset_root(str(root), "owner/dataset") == root / "owner" / "dataset"
