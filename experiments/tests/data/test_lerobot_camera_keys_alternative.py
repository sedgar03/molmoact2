from types import SimpleNamespace

import numpy as np
import pytest

from olmo.data.lerobot_wrapper import (
    LeRobotDatasetWrapper,
    _extract_vector,
    _require_tag_metadata_entry,
    _resolve_repo_camera_keys,
)


PRIMARY_CAMERA_KEYS = [
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
]
ALTERNATIVE_CAMERA_KEYS = [
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]


class _DummyLeRobotDataset:
    def __init__(self):
        self.repo_id = "allenai/droid_lerobot"
        self.root = None
        self.revision = "main"
        self.meta = SimpleNamespace(
            camera_keys=[
                "observation.images.exterior_1_left",
                "observation.images.exterior_2_left",
                "observation.images.wrist_left",
            ]
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise AssertionError("This test should not fetch real frames.")


class _FixedChoiceRng:
    def __init__(self, choice: int):
        self.choice = int(choice)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            high = low
            low = 0
        assert low == 0
        assert high == 2
        assert size is None
        return self.choice


def _build_wrapper(split: str) -> LeRobotDatasetWrapper:
    return LeRobotDatasetWrapper(
        dataset=_DummyLeRobotDataset(),
        split=split,
        camera_keys=PRIMARY_CAMERA_KEYS,
        camera_keys_alternative=ALTERNATIVE_CAMERA_KEYS,
        question_key="task",
        state_keys=["observation.state"],
        action_keys=["action"],
        observation_indices=[0],
        action_indices=[0],
        tag_action_horizon=2,
        tag_n_action_steps=1,
        max_action_horizon=2,
    )


def test_require_tag_metadata_entry_accepts_camera_keys_alternative():
    metadata = _require_tag_metadata_entry(
        tag_metadata_by_tag={
            "franka_droid": {
                "action_key": "action",
                "state_keys": ["observation.state"],
                "camera_keys": PRIMARY_CAMERA_KEYS,
                "camera_keys_alternative": ALTERNATIVE_CAMERA_KEYS,
                "normalize_gripper": False,
                "action_horizon": 15,
                "n_action_steps": 15,
                "setup_type": "single franka robotic arm in droid",
                "control_mode": "absolute joint pose",
            }
        },
        repo_to_tag={"allenai/droid_lerobot": "franka_droid"},
        repo_id="allenai/droid_lerobot",
    )

    assert metadata["camera_keys"] == PRIMARY_CAMERA_KEYS
    assert metadata["camera_keys_alternative"] == ALTERNATIVE_CAMERA_KEYS
    assert metadata["state_keys"] == ["observation.state"]


def test_extract_vector_concatenates_configured_state_keys_in_order():
    state, used_keys = _extract_vector(
        {
            "observation.state.arm": np.asarray([1.0, 2.0], dtype=np.float32),
            "observation.state.gripper": np.asarray([3.0], dtype=np.float32),
        },
        ["observation.state.arm", "observation.state.gripper"],
        "observation.state.",
    )

    np.testing.assert_allclose(state, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert used_keys == ["observation.state.arm", "observation.state.gripper"]


def test_train_split_can_sample_either_camera_key_configuration():
    wrapper = _build_wrapper(split="train")
    example = {key: object() for key in set(PRIMARY_CAMERA_KEYS + ALTERNATIVE_CAMERA_KEYS)}

    primary_keys_used, primary_permutation = wrapper._resolve_camera_order(
        example,
        _FixedChoiceRng(0),
        mapped_index=0,
    )
    alternative_keys_used, alternative_permutation = wrapper._resolve_camera_order(
        example,
        _FixedChoiceRng(1),
        mapped_index=0,
    )

    assert primary_keys_used == PRIMARY_CAMERA_KEYS
    assert primary_permutation == [0, 1]
    assert alternative_keys_used == ALTERNATIVE_CAMERA_KEYS
    assert alternative_permutation == [0, 1]


def test_train_split_accepts_random_state_for_camera_key_sampling():
    wrapper = _build_wrapper(split="train")
    example = {key: object() for key in set(PRIMARY_CAMERA_KEYS + ALTERNATIVE_CAMERA_KEYS)}

    camera_keys_used, permutation = wrapper._resolve_camera_order(
        example,
        np.random.RandomState(0),
        mapped_index=0,
    )

    assert camera_keys_used in (PRIMARY_CAMERA_KEYS, ALTERNATIVE_CAMERA_KEYS)
    assert permutation == [0, 1]


def test_non_train_split_keeps_primary_camera_key_configuration():
    wrapper = _build_wrapper(split="validation")
    example = {key: object() for key in set(PRIMARY_CAMERA_KEYS + ALTERNATIVE_CAMERA_KEYS)}

    camera_keys_used, permutation = wrapper._resolve_camera_order(
        example,
        _FixedChoiceRng(1),
        mapped_index=0,
    )

    assert camera_keys_used == PRIMARY_CAMERA_KEYS
    assert permutation == [0, 1]


def test_camera_keys_alternative_requires_primary_camera_keys():
    with pytest.raises(ValueError, match="camera_keys list when camera_keys_alternative is provided"):
        _require_tag_metadata_entry(
            tag_metadata_by_tag={
                "franka_droid": {
                    "action_key": "action",
                    "state_keys": ["observation.state"],
                    "camera_keys_alternative": ALTERNATIVE_CAMERA_KEYS,
                    "normalize_gripper": False,
                    "action_horizon": 15,
                    "n_action_steps": 15,
                    "setup_type": "single franka robotic arm in droid",
                    "control_mode": "absolute joint pose",
                }
            },
            repo_to_tag={"allenai/droid_lerobot": "franka_droid"},
            repo_id="allenai/droid_lerobot",
            random_camera_order="all",
        )


def test_yam_dual_standard_camera_keys_resolve_to_legacy_repo_keys():
    standard_keys = [
        "observation.images.top",
        "observation.images.left",
        "observation.images.right",
    ]
    legacy_keys = [
        "observation.images.camera_front",
        "observation.images.camera_left",
        "observation.images.camera_right",
    ]

    assert _resolve_repo_camera_keys(
        "yam_dual_molmoact2",
        available_cameras=legacy_keys,
        camera_keys=standard_keys,
    ) == legacy_keys
    assert _resolve_repo_camera_keys(
        "lerobot:yam_dual_molmoact2",
        available_cameras=standard_keys,
        camera_keys=standard_keys,
    ) == standard_keys
    assert _resolve_repo_camera_keys(
        "franka_droid",
        available_cameras=legacy_keys,
        camera_keys=standard_keys,
    ) == standard_keys
