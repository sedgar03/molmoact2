from types import SimpleNamespace

import numpy as np

import olmo.data.lerobot_wrapper as lerobot_wrapper
from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper


class _TinyDataset:
    def __init__(self, row):
        self._row = row
        self.repo_id = "Loki0929/so100_lan"
        self.root = None
        self.revision = "main"
        self.meta = SimpleNamespace(camera_keys=["observation.images.cam"])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return dict(self._row)


def test_depth_companion_uses_loaded_row_position_not_frame_index(monkeypatch):
    monkeypatch.setattr(lerobot_wrapper, "build_discrete_action_string_from_action", lambda *args, **kwargs: "")
    monkeypatch.setattr(LeRobotDatasetWrapper, "_get_action_discrete_processor", lambda self: None)

    main_dataset = _TinyDataset(
        {
            "task": "reach target",
            "action": [0.1, 0.2],
            "observation.state": [0.3, 0.4],
            "observation.images.cam": np.zeros((2, 2, 3), dtype=np.uint8),
            # Simulate truncated parquet rows where stored frame index no longer matches row position.
            "index": 5,
            "frame_index": 0,
            "episode_index": 0,
        }
    )
    depth_dataset = _TinyDataset(
        {
            "buffer_codes": [1, 2, 3],
            "depth_updated_mask": [True, False, True],
            "index": 5,
            "frame_index": 0,
            "episode_index": 0,
        }
    )

    wrapper = LeRobotDatasetWrapper(
        dataset=main_dataset,
        depth_dataset=depth_dataset,
        split="train",
        camera_keys=["observation.images.cam"],
        camera_keys_alternative=None,
        question_key="task",
        state_keys=["observation.state"],
        action_keys=["action"],
        observation_indices=[0],
        action_indices=[0],
        style="robot_depth",
        action_format="both",
        state_format="continuous",
        enable_depth_reasoning=True,
        add_depth_tokens=True,
        num_depth_tokens=8,
        num_depth_tokens_per_image=3,
        discrete_action_tokenizer="allenai/MolmoAct2-FAST-Tokenizer",
        tag_action_horizon=1,
        tag_n_action_steps=1,
        max_action_horizon=1,
        max_action_dim=2,
        metadata_repo_id="Loki0929/so100_lan",
    )

    example = wrapper.get(0, np.random.default_rng(0), allow_random_retry=False)

    assert example["depth_buffer_codes"].tolist() == [1, 2, 3]
    assert example["depth_updated_mask"].tolist() == [True, False, True]
