from pathlib import Path

from olmo.data import lerobot_wrapper


def test_warn_frame_count_mismatch_logs_once_on_rank0_main_process(caplog, monkeypatch):
    lerobot_wrapper._FRAME_COUNT_MISMATCH_WARNING_KEYS.clear()
    monkeypatch.setattr(lerobot_wrapper, "_get_dist_rank", lambda: 0)
    monkeypatch.setattr(lerobot_wrapper.torch.utils.data, "get_worker_info", lambda: None)

    with caplog.at_level("WARNING"):
        lerobot_wrapper._warn_frame_count_mismatch(
            "Loki0929/so100_lan",
            Path("/weka/oe-training-default/molmoact/lerobot/Loki0929/so100_lan/data"),
            256044,
            256427,
        )
        lerobot_wrapper._warn_frame_count_mismatch(
            "Loki0929/so100_lan",
            Path("/weka/oe-training-default/molmoact/lerobot/Loki0929/so100_lan/data"),
            256044,
            256427,
        )

    assert len(caplog.records) == 1
    assert "metadata expects 256427" in caplog.text


def test_warn_frame_count_mismatch_suppressed_in_dataloader_workers(caplog, monkeypatch):
    lerobot_wrapper._FRAME_COUNT_MISMATCH_WARNING_KEYS.clear()
    monkeypatch.setattr(lerobot_wrapper, "_get_dist_rank", lambda: 0)
    monkeypatch.setattr(lerobot_wrapper.torch.utils.data, "get_worker_info", lambda: object())

    with caplog.at_level("WARNING"):
        lerobot_wrapper._warn_frame_count_mismatch(
            "Loki0929/so100_lan",
            Path("/weka/oe-training-default/molmoact/lerobot/Loki0929/so100_lan/data"),
            256044,
            256427,
        )

    assert not caplog.records


def test_warn_frame_count_mismatch_suppressed_on_nonzero_rank(caplog, monkeypatch):
    lerobot_wrapper._FRAME_COUNT_MISMATCH_WARNING_KEYS.clear()
    monkeypatch.setattr(lerobot_wrapper, "_get_dist_rank", lambda: 7)
    monkeypatch.setattr(lerobot_wrapper.torch.utils.data, "get_worker_info", lambda: None)

    with caplog.at_level("WARNING"):
        lerobot_wrapper._warn_frame_count_mismatch(
            "Loki0929/so100_lan",
            Path("/weka/oe-training-default/molmoact/lerobot/Loki0929/so100_lan/data"),
            256044,
            256427,
        )

    assert not caplog.records
