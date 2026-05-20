from pathlib import Path

import numpy as np
import pytest

from musclemimic.utils.root_tracking import (
    compute_root_reference_metrics,
    compute_rollout_root_metrics,
    select_latest_checkpoint,
)


def test_select_latest_checkpoint_uses_largest_numeric_suffix(tmp_path):
    root = tmp_path / "ckpts"
    root.mkdir()
    for name in ["checkpoint_10", "checkpoint_2", "checkpoint_100"]:
        (root / name).mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    assert select_latest_checkpoint(root) == root / "checkpoint_100"


def test_select_latest_checkpoint_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint root does not exist"):
        select_latest_checkpoint(tmp_path / "missing")


def test_select_latest_checkpoint_rejects_empty_checkpoint_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="no checkpoint_\\* directories"):
        select_latest_checkpoint(root)


def test_compute_root_reference_metrics_reports_displacement_and_speed():
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.3, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    qvel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_root_reference_metrics(qpos=qpos, qvel=qvel)

    assert metrics["reference_root_xy_total_displacement"] == pytest.approx(1.0)
    assert metrics["reference_root_xy_path_length"] == pytest.approx(1.0)
    assert metrics["reference_root_xy_peak_speed"] == pytest.approx(10.0)


def test_compute_rollout_root_metrics_compares_rollout_to_reference():
    reference_qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rollout_qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_rollout_root_metrics(reference_qpos=reference_qpos, rollout_qpos=rollout_qpos)

    assert metrics["reference_root_xy_total_displacement"] == pytest.approx(1.0)
    assert metrics["rollout_root_xy_total_displacement"] == pytest.approx(0.5)
    assert metrics["root_displacement_ratio"] == pytest.approx(0.5)
    assert metrics["root_xy_final_error"] == pytest.approx(0.5)
    assert metrics["root_xy_rmse"] == pytest.approx(np.sqrt((0.0**2 + 0.5**2) / 2.0))
