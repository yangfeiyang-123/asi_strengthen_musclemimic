from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musclemimic.badminton.skill_pipeline.stage_local_trajectories import (  # noqa: E402
    SKILL_NAMESPACE,
    stage_action,
)

ACTION = "forehandClear_standard"
ACTION_DIR = REPO_ROOT / "datasets" / ACTION / "muscle_trajectory" / "optimized"

pytestmark = pytest.mark.skipif(
    not ACTION_DIR.is_dir(),
    reason=f"local optimized trajectories for {ACTION} not present",
)


def test_stage_action_symlinks_and_splits(tmp_path) -> None:
    report = stage_action(
        ACTION,
        cache_root=tmp_path,
        link_mode="symlink",
        validate=False,
        val_ratio=0.2,
        seed=0,
        max_clips=6,
    )
    assert report["num_clips"] == 6
    # rel paths follow skill/<action>/<clip>
    for rel in report["train"] + report["val"]:
        assert rel.startswith(f"{SKILL_NAMESPACE}/{ACTION}/")
        staged = tmp_path / f"{rel}.npz"
        assert staged.is_symlink()
        assert staged.resolve().is_file()
    # train/val disjoint and cover all clips
    assert set(report["train"]).isdisjoint(report["val"])
    assert len(set(report["train"]) | set(report["val"])) == 6


def test_stage_action_validate_reports_trajectory_shape(tmp_path) -> None:
    report = stage_action(ACTION, cache_root=tmp_path, validate=True, max_clips=1)
    clip = report["clips"][0]
    assert clip["qpos_dim"] == 89  # MyoFullBody
    assert clip["frequency_hz"] > 0
    assert clip["njnt"] > 0
