from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chinajump_qc10_config_binds_existing_accepted_caches():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        cfg = compose(
            config_name=(
                "config_specific_task/stage1_body/"
                "conf_fullbody_chinajump_optimized_qc10"
            )
        )

    train = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
    val = list(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path)
    assert len(train) == 8
    assert len(val) == 2
    assert not set(train) & set(val)
    assert cfg.experiment.run_id == "chinajump_optimized_qc10_stage1_body_v1"
    assert cfg.experiment.training_action == "ChinaJump"
    assert cfg.experiment.training_source.source_fps == 60
    assert cfg.experiment.training_source.cache_fps == 100
    assert cfg.experiment.total_timesteps == 320_000_000
    assert cfg.experiment.env_params.disable_fingers is True
    assert cfg.experiment.validation.cover_all_trajectories is True
    assert cfg.experiment.promotion.auto_stop is True
    assert cfg.experiment.promotion.require_visual_validation_clips == 2

    for motion in train + val:
        cache = REPO_ROOT / "datasets" / f"{motion}.npz"
        assert cache.is_file(), cache
        with np.load(cache, allow_pickle=True) as data:
            assert float(data["frequency"]) == 100.0
            assert np.isfinite(data["qpos"]).all()
            assert np.isfinite(data["qvel"]).all()
