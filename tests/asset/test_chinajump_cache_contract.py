"""Prepared ChinaJump cache contract checks kept out of source-only CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.asset


def test_chinajump_qc10_caches_exist_and_are_finite():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "fullbody")):
        cfg = compose(config_name=("config_specific_task/stage1_body/conf_fullbody_chinajump_optimized_qc10"))

    train = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
    validation = list(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path)
    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train) & set(validation)

    for motion in train + validation:
        cache = ROOT / "datasets" / f"{motion}.npz"
        assert cache.is_file(), cache
        with np.load(cache, allow_pickle=False) as data:
            required = {"frequency", "qpos", "qvel"}
            assert required <= set(data.files), (cache, data.files)
            assert float(np.asarray(data["frequency"]).item()) == 100.0
            qpos = np.asarray(data["qpos"])
            qvel = np.asarray(data["qvel"])
            assert qpos.ndim == 2
            assert qvel.ndim == 2
            assert qpos.shape[0] == qvel.shape[0]
            assert qpos.shape[0] > 1
            assert np.isfinite(qpos).all()
            assert np.isfinite(qvel).all()
