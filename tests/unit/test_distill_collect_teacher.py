"""Tests for teacher rollout collection helpers."""

from omegaconf import OmegaConf

import musclemimic.distill as distill
from musclemimic.distill.collect_teacher import build_teacher_rollout_config


def test_build_teacher_rollout_config_disables_student_filter():
    exp = OmegaConf.create(
        {
            "num_envs": 1,
            "normalize_env": False,
            "gamma": 0.99,
            "student_obs_filter": {"enabled": True},
        }
    )

    rollout_cfg = build_teacher_rollout_config(exp, num_envs=8)

    assert rollout_cfg.num_envs == 8
    assert rollout_cfg.student_obs_filter.enabled is False


def test_distill_package_lazy_exports_public_functions():
    assert distill.bc_loss is not None
    assert distill.collect_teacher_dataset is not None
    assert distill.collect_dagger_dataset is not None
    assert distill.train_bc is not None
