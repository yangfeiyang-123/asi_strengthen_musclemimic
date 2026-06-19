"""Tests for teacher rollout collection helpers."""

from omegaconf import OmegaConf

import musclemimic.distill as distill
from musclemimic.distill.collect_teacher import build_teacher_rollout_config
from musclemimic.distill.config_overrides import apply_collection_overrides


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


def test_apply_collection_overrides_supports_motion_group_and_fixed_start():
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {},
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["old_motion"],
                            "dataset_group": "OLD",
                        }
                    }
                },
            }
        }
    )

    apply_collection_overrides(
        config,
        motion_group="KIT_TEST",
        traj_index=3,
        traj_start_step=12,
    )

    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    assert dataset_conf.dataset_group == "KIT_TEST"
    assert dataset_conf.rel_dataset_path == ["old_motion"]
    assert config.experiment.env_params.th_params.fixed_start_conf == [3, 12]
    assert config.experiment.env_params.th_params.random_start is False


def test_apply_collection_overrides_motion_path_takes_precedence_over_group():
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {},
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["old_motion"],
                            "dataset_group": "OLD",
                        }
                    }
                },
            }
        }
    )

    apply_collection_overrides(config, motion_path=["new_motion"], motion_group="IGNORED")

    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    assert dataset_conf.rel_dataset_path == ["new_motion"]
    assert dataset_conf.dataset_group is None
