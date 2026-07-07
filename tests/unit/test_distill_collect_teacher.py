"""Tests for teacher rollout collection helpers."""

from omegaconf import OmegaConf
import numpy as np

import musclemimic.distill as distill
from musclemimic.distill.collect_teacher import build_teacher_rollout_config
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.dagger import build_dagger_shard_data
from musclemimic.distill.obs_filter import StudentObsSpec, extract_reference_features


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


def test_distill_collect_cli_defaults_to_teacher_mean_actions():
    from fullbody.distill_collect import build_parser

    args = build_parser().parse_args(
        [
            "--teacher_ckpt",
            "/ckpt/teacher",
            "--output_dir",
            "/tmp/distill",
        ]
    )

    assert args.teacher_action_target == "mean"
    assert args.deterministic_teacher is True


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


def test_extract_reference_features_uses_dropped_goal_lookahead_without_phase():
    spec = StudentObsSpec(
        raw_obs_dim=8,
        goal_indices=np.array([4, 5, 6, 7]),
        state_indices=np.array([0, 1, 2, 3]),
        student_indices=np.array([0, 1, 2, 3, 7]),
        phase_index=7,
    )
    obs = np.arange(16, dtype=np.float32).reshape(2, 8)

    reference = extract_reference_features(obs, spec)

    np.testing.assert_array_equal(reference, obs[:, [4, 5, 6]])


def test_build_dagger_shard_data_can_include_reference_features():
    spec = StudentObsSpec(
        raw_obs_dim=6,
        goal_indices=np.array([3, 4, 5]),
        state_indices=np.array([0, 1, 2]),
        student_indices=np.array([0, 1, 2, 5]),
        phase_index=5,
    )
    full_obs = np.arange(12, dtype=np.float32).reshape(2, 6)

    data = build_dagger_shard_data(
        full_obs=full_obs,
        teacher_mu=np.zeros((2, 2), dtype=np.float32),
        student_action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        done=np.zeros((2,), dtype=bool),
        absorbing=np.zeros((2,), dtype=bool),
        info={},
        spec=spec,
        save_reference_features=True,
    )

    np.testing.assert_array_equal(data["reference_features"], full_obs[:, [3, 4]])
