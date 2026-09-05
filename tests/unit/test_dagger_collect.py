"""Tests for DAgger-style student rollout relabeling helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.distill.dagger import build_dagger_shard_data, collect_dagger_dataset
from musclemimic.distill.obs_filter import StudentObsSpec


def test_dagger_collector_rejects_mixed_policy_coordinate_abis(tmp_path):
    direct = SimpleNamespace(
        config=OmegaConf.create(
            {
                "experiment": {
                    "len_obs_history": 1,
                    "action_representation": {"mode": "full_354"},
                }
            }
        )
    )
    synergy = SimpleNamespace(
        config=OmegaConf.create(
            {
                "experiment": {
                    "len_obs_history": 1,
                    "action_representation": {"mode": "fixed_synergy"},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="action modes differ"):
        collect_dagger_dataset(
            env=object(),
            teacher_agent_conf=direct,
            teacher_agent_state=None,
            student_agent_conf=synergy,
            student_agent_state=None,
            output_dir=tmp_path,
            num_envs=1,
            num_steps=1,
        )


def test_build_dagger_shard_data_uses_student_visited_obs_and_teacher_mean_label():
    spec = StudentObsSpec(
        raw_obs_dim=5,
        goal_indices=np.array([3, 4]),
        state_indices=np.array([0, 1, 2]),
        student_indices=np.array([0, 1, 2, 4]),
        phase_index=4,
    )
    full_obs = np.array(
        [
            [1.0, 2.0, 3.0, 100.0, 0.25],
            [4.0, 5.0, 6.0, 200.0, 0.50],
        ],
        dtype=np.float32,
    )
    teacher_mu = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    student_action = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    rollout_action = np.array([[0.1, 0.2], [0.7, 0.6]], dtype=np.float32)
    used_teacher_action = np.array([True, False])
    reward = np.array([1.0, 0.5], dtype=np.float32)
    done = np.array([False, True])
    absorbing = np.array([False, True])
    info = {
        "traj_no": np.array([2, 3], dtype=np.int32),
        "subtraj_step_no": np.array([10, 20], dtype=np.int32),
    }

    data = build_dagger_shard_data(
        full_obs=full_obs,
        teacher_mu=teacher_mu,
        student_action=student_action,
        rollout_action=rollout_action,
        used_teacher_action=used_teacher_action,
        reward=reward,
        done=done,
        absorbing=absorbing,
        info=info,
        spec=spec,
        teacher_log_prob_teacher_mu=np.array([-0.1, -0.2], dtype=np.float32),
        teacher_log_prob_student_action=np.array([-1.1, -1.2], dtype=np.float32),
        teacher_log_prob_rollout_action=np.array([-0.1, -1.2], dtype=np.float32),
        teacher_log_std=np.zeros((2, 2), dtype=np.float32),
        save_full_obs=True,
    )

    np.testing.assert_allclose(
        data["student_obs"],
        np.array([[1.0, 2.0, 3.0, 0.25], [4.0, 5.0, 6.0, 0.50]], dtype=np.float32),
    )
    np.testing.assert_allclose(data["teacher_action"], teacher_mu)
    np.testing.assert_allclose(data["student_action"], student_action)
    np.testing.assert_allclose(data["rollout_action"], rollout_action)
    np.testing.assert_array_equal(data["used_teacher_action"], used_teacher_action)
    np.testing.assert_allclose(data["teacher_log_prob_teacher_mu"], np.array([-0.1, -0.2], dtype=np.float32))
    np.testing.assert_allclose(data["teacher_log_prob_student_action"], np.array([-1.1, -1.2], dtype=np.float32))
    np.testing.assert_allclose(data["teacher_log_prob_rollout_action"], np.array([-0.1, -1.2], dtype=np.float32))
    np.testing.assert_allclose(data["teacher_log_std"], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(data["phase"], np.array([0.25, 0.50], dtype=np.float32))
    np.testing.assert_array_equal(data["traj_no"], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(data["subtraj_step_no"], np.array([10, 20], dtype=np.int32))
    np.testing.assert_allclose(data["full_obs"], full_obs)


def test_dagger_teacher_target_is_clipped_applied_action_but_raw_mean_is_preserved():
    spec = StudentObsSpec(
        raw_obs_dim=3,
        goal_indices=np.array([2]),
        state_indices=np.array([0, 1]),
        student_indices=np.array([0, 1, 2]),
        phase_index=2,
    )
    raw_mu = np.array([[1.4, -1.2], [0.2, -0.3]], dtype=np.float32)

    data = build_dagger_shard_data(
        full_obs=np.zeros((2, 3), dtype=np.float32),
        teacher_mu=raw_mu,
        student_action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        done=np.zeros(2, dtype=bool),
        absorbing=np.zeros(2, dtype=bool),
        info={},
        spec=spec,
    )

    np.testing.assert_allclose(data["teacher_mu"], raw_mu)
    np.testing.assert_allclose(
        data["teacher_action"],
        np.array([[1.0, -1.0], [0.2, -0.3]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        data["teacher_raw_mean_saturation_fraction"],
        np.array([1.0, 0.0], dtype=np.float32),
    )


def test_dagger_dual_schema_keeps_decoded_body_and_c_rho_coordinates_separate():
    spec = StudentObsSpec(
        raw_obs_dim=3,
        goal_indices=np.array([2]),
        state_indices=np.array([0, 1]),
        student_indices=np.array([0, 1, 2]),
        phase_index=2,
    )
    body = np.array([[0.2, -0.1]], dtype=np.float32)
    policy_mu = np.array([[1.5, -0.4, 0.3]], dtype=np.float32)
    policy_action = np.clip(policy_mu, -1.0, 1.0)
    policy_log_std = np.array([[-0.2, -0.3, -0.4]], dtype=np.float32)

    data = build_dagger_shard_data(
        full_obs=np.zeros((1, 3), dtype=np.float32),
        teacher_mu=body,
        teacher_action=body,
        student_action=body,
        rollout_action=body,
        reward=np.zeros(1, dtype=np.float32),
        done=np.zeros(1, dtype=bool),
        absorbing=np.zeros(1, dtype=bool),
        info={},
        spec=spec,
        teacher_policy_mu=policy_mu,
        teacher_policy_action=policy_action,
        teacher_policy_log_std=policy_log_std,
        student_policy_action=policy_action,
        rollout_policy_action=policy_action,
        teacher_synergy_coefficients=np.array([[0.7, 0.2]], dtype=np.float32),
        teacher_residual_coefficients=np.array([[0.03]], dtype=np.float32),
    )

    assert data["teacher_action"].shape == (1, 2)
    assert data["teacher_policy_action"].shape == (1, 3)
    np.testing.assert_allclose(data["teacher_mu"], body)
    assert "teacher_log_std" not in data
    np.testing.assert_allclose(data["teacher_policy_mu"], policy_mu)
    np.testing.assert_allclose(data["teacher_policy_log_std"], policy_log_std)
    np.testing.assert_allclose(
        data["teacher_synergy_coefficients"], [[0.7, 0.2]]
    )
    np.testing.assert_allclose(data["teacher_residual_coefficients"], [[0.03]])
