"""Tests for DAgger-style student rollout relabeling helpers."""

import numpy as np

from musclemimic.distill.dagger import build_dagger_shard_data
from musclemimic.distill.obs_filter import StudentObsSpec


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
