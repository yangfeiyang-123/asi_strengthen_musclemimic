from __future__ import annotations

import numpy as np
import pytest

from environment.overall_environment.src.body_obs_adapter import (
    BodyObsAdapter,
    BodyObsCompatibilityError,
)
from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
from environment.overall_environment.src.training_scene import default_training_scene_path


CHECKPOINT = "checkpoints/de63059b16c0/checkpoint_7812"


def test_body_obs_adapter_reads_checkpoint_obs_size_and_rejects_overall_obs_size():
    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)

    report = adapter.check_compatibility(overall_obs_size=283)

    assert report.checkpoint_obs_size == 2418
    assert report.overall_obs_size == 283
    assert report.compatible is False
    assert "2418" in report.reason
    assert "283" in report.reason


def test_body_obs_adapter_adapt_requires_exact_shape_until_schema_mapping_exists():
    adapter = BodyObsAdapter(expected_obs_size=2418)

    with pytest.raises(BodyObsCompatibilityError, match="expected 2418"):
        adapter.adapt(np.zeros(283, dtype=float))

    obs = adapter.adapt(np.zeros(2418, dtype=float))
    assert obs.shape == (2418,)


def test_body_obs_adapter_reconstructs_checkpoint_schema_components():
    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)

    assert adapter.schema.total_size == 2418
    assert adapter.schema.kinematic_size == 175
    assert adapter.schema.muscle_size == 1770
    assert adapter.schema.touch_size == 4
    assert adapter.schema.goal_size == 469
    assert adapter.schema.action_size == 354
    assert list(adapter.schema.observation_names[:4]) == [
        "q_free_joint",
        "q_all_pos",
        "dq_free_joint",
        "dq_all_vel",
    ]
    assert adapter.schema.observation_names[-1] == "GoalTrajMimic"


def test_body_obs_adapter_builds_legacy_obs_from_overall_state_with_explicit_goal():
    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)
    env = OverallBadmintonEnvironment(default_training_scene_path())
    env.reset()
    goal_obs = np.zeros(adapter.schema.goal_size, dtype=float)

    obs = adapter.build_from_mujoco(env.model, env.data, goal_obs=goal_obs)

    assert obs.shape == (2418,)
    assert np.isfinite(obs).all()
    assert np.array_equal(obs[-adapter.schema.goal_size :], goal_obs)


def test_trajectory_goal_provider_builds_checkpoint_goal_from_retarget_cache():
    from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider

    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)
    env = OverallBadmintonEnvironment(default_training_scene_path())
    env.reset()
    provider = TrajectoryGoalProvider.from_checkpoint(CHECKPOINT)

    goal_obs = provider.build(env.model, env.data)
    provider.advance()
    next_goal_obs = provider.build(env.model, env.data)

    assert provider.source == "trajectory_cache"
    assert provider.motion_path == "10trajectories/video1_lower_body_full_poses"
    assert goal_obs.shape == (adapter.schema.goal_size,)
    assert np.isfinite(goal_obs).all()
    assert not np.array_equal(goal_obs, np.zeros_like(goal_obs))
    assert next_goal_obs.shape == goal_obs.shape
    assert next_goal_obs[-1] > goal_obs[-1]


def test_body_obs_adapter_requires_explicit_goal_observation():
    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)
    env = OverallBadmintonEnvironment(default_training_scene_path())
    env.reset()

    with pytest.raises(BodyObsCompatibilityError, match="goal_obs is required"):
        adapter.build_from_mujoco(env.model, env.data)

    with pytest.raises(BodyObsCompatibilityError, match="goal_obs must have shape"):
        adapter.build_from_mujoco(env.model, env.data, goal_obs=np.zeros(adapter.schema.goal_size - 1))
