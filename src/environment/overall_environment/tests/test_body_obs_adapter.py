from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from environment.overall_environment.src.body_obs_adapter import (
    BodyObsAdapter,
    BodyObsCompatibilityError,
)
from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
from environment.overall_environment.src.training_scene import default_training_scene_path


CHECKPOINT = "checkpoints/de63059b16c0/checkpoint_7812"


@pytest.fixture
def trajectory_cache_root(tmp_path: Path) -> Path:
    """Materialize the deleted legacy motion name from a current 89/88 cache.

    The checkpoint fixture still carries the historical
    ``10trajectories/video1_lower_body_full_poses`` identifier, but that cache
    is no longer distributed.  These tests exercise goal ABI/root
    normalization rather than the old motion content, so bind the identifier
    to the canonical current MyoFullBody cache in an isolated test root.
    """

    source = Path(
        "datasets/forehandClear_standard/muscle_trajectory/raw_smooth_v1/video1.npz"
    )
    if not source.is_file():
        pytest.fail(f"canonical trajectory test cache is missing: {source}")
    target = (
        tmp_path
        / "MyoFullBody"
        / "gmr"
        / "10trajectories"
        / "video1_lower_body_full_poses.npz"
    )
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    return tmp_path


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


def test_trajectory_goal_provider_builds_checkpoint_goal_from_retarget_cache(
    trajectory_cache_root: Path,
):
    from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider

    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)
    env = OverallBadmintonEnvironment(default_training_scene_path())
    env.reset()
    provider = TrajectoryGoalProvider.from_checkpoint(
        CHECKPOINT, cache_root=trajectory_cache_root
    )

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


def test_trajectory_goal_provider_reference_state_matches_training_root_xy_normalization(
    trajectory_cache_root: Path,
):
    from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider

    provider = TrajectoryGoalProvider.from_checkpoint(
        CHECKPOINT, cache_root=trajectory_cache_root
    )

    reference0 = provider.reference_state(0)
    reference12 = provider.reference_state(12)
    raw0 = provider._trajectory_handler.traj.data.get(0, 0, np).qpos
    raw12 = provider._trajectory_handler.traj.data.get(0, 12, np).qpos

    np.testing.assert_allclose(reference0.qpos[:2], np.zeros(2), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(reference12.qpos[:2], raw12[:2] - raw0[:2], rtol=0.0, atol=1e-8)
    assert np.isfinite(reference12.qpos).all()


def test_body_obs_adapter_requires_explicit_goal_observation():
    adapter = BodyObsAdapter.from_checkpoint(CHECKPOINT)
    env = OverallBadmintonEnvironment(default_training_scene_path())
    env.reset()

    with pytest.raises(BodyObsCompatibilityError, match="goal_obs is required"):
        adapter.build_from_mujoco(env.model, env.data)

    with pytest.raises(BodyObsCompatibilityError, match="goal_obs must have shape"):
        adapter.build_from_mujoco(env.model, env.data, goal_obs=np.zeros(adapter.schema.goal_size - 1))


def test_body_obs_adapter_honors_direct_student_state_plus_phase_schema(tmp_path):
    source = Path(CHECKPOINT)
    checkpoint = tmp_path / "student"
    (checkpoint / "config").mkdir(parents=True)
    (checkpoint / "train_state").mkdir(parents=True)
    metadata = json.loads((source / "config" / "metadata").read_text(encoding="utf-8"))
    metadata["experiment"]["student_obs_filter"] = {
        "enabled": True,
        "drop_goal_lookahead": True,
        "keep_motion_phase": True,
        "require_goal_group": True,
        "require_motion_phase": True,
    }
    (checkpoint / "config" / "metadata").write_text(json.dumps(metadata), encoding="utf-8")
    orbax_metadata = {
        "tree_metadata": {
            "('run_stats', 'RunningMeanStd_0', 'mean')": {
                "value_metadata": {"write_shape": [1950]}
            }
        }
    }
    (checkpoint / "train_state" / "_METADATA").write_text(
        json.dumps(orbax_metadata), encoding="utf-8"
    )

    adapter = BodyObsAdapter.from_checkpoint(checkpoint)

    assert adapter.expected_obs_size == 1950
    assert adapter.schema.student_filtered is True
    assert adapter.schema.goal_size == 1
    assert adapter.schema.total_size == 1950
    assert adapter.schema.observation_names[-1] == "motion_phase"
