from __future__ import annotations

from pathlib import Path

import numpy as np

from environment.overall_environment.src.overall_grip_hold_env import OverallGripHoldEnv
from environment.overall_environment.src.training_scene import default_training_scene_path


def test_overall_grip_hold_env_reset_and_step_report_required_terms():
    env = OverallGripHoldEnv(default_training_scene_path(), residual_groups=["right_hand_fingers"])

    obs, reset_info = env.reset()
    action = np.zeros(env.action_size, dtype=float)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == next_obs.shape
    assert env.action_size > 0
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert reset_info["scene_action_size"] == 416
    assert info["hand_handle_contact_count"] >= 0
    assert info["grip_slip_m"] >= 0.0
    assert info["racket_drop"] is False
    assert info["body_fall"] is False
    assert set(info["reward_terms"]) == {
        "r_mimic_body",
        "r_grip_site",
        "r_contact",
        "r_no_slip",
        "r_no_penetration",
        "r_racket_hand_pose",
        "r_residual_effort",
    }


def test_overall_grip_hold_env_rejects_wrong_action_shape():
    env = OverallGripHoldEnv(default_training_scene_path(), residual_groups=["right_hand_fingers"])
    env.reset()

    try:
        env.step(np.zeros(env.action_size + 1, dtype=float))
    except ValueError as exc:
        assert "action must have shape" in str(exc)
    else:
        raise AssertionError("wrong-sized residual action should fail")


def test_overall_grip_hold_env_steps_with_frozen_body_policy_artifact():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    env.reset()
    action = np.zeros(env.action_size, dtype=float)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["body_policy_source"] == "frozen_artifact"
    assert info["body_obs_size"] == 2418
    assert info["body_action_size"] == 354
    assert info["body_goal_obs_source"] == "trajectory_cache"
    assert info["body_goal_motion_path"] == "10trajectories/video1_lower_body_full_poses"
    assert info["body_goal_traj_step"] == 0
    assert info["raw_body_action_max_abs"] > 0.0
    assert info["clipped_full_ctrl_max_abs"] <= 1.0


def test_tiny_train_writes_metrics_and_checkpoint(tmp_path: Path):
    from BadmintonMimic.scripts.run_forehand_clear_grip_hold import (
        load_grip_hold_spec,
        train_tiny,
    )

    paths = load_grip_hold_spec("BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")

    report = train_tiny(paths, out_dir=tmp_path, total_steps=8, rollout_steps=4, seed=0)

    assert report["runner_stage"] == "train-tiny"
    assert report["policy_source"] == "frozen_artifact"
    assert report["global_step"] == 8
    assert report["updates"] == 2
    assert report["finite"] is True
    assert report["policy_checkpoint"].endswith("policy_latest.pt")
    assert (tmp_path / "policy_latest.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
