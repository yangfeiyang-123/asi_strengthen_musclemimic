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
    assert info["base_runtime_source"] == "checkpoint_imitation_factory"
    assert info["base_obs_size"] == 2834
    assert info["raw_body_action_max_abs"] > 0.0
    assert info["clipped_full_ctrl_max_abs"] <= 1.0


def test_overall_grip_hold_env_defaults_to_no_pose_servo():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert info["pose_servo_enabled"] is False
    assert info["servo_scope"] == "none"
    assert info["servo_force_norm_max"] == 0.0


def test_overall_grip_hold_env_defaults_to_original_body_control_dt():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert info["base_runtime"] == "musclemimic"
    assert info["control_substeps"] == 5
    assert info["physics_timestep_s"] == 0.002
    assert info["policy_control_dt_s"] == 0.01


def test_overall_grip_hold_env_reset_aligns_to_trajectory_step():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    _, info = env.reset(traj_step=12)
    reference = env.body_goal_provider.reference_state(12)

    root_qadr = env.model.jnt_qposadr[env.root_joint_id]
    root_dadr = env.model.jnt_dofadr[env.root_joint_id]
    np.testing.assert_allclose(env.data.qpos[root_qadr : root_qadr + 2], np.array([-2.5, 0.0]))
    np.testing.assert_allclose(env.data.qpos[root_qadr + 2 : root_qadr + 7], reference.qpos[2:7])
    np.testing.assert_allclose(env.data.qvel[root_dadr : root_dadr + 6], reference.qvel[:6])
    assert info["reset_mode"] == "trajectory"
    assert info["reset_traj_step"] == 12
    assert info["body_goal_next_traj_step"] == 12


def test_body_mimic_error_uses_trajectory_reference_not_reset_snapshot():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    env.reset(traj_step=12)
    env._reference_qpos = np.asarray(env.data.qpos, dtype=float) + 10.0

    assert env._body_mimic_error() < 1e-12
    assert env._info()["body_mimic_reference"] == "trajectory"


def test_racket_hand_pose_reward_preserves_reference_offset():
    env = OverallGripHoldEnv(default_training_scene_path(), residual_groups=["right_hand_fingers"])

    env.reset()
    info = env._info()
    terms = env._reward_terms(np.zeros(env.action_size, dtype=float), info)

    assert info["palm_to_grip_m"] > 0.04
    assert info["grip_site_error_m"] == 0.0
    assert info["racket_hand_pose_error_m"] == 0.0
    assert terms["r_racket_hand_pose"] == 0.0


def test_tiny_train_writes_metrics_and_checkpoint(tmp_path: Path):
    from musclemimic.badminton.scripts.run_forehand_clear_grip_hold import (
        load_grip_hold_spec,
        train_tiny,
    )

    paths = load_grip_hold_spec("experiments/posttrain/forehand_clear_grip_hold_v1.yaml")

    report = train_tiny(paths, out_dir=tmp_path, total_steps=8, rollout_steps=4, seed=0)

    assert report["runner_stage"] == "train-tiny"
    assert report["policy_source"] == "frozen_artifact"
    assert report["global_step"] == 8
    assert report["updates"] == 2
    assert report["finite"] is True
    assert report["policy_checkpoint"].endswith("policy_latest.pt")
    assert report["last_info"]["pose_servo_enabled"] is False
    assert report["last_info"]["servo_scope"] == "none"
    assert report["last_info"]["body_goal_obs_source"] == "trajectory_cache"
    assert report["last_info"]["reset_mode"] == "trajectory"
    assert (tmp_path / "policy_latest.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
