from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from BadmintonMimic.scripts.run_forehand_clear_grip_hold import (
    checkpoint_metadata,
    diagnostic_reset,
    load_grip_hold_spec,
    preflight,
    replay_precheck,
    replay_smoke,
)


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")


def test_load_grip_hold_spec_resolves_paths():
    paths = load_grip_hold_spec(SPEC)

    assert paths.runner_type == "forehand_clear_grip_hold"
    assert paths.resume_from.is_dir()
    assert paths.body_policy_artifact is not None
    assert paths.body_policy_artifact.is_dir()
    assert paths.scene_xml.is_file()
    assert paths.scene_xml.name == "overall_badminton_training_scene.xml"
    assert paths.grip_seed.is_file()


def test_preflight_writes_report(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)
    report = preflight(paths, out_dir=tmp_path)

    assert report["runner_type"] == "forehand_clear_grip_hold"
    assert report["checkpoint_exists"] is True
    assert report["scene_exists"] is True
    assert report["grip_seed_exists"] is True
    assert (tmp_path / "preflight_report.json").is_file()


def test_diagnostic_reset_records_video_with_injected_recorder(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)

    def fake_recorder(*, paths, out_dir):
        video_path = out_dir / "diagnostics" / "reset_grip_hold.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake mp4")
        return video_path

    report = diagnostic_reset(paths, out_dir=tmp_path, recorder=fake_recorder)

    assert report["checkpoint_exists"] is True
    assert report["reset_video"].endswith("diagnostics/reset_grip_hold.mp4")
    assert (tmp_path / "diagnostics" / "reset_grip_hold.mp4").is_file()
    assert (tmp_path / "diagnostic_reset_report.json").is_file()


def test_checkpoint_metadata_identifies_forehand_clear_checkpoint():
    paths = load_grip_hold_spec(SPEC)

    metadata = checkpoint_metadata(paths.resume_from)

    assert metadata["wandb"]["project"] == "musclemimic"
    assert "forehand_clear" in metadata["wandb"]["tags"]
    assert metadata["experiment"]["env_params"]["env_name"] == "MjxMyoFullBody"


def test_replay_precheck_writes_report_without_running_policy(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)

    report = replay_precheck(paths, out_dir=tmp_path)

    assert report["checkpoint_exists"] is True
    assert report["checkpoint_tags_match"] is True
    assert report["base_env_name"] == "MjxMyoFullBody"
    assert report["base_disable_fingers"] is True
    assert report["runner_stage"] == "replay-precheck"
    assert report["checkpoint_action_size"] == 354
    assert report["scene_action_size"] == 416
    assert report["action_adapter_ready"] is True
    assert report["adapter_mapped_count"] == 354
    assert report["adapter_extra_in_target_count"] == 62
    assert report["policy_replay_ready"] is False
    assert "action adapter" not in report["blocked_reason"]
    assert (tmp_path / "replay_precheck_report.json").is_file()


def test_replay_smoke_runs_fake_body_policy_on_training_scene(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)

    report = replay_smoke(paths, out_dir=tmp_path, steps=5)

    assert report["runner_stage"] == "replay-smoke"
    assert report["policy_source"] == "fake"
    assert report["scene_validation_ready"] is True
    assert report["fake_policy_replay_ready"] is True
    assert report["policy_replay_ready"] is False
    assert report["steps_requested"] == 5
    assert report["steps_completed"] == 5
    assert report["finite"] is True
    assert report["racket_drop"] is False
    assert report["scene_action_size"] == 416
    assert report["adapter_mapped_count"] == 354
    assert (tmp_path / "replay_smoke_report.json").is_file()


def test_replay_smoke_real_policy_uses_obs_adapter_before_actor_restore(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)

    report = replay_smoke(paths, out_dir=tmp_path, steps=5, policy_source="real")

    assert report["runner_stage"] == "replay-smoke"
    assert report["policy_source"] == "real"
    assert report["fake_policy_replay_ready"] is False
    assert report["policy_replay_ready"] is True
    assert report["steps_completed"] == 5
    assert report["finite"] is True
    assert report["overall_obs_size"] > 0
    assert report["body_obs_compatibility"]["checkpoint_obs_size"] == 2418
    assert report["body_obs_compatibility"]["compatible"] is False
    assert report["actor_checkpoint_shapes"]["valid"] is True
    assert report["actor_checkpoint_shapes"]["actor_output_kernel_shape"] == (1024, 354)
    assert report["actor_checkpoint_shapes"]["log_std_shape"] == (354,)
    assert report["body_policy_artifact_exists"] is True
    assert report["body_obs_size"] == 2418
    assert report["body_action_size"] == 354
    assert report["scene_action_size"] == 416
    assert report["goal_obs_source"] == "trajectory_cache"
    assert report["goal_motion_path"] == "10trajectories/video1_lower_body_full_poses"
    assert report["goal_traj_step"] >= 4
    assert report["raw_body_action_max_abs"] > 0.0
    assert report["clipped_full_action_max_abs"] <= 1.0
    assert report["blocked_reason"] == ""
    assert (tmp_path / "replay_smoke_report.json").is_file()


def test_grip_hold_runner_help_is_diagnostic_only():
    result = subprocess.run(
        [
            sys.executable,
            "BadmintonMimic/scripts/run_forehand_clear_grip_hold.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.lower()
    assert "{preflight,reset-video,replay-precheck,replay-smoke,train-tiny}" in stdout
    assert "{preflight,reset-video,replay-precheck,replay-smoke,train}" not in stdout
