from __future__ import annotations

from pathlib import Path

from BadmintonMimic.scripts.run_forehand_clear_grip_hold import (
    checkpoint_metadata,
    diagnostic_reset,
    load_grip_hold_spec,
    preflight,
    replay_precheck,
)


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")


def test_load_grip_hold_spec_resolves_paths():
    paths = load_grip_hold_spec(SPEC)

    assert paths.runner_type == "forehand_clear_grip_hold"
    assert paths.resume_from.is_dir()
    assert paths.scene_xml.is_file()
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
    assert report["policy_replay_ready"] is False
    assert "action adapter" in report["blocked_reason"]
    assert (tmp_path / "replay_precheck_report.json").is_file()
