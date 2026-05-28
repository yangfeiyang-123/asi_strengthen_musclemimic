from __future__ import annotations

from pathlib import Path

from BadmintonMimic.scripts.run_forehand_clear_grip_hold import (
    diagnostic_reset,
    load_grip_hold_spec,
    preflight,
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
