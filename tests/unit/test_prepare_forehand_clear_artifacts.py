from __future__ import annotations

from pathlib import Path

from BadmintonMimic.scripts.prepare_forehand_clear_grip_hold_artifacts import prepare_artifacts


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")


def test_prepare_forehand_clear_artifacts_reports_ready_dependencies(tmp_path: Path):
    report = prepare_artifacts(
        SPEC,
        out_dir=tmp_path,
        check_trajectory_cache=True,
        check_grip_seed=True,
    )

    assert report["artifact_source_checkpoint_matches_spec"] is True
    assert report["params_npz_exists"] is True
    assert report["run_stats_npz_exists"] is True
    assert report["training_scene_ok"] is True
    assert report["actuation_enabled"] is True
    assert report["hand_racket_contact_allowed"] is True
    assert report["trajectory_cache_ok"] is True
    assert report["grip_seed_ok"] is True
    assert report["body_obs_size"] == 2418
    assert report["goal_size"] == 469
    assert report["body_action_size"] == 354
    assert report["overall_action_size"] == 416
    assert (tmp_path / "prepare_artifacts_report.json").is_file()
