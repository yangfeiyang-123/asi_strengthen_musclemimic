from __future__ import annotations

import json

import pytest

from fullbody.run_forehand_clear_pipeline import (
    DATASET_ROOT,
    PipelineArtifacts,
    _require_metrics_gate,
    build_pipeline_plan,
    execute_pipeline_step,
)
from musclemimic.badminton.data_qc import TRAIN_MOTIONS, VAL_MOTIONS
from musclemimic.badminton.visual_review import (
    REVIEW_SCHEMA_VERSION,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
)


def _write_visual_review(path, *, passed=True, review_kind=STAGE1_REVIEW_KIND):
    def _clip(index, motion):
        clip = {
            "review_kind": review_kind,
            "motion": f"forehandClear_standard/muscle_trajectory/raw_smooth_v1/{motion}",
            "artifact": f"videos/{index}.mp4",
            "major_swing_complete": passed,
            "root_tracking_spike_free": passed,
            "right_hand_tracking_spike_free": passed,
            "passed": passed,
            "notes": "manual review completed",
        }
        if review_kind == STAGE2_REVIEW_KIND:
            clip.update(
                {
                    "racket_head_trajectory_ok": passed,
                    "racket_face_orientation_ok": passed,
                }
            )
        return clip

    path.write_text(
        json.dumps(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "review_kind": review_kind,
                "passed": passed,
                "clips": [
                    _clip(index, motion)
                    for index, motion in enumerate(VAL_MOTIONS)
                ],
            }
        )
    )
    return path


def _write_stage_gate_metrics(tmp_path):
    stage1 = tmp_path / "stage1-passing.json"
    stage2 = tmp_path / "stage2-passing.json"
    stage1_record = {
        "val_early_termination_rate": 0.0,
        "val_frame_coverage": 1.0,
        "val_err_rpos": 0.01,
        "val_action_saturation_fraction": 0.0,
        "val_activation_energy": 0.0,
    }
    stage2_record = {
        "val_early_termination_rate": 0.0,
        "val_frame_coverage": 1.0,
        "val_err_racket_pos": 0.01,
        "val_err_racket_rot": 0.01,
        "val_err_rpos": 0.01,
    }
    stage1.write_text(json.dumps({"validations": [stage1_record] * 3}))
    stage2.write_text(json.dumps({"validations": [stage2_record] * 3}))
    return stage1, stage2


def test_pipeline_plan_records_all_gate_separated_stages(tmp_path):
    steps = build_pipeline_plan(tmp_path, PipelineArtifacts())
    assert [step.name for step in steps] == [
        "data_release_validate", "data_qc", "stage1_train", "stage1_gate", "stage1_visual_gate",
        "stage1_promote", "stage1r_train", "stage1r_eval", "stage1r_gate",
        "stage1r005_train", "stage1r005_eval", "stage1r005_gate",
        "stage2_train", "stage2_extend_160m", "stage2_gate", "stage2_visual_gate",
        "stage2_promote",
        "direct_distill", "direct_distill_resume", "latent_distill",
        "latent_closed_loop_gate", "latent_gate", "stage3_preflight",
        "stage3_base_only", "stage3_feed_gate", "stage3_train",
        "stage3_extend_curriculum", "stage3_evaluate", "stage3_gate",
    ]
    release = next(step for step in steps if step.name == "data_release_validate")
    assert "--validate" in release.command
    data_qc = next(step for step in steps if step.name == "data_qc")
    assert data_qc.command[data_qc.command.index("--source-variant") + 1] == "raw_smooth_v1"
    assert data_qc.command[data_qc.command.index("--cache-variant") + 1] == "raw_smooth_v1"
    assert "--require-clean" in data_qc.command
    direct = next(step for step in steps if step.name == "direct_distill")
    assert {"--collect_train", "--collect_val", "--run_ppo", "--compare"}.issubset(direct.command)
    assert direct.command[direct.command.index("--teacher_promotion_manifest") + 1] == str(
        tmp_path / "stage2_promotion_manifest.json"
    )
    assert len(direct.command[direct.command.index("--train_motion_path") + 1 : direct.command.index("--val_motion_path")]) == 22
    assert len(direct.command[direct.command.index("--val_motion_path") + 1 :]) == 5
    assert all(
        "/raw_smooth_v1/" in path
        for path in direct.command[
            direct.command.index("--train_motion_path") + 1 : direct.command.index("--val_motion_path")
        ]
    )
    resumed_direct = next(
        step for step in steps if step.name == "direct_distill_resume"
    )
    assert "--resume_dataset" in resumed_direct.command
    assert "--resume_dataset" not in direct.command
    latent_closed_loop = next(step for step in steps if step.name == "latent_closed_loop_gate")
    assert "--direct_rollout_metrics" in latent_closed_loop.command
    assert latent_closed_loop.command[-1] == "--require_pass"
    stage2_gate = next(step for step in steps if step.name == "stage2_gate")
    assert stage2_gate.command[stage2_gate.command.index("--baseline-metrics") + 1] == (
        "<required:stage1_metrics>"
    )
    stage2_train = next(step for step in steps if step.name == "stage2_train")
    assert (
        "experiment.promotion.baseline_metrics_path=<required:stage1_metrics>"
        in stage2_train.command
    )
    stage2_promote = next(step for step in steps if step.name == "stage2_promote")
    assert "--parent-promoted-artifact" in stage2_promote.command
    visual_gate = next(step for step in steps if step.name == "stage1_visual_gate")
    expected_start = visual_gate.command.index("--expected_motion") + 1
    expected_end = visual_gate.command.index("--output")
    assert visual_gate.command[expected_start:expected_end] == VAL_MOTIONS
    assert visual_gate.command[visual_gate.command.index("--review-kind") + 1] == (
        STAGE1_REVIEW_KIND
    )
    stage2_visual_gate = next(
        step for step in steps if step.name == "stage2_visual_gate"
    )
    assert stage2_visual_gate.command[
        stage2_visual_gate.command.index("--review-kind") + 1
    ] == STAGE2_REVIEW_KIND
    direct = next(step for step in steps if step.name == "direct_distill")
    latent = next(step for step in steps if step.name == "latent_distill")
    assert latent.command[latent.command.index("--teacher_promotion_manifest") + 1] == str(
        tmp_path / "stage2_promotion_manifest.json"
    )
    assert "stage2_visual_review" in direct.required_artifacts
    assert "stage2_visual_review" in latent.required_artifacts
    stage1r005 = next(step for step in steps if step.name == "stage1r005_train")
    assert stage1r005.environment == (
        ("STAGE1R_003_PROMOTED_CHECKPOINT", "<required:stage1r_checkpoint>"),
    )
    assert stage1r005.required_artifacts == ("stage1r_checkpoint", "stage1r_metrics")
    stage1r_gate = next(step for step in steps if step.name == "stage1r_gate")
    assert stage1r_gate.required_artifacts == (
        "stage1r_checkpoint",
        "stage1r_metrics",
    )
    assert stage1r_gate.command[
        stage1r_gate.command.index("--checkpoint") + 1
    ] == "<required:stage1r_checkpoint>"
    assert stage1r_gate.command[
        stage1r_gate.command.index("--finger-perturb-qpos-scale") + 1
    ] == "0.03"
    stage1r005_gate = next(
        step for step in steps if step.name == "stage1r005_gate"
    )
    assert stage1r005_gate.required_artifacts == (
        "stage1r005_checkpoint",
        "stage1r005_metrics",
    )
    assert stage1r005_gate.command[
        stage1r005_gate.command.index("--finger-perturb-qpos-scale") + 1
    ] == "0.05"
    preflight = next(step for step in steps if step.name == "stage3_preflight")
    feed_gate = next(step for step in steps if step.name == "stage3_feed_gate")
    assert preflight.command[preflight.command.index("--stage") + 1] == "preflight"
    base_only = next(step for step in steps if step.name == "stage3_base_only")
    assert base_only.command[base_only.command.index("--stage") + 1] == "base-only-check"
    assert base_only.command[base_only.command.index("--latent-checkpoint") + 1] == str(
        tmp_path / "latent_distill" / "latent_checkpoint"
    )
    assert feed_gate.command[feed_gate.command.index("--stage") + 1] == "feed-check"
    assert preflight.command[-1] == feed_gate.command[-1] == str(tmp_path / "stage3_lab")
    stage3_evaluate = next(step for step in steps if step.name == "stage3_evaluate")
    assert stage3_evaluate.command[stage3_evaluate.command.index("--episodes") + 1] == "128"
    assert stage3_evaluate.command[stage3_evaluate.command.index("--checkpoint") + 1] == str(
        tmp_path / "stage3_lab" / "policy_latest.json"
    )
    assert "--record-video" in stage3_evaluate.command
    stage3_train = next(step for step in steps if step.name == "stage3_train")
    assert "--auto-resume" in stage3_train.command
    stage3_gate = next(step for step in steps if step.name == "stage3_gate")
    assert stage3_gate.command[stage3_gate.command.index("--metrics") + 1] == str(
        tmp_path / "stage3_lab" / "evaluate" / "evaluate_report.json"
    )
    assert stage3_gate.required_artifacts == ()


def test_pipeline_refuses_legacy_stage1r_report_before_005(tmp_path):
    metrics = tmp_path / "stage1r-003.json"
    metrics.write_text(
        json.dumps(
            {
                "finger_qpos_perturb_scale": 0.03,
                "body_site_relative_degradation": 0.10,
                "right_hand_relative_degradation": 0.0,
                "racket_head_position_relative_degradation": 0.0,
                "racket_head_rotation_relative_degradation": 0.0,
                "early_termination_gap": 0.0,
                "paired_seed_verified": 1.0,
                "new_root_hand_racket_spike_count": 0,
            }
        )
    )
    with pytest.raises(ValueError, match="promotion artifact is invalid"):
        execute_pipeline_step(
            "stage1r005_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(
                stage1r_checkpoint="/tmp/accepted-stage1r-003",
                stage1r_metrics=str(metrics),
            ),
        )


def test_pipeline_refuses_unbound_stage1r_report(tmp_path):
    metrics = tmp_path / "stage1r-wrong-rung.json"
    metrics.write_text(json.dumps({"finger_qpos_perturb_scale": 0.05}))

    with pytest.raises(ValueError, match="promotion artifact is invalid"):
        execute_pipeline_step(
            "stage1r005_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(
                stage1r_checkpoint="/tmp/accepted-stage1r-003",
                stage1r_metrics=str(metrics),
            ),
        )


def test_pipeline_refuses_stage2_when_stage1_gate_did_not_pass(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"validations": [{"val_early_termination_rate": 1.0}] * 3}))
    review = _write_visual_review(tmp_path / "visual-review.json")
    artifacts = PipelineArtifacts(
        stage1_checkpoint="/tmp/ckpt",
        stage1_metrics=str(metrics),
        stage1_visual_review=str(review),
    )
    with pytest.raises(ValueError, match="promotion gate"):
        execute_pipeline_step("stage2_train", output_dir=tmp_path, artifacts=artifacts)


def test_pipeline_gate_consumes_online_promotion_progress_history(tmp_path):
    metrics = tmp_path / "promotion_progress.json"
    passing = {
        "val_early_termination_rate": 0.0,
        "val_frame_coverage": 1.0,
        "val_err_rpos": 0.01,
        "val_action_saturation_fraction": 0.0,
        "val_activation_energy": 0.0,
    }
    metrics.write_text(
        json.dumps(
            {
                "history": [
                    {"update_number": update, "metrics": passing}
                    for update in (10, 20, 30)
                ]
            }
        )
    )

    _require_metrics_gate("stage1", str(metrics), consecutive=3)


def test_pipeline_refuses_stage2_without_passed_human_visual_review(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "validations": [
                    {
                        "val_early_termination_rate": 0.0,
                        "val_frame_coverage": 1.0,
                        "val_err_rpos": 0.01,
                        "val_action_saturation_fraction": 0.0,
                        "val_activation_energy": 0.0,
                    }
                ]
                * 3
            }
        )
    )
    review = _write_visual_review(tmp_path / "visual-review.json", passed=False)
    artifacts = PipelineArtifacts(
        stage1_checkpoint="/tmp/ckpt",
        stage1_metrics=str(metrics),
        stage1_visual_review=str(review),
    )

    with pytest.raises(ValueError, match="visual review gate did not pass"):
        execute_pipeline_step("stage2_train", output_dir=tmp_path, artifacts=artifacts)


def test_pipeline_refuses_distill_when_stage2_visual_gate_was_skipped(tmp_path):
    stage1_metrics, stage2_metrics = _write_stage_gate_metrics(tmp_path)
    with pytest.raises(ValueError, match="stage2_visual_review"):
        execute_pipeline_step(
            "direct_distill",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(
                stage1_metrics=str(stage1_metrics),
                stage2_checkpoint="/tmp/stage2",
                stage2_metrics=str(stage2_metrics),
            ),
        )


def test_pipeline_refuses_wrong_kind_stage2_visual_review(tmp_path):
    stage1_metrics, stage2_metrics = _write_stage_gate_metrics(tmp_path)
    wrong_kind = _write_visual_review(
        tmp_path / "wrong-kind.json",
        review_kind=STAGE1_REVIEW_KIND,
    )
    with pytest.raises(ValueError, match="Stage 2 human visual review gate did not pass"):
        execute_pipeline_step(
            "direct_distill",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(
                stage1_metrics=str(stage1_metrics),
                stage2_checkpoint="/tmp/stage2",
                stage2_metrics=str(stage2_metrics),
                stage2_visual_review=str(wrong_kind),
            ),
        )


def test_pipeline_refuses_stage1_when_cache_environment_was_not_sourced(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MUSCLEMIMIC_GMR_CACHE_PATH", raising=False)
    (tmp_path / "data_qc.json").write_text(
        json.dumps(
            {
                "passed": True,
                "clean_passed": True,
                "dataset_root": str(DATASET_ROOT),
                "source_variant": "raw_smooth_v1",
                "cache_variant": "raw_smooth_v1",
                "resolved_source_dir": str(DATASET_ROOT / "temp" / "raw_smooth_v1"),
                "resolved_cache_dir": str(
                    DATASET_ROOT / "muscle_trajectory" / "raw_smooth_v1"
                ),
                "train_motions": list(TRAIN_MOTIONS),
                "validation_motions": list(VAL_MOTIONS),
            }
        )
    )
    with pytest.raises(ValueError, match="source configs/env.sh"):
        execute_pipeline_step(
            "stage1_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(),
        )


def test_pipeline_refuses_stage1_qc_with_any_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSCLEMIMIC_GMR_CACHE_PATH", str(DATASET_ROOT.parent))
    (tmp_path / "data_qc.json").write_text(
        json.dumps(
            {
                "passed": True,
                "clean_passed": False,
                "warnings": ["one posture-collapse warning"],
                "dataset_root": str(DATASET_ROOT),
                "source_variant": "raw_smooth_v1",
                "cache_variant": "raw_smooth_v1",
            }
        )
    )
    with pytest.raises(ValueError, match="warning-free"):
        execute_pipeline_step(
            "stage1_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(),
        )


def test_pipeline_refuses_stage3_without_promoted_latent_checkpoint(tmp_path):
    checkpoint = tmp_path / "latent"
    checkpoint.mkdir()
    (checkpoint / "eval_metrics.json").write_text(json.dumps({"promotion": {"passed": False}}))
    with pytest.raises(ValueError, match="has not passed"):
        execute_pipeline_step(
            "stage3_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(latent_checkpoint=str(checkpoint)),
        )


def test_pipeline_refuses_stage3_when_preflight_or_feed_report_was_skipped(tmp_path):
    checkpoint = tmp_path / "latent"
    checkpoint.mkdir()
    (checkpoint / "eval_metrics.json").write_text(
        json.dumps({"promotion": {"passed": True}})
    )
    artifacts = PipelineArtifacts(latent_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="preflight report is missing"):
        execute_pipeline_step(
            "stage3_train",
            output_dir=tmp_path,
            artifacts=artifacts,
        )

    stage3_dir = tmp_path / "stage3_lab"
    stage3_dir.mkdir()
    (stage3_dir / "preflight_report.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError, match="base-only check report is missing"):
        execute_pipeline_step(
            "stage3_train",
            output_dir=tmp_path,
            artifacts=artifacts,
        )

    (stage3_dir / "base_only_report.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError, match="feed check report is missing"):
        execute_pipeline_step(
            "stage3_train",
            output_dir=tmp_path,
            artifacts=artifacts,
        )

    (stage3_dir / "feed_check_report.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="feed check did not pass"):
        execute_pipeline_step(
            "stage3_train",
            output_dir=tmp_path,
            artifacts=artifacts,
        )


def test_pipeline_refuses_failed_stage3_base_only_report(tmp_path):
    checkpoint = tmp_path / "latent"
    checkpoint.mkdir()
    (checkpoint / "eval_metrics.json").write_text(
        json.dumps({"promotion": {"passed": True}})
    )
    stage3_dir = tmp_path / "stage3_lab"
    stage3_dir.mkdir()
    (stage3_dir / "preflight_report.json").write_text(json.dumps({"passed": True}))
    (stage3_dir / "base_only_report.json").write_text(json.dumps({"passed": False}))

    with pytest.raises(ValueError, match="base-only check did not pass"):
        execute_pipeline_step(
            "stage3_feed_gate",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(latent_checkpoint=str(checkpoint)),
        )


def test_pipeline_refuses_stage3_evaluate_without_trained_checkpoint(tmp_path):
    checkpoint = tmp_path / "latent"
    checkpoint.mkdir()
    (checkpoint / "eval_metrics.json").write_text(
        json.dumps({"promotion": {"passed": True}})
    )
    stage3_dir = tmp_path / "stage3_lab"
    stage3_dir.mkdir()
    (stage3_dir / "preflight_report.json").write_text(json.dumps({"passed": True}))
    (stage3_dir / "base_only_report.json").write_text(json.dumps({"passed": True}))
    (stage3_dir / "feed_check_report.json").write_text(json.dumps({"passed": True}))

    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        execute_pipeline_step(
            "stage3_evaluate",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(latent_checkpoint=str(checkpoint)),
        )


def test_pipeline_refuses_latent_training_without_direct_acceptance(tmp_path):
    stage1_metrics = tmp_path / "stage1.json"
    stage1_metrics.write_text(
        json.dumps(
            {
                "validations": [
                    {
                        "val_early_termination_rate": 0.0,
                        "val_frame_coverage": 1.0,
                        "val_err_rpos": 0.01,
                    }
                ]
                * 3
            }
        )
    )
    stage2_metrics = tmp_path / "stage2.json"
    stage2_metrics.write_text(
        json.dumps(
            {
                "validations": [
                    {
                        "val_early_termination_rate": 0.0,
                        "val_frame_coverage": 1.0,
                        "val_err_racket_pos": 0.01,
                        "val_err_racket_rot": 0.01,
                        "val_err_rpos": 0.01,
                    }
                ]
                * 3
            }
        )
    )
    stage2_review = _write_visual_review(
        tmp_path / "stage2-review.json",
        review_kind=STAGE2_REVIEW_KIND,
    )
    with pytest.raises(ValueError, match="direct distillation artifacts are incomplete"):
        execute_pipeline_step(
            "latent_distill",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(
                stage2_checkpoint="/tmp/stage2",
                stage2_metrics=str(stage2_metrics),
                stage1_metrics=str(stage1_metrics),
                stage2_visual_review=str(stage2_review),
            ),
        )
