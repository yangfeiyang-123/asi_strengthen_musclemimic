from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace

import pytest

from fullbody.run_forehand_clear_pipeline import (
    DATASET_ROOT,
    REPO_ROOT,
    PipelineArtifacts,
    PipelineStep,
    _canonical_training_launch_command,
    _require_checkpoint_descends_from_stage1_peasd_promotion,
    _require_metrics_gate,
    build_pipeline_plan,
    execute_pipeline_step,
    main,
)
from musclemimic.badminton.action_registry import CHINA_JUMP, FOREHAND_LIFT
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


def test_non_peasd_plan_abi_omits_new_null_evidence_fields(monkeypatch, tmp_path) -> None:
    output = tmp_path / "legacy"
    monkeypatch.setattr(sys, "argv", ["pipeline", "--output_dir", str(output)])
    assert main() == 0
    payload = json.loads((output / "pipeline_plan.json").read_text(encoding="utf-8"))
    assert not any(key.startswith("stage1_peasd_") for key in payload["artifacts"])

    peasd_output = tmp_path / "peasd"
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline", "--output_dir", str(peasd_output), "--profile", "stage1_peasd"],
    )
    assert main() == 0
    peasd = json.loads(
        (peasd_output / "pipeline_plan.json").read_text(encoding="utf-8")
    )
    assert "stage1_peasd_t4_s2_validation_evidence" in peasd["artifacts"]


@pytest.mark.parametrize(
    ("spec", "expected_source", "expected_val_count"),
    (
        (FOREHAND_LIFT, "temp/optimized_root_smooth_v2", "4"),
        (CHINA_JUMP, "wham/optimized_wham", "2"),
    ),
)
def test_stage1_aligned_plan_binds_action_specific_qc_and_visual_count(
    tmp_path,
    spec,
    expected_source,
    expected_val_count,
):
    steps = build_pipeline_plan(
        tmp_path,
        PipelineArtifacts(),
        profile="stage1_aligned",
        spec=spec,
    )
    assert [step.name for step in steps] == [
        "data_release_validate",
        "data_qc",
        "stage1_train",
        "stage1_gate",
        "stage1_visual_gate",
        "stage1_promote",
    ]
    data_qc = next(step for step in steps if step.name == "data_qc")
    assert data_qc.command[data_qc.command.index("--action") + 1] == spec.slug
    assert data_qc.command[data_qc.command.index("--source-variant") + 1] == expected_source
    visual = next(step for step in steps if step.name == "stage1_visual_gate")
    assert visual.command.count("--required_clips") == 1
    assert visual.command[visual.command.index("--required_clips") + 1] == expected_val_count


def test_synergy_plan_names_real_nonclear_readiness_blockers(tmp_path):
    with pytest.raises(ValueError, match="racket_event_bank_config"):
        build_pipeline_plan(
            tmp_path,
            PipelineArtifacts(),
            profile="synergy_v3",
            spec=FOREHAND_LIFT,
        )
    steps = build_pipeline_plan(
        tmp_path,
        PipelineArtifacts(),
        profile="synergy_v3",
        spec=CHINA_JUMP,
    )
    names = [step.name for step in steps]
    assert names == [
        "data_release_validate",
        "data_qc",
        "stage1_train",
        "stage1_gate",
        "stage1_visual_gate",
        "stage1_promote",
        "physical_rollout_collect",
        "physical_rollout_collect_val",
        "physical_rollout_qc",
        "physical_rollout_gate",
        "synergy_fit",
        "synergy_gate",
        "latent_dimension_sweep",
        "latent_dimension_execute",
        "latent_synergy_analysis",
        "latent_synergy_gate",
    ]
    assert not any(
        name.startswith(("stage1r", "racket_mass", "stage2", "stage3"))
        for name in names
    )
    collect = next(step for step in steps if step.name == "physical_rollout_collect")
    assert collect.command[collect.command.index("--teacher-promotion-stage") + 1] == "stage1"
    assert collect.command[collect.command.index("--teacher-promotion-role") + 1] == "body_only"
    assert "--save-event-features" not in collect.command
    assert "--event-reference-bank" not in collect.command
    assert not any("racket" in token.lower() for token in collect.command)
    physical_qc = next(step for step in steps if step.name == "physical_rollout_qc")
    assert physical_qc.command[physical_qc.command.index("--qc-contract") + 1] == (
        "body-only-phase-free"
    )
    physical_gate = next(step for step in steps if step.name == "physical_rollout_gate")
    assert physical_gate.command[physical_gate.command.index("--stage") + 1] == (
        "physical_rollout_body_only_v1"
    )
    sweep = next(step for step in steps if step.name == "latent_dimension_sweep")
    assert sweep.command[sweep.command.index("--expected-validation-motion-count") + 1] == "2"
    assert sweep.command[sweep.command.index("--base-config") + 1] == (
        CHINA_JUMP.latent_synergy_config
    )
    assert "--direct-bc-metrics" not in sweep.command
    assert "--direct-rollout-metrics" not in sweep.command
    assert "--require-causal-interventions" not in sweep.command
    latent_gate = next(step for step in steps if step.name == "latent_synergy_gate")
    assert latent_gate.command[latent_gate.command.index("--stage") + 1] == (
        "latent_synergy_body_only_v1"
    )


def test_lift_does_not_borrow_clear_stage3_after_event_mass_are_ready(tmp_path):
    event_mass_ready = replace(
        FOREHAND_LIFT,
        racket_event_bank_config="config_specific_task/stage2_racket_v2/lift_event",
        racket_mass_v2_configs=tuple(
            f"config_specific_task/stage2_racket_v2/lift_mass_{scale}"
            for scale in ("025", "050", "075", "100")
        ),
    )

    with pytest.raises(ValueError, match="stage3_v2_spec"):
        build_pipeline_plan(
            tmp_path,
            PipelineArtifacts(),
            profile="synergy_v3",
            spec=event_mass_ready,
        )


def test_chinajump_four_arms_share_collection_and_only_change_context(
    tmp_path, monkeypatch
):
    promotion = tmp_path / "promotion.json"
    tube = tmp_path / "emg_reference_manifest.json"
    shared_path = tmp_path / "stage2_shared_inputs.json"
    lock = tmp_path / "stage2_s2b_architecture_lock.json"
    for path in (promotion, tube, shared_path, lock):
        path.write_text("{}\n", encoding="utf-8")
    common = {
        "stage1_checkpoint": "/sealed/chinajump/t3/checkpoint_100",
        "stage1_peasd_promotion_manifest": str(promotion),
        "emg_reference_manifest": str(tube),
    }
    baseline = PipelineArtifacts(**common, stage1_peasd_latent_arm="disabled")
    baseline_steps = build_pipeline_plan(
        tmp_path / "s2b", baseline, profile="synergy_v3", spec=CHINA_JUMP
    )
    baseline_names = [step.name for step in baseline_steps]
    assert "physical_rollout_collect" in baseline_names
    assert "stage2_shared_inputs_seal" in baseline_names
    assert "stage2_s2b_architecture_lock" in baseline_names
    baseline_sweep = next(
        step for step in baseline_steps if step.name == "latent_dimension_sweep"
    ).command
    assert baseline_sweep[baseline_sweep.index("--stage2-arm") + 1] == "S2-B"
    assert not any("emg-privileged" in token for token in baseline_sweep)

    shared = {
        "binding_sha256": "1" * 64,
        "stage1_peasd": {
            "promotion": {"path": str(promotion.resolve())},
            "emg_reference": {"path": str(tube.resolve())},
        },
        "teacher": {
            "checkpoint": {"resolved_path": "/sealed/chinajump/t3/checkpoint_100"},
            "promotion": {"path": str(promotion.resolve())},
        },
        "datasets": {
            "train": {"path": "/sealed/shared/train"},
            "validation": {"path": "/sealed/shared/val"},
        },
        "direct_s2a_evidence": {"required": False},
        "synergy": {
            "basis": {"path": "/sealed/shared/basis", "artifact_fingerprint": "2" * 64},
            "frozen_body_decoder": {
                "path": "/sealed/shared/decoder",
                "artifact_fingerprint": "3" * 64,
                "body_synergy_contract_fingerprint": "4" * 64,
                "portable_decoder_core_fingerprint": "5" * 64,
            },
        },
    }
    monkeypatch.setattr(
        "musclemimic.badminton.stage2_context_family.validate_stage2_shared_inputs",
        lambda _path, expected_action=None: shared,
    )
    arms = {
        "peasd": PipelineArtifacts(
            **common,
            stage2_shared_inputs_manifest=str(shared_path),
            stage2_architecture_lock_manifest=str(lock),
            stage1_peasd_latent_arm="real",
            emg_synergy_dim=3,
        ),
        "shuffled": PipelineArtifacts(
            **common,
            stage2_shared_inputs_manifest=str(shared_path),
            stage2_architecture_lock_manifest=str(lock),
            stage1_peasd_latent_arm="shuffled",
            emg_synergy_dim=3,
            emg_shuffle_context_ablation=True,
        ),
        "nodropout": PipelineArtifacts(
            **common,
            stage2_shared_inputs_manifest=str(shared_path),
            stage2_architecture_lock_manifest=str(lock),
            stage1_peasd_latent_arm="real_no_dropout",
            emg_synergy_dim=3,
            emg_no_context_dropout=True,
        ),
    }
    commands = {}
    for name, artifacts in arms.items():
        steps = build_pipeline_plan(
            tmp_path / name,
            artifacts,
            profile="synergy_v3",
            spec=CHINA_JUMP,
        )
        assert "physical_rollout_collect" not in {step.name for step in steps}
        sweep = next(
            step for step in steps if step.name == "latent_dimension_sweep"
        )
        commands[name] = sweep.command

    for name in ("peasd", "shuffled", "nodropout"):
        sweep = commands[name]
        assert "--emg-privileged-enabled" in sweep
        assert sweep[sweep.index("--emg-synergy-dim") + 1] == "3"
        assert sweep[sweep.index("--emg-synergy-loss-weight") + 1] == "0.05"
        assert sweep[
            sweep.index("--stage1-peasd-promotion-manifest") + 1
        ] == common["stage1_peasd_promotion_manifest"]
    assert "--emg-shuffle-context-ablation" in commands["shuffled"]
    assert "--emg-context-dropout" in commands["nodropout"]
    assert commands["nodropout"][
        commands["nodropout"].index("--emg-context-dropout") + 1
    ] == "0.0"


def test_context_treatment_arm_cannot_recollect_without_shared_inputs(tmp_path):
    with pytest.raises(ValueError, match="requires stage2_shared_inputs_manifest"):
        build_pipeline_plan(
            tmp_path,
            PipelineArtifacts(
                stage1_checkpoint="/sealed/t3/checkpoint_100",
                stage1_peasd_promotion_manifest="/sealed/t3/promotion.json",
                emg_reference_manifest="/sealed/emg/reference.json",
                stage1_peasd_latent_arm="real",
                emg_synergy_dim=3,
            ),
            profile="synergy_v3",
            spec=CHINA_JUMP,
        )


def test_stage2_direct_profile_expands_canonical_component_steps(
    tmp_path, monkeypatch
):
    from musclemimic.distill.stage2_direct_lifecycle import Stage2DirectStep

    shared_path = tmp_path / "stage2_shared_inputs.json"
    shared_path.write_text("{}\n", encoding="utf-8")
    shared = {
        "teacher": {
            "checkpoint": {"resolved_path": "/sealed/teacher/checkpoint_100"},
            "promotion": {"path": "/sealed/teacher/promotion.json"},
        },
        "datasets": {
            "train": {"path": "/sealed/shared/train"},
            "validation": {"path": "/sealed/shared/val"},
        },
    }
    monkeypatch.setattr(
        "musclemimic.badminton.stage2_context_family.validate_stage2_shared_inputs",
        lambda _path, expected_action=None: shared,
    )
    lifecycle_step = Stage2DirectStep(
        name="s2a_seed0_bc",
        seed=0,
        command=("scripts/run_fullbody_training.sh", "--distill-bc", "--seed", "0"),
        environment={
            "CUDA_VISIBLE_DEVICES": "0",
            "MUSCLEMIMIC_JAX_CACHE_KEY": "clear_s2a_s0_bc",
            "MUSCLEMIMIC_TRAIN_LOG": "/tmp/clear_s2a_s0_bc.log",
        },
        output_artifact="/sealed/s2a/seed0/bc",
    )
    monkeypatch.setattr(
        "musclemimic.distill.stage2_direct_lifecycle.build_stage2_direct_family_plan",
        lambda _config: ({"schema_version": "stage2_direct_family_plan_v1"}, (lifecycle_step,)),
    )

    steps = build_pipeline_plan(
        tmp_path,
        PipelineArtifacts(
            stage2_shared_inputs_manifest=str(shared_path),
            stage2_direct_physical_gpu=0,
            stage2_direct_cache_key_prefix="clear_s2a",
        ),
        profile="stage2_direct",
    )

    assert [step.name for step in steps] == ["stage2_direct_plan", "s2a_seed0_bc"]
    assert steps[1].command[:2] == (
        "scripts/run_fullbody_training.sh",
        "--distill-bc",
    )
    assert dict(steps[1].environment)["CUDA_VISIBLE_DEVICES"] == "0"


def test_peasd_stage1r_parent_lineage_binds_promotion_manifest(tmp_path) -> None:
    artifacts = PipelineArtifacts(
        stage1_checkpoint="/sealed/t3/checkpoint_100",
        stage1_peasd_promotion_manifest="/sealed/t3/teacher_promotion.json",
        emg_reference_manifest="/sealed/emg/emg_reference_manifest.json",
        stage1_peasd_latent_arm="disabled",
    )
    steps = build_pipeline_plan(tmp_path, artifacts, profile="synergy_v3")
    names = {step.name for step in steps}
    assert "stage1_train" not in names
    stage1r = next(step for step in steps if step.name == "stage1r_train")
    assert (
        "+experiment.parent_checkpoint_lineage.promotion_manifest="
        "/sealed/t3/teacher_promotion.json"
    ) in stage1r.command
    assert "stage1_peasd_promotion_manifest" in stage1r.required_artifacts


def test_racket_teacher_must_recursively_descend_from_selected_peasd_promotion(
    tmp_path,
) -> None:
    promotion = tmp_path / "teacher_promotion.json"
    promotion.write_text(
        json.dumps({"binding_sha256": "a" * 64}) + "\n", encoding="utf-8"
    )
    content = hashlib.sha256(promotion.read_bytes()).hexdigest()
    checkpoint = {
        "parent_checkpoint_lineage": {
            "promotion": {
                "evidence_kind": "verified_stage1_peasd_promotion_v1",
                "artifact_content_sha256": content,
                "artifact_binding_sha256": "a" * 64,
            },
            "parent_checkpoint_lineage": None,
        }
    }
    _require_checkpoint_descends_from_stage1_peasd_promotion(
        checkpoint, str(promotion)
    )
    checkpoint["parent_checkpoint_lineage"]["promotion"][
        "artifact_content_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="ancestry does not contain"):
        _require_checkpoint_descends_from_stage1_peasd_promotion(
            checkpoint, str(promotion)
        )


@pytest.mark.parametrize(
    ("step", "expected_prefix"),
    (
        (
            PipelineStep("stage1_train", ("python", "fullbody/experiment.py", "--config-name=x")),
            (str(REPO_ROOT / "scripts/run_fullbody_training.sh"), "--config-name=x"),
        ),
        (
            PipelineStep("latent", ("python", "-m", "fullbody.latent_train", "--config", "x.yaml")),
            (str(REPO_ROOT / "scripts/run_fullbody_training.sh"), "--latent"),
        ),
        (
            PipelineStep("bc", ("python", "-m", "fullbody.distill_train_bc", "--dataset_dir", "d")),
            (str(REPO_ROOT / "scripts/run_fullbody_training.sh"), "--distill-bc"),
        ),
        (
            PipelineStep(
                "stage1_peasd_t0_s0_posthoc_physiology",
                (
                    "python",
                    "scripts/evaluate_stage1_peasd.py",
                    "--checkpoint",
                    "/tmp/checkpoint_10",
                ),
            ),
            (str(REPO_ROOT / "scripts/run_fullbody_training.sh"), "--stage1-peasd-eval"),
        ),
    ),
)
def test_production_training_steps_route_through_canonical_launcher(step, expected_prefix):
    assert tuple(_canonical_training_launch_command(step)[:2]) == expected_prefix


def test_monolithic_legacy_distill_cannot_bypass_canonical_launcher():
    step = PipelineStep(
        "direct_distill",
        ("python", "-m", "fullbody.run_distill_experiment", "--train_bc"),
    )
    with pytest.raises(ValueError, match="nested trainers bypass"):
        _canonical_training_launch_command(step)


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


def test_pipeline_uses_launcher_environment_default_when_shell_was_not_sourced(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MUSCLEMIMIC_GMR_CACHE_PATH", raising=False)
    monkeypatch.delenv("MUSCLEMIMIC_DATASETS_ROOT", raising=False)
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "datasets" / "forehandClear_standard"
    cache_dir = dataset_root / "muscle_trajectory" / "raw_smooth_v1"
    cache_dir.mkdir(parents=True)
    for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS):
        (cache_dir / f"{motion}.npz").write_bytes(b"source-only fixture")
    release_manifest = (
        dataset_root / "manifests" / "raw_smooth_v1" / "release_manifest.json"
    )
    release_manifest.parent.mkdir(parents=True)
    release_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("fullbody.run_forehand_clear_pipeline.REPO_ROOT", repo_root)
    monkeypatch.setattr("musclemimic.badminton.action_registry.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "fullbody.run_forehand_clear_pipeline.validate_action_release",
        lambda _spec: {
            "passed": True,
            "release_binding_sha256": "a" * 64,
            "formal_release_manifest": True,
            "review_evidence_kind": "source_only_fixture",
            "evidence_limitations": [],
            "visual_qc_path": None,
            "visual_qc_sha256": None,
        },
    )
    (tmp_path / "data_qc.json").write_text(
        json.dumps(
            {
                "passed": True,
                "clean_passed": True,
                "dataset_root": str(dataset_root),
                "source_variant": "raw_smooth_v1",
                "cache_variant": "raw_smooth_v1",
                "resolved_source_dir": str(dataset_root / "temp" / "raw_smooth_v1"),
                "resolved_cache_dir": str(cache_dir),
                "train_motions": list(TRAIN_MOTIONS),
                "validation_motions": list(VAL_MOTIONS),
            }
        )
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        "fullbody.run_forehand_clear_pipeline.subprocess.run",
        lambda command, **_kwargs: launched.append(command),
    )
    execute_pipeline_step(
        "stage1_train",
        output_dir=tmp_path,
        artifacts=PipelineArtifacts(),
    )
    assert launched and launched[0][0].endswith("scripts/run_fullbody_training.sh")


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
