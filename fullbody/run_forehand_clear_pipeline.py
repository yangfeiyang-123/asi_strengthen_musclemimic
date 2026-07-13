"""Gate-enforced launcher for the canonical Stage1 -> Stage2 -> distill -> LAB pipeline.

Long GPU jobs produce checkpoints and validation artifacts asynchronously, so
this launcher executes one stage at a time and refuses unsafe stage changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musclemimic.badminton.data_qc import TRAIN_MOTIONS, VAL_MOTIONS
from musclemimic.badminton.promotion_artifact import validate_promoted_artifact
from musclemimic.badminton.scripts.data_release import validate_release_manifest
from musclemimic.badminton.scripts.finalize_raw_smooth_visual_qc import (
    validate_report as validate_visual_qc_report,
)
from musclemimic.badminton.stage1r_artifact import validate_stage1r_report
from musclemimic.badminton.training_gates import (
    evaluate_promotion,
    extract_validation_records,
    latest_validation_record,
)
from musclemimic.badminton.visual_review import (
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
    validate_visual_review,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_VARIANT = "raw_smooth_v1"
DATASET_ROOT = REPO_ROOT / "datasets" / "forehandClear_standard"
RELEASE_MANIFEST = (
    DATASET_ROOT / "manifests" / DATA_VARIANT / "release_manifest.json"
)


@dataclass(frozen=True)
class PipelineArtifacts:
    stage1_checkpoint: str | None = None
    stage1_metrics: str | None = None
    stage1_visual_review: str | None = None
    stage1_promotion_manifest: str | None = None
    stage1r_checkpoint: str | None = None
    stage1r_metrics: str | None = None
    stage1r005_metrics: str | None = None
    stage1r005_checkpoint: str | None = None
    stage2_checkpoint: str | None = None
    stage2_metrics: str | None = None
    stage2_visual_review: str | None = None
    stage2_promotion_manifest: str | None = None
    direct_bc_metrics: str | None = None
    direct_rollout_metrics: str | None = None
    direct_acceptance: str | None = None
    latent_checkpoint: str | None = None
    stage3_checkpoint: str | None = None
    stage3_metrics: str | None = None


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    required_artifacts: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


def build_pipeline_plan(output_dir: str | Path, artifacts: PipelineArtifacts) -> tuple[PipelineStep, ...]:
    out = Path(output_dir)
    python = sys.executable
    stage1_ckpt = artifacts.stage1_checkpoint or "<required:stage1_checkpoint>"
    stage1r_ckpt = artifacts.stage1r_checkpoint or "<required:stage1r_checkpoint>"
    stage2_ckpt = artifacts.stage2_checkpoint or "<required:stage2_checkpoint>"
    stage1_promotion = artifacts.stage1_promotion_manifest or str(
        out / "stage1_promotion_manifest.json"
    )
    stage2_promotion = artifacts.stage2_promotion_manifest or str(
        out / "stage2_promotion_manifest.json"
    )
    train_motions = tuple(
        f"forehandClear_standard/muscle_trajectory/{DATA_VARIANT}/{name}"
        for name in TRAIN_MOTIONS
    )
    val_motions = tuple(
        f"forehandClear_standard/muscle_trajectory/{DATA_VARIANT}/{name}"
        for name in VAL_MOTIONS
    )
    direct_dir = out / "direct_distill"
    latent_dir = out / "latent_distill"
    stage3_dir = out / "stage3_lab"
    direct_bc_metrics = artifacts.direct_bc_metrics or str(direct_dir / "bc" / "distill_metadata.json")
    direct_rollout_metrics = artifacts.direct_rollout_metrics or str(
        direct_dir / "compare" / "comparison_metrics.json"
    )
    latent_checkpoint = artifacts.latent_checkpoint or str(latent_dir / "latent_checkpoint")
    stage3_checkpoint = artifacts.stage3_checkpoint or str(stage3_dir / "policy_latest.json")
    stage3_metrics = artifacts.stage3_metrics or str(
        stage3_dir / "evaluate" / "evaluate_report.json"
    )
    direct_distill_command = (
        python,
        "-m",
        "fullbody.run_distill_experiment",
        "--teacher_ckpt",
        stage2_ckpt,
        "--teacher_promotion_manifest",
        stage2_promotion,
        "--student_config",
        "fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc.yaml",
        "--student_ppo_config",
        "config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_ppo",
        "--out_dir",
        str(direct_dir),
        "--collect_train",
        "--collect_val",
        "--train_bc",
        "--run_dagger",
        "3",
        "--run_ppo",
        "--compare",
        "--train_motion_path",
        *train_motions,
        "--val_motion_path",
        *val_motions,
    )
    return (
        PipelineStep(
            "data_release_validate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.data_release",
                "--dataset-root",
                str(DATASET_ROOT),
                "--output",
                str(RELEASE_MANIFEST),
                "--validate",
            ),
        ),
        PipelineStep(
            "data_qc",
            (
                python,
                "-m",
                "musclemimic.badminton.data_qc",
                "--dataset-root",
                str(DATASET_ROOT),
                "--source-variant",
                DATA_VARIANT,
                "--cache-variant",
                DATA_VARIANT,
                "--require-clean",
                "--output",
                str(out / "data_qc.json"),
            ),
        ),
        PipelineStep(
            "stage1_train",
            (python, "fullbody/experiment.py", "--config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local"),
        ),
        PipelineStep(
            "stage1_gate",
            _gate_command(python, "stage1", artifacts.stage1_metrics, out / "stage1_gate.json", consecutive=3),
            ("stage1_metrics",),
        ),
        PipelineStep(
            "stage1_visual_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.visual_review",
                "--review",
                artifacts.stage1_visual_review or "<required:stage1_visual_review>",
                "--required_clips",
                "5",
                "--review-kind",
                STAGE1_REVIEW_KIND,
                "--checkpoint",
                stage1_ckpt,
                "--expected_motion",
                *VAL_MOTIONS,
                "--output",
                str(out / "stage1_visual_gate.json"),
                "--require_pass",
            ),
            ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review"),
        ),
        PipelineStep(
            "stage1_promote",
            (
                python,
                "-m",
                "musclemimic.badminton.promotion_artifact",
                "--stage",
                "stage1",
                "--checkpoint",
                stage1_ckpt,
                "--promotion-progress",
                artifacts.stage1_metrics or "<required:stage1_metrics>",
                "--visual-review",
                artifacts.stage1_visual_review or "<required:stage1_visual_review>",
                "--output",
                stage1_promotion,
                "--require-pass",
            ),
            ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review"),
        ),
        PipelineStep(
            "stage1r_train",
            (python, "fullbody/experiment.py", "--config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_finger_isolated"),
            ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review"),
            (("STAGE1_PROMOTED_CHECKPOINT", stage1_ckpt),),
        ),
        PipelineStep(
            "stage1r_eval",
            (
                python,
                "-m",
                "fullbody.eval_finger_robustness",
                "--checkpoint",
                stage1r_ckpt,
                "--motion_path",
                *val_motions,
                "--perturb_qpos_scale",
                "0.03",
                "--perturb_qvel_scale",
                "0.0",
                "--output",
                artifacts.stage1r_metrics
                or str(out / "stage1r_003" / "paired_robustness.json"),
                "--require_pass",
            ),
            ("stage1r_checkpoint",),
        ),
        PipelineStep(
            "stage1r_gate",
            _gate_command(
                python,
                "stage1r",
                artifacts.stage1r_metrics,
                out / "stage1r_gate.json",
                checkpoint=stage1r_ckpt,
                finger_perturb_qpos_scale=0.03,
            ),
            ("stage1r_checkpoint", "stage1r_metrics"),
        ),
        PipelineStep(
            "stage1r005_train",
            (
                python,
                "fullbody/experiment.py",
                "--config-name=config_specific_task/stage1_body/"
                "conf_fullbody_forehand_clear_body_finger_isolated_005",
            ),
            ("stage1r_checkpoint", "stage1r_metrics"),
            (("STAGE1R_003_PROMOTED_CHECKPOINT", stage1r_ckpt),),
        ),
        PipelineStep(
            "stage1r005_eval",
            (
                python,
                "-m",
                "fullbody.eval_finger_robustness",
                "--checkpoint",
                artifacts.stage1r005_checkpoint
                or "<required:stage1r005_checkpoint>",
                "--motion_path",
                *val_motions,
                "--perturb_qpos_scale",
                "0.05",
                "--perturb_qvel_scale",
                "0.0",
                "--output",
                artifacts.stage1r005_metrics
                or str(out / "stage1r_005" / "paired_robustness.json"),
                "--require_pass",
            ),
            ("stage1r005_checkpoint",),
        ),
        PipelineStep(
            "stage1r005_gate",
            _gate_command(
                python,
                "stage1r",
                artifacts.stage1r005_metrics,
                out / "stage1r005_gate.json",
                checkpoint=(
                    artifacts.stage1r005_checkpoint
                    or "<required:stage1r005_checkpoint>"
                ),
                finger_perturb_qpos_scale=0.05,
            ),
            ("stage1r005_checkpoint", "stage1r005_metrics"),
        ),
        PipelineStep(
            "stage2_train",
            (
                python,
                "fullbody/experiment.py",
                "--config-name=config_specific_task/stage2_racket/conf_fullbody_badminton_racket_local",
                f"experiment.resume_from={stage1_ckpt}",
                "experiment.promotion.baseline_metrics_path="
                f"{artifacts.stage1_metrics or '<required:stage1_metrics>'}",
            ),
            ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review"),
        ),
        PipelineStep(
            "stage2_extend_160m",
            (
                python,
                "fullbody/experiment.py",
                "--config-name=config_specific_task/stage2_racket/"
                "conf_fullbody_badminton_racket_local_extend_160m",
                f"experiment.resume_from={stage2_ckpt}",
                "experiment.promotion.baseline_metrics_path="
                f"{artifacts.stage1_metrics or '<required:stage1_metrics>'}",
            ),
            (
                "stage1_checkpoint",
                "stage1_metrics",
                "stage2_checkpoint",
                "stage2_metrics",
            ),
        ),
        PipelineStep(
            "stage2_gate",
            _gate_command(
                python,
                "stage2",
                artifacts.stage2_metrics,
                out / "stage2_gate.json",
                consecutive=3,
                baseline_metrics=artifacts.stage1_metrics or "<required:stage1_metrics>",
            ),
            ("stage1_metrics", "stage2_metrics"),
        ),
        PipelineStep(
            "stage2_visual_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.visual_review",
                "--review",
                artifacts.stage2_visual_review or "<required:stage2_visual_review>",
                "--required_clips",
                "5",
                "--review-kind",
                STAGE2_REVIEW_KIND,
                "--checkpoint",
                stage2_ckpt,
                "--expected_motion",
                *VAL_MOTIONS,
                "--output",
                str(out / "stage2_visual_gate.json"),
                "--require_pass",
            ),
            (
                "stage1_metrics",
                "stage2_checkpoint",
                "stage2_metrics",
                "stage2_visual_review",
            ),
        ),
        PipelineStep(
            "stage2_promote",
            (
                python,
                "-m",
                "musclemimic.badminton.promotion_artifact",
                "--stage",
                "stage2",
                "--checkpoint",
                stage2_ckpt,
                "--promotion-progress",
                artifacts.stage2_metrics or "<required:stage2_metrics>",
                "--visual-review",
                artifacts.stage2_visual_review or "<required:stage2_visual_review>",
                "--parent-promoted-artifact",
                stage1_promotion,
                "--output",
                stage2_promotion,
                "--require-pass",
            ),
            ("stage2_checkpoint", "stage2_metrics", "stage2_visual_review"),
        ),
        PipelineStep(
            "direct_distill",
            direct_distill_command,
            (
                "stage1_metrics",
                "stage2_checkpoint",
                "stage2_metrics",
                "stage2_visual_review",
            ),
        ),
        PipelineStep(
            "direct_distill_resume",
            (*direct_distill_command, "--resume_dataset"),
            (
                "stage1_metrics",
                "stage2_checkpoint",
                "stage2_metrics",
                "stage2_visual_review",
            ),
        ),
        PipelineStep(
            "latent_distill",
            (
                python,
                "fullbody/latent_train.py",
                "--config",
                "fullbody/config_specific_task/distill/latent_forehandclear_lab.yaml",
                "--dataset_dir",
                str(direct_dir / "dataset"),
                "--output_dir",
                str(latent_dir),
                "--teacher_ckpt",
                stage2_ckpt,
                "--teacher_promotion_manifest",
                stage2_promotion,
                "--direct_bc_metrics",
                direct_bc_metrics,
            ),
            (
                "stage1_metrics",
                "stage2_checkpoint",
                "stage2_metrics",
                "stage2_visual_review",
            ),
        ),
        PipelineStep(
            "latent_closed_loop_gate",
            (
                python,
                "-m",
                "fullbody.latent_closed_loop_eval",
                "--latent_checkpoint",
                latent_checkpoint,
                "--teacher_ckpt",
                stage2_ckpt,
                "--direct_rollout_metrics",
                direct_rollout_metrics,
                "--promotion_policy",
                "student_bc_ppo",
                "--motion_path",
                *val_motions,
                "--lambdas",
                "0.0",
                "0.25",
                "0.5",
                "--max_steps",
                "120",
                "--require_pass",
            ),
            ("stage2_checkpoint",),
        ),
        PipelineStep(
            "latent_gate",
            (
                python,
                "-m",
                "fullbody.latent_eval",
                "--checkpoint_dir",
                latent_checkpoint,
                "--require_pass",
            ),
        ),
        PipelineStep(
            "stage3_preflight",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--stage",
                "preflight",
                "--out-dir",
                str(stage3_dir),
            ),
        ),
        PipelineStep(
            "stage3_base_only",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--stage",
                "base-only-check",
                "--latent-checkpoint",
                latent_checkpoint,
                "--out-dir",
                str(stage3_dir),
            ),
        ),
        PipelineStep(
            "stage3_feed_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--stage",
                "feed-check",
                "--out-dir",
                str(stage3_dir),
            ),
        ),
        PipelineStep(
            "stage3_train",
            (
                python,
                "-m",
                "environment.overall_environment.src.train_incoming_hit_mjx",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--latent-checkpoint",
                latent_checkpoint,
                "--total-env-steps",
                "20000000",
                "--auto-resume",
                "--out-dir",
                str(stage3_dir),
            ),
        ),
        PipelineStep(
            "stage3_extend_curriculum",
            (
                python,
                "-m",
                "environment.overall_environment.src.train_incoming_hit_mjx",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--latent-checkpoint",
                latent_checkpoint,
                "--total-env-steps",
                "40000000",
                "--auto-resume",
                "--out-dir",
                str(stage3_dir),
            ),
        ),
        PipelineStep(
            "stage3_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
                "--stage",
                "evaluate",
                "--checkpoint",
                stage3_checkpoint,
                "--episodes",
                "128",
                "--record-video",
                "--out-dir",
                str(stage3_dir / "evaluate"),
            ),
        ),
        PipelineStep(
            "stage3_gate",
            _gate_command(python, "stage3", stage3_metrics, out / "stage3_gate.json"),
        ),
    )


def _gate_command(
    python: str,
    stage: str,
    metrics: str | None,
    output: Path,
    *,
    consecutive: int | None = None,
    baseline_metrics: str | None = None,
    checkpoint: str | None = None,
    finger_perturb_qpos_scale: float | None = None,
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "musclemimic.badminton.training_gates",
        "--stage",
        stage,
        "--metrics",
        metrics or f"<required:{stage}_metrics>",
        "--output",
        str(output),
        "--require_pass",
    ]
    if consecutive is not None:
        command.extend(("--consecutive", str(int(consecutive))))
    if baseline_metrics is not None:
        command.extend(("--baseline-metrics", str(baseline_metrics)))
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if finger_perturb_qpos_scale is not None:
        command.extend(
            (
                "--finger-perturb-qpos-scale",
                str(float(finger_perturb_qpos_scale)),
            )
        )
    return tuple(command)


def execute_pipeline_step(
    step_name: str,
    *,
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
) -> None:
    steps = {step.name: step for step in build_pipeline_plan(output_dir, artifacts)}
    if step_name not in steps:
        raise ValueError(f"unknown pipeline step {step_name!r}; expected one of {sorted(steps)}")
    step = steps[step_name]
    missing = [name for name in step.required_artifacts if not getattr(artifacts, name)]
    if missing:
        raise ValueError(f"pipeline step {step_name} is missing required artifacts: {missing}")
    _verify_upstream_gates(step_name, artifacts, output_dir=output_dir)
    env = os.environ.copy()
    env.update(dict(step.environment))
    subprocess.run(list(step.command), cwd=REPO_ROOT, env=env, check=True)


def _verify_upstream_gates(
    step_name: str,
    artifacts: PipelineArtifacts,
    *,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    if step_name == "latent_distill":
        _require_direct_outputs(out, artifacts)
    if step_name == "stage1_train":
        qc_path = out / "data_qc.json"
        if not qc_path.is_file():
            raise ValueError(f"Stage 1 requires a completed data QC report: {qc_path}")
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("passed") is not True:
            raise ValueError("canonical raw data QC did not pass")
        _require_canonical_cache_environment(
            qc,
            qc_path=qc_path,
            preflight_path=out / "data_preflight_binding.json",
        )
    if step_name in {
        "stage1_visual_gate",
        "stage1_promote",
        "stage1r_train",
        "stage2_train",
    }:
        _require_metrics_gate("stage1", artifacts.stage1_metrics, consecutive=3)
    if step_name in {"stage1_promote", "stage1r_train", "stage2_train"}:
        _require_visual_review(
            artifacts.stage1_visual_review,
            review_kind=STAGE1_REVIEW_KIND,
            stage_label="Stage 1",
            checkpoint=artifacts.stage1_checkpoint,
        )
    if step_name in {
        "stage1r_train",
        "stage2_train",
        "stage2_extend_160m",
        "stage2_promote",
    }:
        _require_promoted_artifact(
            artifacts.stage1_promotion_manifest
            or str(out / "stage1_promotion_manifest.json"),
            stage="stage1",
            checkpoint=artifacts.stage1_checkpoint,
        )
    if step_name in {"stage1r_gate", "stage1r005_train"}:
        _require_stage1r_artifact(
            artifacts.stage1r_metrics,
            checkpoint=artifacts.stage1r_checkpoint,
            expected_scale=0.03,
        )
    if step_name == "stage1r005_train":
        _require_metrics_gate("stage1r", artifacts.stage1r_metrics, consecutive=1)
    if step_name == "stage1r005_gate":
        _require_stage1r_artifact(
            artifacts.stage1r005_metrics,
            checkpoint=artifacts.stage1r005_checkpoint,
            expected_scale=0.05,
        )
    if step_name == "stage2_extend_160m":
        _require_stage2_extension_eligible(artifacts.stage2_metrics)
    if step_name in {
        "stage2_visual_gate",
        "stage2_promote",
        "direct_distill",
        "direct_distill_resume",
        "latent_distill",
    }:
        _require_metrics_gate(
            "stage2",
            artifacts.stage2_metrics,
            consecutive=3,
            baseline_path=artifacts.stage1_metrics,
        )
    if step_name in {
        "stage2_promote",
        "direct_distill",
        "direct_distill_resume",
        "latent_distill",
    }:
        _require_visual_review(
            artifacts.stage2_visual_review,
            review_kind=STAGE2_REVIEW_KIND,
            stage_label="Stage 2",
            checkpoint=artifacts.stage2_checkpoint,
        )
    if step_name in {"direct_distill", "direct_distill_resume", "latent_distill"}:
        _require_promoted_artifact(
            artifacts.stage2_promotion_manifest
            or str(out / "stage2_promotion_manifest.json"),
            stage="stage2",
            checkpoint=artifacts.stage2_checkpoint,
        )
    if step_name in {"latent_closed_loop_gate", "latent_gate"}:
        _require_direct_outputs(out, artifacts)
    if step_name in {
        "latent_gate",
        "stage3_preflight",
        "stage3_base_only",
        "stage3_feed_gate",
        "stage3_train",
        "stage3_extend_curriculum",
        "stage3_evaluate",
        "stage3_gate",
    }:
        checkpoint = Path(
            artifacts.latent_checkpoint or out / "latent_distill" / "latent_checkpoint"
        )
        metrics_path = checkpoint / "eval_metrics.json"
        if not metrics_path.is_file():
            raise ValueError(f"latent checkpoint has no eval_metrics.json: {checkpoint}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("promotion", {}).get("passed") is not True:
            raise ValueError("latent checkpoint has not passed prior/decoder promotion gates")
    stage3_dir = out / "stage3_lab"
    if step_name in {
        "stage3_base_only",
        "stage3_feed_gate",
        "stage3_train",
        "stage3_extend_curriculum",
        "stage3_evaluate",
        "stage3_gate",
    }:
        _require_passed_report(
            stage3_dir / "preflight_report.json",
            label="Stage-3 preflight",
        )
    if step_name in {
        "stage3_feed_gate",
        "stage3_train",
        "stage3_evaluate",
        "stage3_gate",
    }:
        _require_passed_report(
            stage3_dir / "base_only_report.json",
            label="Stage-3 base-only check",
        )
    if step_name in {
        "stage3_train",
        "stage3_extend_curriculum",
        "stage3_evaluate",
        "stage3_gate",
    }:
        _require_passed_report(
            stage3_dir / "feed_check_report.json",
            label="Stage-3 feed check",
        )
    if step_name == "stage3_evaluate":
        stage3_checkpoint = Path(
            artifacts.stage3_checkpoint or stage3_dir / "policy_latest.json"
        )
        if not stage3_checkpoint.is_file():
            raise ValueError(
                f"Stage-3 evaluation requires a trained checkpoint: {stage3_checkpoint}"
            )
    if step_name in {"stage3_evaluate", "stage3_gate"}:
        _require_stage3_curriculum_complete(stage3_dir / "train_report.json")
    if step_name == "stage3_extend_curriculum":
        _require_stage3_curriculum_incomplete(stage3_dir / "train_report.json")
    if step_name == "stage3_gate":
        _require_stage3_artifact_binding(
            Path(artifacts.stage3_metrics or stage3_dir / "evaluate" / "evaluate_report.json")
        )


def _require_stage3_curriculum_complete(path: Path) -> None:
    report = _load_json_mapping(path, label="Stage-3 train report")
    if report.get("curriculum_complete") is not True:
        raise ValueError(
            "Stage-3 curriculum is incomplete; run stage3_extend_curriculum before evaluation"
        )
    if report.get("promotion_eligible") is not True:
        raise ValueError("Stage-3 train report is not promotion eligible")


def _require_stage3_curriculum_incomplete(path: Path) -> None:
    report = _load_json_mapping(path, label="Stage-3 train report")
    if report.get("curriculum_complete") is True:
        raise ValueError("Stage-3 curriculum is already complete; extension is not allowed")
    if report.get("extension_required") is not True:
        raise ValueError("Stage-3 train report did not explicitly request an extension")


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_stage3_artifact_binding(report_path: Path) -> None:
    report = _load_json_mapping(report_path, label="Stage-3 evaluation report")
    binding = report.get("artifact_binding")
    if not isinstance(binding, dict) or binding.get("verified") is not True:
        raise ValueError("Stage-3 evaluation artifact binding is missing or unverified")
    if binding.get("schema_version") != "incoming_hit_evaluation_artifact_binding_v3":
        raise ValueError("Stage-3 evaluation artifact binding schema is incompatible")
    recorded_binding_hash = binding.get("binding_sha256")
    unbound = dict(binding)
    unbound.pop("binding_sha256", None)
    if recorded_binding_hash != _canonical_mapping_sha256(unbound):
        raise ValueError("Stage-3 evaluation artifact binding hash mismatch")

    from environment.overall_environment.src.train_incoming_hit_mjx import (
        resolve_training_checkpoint,
    )
    from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
        _stage3_evaluation_content_sha256,
        _validate_stage3_training_prerequisite_binding,
    )

    if binding.get("evaluation_content_sha256") != _stage3_evaluation_content_sha256(
        report
    ):
        raise ValueError("Stage-3 evaluation metrics/content changed after binding")

    checkpoint_value = Path(str(report.get("checkpoint", ""))).expanduser()
    if not checkpoint_value.is_absolute():
        checkpoint_value = REPO_ROOT / checkpoint_value
    payload_path, metadata_path = resolve_training_checkpoint(checkpoint_value)
    recorded_payload_path = Path(str(binding.get("checkpoint_payload_path", ""))).expanduser()
    recorded_metadata_path = Path(str(binding.get("checkpoint_metadata_path", ""))).expanduser()
    if recorded_payload_path.resolve() != payload_path.resolve():
        raise ValueError("Stage-3 binding names a different checkpoint payload")
    if recorded_metadata_path.resolve() != metadata_path.resolve():
        raise ValueError("Stage-3 binding names different checkpoint metadata")
    file_contracts = (
        (payload_path, "checkpoint_payload_sha256"),
        (metadata_path, "checkpoint_metadata_sha256"),
        (Path(str(binding.get("spec_path", ""))), "spec_sha256"),
        (Path(str(binding.get("scene_path", ""))), "scene_sha256"),
        (Path(str(binding.get("train_report_path", ""))), "train_report_sha256"),
    )
    for path, hash_key in file_contracts:
        if not path.is_file():
            raise ValueError(f"Stage-3 bound artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if binding.get(hash_key) != actual:
            raise ValueError(f"Stage-3 bound artifact changed: {path}")
    metadata = _load_json_mapping(metadata_path, label="Stage-3 checkpoint metadata")
    train_report = _load_json_mapping(
        Path(str(binding.get("train_report_path", ""))),
        label="Stage-3 bound train report",
    )
    prerequisite_binding = _validate_stage3_training_prerequisite_binding(
        metadata.get("training_prerequisite_binding")
    )
    if train_report.get("training_prerequisite_binding") != prerequisite_binding:
        raise ValueError(
            "Stage-3 bound train report and checkpoint prerequisite evidence differ"
        )
    if binding.get("training_prerequisite_binding_sha256") != prerequisite_binding.get(
        "binding_sha256"
    ):
        raise ValueError("Stage-3 evaluation names different prerequisite evidence")
    curriculum_state = metadata.get("curriculum_state")
    if not isinstance(curriculum_state, dict):
        raise ValueError("Stage-3 checkpoint metadata has no curriculum state")
    consistency_fields = (
        ("iterations", "iteration", metadata),
        ("env_steps", "env_steps", metadata),
        ("curriculum_effective_steps", "effective_steps", curriculum_state),
        ("curriculum_phase", "phase", curriculum_state),
        ("curriculum_complete", "curriculum_complete", metadata),
        ("promotion_eligible", "promotion_eligible", metadata),
    )
    for report_key, metadata_key, source in consistency_fields:
        if train_report.get(report_key) != source.get(metadata_key):
            raise ValueError(
                "Stage-3 bound train report and checkpoint disagree on "
                f"{report_key}"
            )
    if metadata.get("curriculum_complete") is not True or metadata.get(
        "promotion_eligible"
    ) is not True:
        raise ValueError("Stage-3 bound checkpoint is not curriculum complete")
    if binding.get("checkpoint_iteration") != metadata.get("iteration"):
        raise ValueError("Stage-3 binding checkpoint iteration changed")
    if binding.get("checkpoint_env_steps") != metadata.get("env_steps"):
        raise ValueError("Stage-3 binding checkpoint env-step identity changed")
    control = report.get("control_manifest")
    if not isinstance(control, dict) or binding.get("control_hash") != control.get(
        "control_hash"
    ):
        raise ValueError("Stage-3 bound control manifest changed")
    if binding.get("latent_checkpoint_fingerprint") != control.get(
        "latent_checkpoint_fingerprint"
    ):
        raise ValueError("Stage-3 bound latent checkpoint changed")
    if prerequisite_binding.get("control_hash") != control.get(
        "control_hash"
    ) or prerequisite_binding.get(
        "latent_checkpoint_fingerprint"
    ) != control.get("latent_checkpoint_fingerprint"):
        raise ValueError("Stage-3 prerequisite control/latent identity changed")
    for report_key, binding_key in (
        ("training_feed_manifest", "training_feed_manifest_sha256"),
        ("evaluation_feed_manifest", "evaluation_feed_manifest_sha256"),
    ):
        manifest = report.get(report_key)
        if not isinstance(manifest, dict):
            raise ValueError(f"Stage-3 report has no {report_key}")
        if binding.get(binding_key) != _canonical_mapping_sha256(manifest):
            raise ValueError(f"Stage-3 bound {report_key} changed")
        if report_key == "training_feed_manifest" and prerequisite_binding.get(
            "training_feed_manifest_sha256"
        ) != _canonical_mapping_sha256(manifest):
            raise ValueError("Stage-3 prerequisite training feed identity changed")


def _canonical_mapping_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_passed_report(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} report is unreadable: {path}") from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError(f"{label} did not pass: {path}")


def _require_direct_outputs(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    direct_dir = output_dir / "direct_distill"
    bc_path = Path(
        artifacts.direct_bc_metrics or direct_dir / "bc" / "distill_metadata.json"
    )
    rollout_path = Path(
        artifacts.direct_rollout_metrics
        or direct_dir / "compare" / "comparison_metrics.json"
    )
    acceptance_path = Path(
        artifacts.direct_acceptance or direct_dir / "compare" / "acceptance.json"
    )
    missing = [str(path) for path in (bc_path, rollout_path, acceptance_path) if not path.is_file()]
    if missing:
        raise ValueError(f"direct distillation artifacts are incomplete: {missing}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    promoted = (
        acceptance.get("student_bc_ppo")
        or acceptance.get("student_bc_dagger")
        or acceptance.get("student_bc")
    )
    if not isinstance(promoted, dict) or promoted.get("passed") is not True:
        raise ValueError("direct distilled policy has not passed held-out acceptance gates")


def _require_visual_review(
    path: str | None,
    *,
    review_kind: str,
    stage_label: str,
    checkpoint: str | None,
) -> None:
    if path is None:
        raise ValueError(f"{stage_label} human visual review artifact is required")
    review_path = Path(path)
    if not review_path.is_file():
        raise ValueError(
            f"{stage_label} human visual review artifact does not exist: {review_path}"
        )
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{stage_label} human visual review artifact is unreadable: {review_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{stage_label} human visual review root must be a JSON object")
    basic_report = validate_visual_review(
        payload,
        required_clips=5,
        expected_motions=VAL_MOTIONS,
        required_review_kind=review_kind,
    )
    if basic_report["passed"] is not True:
        raise ValueError(
            f"{stage_label} human visual review gate did not pass: "
            + "; ".join(str(error) for error in basic_report["errors"])
        )
    if checkpoint is None:
        raise ValueError(f"{stage_label} promoted checkpoint is required for visual binding")
    from musclemimic.badminton.promotion_artifact import checkpoint_identity

    report = validate_visual_review(
        payload,
        required_clips=5,
        expected_motions=VAL_MOTIONS,
        required_review_kind=review_kind,
        expected_candidate=checkpoint_identity(checkpoint),
    )
    if report["passed"] is not True:
        raise ValueError(
            f"{stage_label} human visual review gate did not pass: "
            + "; ".join(str(error) for error in report["errors"])
        )


def _require_promoted_artifact(
    path: str,
    *,
    stage: str,
    checkpoint: str | None,
) -> None:
    if checkpoint is None:
        raise ValueError(f"{stage} promoted checkpoint is required")
    try:
        validate_promoted_artifact(
            path,
            expected_stage=stage,
            expected_checkpoint=checkpoint,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{stage} promoted artifact is invalid: {exc}") from exc


def _require_stage2_extension_eligible(path: str | None) -> None:
    """Permit the second 80M only after a useful, incomplete first tranche."""

    if path is None:
        raise ValueError("Stage-2 80M promotion progress is required for extension")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("stopped_early") is True:
        raise ValueError("Stage-2 already passed promotion; extension is not allowed")
    if payload.get("hard_cap_reached") is not True:
        raise ValueError("Stage-2 extension requires the initial 80M hard cap")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) < 3:
        raise ValueError("Stage-2 extension requires at least three held-out validations")
    records = [event.get("metrics") for event in history[-3:] if isinstance(event, dict)]
    if len(records) != 3 or not all(isinstance(record, dict) for record in records):
        raise ValueError("Stage-2 extension history has malformed validation records")

    lower_is_better = (
        "val_early_termination_rate",
        "val_err_racket_pos",
        "val_err_racket_rot",
        "val_err_rpos",
    )
    improvements: list[float] = []
    for name in lower_is_better:
        try:
            values = [float(record[name]) for record in records]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage-2 extension history is missing finite {name}"
            ) from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"Stage-2 extension history has invalid {name}")
        # No mandatory metric may deteriorate by more than five percent over
        # the final three boundaries; at least one must improve materially.
        if values[-1] > values[0] * 1.05 + 1e-8:
            raise ValueError(f"Stage-2 {name} is deteriorating; extension is unsafe")
        improvements.append((values[0] - values[-1]) / max(values[0], 1e-8))
    coverage = [float(record["val_frame_coverage"]) for record in records]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coverage):
        raise ValueError("Stage-2 extension history has invalid frame coverage")
    if coverage[-1] + 0.01 < coverage[0]:
        raise ValueError("Stage-2 frame coverage is deteriorating; extension is unsafe")
    if max([*improvements, coverage[-1] - coverage[0]]) < 0.01:
        raise ValueError("Stage-2 metrics are plateaued; repair/tune instead of extending")


def _require_stage1r_artifact(
    path: str | None,
    *,
    checkpoint: str | None,
    expected_scale: float,
) -> None:
    if path is None:
        raise ValueError("Stage-1R paired robustness report is required")
    if checkpoint is None:
        raise ValueError("Stage-1R checkpoint is required for report binding")
    try:
        validate_stage1r_report(
            path,
            expected_checkpoint=checkpoint,
            expected_perturb_qpos_scale=expected_scale,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Stage-1R promotion artifact is invalid: {exc}") from exc


def _require_canonical_cache_environment(
    qc_report: dict[str, object],
    *,
    qc_path: Path,
    preflight_path: Path,
) -> None:
    """Bind Stage 1 to the clean, immutable ``raw_smooth_v1`` release."""

    if qc_report.get("passed") is not True or qc_report.get("clean_passed") is not True:
        raise ValueError("Stage 1 requires warning-free raw_smooth_v1 data QC")
    if qc_report.get("source_variant") != DATA_VARIANT:
        raise ValueError("Stage 1 data QC used the wrong source variant")
    if qc_report.get("cache_variant") != DATA_VARIANT:
        raise ValueError("Stage 1 data QC used the wrong cache variant")
    expected_source_dir = DATASET_ROOT / "temp" / DATA_VARIANT
    expected_cache_dir = DATASET_ROOT / "muscle_trajectory" / DATA_VARIANT
    if Path(str(qc_report.get("resolved_source_dir", ""))).resolve() != expected_source_dir.resolve():
        raise ValueError("Stage 1 data QC source namespace is not canonical raw_smooth_v1")
    if Path(str(qc_report.get("resolved_cache_dir", ""))).resolve() != expected_cache_dir.resolve():
        raise ValueError("Stage 1 data QC cache namespace is not canonical raw_smooth_v1")
    if tuple(qc_report.get("train_motions", ())) != TRAIN_MOTIONS:
        raise ValueError("Stage 1 data QC train split is not the canonical ordered 22-motion split")
    if tuple(qc_report.get("validation_motions", ())) != VAL_MOTIONS:
        raise ValueError("Stage 1 data QC validation split is not the canonical ordered 5-motion split")

    cache_root_value = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH")
    if not cache_root_value:
        raise ValueError(
            "MUSCLEMIMIC_GMR_CACHE_PATH is unset; run `source configs/env.sh` "
            "before starting Stage 1"
        )
    cache_root = Path(cache_root_value).expanduser().resolve()
    qc_dataset_root = Path(str(qc_report.get("dataset_root", ""))).resolve()
    expected_dataset_root = cache_root / "forehandClear_standard"
    if expected_dataset_root != qc_dataset_root:
        raise ValueError(
            "data QC and runtime cache roots differ: "
            f"qc={qc_dataset_root} runtime={expected_dataset_root}"
        )
    raw_dir = expected_dataset_root / "muscle_trajectory" / DATA_VARIANT
    missing = [
        str(raw_dir / f"{motion}.npz")
        for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS)
        if not (raw_dir / f"{motion}.npz").is_file()
    ]
    if missing:
        raise ValueError(f"runtime raw_smooth_v1 cache is incomplete: {missing}")

    release_validation = validate_release_manifest(DATASET_ROOT, RELEASE_MANIFEST)
    if release_validation.get("passed") is not True:
        raise ValueError(
            "raw_smooth_v1 release manifest validation failed: "
            + "; ".join(str(error) for error in release_validation.get("errors", ()))
        )
    release_sha = release_validation.get("release_sha256")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise ValueError("raw_smooth_v1 release manifest has no valid content identity")
    visual_qc_path = RELEASE_MANIFEST.with_name("visual_qc_report.json")
    visual_validation = validate_visual_qc_report(REPO_ROOT, visual_qc_path)
    if visual_validation.get("passed") is not True:
        raise ValueError(
            "raw_smooth_v1 visual QC validation failed: "
            + "; ".join(str(error) for error in visual_validation.get("errors", ()))
        )
    binding: dict[str, object] = {
        "schema_version": "forehand_clear_data_preflight_binding_v1",
        "dataset_root": str(DATASET_ROOT.resolve()),
        "source_variant": DATA_VARIANT,
        "cache_variant": DATA_VARIANT,
        "qc_report_path": str(qc_path.resolve()),
        "qc_report_sha256": hashlib.sha256(qc_path.read_bytes()).hexdigest(),
        "release_manifest_path": str(RELEASE_MANIFEST.resolve()),
        "release_manifest_sha256": hashlib.sha256(RELEASE_MANIFEST.read_bytes()).hexdigest(),
        "release_sha256": release_sha,
        "visual_qc_report_path": str(visual_qc_path.resolve()),
        "visual_qc_report_sha256": visual_validation["report_sha256"],
        "clean_passed": True,
    }
    binding["binding_sha256"] = _canonical_mapping_sha256(binding)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = preflight_path.with_name(f".{preflight_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, preflight_path)


def _require_metrics_gate(
    stage: str,
    path: str | None,
    *,
    consecutive: int,
    baseline_path: str | None = None,
) -> None:
    if path is None:
        raise ValueError(f"{stage} validation metrics are required")
    metrics = extract_validation_records(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    baseline = None
    if baseline_path is not None:
        baseline = latest_validation_record(
            json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        )
    if not evaluate_promotion(
        stage,
        metrics,
        consecutive=consecutive,
        baseline_metrics=baseline,
    ).passed:
        raise ValueError(f"{stage} promotion gate did not pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/forehand_clear_three_stage_v1")
    parser.add_argument("--execute_step", default=None)
    for field_name in PipelineArtifacts.__dataclass_fields__:
        parser.add_argument(f"--{field_name}", default=None)
    args = parser.parse_args()
    artifacts = PipelineArtifacts(
        **{field_name: getattr(args, field_name) for field_name in PipelineArtifacts.__dataclass_fields__}
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = build_pipeline_plan(output, artifacts)
    plan_path = output / "pipeline_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "forehand_clear_pipeline_v2",
                "artifacts": asdict(artifacts),
                "steps": [asdict(step) for step in steps],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"pipeline_plan: {plan_path}")
    if args.execute_step:
        execute_pipeline_step(args.execute_step, output_dir=output, artifacts=artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
