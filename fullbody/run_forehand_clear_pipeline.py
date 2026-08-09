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

from musclemimic.badminton.action_registry import (
    DEFAULT_ACTION,
    ActionSpec,
    action_choices,
    resolve,
)
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
DEFAULT_SPEC = resolve(DEFAULT_ACTION)
DATA_VARIANT = DEFAULT_SPEC.data_variant
DATASET_ROOT = DEFAULT_SPEC.dataset_root
RELEASE_MANIFEST = REPO_ROOT / DEFAULT_SPEC.release_manifest
# ``latent_synergy_sweep.py`` falls back to this config when ``--base-config``
# is absent (see its DEFAULT_CONFIG).  The sweep step therefore passes the flag
# only for actions whose config differs, keeping the sealed forehand-clear
# command byte-identical.
_SWEEP_DEFAULT_BASE_CONFIG = "fullbody/config_specific_task/distill/latent_forehandclear_synergy_v3.yaml"


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
    stage2_checkpoint_fingerprint: str | None = None
    racket_mass_025_checkpoint: str | None = None
    racket_mass_025_metrics: str | None = None
    racket_mass_025_visual_review: str | None = None
    racket_mass_025_physics_manifest: str | None = None
    racket_mass_025_promotion_manifest: str | None = None
    racket_mass_050_checkpoint: str | None = None
    racket_mass_050_metrics: str | None = None
    racket_mass_050_visual_review: str | None = None
    racket_mass_050_physics_manifest: str | None = None
    racket_mass_050_promotion_manifest: str | None = None
    racket_mass_075_checkpoint: str | None = None
    racket_mass_075_metrics: str | None = None
    racket_mass_075_visual_review: str | None = None
    racket_mass_075_physics_manifest: str | None = None
    racket_mass_075_promotion_manifest: str | None = None
    racket_mass_100_checkpoint: str | None = None
    racket_mass_100_metrics: str | None = None
    racket_mass_100_visual_review: str | None = None
    racket_mass_100_physics_manifest: str | None = None
    racket_mass_100_promotion_manifest: str | None = None
    racket_mass_100_checkpoint_fingerprint: str | None = None
    direct_bc_metrics: str | None = None
    direct_rollout_metrics: str | None = None
    direct_acceptance: str | None = None
    latent_checkpoint: str | None = None
    stage3_checkpoint: str | None = None
    stage3_metrics: str | None = None
    event_reference_metrics: str | None = None
    train_event_reference_manifest_list: str | None = None
    val_event_reference_manifest_list: str | None = None
    train_event_reference_bank: str | None = None
    val_event_reference_bank: str | None = None
    train_event_reference_fingerprint: str | None = None
    val_event_reference_fingerprint: str | None = None
    physical_rollout_metrics: str | None = None
    synergy_basis: str | None = None
    synergy_basis_fingerprint: str | None = None
    frozen_body_decoder: str | None = None
    frozen_body_decoder_fingerprint: str | None = None
    body_synergy_contract_fingerprint: str | None = None
    body_synergy_portable_core_fingerprint: str | None = None
    synergy_grouping: str | None = None
    synergy_metrics: str | None = None
    latent_dimension_metrics: str | None = None
    latent_direct_checkpoint: str | None = None
    latent_synergy_checkpoint: str | None = None
    latent_synergy_metrics: str | None = None
    latent_selection_manifest: str | None = None
    latent_causal_adapter_config: str | None = None
    static_target_checkpoint: str | None = None
    static_target_metrics: str | None = None
    stage3_v2_checkpoint: str | None = None
    stage3_v2_metrics: str | None = None
    direct_static_target_checkpoint: str | None = None
    direct_static_target_metrics: str | None = None
    direct_stage3_v2_checkpoint: str | None = None
    direct_stage3_v2_metrics: str | None = None
    stage3_paired_metrics: str | None = None
    stage3_task_causal_config: str | None = None
    stage3_task_causal_metrics: str | None = None
    stage3_signal_identity_json: str | None = None
    stage3_signal_npz: str | None = None
    stage3_signal_sidecar_json: str | None = None
    recovery_target_bank: str | None = None
    recovery_eval_target_bank: str | None = None
    recovery_train_feed_bank: str | None = None
    recovery_eval_feed_bank: str | None = None
    desired_impact_targets_jsonl: str | None = None
    desired_impact_targets_eval_jsonl: str | None = None
    emg_metrics: str | None = None
    emg_simulation_npz: str | None = None
    emg_measurement_npz: str | None = None
    emg_mapping_json: str | None = None
    # Training-time privileged EMG (PEASD).  Distinct from the fields above,
    # which validate a trained policy after the fact: this one feeds a reviewed
    # phase-reference tube to the posterior during latent distillation.  Absent
    # by default so the EMG-free baseline arm stays the launcher default.
    emg_reference_manifest: str | None = None
    emg_synergy_dim: int | None = None
    # Negative control (§26.2 S2-D).  A gate, not a decoration: if the real
    # context does not beat its shuffled twin, the privileged claim is unearned.
    emg_shuffle_context_ablation: bool = False
    # §26.2 S2-E: privileged latent trained with context dropout forced to 0.
    # A distinct arm of the ablation matrix, not a knob for the real arm.
    emg_no_context_dropout: bool = False
    # §26.3 H3: grouped right-arm correction on top of the frozen latent/decoder.
    # Path to a JSON mapping of bounded_residual.groups; when set, Stage-3
    # train/eval/base-only inject it and enable the grouped residual.
    stage3_bounded_residual_groups_json: str | None = None
    expected_policy_checkpoint_fingerprint: str | None = None
    expected_policy_promotion_fingerprint: str | None = None
    expected_formal_synergy_basis_fingerprint: str | None = None
    expected_event_reference_fingerprint: str | None = None
    expected_session_uid: str | None = None
    expected_policy_decoder_type: str | None = None
    physiology_input_npz: str | None = None
    physiology_config_json: str | None = None
    physiology_metrics: str | None = None
    ablation_jsonl: str | None = None


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    required_artifacts: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


def _build_stage1_aligned_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
) -> tuple[PipelineStep, ...]:
    out = Path(output_dir)
    python = sys.executable
    dataset_root = spec.dataset_root
    release_manifest = REPO_ROOT / spec.release_manifest
    stage1_ckpt = artifacts.stage1_checkpoint or "<required:stage1_checkpoint>"
    stage1_promotion = artifacts.stage1_promotion_manifest or str(out / "stage1_promotion_manifest.json")
    return (
        PipelineStep(
            "data_release_validate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.data_release",
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(release_manifest),
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
                str(dataset_root),
                "--source-variant",
                spec.source_variant,
                "--cache-variant",
                spec.cache_variant,
                "--require-clean",
                "--output",
                str(out / "data_qc.json"),
            ),
        ),
        PipelineStep(
            "stage1_train",
            (
                python,
                "fullbody/experiment.py",
                f"--config-name={spec.stage1_config}",
            ),
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
                *spec.val_motions,
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
    )


def _build_legacy_pipeline_plan(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
) -> tuple[PipelineStep, ...]:
    out = Path(output_dir)
    python = sys.executable
    stage1_ckpt = artifacts.stage1_checkpoint or "<required:stage1_checkpoint>"
    stage1r_ckpt = artifacts.stage1r_checkpoint or "<required:stage1r_checkpoint>"
    stage2_ckpt = artifacts.stage2_checkpoint or "<required:stage2_checkpoint>"
    stage1_promotion = artifacts.stage1_promotion_manifest or str(out / "stage1_promotion_manifest.json")

    stage2_promotion = artifacts.stage2_promotion_manifest or str(out / "stage2_promotion_manifest.json")
    train_motions = spec.train_motion_paths
    val_motions = spec.val_motion_paths
    direct_dir = out / "direct_distill"
    latent_dir = out / "latent_distill"
    stage3_dir = out / "stage3_lab"
    direct_bc_metrics = artifacts.direct_bc_metrics or str(direct_dir / "bc" / "distill_metadata.json")
    direct_rollout_metrics = artifacts.direct_rollout_metrics or str(direct_dir / "compare" / "comparison_metrics.json")
    latent_checkpoint = artifacts.latent_checkpoint or str(latent_dir / "latent_checkpoint")
    stage3_checkpoint = artifacts.stage3_checkpoint or str(stage3_dir / "policy_latest.json")
    stage3_metrics = artifacts.stage3_metrics or str(stage3_dir / "evaluate" / "evaluate_report.json")
    direct_distill_command = (
        python,
        "-m",
        "fullbody.run_distill_experiment",
        "--teacher_ckpt",
        stage2_ckpt,
        "--teacher_promotion_manifest",
        stage2_promotion,
        "--student_config",
        spec.require("student_bc_config"),
        "--student_ppo_config",
        spec.require("student_ppo_config"),
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
        *_build_stage1_aligned_steps(output_dir, artifacts, spec=spec),
        PipelineStep(
            "stage1r_train",
            (
                python,
                "fullbody/experiment.py",
                f"--config-name={spec.require('stage1r_config')}",
            ),
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
                artifacts.stage1r_metrics or str(out / "stage1r_003" / "paired_robustness.json"),
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
                f"--config-name={spec.require('stage1r005_config')}",
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
                artifacts.stage1r005_checkpoint or "<required:stage1r005_checkpoint>",
                "--motion_path",
                *val_motions,
                "--perturb_qpos_scale",
                "0.05",
                "--perturb_qvel_scale",
                "0.0",
                "--output",
                artifacts.stage1r005_metrics or str(out / "stage1r_005" / "paired_robustness.json"),
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
                checkpoint=(artifacts.stage1r005_checkpoint or "<required:stage1r005_checkpoint>"),
                finger_perturb_qpos_scale=0.05,
            ),
            ("stage1r005_checkpoint", "stage1r005_metrics"),
        ),
        PipelineStep(
            "stage2_train",
            (
                python,
                "fullbody/experiment.py",
                f"--config-name={spec.require('stage2_config')}",
                f"experiment.resume_from={stage1_ckpt}",
                f"experiment.promotion.baseline_metrics_path={artifacts.stage1_metrics or '<required:stage1_metrics>'}",
            ),
            ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review"),
        ),
        PipelineStep(
            "stage2_extend_160m",
            (
                python,
                "fullbody/experiment.py",
                f"--config-name={spec.require('stage2_extend_config')}",
                f"experiment.resume_from={stage2_ckpt}",
                f"experiment.promotion.baseline_metrics_path={artifacts.stage1_metrics or '<required:stage1_metrics>'}",
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
                *spec.val_motions,
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
                spec.require("latent_lab_config"),
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
                spec.require("stage3_spec"),
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
                spec.require("stage3_spec"),
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
                spec.require("stage3_spec"),
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
                spec.require("stage3_spec"),
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
                spec.require("stage3_spec"),
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
                spec.require("stage3_spec"),
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


def build_pipeline_plan(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    profile: str = "legacy_v2",
    spec: ActionSpec = DEFAULT_SPEC,
) -> tuple[PipelineStep, ...]:
    """Return a command-only plan; constructing it never launches training."""

    if profile == "stage1_aligned":
        return _build_stage1_aligned_steps(output_dir, artifacts, spec=spec)
    legacy = _build_legacy_pipeline_plan(output_dir, artifacts, spec=spec)
    if profile == "legacy_v2":
        return legacy
    if profile != "synergy_v3":
        raise ValueError("profile must be 'legacy_v2', 'synergy_v3', or 'stage1_aligned'")
    stage1r_end = next(index for index, step in enumerate(legacy) if step.name == "stage1r005_gate")
    return (*legacy[: stage1r_end + 1], *_build_synergy_v3_steps(output_dir, artifacts, spec=spec))


def _build_synergy_v3_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
) -> tuple[PipelineStep, ...]:
    root = Path(output_dir) / "synergy_v3"
    python = sys.executable
    stage3_v2_spec = spec.require("stage3_v2_spec")
    stage3_direct_spec = spec.require("stage3_direct_spec")
    # The v3 physical dataset is deliberately collected from the final 100%
    # load rung.  Falling back to the legacy Stage-2 teacher would make a
    # nominal mass-curriculum run silently train on the wrong dynamics.
    teacher = artifacts.racket_mass_100_checkpoint or ("<required:racket_mass_100_checkpoint>")
    physical_train = root / "physical_rollout" / "train"
    physical_val = root / "physical_rollout" / "val"
    physical_metrics = artifacts.physical_rollout_metrics or str(root / "physical_rollout" / "promotion_metrics.json")
    direct_dir = root / "direct_baseline"
    direct_checkpoint = direct_dir / "bc" / "checkpoints" / "checkpoint_200000"
    direct_bc_metrics = artifacts.direct_bc_metrics or str(direct_dir / "bc" / "distill_metadata.json")
    direct_rollout_metrics = artifacts.direct_rollout_metrics or str(direct_dir / "compare" / "comparison_metrics.json")
    direct_acceptance = artifacts.direct_acceptance or str(direct_dir / "compare" / "direct_promotion_evidence.json")
    synergy_dir = root / "synergy"
    synergy_metrics = artifacts.synergy_metrics or str(synergy_dir / "promotion_metrics.json")
    basis = artifacts.synergy_basis or str(synergy_dir / "physical_excitation_unit" / "regional_composite")
    basis_fingerprint = artifacts.synergy_basis_fingerprint or "<required:synergy_basis_fingerprint>"
    frozen_body_decoder = artifacts.frozen_body_decoder or ("<required:frozen_body_decoder_from_stage1_release>")
    frozen_body_decoder_fingerprint = (
        artifacts.frozen_body_decoder_fingerprint or "<required:frozen_body_decoder_fingerprint>"
    )
    body_synergy_contract_fingerprint = (
        artifacts.body_synergy_contract_fingerprint or "<required:body_synergy_contract_fingerprint>"
    )
    body_synergy_portable_core_fingerprint = (
        artifacts.body_synergy_portable_core_fingerprint or "<required:body_synergy_portable_core_fingerprint>"
    )
    latent_dir = root / "latent_synergy"
    latent_checkpoint = artifacts.latent_synergy_checkpoint or str(latent_dir / "selected" / "best_synergy")
    # ``best_direct`` from the latent sweep is retained only as an explicitly
    # named latent-decoder ablation.  The formal Stage-3 direct comparator
    # below is a fresh policy whose output is the ordered 354-muscle action.
    latent_synergy_metrics = artifacts.latent_synergy_metrics or str(latent_dir / "promotion_metrics.json")
    latent_causal_adapter_config = artifacts.latent_causal_adapter_config or ("<required:latent_causal_adapter_config>")
    stage3_dir = root / "stage3_impact_recovery"
    direct_stage3_dir = root / "stage3_impact_recovery_direct"
    # Static C0--C3 and resumed C4--C7 share one versioned checkpoint root so
    # immutable prerequisite reports and resume ancestry remain identical.
    static_dir = stage3_dir
    static_checkpoint = artifacts.static_target_checkpoint or str(static_dir / "policy_latest.json")
    static_metrics = artifacts.static_target_metrics or str(static_dir / "evaluate_static" / "evaluate_report.json")
    stage3_metrics = artifacts.stage3_v2_metrics or str(stage3_dir / "evaluate" / "evaluate_report.json")
    direct_static_checkpoint = artifacts.direct_static_target_checkpoint or str(
        direct_stage3_dir / "policy_latest.json"
    )
    direct_final_checkpoint = artifacts.direct_stage3_v2_checkpoint or str(direct_stage3_dir / "policy_latest.json")
    direct_static_metrics = artifacts.direct_static_target_metrics or str(
        direct_stage3_dir / "evaluate_static" / "evaluate_report.json"
    )
    direct_stage3_metrics = artifacts.direct_stage3_v2_metrics or str(
        direct_stage3_dir / "evaluate" / "evaluate_report.json"
    )
    paired_stage3_metrics = artifacts.stage3_paired_metrics or str(root / "stage3_paired" / "paired_comparison.json")
    stage3_task_causal_metrics = artifacts.stage3_task_causal_metrics or str(
        root / "stage3_task_causal" / "promotion_metrics.json"
    )
    stage3_signal_npz = artifacts.stage3_signal_npz or str(root / "stage3_signal" / "simulation_signals.npz")
    stage3_signal_sidecar = artifacts.stage3_signal_sidecar_json or str(
        root / "stage3_signal" / "simulation_signals.manifest.json"
    )
    target_bank = artifacts.recovery_target_bank or str(root / "targets" / "targets_train_v2.json")
    eval_target_bank = artifacts.recovery_eval_target_bank or str(root / "targets" / "targets_eval_v2.json")
    train_feed_bank = artifacts.recovery_train_feed_bank or "<required:recovery_train_feed_bank>"
    eval_feed_bank = artifacts.recovery_eval_feed_bank or "<required:recovery_eval_feed_bank>"
    grouping = artifacts.synergy_grouping or str(REPO_ROOT / spec.synergy_grouping)
    train_event_reference_bank = artifacts.train_event_reference_bank or "<required:train_event_reference_bank>"
    val_event_reference_bank = artifacts.val_event_reference_bank or "<required:val_event_reference_bank>"
    teacher_promotion = artifacts.racket_mass_100_promotion_manifest or str(
        root / "racket_mass_v2" / "mass_100_promotion_manifest.json"
    )
    teacher_fingerprint = artifacts.racket_mass_100_checkpoint_fingerprint or (
        "<required:racket_mass_100_checkpoint_fingerprint>"
    )
    event_reference_metrics = artifacts.event_reference_metrics or str(
        root / "event_reference" / "promotion_metrics.json"
    )
    train_reference_manifests = (
        artifacts.train_event_reference_manifest_list or "<required:train_event_reference_manifest_list>"
    )
    val_reference_manifests = (
        artifacts.val_event_reference_manifest_list or "<required:val_event_reference_manifest_list>"
    )
    train_motions = spec.train_motion_paths
    val_motions = spec.val_motion_paths
    sweep_base_config = spec.require("latent_synergy_config")
    sweep_base_config_flags: tuple[str, ...] = (
        () if sweep_base_config == _SWEEP_DEFAULT_BASE_CONFIG else ("--base-config", sweep_base_config)
    )

    # PEASD arm selection.  With no reference manifest the sweep command stays
    # byte-identical to the historical EMG-free baseline, so the arms differ by
    # exactly these flags and nothing else.  The arms map to the doc §26.2
    # matrix: S2-B baseline (no flags), S2-C privileged, S2-D shuffled control,
    # S2-E privileged without context dropout.
    if artifacts.emg_reference_manifest is None:
        if artifacts.emg_synergy_dim is not None:
            raise ValueError("emg_synergy_dim requires emg_reference_manifest for the privileged latent arm")
        if artifacts.emg_shuffle_context_ablation:
            raise ValueError("emg_shuffle_context_ablation is a control for the privileged arm; enable that arm first")
        if artifacts.emg_no_context_dropout:
            raise ValueError("emg_no_context_dropout requires emg_reference_manifest for the privileged latent arm")
        emg_privileged_flags: tuple[str, ...] = ()
    else:
        if artifacts.emg_synergy_dim is None or int(artifacts.emg_synergy_dim) <= 0:
            raise ValueError("privileged latent arm requires a positive emg_synergy_dim")
        emg_privileged_flags = (
            "--emg-privileged-enabled",
            "--emg-synergy-dim",
            str(int(artifacts.emg_synergy_dim)),
            "--emg-reference-manifest",
            str(artifacts.emg_reference_manifest),
        )
        if artifacts.emg_shuffle_context_ablation:
            emg_privileged_flags += ("--emg-shuffle-context-ablation",)
        if artifacts.emg_no_context_dropout:
            emg_privileged_flags += ("--emg-context-dropout", "0.0")

    # §26.3 H3: grouped right-arm correction on the frozen latent/decoder.
    stage3_residual_flag: tuple[str, ...] = ()
    if artifacts.stage3_bounded_residual_groups_json:
        stage3_residual_flag = (
            "--bounded-residual-groups-json",
            str(artifacts.stage3_bounded_residual_groups_json),
        )

    def gate(stage: str, metrics: str | None, name: str) -> PipelineStep:
        return PipelineStep(
            name,
            _gate_command(python, stage, metrics, root / f"{name}.json"),
        )

    emg_requested = any(
        (
            artifacts.emg_simulation_npz,
            artifacts.emg_measurement_npz,
            artifacts.emg_mapping_json,
        )
    )
    if emg_requested and not all((artifacts.emg_measurement_npz, artifacts.emg_mapping_json)):
        raise ValueError("EMG validation requires both measurement NPZ and channel mapping JSON")
    emg_simulation_npz = artifacts.emg_simulation_npz or stage3_signal_npz
    emg_command = (
        (
            python,
            "-m",
            "musclemimic.evaluation.emg_eval",
            "--simulation-npz",
            str(emg_simulation_npz),
            "--emg-npz",
            str(artifacts.emg_measurement_npz),
            "--mapping-json",
            str(artifacts.emg_mapping_json),
            "--policy-evidence-json",
            paired_stage3_metrics,
            "--output-json",
            artifacts.emg_metrics or str(root / "emg" / "report.json"),
        )
        if emg_requested
        else (python, "-m", "musclemimic.evaluation.emg_eval", "--dry-run")
    )
    physiology_requested = any((artifacts.physiology_input_npz, artifacts.physiology_config_json))
    if physiology_requested and artifacts.physiology_config_json is None:
        raise ValueError("physiology validation requires an evaluation config JSON")
    physiology_input_npz = artifacts.physiology_input_npz or stage3_signal_npz
    physiology_command = (
        (
            python,
            "-m",
            "musclemimic.evaluation.physiology",
            "--input-npz",
            str(physiology_input_npz),
            "--evaluation-config-json",
            str(artifacts.physiology_config_json),
            "--policy-evidence-json",
            paired_stage3_metrics,
            "--signal-identity-json",
            artifacts.stage3_signal_identity_json or "<required:stage3_signal_identity_json>",
            "--output-json",
            artifacts.physiology_metrics or str(root / "physiology" / "report.json"),
        )
        if physiology_requested
        else (python, "-m", "musclemimic.evaluation.physiology", "--dry-run")
    )

    mass_steps: list[PipelineStep] = []
    previous_checkpoint_value = artifacts.stage1r005_checkpoint or ("<required:stage1r005_checkpoint>")
    previous_checkpoint_field = "stage1r005_checkpoint"
    previous_metrics_value = artifacts.stage1r005_metrics or ("<required:stage1r005_metrics>")
    previous_metrics_field = "stage1r005_metrics"
    previous_promotion_value: str | None = None
    for scale in ("025", "050", "075", "100"):
        checkpoint_field = f"racket_mass_{scale}_checkpoint"
        metrics_field = f"racket_mass_{scale}_metrics"
        visual_review_field = f"racket_mass_{scale}_visual_review"
        physics_manifest_field = f"racket_mass_{scale}_physics_manifest"
        promotion_field = f"racket_mass_{scale}_promotion_manifest"
        checkpoint_value = getattr(artifacts, checkpoint_field) or (f"<required:{checkpoint_field}>")
        metrics_value = getattr(artifacts, metrics_field) or (f"<required:{metrics_field}>")
        visual_review_value = getattr(artifacts, visual_review_field) or (f"<required:{visual_review_field}>")
        promotion_value = getattr(artifacts, promotion_field) or str(
            root / "racket_mass_v2" / f"mass_{scale}_promotion_manifest.json"
        )
        physics_manifest_value = getattr(artifacts, physics_manifest_field) or str(
            root / "racket_mass_v2" / f"mass_{scale}_physics_manifest.json"
        )
        mass_steps.append(
            PipelineStep(
                f"racket_mass_{scale}_physics",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.racket_mass_curriculum",
                    "--physics-stage",
                    f"mass_{scale}",
                    "--output",
                    physics_manifest_value,
                ),
            )
        )
        mass_steps.append(
            PipelineStep(
                f"racket_mass_{scale}_train",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.racket_mass_curriculum",
                    "--launch-stage",
                    f"mass_{scale}",
                    "--physics-manifest",
                    physics_manifest_value,
                    "--resume-from",
                    previous_checkpoint_value,
                    "--baseline-metrics",
                    previous_metrics_value,
                    "--train-event-bank",
                    train_event_reference_bank,
                    "--val-event-bank",
                    val_event_reference_bank,
                ),
                (
                    previous_checkpoint_field,
                    previous_metrics_field,
                    "train_event_reference_bank",
                    "val_event_reference_bank",
                ),
                (
                    (
                        "RACKET_MASS_PARENT_PROMOTION",
                        previous_promotion_value or artifacts.stage1r005_metrics or "<required:stage1r005_metrics>",
                    ),
                    ("RACKET_MASS_PHYSICS_MANIFEST", physics_manifest_value),
                ),
            )
        )
        mass_steps.append(
            PipelineStep(
                f"racket_mass_{scale}_gate",
                _gate_command(
                    python,
                    "stage2",
                    metrics_value,
                    root / "racket_mass_v2" / f"mass_{scale}_gate.json",
                    consecutive=1,
                    baseline_metrics=previous_metrics_value,
                    checkpoint=checkpoint_value,
                ),
                (checkpoint_field, metrics_field, previous_metrics_field),
            )
        )
        mass_steps.append(
            PipelineStep(
                f"racket_mass_{scale}_visual_gate",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.visual_review",
                    "--review",
                    visual_review_value,
                    "--required_clips",
                    "5",
                    "--review-kind",
                    STAGE2_REVIEW_KIND,
                    "--checkpoint",
                    checkpoint_value,
                    "--expected_motion",
                    *spec.val_motions,
                    "--output",
                    str(root / "racket_mass_v2" / f"mass_{scale}_visual_gate.json"),
                    "--require_pass",
                ),
                (checkpoint_field, metrics_field, visual_review_field),
            )
        )
        mass_steps.append(
            PipelineStep(
                f"racket_mass_{scale}_promote",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.racket_mass_curriculum",
                    "--promote-stage",
                    f"mass_{scale}",
                    "--checkpoint",
                    checkpoint_value,
                    "--promotion-progress",
                    metrics_value,
                    "--visual-review",
                    visual_review_value,
                    "--physics-manifest",
                    physics_manifest_value,
                    *(
                        (
                            "--parent-checkpoint",
                            previous_checkpoint_value,
                            "--parent-stage1r-report",
                            previous_metrics_value,
                        )
                        if scale == "025"
                        else (
                            "--parent-mass-promotion",
                            str(previous_promotion_value),
                        )
                    ),
                    "--output",
                    promotion_value,
                ),
                (checkpoint_field, metrics_field, visual_review_field),
            )
        )
        previous_checkpoint_value = checkpoint_value
        previous_checkpoint_field = checkpoint_field
        previous_metrics_value = metrics_value
        previous_metrics_field = metrics_field
        previous_promotion_value = promotion_value
    collect_common = (
        python,
        "-m",
        "fullbody.distill_collect",
        "--teacher_ckpt",
        teacher,
        "--num_transitions",
        "1000000",
        "--save-physical-muscle-state",
        "--save-event-features",
        "--save_reference_features",
        "--teacher-promotion-manifest",
        teacher_promotion,
        "--physical-racket-site-name",
        "racket_stringbed_center_site",
    )
    steps = (
        PipelineStep(
            "racket_mass_curriculum_plan",
            (
                python,
                "-m",
                "musclemimic.badminton.racket_mass_curriculum",
                "--output",
                str(root / "racket_mass_v2" / "curriculum_plan.json"),
            ),
        ),
        PipelineStep(
            "event_reference_qc",
            (
                python,
                "-m",
                "musclemimic.badminton.data.event_qc",
                "--train-manifests-json",
                train_reference_manifests,
                "--val-manifests-json",
                val_reference_manifests,
                "--train-event-bank",
                train_event_reference_bank,
                "--val-event-bank",
                val_event_reference_bank,
                "--output",
                event_reference_metrics,
            ),
            (
                "train_event_reference_manifest_list",
                "val_event_reference_manifest_list",
                "train_event_reference_bank",
                "val_event_reference_bank",
            ),
        ),
        gate("event_reference_v2", event_reference_metrics, "event_reference_gate"),
        *tuple(mass_steps),
        PipelineStep(
            "physical_rollout_collect",
            (
                *collect_common,
                "--event-reference-bank",
                train_event_reference_bank,
                "--motion_path",
                *train_motions,
                "--output_dir",
                str(physical_train),
                "--split",
                "train",
            ),
            (
                "racket_mass_100_checkpoint",
                "train_event_reference_bank",
                "racket_mass_100_checkpoint_fingerprint",
            ),
        ),
        PipelineStep(
            "physical_rollout_collect_val",
            (
                *collect_common,
                "--event-reference-bank",
                val_event_reference_bank,
                "--motion_path",
                *val_motions,
                "--output_dir",
                str(physical_val),
                "--split",
                "val",
            ),
            (
                "racket_mass_100_checkpoint",
                "val_event_reference_bank",
                "racket_mass_100_checkpoint_fingerprint",
            ),
        ),
        PipelineStep(
            "physical_rollout_qc",
            (
                python,
                "-m",
                "musclemimic.distill.physical_qc",
                "--train",
                str(physical_train),
                "--val",
                str(physical_val),
                "--output",
                physical_metrics,
                "--teacher-checkpoint-fingerprint",
                teacher_fingerprint,
                "--event-reference-metrics",
                event_reference_metrics,
            ),
            (
                "racket_mass_100_checkpoint_fingerprint",
                "event_reference_metrics",
            ),
        ),
        gate("physical_rollout_v2", physical_metrics, "physical_rollout_gate"),
        PipelineStep(
            "direct_baseline_train",
            (
                python,
                "-m",
                "fullbody.distill_train_bc",
                "--dataset_dir",
                str(physical_train),
                "--student_config",
                spec.require("student_bc_config"),
                "--output_dir",
                str(direct_dir / "bc"),
                "--batch_size",
                "4096",
                "--num_steps",
                "200000",
                "--lr",
                "0.0003",
                "--seed",
                "0",
                "--require_dataset_manifest",
            ),
        ),
        PipelineStep(
            "direct_baseline_evaluate",
            (
                python,
                "-m",
                "fullbody.distill_compare",
                "--teacher_ckpt",
                teacher,
                "--student_ckpt",
                str(direct_checkpoint),
                "--output_dir",
                str(direct_dir / "compare"),
                "--dataset_dir",
                str(physical_val),
                "--convergence_metrics",
                direct_bc_metrics,
                "--motion_path",
                *val_motions,
                "--metrics_envs",
                "20",
                "--metrics_steps",
                "500",
                "--eval_seed",
                "0",
                "--deterministic",
                "--promotion_policy",
                "student_bc",
                "--require_pass",
            ),
            (
                "racket_mass_100_checkpoint",
                "racket_mass_100_promotion_manifest",
            ),
        ),
        PipelineStep(
            "synergy_fit",
            (
                python,
                "-m",
                "musclemimic.synergy.fit",
                "--train",
                str(physical_train),
                "--val",
                str(physical_val),
                "--output-dir",
                str(synergy_dir),
                "--mode",
                "both",
                "--grouping-json",
                grouping,
            ),
        ),
        gate("synergy_v2", synergy_metrics, "synergy_gate"),
        PipelineStep(
            "latent_dimension_sweep",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "plan",
                "--dataset-dir",
                str(physical_train),
                "--val-dataset-dir",
                str(physical_val),
                "--teacher-ckpt",
                teacher,
                "--teacher-promotion-manifest",
                teacher_promotion,
                "--direct-bc-metrics",
                direct_bc_metrics,
                "--direct-rollout-metrics",
                direct_rollout_metrics,
                "--direct-promotion-evidence",
                direct_acceptance,
                "--synergy-basis",
                basis,
                "--synergy-basis-fingerprint",
                basis_fingerprint,
                "--frozen-body-decoder",
                frozen_body_decoder,
                "--frozen-body-decoder-fingerprint",
                frozen_body_decoder_fingerprint,
                "--body-synergy-contract-fingerprint",
                body_synergy_contract_fingerprint,
                "--body-synergy-portable-core-fingerprint",
                body_synergy_portable_core_fingerprint,
                "--output-dir",
                str(latent_dir),
                "--dimensions",
                "2",
                "4",
                "8",
                "16",
                "32",
                "--seeds",
                "0",
                "1",
                "2",
                "--require-causal-interventions",
                *emg_privileged_flags,
                *sweep_base_config_flags,
            ),
            (
                "synergy_basis",
                "synergy_basis_fingerprint",
                "frozen_body_decoder",
                "frozen_body_decoder_fingerprint",
                "body_synergy_contract_fingerprint",
                "body_synergy_portable_core_fingerprint",
                "racket_mass_100_checkpoint",
                "racket_mass_100_promotion_manifest",
                "racket_mass_100_checkpoint_fingerprint",
            ),
        ),
        PipelineStep(
            "latent_dimension_execute",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "execute",
                "--output-dir",
                str(latent_dir),
                "--stage",
                "full",
            ),
        ),
        PipelineStep(
            "latent_causal_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "causal-evaluate",
                "--output-dir",
                str(latent_dir),
                "--shared-config",
                latent_causal_adapter_config,
            ),
            ("latent_causal_adapter_config",),
        ),
        PipelineStep(
            "latent_causal_finalize",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "finalize-causal",
                "--output-dir",
                str(latent_dir),
            ),
        ),
        PipelineStep(
            "latent_synergy_analysis",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "analyze",
                "--output-dir",
                str(latent_dir),
                "--require-all-phases",
                "--require-causal-interventions",
            ),
        ),
        gate("latent_synergy_v2", latent_synergy_metrics, "latent_synergy_gate"),
        PipelineStep(
            "recovery_target",
            (
                python,
                "-m",
                "environment.overall_environment.src.stage3_target_bank_v2",
                "--input-jsonl",
                artifacts.desired_impact_targets_jsonl or "<required:desired_impact_targets_jsonl>",
                "--event-reference-metrics",
                event_reference_metrics,
                "--reference-split",
                "train",
                "--feed-bank-path",
                train_feed_bank,
                "--consumer-order",
                "difficulty_sorted",
                "--output",
                target_bank,
            ),
            (
                "desired_impact_targets_jsonl",
                "recovery_train_feed_bank",
                "event_reference_metrics",
            ),
        ),
        PipelineStep(
            "recovery_target_eval",
            (
                python,
                "-m",
                "environment.overall_environment.src.stage3_target_bank_v2",
                "--input-jsonl",
                artifacts.desired_impact_targets_eval_jsonl or "<required:desired_impact_targets_eval_jsonl>",
                "--event-reference-metrics",
                event_reference_metrics,
                "--reference-split",
                "validation",
                "--feed-bank-path",
                eval_feed_bank,
                "--consumer-order",
                "stored",
                "--output",
                eval_target_bank,
            ),
            (
                "desired_impact_targets_eval_jsonl",
                "recovery_eval_feed_bank",
                "event_reference_metrics",
            ),
        ),
        PipelineStep(
            "stage3_v2_preflight",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "preflight",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(stage3_dir),
            ),
            ("recovery_target_bank", "recovery_eval_target_bank"),
        ),
        PipelineStep(
            "stage3_v2_feed_check",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "feed-check",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(stage3_dir),
            ),
            ("recovery_target_bank", "recovery_eval_target_bank"),
        ),
        PipelineStep(
            "stage3_v2_base_only",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "base-only-check",
                "--latent-checkpoint",
                latent_checkpoint,
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(stage3_dir),
                *stage3_residual_flag,
            ),
            (
                "latent_synergy_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        PipelineStep(
            "stage3_static_target_train",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "train-gpu",
                "--latent-checkpoint",
                latent_checkpoint,
                "--total-env-steps",
                "6000000",
                "--curriculum-max-stage",
                "C3_static_velocity",
                "--seed",
                "0",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(static_dir),
                *stage3_residual_flag,
            ),
            (
                "latent_synergy_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        PipelineStep(
            "stage3_static_target_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "evaluate",
                "--checkpoint",
                static_checkpoint,
                "--episodes",
                "128",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(static_dir / "evaluate_static"),
            ),
            (
                "static_target_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        gate("static_target_v2", static_metrics, "stage3_static_target_gate"),
        PipelineStep(
            "stage3_v2_train",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "train-gpu",
                "--latent-checkpoint",
                latent_checkpoint,
                "--total-env-steps",
                "30000000",
                "--curriculum-max-stage",
                "C7_recovery",
                "--resume-from",
                static_checkpoint,
                "--seed",
                "0",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(stage3_dir),
                *stage3_residual_flag,
            ),
            (
                "latent_synergy_checkpoint",
                "static_target_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        PipelineStep(
            "stage3_v2_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "evaluate",
                "--checkpoint",
                artifacts.stage3_v2_checkpoint or str(stage3_dir / "policy_latest.json"),
                "--episodes",
                "128",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(stage3_dir / "evaluate"),
                *stage3_residual_flag,
            ),
            (
                "stage3_v2_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        gate("stage3_v2", stage3_metrics, "stage3_v2_gate"),
        PipelineStep(
            "stage3_task_causal_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.stage3_task_causal",
                "--config",
                artifacts.stage3_task_causal_config or "<required:stage3_task_causal_config>",
            ),
            (
                "stage3_task_causal_config",
                "stage3_v2_metrics",
                "latent_selection_manifest",
            ),
        ),
        gate(
            "latent_task_causal_v2",
            stage3_task_causal_metrics,
            "stage3_task_causal_gate",
        ),
        PipelineStep(
            "direct_stage3_v2_preflight",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "preflight",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir),
            ),
            ("recovery_target_bank", "recovery_eval_target_bank"),
        ),
        PipelineStep(
            "direct_stage3_v2_feed_check",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "feed-check",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir),
            ),
            ("recovery_target_bank", "recovery_eval_target_bank"),
        ),
        PipelineStep(
            "direct_stage3_static_target_train",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "train-gpu",
                "--total-env-steps",
                "6000000",
                "--curriculum-max-stage",
                "C3_static_velocity",
                "--seed",
                "0",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir),
            ),
            (
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        PipelineStep(
            "direct_stage3_static_target_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "evaluate",
                "--checkpoint",
                direct_static_checkpoint,
                "--episodes",
                "128",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir / "evaluate_static"),
            ),
            (
                "direct_static_target_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        gate(
            "static_target_v2",
            direct_static_metrics,
            "direct_stage3_static_target_gate",
        ),
        PipelineStep(
            "direct_stage3_v2_train",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "train-gpu",
                "--total-env-steps",
                "30000000",
                "--curriculum-max-stage",
                "C7_recovery",
                "--resume-from",
                direct_static_checkpoint,
                "--seed",
                "0",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir),
            ),
            (
                "direct_static_target_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        PipelineStep(
            "direct_stage3_v2_evaluate",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_direct_spec,
                "--stage",
                "evaluate",
                "--checkpoint",
                direct_final_checkpoint,
                "--episodes",
                "128",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(direct_stage3_dir / "evaluate"),
            ),
            (
                "direct_stage3_v2_checkpoint",
                "recovery_target_bank",
                "recovery_eval_target_bank",
            ),
        ),
        gate("stage3_v2", direct_stage3_metrics, "direct_stage3_v2_gate"),
        PipelineStep(
            "stage3_paired_comparison",
            (
                python,
                "-m",
                "musclemimic.badminton.stage3_paired_comparison",
                "--direct-report",
                direct_stage3_metrics,
                "--synergy-report",
                stage3_metrics,
                "--selection-manifest",
                artifacts.latent_selection_manifest or str(latent_dir / "selected" / "selection_manifest.json"),
                "--output",
                paired_stage3_metrics,
            ),
            (
                "direct_stage3_v2_metrics",
                "stage3_v2_metrics",
            ),
        ),
        PipelineStep(
            "stage3_signal_export",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                "--spec",
                stage3_v2_spec,
                "--stage",
                "evaluate",
                "--checkpoint",
                artifacts.stage3_v2_checkpoint or str(stage3_dir / "policy_latest.json"),
                "--episodes",
                "128",
                "--target-bank",
                target_bank,
                "--eval-target-bank",
                eval_target_bank,
                "--out-dir",
                str(root / "stage3_signal" / "evaluate"),
                "--export-simulation-npz",
                stage3_signal_npz,
                "--signal-identity-json",
                artifacts.stage3_signal_identity_json or "<required:stage3_signal_identity_json>",
                "--policy-evidence-json",
                paired_stage3_metrics,
                "--signal-sidecar-json",
                stage3_signal_sidecar,
            ),
            (
                "stage3_v2_checkpoint",
                "stage3_signal_identity_json",
                "recovery_train_feed_bank",
                "recovery_eval_feed_bank",
            ),
        ),
        PipelineStep(
            "emg_validation",
            emg_command,
            (
                (
                    "emg_measurement_npz",
                    "emg_mapping_json",
                    "stage3_paired_metrics",
                    *(("emg_simulation_npz",) if artifacts.emg_simulation_npz else ()),
                )
                if emg_requested
                else ()
            ),
        ),
        PipelineStep(
            "physiology_validation",
            physiology_command,
            (
                (
                    "physiology_config_json",
                    "stage3_paired_metrics",
                    "stage3_signal_identity_json",
                    *(("physiology_input_npz",) if artifacts.physiology_input_npz else ()),
                )
                if physiology_requested
                else ()
            ),
        ),
        PipelineStep(
            "ablation_report",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.build_forehand_clear_ablation_report",
                "--input-jsonl",
                artifacts.ablation_jsonl or "<required:ablation_jsonl>",
                "--output-dir",
                str(root / "ablation"),
            ),
            ("ablation_jsonl",),
        ),
    )
    # Every Stage-3 v2 consumer must load the exact feed banks used to build
    # its ordered target banks.  Central insertion keeps future direct/synergy
    # runner steps symmetric and avoids depending on paths inside a YAML file.
    runner_module = "musclemimic.badminton.scripts.run_incoming_shuttle_hit"
    bound_steps: list[PipelineStep] = []
    for step in steps:
        if runner_module not in step.command:
            bound_steps.append(step)
            continue
        command = list(step.command)
        if "--feed-bank" in command or "--eval-feed-bank" in command:
            raise ValueError(f"Stage-3 feed override duplicated in pipeline step {step.name}")
        try:
            insertion = command.index("--target-bank")
        except ValueError as exc:
            raise ValueError(f"Stage-3 runner step {step.name} has no target-bank binding") from exc
        command[insertion:insertion] = [
            "--feed-bank",
            train_feed_bank,
            "--eval-feed-bank",
            eval_feed_bank,
        ]
        required_artifacts = tuple(
            dict.fromkeys(
                (
                    *step.required_artifacts,
                    "recovery_train_feed_bank",
                    "recovery_eval_feed_bank",
                )
            )
        )
        bound_steps.append(
            PipelineStep(
                name=step.name,
                command=tuple(command),
                required_artifacts=required_artifacts,
                environment=step.environment,
            )
        )
    return tuple(bound_steps)


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
    profile: str = "legacy_v2",
    spec: ActionSpec = DEFAULT_SPEC,
) -> None:
    steps = {step.name: step for step in build_pipeline_plan(output_dir, artifacts, profile=profile, spec=spec)}
    if step_name not in steps:
        raise ValueError(f"unknown pipeline step {step_name!r}; expected one of {sorted(steps)}")
    step = steps[step_name]
    missing = [name for name in step.required_artifacts if not getattr(artifacts, name)]
    if missing:
        raise ValueError(f"pipeline step {step_name} is missing required artifacts: {missing}")
    _verify_upstream_gates(step_name, artifacts, output_dir=output_dir, spec=spec)
    env = os.environ.copy()
    env.update(dict(step.environment))
    subprocess.run(list(step.command), cwd=REPO_ROOT, env=env, check=True)


def _verify_upstream_gates(
    step_name: str,
    artifacts: PipelineArtifacts,
    *,
    output_dir: str | Path,
    spec: ActionSpec = DEFAULT_SPEC,
) -> None:
    out = Path(output_dir)
    v3 = out / "synergy_v3"
    mass_root = v3 / "racket_mass_v2"
    mass_scales = ("025", "050", "075", "100")
    for index, scale in enumerate(mass_scales):
        prefix = f"racket_mass_{scale}"
        if step_name == f"{prefix}_train":
            from musclemimic.badminton.racket_mass_curriculum import (
                validate_mass_promoted_artifact,
                validate_racket_physics_manifest,
            )

            _require_passed_report(
                v3 / "event_reference_gate.json",
                label="event-reference v2 gate",
                expected_metrics=(
                    artifacts.event_reference_metrics or v3 / "event_reference" / "promotion_metrics.json"
                ),
            )
            physics_path = getattr(artifacts, f"racket_mass_{scale}_physics_manifest") or str(
                mass_root / f"mass_{scale}_physics_manifest.json"
            )
            validate_racket_physics_manifest(
                physics_path,
                expected_stage=f"mass_{scale}",
                verify_compiled_model=True,
            )
            if index == 0:
                from musclemimic.badminton.stage1r_artifact import (
                    validate_stage1r_report,
                )

                if artifacts.stage1r005_checkpoint is None:
                    raise ValueError("mass_025 requires the Stage-1R 0% checkpoint")
                if artifacts.stage1r005_metrics is None:
                    raise ValueError("mass_025 requires the Stage-1R 0% report")
                validate_stage1r_report(
                    artifacts.stage1r005_metrics,
                    expected_checkpoint=artifacts.stage1r005_checkpoint,
                    expected_perturb_qpos_scale=0.05,
                )
            else:
                parent_scale = mass_scales[index - 1]
                parent_checkpoint = getattr(artifacts, f"racket_mass_{parent_scale}_checkpoint")
                parent_promotion = getattr(artifacts, f"racket_mass_{parent_scale}_promotion_manifest") or str(
                    mass_root / f"mass_{parent_scale}_promotion_manifest.json"
                )
                if parent_checkpoint is None:
                    raise ValueError(f"racket_mass_{parent_scale} promoted checkpoint is required")
                validate_mass_promoted_artifact(
                    parent_promotion,
                    expected_stage=f"mass_{parent_scale}",
                    expected_checkpoint=parent_checkpoint,
                )
        if step_name in {f"{prefix}_visual_gate", f"{prefix}_promote"}:
            _require_passed_report(
                mass_root / f"mass_{scale}_gate.json",
                label=f"racket-mass {scale} numerical gate",
                expected_metrics=getattr(artifacts, f"racket_mass_{scale}_metrics"),
            )
        if step_name == f"{prefix}_promote":
            _require_passed_report(
                mass_root / f"mass_{scale}_visual_gate.json",
                label=f"racket-mass {scale} visual gate",
            )
    if step_name in {
        "physical_rollout_collect",
        "physical_rollout_collect_val",
        "physical_rollout_qc",
    }:
        _require_passed_report(
            v3 / "event_reference_gate.json",
            label="event-reference v2 gate",
            expected_metrics=(artifacts.event_reference_metrics or v3 / "event_reference" / "promotion_metrics.json"),
        )
        from musclemimic.badminton.racket_mass_curriculum import (
            validate_mass_promoted_artifact,
        )

        if artifacts.racket_mass_100_checkpoint is None:
            raise ValueError("physical rollout requires the promoted 100% racket checkpoint")
        validate_mass_promoted_artifact(
            artifacts.racket_mass_100_promotion_manifest or str(mass_root / "mass_100_promotion_manifest.json"),
            expected_stage="mass_100",
            expected_checkpoint=artifacts.racket_mass_100_checkpoint,
        )
    if step_name in {"synergy_fit", "synergy_gate"}:
        _require_passed_report(
            v3 / "physical_rollout_gate.json",
            label="physical rollout v2 gate",
            expected_metrics=(artifacts.physical_rollout_metrics or v3 / "physical_rollout" / "promotion_metrics.json"),
        )
    if step_name in {"direct_baseline_train", "direct_baseline_evaluate"}:
        _require_passed_report(
            v3 / "physical_rollout_gate.json",
            label="physical rollout v2 gate",
            expected_metrics=(artifacts.physical_rollout_metrics or v3 / "physical_rollout" / "promotion_metrics.json"),
        )
    if step_name == "direct_baseline_evaluate":
        direct_bc = Path(artifacts.direct_bc_metrics or v3 / "direct_baseline" / "bc" / "distill_metadata.json")
        direct_checkpoint = v3 / "direct_baseline" / "bc" / "checkpoints" / "checkpoint_200000"
        if not direct_bc.is_file() or not direct_checkpoint.is_dir():
            raise ValueError("direct baseline BC checkpoint/metrics are incomplete")
    if step_name in {
        "latent_dimension_sweep",
        "latent_dimension_execute",
        "latent_causal_evaluate",
        "latent_causal_finalize",
        "latent_synergy_analysis",
        "latent_synergy_gate",
    }:
        _require_passed_report(
            v3 / "synergy_gate.json",
            label="synergy v2 gate",
            expected_metrics=(artifacts.synergy_metrics or v3 / "synergy" / "promotion_metrics.json"),
        )
        _require_v3_direct_baseline(artifacts, v3=v3, spec=spec)
    if step_name in {
        "latent_dimension_execute",
        "latent_causal_evaluate",
        "latent_causal_finalize",
        "latent_synergy_analysis",
    }:
        sweep_plan = v3 / "latent_synergy" / "sweep_plan.json"
        if not sweep_plan.is_file():
            raise ValueError(f"latent-synergy sweep plan is missing: {sweep_plan}")
    stage3_synergy_steps = {
        "stage3_v2_preflight",
        "stage3_v2_feed_check",
        "stage3_v2_base_only",
        "stage3_static_target_train",
        "stage3_static_target_evaluate",
        "stage3_static_target_gate",
        "stage3_v2_train",
        "stage3_v2_evaluate",
        "stage3_v2_gate",
    }
    stage3_direct_steps = {
        "direct_stage3_v2_preflight",
        "direct_stage3_v2_feed_check",
        "direct_stage3_static_target_train",
        "direct_stage3_static_target_evaluate",
        "direct_stage3_static_target_gate",
        "direct_stage3_v2_train",
        "direct_stage3_v2_evaluate",
        "direct_stage3_v2_gate",
    }
    if step_name in {
        *stage3_synergy_steps,
        "stage3_paired_comparison",
        "stage3_task_causal_evaluate",
        "stage3_task_causal_gate",
        "stage3_signal_export",
    }:
        _require_passed_report(
            v3 / "latent_synergy_gate.json",
            label="latent-synergy v2 gate",
            expected_metrics=(artifacts.latent_synergy_metrics or v3 / "latent_synergy" / "promotion_metrics.json"),
        )
        _require_latent_selection_binding(artifacts, v3=v3)
    if step_name in {
        *stage3_synergy_steps,
        *stage3_direct_steps,
        "stage3_paired_comparison",
        "stage3_task_causal_evaluate",
        "stage3_task_causal_gate",
        "stage3_signal_export",
    }:
        event_metrics = Path(artifacts.event_reference_metrics or v3 / "event_reference" / "promotion_metrics.json")
        _require_target_event_binding(
            artifacts.recovery_target_bank or v3 / "targets" / "targets_train_v2.json",
            event_metrics=event_metrics,
            split="train",
        )
        _require_target_event_binding(
            artifacts.recovery_eval_target_bank or v3 / "targets" / "targets_eval_v2.json",
            event_metrics=event_metrics,
            split="validation",
        )
    if step_name in {
        "stage3_static_target_train",
        "direct_stage3_static_target_train",
    }:
        direct = step_name.startswith("direct_")
        branch = "stage3_impact_recovery_direct" if direct else "stage3_impact_recovery"
        prerequisite_reports = [
            ("preflight_report.json", "Stage-3 v2 preflight"),
            ("feed_check_report.json", "Stage-3 v2 feed check"),
        ]
        if not direct:
            prerequisite_reports.append(("base_only_report.json", "Stage-3 v2 base-only check"))
        for filename, label in prerequisite_reports:
            _require_passed_report(
                v3 / branch / filename,
                label=label,
            )
    if step_name in {
        "stage3_static_target_evaluate",
        "stage3_static_target_gate",
        "direct_stage3_static_target_evaluate",
        "direct_stage3_static_target_gate",
    }:
        branch = "stage3_impact_recovery_direct" if step_name.startswith("direct_") else "stage3_impact_recovery"
        _require_stage3_task_curriculum_complete(
            v3 / branch / "train_report.json",
            expected_max_stage="C3_static_velocity",
        )
    if step_name in {"stage3_v2_train", "direct_stage3_v2_train"}:
        direct = step_name.startswith("direct_")
        _require_passed_report(
            v3 / ("direct_stage3_static_target_gate.json" if direct else "stage3_static_target_gate.json"),
            label="static-target v2 gate",
            expected_metrics=(
                (
                    artifacts.direct_static_target_metrics
                    or v3 / "stage3_impact_recovery_direct" / "evaluate_static" / "evaluate_report.json"
                )
                if direct
                else (
                    artifacts.static_target_metrics
                    or v3 / "stage3_impact_recovery" / "evaluate_static" / "evaluate_report.json"
                )
            ),
        )
    if step_name in {
        "stage3_v2_evaluate",
        "stage3_v2_gate",
        "direct_stage3_v2_evaluate",
        "direct_stage3_v2_gate",
    }:
        branch = "stage3_impact_recovery_direct" if step_name.startswith("direct_") else "stage3_impact_recovery"
        _require_stage3_curriculum_complete(v3 / branch / "train_report.json")
    if step_name in {"stage3_v2_gate", "direct_stage3_v2_gate"}:
        direct = step_name.startswith("direct_")
        _require_stage3_artifact_binding(
            Path(
                (
                    artifacts.direct_stage3_v2_metrics
                    or v3 / "stage3_impact_recovery_direct" / "evaluate" / "evaluate_report.json"
                )
                if direct
                else (
                    artifacts.stage3_v2_metrics or v3 / "stage3_impact_recovery" / "evaluate" / "evaluate_report.json"
                )
            )
        )
    if step_name == "stage3_paired_comparison":
        direct_metrics = Path(
            artifacts.direct_stage3_v2_metrics
            or v3 / "stage3_impact_recovery_direct" / "evaluate" / "evaluate_report.json"
        )
        synergy_metrics = Path(
            artifacts.stage3_v2_metrics or v3 / "stage3_impact_recovery" / "evaluate" / "evaluate_report.json"
        )
        _require_passed_report(
            v3 / "direct_stage3_v2_gate.json",
            label="direct Stage-3 v2 gate",
            expected_metrics=direct_metrics,
        )
        _require_passed_report(
            v3 / "stage3_v2_gate.json",
            label="synergy Stage-3 v2 gate",
            expected_metrics=synergy_metrics,
        )
        _require_stage3_artifact_binding(direct_metrics)
        _require_stage3_artifact_binding(synergy_metrics)
    if step_name in {
        "stage3_task_causal_evaluate",
        "stage3_task_causal_gate",
    }:
        synergy_metrics = Path(
            artifacts.stage3_v2_metrics or v3 / "stage3_impact_recovery" / "evaluate" / "evaluate_report.json"
        )
        _require_passed_report(
            v3 / "stage3_v2_gate.json",
            label="synergy Stage-3 v2 gate",
            expected_metrics=synergy_metrics,
        )
        _require_stage3_artifact_binding(synergy_metrics)
    if step_name == "stage3_signal_export":
        from musclemimic.badminton.stage3_paired_comparison import (
            validate_paired_comparison,
        )

        synergy_metrics = Path(
            artifacts.stage3_v2_metrics or v3 / "stage3_impact_recovery" / "evaluate" / "evaluate_report.json"
        )
        _require_passed_report(
            v3 / "stage3_v2_gate.json",
            label="synergy Stage-3 v2 gate",
            expected_metrics=synergy_metrics,
        )
        _require_stage3_artifact_binding(synergy_metrics)
        validate_paired_comparison(artifacts.stage3_paired_metrics or v3 / "stage3_paired" / "paired_comparison.json")
    if step_name == "emg_validation" and all(
        (
            artifacts.emg_simulation_npz,
            artifacts.emg_measurement_npz,
            artifacts.emg_mapping_json,
        )
    ):
        from musclemimic.badminton.stage3_paired_comparison import (
            validate_paired_comparison,
        )

        paired = validate_paired_comparison(
            artifacts.stage3_paired_metrics or v3 / "stage3_paired" / "paired_comparison.json"
        )
        selected = paired.get("selected_policy_for_emg")
        if not isinstance(selected, dict):
            raise ValueError("paired Stage-3 report has no sealed EMG policy")
        expected = {
            "policy_checkpoint_fingerprint": artifacts.expected_policy_checkpoint_fingerprint,
            "policy_promotion_fingerprint": artifacts.expected_policy_promotion_fingerprint,
            "formal_synergy_basis_fingerprint": artifacts.expected_formal_synergy_basis_fingerprint,
        }
        for key, value in expected.items():
            if value is not None and selected.get(key) != value:
                raise ValueError(f"EMG policy evidence differs from paired Stage-3 selection: {key}")
    if step_name == "physiology_validation" and all(
        (
            artifacts.physiology_input_npz,
            artifacts.physiology_config_json,
        )
    ):
        from musclemimic.badminton.stage3_paired_comparison import (
            validate_paired_comparison,
        )

        paired = validate_paired_comparison(
            artifacts.stage3_paired_metrics or v3 / "stage3_paired" / "paired_comparison.json"
        )
        selected = paired.get("selected_policy_for_emg")
        if not isinstance(selected, dict):
            raise ValueError("paired Stage-3 report has no sealed physiology policy")
        expected = {
            "policy_checkpoint_fingerprint": artifacts.expected_policy_checkpoint_fingerprint,
            "policy_promotion_fingerprint": artifacts.expected_policy_promotion_fingerprint,
            "formal_synergy_basis_fingerprint": artifacts.expected_formal_synergy_basis_fingerprint,
            "event_reference_fingerprint": artifacts.expected_event_reference_fingerprint,
            "policy_decoder_type": artifacts.expected_policy_decoder_type,
        }
        for key, value in expected.items():
            if value is not None and selected.get(key) != value:
                raise ValueError(f"physiology evidence differs from paired Stage-3 selection: {key}")
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
            spec=spec,
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
            spec=spec,
        )
    if step_name in {
        "stage1r_train",
        "stage2_train",
        "stage2_extend_160m",
        "stage2_promote",
    }:
        _require_promoted_artifact(
            artifacts.stage1_promotion_manifest or str(out / "stage1_promotion_manifest.json"),
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
            spec=spec,
        )
    if step_name in {"direct_distill", "direct_distill_resume", "latent_distill"}:
        _require_promoted_artifact(
            artifacts.stage2_promotion_manifest or str(out / "stage2_promotion_manifest.json"),
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
        checkpoint = Path(artifacts.latent_checkpoint or out / "latent_distill" / "latent_checkpoint")
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
        stage3_checkpoint = Path(artifacts.stage3_checkpoint or stage3_dir / "policy_latest.json")
        if not stage3_checkpoint.is_file():
            raise ValueError(f"Stage-3 evaluation requires a trained checkpoint: {stage3_checkpoint}")
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
        raise ValueError("Stage-3 curriculum is incomplete; run stage3_extend_curriculum before evaluation")
    if report.get("promotion_eligible") is not True:
        raise ValueError("Stage-3 train report is not promotion eligible")


def _require_stage3_task_curriculum_complete(path: Path, *, expected_max_stage: str) -> None:
    report = _load_json_mapping(path, label="Stage-3 v2 train report")
    if report.get("task_curriculum_complete") is not True:
        raise ValueError("Stage-3 v2 task curriculum is incomplete")
    checkpoint = Path(str(report.get("checkpoint", ""))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        load_training_checkpoint_metadata,
    )

    metadata = load_training_checkpoint_metadata(checkpoint)
    task_state = metadata.get("task_curriculum_state")
    if (
        not isinstance(task_state, dict)
        or task_state.get("complete") is not True
        or task_state.get("max_stage") != expected_max_stage
    ):
        raise ValueError("Stage-3 v2 checkpoint task-stage binding is incomplete")


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
    impact_recovery_v2 = "policy_abi_hash" in binding
    recorded_binding_hash = binding.get("binding_sha256")
    unbound = dict(binding)
    unbound.pop("binding_sha256", None)
    if recorded_binding_hash != _canonical_mapping_sha256(unbound):
        raise ValueError("Stage-3 evaluation artifact binding hash mismatch")

    from environment.overall_environment.src.train_incoming_hit_mjx import (
        resolve_training_checkpoint,
    )
    from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
        _stage3_action_family,
        _stage3_evaluation_content_sha256,
        _validate_stage3_training_prerequisite_binding,
    )

    if binding.get("evaluation_content_sha256") != _stage3_evaluation_content_sha256(report):
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
    for label in ("training", "evaluation"):
        target_path_value = binding.get(f"{label}_target_path")
        if target_path_value is None:
            continue
        target_path = Path(str(target_path_value))
        if not target_path.is_file():
            raise ValueError(f"Stage-3 bound {label} target bank is missing")
        if binding.get(f"{label}_target_file_sha256") != hashlib.sha256(target_path.read_bytes()).hexdigest():
            raise ValueError(f"Stage-3 bound {label} target bank changed")
    metadata = _load_json_mapping(metadata_path, label="Stage-3 checkpoint metadata")
    if impact_recovery_v2:
        checkpoint_config = metadata.get("config")
        if not isinstance(checkpoint_config, dict) or isinstance(checkpoint_config.get("seed"), bool):
            raise ValueError("Stage-3 checkpoint has no exact training seed")
        try:
            training_seed = int(checkpoint_config["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage-3 checkpoint has no exact training seed") from exc
        if float(checkpoint_config["seed"]) != float(training_seed) or binding.get("training_seed") != training_seed:
            raise ValueError("Stage-3 binding training seed changed")
        evaluation_seed = report.get("evaluation_seed")
        if (
            isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
            or binding.get("evaluation_seed") != evaluation_seed
        ):
            raise ValueError("Stage-3 binding evaluation seed changed")
    train_report = _load_json_mapping(
        Path(str(binding.get("train_report_path", ""))),
        label="Stage-3 bound train report",
    )
    prerequisite_binding = _validate_stage3_training_prerequisite_binding(metadata.get("training_prerequisite_binding"))
    if train_report.get("training_prerequisite_binding") != prerequisite_binding:
        raise ValueError("Stage-3 bound train report and checkpoint prerequisite evidence differ")
    if binding.get("training_prerequisite_binding_sha256") != prerequisite_binding.get("binding_sha256"):
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
            raise ValueError(f"Stage-3 bound train report and checkpoint disagree on {report_key}")
    if metadata.get("curriculum_complete") is not True or metadata.get("promotion_eligible") is not True:
        raise ValueError("Stage-3 bound checkpoint is not curriculum complete")
    if binding.get("checkpoint_iteration") != metadata.get("iteration"):
        raise ValueError("Stage-3 binding checkpoint iteration changed")
    if binding.get("checkpoint_env_steps") != metadata.get("env_steps"):
        raise ValueError("Stage-3 binding checkpoint env-step identity changed")
    control = report.get("control_manifest")
    if not isinstance(control, dict):
        raise ValueError("Stage-3 bound control manifest changed")
    action_family = _stage3_action_family(control)
    if binding.get("action_family") != action_family:
        raise ValueError("Stage-3 bound action family changed")
    if impact_recovery_v2:
        if binding.get("evaluation_control_hash") != control.get("control_hash"):
            raise ValueError("Stage-3 bound control manifest changed")
        if binding.get("policy_abi_hash") != control.get("policy_abi_hash"):
            raise ValueError("Stage-3 bound policy ABI changed")
    elif binding.get("control_hash") != control.get("control_hash"):
        raise ValueError("Stage-3 bound control manifest changed")
    if binding.get("latent_checkpoint_fingerprint") != control.get("latent_checkpoint_fingerprint"):
        raise ValueError("Stage-3 bound latent checkpoint changed")
    if action_family == "full_354" and binding.get("latent_checkpoint_fingerprint") is not None:
        raise ValueError("Stage-3 full_354 binding must have a null latent fingerprint")
    metadata_control = metadata.get("control_manifest")
    if impact_recovery_v2:
        if not isinstance(metadata_control, dict):
            raise ValueError("Stage-3 checkpoint has no control manifest")
        training_control = metadata_control
    else:
        training_control = control
    if prerequisite_binding.get("control_hash") != training_control.get("control_hash") or prerequisite_binding.get(
        "latent_checkpoint_fingerprint"
    ) != control.get("latent_checkpoint_fingerprint"):
        raise ValueError("Stage-3 prerequisite control/latent identity changed")
    if impact_recovery_v2 and prerequisite_binding.get("action_family") != action_family:
        raise ValueError("Stage-3 prerequisite action family changed")
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
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _require_passed_report(
    path: Path,
    *,
    label: str,
    expected_metrics: str | Path | None = None,
) -> None:
    if not path.is_file():
        raise ValueError(f"{label} report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} report is unreadable: {path}") from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError(f"{label} did not pass: {path}")
    if expected_metrics is None:
        return
    metrics_path = Path(expected_metrics).expanduser().resolve(strict=True)
    binding = report.get("source_binding")
    if not isinstance(binding, dict) or binding.get("schema_version") != ("promotion_gate_source_binding_v1"):
        raise ValueError(f"{label} has no source-bound gate evidence")
    if Path(str(binding.get("metrics_path", ""))).resolve(strict=True) != metrics_path:
        raise ValueError(f"{label} is bound to a different metrics artifact")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if binding.get("metrics_content_sha256") != digest:
        raise ValueError(f"{label} metrics changed after gate evaluation")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_schema = metrics.get("schema_version") if isinstance(metrics, dict) else None
    if binding.get("metrics_schema_version") != expected_schema:
        raise ValueError(f"{label} source schema differs from the gated artifact")
    expected_self = {}
    if isinstance(metrics, dict):
        for key in (
            "metrics_fingerprint",
            "report_fingerprint",
            "manifest_fingerprint",
            "binding_sha256",
            "bank_sha256",
            "artifact_binding_sha256",
        ):
            value = metrics.get(key)
            if isinstance(value, str):
                expected_self[key] = value
    if binding.get("metrics_self_fingerprints") != expected_self:
        raise ValueError(f"{label} source self-fingerprint binding is stale")
    unsigned_gate = {key: value for key, value in report.items() if key != "source_binding"}
    unsigned_source = {key: value for key, value in binding.items() if key != "binding_sha256"}
    expected_binding = _canonical_mapping_sha256({"gate": unsigned_gate, "source": unsigned_source})
    if binding.get("binding_sha256") != expected_binding:
        raise ValueError(f"{label} gate/source binding hash is stale")


def _require_v3_direct_baseline(artifacts: PipelineArtifacts, *, v3: Path, spec: ActionSpec = DEFAULT_SPEC) -> None:
    """Revalidate the mass-100 direct comparator before any latent claim."""

    from musclemimic.distill.motion_identity import normalize_motion_path
    from musclemimic.distill.provenance import (
        canonical_json_sha256,
        checkpoint_content_fingerprint,
        file_sha256,
        validate_dataset_manifest,
        validate_direct_acceptance_record,
    )

    root = v3 / "direct_baseline"
    paths = {
        "bc": Path(artifacts.direct_bc_metrics or root / "bc" / "distill_metadata.json"),
        "rollout": Path(artifacts.direct_rollout_metrics or root / "compare" / "comparison_metrics.json"),
        "evidence": Path(artifacts.direct_acceptance or root / "compare" / "direct_promotion_evidence.json"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"v3 direct baseline artifacts are incomplete: {missing}")
    evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schema_version") != ("direct_distill_promotion_evidence_v2"):
        raise ValueError("v3 direct baseline promotion evidence schema is invalid")
    recorded = evidence.get("evidence_fingerprint")
    unsigned = dict(evidence)
    unsigned.pop("evidence_fingerprint", None)
    if recorded != canonical_json_sha256(unsigned):
        raise ValueError("v3 direct baseline evidence fingerprint is stale")
    if evidence.get("promotion_policy") != "student_bc" or evidence.get("deterministic") is not True:
        raise ValueError("v3 direct baseline is not deterministic held-out BC")
    teacher_path = artifacts.racket_mass_100_checkpoint
    if teacher_path is None:
        raise ValueError("v3 direct baseline requires the mass-100 teacher")
    if evidence.get("teacher_checkpoint") != checkpoint_content_fingerprint(teacher_path):
        raise ValueError("v3 direct baseline belongs to a different teacher")
    heldout = evidence.get("heldout")
    expected_motions = [normalize_motion_path(path) for path in spec.val_motion_paths]
    if not isinstance(heldout, dict) or heldout.get("motion_paths") != expected_motions:
        raise ValueError("v3 direct baseline does not use the canonical validation split")
    bound_artifacts = evidence.get("artifacts")
    if not isinstance(bound_artifacts, dict):
        raise ValueError("v3 direct baseline has no bound source artifacts")
    for key in ("comparison_metrics", "acceptance", "convergence", "temporal_audit"):
        record = bound_artifacts.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"v3 direct baseline is missing {key}")
        source = Path(str(record.get("path", ""))).resolve(strict=True)
        if file_sha256(source) != record.get("sha256"):
            raise ValueError(f"v3 direct baseline {key} content changed")
    if Path(bound_artifacts["comparison_metrics"]["path"]).resolve() != paths["rollout"].resolve():
        raise ValueError("v3 direct baseline rollout path differs from selected evidence")
    if Path(bound_artifacts["convergence"]["path"]).resolve() != paths["bc"].resolve():
        raise ValueError("v3 direct baseline BC path differs from selected evidence")
    acceptance = json.loads(Path(bound_artifacts["acceptance"]["path"]).read_text(encoding="utf-8"))
    validate_direct_acceptance_record(acceptance.get("student_bc"))
    validation_dataset = validate_dataset_manifest(
        v3 / "physical_rollout" / "val",
        expected_teacher=checkpoint_content_fingerprint(teacher_path),
        require_promoted_teacher=True,
    )
    if evidence.get("dataset_manifest_fingerprint") != validation_dataset.get("manifest_fingerprint"):
        raise ValueError("v3 direct baseline is not bound to the physical val split")


def _require_target_event_binding(
    target_bank: str | Path,
    *,
    event_metrics: Path,
    split: str,
) -> None:
    from environment.overall_environment.src.stage3_target_bank_v2 import (
        load_target_bank,
        source_fingerprint_from_event_metrics,
    )

    source_fingerprint, source_metadata = source_fingerprint_from_event_metrics(
        event_metrics,
        split=split,
    )
    bank = load_target_bank(
        target_bank,
        expected_source_fingerprint=source_fingerprint,
    )
    for key, expected in source_metadata.items():
        if bank.metadata.get(key) != expected:
            raise ValueError(f"Stage-3 {split} target bank is not bound to current event metrics: {key}")


def _require_latent_selection_binding(artifacts: PipelineArtifacts, *, v3: Path) -> None:
    """Seal formal Stage-3 only to the selected fixed-synergy checkpoint."""

    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        validate_selected_artifact,
    )

    root = v3 / "latent_synergy"
    checkpoint = Path(artifacts.latent_synergy_checkpoint or root / "selected" / "best_synergy")
    manifest_path = Path(artifacts.latent_selection_manifest or root / "selected" / "selection_manifest.json")
    promotion_path = Path(artifacts.latent_synergy_metrics or root / "promotion_metrics.json")
    if not (checkpoint.is_dir() and manifest_path.is_file() and promotion_path.is_file()):
        raise ValueError("Stage-3 requires a sealed fixed-synergy selection")
    manifest = validate_selected_artifact(manifest_path)
    checkpoints = manifest.get("checkpoints") or {}
    if "best_synergy" not in checkpoints:
        raise ValueError("Stage-3 requires a selected best_synergy checkpoint")
    if Path(manifest["promotion_metrics_path"]).resolve(strict=True) != (promotion_path.resolve(strict=True)):
        raise ValueError("latent selection uses a different promotion metrics artifact")
    selected = checkpoints["best_synergy"]
    if Path(selected["stable_checkpoint_path"]).resolve(strict=True) != checkpoint.resolve(strict=True):
        raise ValueError("Stage-3 best_synergy checkpoint differs from sealed selection")
    alias = manifest.get("compatibility_alias") or {}
    if alias.get("target_family") != "best_synergy":
        raise ValueError("canonical Stage-3 alias must select best_synergy")
    if Path(alias.get("stable_checkpoint_path", "")).resolve(strict=True) != (checkpoint.resolve(strict=True)):
        raise ValueError("Stage-3 latent checkpoint differs from sealed best_synergy")


def _require_direct_outputs(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    direct_dir = output_dir / "direct_distill"
    bc_path = Path(artifacts.direct_bc_metrics or direct_dir / "bc" / "distill_metadata.json")
    rollout_path = Path(artifacts.direct_rollout_metrics or direct_dir / "compare" / "comparison_metrics.json")
    acceptance_path = Path(artifacts.direct_acceptance or direct_dir / "compare" / "acceptance.json")
    missing = [str(path) for path in (bc_path, rollout_path, acceptance_path) if not path.is_file()]
    if missing:
        raise ValueError(f"direct distillation artifacts are incomplete: {missing}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    promoted = acceptance.get("student_bc_ppo") or acceptance.get("student_bc_dagger") or acceptance.get("student_bc")
    if not isinstance(promoted, dict) or promoted.get("passed") is not True:
        raise ValueError("direct distilled policy has not passed held-out acceptance gates")


def _require_visual_review(
    path: str | None,
    *,
    review_kind: str,
    stage_label: str,
    checkpoint: str | None,
    spec: ActionSpec = DEFAULT_SPEC,
) -> None:
    if path is None:
        raise ValueError(f"{stage_label} human visual review artifact is required")
    review_path = Path(path)
    if not review_path.is_file():
        raise ValueError(f"{stage_label} human visual review artifact does not exist: {review_path}")
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{stage_label} human visual review artifact is unreadable: {review_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{stage_label} human visual review root must be a JSON object")
    basic_report = validate_visual_review(
        payload,
        required_clips=5,
        expected_motions=spec.val_motions,
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
        expected_motions=spec.val_motions,
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
            raise ValueError(f"Stage-2 extension history is missing finite {name}") from exc
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
    spec: ActionSpec = DEFAULT_SPEC,
) -> None:
    """Bind Stage 1 to the action's clean, immutable release variant."""

    dataset_root = REPO_ROOT / spec.dataset_root
    release_manifest = REPO_ROOT / spec.release_manifest
    if qc_report.get("passed") is not True or qc_report.get("clean_passed") is not True:
        raise ValueError(f"Stage 1 requires warning-free {spec.cache_variant} data QC")
    if qc_report.get("source_variant") != spec.source_variant:
        raise ValueError("Stage 1 data QC used the wrong source variant")
    if qc_report.get("cache_variant") != spec.cache_variant:
        raise ValueError("Stage 1 data QC used the wrong cache variant")
    expected_source_dir = dataset_root / spec.source_namespace
    expected_cache_dir = dataset_root / spec.cache_namespace
    if Path(str(qc_report.get("resolved_source_dir", ""))).resolve() != expected_source_dir.resolve():
        raise ValueError(f"Stage 1 data QC source namespace is not canonical {spec.source_variant}")
    if Path(str(qc_report.get("resolved_cache_dir", ""))).resolve() != expected_cache_dir.resolve():
        raise ValueError(f"Stage 1 data QC cache namespace is not canonical {spec.cache_variant}")
    if tuple(qc_report.get("train_motions", ())) != spec.train_motions:
        raise ValueError(
            "Stage 1 data QC train split is not the canonical ordered "
            f"{len(spec.train_motions)}-motion split"
        )
    if tuple(qc_report.get("validation_motions", ())) != spec.val_motions:
        raise ValueError(
            "Stage 1 data QC validation split is not the canonical ordered "
            f"{len(spec.val_motions)}-motion split"
        )

    cache_root_value = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH")
    if not cache_root_value:
        raise ValueError("MUSCLEMIMIC_GMR_CACHE_PATH is unset; run `source configs/env.sh` before starting Stage 1")
    cache_root = Path(cache_root_value).expanduser().resolve()
    qc_dataset_root = Path(str(qc_report.get("dataset_root", ""))).resolve()
    expected_dataset_root = cache_root / spec.action_id
    if expected_dataset_root != qc_dataset_root:
        raise ValueError(
            f"data QC and runtime cache roots differ: qc={qc_dataset_root} runtime={expected_dataset_root}"
        )
    raw_dir = expected_dataset_root / spec.cache_namespace
    missing = [
        str(raw_dir / f"{motion}.npz")
        for motion in spec.all_motions
        if not (raw_dir / f"{motion}.npz").is_file()
    ]
    if missing:
        raise ValueError(f"runtime {spec.cache_variant} cache is incomplete: {missing}")

    release_validation = validate_release_manifest(dataset_root, release_manifest)
    if release_validation.get("passed") is not True:
        raise ValueError(
            f"{spec.cache_variant} release manifest validation failed: "
            + "; ".join(str(error) for error in release_validation.get("errors", ()))
        )
    release_sha = release_validation.get("release_sha256")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise ValueError(f"{spec.cache_variant} release manifest has no valid content identity")
    visual_qc_path = release_manifest.with_name("visual_qc_report.json")
    visual_validation = validate_visual_qc_report(REPO_ROOT, visual_qc_path)
    if visual_validation.get("passed") is not True:
        raise ValueError(
            f"{spec.cache_variant} visual QC validation failed: "
            + "; ".join(str(error) for error in visual_validation.get("errors", ()))
        )
    binding: dict[str, object] = {
        "schema_version": f"{spec.slug}_data_preflight_binding_v1",
        "dataset_root": str(dataset_root.resolve()),
        "source_variant": spec.source_variant,
        "cache_variant": spec.cache_variant,
        "qc_report_path": str(qc_path.resolve()),
        "qc_report_sha256": hashlib.sha256(qc_path.read_bytes()).hexdigest(),
        "release_manifest_path": str(release_manifest.resolve()),
        "release_manifest_sha256": hashlib.sha256(release_manifest.read_bytes()).hexdigest(),
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
    metrics = extract_validation_records(json.loads(Path(path).read_text(encoding="utf-8")))
    baseline = None
    if baseline_path is not None:
        baseline = latest_validation_record(json.loads(Path(baseline_path).read_text(encoding="utf-8")))
    if not evaluate_promotion(
        stage,
        metrics,
        consecutive=consecutive,
        baseline_metrics=baseline,
    ).passed:
        raise ValueError(f"{stage} promotion gate did not pass")


_CLI_TRUE = frozenset({"true", "1", "yes", "y", "on"})
_CLI_FALSE = frozenset({"false", "0", "no", "n", "off"})


def _cli_bool(value: str) -> bool:
    """Parse a boolean CLI value, refusing anything ambiguous.

    ``bool("False")`` is ``True``, so an untyped flag would silently select the
    shuffled-context control arm for a user who wrote ``False`` to turn it off,
    and the privileged-vs-shuffled gate would then compare an arm against
    itself.  Unrecognised spellings raise instead of guessing.
    """

    text = str(value).strip().lower()
    if text in _CLI_TRUE:
        return True
    if text in _CLI_FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (one of {sorted(_CLI_TRUE | _CLI_FALSE)}), got {value!r}")


def _add_artifact_argument(parser: argparse.ArgumentParser, field_name: str, annotation: str) -> None:
    """Register one ``PipelineArtifacts`` field, typed from its annotation."""

    text = str(annotation)
    if "bool" in text:
        # ``--flag`` alone means True; ``--flag false`` stays available so the
        # field can be switched off explicitly from a generated command line.
        parser.add_argument(
            f"--{field_name}",
            type=_cli_bool,
            nargs="?",
            const=True,
            default=False,
        )
    elif "int" in text:
        parser.add_argument(f"--{field_name}", type=int, default=None)
    else:
        parser.add_argument(f"--{field_name}", default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=action_choices(), default=DEFAULT_ACTION)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--profile", choices=("legacy_v2", "synergy_v3", "stage1_aligned"), default="legacy_v2")
    parser.add_argument("--execute_step", default=None)
    for field_name, field_spec in PipelineArtifacts.__dataclass_fields__.items():
        _add_artifact_argument(parser, field_name, field_spec.type)
    args = parser.parse_args()
    spec = resolve(args.action)
    artifacts = PipelineArtifacts(
        **{field_name: getattr(args, field_name) for field_name in PipelineArtifacts.__dataclass_fields__}
    )
    output = Path(args.output_dir or f"outputs/{spec.slug}_three_stage_v1")
    output.mkdir(parents=True, exist_ok=True)
    steps = build_pipeline_plan(output, artifacts, profile=args.profile, spec=spec)
    plan_path = output / "pipeline_plan.json"
    payload: dict[str, object] = {
        "schema_version": {
            "legacy_v2": f"{spec.slug}_pipeline_v2",
            "synergy_v3": f"{spec.slug}_pipeline_synergy_v3",
            "stage1_aligned": f"{spec.slug}_pipeline_stage1_aligned",
        }[args.profile],
        "profile": args.profile,
        "artifacts": asdict(artifacts),
        "steps": [asdict(step) for step in steps],
    }
    # The sealed forehand-clear plan file has no "action" key; keep it byte-identical.
    if spec.slug != DEFAULT_ACTION:
        payload["action"] = spec.slug
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"pipeline_plan: {plan_path}")
    if args.execute_step:
        execute_pipeline_step(
            args.execute_step,
            output_dir=output,
            artifacts=artifacts,
            profile=args.profile,
            spec=spec,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
