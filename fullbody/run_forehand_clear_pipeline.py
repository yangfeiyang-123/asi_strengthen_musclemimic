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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import (
    DEFAULT_ACTION,
    STAGE1_PEASD_ARMS,
    ActionSpec,
    action_choices,
    resolve,
)
from musclemimic.badminton.action_release import validate_action_release
from musclemimic.badminton.promotion_artifact import validate_promoted_artifact
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
PEASD_EMG_SYNERGY_LOSS_WEIGHT = 0.05


@dataclass(frozen=True)
class PipelineArtifacts:
    stage1_checkpoint: str | None = None
    # Body-only physical QC binds the immutable train/val collections to the
    # exact promoted Stage-1 content.  This field is omitted from the default
    # sealed Forehand-Clear JSON when unset (see main), preserving its ABI.
    stage1_checkpoint_fingerprint: str | None = None
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
    # Immutable Stage-2 context-family handoff.  S2-B creates the shared
    # collection seal and architecture lock; S2-C/D/E may only consume those
    # exact artifacts and therefore never recollect physical rollouts or pick
    # their own latent architecture.
    stage2_shared_inputs_manifest: str | None = None
    stage2_architecture_lock_manifest: str | None = None
    stage2_s2b_output_dir: str | None = None
    stage2_s2c_output_dir: str | None = None
    stage2_s2d_output_dir: str | None = None
    stage2_s2e_output_dir: str | None = None
    stage2_context_family_index: str | None = None
    stage2_context_family_gate: str | None = None
    stage2_direct_family_promotion: str | None = None
    stage2_direct_physical_gpu: int | None = None
    stage2_direct_cache_key_prefix: str | None = None
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
    # Stage-3 reachability chain for the selected latent branch.  The source
    # checkpoint/control/feed identities are explicit; no Stage-3 PPO may
    # infer them from a directory name or jump directly to a later curriculum.
    stage3_reachability_source_checkpoint: str | None = None
    stage3_expected_feed_fingerprint: str | None = None
    stage3_expected_control_hash: str | None = None
    stage3_expected_latent_fingerprint: str | None = None
    stage3_cem_contract: str | None = None
    stage3_cem_report: str | None = None
    stage3_cem_candidate: str | None = None
    stage3_cpu_audit_report: str | None = None
    stage3_cpu_audit_trace: str | None = None
    stage3_cross_backend_seal_report: str | None = None
    stage3_correction_dataset: str | None = None
    stage3_correction_dataset_manifest: str | None = None
    stage3_short_bc_checkpoint: str | None = None
    stage3_short_bc_metrics: str | None = None
    stage3_short_bc_train_report: str | None = None
    stage3_reachability_release: str | None = None
    # The full-354 comparator needs its own reachability chain because its
    # control/latent identity differs from the fixed-synergy branch.
    direct_stage3_reachability_source_checkpoint: str | None = None
    direct_stage3_expected_feed_fingerprint: str | None = None
    direct_stage3_expected_control_hash: str | None = None
    direct_stage3_cem_report: str | None = None
    direct_stage3_cem_candidate: str | None = None
    direct_stage3_cpu_audit_report: str | None = None
    direct_stage3_cross_backend_seal_report: str | None = None
    direct_stage3_correction_dataset: str | None = None
    direct_stage3_correction_dataset_manifest: str | None = None
    direct_stage3_short_bc_checkpoint: str | None = None
    direct_stage3_short_bc_metrics: str | None = None
    direct_stage3_reachability_release: str | None = None
    # One formal Stage-3 family leaf.  The leaf profile derives the selected
    # latent from the sealed Stage-2 family (H1 <- S2-B, H2/H3 <- S2-C), then
    # runs an independent reachability/short-BC/PPO chain for one exact seed.
    stage3_peasd_arm: str | None = None
    stage3_training_seed: int | None = None
    stage3_physical_gpu: int | None = None
    stage3_cache_key_prefix: str | None = None
    # Cross-root H1/H2/H3 family aggregation (three reports and releases per
    # arm, exact seeds 0/1/2).
    stage3_peasd_comparison_contract: str | None = None
    stage3_peasd_family_index: str | None = None
    stage3_peasd_family_gate: str | None = None
    stage3_h1_s0_report: str | None = None
    stage3_h1_s1_report: str | None = None
    stage3_h1_s2_report: str | None = None
    stage3_h2_s0_report: str | None = None
    stage3_h2_s1_report: str | None = None
    stage3_h2_s2_report: str | None = None
    stage3_h3_s0_report: str | None = None
    stage3_h3_s1_report: str | None = None
    stage3_h3_s2_report: str | None = None
    stage3_h1_s0_reachability_release: str | None = None
    stage3_h1_s1_reachability_release: str | None = None
    stage3_h1_s2_reachability_release: str | None = None
    stage3_h2_s0_reachability_release: str | None = None
    stage3_h2_s1_reachability_release: str | None = None
    stage3_h2_s2_reachability_release: str | None = None
    stage3_h3_s0_reachability_release: str | None = None
    stage3_h3_s1_reachability_release: str | None = None
    stage3_h3_s2_reachability_release: str | None = None
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
    # Explicit §24 latent arm.  A promoted PEASD teacher and reviewed tube are
    # shared by S2-B/C/D/E; only latent use of that context changes.
    stage1_peasd_latent_arm: str | None = None
    # Negative control (§26.2 S2-D).  A gate, not a decoration: if the real
    # context does not beat its shuffled twin, the privileged claim is unearned.
    emg_shuffle_context_ablation: bool = False
    # §26.2 S2-E: privileged latent trained with context dropout forced to 0.
    # A distinct arm of the ablation matrix, not a knob for the real arm.
    emg_no_context_dropout: bool = False
    # Stage-1 PEASD-Lite promotion is evaluated from a separately assembled,
    # self-fingerprinted T0/T3/T4 seed-paired metrics document.  Keeping it
    # separate from Stage-2 EMG metrics prevents the two claims from being
    # accidentally interchanged.
    stage1_peasd_pairwise_metrics: str | None = None
    # Exact T0 fixed-budget leaves used by the reward-neutral, verified-tube
    # post-hoc physiology evaluator.  Training itself remains tube-free.
    stage1_peasd_t0_s0_checkpoint: str | None = None
    stage1_peasd_t0_s1_checkpoint: str | None = None
    stage1_peasd_t0_s2_checkpoint: str | None = None
    # Runner-sealed endpoint evidence for the complete matched family.  The
    # pipeline assembles these 15 immutable inputs into the pairwise index;
    # scalar metrics are never accepted here.
    stage1_peasd_t0_s0_validation_evidence: str | None = None
    stage1_peasd_t0_s1_validation_evidence: str | None = None
    stage1_peasd_t0_s2_validation_evidence: str | None = None
    stage1_peasd_t1_s0_validation_evidence: str | None = None
    stage1_peasd_t1_s1_validation_evidence: str | None = None
    stage1_peasd_t1_s2_validation_evidence: str | None = None
    stage1_peasd_t2_s0_validation_evidence: str | None = None
    stage1_peasd_t2_s1_validation_evidence: str | None = None
    stage1_peasd_t2_s2_validation_evidence: str | None = None
    stage1_peasd_t3_s0_validation_evidence: str | None = None
    stage1_peasd_t3_s1_validation_evidence: str | None = None
    stage1_peasd_t3_s2_validation_evidence: str | None = None
    stage1_peasd_t4_s0_validation_evidence: str | None = None
    stage1_peasd_t4_s1_validation_evidence: str | None = None
    stage1_peasd_t4_s2_validation_evidence: str | None = None
    # Structured Stage-1 visual review of the pre-registered T3/seed-0
    # checkpoint.  The PEASD gate binds every held-out clip to that identity.
    stage1_peasd_visual_review: str | None = None
    # Formal promotion accepts only the reviewer-visible opaque package and
    # the separately held private mapping back to the sealed endpoint clips.
    stage1_peasd_blind_review: str | None = None
    stage1_peasd_blind_private_mapping: str | None = None
    # New-schema promotion emitted only after the full T0--T4 paired gate and
    # the T3/seed-0 visual review pass.  Downstream PEASD routes must use this
    # artifact rather than relabeling a historical Stage-1 baseline.
    stage1_peasd_promotion_manifest: str | None = None
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


def _resolve_stage1_peasd_latent_arm(artifacts: PipelineArtifacts) -> str | None:
    """Separate PEASD teacher selection from latent-context treatment."""

    has_teacher = artifacts.stage1_peasd_promotion_manifest is not None
    mode = artifacts.stage1_peasd_latent_arm
    allowed = {"disabled", "real", "shuffled", "real_no_dropout"}
    if not has_teacher:
        if (
            mode is not None
            or artifacts.emg_reference_manifest is not None
            or artifacts.emg_shuffle_context_ablation
            or artifacts.emg_no_context_dropout
        ):
            raise ValueError(
                "Stage1 PEASD latent arms require stage1_peasd_promotion_manifest; "
                "a tube alone must not switch the teacher"
            )
        if artifacts.emg_synergy_dim is not None:
            raise ValueError("emg_synergy_dim requires an explicit Stage1 PEASD latent arm")
        return None
    if artifacts.stage1_checkpoint is None:
        raise ValueError("Stage1 PEASD latent arms require the promoted T3 checkpoint")
    if artifacts.emg_reference_manifest is None:
        raise ValueError("all matched Stage1 PEASD latent arms require the common reviewed tube collection")
    if mode not in allowed:
        raise ValueError("stage1_peasd_latent_arm must explicitly select disabled, real, shuffled, or real_no_dropout")
    if artifacts.emg_shuffle_context_ablation and mode != "shuffled":
        raise ValueError("emg_shuffle_context_ablation conflicts with the explicit latent arm")
    if artifacts.emg_no_context_dropout and mode != "real_no_dropout":
        raise ValueError("emg_no_context_dropout conflicts with the explicit latent arm")
    if mode == "disabled":
        if artifacts.emg_synergy_dim is not None:
            raise ValueError("disabled PEASD latent arm must not consume emg_synergy_dim")
    elif artifacts.emg_synergy_dim is None or int(artifacts.emg_synergy_dim) <= 0:
        raise ValueError(f"{mode} PEASD latent arm requires a positive emg_synergy_dim")
    return mode


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
    if spec.slug == DEFAULT_SPEC.slug:
        # Preserve the sealed forehand-clear command exactly.  Its bespoke
        # validator reconstructs raw_smooth_v1 byte for byte.
        release_command = (
            python,
            "-m",
            "musclemimic.badminton.scripts.data_release",
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(release_manifest),
            "--validate",
        )
    else:
        release_command = (
            python,
            "-m",
            "musclemimic.badminton.action_release",
            "--action",
            spec.slug,
            "--output",
            str(out / "data_release_validation.json"),
            "--require-pass",
        )
    # Keep the sealed Clear command byte-identical.  Other actions must carry
    # their slug because the historical CLI default is Forehand Clear, and
    # ChinaJump must carry its explicit ``wham/...`` source namespace.
    data_qc_action_flags: tuple[str, ...] = ()
    source_variant = spec.source_variant
    if spec.slug != DEFAULT_SPEC.slug:
        data_qc_action_flags = ("--action", spec.slug)
        source_variant = spec.source_namespace
    return (
        PipelineStep(
            "data_release_validate",
            release_command,
        ),
        PipelineStep(
            "data_qc",
            (
                python,
                "-m",
                "musclemimic.badminton.data_qc",
                *data_qc_action_flags,
                "--dataset-root",
                str(dataset_root),
                "--source-variant",
                source_variant,
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
                str(len(spec.val_motions)),
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


def _build_stage1r_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
    peasd_teacher: bool = False,
) -> tuple[PipelineStep, ...]:
    """Build only the two finger-robustness rungs.

    Keeping this segment independent is what lets a body-only action stop after
    Stage 1 without eagerly requiring racket, student, or Stage-3 assets.
    """

    out = Path(output_dir)
    python = sys.executable
    stage1_ckpt = artifacts.stage1_checkpoint or "<required:stage1_checkpoint>"
    stage1r_ckpt = artifacts.stage1r_checkpoint or "<required:stage1r_checkpoint>"
    stage1_requirements = (
        ("stage1_checkpoint", "stage1_peasd_promotion_manifest")
        if peasd_teacher
        else ("stage1_checkpoint", "stage1_metrics", "stage1_visual_review")
    )
    stage1_promotion = artifacts.stage1_peasd_promotion_manifest or "<required:stage1_peasd_promotion_manifest>"
    stage1r_command = [
        python,
        "fullbody/experiment.py",
        f"--config-name={spec.require('stage1r_config')}",
    ]
    if peasd_teacher:
        stage1r_command.append(f"+experiment.parent_checkpoint_lineage.promotion_manifest={stage1_promotion}")
    return (
        PipelineStep(
            "stage1r_train",
            tuple(stage1r_command),
            stage1_requirements,
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
                *spec.val_motion_paths,
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
            (python, "fullbody/experiment.py", f"--config-name={spec.require('stage1r005_config')}"),
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
                *spec.val_motion_paths,
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
                str(len(spec.val_motions)),
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
    if profile == "stage1_peasd":
        return _build_stage1_peasd_steps(output_dir, artifacts, spec=spec)
    if profile == "stage2_context_family":
        return _build_stage2_context_family_steps(
            output_dir,
            artifacts,
            spec=spec,
        )
    if profile == "stage2_direct":
        return _build_stage2_direct_steps(output_dir, artifacts, spec=spec)
    if profile == "stage3_peasd_family":
        return _build_stage3_peasd_family_steps(
            output_dir,
            artifacts,
            spec=spec,
        )
    if profile == "stage3_peasd_arm":
        return _build_stage3_peasd_arm_steps(
            output_dir,
            artifacts,
            spec=spec,
        )
    if profile == "legacy_v2":
        return _build_legacy_pipeline_plan(output_dir, artifacts, spec=spec)
    if profile != "synergy_v3":
        raise ValueError(
            "profile must be 'legacy_v2', 'synergy_v3', 'stage1_aligned', "
            "'stage1_peasd', 'stage2_direct', 'stage2_context_family', or "
            "'stage3_peasd_arm', or 'stage3_peasd_family'"
        )

    # Assemble only the stages that are scientifically applicable.  Building
    # the complete legacy plan first used to require student/Stage-3 assets
    # before the v3 branch could even be selected, which made body-only actions
    # fail on an unrelated Forehand-Clear assumption.
    _resolve_stage1_peasd_latent_arm(artifacts)
    peasd_teacher = artifacts.stage1_peasd_promotion_manifest is not None
    aligned = _build_stage1_aligned_steps(output_dir, artifacts, spec=spec)
    if peasd_teacher and artifacts.stage2_shared_inputs_manifest is not None:
        # S2-C/D/E (and a replayed B) consume the already sealed family-owned
        # collection/basis.  They must never re-enter Stage1R, racket mass,
        # collection, direct BC, or synergy fitting in an arm-specific root.
        return (
            *aligned[:2],
            *_build_stage2_context_arm_steps(output_dir, artifacts, spec=spec),
        )
    if peasd_teacher and _resolve_stage1_peasd_latent_arm(artifacts) != "disabled":
        raise ValueError(
            "formal S2-C/D/E planning requires stage2_shared_inputs_manifest and the S2-B architecture lock"
        )
    # A privileged route consumes the separately completed PEASD experiment;
    # it must not silently schedule/use a new historical Stage-1 baseline.
    steps: tuple[PipelineStep, ...] = aligned[:2] if peasd_teacher else aligned
    if spec.stage1r_applicable:
        steps = (
            *steps,
            *_build_stage1r_steps(
                output_dir,
                artifacts,
                spec=spec,
                peasd_teacher=peasd_teacher,
            ),
        )

    if not spec.racket_applicable:
        # Body-only actions use a separately versioned phase-free contract.
        # An empty phase vocabulary is deliberate: it prevents ChinaJump from
        # inheriting Forehand Clear's impact/recovery claims while still
        # allowing duration-normalized privileged EMG context.
        spec.require("latent_lab_config")
        spec.require("latent_synergy_config")
        return (*steps, *_build_body_only_synergy_v3_steps(output_dir, artifacts, spec=spec))

    # The v3 racket path is defined by an action-specific event bank and all
    # four calibrated load configs.  Legacy Stage-2 YAMLs are not substitutes.
    spec.require("racket_event_bank_config")
    if spec.racket_mass_v2_configs is None:
        spec.require("racket_mass_v2_configs")
    return (
        *steps,
        *_build_synergy_v3_steps(
            output_dir,
            artifacts,
            spec=spec,
            include_stage3=spec.stage3_applicable,
        ),
    )


def _build_stage1_peasd_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
) -> tuple[PipelineStep, ...]:
    """Plan the matched T0--T4 Stage-1 PEASD-Lite ablation.

    Plan construction is intentionally side-effect free and therefore accepts
    a placeholder tube.  The matched no-EMG T0 baseline is deliberately
    executable before the tube is reviewed and contains no tube path/token.
    Execution of every EMG treatment arm remains fail-closed behind the
    verified, training-enabled tube gate.
    """

    root = Path(output_dir) / "stage1_peasd"
    python = sys.executable
    tube = artifacts.emg_reference_manifest or "<required:emg_reference_manifest>"
    tube_gate = root / "verified_tube_gate.json"
    pairwise_metrics = artifacts.stage1_peasd_pairwise_metrics or str(root / "pairwise_evidence_index.json")
    blind_review = artifacts.stage1_peasd_blind_review or "<required:stage1_peasd_blind_review>"
    blind_mapping = artifacts.stage1_peasd_blind_private_mapping or "<required:stage1_peasd_blind_private_mapping>"
    promotion_output = artifacts.stage1_peasd_promotion_manifest or str(root / "stage1_peasd_teacher_promotion.json")

    # Reuse only the action-specific immutable release/QC prefix.  The
    # historical Stage-1 expert is not a substitute for the matched T0 arm.
    aligned_prefix = _build_stage1_aligned_steps(output_dir, artifacts, spec=spec)[:2]
    steps: list[PipelineStep] = [*aligned_prefix]

    # §22 pre-registration order: establish the tube-independent matched
    # baseline first.  Do not even serialize a tube path into these commands.
    for seed in (0, 1, 2):
        arm = "T0"
        run_id = f"{spec.slug}_stage1_peasd_lite_v1_t0_s{seed}"
        steps.append(
            PipelineStep(
                f"stage1_peasd_t0_s{seed}_train",
                (
                    python,
                    "fullbody/experiment.py",
                    f"--config-name={spec.stage1_peasd_config(arm)}",
                    f"experiment.run_id={run_id}",
                    f"wandb.name={run_id}",
                    "experiment.auto_resume=false",
                    "experiment.resume_from=null",
                    "experiment.reset_optimizer_on_resume=true",
                    "experiment.n_seeds=1",
                    f"experiment.seeds=[{seed}]",
                ),
            )
        )

    steps.append(
        PipelineStep(
            "stage1_peasd_tube_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.stage1_peasd_gate",
                "tube",
                "--action",
                spec.slug,
                "--tube",
                tube,
                "--output",
                str(tube_gate),
                "--require-pass",
            ),
            ("emg_reference_manifest",),
        )
    )
    for seed in (0, 1, 2):
        field = f"stage1_peasd_t0_s{seed}_checkpoint"
        checkpoint = getattr(artifacts, field) or f"<required:{field}>"
        steps.append(
            PipelineStep(
                f"stage1_peasd_t0_s{seed}_posthoc_physiology",
                (
                    python,
                    "scripts/evaluate_stage1_peasd.py",
                    "--checkpoint",
                    checkpoint,
                    "--reference-cache",
                    tube,
                ),
                ("emg_reference_manifest", field),
            )
        )
    for arm in STAGE1_PEASD_ARMS:
        if arm == "T0":
            continue
        config = spec.stage1_peasd_config(arm)
        for seed in (0, 1, 2):
            arm_lower = arm.lower()
            run_id = f"{spec.slug}_stage1_peasd_lite_v1_{arm_lower}_s{seed}"
            reference_override = (
                f"experiment.env_params.reward_params.emg_consistency.reference_cache={tube}",
                f"experiment.env_params.reward_params.emg_consistency.action_id={spec.emg_trial_actions[0]}",
            )
            steps.append(
                PipelineStep(
                    f"stage1_peasd_{arm_lower}_s{seed}_train",
                    (
                        python,
                        "fullbody/experiment.py",
                        f"--config-name={config}",
                        f"experiment.run_id={run_id}",
                        f"wandb.name={run_id}",
                        "experiment.auto_resume=false",
                        "experiment.resume_from=null",
                        "experiment.reset_optimizer_on_resume=true",
                        "experiment.n_seeds=1",
                        f"experiment.seeds=[{seed}]",
                        *reference_override,
                    ),
                    ("emg_reference_manifest",),
                )
            )

    evidence_fields = tuple(
        f"stage1_peasd_{arm.lower()}_s{seed}_validation_evidence" for arm in STAGE1_PEASD_ARMS for seed in (0, 1, 2)
    )
    evidence_arguments: list[str] = []
    for arm in STAGE1_PEASD_ARMS:
        for seed in (0, 1, 2):
            field = f"stage1_peasd_{arm.lower()}_s{seed}_validation_evidence"
            path = getattr(artifacts, field) or f"<required:{field}>"
            evidence_arguments.extend(("--evidence", f"{arm}:{seed}:{path}"))
    steps.append(
        PipelineStep(
            "stage1_peasd_evidence_index",
            (
                python,
                "-m",
                "musclemimic.badminton.stage1_peasd_gate",
                "index",
                "--action",
                spec.slug,
                *evidence_arguments,
                "--output",
                pairwise_metrics,
            ),
            evidence_fields,
        )
    )
    steps.append(
        PipelineStep(
            "stage1_peasd_pairwise_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.stage1_peasd_gate",
                "pairwise",
                "--action",
                spec.slug,
                "--metrics",
                pairwise_metrics,
                "--blind-review",
                blind_review,
                "--blind-mapping",
                blind_mapping,
                "--output",
                str(root / "pairwise_promotion_gate.json"),
                "--promotion-output",
                promotion_output,
                "--require-pass",
            ),
            (
                "emg_reference_manifest",
                "stage1_peasd_blind_review",
                "stage1_peasd_blind_private_mapping",
            ),
        )
    )
    return tuple(steps)


def _build_stage2_context_family_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Seal and gate an already completed exact B/C/D/E family."""

    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_shared_inputs,
    )

    required_values = {
        "stage2_shared_inputs_manifest": artifacts.stage2_shared_inputs_manifest,
        "stage2_architecture_lock_manifest": artifacts.stage2_architecture_lock_manifest,
        "stage2_s2b_output_dir": artifacts.stage2_s2b_output_dir,
        "stage2_s2c_output_dir": artifacts.stage2_s2c_output_dir,
        "stage2_s2d_output_dir": artifacts.stage2_s2d_output_dir,
        "stage2_s2e_output_dir": artifacts.stage2_s2e_output_dir,
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        raise ValueError("stage2_context_family requires completed immutable inputs: " + ", ".join(missing))
    shared_path = Path(str(artifacts.stage2_shared_inputs_manifest)).expanduser().resolve(strict=True)
    validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    root = Path(output_dir) / "stage2_context_family"
    family_index = artifacts.stage2_context_family_index or str(root / "family_index.json")
    family_gate = artifacts.stage2_context_family_gate or str(root / "family_gate.json")
    python = sys.executable
    required = tuple(required_values)
    return (
        PipelineStep(
            "stage2_context_family_index",
            (
                python,
                "-m",
                "musclemimic.badminton.stage2_context_family",
                "index",
                "--shared-inputs",
                str(shared_path),
                "--architecture-lock",
                str(artifacts.stage2_architecture_lock_manifest),
                "--s2b-output-dir",
                str(artifacts.stage2_s2b_output_dir),
                "--s2c-output-dir",
                str(artifacts.stage2_s2c_output_dir),
                "--s2d-output-dir",
                str(artifacts.stage2_s2d_output_dir),
                "--s2e-output-dir",
                str(artifacts.stage2_s2e_output_dir),
                "--output",
                family_index,
            ),
            required,
        ),
        PipelineStep(
            "stage2_context_family_gate",
            (
                python,
                "-m",
                "musclemimic.badminton.stage2_context_family",
                "gate",
                "--family-index",
                family_index,
                "--output",
                family_gate,
                "--require-pass",
            ),
        ),
    )


def _build_stage2_direct_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Expand the complete componentized S2-A plan into pipeline steps."""

    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_shared_inputs,
    )
    from musclemimic.distill.stage2_direct_lifecycle import (
        Stage2DirectFamilyConfig,
        build_stage2_direct_family_plan,
    )

    if not spec.racket_applicable:
        raise ValueError(f"S2-A direct lifecycle is not applicable to {spec.slug}")
    if artifacts.stage2_shared_inputs_manifest is None:
        raise ValueError("stage2_direct requires stage2_shared_inputs_manifest")
    if artifacts.stage2_direct_physical_gpu is None:
        raise ValueError("stage2_direct requires an explicit physical GPU")
    if not str(artifacts.stage2_direct_cache_key_prefix or "").strip():
        raise ValueError("stage2_direct requires a stable cache-key prefix")
    shared_path = Path(artifacts.stage2_shared_inputs_manifest).expanduser().resolve(strict=True)
    shared = validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    teacher = shared["teacher"]
    datasets = shared["datasets"]
    root = Path(output_dir) / "stage2_direct"
    config = Stage2DirectFamilyConfig(
        action=spec.slug,
        shared_inputs=str(shared_path),
        source_train_dataset_dir=str(datasets["train"]["path"]),
        source_val_dataset_dir=str(datasets["validation"]["path"]),
        teacher_checkpoint=str(teacher["checkpoint"]["resolved_path"]),
        teacher_promotion_manifest=str(teacher["promotion"]["path"]),
        output_dir=str(root),
        physical_gpu=int(artifacts.stage2_direct_physical_gpu),
        cache_key_prefix=str(artifacts.stage2_direct_cache_key_prefix),
        student_bc_config=spec.require("student_bc_config"),
        student_ppo_config=spec.require("student_ppo_config"),
    )
    _payload, lifecycle_steps = build_stage2_direct_family_plan(config)
    python = sys.executable
    plan_step = PipelineStep(
        "stage2_direct_plan",
        (
            python,
            "-m",
            "fullbody.stage2_direct_lifecycle",
            "plan",
            "--action",
            spec.slug,
            "--shared-inputs",
            str(shared_path),
            "--teacher-checkpoint",
            str(teacher["checkpoint"]["resolved_path"]),
            "--teacher-promotion-manifest",
            str(teacher["promotion"]["path"]),
            "--output-dir",
            str(root),
            "--physical-gpu",
            str(int(artifacts.stage2_direct_physical_gpu)),
            "--cache-key-prefix",
            str(artifacts.stage2_direct_cache_key_prefix),
            "--student-bc-config",
            spec.require("student_bc_config"),
            "--student-ppo-config",
            spec.require("student_ppo_config"),
        ),
        ("stage2_shared_inputs_manifest",),
    )
    expanded = tuple(
        PipelineStep(
            name=step.name,
            command=step.command,
            required_artifacts=("stage2_shared_inputs_manifest",),
            environment=tuple(sorted(step.environment.items())),
        )
        for step in lifecycle_steps
    )
    return (plan_step, *expanded)


def _build_stage3_peasd_arm_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Build one immutable H1/H2/H3 x seed Stage-3 leaf.

    The selected latent is resolved from the passed Stage-2 family gate, not
    from a user-chosen checkpoint.  Every positive-step PPO launch is preceded
    by the CEM -> CPU audit -> cross-backend seal -> correction dataset ->
    zero-step short-BC release chain and stays in one checkpoint root.
    """

    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        validate_selected_artifact,
    )
    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_context_family_gate,
        validate_stage2_context_family_index,
    )

    if not spec.stage3_applicable:
        raise ValueError(f"Stage-3 PEASD leaves are not applicable to {spec.slug}")
    arm = str(artifacts.stage3_peasd_arm or "").upper()
    if arm not in {"H1", "H2", "H3"}:
        raise ValueError("stage3_peasd_arm must be exactly H1, H2, or H3")
    if artifacts.stage3_training_seed not in {0, 1, 2}:
        raise ValueError("stage3_training_seed must be exactly 0, 1, or 2")
    seed = int(artifacts.stage3_training_seed)
    if artifacts.stage3_physical_gpu is None:
        raise ValueError("stage3_peasd_arm requires an explicit physical GPU")
    cache_prefix = str(artifacts.stage3_cache_key_prefix or "").strip()
    if not cache_prefix:
        raise ValueError("stage3_peasd_arm requires a stable cache-key prefix")
    if artifacts.stage2_context_family_gate is None:
        raise ValueError("stage3_peasd_arm requires stage2_context_family_gate")

    gate_path = Path(artifacts.stage2_context_family_gate).expanduser().resolve(strict=True)
    stage2_gate = validate_stage2_context_family_gate(gate_path, require_pass=True)
    action = stage2_gate.get("action") or {}
    if action.get("slug") != spec.slug:
        raise ValueError("Stage-2 family gate belongs to a different action")
    family_record = stage2_gate.get("family_index") or {}
    family_path = Path(str(family_record.get("path", ""))).expanduser().resolve(strict=True)
    stage2_index = validate_stage2_context_family_index(family_path)
    source_arm = "S2-B" if arm == "H1" else "S2-C"
    arm_record = (stage2_index.get("arms") or {}).get(source_arm) or {}
    selection_record = arm_record.get("selection_manifest") or {}
    selection_path = Path(str(selection_record.get("path", ""))).expanduser().resolve(strict=True)
    selection = validate_selected_artifact(selection_path)
    selected = (selection.get("checkpoints") or {}).get("best_synergy")
    if not isinstance(selected, Mapping):
        raise ValueError(f"{source_arm} has no sealed best_synergy checkpoint")
    latent_checkpoint = Path(str(selected.get("stable_checkpoint_path", ""))).expanduser().resolve(strict=True)
    latent_fingerprint = str(selected.get("checkpoint_fingerprint", ""))
    if artifacts.stage3_expected_latent_fingerprint is not None and (
        artifacts.stage3_expected_latent_fingerprint != latent_fingerprint
    ):
        raise ValueError("stage3_expected_latent_fingerprint differs from the Stage-2 selection")
    if artifacts.latent_synergy_checkpoint is not None:
        from musclemimic.latent_muscle.checkpoint import (
            latent_checkpoint_fingerprint,
        )

        supplied_latent = Path(artifacts.latent_synergy_checkpoint).expanduser().resolve(strict=True)
        if latent_checkpoint_fingerprint(supplied_latent) != latent_fingerprint:
            raise ValueError("latent_synergy_checkpoint differs from the sealed Stage-2 selection")

    residual_groups = artifacts.stage3_bounded_residual_groups_json
    if arm == "H3":
        if residual_groups is None:
            raise ValueError("H3 requires stage3_bounded_residual_groups_json")
        Path(residual_groups).expanduser().resolve(strict=True)
    elif residual_groups is not None:
        raise ValueError(f"{arm} must disable the grouped bounded residual")

    required_leaf_fields = (
        "stage3_reachability_source_checkpoint",
        "stage3_expected_feed_fingerprint",
        "stage3_expected_control_hash",
        "recovery_target_bank",
        "recovery_eval_target_bank",
        "recovery_train_feed_bank",
        "recovery_eval_feed_bank",
    )
    missing = [name for name in required_leaf_fields if not getattr(artifacts, name)]
    if missing:
        raise ValueError("stage3_peasd_arm is missing required inputs: " + ", ".join(missing))

    python = sys.executable
    root = Path(output_dir) / "stage3_peasd_arm"
    reachability = root / "reachability"
    cem_dir = reachability / "single_feed_cem"
    seal_dir = reachability / "cross_backend_seal"
    stage3_spec = spec.require("stage3_v2_spec")
    source_checkpoint = str(artifacts.stage3_reachability_source_checkpoint)
    target_bank = str(artifacts.recovery_target_bank)
    eval_target_bank = str(artifacts.recovery_eval_target_bank)
    train_feed_bank = str(artifacts.recovery_train_feed_bank)
    eval_feed_bank = str(artifacts.recovery_eval_feed_bank)
    feed_fingerprint = str(artifacts.stage3_expected_feed_fingerprint)
    control_hash = str(artifacts.stage3_expected_control_hash)

    def fixed_output(field: str, expected: Path) -> str:
        supplied = getattr(artifacts, field)
        if supplied is not None and (Path(supplied).expanduser().resolve() != expected.expanduser().resolve()):
            raise ValueError(f"{field} must be the producer-owned path {expected}; got {supplied}")
        return str(expected)

    fixed_output("stage3_cem_contract", cem_dir / "cem_contract.json")
    cem_report = fixed_output("stage3_cem_report", cem_dir / "cem_report.json")
    cem_candidate = fixed_output("stage3_cem_candidate", cem_dir / "best_teacher.json")
    cpu_trace = artifacts.stage3_cpu_audit_trace or str(reachability / "cpu_audit_trace.npz")
    cpu_report = fixed_output("stage3_cpu_audit_report", Path(cpu_trace).with_suffix(".json"))
    cross_backend_report = fixed_output("stage3_cross_backend_seal_report", seal_dir / "cem_report.json")
    correction_dataset = fixed_output(
        "stage3_correction_dataset",
        seal_dir / "teacher_trajectory_cpu_quality.npz",
    )
    correction_manifest = artifacts.stage3_correction_dataset_manifest or str(
        reachability / "correction_dataset_manifest.json"
    )
    short_bc_checkpoint = artifacts.stage3_short_bc_checkpoint or ("<required:stage3_short_bc_checkpoint>")
    short_bc_metrics = fixed_output("stage3_short_bc_metrics", root / "teacher_bc_pretrain_report.json")
    fixed_output("stage3_short_bc_train_report", root / "train_report.json")
    reachability_release = artifacts.stage3_reachability_release or str(reachability / "reachability_release.json")
    static_checkpoint = artifacts.static_target_checkpoint or ("<required:static_target_checkpoint>")
    static_metrics = fixed_output("static_target_metrics", root / "evaluate_static" / "evaluate_report.json")
    final_checkpoint = artifacts.stage3_v2_checkpoint or ("<required:stage3_v2_checkpoint>")
    final_metrics = fixed_output("stage3_v2_metrics", root / "evaluate" / "evaluate_report.json")
    residual_flags: tuple[str, ...] = (
        (
            "--bounded-residual-groups-json",
            str(residual_groups),
        )
        if residual_groups is not None
        else ()
    )
    runner_common = (
        "--spec",
        stage3_spec,
        "--latent-checkpoint",
        str(latent_checkpoint),
        "--feed-bank",
        train_feed_bank,
        "--eval-feed-bank",
        eval_feed_bank,
        "--target-bank",
        target_bank,
        "--eval-target-bank",
        eval_target_bank,
        *residual_flags,
    )
    launch_environment = tuple(
        sorted(
            {
                "CUDA_VISIBLE_DEVICES": str(int(artifacts.stage3_physical_gpu)),
                "MUSCLEMIMIC_JAX_CACHE_KEY": (f"{cache_prefix}_{arm.lower()}_s{seed}"),
                "MUSCLEMIMIC_TRAIN_LOG": str(root / "training.log"),
            }.items()
        )
    )

    def runner_step(
        name: str,
        stage: str,
        *extra: str,
        required: tuple[str, ...] = (),
        gpu: bool = False,
        run_dir: str | Path | None = None,
    ) -> PipelineStep:
        return PipelineStep(
            name,
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
                *runner_common,
                "--stage",
                stage,
                *extra,
                "--out-dir",
                str(run_dir or root),
            ),
            required,
            launch_environment if gpu else (),
        )

    return (
        runner_step(
            "stage3_v2_preflight",
            "preflight",
            required=required_leaf_fields,
        ),
        runner_step(
            "stage3_v2_feed_check",
            "feed-check",
            required=required_leaf_fields,
        ),
        runner_step(
            "stage3_v2_base_only",
            "base-only-check",
            required=required_leaf_fields,
        ),
        PipelineStep(
            "stage3_single_feed_cem",
            (
                str(REPO_ROOT / "scripts" / "run_fullbody_training.sh"),
                "--incoming-hit-cem",
                "--spec",
                stage3_spec,
                "--checkpoint",
                source_checkpoint,
                "--out-dir",
                str(cem_dir),
                "--feed-fingerprint",
                feed_fingerprint,
                "--seed",
                str(seed),
            ),
            (*required_leaf_fields, "stage2_context_family_gate"),
            launch_environment,
        ),
        PipelineStep(
            "stage3_candidate_cpu_audit",
            (
                python,
                str(REPO_ROOT / "scripts" / "audit_cem_candidate_cpu.py"),
                "--run-dir",
                str(cem_dir),
                "--candidate",
                cem_candidate,
                "--feed-fingerprint",
                feed_fingerprint,
                "--output",
                cpu_trace,
            ),
            ("stage3_cem_contract", "stage3_cem_report", "stage3_cem_candidate"),
        ),
        PipelineStep(
            "stage3_cross_backend_seal",
            (
                python,
                str(REPO_ROOT / "scripts" / "seal_cross_backend_hit_teacher.py"),
                "--source-cem-report",
                cem_report,
                "--out-dir",
                str(seal_dir),
            ),
            ("stage3_cem_report", "stage3_cpu_audit_report", "stage3_cpu_audit_trace"),
        ),
        PipelineStep(
            "stage3_correction_dataset_seal",
            (
                python,
                "-m",
                "musclemimic.badminton.stage3_reachability_release",
                "build-dataset-manifest",
                "--action",
                spec.slug,
                "--expected-stage3-spec",
                stage3_spec,
                "--expected-feed-fingerprint",
                feed_fingerprint,
                "--expected-control-hash",
                control_hash,
                "--expected-latent-fingerprint",
                latent_fingerprint,
                "--source-cem-report",
                cem_report,
                "--candidate",
                cem_candidate,
                "--cpu-audit-report",
                cpu_report,
                "--cross-backend-seal-report",
                cross_backend_report,
                "--correction-dataset",
                correction_dataset,
                "--output",
                correction_manifest,
            ),
            (
                "stage3_cem_report",
                "stage3_cem_candidate",
                "stage3_cpu_audit_report",
                "stage3_cross_backend_seal_report",
                "stage3_correction_dataset",
            ),
        ),
        runner_step(
            "stage3_short_bc",
            "train-gpu",
            "--initialize-policy-from",
            source_checkpoint,
            "--teacher-dataset",
            correction_dataset,
            "--total-env-steps",
            "0",
            "--curriculum-max-stage",
            "C3_static_velocity",
            "--seed",
            str(seed),
            required=(
                "stage3_reachability_source_checkpoint",
                "stage3_correction_dataset",
                "stage3_correction_dataset_manifest",
            ),
            gpu=True,
        ),
        PipelineStep(
            "stage3_reachability_release",
            (
                python,
                "-m",
                "musclemimic.badminton.stage3_reachability_release",
                "build-release",
                "--correction-dataset-manifest",
                correction_manifest,
                "--short-bc-checkpoint",
                short_bc_checkpoint,
                "--short-bc-metrics",
                short_bc_metrics,
                "--output",
                reachability_release,
            ),
            (
                "stage3_correction_dataset_manifest",
                "stage3_short_bc_checkpoint",
                "stage3_short_bc_metrics",
                "stage3_short_bc_train_report",
            ),
        ),
        runner_step(
            "stage3_static_target_train",
            "train-gpu",
            "--resume-from",
            short_bc_checkpoint,
            "--teacher-dataset",
            correction_dataset,
            "--stage3-reachability-release",
            reachability_release,
            "--total-env-steps",
            "6000000",
            "--curriculum-max-stage",
            "C3_static_velocity",
            "--seed",
            str(seed),
            required=(
                "stage3_short_bc_checkpoint",
                "stage3_correction_dataset",
                "stage3_correction_dataset_manifest",
                "stage3_reachability_release",
            ),
            gpu=True,
        ),
        runner_step(
            "stage3_static_target_evaluate",
            "evaluate",
            "--checkpoint",
            static_checkpoint,
            "--episodes",
            "128",
            required=("static_target_checkpoint", "stage3_reachability_release"),
            run_dir=root / "evaluate_static",
        ),
        PipelineStep(
            "stage3_static_target_gate",
            _gate_command(
                python,
                "static_target_v2",
                static_metrics,
                root / "stage3_static_target_gate.json",
            ),
            ("static_target_metrics", "stage3_reachability_release"),
        ),
        runner_step(
            "stage3_v2_train",
            "train-gpu",
            "--resume-from",
            static_checkpoint,
            "--teacher-dataset",
            correction_dataset,
            "--stage3-reachability-release",
            reachability_release,
            "--total-env-steps",
            "30000000",
            "--curriculum-max-stage",
            "C7_recovery",
            "--seed",
            str(seed),
            required=(
                "static_target_checkpoint",
                "static_target_metrics",
                "stage3_correction_dataset",
                "stage3_correction_dataset_manifest",
                "stage3_reachability_release",
            ),
            gpu=True,
        ),
        runner_step(
            "stage3_v2_evaluate",
            "evaluate",
            "--checkpoint",
            final_checkpoint,
            "--episodes",
            "128",
            required=("stage3_v2_checkpoint", "stage3_reachability_release"),
            run_dir=root / "evaluate",
        ),
        PipelineStep(
            "stage3_v2_gate",
            _gate_command(
                python,
                "stage3_v2",
                final_metrics,
                root / "stage3_v2_gate.json",
            ),
            ("stage3_v2_metrics", "stage3_reachability_release"),
        ),
    )


def _build_stage3_peasd_family_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Index and gate the exact H1/H2/H3 x seed family."""

    if not spec.stage3_applicable:
        raise ValueError(f"Stage-3 PEASD family is not applicable to {spec.slug}")
    if artifacts.stage2_context_family_gate is None:
        raise ValueError("stage3_peasd_family requires stage2_context_family_gate")
    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_context_family_gate,
    )
    from musclemimic.badminton.stage3_peasd_family import (
        DEFAULT_COMPARISON_CONTRACT,
    )

    validate_stage2_context_family_gate(
        artifacts.stage2_context_family_gate,
        require_pass=True,
    )
    leaf_fields = tuple(
        f"stage3_h{arm}_s{seed}_{kind}"
        for arm in (1, 2, 3)
        for seed in (0, 1, 2)
        for kind in ("report", "reachability_release")
    )
    missing = [name for name in leaf_fields if getattr(artifacts, name) is None]
    if missing:
        raise ValueError("stage3_peasd_family requires exact H1/H2/H3 x seeds 0/1/2: " + ", ".join(missing))
    root = Path(output_dir) / "stage3_peasd_family"
    family_index = artifacts.stage3_peasd_family_index or str(root / "family_index.json")
    family_gate = artifacts.stage3_peasd_family_gate or str(root / "family_gate.json")
    command: list[str] = [
        sys.executable,
        "-m",
        "musclemimic.badminton.stage3_peasd_family",
        "index",
        "--stage2-family-gate",
        str(artifacts.stage2_context_family_gate),
        "--comparison-contract",
        str(artifacts.stage3_peasd_comparison_contract or DEFAULT_COMPARISON_CONTRACT),
    ]
    for arm in (1, 2, 3):
        for seed in (0, 1, 2):
            command.extend(
                (
                    f"--h{arm}-report",
                    f"{seed}={getattr(artifacts, f'stage3_h{arm}_s{seed}_report')}",
                    f"--h{arm}-reachability-release",
                    f"{seed}={getattr(artifacts, f'stage3_h{arm}_s{seed}_reachability_release')}",
                )
            )
    command.extend(("--output", family_index))
    return (
        PipelineStep(
            "stage3_peasd_family_index",
            tuple(command),
            ("stage2_context_family_gate", *leaf_fields),
        ),
        PipelineStep(
            "stage3_peasd_family_gate",
            (
                sys.executable,
                "-m",
                "musclemimic.badminton.stage3_peasd_family",
                "gate",
                "--family-index",
                family_index,
                "--output",
                family_gate,
                "--require-pass",
            ),
        ),
    )


def _build_body_only_synergy_v3_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Build Stage-1-teacher PEASD without racket/event/Stage-3 claims.

    This path shares the physical collector, synergy fitter, latent sweep, and
    numerical thresholds with the racket path, but it has two deliberately
    different evidence schemas: a formal Stage-1/body-only teacher binding and
    phase-free physical/latent promotion reports.  That separation is what
    makes ChinaJump a valid generalization endpoint rather than a mislabeled
    Stage-2 hitting experiment.
    """

    root = Path(output_dir) / "synergy_v3"
    python = sys.executable
    teacher = artifacts.stage1_checkpoint or "<required:stage1_checkpoint>"
    latent_arm = _resolve_stage1_peasd_latent_arm(artifacts)
    peasd_teacher = artifacts.stage1_peasd_promotion_manifest is not None
    teacher_promotion = (
        artifacts.stage1_peasd_promotion_manifest or "<required:stage1_peasd_promotion_manifest>"
        if peasd_teacher
        else artifacts.stage1_promotion_manifest or str(Path(output_dir) / "stage1_promotion_manifest.json")
    )
    promotion_artifact_name = "stage1_peasd_promotion_manifest" if peasd_teacher else "stage1_promotion_manifest"
    teacher_fingerprint = artifacts.stage1_checkpoint_fingerprint or "<required:stage1_checkpoint_fingerprint>"
    physical_train = root / "physical_rollout" / "train"
    physical_val = root / "physical_rollout" / "val"
    physical_metrics = artifacts.physical_rollout_metrics or str(root / "physical_rollout" / "promotion_metrics.json")
    synergy_dir = root / "synergy"
    synergy_metrics = artifacts.synergy_metrics or str(synergy_dir / "promotion_metrics.json")
    basis = artifacts.synergy_basis or str(synergy_dir / "physical_excitation_unit" / "regional_composite")
    basis_fingerprint = artifacts.synergy_basis_fingerprint or "<required:synergy_basis_fingerprint>"
    frozen_body_decoder = artifacts.frozen_body_decoder or "<required:frozen_body_decoder_from_stage1_release>"
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
    latent_synergy_metrics = artifacts.latent_synergy_metrics or str(latent_dir / "promotion_metrics.json")
    grouping = artifacts.synergy_grouping or str(REPO_ROOT / spec.synergy_grouping)
    train_motions = spec.train_motion_paths
    val_motions = spec.val_motion_paths
    sweep_base_config = spec.require("latent_synergy_config")
    stage2_shared = artifacts.stage2_shared_inputs_manifest or str(root / "stage2_shared_inputs.json")
    stage2_family_flags: tuple[str, ...] = ()
    stage2_shared_steps: tuple[PipelineStep, ...] = ()
    stage2_lock_steps: tuple[PipelineStep, ...] = ()
    if peasd_teacher:
        if latent_arm != "disabled":
            raise ValueError(
                "S2-C/D/E must consume an existing stage2_shared_inputs_manifest; "
                "they must not recollect the body-only physical dataset"
            )
        stage2_shared_steps = (
            PipelineStep(
                "stage2_shared_inputs_seal",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.stage2_context_family",
                    "seal-shared",
                    "--action",
                    spec.slug,
                    "--train-dataset-dir",
                    str(physical_train),
                    "--val-dataset-dir",
                    str(physical_val),
                    "--teacher-checkpoint",
                    teacher,
                    "--teacher-promotion-manifest",
                    teacher_promotion,
                    "--stage1-peasd-promotion-manifest",
                    str(artifacts.stage1_peasd_promotion_manifest),
                    "--emg-reference-manifest",
                    str(artifacts.emg_reference_manifest),
                    "--physical-qc-metrics",
                    physical_metrics,
                    "--physical-qc-gate",
                    str(root / "physical_rollout_gate.json"),
                    "--synergy-basis",
                    basis,
                    "--frozen-body-decoder",
                    frozen_body_decoder,
                    "--output",
                    stage2_shared,
                ),
                (
                    "stage1_checkpoint",
                    "stage1_peasd_promotion_manifest",
                    "emg_reference_manifest",
                    "frozen_body_decoder",
                ),
            ),
        )
        stage2_family_flags = (
            "--stage2-arm",
            "S2-B",
            "--stage2-shared-inputs",
            stage2_shared,
        )
        stage2_lock_steps = (
            PipelineStep(
                "stage2_s2b_architecture_lock",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.stage2_context_family",
                    "lock-architecture",
                    "--shared-inputs",
                    stage2_shared,
                    "--s2b-output-dir",
                    str(latent_dir),
                    "--output",
                    artifacts.stage2_architecture_lock_manifest or str(root / "stage2_s2b_architecture_lock.json"),
                ),
            ),
        )

    if spec.racket_applicable or spec.stage3_applicable or spec.stage1r_applicable:
        raise ValueError(
            f"action {spec.action_id!r} cannot use the body-only PEASD builder "
            "while racket, Stage1R, or Stage3 is applicable"
        )
    if spec.latent_phase_ready:
        raise ValueError(
            f"action {spec.action_id!r} declares observed phases; use an "
            "action-specific event-aware builder instead of the phase-free contract"
        )

    if artifacts.emg_reference_manifest is None:
        emg_collect_flags: tuple[str, ...] = ()
        emg_privileged_flags: tuple[str, ...] = ()
    else:
        emg_collect_flags = (
            "--save-emg-reference",
            "--emg-reference-cache",
            str(artifacts.emg_reference_manifest),
            "--stage1-peasd-promotion-manifest",
            teacher_promotion,
        )
        if latent_arm == "disabled":
            emg_privileged_flags = ()
        else:
            emg_privileged_flags = (
                "--emg-privileged-enabled",
                "--emg-synergy-dim",
                str(int(artifacts.emg_synergy_dim)),
                "--emg-reference-manifest",
                str(artifacts.emg_reference_manifest),
                "--emg-synergy-loss-weight",
                str(PEASD_EMG_SYNERGY_LOSS_WEIGHT),
            )
            if latent_arm == "shuffled":
                emg_privileged_flags += ("--emg-shuffle-context-ablation",)
            if latent_arm == "real_no_dropout":
                emg_privileged_flags += ("--emg-context-dropout", "0.0")

    collect_common = (
        python,
        "-m",
        "fullbody.distill_collect",
        "--teacher_ckpt",
        teacher,
        "--num_transitions",
        "1000000",
        "--save-physical-muscle-state",
        "--save_reference_features",
        "--teacher-promotion-manifest",
        teacher_promotion,
        "--teacher-promotion-stage",
        "stage1",
        "--teacher-promotion-role",
        "body_only",
        *emg_collect_flags,
    )

    def gate(stage: str, metrics: str | None, name: str) -> PipelineStep:
        return PipelineStep(
            name,
            _gate_command(python, stage, metrics, root / f"{name}.json"),
        )

    return (
        PipelineStep(
            "physical_rollout_collect",
            (
                *collect_common,
                "--motion_path",
                *train_motions,
                "--output_dir",
                str(physical_train),
                "--split",
                "train",
            ),
            (
                "stage1_checkpoint",
                promotion_artifact_name,
                *(("emg_reference_manifest",) if artifacts.emg_reference_manifest else ()),
            ),
        ),
        PipelineStep(
            "physical_rollout_collect_val",
            (
                *collect_common,
                "--motion_path",
                *val_motions,
                "--output_dir",
                str(physical_val),
                "--split",
                "val",
            ),
            (
                "stage1_checkpoint",
                promotion_artifact_name,
                *(("emg_reference_manifest",) if artifacts.emg_reference_manifest else ()),
            ),
        ),
        PipelineStep(
            "physical_rollout_qc",
            (
                python,
                "-m",
                "musclemimic.distill.physical_qc",
                "--qc-contract",
                "body-only-phase-free",
                "--train",
                str(physical_train),
                "--val",
                str(physical_val),
                "--output",
                physical_metrics,
                "--teacher-checkpoint-fingerprint",
                teacher_fingerprint,
            ),
            (
                "stage1_checkpoint",
                promotion_artifact_name,
                "stage1_checkpoint_fingerprint",
            ),
        ),
        gate(
            "physical_rollout_body_only_v1",
            physical_metrics,
            "physical_rollout_gate",
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
        *stage2_shared_steps,
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
                *(
                    (
                        "--stage1-peasd-promotion-manifest",
                        str(artifacts.stage1_peasd_promotion_manifest),
                    )
                    if peasd_teacher
                    else ()
                ),
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
                "--base-config",
                sweep_base_config,
                "--expected-validation-motion-count",
                str(spec.latent_expected_val_motion_count),
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
                *stage2_family_flags,
                *emg_privileged_flags,
            ),
            (
                "synergy_basis",
                "synergy_basis_fingerprint",
                "frozen_body_decoder",
                "frozen_body_decoder_fingerprint",
                "body_synergy_contract_fingerprint",
                "body_synergy_portable_core_fingerprint",
                "stage1_checkpoint",
                promotion_artifact_name,
                "stage1_checkpoint_fingerprint",
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
            "latent_synergy_analysis",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "analyze",
                "--output-dir",
                str(latent_dir),
            ),
        ),
        gate(
            "latent_synergy_body_only_v1",
            latent_synergy_metrics,
            "latent_synergy_gate",
        ),
        *stage2_lock_steps,
    )


def _build_stage2_context_arm_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec,
) -> tuple[PipelineStep, ...]:
    """Plan one C/D/E arm from a sealed shared collection and S2-B lock.

    This deliberately has no collector, direct-student, synergy-fit, or
    architecture-selection step.  Those are family-owned inputs.  Repeating
    them per arm would destroy the matched comparison even if the CLI flags
    happened to look alike.
    """

    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_shared_inputs,
    )

    latent_mode = _resolve_stage1_peasd_latent_arm(artifacts)
    arm_by_mode = {
        "disabled": "S2-B",
        "real": "S2-C",
        "shuffled": "S2-D",
        "real_no_dropout": "S2-E",
    }
    if latent_mode not in arm_by_mode:
        raise ValueError("sealed Stage-2 context arms require a promoted PEASD teacher")
    stage2_arm = arm_by_mode[latent_mode]
    if artifacts.stage2_shared_inputs_manifest is None:
        raise ValueError(f"{stage2_arm} requires stage2_shared_inputs_manifest")
    shared_path = Path(artifacts.stage2_shared_inputs_manifest).expanduser().resolve(strict=True)
    shared = validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    if artifacts.stage1_peasd_promotion_manifest is None:
        raise ValueError(f"{stage2_arm} requires stage1_peasd_promotion_manifest")
    if artifacts.emg_reference_manifest is None:
        raise ValueError(f"{stage2_arm} requires the family EMG reference manifest")

    def same_path(left: str | Path, right: str | Path, *, label: str) -> None:
        if Path(left).expanduser().resolve(strict=True) != Path(right).expanduser().resolve(strict=True):
            raise ValueError(f"{stage2_arm} {label} differs from the shared family")

    stage1 = shared["stage1_peasd"]
    same_path(
        artifacts.stage1_peasd_promotion_manifest,
        stage1["promotion"]["path"],
        label="Stage-1 PEASD promotion",
    )
    same_path(
        artifacts.emg_reference_manifest,
        stage1["emg_reference"]["path"],
        label="EMG reference",
    )

    lock_flags: tuple[str, ...] = ()
    required: tuple[str, ...] = (
        "stage2_shared_inputs_manifest",
        "stage1_peasd_promotion_manifest",
        "stage1_checkpoint",
        "emg_reference_manifest",
    )
    if stage2_arm != "S2-B":
        if artifacts.stage2_architecture_lock_manifest is None:
            raise ValueError(f"{stage2_arm} requires stage2_architecture_lock_manifest")
        lock_flags = (
            "--stage2-architecture-lock",
            str(artifacts.stage2_architecture_lock_manifest),
        )
        required = (*required, "stage2_architecture_lock_manifest")

    datasets = shared["datasets"]
    teacher = shared["teacher"]
    synergy = shared["synergy"]
    direct = shared["direct_s2a_evidence"]
    root = Path(output_dir) / "synergy_v3"
    latent_dir = root / "latent_synergy"
    python = sys.executable
    direct_flags: tuple[str, ...] = ()
    if bool(direct.get("required")):
        if artifacts.stage2_direct_family_promotion is None:
            raise ValueError(f"{stage2_arm} requires stage2_direct_family_promotion")
        from musclemimic.distill.stage2_direct_lifecycle import (
            validate_stage2_direct_family_promotion,
        )

        validate_stage2_direct_family_promotion(
            artifacts.stage2_direct_family_promotion,
            expected_action=spec.slug,
            expected_shared_inputs=shared_path,
        )
        direct_flags = (
            "--direct-bc-metrics",
            str(direct["bc_metrics"]["path"]),
            "--direct-rollout-metrics",
            str(direct["rollout_metrics"]["path"]),
            "--direct-promotion-evidence",
            str(direct["promotion_evidence"]["path"]),
            "--stage2-direct-family-promotion",
            str(artifacts.stage2_direct_family_promotion),
        )
        required = (*required, "stage2_direct_family_promotion")

    emg_flags: tuple[str, ...] = ()
    if stage2_arm != "S2-B":
        emg_flags = (
            "--emg-privileged-enabled",
            "--emg-synergy-dim",
            str(int(artifacts.emg_synergy_dim)),
            "--emg-reference-manifest",
            str(artifacts.emg_reference_manifest),
            "--emg-synergy-loss-weight",
            str(PEASD_EMG_SYNERGY_LOSS_WEIGHT),
        )
        if stage2_arm == "S2-D":
            emg_flags += ("--emg-shuffle-context-ablation",)
        if stage2_arm == "S2-E":
            emg_flags += ("--emg-context-dropout", "0.0")

    causal_flags = ("--require-causal-interventions",) if spec.latent_phase_ready else ()
    latent_metrics = artifacts.latent_synergy_metrics or str(latent_dir / "promotion_metrics.json")
    gate_stage = "latent_synergy_v2" if spec.latent_phase_ready else "latent_synergy_body_only_v1"
    steps: list[PipelineStep] = [
        PipelineStep(
            "latent_dimension_sweep",
            (
                python,
                "-m",
                "musclemimic.badminton.scripts.latent_synergy_sweep",
                "plan",
                "--dataset-dir",
                str(datasets["train"]["path"]),
                "--val-dataset-dir",
                str(datasets["validation"]["path"]),
                "--teacher-ckpt",
                str(teacher["checkpoint"]["resolved_path"]),
                "--teacher-promotion-manifest",
                str(teacher["promotion"]["path"]),
                "--stage1-peasd-promotion-manifest",
                str(stage1["promotion"]["path"]),
                *direct_flags,
                "--synergy-basis",
                str(synergy["basis"]["path"]),
                "--synergy-basis-fingerprint",
                str(synergy["basis"]["artifact_fingerprint"]),
                "--frozen-body-decoder",
                str(synergy["frozen_body_decoder"]["path"]),
                "--frozen-body-decoder-fingerprint",
                str(synergy["frozen_body_decoder"]["artifact_fingerprint"]),
                "--body-synergy-contract-fingerprint",
                str(synergy["frozen_body_decoder"]["body_synergy_contract_fingerprint"]),
                "--body-synergy-portable-core-fingerprint",
                str(synergy["frozen_body_decoder"]["portable_decoder_core_fingerprint"]),
                "--output-dir",
                str(latent_dir),
                "--base-config",
                spec.require("latent_synergy_config"),
                "--expected-validation-motion-count",
                str(spec.latent_expected_val_motion_count),
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
                "--stage2-arm",
                stage2_arm,
                "--stage2-shared-inputs",
                str(shared_path),
                *lock_flags,
                *causal_flags,
                *emg_flags,
            ),
            required,
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
    ]
    if spec.latent_phase_ready:
        steps.extend(
            (
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
                        artifacts.latent_causal_adapter_config or "<required:latent_causal_adapter_config>",
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
            )
        )
    steps.extend(
        (
            PipelineStep(
                "latent_synergy_analysis",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.scripts.latent_synergy_sweep",
                    "analyze",
                    "--output-dir",
                    str(latent_dir),
                    *(("--require-all-phases",) if spec.latent_require_all_phases else ()),
                    *causal_flags,
                ),
            ),
            PipelineStep(
                "latent_synergy_gate",
                _gate_command(
                    python,
                    gate_stage,
                    latent_metrics,
                    root / "latent_synergy_gate.json",
                ),
            ),
        )
    )
    if stage2_arm == "S2-B":
        lock_output = artifacts.stage2_architecture_lock_manifest or str(root / "stage2_s2b_architecture_lock.json")
        steps.append(
            PipelineStep(
                "stage2_s2b_architecture_lock",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.stage2_context_family",
                    "lock-architecture",
                    "--shared-inputs",
                    str(shared_path),
                    "--s2b-output-dir",
                    str(latent_dir),
                    "--output",
                    lock_output,
                ),
                ("stage2_shared_inputs_manifest",),
            )
        )
    return tuple(steps)


def _build_synergy_v3_steps(
    output_dir: str | Path,
    artifacts: PipelineArtifacts,
    *,
    spec: ActionSpec = DEFAULT_SPEC,
    include_stage3: bool = True,
) -> tuple[PipelineStep, ...]:
    root = Path(output_dir) / "synergy_v3"
    python = sys.executable
    latent_arm = _resolve_stage1_peasd_latent_arm(artifacts)
    stage3_v2_spec = spec.require("stage3_v2_spec") if include_stage3 else "<not-applicable:stage3_v2_spec>"
    stage3_direct_spec = spec.require("stage3_direct_spec") if include_stage3 else "<not-applicable:stage3_direct_spec>"
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
    validation_count_flags: tuple[str, ...] = (
        ()
        if spec.latent_expected_val_motion_count == DEFAULT_SPEC.latent_expected_val_motion_count
        else (
            "--expected-validation-motion-count",
            str(spec.latent_expected_val_motion_count),
        )
    )
    causal_plan_flags: tuple[str, ...] = ("--require-causal-interventions",) if spec.latent_phase_ready else ()
    stage2_shared = artifacts.stage2_shared_inputs_manifest or str(root / "stage2_shared_inputs.json")
    stage2_family_flags: tuple[str, ...] = ()
    stage2_shared_steps: tuple[PipelineStep, ...] = ()
    stage2_lock_steps: tuple[PipelineStep, ...] = ()
    if latent_arm is not None:
        if latent_arm != "disabled":
            raise ValueError(
                "S2-C/D/E must consume an existing stage2_shared_inputs_manifest; "
                "they must not recollect racket physical rollouts or refit the basis"
            )
        stage2_shared_steps = (
            PipelineStep(
                "stage2_shared_inputs_seal",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.stage2_context_family",
                    "seal-shared",
                    "--action",
                    spec.slug,
                    "--train-dataset-dir",
                    str(physical_train),
                    "--val-dataset-dir",
                    str(physical_val),
                    "--teacher-checkpoint",
                    teacher,
                    "--teacher-promotion-manifest",
                    teacher_promotion,
                    "--stage1-peasd-promotion-manifest",
                    str(artifacts.stage1_peasd_promotion_manifest),
                    "--emg-reference-manifest",
                    str(artifacts.emg_reference_manifest),
                    "--physical-qc-metrics",
                    physical_metrics,
                    "--physical-qc-gate",
                    str(root / "physical_rollout_gate.json"),
                    "--synergy-basis",
                    basis,
                    "--frozen-body-decoder",
                    frozen_body_decoder,
                    "--direct-bc-metrics",
                    direct_bc_metrics,
                    "--direct-rollout-metrics",
                    direct_rollout_metrics,
                    "--direct-promotion-evidence",
                    direct_acceptance,
                    "--output",
                    stage2_shared,
                ),
                (
                    "stage1_peasd_promotion_manifest",
                    "stage1_checkpoint",
                    "emg_reference_manifest",
                    "racket_mass_100_checkpoint",
                    "racket_mass_100_promotion_manifest",
                    "frozen_body_decoder",
                ),
            ),
        )
        stage2_family_flags = (
            "--stage2-arm",
            "S2-B",
            "--stage2-shared-inputs",
            stage2_shared,
        )
        stage2_lock_steps = (
            PipelineStep(
                "stage2_s2b_architecture_lock",
                (
                    python,
                    "-m",
                    "musclemimic.badminton.stage2_context_family",
                    "lock-architecture",
                    "--shared-inputs",
                    stage2_shared,
                    "--s2b-output-dir",
                    str(latent_dir),
                    "--output",
                    artifacts.stage2_architecture_lock_manifest or str(root / "stage2_s2b_architecture_lock.json"),
                ),
            ),
        )

    # S2-B/C/D/E share one promoted PEASD teacher, collection and reviewed
    # tube.  Only the latent-context flags differ.
    if artifacts.emg_reference_manifest is None:
        emg_privileged_flags: tuple[str, ...] = ()
    else:
        if latent_arm == "disabled":
            emg_privileged_flags = ()
        else:
            emg_privileged_flags = (
                "--emg-privileged-enabled",
                "--emg-synergy-dim",
                str(int(artifacts.emg_synergy_dim)),
                "--emg-reference-manifest",
                str(artifacts.emg_reference_manifest),
                "--emg-synergy-loss-weight",
                str(PEASD_EMG_SYNERGY_LOSS_WEIGHT),
            )
            if latent_arm == "shuffled":
                emg_privileged_flags += ("--emg-shuffle-context-ablation",)
            if latent_arm == "real_no_dropout":
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
    mass_configs = spec.racket_mass_v2_configs
    if mass_configs is None:  # guarded by build_pipeline_plan; keep direct callers safe.
        spec.require("racket_mass_v2_configs")
        raise AssertionError("unreachable")
    default_mass_configs = DEFAULT_SPEC.racket_mass_v2_configs
    default_mass_config_by_scale = dict(zip(("025", "050", "075", "100"), default_mass_configs or (), strict=True))
    for scale, mass_config in zip(("025", "050", "075", "100"), mass_configs, strict=True):
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
                    *(() if mass_config == default_mass_config_by_scale.get(scale) else ("--config-name", mass_config)),
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
                    str(len(spec.val_motions)),
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
    emg_collect_flags: tuple[str, ...] = ()
    if artifacts.emg_reference_manifest is not None:
        emg_collect_flags = (
            "--save-emg-reference",
            "--emg-reference-cache",
            str(artifacts.emg_reference_manifest),
            "--stage1-peasd-promotion-manifest",
            (artifacts.stage1_peasd_promotion_manifest or "<required:stage1_peasd_promotion_manifest>"),
        )
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
        *emg_collect_flags,
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
                *(
                    (
                        "emg_reference_manifest",
                        "stage1_peasd_promotion_manifest",
                        "stage1_checkpoint",
                    )
                    if artifacts.emg_reference_manifest is not None
                    else ()
                ),
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
                *(
                    (
                        "emg_reference_manifest",
                        "stage1_peasd_promotion_manifest",
                        "stage1_checkpoint",
                    )
                    if artifacts.emg_reference_manifest is not None
                    else ()
                ),
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
        *stage2_shared_steps,
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
                *(
                    (
                        "--stage1-peasd-promotion-manifest",
                        str(artifacts.stage1_peasd_promotion_manifest),
                    )
                    if latent_arm is not None
                    else ()
                ),
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
                *stage2_family_flags,
                *validation_count_flags,
                *causal_plan_flags,
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
                *(("--require-all-phases",) if spec.latent_require_all_phases else ()),
                *causal_plan_flags,
            ),
        ),
        gate("latent_synergy_v2", latent_synergy_metrics, "latent_synergy_gate"),
        *stage2_lock_steps,
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
    if latent_arm == "disabled" and artifacts.stage2_shared_inputs_manifest is None:
        # Formal racket PEASD first seals the family-owned physical inputs.
        # Complete S2-A is then run through the separate component profile;
        # only after its family promotion may S2-B/C/D/E consume the seal.
        cutoff = next(index for index, step in enumerate(steps) if step.name == "stage2_shared_inputs_seal")
        steps = steps[: cutoff + 1]
    if not spec.latent_phase_ready:
        # The current causal adapter and its outcome contract are Stage-2
        # event/phase aware.  A canonical disabled phase contract still permits
        # phase-free latent metrics, but must not manufacture causal phase
        # evidence for another action.
        steps = tuple(step for step in steps if step.name not in {"latent_causal_evaluate", "latent_causal_finalize"})
    if not include_stage3:
        # Racket actions may provide valid Stage-1/2 + latent generalization
        # evidence before a scientifically matched incoming-shuttle protocol is
        # available.  Stop at the latent gate; never borrow clear's Stage-3.
        cutoff = next(index for index, step in enumerate(steps) if step.name == "recovery_target")
        steps = steps[:cutoff]

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
    _verify_upstream_gates(
        step_name,
        artifacts,
        output_dir=output_dir,
        profile=profile,
        spec=spec,
    )
    env = os.environ.copy()
    env.update(dict(step.environment))
    command = _canonical_training_launch_command(step)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _canonical_training_launch_command(step: PipelineStep) -> list[str]:
    """Route every production trainer through the repository launcher.

    The serialized forehand-clear plan intentionally retains its historical
    commands for reproducibility.  Execution is stricter: the root launch
    contract owns CUDA selection, environment paths, compilation cache, Orbax
    limits and append-only logging.
    """

    command = list(step.command)
    launcher = str(REPO_ROOT / "scripts" / "run_fullbody_training.sh")
    if command[1:3] == ["-m", "fullbody.run_distill_experiment"]:
        raise ValueError(
            f"pipeline step {step.name!r} uses the retired monolithic legacy "
            "distillation orchestrator, whose nested trainers bypass the "
            "canonical launcher; use profile='synergy_v3' component steps"
        )
    if len(command) >= 2 and command[1] == "fullbody/experiment.py":
        return [launcher, *command[2:]]
    if len(command) >= 3 and command[1:3] == ["-m", "fullbody.distill_train_bc"]:
        return [launcher, "--distill-bc", *command[3:]]
    if len(command) >= 3 and command[1:3] == ["-m", "fullbody.latent_train"]:
        return [launcher, "--latent", *command[3:]]
    if len(command) >= 2 and command[1] == "scripts/evaluate_stage1_peasd.py":
        return [launcher, "--stage1-peasd-eval", *command[2:]]
    incoming = "musclemimic.badminton.scripts.run_incoming_shuttle_hit"
    if len(command) >= 3 and command[1:3] == ["-m", incoming]:
        try:
            stage = command[command.index("--stage") + 1]
        except (ValueError, IndexError):
            stage = None
        if stage == "train-gpu":
            return [launcher, "--incoming-hit", *command[3:]]
    if len(command) >= 3 and command[1:3] == [
        "-m",
        "environment.overall_environment.src.train_incoming_hit_mjx",
    ]:
        raise ValueError(
            f"pipeline step {step.name!r} uses the retired direct Stage-3 trainer; "
            "execute Stage-3 through run_incoming_shuttle_hit --stage train-gpu"
        )
    return command


def _verify_stage3_peasd_arm_upstream(
    step_name: str,
    artifacts: PipelineArtifacts,
    *,
    output_dir: Path,
    spec: ActionSpec,
) -> None:
    """Rebuild every formal Stage-3 leaf prerequisite before execution."""

    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_context_family_gate,
    )
    from musclemimic.badminton.stage3_reachability_release import (
        validate_stage3_reachability_release,
        validate_successful_correction_dataset_manifest,
    )

    if artifacts.stage2_context_family_gate is None:
        raise ValueError("Stage-3 PEASD execution requires a Stage-2 family gate")
    gate = validate_stage2_context_family_gate(
        artifacts.stage2_context_family_gate,
        require_pass=True,
    )
    if (gate.get("action") or {}).get("slug") != spec.slug:
        raise ValueError("Stage-3 PEASD execution uses a different action family gate")

    root = output_dir / "stage3_peasd_arm"
    if step_name in {
        "stage3_single_feed_cem",
        "stage3_short_bc",
        "stage3_static_target_train",
    }:
        for filename, label in (
            ("preflight_report.json", "Stage-3 preflight"),
            ("feed_check_report.json", "Stage-3 feed check"),
            ("base_only_report.json", "Stage-3 base-only check"),
        ):
            _require_passed_report(root / filename, label=label)

    if step_name in {
        "stage3_short_bc",
        "stage3_reachability_release",
        "stage3_static_target_train",
        "stage3_v2_train",
    }:
        if artifacts.stage3_correction_dataset_manifest is None:
            raise ValueError(f"{step_name} requires stage3_correction_dataset_manifest")
        correction = validate_successful_correction_dataset_manifest(artifacts.stage3_correction_dataset_manifest)
        recorded_dataset = (correction.get("correction_dataset") or {}).get("path")
        if artifacts.stage3_correction_dataset is None or Path(
            artifacts.stage3_correction_dataset
        ).expanduser().resolve(strict=True) != Path(str(recorded_dataset)).expanduser().resolve(strict=True):
            raise ValueError("Stage-3 correction dataset differs from its immutable manifest")

    if step_name in {
        "stage3_static_target_train",
        "stage3_v2_train",
        "stage3_static_target_evaluate",
        "stage3_static_target_gate",
        "stage3_v2_evaluate",
        "stage3_v2_gate",
    }:
        if artifacts.stage3_reachability_release is None:
            raise ValueError(f"{step_name} requires stage3_reachability_release")
        release = validate_stage3_reachability_release(artifacts.stage3_reachability_release)
        if step_name == "stage3_static_target_train":
            checkpoint = (release.get("short_bc") or {}).get("checkpoint") or {}
            expected = Path(str(checkpoint.get("payload_path", ""))).expanduser().resolve(strict=True)
            if (
                artifacts.stage3_short_bc_checkpoint is None
                or Path(artifacts.stage3_short_bc_checkpoint).expanduser().resolve(strict=True) != expected
            ):
                raise ValueError("C3 must resume the immutable short-BC payload sealed by the release")

    if step_name in {
        "stage3_static_target_evaluate",
        "stage3_static_target_gate",
        "stage3_v2_train",
    }:
        _require_stage3_task_curriculum_complete(
            root / "train_report.json",
            expected_max_stage="C3_static_velocity",
        )
    if step_name == "stage3_v2_train":
        _require_passed_report(
            root / "stage3_static_target_gate.json",
            label="Stage-3 static-target gate",
            expected_metrics=artifacts.static_target_metrics,
        )
    if step_name in {"stage3_v2_evaluate", "stage3_v2_gate"}:
        _require_stage3_curriculum_complete(root / "train_report.json")


def _verify_upstream_gates(
    step_name: str,
    artifacts: PipelineArtifacts,
    *,
    output_dir: str | Path,
    profile: str = "legacy_v2",
    spec: ActionSpec = DEFAULT_SPEC,
) -> None:
    out = Path(output_dir)
    if profile == "stage3_peasd_arm":
        _verify_stage3_peasd_arm_upstream(
            step_name,
            artifacts,
            output_dir=out,
            spec=spec,
        )
        return
    v3 = out / "synergy_v3"
    is_stage1_peasd_train = step_name.startswith("stage1_peasd_") and step_name.endswith("_train")
    if is_stage1_peasd_train:
        qc_path = out / "data_qc.json"
        if not qc_path.is_file():
            raise ValueError(f"Stage-1 PEASD train step {step_name!r} requires completed data QC: {qc_path}")
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Stage-1 PEASD data QC is unreadable: {qc_path}") from exc
        if not isinstance(qc, dict) or qc.get("passed") is not True:
            raise ValueError("Stage-1 PEASD requires passing canonical action data QC")
        # This also rebuilds and checks the action-owned release inventory; a
        # skipped or stale release step therefore cannot be hidden by a copied
        # QC JSON.
        _require_canonical_cache_environment(
            qc,
            qc_path=qc_path,
            preflight_path=out / "stage1_peasd" / "data_preflight_binding.json",
            spec=spec,
        )
    if (
        step_name.startswith("stage1_peasd_")
        and step_name != "stage1_peasd_tube_gate"
        and not (step_name.startswith("stage1_peasd_t0_") and step_name.endswith("_train"))
    ):
        from musclemimic.badminton.stage1_peasd_gate import (
            validate_verified_tube_gate,
        )

        if artifacts.emg_reference_manifest is None:
            raise ValueError(f"pipeline step {step_name!r} requires emg_reference_manifest")
        validate_verified_tube_gate(
            out / "stage1_peasd" / "verified_tube_gate.json",
            expected_action=spec.slug,
            expected_tube=artifacts.emg_reference_manifest,
        )
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
        if spec.racket_applicable:
            if artifacts.emg_reference_manifest is not None:
                _require_stage1_teacher_promotion(
                    artifacts.stage1_peasd_promotion_manifest,
                    checkpoint=artifacts.stage1_checkpoint,
                    require_peasd=True,
                    expected_tube=artifacts.emg_reference_manifest,
                )
            _require_passed_report(
                v3 / "event_reference_gate.json",
                label="event-reference v2 gate",
                expected_metrics=(
                    artifacts.event_reference_metrics or v3 / "event_reference" / "promotion_metrics.json"
                ),
            )
            from musclemimic.badminton.racket_mass_curriculum import (
                validate_mass_promoted_artifact,
            )

            if artifacts.racket_mass_100_checkpoint is None:
                raise ValueError("physical rollout requires the promoted 100% racket checkpoint")
            mass_promotion = validate_mass_promoted_artifact(
                artifacts.racket_mass_100_promotion_manifest or str(mass_root / "mass_100_promotion_manifest.json"),
                expected_stage="mass_100",
                expected_checkpoint=artifacts.racket_mass_100_checkpoint,
            )
            if artifacts.emg_reference_manifest is not None:
                _require_checkpoint_descends_from_stage1_peasd_promotion(
                    mass_promotion["checkpoint"],
                    artifacts.stage1_peasd_promotion_manifest,
                )
        else:
            if artifacts.stage1_checkpoint is None:
                raise ValueError("body-only physical rollout requires a promoted Stage-1 checkpoint")
            _require_stage1_teacher_promotion(
                (
                    artifacts.stage1_peasd_promotion_manifest
                    if artifacts.emg_reference_manifest is not None
                    else artifacts.stage1_promotion_manifest or str(out / "stage1_promotion_manifest.json")
                ),
                checkpoint=artifacts.stage1_checkpoint,
                require_peasd=artifacts.emg_reference_manifest is not None,
                expected_tube=artifacts.emg_reference_manifest,
            )
    if step_name in {"synergy_fit", "synergy_gate"}:
        _require_passed_report(
            v3 / "physical_rollout_gate.json",
            label=(
                "physical rollout v2 gate" if spec.racket_applicable else "body-only phase-free physical rollout gate"
            ),
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
        if artifacts.stage2_shared_inputs_manifest is not None:
            from musclemimic.badminton.stage2_context_family import (
                validate_stage2_s2b_architecture_lock,
                validate_stage2_shared_inputs,
            )

            shared = validate_stage2_shared_inputs(
                artifacts.stage2_shared_inputs_manifest,
                expected_action=spec.slug,
            )
            if _resolve_stage1_peasd_latent_arm(artifacts) != "disabled":
                if artifacts.stage2_architecture_lock_manifest is None:
                    raise ValueError("S2-C/D/E latent steps require the S2-B architecture lock")
                validate_stage2_s2b_architecture_lock(
                    artifacts.stage2_architecture_lock_manifest,
                    expected_shared_inputs=artifacts.stage2_shared_inputs_manifest,
                )
            if shared.get("binding_sha256") is None:
                raise ValueError("Stage-2 shared inputs have no immutable binding")
        else:
            _require_passed_report(
                v3 / "synergy_gate.json",
                label="synergy v2 gate",
                expected_metrics=(artifacts.synergy_metrics or v3 / "synergy" / "promotion_metrics.json"),
            )
            if spec.racket_applicable:
                _require_v3_direct_baseline(artifacts, v3=v3, spec=spec)
    if step_name == "stage2_shared_inputs_seal":
        _require_passed_report(
            v3 / "synergy_gate.json",
            label="synergy v2 gate",
            expected_metrics=(artifacts.synergy_metrics or v3 / "synergy" / "promotion_metrics.json"),
        )
        if spec.racket_applicable:
            _require_v3_direct_baseline(artifacts, v3=v3, spec=spec)
    if step_name == "stage2_s2b_architecture_lock":
        _require_passed_report(
            v3 / "latent_synergy_gate.json",
            label=("latent-synergy v2 gate" if spec.latent_phase_ready else "body-only latent-synergy gate"),
            expected_metrics=(artifacts.latent_synergy_metrics or v3 / "latent_synergy" / "promotion_metrics.json"),
        )
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
    peasd_stage1_teacher = artifacts.stage1_peasd_promotion_manifest is not None
    if step_name in {
        "stage1_visual_gate",
        "stage1_promote",
        "stage2_train",
    } or (step_name == "stage1r_train" and not peasd_stage1_teacher):
        _require_metrics_gate("stage1", artifacts.stage1_metrics, consecutive=3)
    if step_name in {"stage1_promote", "stage2_train"} or (step_name == "stage1r_train" and not peasd_stage1_teacher):
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
        _require_stage1_teacher_promotion(
            (
                artifacts.stage1_peasd_promotion_manifest
                if peasd_stage1_teacher
                else artifacts.stage1_promotion_manifest or str(out / "stage1_promotion_manifest.json")
            ),
            checkpoint=artifacts.stage1_checkpoint,
            require_peasd=peasd_stage1_teacher,
            expected_tube=(artifacts.emg_reference_manifest if peasd_stage1_teacher else None),
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
        required_clips=len(spec.val_motions),
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
        required_clips=len(spec.val_motions),
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


def _require_stage1_teacher_promotion(
    path: str | None,
    *,
    checkpoint: str | None,
    require_peasd: bool,
    expected_tube: str | None = None,
) -> dict[str, Any]:
    """Validate one Stage-1 teacher without conflating promotion schemas."""

    if path is None:
        kind = "PEASD " if require_peasd else ""
        raise ValueError(f"{kind}Stage-1 teacher promotion artifact is required")
    if checkpoint is None:
        raise ValueError("Stage-1 promoted checkpoint is required")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stage-1 teacher promotion is unreadable: {path}") from exc
    from musclemimic.badminton.stage1_peasd_gate import (
        PEASD_TEACHER_PROMOTION_SCHEMA_VERSION,
        validate_stage1_peasd_teacher_promotion,
    )

    if isinstance(payload, dict) and payload.get("schema_version") == (PEASD_TEACHER_PROMOTION_SCHEMA_VERSION):
        return validate_stage1_peasd_teacher_promotion(
            path,
            expected_checkpoint=checkpoint,
            expected_tube=expected_tube,
        )
    if require_peasd:
        raise ValueError(
            "privileged/PEASD downstream route requires the new paired T3 "
            "teacher promotion schema; a legacy Stage-1 artifact is not a substitute"
        )
    return validate_promoted_artifact(
        path,
        expected_stage="stage1",
        expected_checkpoint=checkpoint,
    )


def _require_checkpoint_descends_from_stage1_peasd_promotion(
    checkpoint: Mapping[str, Any],
    promotion_path: str | None,
) -> None:
    """Prove a later racket teacher recursively descends from the PEASD teacher."""

    if promotion_path is None:
        raise ValueError("racket PEASD route requires the Stage-1 promotion artifact")
    path = Path(promotion_path).expanduser().resolve(strict=True)
    try:
        promotion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-1 PEASD promotion artifact is unreadable") from exc
    if not isinstance(promotion, dict):
        raise ValueError("Stage-1 PEASD promotion artifact must be a JSON object")
    target_content = hashlib.sha256(path.read_bytes()).hexdigest()
    target_binding = str(promotion.get("binding_sha256", ""))
    lineage: Any = checkpoint.get("parent_checkpoint_lineage")
    while isinstance(lineage, Mapping):
        binding = lineage.get("promotion")
        if isinstance(binding, Mapping) and (
            binding.get("evidence_kind") == "verified_stage1_peasd_promotion_v1"
            and binding.get("artifact_content_sha256") == target_content
            and binding.get("artifact_binding_sha256") == target_binding
        ):
            return
        lineage = lineage.get("parent_checkpoint_lineage")
    raise ValueError("racket-mass teacher ancestry does not contain the selected Stage-1 PEASD promotion")


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
    # The sealed Forehand Clear command keeps its historical bare variant,
    # while action-owned QC commands carry the full source namespace so that
    # ``wham/...`` (ChinaJump) can never be mistaken for ``temp/...``.  Bind
    # the report to the exact spelling emitted by the planner; the resolved
    # path check below remains the authoritative namespace identity.
    expected_qc_source_variant = spec.source_variant if spec.slug == DEFAULT_SPEC.slug else spec.source_namespace
    if qc_report.get("source_variant") != expected_qc_source_variant:
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
            f"Stage 1 data QC train split is not the canonical ordered {len(spec.train_motions)}-motion split"
        )
    if tuple(qc_report.get("validation_motions", ())) != spec.val_motions:
        raise ValueError(
            f"Stage 1 data QC validation split is not the canonical ordered {len(spec.val_motions)}-motion split"
        )

    # ``execute_pipeline_step`` performs this check before it hands the actual
    # trainer to ``scripts/run_fullbody_training.sh``.  The launcher sources
    # configs/env.sh itself, so requiring an already-sourced interactive shell
    # here would contradict the production launch contract.  Mirror env.sh's
    # deterministic defaults when the caller has not explicitly overridden
    # either root, then let the launcher establish the real process env.
    datasets_root_value = os.environ.get("MUSCLEMIMIC_DATASETS_ROOT", str(REPO_ROOT / "datasets"))
    cache_root_value = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH", datasets_root_value)
    cache_root = Path(cache_root_value).expanduser().resolve()
    qc_dataset_root = Path(str(qc_report.get("dataset_root", ""))).resolve()
    expected_dataset_root = cache_root / spec.action_id
    if expected_dataset_root != qc_dataset_root:
        raise ValueError(
            f"data QC and runtime cache roots differ: qc={qc_dataset_root} runtime={expected_dataset_root}"
        )
    raw_dir = expected_dataset_root / spec.cache_namespace
    missing = [
        str(raw_dir / f"{motion}.npz") for motion in spec.all_motions if not (raw_dir / f"{motion}.npz").is_file()
    ]
    if missing:
        raise ValueError(f"runtime {spec.cache_variant} cache is incomplete: {missing}")

    release_validation = validate_action_release(spec)
    if release_validation.get("passed") is not True:
        raise ValueError(
            f"{spec.cache_variant} release manifest validation failed: "
            + "; ".join(str(error) for error in release_validation.get("errors", ()))
        )
    release_sha = release_validation.get("release_binding_sha256")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise ValueError(f"{spec.cache_variant} release manifest has no valid content identity")
    visual_qc_value = release_validation.get("visual_qc_path")
    visual_qc_path = (
        (REPO_ROOT / str(visual_qc_value)).resolve() if isinstance(visual_qc_value, str) and visual_qc_value else None
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
        "formal_release_manifest": bool(release_validation.get("formal_release_manifest", False)),
        "review_evidence_kind": release_validation.get("review_evidence_kind"),
        "evidence_limitations": list(release_validation.get("evidence_limitations", ())),
        "visual_qc_report_path": None if visual_qc_path is None else str(visual_qc_path),
        "visual_qc_report_sha256": release_validation.get("visual_qc_sha256"),
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
    parser.add_argument(
        "--profile",
        choices=(
            "legacy_v2",
            "synergy_v3",
            "stage1_aligned",
            "stage1_peasd",
            "stage2_direct",
            "stage2_context_family",
            "stage3_peasd_arm",
            "stage3_peasd_family",
        ),
        default="legacy_v2",
    )
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
    serialized_artifacts = asdict(artifacts)
    if spec.slug == DEFAULT_ACTION and serialized_artifacts.get("stage1_checkpoint_fingerprint") is None:
        # Added only for the body-only Stage-1 teacher contract.  Do not grow
        # the sealed default Forehand-Clear payload with an irrelevant null.
        serialized_artifacts.pop("stage1_checkpoint_fingerprint")
    if args.profile != "stage1_peasd" and serialized_artifacts.get("stage1_peasd_pairwise_metrics") is None:
        # These fields belong only to the opt-in Stage-1 matched-ablation
        # profile; keep every other profile's serialized ABI unchanged.
        serialized_artifacts.pop("stage1_peasd_pairwise_metrics")
    for optional_peasd_field in (
        "stage1_peasd_visual_review",
        "stage1_peasd_blind_review",
        "stage1_peasd_blind_private_mapping",
        "stage1_peasd_promotion_manifest",
        "stage1_peasd_t0_s0_checkpoint",
        "stage1_peasd_t0_s1_checkpoint",
        "stage1_peasd_t0_s2_checkpoint",
        *(f"stage1_peasd_{arm.lower()}_s{seed}_validation_evidence" for arm in STAGE1_PEASD_ARMS for seed in (0, 1, 2)),
    ):
        if args.profile != "stage1_peasd" and serialized_artifacts.get(optional_peasd_field) is None:
            serialized_artifacts.pop(optional_peasd_field)
    if serialized_artifacts.get("stage1_peasd_latent_arm") is None:
        serialized_artifacts.pop("stage1_peasd_latent_arm")
    for optional_stage2_family_field in (
        "stage2_shared_inputs_manifest",
        "stage2_architecture_lock_manifest",
        "stage2_s2b_output_dir",
        "stage2_s2c_output_dir",
        "stage2_s2d_output_dir",
        "stage2_s2e_output_dir",
        "stage2_context_family_index",
        "stage2_context_family_gate",
        "stage2_direct_family_promotion",
        "stage2_direct_physical_gpu",
        "stage2_direct_cache_key_prefix",
    ):
        if serialized_artifacts.get(optional_stage2_family_field) is None:
            serialized_artifacts.pop(optional_stage2_family_field)
    for optional_stage3_formal_field in (
        "stage3_reachability_source_checkpoint",
        "stage3_expected_feed_fingerprint",
        "stage3_expected_control_hash",
        "stage3_expected_latent_fingerprint",
        "stage3_cem_contract",
        "stage3_cem_report",
        "stage3_cem_candidate",
        "stage3_cpu_audit_report",
        "stage3_cpu_audit_trace",
        "stage3_cross_backend_seal_report",
        "stage3_correction_dataset",
        "stage3_correction_dataset_manifest",
        "stage3_short_bc_checkpoint",
        "stage3_short_bc_metrics",
        "stage3_short_bc_train_report",
        "stage3_reachability_release",
        "direct_stage3_reachability_source_checkpoint",
        "direct_stage3_expected_feed_fingerprint",
        "direct_stage3_expected_control_hash",
        "direct_stage3_cem_report",
        "direct_stage3_cem_candidate",
        "direct_stage3_cpu_audit_report",
        "direct_stage3_cross_backend_seal_report",
        "direct_stage3_correction_dataset",
        "direct_stage3_correction_dataset_manifest",
        "direct_stage3_short_bc_checkpoint",
        "direct_stage3_short_bc_metrics",
        "direct_stage3_reachability_release",
        "stage3_peasd_arm",
        "stage3_training_seed",
        "stage3_physical_gpu",
        "stage3_cache_key_prefix",
        "stage3_peasd_comparison_contract",
        "stage3_peasd_family_index",
        "stage3_peasd_family_gate",
        *(
            f"stage3_h{arm}_s{seed}_{kind}"
            for arm in (1, 2, 3)
            for seed in (0, 1, 2)
            for kind in ("report", "reachability_release")
        ),
    ):
        if serialized_artifacts.get(optional_stage3_formal_field) is None:
            serialized_artifacts.pop(optional_stage3_formal_field)
    payload: dict[str, object] = {
        "schema_version": {
            "legacy_v2": f"{spec.slug}_pipeline_v2",
            "synergy_v3": f"{spec.slug}_pipeline_synergy_v3",
            "stage1_aligned": f"{spec.slug}_pipeline_stage1_aligned",
            "stage1_peasd": f"{spec.slug}_pipeline_stage1_peasd_lite_v1",
            "stage2_direct": f"{spec.slug}_pipeline_stage2_direct_v1",
            "stage2_context_family": f"{spec.slug}_pipeline_stage2_context_family_v1",
            "stage3_peasd_arm": f"{spec.slug}_pipeline_stage3_peasd_arm_v1",
            "stage3_peasd_family": f"{spec.slug}_pipeline_stage3_peasd_family_v1",
        }[args.profile],
        "profile": args.profile,
        "artifacts": serialized_artifacts,
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
