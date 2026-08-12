"""Content-bound Stage-1 PEASD tube, evidence, and promotion gates."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.badminton.action_registry import FOREHAND_CLEAR
from musclemimic.badminton.action_release import validate_action_release
from musclemimic.badminton.data_qc import inspect_canonical_dataset
from musclemimic.badminton.promotion_artifact import checkpoint_identity
from musclemimic.badminton.stage1_peasd_gate import (
    CANONICAL_ARMS,
    CANONICAL_SEEDS,
    _canonical_sha256,
    _validate_delivered_treatment,
    _validate_runtime_binding,
    build_pairwise_evidence_index,
    build_pairwise_metrics_document,
    build_stage1_peasd_teacher_promotion,
    build_verified_tube_gate,
    evaluate_pairwise_promotion,
    validate_stage1_peasd_teacher_promotion,
)
from musclemimic.badminton.visual_review import STAGE1_REVIEW_KIND
from musclemimic.physiology.emg_consistency_runtime import (
    EmgConsistencyConfig,
    EmgReferenceBundle,
    build_emg_consistency_preflight_contract,
)
from musclemimic.physiology.emg_reference import (
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    build_emg_dual_track_normalization,
    build_phase_reference_tube,
    save_emg_phase_reference_tube,
)
from musclemimic.runner.checkpointing import (
    STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES,
    STAGE1_SOURCE_TREE_SCOPES,
    STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION,
    bind_explicit_parent_checkpoint,
)
from musclemimic.runner.stage1_peasd_validation import (
    STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS,
    STAGE1_PEASD_VALIDATION_METRIC_KEYS,
    append_stage1_peasd_validation_history,
    build_stage1_peasd_training_treatment_sample,
    build_stage1_peasd_validation_evidence,
    stage1_peasd_training_treatment_targets,
    write_stage1_peasd_validation_evidence,
)
from musclemimic.runner.validation_video_recorder import (
    ENDPOINT_REVIEW_SET_SCHEMA_VERSION,
    build_stage1_peasd_blind_review_package,
    validate_stage1_peasd_blind_mapping,
    validate_stage1_peasd_blind_review,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_snapshot(
    *, git_sha: str, source_tree_drift: bool = False
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION,
        "git_sha": git_sha,
        "scopes": list(STAGE1_SOURCE_TREE_SCOPES),
        "included_suffixes": sorted(STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES),
        "file_count": 100,
        "worktree_dirty": False,
        "worktree_status_sha256": "0" * 64,
        "source_tree_fingerprint": ("2" if source_tree_drift else "1") * 64,
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def _runtime_contract(
    arm: str,
    *,
    reference_fingerprint: str = "1" * 64,
    add_event: bool = False,
    anchor_weight_override: float | None = None,
    synergy_weight_override: float | None = None,
) -> dict[str, object]:
    weights = {
        "T0": (0.0, 0.0),
        "T1": (0.02, 0.0),
        "T2": (0.0, 0.05),
        "T3": (0.02, 0.05),
        "T4": (0.02, 0.05),
    }
    anchor, synergy = weights[arm]
    if anchor_weight_override is not None:
        anchor = anchor_weight_override
    if synergy_weight_override is not None:
        synergy = synergy_weight_override
    contract: dict[str, object] = {
        "schema_version": "stage1_peasd_lite_runtime_contract_v1",
        "enabled": True,
        "arm": arm,
        "mode": "diagnostics_only" if arm == "T0" else "reward",
        "training_signal_enabled": arm != "T0",
        "action_id": FOREHAND_CLEAR.emg_trial_actions[0],
        "action_index": 0,
        "reference_path": "/evidence/forehand_clear/emg_reference_manifest.json",
        "reference_id": "P002_forehand_clear_v1",
        "reference_fingerprint": reference_fingerprint,
        "array_bundle_sha256": "2" * 64,
        "reference_review_status": "verified",
        "reference_training_enabled": True,
        "trial_qc_review_schema_version": "jidian_trial_qc_review_v1",
        "trial_qc_review_sha256": "9" * 64,
        "mapping_path": "/evidence/forehand_clear/emg_observation_mapping.json",
        "mapping_id": "P002_15_of_16_v1",
        "mapping_sha256": "3" * 64,
        "mapping_review_status": "verified",
        "phase_coordinate": "normalized_trajectory_progress",
        "signal": "mujoco_scalar_activation_state",
        "phase_bin_count": 20,
        "channel_count": 15,
        "synergy_count": 4,
        "ordered_actuator_count": 354,
        "actuator_schema_hash": "4" * 64,
        "runtime_model_hash": "5" * 64,
        "muscle_channel_core_fingerprint": "6" * 64,
        "anchor_loss_spec_fingerprint": "7" * 64,
        "anchor_weight_max": anchor,
        "synergy_weight_max": synergy,
        "anchor_max_penalty_each": 1.0,
        "synergy_max_penalty_each": 1.0,
        "start_update": 0,
        "ramp_updates": 0,
        "tube_kappa": 1.0,
        "huber_delta": 1.0,
        "synergy_shape_weight": 1.0,
        "synergy_intensity_weight": 0.25,
        "synergy_phase_shuffle_offset_bins": 10 if arm == "T4" else 0,
        "synergy_phase_shuffled": arm == "T4",
        "synergy_phase_shift_strategy": "half_cycle_circular" if arm == "T4" else "none",
        "matched_reward_core_fingerprint": (
            "8" * 64 if arm in {"T3", "T4"} else f"{CANONICAL_ARMS.index(arm) + 10:x}" * 64
        ),
    }
    if add_event:
        contract["fake_event_shuffle"] = True
    contract["binding_sha256"] = _canonical_sha256(contract)
    return contract


@cache
def _release_contract() -> dict[str, object]:
    report = validate_action_release(FOREHAND_CLEAR)
    assert report["passed"] is True
    return report


@cache
def _numeric_data_qc_contract() -> dict[str, object]:
    report = inspect_canonical_dataset(
        FOREHAND_CLEAR.dataset_root,
        source_variant=FOREHAND_CLEAR.source_variant,
        cache_variant=FOREHAND_CLEAR.cache_variant,
        action=FOREHAND_CLEAR.slug,
    )
    assert report["clean_passed"] is True
    report = json.loads(json.dumps(report, sort_keys=True, allow_nan=False))
    unsigned: dict[str, object] = {
        "schema_version": "stage1_peasd_numeric_data_qc_contract_v1",
        "action_id": FOREHAND_CLEAR.action_id,
        "action_slug": FOREHAND_CLEAR.slug,
        "source_variant": FOREHAND_CLEAR.source_variant,
        "cache_variant": FOREHAND_CLEAR.cache_variant,
        "report_sha256": _canonical_sha256(report),
        "report": report,
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def _budget_contract(
    arm: str,
    seed: int,
    *,
    updates: int,
    anchor_weight: float | None = None,
    synergy_weight: float | None = None,
) -> dict[str, object]:
    weights = {
        "T0": (0.0, 0.0, "off"),
        "T1": (0.02, 0.0, "reward"),
        "T2": (0.0, 0.05, "reward"),
        "T3": (0.02, 0.05, "reward"),
        "T4": (0.02, 0.05, "reward"),
    }
    anchor, synergy, mode = weights[arm]
    if anchor_weight is not None:
        anchor = anchor_weight
    if synergy_weight is not None:
        synergy = synergy_weight
    total = updates * 100
    unsigned: dict[str, object] = {
        "schema_version": "stage1_peasd_fixed_budget_contract_v1",
        "action_slug": FOREHAND_CLEAR.slug,
        "action_id": FOREHAND_CLEAR.action_id,
        "tube_action_id": FOREHAND_CLEAR.emg_trial_actions[0],
        "arm": arm,
        "seed": seed,
        "canonical_seeds": list(CANONICAL_SEEDS),
        "fresh_optimizer": True,
        "promotion_auto_stop": False,
        "total_timesteps": total,
        "num_updates": updates,
        "num_steps": 1,
        "num_envs": 100,
        "rollout_batch_size": 100,
        "expected_endpoint_update_number": updates,
        "expected_endpoint_global_timestep": total,
        "validation_interval_updates": 1,
        "requested_validation_count": updates,
        "scheduled_validation_count": updates,
        "endpoint_requires_independent_validation": False,
        "expected_history_count": updates,
        # Compatibility with the validator while the production writer and
        # validator converge on the final field spelling.
        "expected_validation_count": updates,
        "emg_curriculum": {
            "mode": mode,
            "anchor_weight_max": anchor,
            "synergy_weight_max": synergy,
            "start_update": 0,
            "ramp_updates": 0,
            "full_weight_update": 0,
            "fully_exposed_within_budget": True,
        },
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def _metrics(
    arm: str,
    *,
    real_loss: float,
    shifted_loss: float,
    degraded: bool,
    activation_saturation_degraded: bool,
    control_effort_degraded: bool,
    return_degraded: bool,
    absolute_quality_failed: bool,
    anchor_weight: float,
    synergy_weight: float,
    anchor_loss_degraded: bool = False,
    anchor_correlation_degraded: bool = False,
) -> dict[str, float]:
    factor = 1.2 if arm == "T3" and degraded else (1.05 if arm in {"T1", "T2", "T3", "T4"} else 1.0)
    values = dict.fromkeys(STAGE1_PEASD_VALIDATION_METRIC_KEYS, 0.0)
    values.update(
        {
            "val_mean_episode_return": 1.0 / factor,
            "val_early_termination_rate": 0.01,
            "val_frame_coverage": 0.99,
            "val_err_root_xyz": 0.02 * factor,
            "val_err_root_yaw": 0.01 * factor,
            "val_err_joint_pos": 0.03 * factor,
            "val_err_joint_vel": 0.04 * factor,
            "val_err_rpos": 0.07 * factor,
            "val_activation_energy": 0.20 * factor,
            "val_action_saturation_fraction": 0.01,
            "val_action_rate_mean_square": 0.02 * factor,
            "val_activation_rate_mean_square": 0.03 * factor,
            "val_emg_anchor_loss": (
                0.35
                if arm == "T3" and anchor_loss_degraded
                else (0.30 if arm == "T0" else 0.20)
            ),
            "val_emg_anchor_violation_fraction": 0.30 if arm == "T0" else 0.20,
            "val_emg_anchor_mean_abs_deviation": 0.15 if arm == "T0" else 0.10,
            "val_emg_anchor_max_abs_deviation": 0.40 if arm == "T0" else 0.30,
            "val_emg_anchor_correlation": (
                0.60
                if arm == "T3" and anchor_correlation_degraded
                else (0.70 if arm == "T0" else 0.80)
            ),
            "val_emg_anchor_valid_channel_fraction": 1.0,
            "val_emg_synergy_loss": shifted_loss if arm == "T4" else real_loss,
            "val_emg_synergy_real_reference_loss": shifted_loss if arm == "T4" else real_loss,
            "val_emg_synergy_real_reference_shape_cosine": 0.8,
            "val_emg_synergy_real_reference_intensity": 0.4,
            "val_emg_anchor_weight": anchor_weight,
            "val_emg_synergy_weight": synergy_weight,
            "val_emg_curriculum_factor_anchor": 1.0 if anchor_weight > 0.0 else 0.0,
            "val_emg_curriculum_factor_synergy": 1.0 if synergy_weight > 0.0 else 0.0,
            "val_emg_consistency_penalty_masked_fraction": 0.0,
        }
    )
    if arm == "T3" and activation_saturation_degraded:
        values["val_activation_saturation_fraction"] = 0.20
    if arm == "T3" and control_effort_degraded:
        values["val_activation_energy"] = 0.50
        values["val_action_rate_mean_square"] = 0.10
        values["val_activation_rate_mean_square"] = 0.10
    if arm == "T3" and return_degraded:
        values["val_mean_episode_return"] = -100.0
    if absolute_quality_failed:
        values.update(
            {
                "val_early_termination_rate": 0.20,
                "val_frame_coverage": 0.50,
                "val_err_rpos": 0.20,
                "val_action_saturation_fraction": 0.20,
                "val_activation_energy": 0.70,
            }
        )
    return values


def _write_checkpoint(run_root: Path, *, update: int, total_updates: int) -> dict[str, object]:
    checkpoint = run_root / f"checkpoint_{update}"
    metadata = checkpoint / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "metadata").write_text(
        json.dumps(
            {
                "update_number": update,
                "global_timestep": update * 100,
                "target_global_timestep": total_updates * 100,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "state.bin").write_bytes(f"state-{update}".encode())
    return checkpoint_identity(checkpoint)


def _sealed_run(
    tmp_path: Path,
    *,
    arm: str,
    seed: int,
    index: int,
    real_loss: float,
    shifted_loss: float,
    degraded: bool,
    t4_reference_fingerprint: str,
    add_t4_event: bool,
    reward_drift: bool,
    endpoint_updates: int,
    activation_saturation_degraded: bool,
    control_effort_degraded: bool,
    return_degraded: bool,
    absolute_quality_failed: bool,
    git_sha: str,
    source_tree_drift: bool,
    anchor_loss_degraded: bool,
    anchor_correlation_degraded: bool,
    t1_anchor_weight: float | None,
    t2_synergy_weight: float | None,
) -> dict[str, object]:
    run_id = f"forehand_clear_stage1_peasd_lite_v1_{arm.lower()}_s{seed}"
    config_hash = f"{index + 1:012x}"
    run_root = tmp_path / f"{arm.lower()}_s{seed}"
    run_root.mkdir(parents=True)
    runtime = _runtime_contract(
        arm,
        reference_fingerprint=(t4_reference_fingerprint if arm == "T4" else "1" * 64),
        add_event=add_t4_event and arm == "T4",
        anchor_weight_override=(t1_anchor_weight if arm == "T1" else None),
        synergy_weight_override=(t2_synergy_weight if arm == "T2" else None),
    )
    budget = _budget_contract(
        arm,
        seed,
        updates=endpoint_updates,
        anchor_weight=float(runtime["anchor_weight_max"]),
        synergy_weight=float(runtime["synergy_weight_max"]),
    )
    experiment: dict[str, object] = {
        "run_id": run_id,
        "checkpoint_dir": str(run_root),
        "validation_video_dir": str(run_root / "videos"),
        "auto_resume": False,
        "resume_from": None,
        "reset_optimizer_on_resume": True,
        "n_seeds": 1,
        "seeds": [seed],
        "total_timesteps": endpoint_updates * 100,
        "num_envs": 100,
        "ppo_config": {"num_steps": 1},
        "save_checkpoints": True,
        "checkpoints_on_validation": True,
        "promotion": {
            "auto_stop": False,
            "max_early_termination_rate": 0.05,
            "min_frame_coverage": 0.95,
            "max_relative_site_position_error_m": 0.09,
            "max_action_saturation_fraction": 0.05,
            "max_activation_energy": 0.35,
        },
        "stage1_peasd": {
            "schema_version": "stage1_peasd_lite_matched_arm_v1",
            "arm": arm,
            "action_id": FOREHAND_CLEAR.action_id,
            "canonical_seeds": list(CANONICAL_SEEDS),
            "fresh_optimizer_required": True,
            "parent_initialization_checkpoint": None,
            "control_kind": arm.lower(),
        },
        "stage1_peasd_fixed_budget_contract": budget,
        "stage1_peasd_action_release_contract": _release_contract(),
        "stage1_peasd_numeric_data_qc_contract": _numeric_data_qc_contract(),
        "task_factory": {
            "params": {
                "amass_dataset_conf": {
                    "rel_dataset_path": list(FOREHAND_CLEAR.train_motion_paths),
                }
            }
        },
        "validation": {
            "active": True,
            "deterministic": True,
            "start_from_beginning": True,
            "num": endpoint_updates,
            "amass_dataset_conf": {
                "rel_dataset_path": list(FOREHAND_CLEAR.val_motion_paths),
            },
        },
        "env_params": {
            "reward_params": {
                "qpos_w_sum": 9.0 if reward_drift else 1.0,
                "emg_consistency": {
                    "enabled": arm != "T0",
                    "arm": arm,
                    "mode": "off" if arm == "T0" else "reward",
                    "action_id": FOREHAND_CLEAR.emg_trial_actions[0],
                    "reference_cache": None if arm == "T0" else "/tube",
                    "mapping_path": None,
                    "anchor_weight_max": runtime["anchor_weight_max"],
                    "synergy_weight_max": runtime["synergy_weight_max"],
                    "start_update": 0,
                    "ramp_updates": 0,
                    "tube_kappa": 1.0,
                    "huber_delta": 1.0,
                    "synergy_shape_weight": 1.0,
                    "synergy_intensity_weight": 0.25,
                    "synergy_phase_shuffle_offset_bins": runtime[
                        "synergy_phase_shuffle_offset_bins"
                    ],
                },
            }
        },
    }
    if arm != "T0":
        experiment["emg_consistency_runtime_contract"] = runtime
    source_snapshot = _source_tree_snapshot(
        git_sha=git_sha,
        source_tree_drift=source_tree_drift,
    )
    manifest = {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "source_tree_snapshot": source_snapshot,
        "experiment_config": experiment,
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    final_metrics = _metrics(
        arm,
        real_loss=real_loss,
        shifted_loss=shifted_loss,
        degraded=degraded,
        activation_saturation_degraded=activation_saturation_degraded,
        control_effort_degraded=control_effort_degraded,
        return_degraded=return_degraded,
        absolute_quality_failed=absolute_quality_failed,
        anchor_weight=float(runtime["anchor_weight_max"]),
        synergy_weight=float(runtime["synergy_weight_max"]),
        anchor_loss_degraded=anchor_loss_degraded,
        anchor_correlation_degraded=anchor_correlation_degraded,
    )
    history_metrics = dict(final_metrics)
    if arm == "T0":
        for key in history_metrics:
            if key.startswith("val_emg_"):
                history_metrics[key] = 0.0
    provenance = {
        "semantics": "evaluate_all_once_per_heldout_v1",
        "deterministic_policy": True,
        "start_frame": 0,
        "auto_reset_samples_included": False,
        "run_stats_frozen": True,
        "heldout_trajectory_count": len(FOREHAND_CLEAR.val_motion_paths),
        "eval_seed": 17,
    }
    final_identity = None
    history_path = None
    training_metric_values = dict.fromkeys(
        STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS, 0.0
    )
    training_metric_values.update(
        {
            "emg_anchor_weight": float(runtime["anchor_weight_max"]),
            "emg_synergy_weight": float(runtime["synergy_weight_max"]),
            "emg_curriculum_factor_anchor": (
                1.0 if float(runtime["anchor_weight_max"]) > 0.0 else 0.0
            ),
            "emg_curriculum_factor_synergy": (
                1.0 if float(runtime["synergy_weight_max"]) > 0.0 else 0.0
            ),
            "emg_anchor_valid_channel_fraction": 1.0,
            "emg_synergy_real_reference_intensity": 0.4,
        }
    )
    treatment_samples = [
        build_stage1_peasd_training_treatment_sample(
            experiment=experiment,
            rollout_update_index=int(target["rollout_update_index"]),
            training_metrics=training_metric_values,
        )
        for target in stage1_peasd_training_treatment_targets(experiment)
    ]
    with patch(
        "musclemimic.runner.stage1_peasd_validation.stage1_source_tree_snapshot",
        return_value=source_snapshot,
    ):
        for update in range(1, endpoint_updates + 1):
            identity = _write_checkpoint(run_root, update=update, total_updates=endpoint_updates)
            _, _, history_path = append_stage1_peasd_validation_history(
                checkpoint_identity_payload=identity,
                validation_provenance=provenance,
                metrics=history_metrics,
                training_treatment_samples=[
                    sample
                    for sample in treatment_samples
                    if int(sample["completed_update"]) <= update
                ],
            )
            final_identity = identity
        assert final_identity is not None and history_path is not None
        evidence = build_stage1_peasd_validation_evidence(
            checkpoint_identity_payload=final_identity,
            validation_provenance=provenance,
            metrics=final_metrics,
            validation_history=history_path,
            evaluation_runtime_contract=(runtime if arm == "T0" else None),
        )
    versioned, _latest = write_stage1_peasd_validation_evidence(evidence)
    return {
        "seed": seed,
        "validation_metrics_path": str(versioned),
        "validation_metrics_content_sha256": _sha(versioned),
    }


def _evidence_index(
    tmp_path: Path,
    *,
    real_loss: float = 0.70,
    shifted_loss: float = 1.0,
    degraded: bool = False,
    t4_reference_fingerprint: str = "1" * 64,
    add_t4_event: bool = False,
    reward_drift_arm: str | None = None,
    endpoint_drift_arm: str | None = None,
    activation_saturation_degraded: bool = False,
    control_effort_degraded: bool = False,
    return_degraded: bool = False,
    absolute_quality_failed: bool = False,
    git_drift_arm: str | None = None,
    source_tree_drift_arm: str | None = None,
    anchor_loss_degraded: bool = False,
    anchor_correlation_degraded: bool = False,
    t1_anchor_weight: float | None = None,
    t2_synergy_weight: float | None = None,
) -> Path:
    runs = {arm: [] for arm in CANONICAL_ARMS}
    index = 0
    for arm in CANONICAL_ARMS:
        for seed in CANONICAL_SEEDS:
            runs[arm].append(
                _sealed_run(
                    tmp_path,
                    arm=arm,
                    seed=seed,
                    index=index,
                    real_loss=real_loss,
                    shifted_loss=shifted_loss,
                    degraded=degraded,
                    t4_reference_fingerprint=t4_reference_fingerprint,
                    add_t4_event=add_t4_event,
                    reward_drift=arm == reward_drift_arm,
                    endpoint_updates=4 if arm == endpoint_drift_arm else 3,
                    activation_saturation_degraded=activation_saturation_degraded,
                    control_effort_degraded=control_effort_degraded,
                    return_degraded=return_degraded,
                    absolute_quality_failed=absolute_quality_failed,
                    git_sha=(
                        "b" * 40 if arm == git_drift_arm else "a" * 40
                    ),
                    source_tree_drift=arm == source_tree_drift_arm,
                    anchor_loss_degraded=anchor_loss_degraded,
                    anchor_correlation_degraded=anchor_correlation_degraded,
                    t1_anchor_weight=t1_anchor_weight,
                    t2_synergy_weight=t2_synergy_weight,
                )
            )
            index += 1
    payload = build_pairwise_metrics_document(
        action=FOREHAND_CLEAR.slug,
        runs=runs,
        notes="sealed synthetic evidence",
    )
    path = tmp_path / "pairwise_evidence_index.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _blind_review(root: Path, candidate: dict[str, object]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    video_dir = root / "source_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    endpoint_clips = []
    for index, motion in enumerate(FOREHAND_CLEAR.val_motion_paths):
        artifact = video_dir / f"source_motion_{index}.mp4"
        artifact.write_bytes(f"sealed-video-{index}".encode())
        artifact_sha = _sha(artifact)
        endpoint_clips.append(
            {
                "motion": motion,
                "artifact": str(artifact),
                "artifact_content_sha256": artifact_sha,
            }
        )
    endpoint_unsigned = {
        "schema_version": ENDPOINT_REVIEW_SET_SCHEMA_VERSION,
        "review_kind": STAGE1_REVIEW_KIND,
        "candidate": candidate,
        "deterministic_policy": True,
        "run_stats_frozen": True,
        "validation_motion_paths": list(FOREHAND_CLEAR.val_motion_paths),
        "clips": endpoint_clips,
    }
    endpoint = {
        **endpoint_unsigned,
        "binding_sha256": _canonical_sha256(endpoint_unsigned),
    }
    endpoint_path = root / "stage1_endpoint_review_set.json"
    endpoint_path.write_text(
        json.dumps(endpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blind = build_stage1_peasd_blind_review_package(
        endpoint_review_set=endpoint_path,
        output_dir=root,
        expected_candidate=candidate,
        expected_motions=list(FOREHAND_CLEAR.val_motion_paths),
    )
    review = json.loads(blind["review_template"].read_text(encoding="utf-8"))
    review["reviewer_id"] = "reviewer-fixture"
    review["passed"] = True
    for row in review["clips"]:
        row["major_swing_complete"] = True
        row["root_tracking_spike_free"] = True
        row["right_hand_tracking_spike_free"] = True
        row["passed"] = True
        row["notes"] = "opaque held-out clip passed"
    review_path = blind["review_submission"]
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **blind,
        "review": review_path,
        "endpoint_review_set": endpoint_path,
    }


def test_pairwise_gate_binds_all_arms_checkpoints_histories_and_seed_statistics(tmp_path) -> None:
    # T1/T2 are legitimate distinct treatments; only T3/T4 must share the
    # matched reward-core fingerprint.
    assert (
        _runtime_contract("T1")["matched_reward_core_fingerprint"]
        != _runtime_contract("T2")["matched_reward_core_fingerprint"]
    )
    report = evaluate_pairwise_promotion(_evidence_index(tmp_path), action=FOREHAND_CLEAR.slug)

    assert report["passed"] is True
    assert report["source_binding"]["arms"] == list(CANONICAL_ARMS)
    assert report["source_binding"]["seeds"] == list(CANONICAL_SEEDS)
    assert report["source_binding"]["matched_config_core_sha256"]
    assert report["source_binding"]["matched_budget_core_sha256"]
    assert report["source_binding"]["git_sha"] == "a" * 40
    assert report["source_binding"]["source_tree_snapshot"]["source_tree_fingerprint"] == (
        "1" * 64
    )
    assert report["source_binding"]["source_tree_snapshot_binding_sha256"] == report[
        "source_binding"
    ]["source_tree_snapshot"]["binding_sha256"]
    assert report["source_binding"]["absolute_stage1_promotion_thresholds"] == {
        "max_early_termination_rate": 0.05,
        "min_frame_coverage": 0.95,
        "max_relative_site_position_error_m": 0.09,
        "max_action_saturation_fraction": 0.05,
        "max_activation_energy": 0.35,
    }
    assert report["source_binding"]["tube_reference_fingerprint"] == "1" * 64
    assert report["statistics"]["n_seeds"] == 3
    assert report["statistics"]["primary_synergy"]["confidence_interval"]["scope"] == (
        "seed_level_small_sample_not_population"
    )
    activation = report["statistics"]["measured_activation_vs_t0"]
    assert set(activation) == {
        "val_emg_anchor_loss",
        "val_emg_anchor_violation_fraction",
        "val_emg_anchor_mean_abs_deviation",
        "val_emg_anchor_max_abs_deviation",
        "val_emg_anchor_correlation",
    }
    assert activation["val_emg_anchor_loss"]["values_by_seed"] == {
        "0": pytest.approx(0.10),
        "1": pytest.approx(0.10),
        "2": pytest.approx(0.10),
    }
    assert activation["val_emg_anchor_loss"]["mean"] == pytest.approx(0.10)
    activation_seed_checks = [
        check
        for check in report["checks"]
        if check["name"].endswith("measured_activation_anchor_improves_vs_t0")
    ]
    assert len(activation_seed_checks) == len(CANONICAL_SEEDS)
    assert all(check["passed"] for check in activation_seed_checks)
    assert next(
        check
        for check in report["checks"]
        if check["name"] == "aggregate_measured_activation_anchor_direction"
    )["passed"] is True
    assert set(report["statistics"]["t1_t2_decomposition_vs_t0"]) == {"T1", "T2"}
    assert report["binding_sha256"] == _canonical_sha256(
        {key: value for key, value in report.items() if key != "binding_sha256"}
    )


def test_evidence_index_producer_requires_and_revalidates_exact_15(tmp_path) -> None:
    source = _evidence_index(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    selectors = [
        f"{arm}:{row['seed']}:{row['validation_metrics_path']}"
        for arm in CANONICAL_ARMS
        for row in payload["runs"][arm]
    ]
    rebuilt = build_pairwise_evidence_index(
        action=FOREHAND_CLEAR.slug,
        evidence_selectors=selectors,
        notes="producer regression",
    )
    assert set(rebuilt["runs"]) == set(CANONICAL_ARMS)
    assert sum(len(rows) for rows in rebuilt["runs"].values()) == 15
    with pytest.raises(ValueError, match="exactly T0--T4 x seeds"):
        build_pairwise_evidence_index(
            action=FOREHAND_CLEAR.slug,
            evidence_selectors=selectors[:-1],
        )


def test_pairwise_gate_fails_claim_when_real_does_not_beat_half_cycle_control(tmp_path) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, real_loss=0.99, shifted_loss=1.0),
        action=FOREHAND_CLEAR.slug,
    )
    assert report["passed"] is False
    assert any("real_synergy" in check["name"] and not check["passed"] for check in report["checks"])


def test_pairwise_gate_requires_measured_activation_improvement_for_every_seed(
    tmp_path,
) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, anchor_loss_degraded=True),
        action=FOREHAND_CLEAR.slug,
    )

    assert report["passed"] is False
    seed_checks = [
        check
        for check in report["checks"]
        if check["name"].endswith("measured_activation_anchor_improves_vs_t0")
    ]
    assert len(seed_checks) == len(CANONICAL_SEEDS)
    assert all(check["passed"] is False for check in seed_checks)
    aggregate = next(
        check
        for check in report["checks"]
        if check["name"] == "aggregate_measured_activation_anchor_direction"
    )
    assert aggregate["passed"] is False
    assert aggregate["value"] < 0.0


def test_pairwise_gate_rejects_measured_activation_correlation_degradation(
    tmp_path,
) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, anchor_correlation_degraded=True),
        action=FOREHAND_CLEAR.slug,
    )

    assert report["passed"] is False
    seed_checks = [
        check
        for check in report["checks"]
        if check["name"].endswith("anchor_correlation_non_degraded_vs_t0")
        and check["name"].startswith("seed_")
    ]
    assert len(seed_checks) == len(CANONICAL_SEEDS)
    assert all(check["passed"] is False for check in seed_checks)
    aggregate = next(
        check
        for check in report["checks"]
        if check["name"]
        == "aggregate_anchor_correlation_non_degraded_vs_t0"
    )
    assert aggregate["passed"] is False
    assert aggregate["value"] < 0.0


def test_pairwise_gate_fails_when_root_joint_and_keypoint_tracking_degrade(tmp_path) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, degraded=True), action=FOREHAND_CLEAR.slug
    )
    assert report["passed"] is False
    failed_metrics = {check["metric"] for check in report["checks"] if not check["passed"]}
    assert {"val_err_root_xyz", "val_err_joint_pos", "val_err_rpos"}.issubset(failed_metrics)


def test_pairwise_gate_rejects_common_absolute_stage1_failure(tmp_path) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, absolute_quality_failed=True),
        action=FOREHAND_CLEAR.slug,
    )
    assert report["passed"] is False
    absolute = [
        check
        for check in report["checks"]
        if "absolute_stage1" in check["name"]
    ]
    assert len(absolute) == len(CANONICAL_ARMS) * len(CANONICAL_SEEDS) * 5
    assert all(check["passed"] is False for check in absolute)
    assert all(
        check["passed"] is True
        for check in report["checks"]
        if check["name"].endswith("non_degraded_vs_t0")
    )


@pytest.mark.parametrize(
    ("fixture_option", "expected_metric"),
    [
        ("activation_saturation_degraded", "val_activation_saturation_fraction"),
        ("control_effort_degraded", "val_activation_energy"),
    ],
)
def test_pairwise_gate_hard_fails_muscle_saturation_and_control_effort(
    tmp_path, fixture_option: str, expected_metric: str
) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, **{fixture_option: True}),
        action=FOREHAND_CLEAR.slug,
    )
    assert report["passed"] is False
    assert any(
        check["metric"] == expected_metric and check["passed"] is False
        for check in report["checks"]
    )


def test_pairwise_gate_reports_but_does_not_compare_cross_reward_returns(tmp_path) -> None:
    report = evaluate_pairwise_promotion(
        _evidence_index(tmp_path, return_degraded=True),
        action=FOREHAND_CLEAR.slug,
    )
    assert report["passed"] is True
    assert not any(check["metric"] == "val_mean_episode_return" for check in report["checks"])
    diagnostics = report["statistics"]["noncomparable_reward_diagnostics_vs_t0"]
    assert diagnostics["val_mean_episode_return"]["mean"] < 0.0


def test_pairwise_gate_rejects_tube_drift_and_fake_event_control(tmp_path) -> None:
    with pytest.raises(ValueError, match="tube/loss/model contract"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "tube_drift", t4_reference_fingerprint="a" * 64),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match="must not fabricate/shuffle events"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "fake_event", add_t4_event=True),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match="single-leg decomposition"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "t1_weight_drift", t1_anchor_weight=0.001),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match="single-leg decomposition"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "t2_weight_drift", t2_synergy_weight=0.5),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match="one non-empty git_sha"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "git_drift", git_drift_arm="T2"),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match="one source-tree snapshot"):
        evaluate_pairwise_promotion(
            _evidence_index(
                tmp_path / "source_tree_drift",
                source_tree_drift_arm="T2",
            ),
            action=FOREHAND_CLEAR.slug,
        )


def test_delivered_treatment_gate_rejects_final_reward_floor_masking() -> None:
    runtime = _runtime_contract("T3")
    metrics = _metrics(
        "T3",
        real_loss=0.7,
        shifted_loss=1.0,
        degraded=False,
        activation_saturation_degraded=False,
        control_effort_degraded=False,
        return_degraded=False,
        absolute_quality_failed=False,
        anchor_weight=float(runtime["anchor_weight_max"]),
        synergy_weight=float(runtime["synergy_weight_max"]),
    )
    metrics["val_emg_consistency_final_reward_masked_fraction"] = 1.0
    with pytest.raises(ValueError, match="final reward floor"):
        _validate_delivered_treatment(metrics, runtime=runtime, arm="T3", seed=0)


def test_pairwise_gate_rejects_tampered_evidence_nonemg_reward_and_endpoint_drift(tmp_path) -> None:
    index_path = _evidence_index(tmp_path / "tamper")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    evidence = Path(payload["runs"]["T3"][0]["validation_metrics_path"])
    evidence.write_text(evidence.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence content changed"):
        evaluate_pairwise_promotion(index_path, action=FOREHAND_CLEAR.slug)

    with pytest.raises(ValueError, match="matched config core"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "reward_drift", reward_drift_arm="T4"),
            action=FOREHAND_CLEAR.slug,
        )
    with pytest.raises(ValueError, match=r"fixed-budget endpoint|fixed-budget/validation schedule"):
        evaluate_pairwise_promotion(
            _evidence_index(tmp_path / "endpoint_drift", endpoint_drift_arm="T2"),
            action=FOREHAND_CLEAR.slug,
        )


def test_pre_registered_t3_seed0_promotion_requires_opaque_sealed_visual_review(
    monkeypatch, tmp_path
) -> None:
    index_path = _evidence_index(tmp_path)
    gate = evaluate_pairwise_promotion(index_path, action=FOREHAND_CLEAR.slug)
    gate_path = tmp_path / "pairwise_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = gate["source_binding"]["validation_evidence_by_arm_seed"]["T3"]["0"]
    candidate = selected["checkpoint_identity"]
    blind = _blind_review(tmp_path / "blind_qc", candidate)
    review_path = blind["review"]
    mapping_path = blind["private_mapping"]

    promotion = build_stage1_peasd_teacher_promotion(
        pairwise_gate=gate_path,
        blind_review=review_path,
        blind_mapping=mapping_path,
        action=FOREHAND_CLEAR.slug,
    )
    promotion_path = tmp_path / "teacher_promotion.json"
    promotion_path.write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validated = validate_stage1_peasd_teacher_promotion(
        promotion_path,
        expected_action=FOREHAND_CLEAR.slug,
        expected_checkpoint=candidate["checkpoint_path"],
    )
    assert validated["teacher_arm"] == "T3"
    assert validated["teacher_seed"] == 0
    assert validated["selection_rule"] == "pre_registered_t3_seed_0_v1"
    assert validated["visual_review"]["schema_version"] == (
        "stage1_peasd_blind_visual_evidence_v1"
    )
    assert "blinding" not in validated["visual_review"]

    mapping = validate_stage1_peasd_blind_mapping(
        mapping_path,
        expected_candidate=candidate,
        expected_motions=list(FOREHAND_CLEAR.val_motion_paths),
    )
    reconstructed = validate_stage1_peasd_blind_review(
        review_path,
        private_mapping=mapping_path,
        expected_candidate=candidate,
        expected_motions=list(FOREHAND_CLEAR.val_motion_paths),
    )
    assert reconstructed["validation_report"]["passed"] is True
    assert [row["source_motion"] for row in mapping["clips"]] == list(
        FOREHAND_CLEAR.val_motion_paths
    )

    reviewer_visible = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            blind["package_manifest"],
            blind["review_template"],
            review_path,
        )
    )
    reviewer_visible_lower = reviewer_visible.lower()
    for forbidden in (
        "checkpoint",
        "run_id",
        '"arm"',
        '"seed"',
        '"motion"',
        '"source"',
        "t3",
        "seed0",
        "s0",
        str(candidate["run_id"]).lower(),
        str(candidate["checkpoint_path"]).lower(),
        *(str(path).lower() for path in FOREHAND_CLEAR.val_motion_paths),
    ):
        assert forbidden not in reviewer_visible_lower
    package = json.loads(blind["package_manifest"].read_text(encoding="utf-8"))
    assert all(
        row["artifact"] == f"clips/{row['opaque_clip_id']}.mp4"
        for row in package["clips"]
    )
    child = OmegaConf.create(
        {
            "resume_from": candidate["checkpoint_path"],
            "parent_checkpoint_lineage": {
                "required": True,
                "role": "stage1_promoted",
                "promotion_manifest": str(promotion_path),
            },
        }
    )
    lineage = bind_explicit_parent_checkpoint(child, launch_dir=tmp_path)
    assert lineage is not None
    assert lineage["promotion"]["evidence_kind"] == (
        "verified_stage1_peasd_promotion_v1"
    )
    assert lineage["promotion"]["artifact_content_sha256"] == _sha(promotion_path)
    first_video = blind["package_manifest"].parent / package["clips"][0]["artifact"]
    original_video = first_video.read_bytes()
    first_video.write_bytes(original_video + b"tampered")
    with pytest.raises(ValueError, match=r"package video bytes changed|video bytes"):
        validate_stage1_peasd_teacher_promotion(
            promotion_path,
            expected_checkpoint=candidate["checkpoint_path"],
        )
    first_video.write_bytes(original_video)
    expected_source = {
        "reference_id": validated["emg_reference_binding"]["reference_id"],
        "reference_fingerprint": "f" * 64,
        "array_bundle_sha256": validated["emg_reference_binding"][
            "array_bundle_sha256"
        ],
        "mapping_sha256": validated["emg_reference_binding"]["mapping_sha256"],
        "trial_qc_review_schema_version": validated["emg_reference_binding"][
            "trial_qc_review_schema_version"
        ],
        "trial_qc_review_sha256": validated["emg_reference_binding"][
            "trial_qc_review_sha256"
        ],
        "phase_bin_count": validated["emg_reference_binding"]["phase_bin_count"],
    }
    monkeypatch.setattr(
        "musclemimic.badminton.stage1_peasd_gate.build_verified_tube_gate",
        lambda *_args, **_kwargs: {"source": expected_source},
    )
    with pytest.raises(ValueError, match="tube differs from downstream"):
        validate_stage1_peasd_teacher_promotion(
            promotion_path,
            expected_action=FOREHAND_CLEAR.slug,
            expected_tube=tmp_path / "different_tube",
        )

    original_review_bytes = review_path.read_bytes()
    review = json.loads(original_review_bytes)
    review["clips"] = review["clips"][:-1]
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="missing one or more opaque clips"):
        build_stage1_peasd_teacher_promotion(
            pairwise_gate=gate_path,
            blind_review=review_path,
            blind_mapping=mapping_path,
            action=FOREHAND_CLEAR.slug,
        )
    review_path.write_bytes(original_review_bytes)

    leaking_review = json.loads(original_review_bytes)
    leaking_review["clips"][0]["notes"] = "T3 seed0 checkpoint accepted"
    review_path.write_text(json.dumps(leaking_review), encoding="utf-8")
    with pytest.raises(ValueError, match=r"identity|leaks"):
        build_stage1_peasd_teacher_promotion(
            pairwise_gate=gate_path,
            blind_review=review_path,
            blind_mapping=mapping_path,
            action=FOREHAND_CLEAR.slug,
        )
    review_path.write_bytes(original_review_bytes)

    original_mapping_bytes = mapping_path.read_bytes()
    tampered_mapping = json.loads(original_mapping_bytes)
    tampered_mapping["clips"][0]["source_motion"] = "foreign_motion"
    mapping_path.write_text(json.dumps(tampered_mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="private mapping binding mismatch"):
        build_stage1_peasd_teacher_promotion(
            pairwise_gate=gate_path,
            blind_review=review_path,
            blind_mapping=mapping_path,
            action=FOREHAND_CLEAR.slug,
        )
    mapping_path.write_bytes(original_mapping_bytes)

    legacy_path = review_path.parent / "review.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": "forehand_clear_visual_review_v2",
                "blinding": {
                    "reviewer_blinded_to_arm": True,
                    "reviewer_blinded_to_seed": True,
                },
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"opaque blind review schema|leaks identity"):
        build_stage1_peasd_teacher_promotion(
            pairwise_gate=gate_path,
            blind_review=legacy_path,
            blind_mapping=mapping_path,
            action=FOREHAND_CLEAR.slug,
        )


def test_gate_accepts_fields_from_real_preflight_contract_builder(monkeypatch, tmp_path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    mapping_path = reference / "emg_observation_mapping.json"
    mapping_path.write_text("{}", encoding="utf-8")
    config = EmgConsistencyConfig(
        arm="T3",
        mode="reward",
        action_id=FOREHAND_CLEAR.emg_trial_actions[0],
        reference_cache=reference,
        mapping_path=None,
        expected_reference_fingerprint=None,
        expected_mapping_sha256=None,
        anchor_weight_max=0.02,
        synergy_weight_max=0.05,
        start_update=0,
        ramp_updates=0,
        tube_kappa=1.0,
        huber_delta=1.0,
        anchor_max_penalty_each=1.0,
        synergy_max_penalty_each=1.0,
        synergy_shape_weight=1.0,
        synergy_intensity_weight=0.25,
        synergy_phase_shuffle_offset_bins=0,
    )
    tube = SimpleNamespace(
        provenance={
            "trial_qc_review": {
                "schema_version": "jidian_trial_qc_review_v1",
                "review_sha256": "9" * 64,
            }
        },
        reference_id="P002_forehand_clear_v1",
        reference_fingerprint="1" * 64,
        array_bundle_sha256="2" * 64,
        review_status="verified",
        training_enabled=True,
        phase_bin_count=20,
        channel_count=15,
        synergy_count=4,
        normalization_binding={
            "schema_version": "emg_dual_track_normalization_v1",
            "audit_normalization": "percent_mvc_unclipped",
            "model_normalization": "train_p99_per_channel",
            "actions": [
                {
                    "training_cohort_sha256": "a" * 64,
                    "channels": [
                        {"mvc_quality": "good", "task_p99_over_mvc": 1.0}
                        for _index in range(15)
                    ],
                }
            ],
        },
        amplitude_confidence=np.ones((1, 15), dtype=np.float32),
        action_index=lambda _action: 0,
    )
    bundle = EmgReferenceBundle(
        root=reference,
        tube=tube,
        mapping={"mapping_id": "P002_15_of_16_v1", "review_status": "verified"},
        mapping_path=mapping_path,
        mapping_sha256="3" * 64,
    )
    monkeypatch.setattr(
        "musclemimic.physiology.emg_consistency_runtime.validate_emg_consistency_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "musclemimic.physiology.emg_consistency_runtime.load_verified_emg_reference_bundle",
        lambda _config: bundle,
    )
    preflight = build_emg_consistency_preflight_contract({}, base_dir=tmp_path)
    assert preflight is not None
    runtime = {
        **{key: value for key, value in preflight.items() if key != "binding_sha256"},
        "schema_version": "stage1_peasd_lite_runtime_contract_v1",
        "ordered_actuator_count": 354,
        "actuator_schema_hash": "4" * 64,
        "runtime_model_hash": "5" * 64,
        "muscle_channel_core_fingerprint": "6" * 64,
        "anchor_loss_spec_fingerprint": "7" * 64,
        "matched_reward_core_fingerprint": "8" * 64,
    }
    runtime["binding_sha256"] = _canonical_sha256(runtime)
    validated = _validate_runtime_binding(runtime, arm="T3", spec=FOREHAND_CLEAR)
    assert validated["reference_review_status"] == "verified"
    assert validated["trial_qc_review_sha256"] == "9" * 64


def test_provisional_tube_cannot_pass_stage1_preflight(tmp_path) -> None:
    channels = ("c0", "c1", "c2", "c3")
    envelopes = np.full((6, 40, len(channels)), 0.25, dtype=np.float64)
    normalization_binding = build_emg_dual_track_normalization(
        action_samples={
            FOREHAND_CLEAR.emg_trial_actions[0]: [
                envelopes[index] for index in range(envelopes.shape[0])
            ]
        },
        channel_names=channels,
        training_cohorts={
            FOREHAND_CLEAR.emg_trial_actions[0]: [
                {
                    "trial_id": f"trial_{index:03d}",
                    "mvc_normalized_emg_sha256": f"{index + 1:064x}",
                }
                for index in range(envelopes.shape[0])
            ]
        },
        mvc_final_reference_mv=np.ones(len(channels)),
        mvc_reference_binding={
            "path": "/controlled/preprocessing_mvc_reference.json",
            "sha256": "f" * 64,
            "scope": "participant",
            "algorithm": "controlled fixture MVC",
        },
    )
    tube = build_phase_reference_tube(
        reference_id="provisional_forehand_clear",
        action_envelopes={
            FOREHAND_CLEAR.emg_trial_actions[0]: envelopes
        },
        channel_names=channels,
        synergy_basis=np.asarray(
            [[1.0, 0.0], [0.8, 0.1], [0.1, 0.9], [0.0, 0.7]], dtype=np.float64
        ),
        mapping_binding={
            "mapping_id": "fixture",
            "mapping_sha256": "a" * 64,
            "mapping_review_status": "verified",
            "acquired_channel_count": 4,
            "comparable_channel_count": 4,
            "actuator_schema_hash": "b" * 64,
        },
        synergy_binding={
            "basis_id": "fixture_k2",
            "basis_sha256": "c" * 64,
            "synergy_count": 2,
            "channel_normalization": "unit_variance_per_channel",
            "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
            "projection_ridge": EMG_SYNERGY_RIDGE,
        },
        normalization_binding=normalization_binding,
        provenance={
            "subject": "fixture",
            "session": "fixture",
            "normalization": "mvc_percent",
            "review_evidence": [],
        },
        phase_bin_count=20,
    )
    root = tmp_path / "tube"
    save_emg_phase_reference_tube(tube, root)

    with pytest.raises(ValueError, match="mapping review must complete"):
        build_verified_tube_gate(root, action=FOREHAND_CLEAR.slug)
