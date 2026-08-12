from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from environment.overall_environment.src.train_incoming_hit_mjx import (
    _resume_action_prior_source,
    _training_iteration_budget,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import train_gpu
from musclemimic.badminton.stage3_reachability_release import (
    CORRECTION_DATASET_SCHEMA,
    REACHABILITY_RELEASE_SCHEMA,
    STATIC_PPO_ENTRY_SCHEMA,
    attach_static_ppo_entry_to_prerequisites,
    build_stage3_reachability_release,
    build_successful_correction_dataset_manifest,
    canonical_cem_launch_command,
    canonical_short_bc_launch_command,
    validate_post_static_ppo_continuation,
    validate_stage3_reachability_release,
    validate_static_ppo_entry,
    validate_static_ppo_prerequisite_extension,
)
from scripts.seal_cross_backend_hit_teacher import seal_cross_backend_teacher

FEED_FINGERPRINT = "f" * 64
CONTROL_HASH = "c" * 64
LATENT_FINGERPRINT = "d" * 64
OUTGOING_SEMANTICS = "post_control_step_after_all_physics_substeps"
EVENT_SEMANTICS = (
    "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
)
TIMING_SEMANTICS = (
    "frozen_base_swing_phase_advance_applied_identically_to_search_"
    "backend_and_cpu_replays"
)
VERIFICATION_SEMANTICS = (
    "same_candidate_relocated_across_deterministic_warp_batch_lanes"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DISABLED_BOUNDED_RESIDUAL = {
    "enabled": False,
    "dimension": 0,
    "schema_sha256": None,
    "groups": None,
}
H3_BOUNDED_RESIDUAL = {
    "enabled": True,
    "dimension": 2,
    "schema_sha256": "b" * 64,
    "groups": [
        {
            "name": "wrist_forearm",
            "actuator_names": ["SUP", "BRA"],
            "alpha": 0.05,
            "dim": 2,
        }
    ],
}


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _with_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = _canonical_sha256(result)
    return result


def _policy_contract() -> dict[str, Any]:
    return _with_hash(
        {
            "mode": "selected_physical_correction",
            "trainable_action_indices": [0, 1],
            "trainable_actuator_names": ["right_shoulder", "right_wrist"],
            "correction_physical_scales": [0.1, 0.2],
        },
        "contract_sha256",
    )


def _control_manifest(
    treatment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    residual = DISABLED_BOUNDED_RESIDUAL if treatment is None else treatment
    return {
        "schema_version": "stage3_lab_control_v1",
        "control_hash": CONTROL_HASH,
        "latent_checkpoint_fingerprint": LATENT_FINGERPRINT,
        "bounded_residual_dim": residual["dimension"],
        "bounded_residual_schema_hash": residual["schema_sha256"],
        "bounded_residual_groups": residual["groups"],
    }


def _write_source_checkpoint(
    root: Path,
    *,
    bounded_residual_treatment: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload = root / "source" / "policy.npz"
    payload.parent.mkdir(parents=True, exist_ok=True)
    np.savez(payload, actor=np.asarray([1.0], dtype=np.float32))
    policy_contract = _policy_contract()
    prerequisite_binding = _with_hash(
        {
            "schema_version": "stage3_training_prerequisite_binding_v1",
            "verified": True,
            "fixture_evidence": "exact-short-bc-base",
        },
        "binding_sha256",
    )
    metadata = {
        "checkpoint_version": "incoming_hit_training_v3",
        "iteration": 7,
        "env_steps": 700,
        "training_payload_sha256": _file_sha256(payload),
        "control_hash": CONTROL_HASH,
        "control_manifest": _control_manifest(bounded_residual_treatment),
        "policy_update_contract": policy_contract,
        "training_prerequisite_binding": prerequisite_binding,
    }
    _write_json(payload.with_suffix(".json"), metadata)
    return payload, metadata


def _write_cpu_trace(path: Path) -> None:
    count = 32
    event_index = 20
    event = np.zeros((count,), dtype=bool)
    event[event_index] = True
    hit = np.zeros((count,), dtype=bool)
    hit[event_index] = True
    velocity = np.zeros((count, 3), dtype=np.float32)
    velocity[event_index] = [3.0, 0.0, 1.0]
    stringbed = np.zeros((count, 3), dtype=np.float32)
    stringbed[:, 2] = 2.80
    stringbed[event_index, 2] = 2.75
    right_arm = np.zeros((count, 7, 3), dtype=np.float32)
    right_arm[:, -1, 2] = 2.70
    right_arm[event_index, -1, 2] = 2.65
    np.savez_compressed(
        path,
        observation_normalized=np.zeros((count, 3), dtype=np.float32),
        correction_raw=np.full((count, 2), 0.25, dtype=np.float32),
        correction_window=np.ones((count,), dtype=np.float32),
        time_to_intercept_s=np.linspace(0.31, 0.0, count, dtype=np.float32),
        hit_event=hit,
        event_rebound=event,
        shuttle_velocity=velocity,
        stringbed_position=stringbed,
        right_arm_body_position_xyz_m=right_arm,
        body_fall=np.zeros((count,), dtype=bool),
        selected_action_indices=np.asarray([0, 1], dtype=np.int32),
        physical_scales=np.asarray([0.1, 0.2], dtype=np.float32),
        feed_fingerprint=np.asarray(FEED_FINGERPRINT),
        swing_phase_advance_s=np.asarray(0.18, dtype=np.float32),
        outgoing_velocity_semantics=np.asarray(OUTGOING_SEMANTICS),
        event_rebound_contact_semantics=np.asarray(EVENT_SEMANTICS),
    )


def _write_cem_chain(
    root: Path,
    source_checkpoint: Path,
    *,
    spec_path: Path,
    scene_path: Path,
) -> dict[str, Path]:
    cem_dir = root / "cem"
    cem_dir.mkdir(parents=True, exist_ok=True)
    cpu_trace = cem_dir / "teacher_trajectory_cpu_audit.npz"
    _write_cpu_trace(cpu_trace)
    parameters = [0.25, -0.10]
    parameter_sha256 = hashlib.sha256(
        np.asarray(parameters, dtype=np.float32).tobytes(order="C")
    ).hexdigest()
    contract: dict[str, Any] = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "spec": str(spec_path.resolve()),
        "scene_sha256": _file_sha256(scene_path),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _file_sha256(source_checkpoint),
        "policy_update_contract_sha256": _policy_contract()["contract_sha256"],
        "feed_fingerprint": FEED_FINGERPRINT,
        "configured_swing_phase_advance_s": 0.18,
        "swing_phase_advance_s": 0.18,
        "swing_phase_timing_semantics": TIMING_SEMANTICS,
        "mjx_impl": "warp",
        "outgoing_velocity_semantics": OUTGOING_SEMANTICS,
        "event_rebound_contact_semantics": EVENT_SEMANTICS,
        "min_replica_fraction": 2.0 / 3.0,
        "replicas_per_candidate": 3,
        "required_replica_count": 2,
        "population": 8,
        "verification_repeats": 2,
        "verification_context_semantics": VERIFICATION_SEMANTICS,
        "return_quality_search_margin": {
            "semantics": "same_replica_training_backend_margin_gate",
            "min_outgoing_z_m_s": 0.8,
            "min_forward_m_s": 2.5,
        },
        "high_region_contact": {
            "semantics": "soft_window_teacher_gate_not_exact_apex",
            "max_stringbed_height_deficit_m": 0.10,
            "max_hand_height_deficit_m": 0.10,
        },
        "parameterization": "muscle_knots",
        "parameter_count": 2,
        "time_knots": 1,
        "physical_scales": [0.1, 0.2],
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    contract_path = cem_dir / "cem_contract.json"
    _write_json(contract_path, contract)

    best_metrics = {
        "teacher_success": True,
        "teacher_success_rate": 2.0 / 3.0,
        "high_region_contact": True,
        "no_fall": True,
    }
    verified_metrics = {
        "teacher_success": True,
        "teacher_success_rate": 2.0 / 3.0,
        "event_rebound": True,
        "event_rebound_rate": 2.0 / 3.0,
        "high_region_contact": True,
        "no_fall": True,
        "return_quality": True,
        "return_quality_rate": 2.0 / 3.0,
        "positive_outgoing_z_rate": 2.0 / 3.0,
        "positive_outgoing_forward_rate": 2.0 / 3.0,
        "outgoing_z_m_s": 0.8,
        "outgoing_forward_m_s": 3.0,
    }
    inline_cpu_audit = {
        "schema_version": "stage3_cem_inline_cpu_quality_gate_v3",
        "candidate_parameter_f32_sha256": parameter_sha256,
        "candidate_parameters": parameters,
        "cpu_quality_passed": True,
        "hit": True,
        "event_rebound": True,
        "high_region_contact": True,
        "body_fall": False,
        "feed_fingerprint": FEED_FINGERPRINT,
        "swing_phase_advance_s": 0.18,
        "outgoing_z_m_s": 1.0,
        "outgoing_forward_m_s": 3.0,
    }
    source_report = {
        "schema_version": "stage3_single_feed_mjx_cem_report_v3",
        "passed": True,
        "mjx_teacher_passed": True,
        "cpu_replay_passed": True,
        "cpu_replay_event_equivalent": True,
        "contract": contract,
        "best_search_metrics": best_metrics,
        "verified_metrics": verified_metrics,
        "teacher_trace": {
            "verification_context_semantics": VERIFICATION_SEMANTICS,
            "verification_group_indices": [1, 5],
            "selected_batch_group": 1,
        },
        "cpu_replay_audit": {
            "hit": True,
            "event_rebound": True,
            "body_fall": False,
            "feed_fingerprint": FEED_FINGERPRINT,
            "swing_phase_advance_s": 0.18,
            "trace_path": str(cpu_trace.resolve()),
            "trace_sha256": _file_sha256(cpu_trace),
        },
        "cpu_gated_best_audit": inline_cpu_audit,
    }
    source_report_path = cem_dir / "cem_report.json"
    _write_json(source_report_path, source_report)
    candidate = {
        "schema_version": "stage3_cem_teacher_candidate_v1",
        "contract_sha256": contract["contract_sha256"],
        "iteration": 4,
        "parameters": parameters,
        "metrics": best_metrics,
        "cpu_quality_audit": inline_cpu_audit,
    }
    candidate_path = cem_dir / "best_teacher.json"
    _write_json(candidate_path, candidate)

    standalone_audit = {
        "schema_version": "stage3_cem_intermediate_cpu_quality_audit_v1",
        "candidate_path": str(candidate_path.resolve()),
        "candidate_schema_version": "stage3_cem_teacher_candidate_v1",
        "candidate_role": "teacher_candidate",
        "candidate_file_sha256": _file_sha256(candidate_path),
        "candidate_iteration": 4,
        "candidate_parameter_sha256": parameter_sha256,
        "candidate_changed_during_audit": False,
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": contract["contract_sha256"],
        "source_feed_fingerprint": FEED_FINGERPRINT,
        "audited_feed_fingerprint": FEED_FINGERPRINT,
        "alternate_feed_for_unqualified_seed": False,
        "source_swing_phase_advance_s": 0.18,
        "audited_swing_phase_advance_s": 0.18,
        "alternate_timing_for_unqualified_seed": False,
        "trace_path": str(cpu_trace.resolve()),
        "trace_sha256": _file_sha256(cpu_trace),
        "hit_step": 20,
        "event_step": 20,
        "fall_step": None,
        "outgoing_z_m_s": 1.0,
        "outgoing_forward_m_s": 3.0,
        "high_region_contact": True,
        "deployment_quality_passed": True,
        "search_margin_quality_passed": True,
        "deployment_gate": {
            "min_outgoing_z_m_s": 0.5,
            "min_forward_m_s": 2.0,
        },
    }
    audit_path = cem_dir / "standalone_cpu_audit.json"
    _write_json(audit_path, standalone_audit)

    sealed_dir = root / "sealed"
    seal_cross_backend_teacher(source_report_path, sealed_dir)
    return {
        "source_report": source_report_path,
        "candidate": candidate_path,
        "audit": audit_path,
        "seal": sealed_dir / "cem_report.json",
        "dataset": sealed_dir / "teacher_trajectory_cpu_quality.npz",
    }


def _write_short_bc_checkpoint(
    root: Path,
    *,
    source_checkpoint: Path,
    correction_manifest: dict[str, Any],
    passing: bool = True,
    runtime_bounded_residual_treatment: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    run_root = root / "short_bc"
    version_name = "checkpoint_000000000000"
    version_dir = run_root / "checkpoints" / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    payload_path = version_dir / "policy.npz"
    np.savez(payload_path, actor=np.asarray([2.0], dtype=np.float32))

    teacher_binding = correction_manifest["correction_dataset"]["teacher_binding"]
    initial = 1.0
    final = 0.25 if passing else 1.25
    metrics = {
        "schema_version": "stage3_selected_correction_bc_pretrain_v1",
        "teacher_binding": teacher_binding,
        "steps": 10,
        "batch_size": 32,
        "learning_rate": 3.0e-4,
        "initial_weighted_mse": initial,
        "last_minibatch_mse": final,
        "final_weighted_mse": final,
        "improvement_fraction": (initial - final) / initial,
        "passed": passing,
    }
    metrics = _with_hash(metrics, "report_sha256")
    metrics_path = run_root / "teacher_bc_pretrain_report.json"
    _write_json(metrics_path, metrics)

    runtime_control = _control_manifest(runtime_bounded_residual_treatment)
    runtime_feed = {
        "schema_version": "incoming_hit_feed_bank_consumer_v1",
        "producer": {
            "schema_version": "incoming_hit_feed_bank_v1",
            "sample_fingerprints": [FEED_FINGERPRINT],
        },
        "consumer_order": {
            "mode": "explicit_fingerprint_order",
            "sample_fingerprints": [FEED_FINGERPRINT],
        },
    }
    initialization = {
        "schema_version": "stage3_actor_only_reward_repair_initialization_v1",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_payload_sha256": _file_sha256(source_checkpoint),
        "source_control_hash": CONTROL_HASH,
        "runtime_control_hash": CONTROL_HASH,
        "source_iteration": 7,
        "source_env_steps": 700,
        "transferred": ["policy_actor", "observation_normalizer"],
        "reset": ["value_network", "optimizer_state", "rng", "curriculum_state"],
    }
    initialization = _with_hash(initialization, "binding_sha256")
    policy_contract = _policy_contract()
    source_metadata = json.loads(
        source_checkpoint.with_suffix(".json").read_text(encoding="utf-8")
    )
    metadata = {
        "checkpoint_version": "incoming_hit_training_v3",
        "checkpoint_stage": "post_teacher_bc_pre_ppo",
        "iteration": 0,
        "env_steps": 0,
        "versioned_checkpoint_schema": "incoming_hit_versioned_checkpoint_v1",
        "version_name": version_name,
        "config": {
            "policy_update_mode": "selected_physical_correction",
            "total_env_steps": 0,
            "teacher_bc_pretrain_steps": 10,
            "teacher_bc_batch_size": 32,
            "teacher_bc_learning_rate": 3.0e-4,
        },
        "control_hash": CONTROL_HASH,
        "control_manifest": runtime_control,
        "training_feed_manifest": runtime_feed,
        "policy_update_contract": policy_contract,
        "training_prerequisite_binding": source_metadata[
            "training_prerequisite_binding"
        ],
        "actor_initialization": initialization,
        "teacher_bc_pretrain_report": metrics,
        "task_curriculum_state": {
            "schema_version": "stage3_task_curriculum_state_v2",
            "max_stage": "C3_static_velocity",
            "stage": "C0_ready_pose",
            "complete": False,
        },
        "training_payload_sha256": _file_sha256(payload_path),
    }
    metadata_path = version_dir / "policy.json"
    _write_json(metadata_path, metadata)
    completion = {
        "schema_version": "incoming_hit_versioned_checkpoint_v1",
        "version_name": version_name,
        "iteration": 0,
        "env_steps": 0,
        "payload_sha256": _file_sha256(payload_path),
        "metadata_sha256": _file_sha256(metadata_path),
    }
    completion = _with_hash(completion, "binding_sha256")
    _write_json(version_dir / "_COMPLETE.json", completion)
    pointer = {
        "schema_version": "incoming_hit_checkpoint_pointer_v1",
        "version_name": version_name,
        "checkpoint_dir": str(version_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "payload_sha256": completion["payload_sha256"],
        "metadata_sha256": completion["metadata_sha256"],
        "iteration": 0,
        "env_steps": 0,
    }
    pointer = _with_hash(pointer, "binding_sha256")
    pointer_path = run_root / "policy_latest.json"
    _write_json(pointer_path, pointer)
    _write_json(
        run_root / "train_report.json",
        {
            "requested_env_step_cap": 0,
            "env_steps": 0,
            "iterations": 0,
            "already_at_absolute_cap": True,
        },
    )
    return pointer_path, metrics_path


def _build_fixture(
    root: Path,
    *,
    passing_bc: bool = True,
    source_bounded_residual_treatment: dict[str, Any] | None = None,
    runtime_bounded_residual_treatment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec_path = (
        REPO_ROOT
        / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml"
    ).resolve(strict=True)
    scene_path = (
        REPO_ROOT
        / "environment/overall_environment/assets/overall_incoming_hit_scene.xml"
    ).resolve(strict=True)
    source_checkpoint, _metadata = _write_source_checkpoint(
        root,
        bounded_residual_treatment=source_bounded_residual_treatment,
    )
    artifacts = _write_cem_chain(
        root,
        source_checkpoint,
        spec_path=spec_path,
        scene_path=scene_path,
    )
    correction = build_successful_correction_dataset_manifest(
        action="forehand_clear",
        expected_stage3_spec=spec_path,
        expected_feed_fingerprint=FEED_FINGERPRINT,
        expected_control_hash=CONTROL_HASH,
        expected_latent_checkpoint_fingerprint=LATENT_FINGERPRINT,
        source_cem_report=artifacts["source_report"],
        candidate=artifacts["candidate"],
        cpu_audit_report=artifacts["audit"],
        cross_backend_seal_report=artifacts["seal"],
        correction_dataset=artifacts["dataset"],
    )
    correction_path = root / "successful_correction_manifest.json"
    _write_json(correction_path, correction)
    short_bc_checkpoint, short_bc_metrics = _write_short_bc_checkpoint(
        root,
        source_checkpoint=source_checkpoint,
        correction_manifest=correction,
        passing=passing_bc,
        runtime_bounded_residual_treatment=(
            source_bounded_residual_treatment
            if runtime_bounded_residual_treatment is None
            else runtime_bounded_residual_treatment
        ),
    )
    return {
        **artifacts,
        "spec": spec_path,
        "scene": scene_path,
        "source_checkpoint": source_checkpoint,
        "correction": correction,
        "correction_path": correction_path,
        "short_bc_checkpoint": short_bc_checkpoint,
        "short_bc_metrics": short_bc_metrics,
    }


def _build_release(
    root: Path,
    **fixture_kwargs: Any,
) -> dict[str, Any]:
    fixture = _build_fixture(root, **fixture_kwargs)
    release = build_stage3_reachability_release(
        correction_dataset_manifest=fixture["correction_path"],
        short_bc_checkpoint=fixture["short_bc_checkpoint"],
        short_bc_metrics=fixture["short_bc_metrics"],
    )
    release_path = root / "stage3_reachability_release.json"
    _write_json(release_path, release)
    fixture.update({"release": release, "release_path": release_path})
    return fixture


def _write_completed_static_checkpoint(
    root: Path,
    *,
    entry: dict[str, Any],
) -> Path:
    version_name = "checkpoint_000000000100"
    version_dir = root / "short_bc" / "checkpoints" / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    payload_path = version_dir / "policy.npz"
    np.savez(payload_path, actor=np.asarray([3.0], dtype=np.float32))
    prerequisites = _with_hash(
        {
            "schema_version": "stage3_training_prerequisite_binding_v1",
            "verified": True,
            "stage3_reachability_release": entry,
        },
        "binding_sha256",
    )
    metadata = {
        "checkpoint_version": "incoming_hit_training_v3",
        "checkpoint_stage": "ppo_iteration_boundary",
        "iteration": 1,
        "env_steps": 100,
        "versioned_checkpoint_schema": "incoming_hit_versioned_checkpoint_v1",
        "version_name": version_name,
        "task_curriculum_state": {
            "schema_version": "stage3_task_curriculum_state_v2",
            "max_stage": "C3_static_velocity",
            "stage": "C3_static_velocity",
            "complete": True,
        },
        "training_prerequisite_binding": prerequisites,
        "training_payload_sha256": _file_sha256(payload_path),
    }
    metadata_path = version_dir / "policy.json"
    _write_json(metadata_path, metadata)
    completion = _with_hash(
        {
            "schema_version": "incoming_hit_versioned_checkpoint_v1",
            "version_name": version_name,
            "iteration": 1,
            "env_steps": 100,
            "payload_sha256": _file_sha256(payload_path),
            "metadata_sha256": _file_sha256(metadata_path),
        },
        "binding_sha256",
    )
    _write_json(version_dir / "_COMPLETE.json", completion)
    return version_dir


def _overwrite_mutable_run_state_after_c3(
    root: Path,
    static_checkpoint: Path,
) -> None:
    payload_path = static_checkpoint / "policy.npz"
    metadata_path = static_checkpoint / "policy.json"
    completion = json.loads(
        (static_checkpoint / "_COMPLETE.json").read_text(encoding="utf-8")
    )
    pointer = _with_hash(
        {
            "schema_version": "incoming_hit_checkpoint_pointer_v1",
            "version_name": completion["version_name"],
            "checkpoint_dir": str(static_checkpoint.resolve()),
            "payload_path": str(payload_path.resolve()),
            "payload_sha256": _file_sha256(payload_path),
            "metadata_sha256": _file_sha256(metadata_path),
            "iteration": 1,
            "env_steps": 100,
        },
        "binding_sha256",
    )
    run_root = root / "short_bc"
    _write_json(run_root / "policy_latest.json", pointer)
    _write_json(
        run_root / "train_report.json",
        {
            "requested_env_step_cap": 100,
            "env_steps": 100,
            "iterations": 1,
            "already_at_absolute_cap": True,
        },
    )


def _build_completed_static_lineage(
    root: Path,
    fixture: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    short_bc = fixture["release"]["short_bc"]
    entry = validate_static_ppo_entry(
        release_path=fixture["release_path"],
        start_checkpoint=fixture["short_bc_checkpoint"],
        teacher_dataset=fixture["dataset"],
        runtime_run_dir=root / "short_bc",
        runtime_control_manifest=short_bc["runtime_control_manifest"],
        runtime_training_feed_manifest=short_bc[
            "runtime_training_feed_manifest"
        ],
    )
    return entry, _write_completed_static_checkpoint(root, entry=entry)


def test_stage3_reachability_release_binds_complete_ordered_chain(
    tmp_path: Path,
) -> None:
    fixture = _build_release(tmp_path)

    assert fixture["correction"]["schema_version"] == CORRECTION_DATASET_SCHEMA
    assert fixture["release"]["schema_version"] == REACHABILITY_RELEASE_SCHEMA
    assert fixture["release"]["authorized_next_step"] == (
        "stage3_static_single_feed_ppo"
    )
    assert fixture["release"]["target_identity"]["action_slug"] == (
        "forehand_clear"
    )
    assert fixture["release"]["target_identity"]["dataset_action_id"] == (
        "forehandClear_standard"
    )
    assert fixture["release"]["control_identity"][
        "bounded_residual_treatment"
    ] == DISABLED_BOUNDED_RESIDUAL
    assert validate_stage3_reachability_release(fixture["release_path"]) == (
        fixture["release"]
    )

    short_bc = fixture["release"]["short_bc"]
    entry = validate_static_ppo_entry(
        release_path=fixture["release_path"],
        start_checkpoint=fixture["short_bc_checkpoint"],
        teacher_dataset=fixture["dataset"],
        runtime_run_dir=tmp_path / "short_bc",
        runtime_control_manifest=short_bc["runtime_control_manifest"],
        runtime_training_feed_manifest=short_bc["runtime_training_feed_manifest"],
    )
    assert entry["schema_version"] == STATIC_PPO_ENTRY_SCHEMA
    assert entry["authorized_stage"] == "stage3_static_single_feed_ppo"
    short_bc_metadata = json.loads(
        Path(short_bc["checkpoint"]["metadata_path"]).read_text(encoding="utf-8")
    )
    resumed_binding, resumed_source_sha256, _timing = _resume_action_prior_source(
        short_bc_metadata
    )
    assert resumed_binding == fixture["correction"]["correction_dataset"][
        "teacher_binding"
    ]
    assert resumed_source_sha256 == _file_sha256(fixture["source_checkpoint"])
    runtime_prerequisites = attach_static_ppo_entry_to_prerequisites(
        short_bc_metadata["training_prerequisite_binding"],
        entry,
    )
    validate_static_ppo_prerequisite_extension(
        checkpoint_binding=short_bc_metadata["training_prerequisite_binding"],
        runtime_binding=runtime_prerequisites,
        checkpoint_payload_sha256=short_bc["checkpoint"]["payload_sha256"],
    )
    static_checkpoint = _write_completed_static_checkpoint(
        tmp_path,
        entry=entry,
    )
    assert validate_post_static_ppo_continuation(
        release_path=fixture["release_path"],
        static_checkpoint=static_checkpoint,
        teacher_dataset=fixture["dataset"],
        runtime_run_dir=tmp_path / "short_bc",
    ) == entry

    cem_command = canonical_cem_launch_command(
        spec=fixture["spec"],
        checkpoint=fixture["source_checkpoint"],
        out_dir=tmp_path / "cem-production",
    )
    bc_command = canonical_short_bc_launch_command(
        spec=fixture["spec"],
        source_checkpoint=fixture["source_checkpoint"],
        correction_dataset=fixture["dataset"],
        out_dir=tmp_path / "bc-production",
    )
    assert cem_command[:2] == (
        "scripts/run_fullbody_training.sh",
        "--incoming-hit-cem",
    )
    assert bc_command[0] == "scripts/run_fullbody_training.sh"
    assert bc_command[bc_command.index("--total-env-steps") + 1] == "0"
    assert bc_command[bc_command.index("--curriculum-max-stage") + 1] == (
        "C3_static_velocity"
    )
    with pytest.raises(ValueError, match="cannot override sealed option"):
        canonical_short_bc_launch_command(
            spec=fixture["spec"],
            source_checkpoint=fixture["source_checkpoint"],
            correction_dataset=fixture["dataset"],
            out_dir=tmp_path / "bc-invalid",
            extra_args=("--total-env-steps=1",),
        )


def test_release_and_continuation_survive_mutable_c3_run_updates(
    tmp_path: Path,
) -> None:
    fixture = _build_release(tmp_path)
    entry, static_checkpoint = _build_completed_static_lineage(
        tmp_path,
        fixture,
    )
    _overwrite_mutable_run_state_after_c3(tmp_path, static_checkpoint)

    assert "pointer_path" not in fixture["release"]["short_bc"]["checkpoint"]
    assert fixture["release"]["short_bc"][
        "zero_step_train_report_snapshot"
    ]["snapshot"]["env_steps"] == 0
    assert validate_stage3_reachability_release(fixture["release_path"]) == (
        fixture["release"]
    )
    assert validate_post_static_ppo_continuation(
        release_path=fixture["release_path"],
        static_checkpoint=static_checkpoint,
        teacher_dataset=fixture["dataset"],
        runtime_run_dir=tmp_path / "short_bc",
    ) == entry


def test_initial_release_build_still_rejects_post_c3_mutable_state(
    tmp_path: Path,
) -> None:
    fixture = _build_release(tmp_path)
    _entry, static_checkpoint = _build_completed_static_lineage(
        tmp_path,
        fixture,
    )
    _overwrite_mutable_run_state_after_c3(tmp_path, static_checkpoint)

    with pytest.raises(ValueError, match="executed PPO environment steps"):
        build_stage3_reachability_release(
            correction_dataset_manifest=fixture["correction_path"],
            short_bc_checkpoint=fixture["release"]["short_bc"]["checkpoint"][
                "payload_path"
            ],
            short_bc_metrics=fixture["short_bc_metrics"],
        )


def test_initial_release_build_still_requires_short_bc_as_latest(
    tmp_path: Path,
) -> None:
    fixture = _build_release(tmp_path)
    _entry, static_checkpoint = _build_completed_static_lineage(
        tmp_path,
        fixture,
    )
    _overwrite_mutable_run_state_after_c3(tmp_path, static_checkpoint)
    zero_step_snapshot = fixture["release"]["short_bc"][
        "zero_step_train_report_snapshot"
    ]["snapshot"]
    _write_json(tmp_path / "short_bc" / "train_report.json", zero_step_snapshot)

    with pytest.raises(ValueError, match="immutable latest pointer"):
        build_stage3_reachability_release(
            correction_dataset_manifest=fixture["correction_path"],
            short_bc_checkpoint=fixture["release"]["short_bc"]["checkpoint"][
                "payload_path"
            ],
            short_bc_metrics=fixture["short_bc_metrics"],
        )


@pytest.mark.parametrize("changed_artifact", ["payload", "metadata"])
def test_release_rejects_replaced_immutable_short_bc_files(
    tmp_path: Path,
    changed_artifact: str,
) -> None:
    fixture = _build_release(tmp_path)
    checkpoint = fixture["release"]["short_bc"]["checkpoint"]
    payload_path = Path(checkpoint["payload_path"])
    metadata_path = Path(checkpoint["metadata_path"])
    completion_path = Path(checkpoint["completion_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if changed_artifact == "payload":
        np.savez(payload_path, actor=np.asarray([99.0], dtype=np.float32))
        metadata["training_payload_sha256"] = _file_sha256(payload_path)
    else:
        metadata["replacement_marker"] = "changed immutable metadata"
    _write_json(metadata_path, metadata)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.pop("binding_sha256")
    completion["payload_sha256"] = _file_sha256(payload_path)
    completion["metadata_sha256"] = _file_sha256(metadata_path)
    _write_json(
        completion_path,
        _with_hash(completion, "binding_sha256"),
    )

    with pytest.raises(ValueError, match="immutable short-BC checkpoint changed"):
        validate_stage3_reachability_release(fixture["release_path"])


def test_stage3_release_rejects_tampered_upstream_artifact(tmp_path: Path) -> None:
    fixture = _build_release(tmp_path)
    source_report = json.loads(
        fixture["source_report"].read_text(encoding="utf-8")
    )
    source_report["wall_seconds"] = 999.0
    _write_json(fixture["source_report"], source_report)

    with pytest.raises(ValueError, match=r"fingerprint|stale"):
        validate_stage3_reachability_release(fixture["release_path"])


def test_correction_manifest_rejects_wrong_target_feed(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="wrong target feed"):
        build_successful_correction_dataset_manifest(
            action="forehand_clear",
            expected_stage3_spec=fixture["spec"],
            expected_feed_fingerprint="a" * 64,
            expected_control_hash=CONTROL_HASH,
            expected_latent_checkpoint_fingerprint=LATENT_FINGERPRINT,
            source_cem_report=fixture["source_report"],
            candidate=fixture["candidate"],
            cpu_audit_report=fixture["audit"],
            cross_backend_seal_report=fixture["seal"],
            correction_dataset=fixture["dataset"],
        )


def test_correction_manifest_rejects_action_without_registered_stage3_v2_spec(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="no 'stage3_v2_spec' asset"):
        build_successful_correction_dataset_manifest(
            action="forehand_lift",
            expected_stage3_spec=fixture["spec"],
            expected_feed_fingerprint=FEED_FINGERPRINT,
            expected_control_hash=CONTROL_HASH,
            expected_latent_checkpoint_fingerprint=LATENT_FINGERPRINT,
            source_cem_report=fixture["source_report"],
            candidate=fixture["candidate"],
            cpu_audit_report=fixture["audit"],
            cross_backend_seal_report=fixture["seal"],
            correction_dataset=fixture["dataset"],
        )


def test_correction_manifest_rejects_unregistered_action_spec_wrapper(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    wrapped_lift_spec = tmp_path / "clear-assets-wrapped-as-lift.yaml"
    wrapped_lift_spec.write_text(
        "runner_type: incoming_shuttle_hit\n"
        "action: ForehandNetLift\n"
        "experiment_id: dishonest_lift_wrapper\n"
        f"scene:\n  xml: {fixture['scene']}\n"
        "stage3_v2:\n  profile: impact_recovery_v2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registered Stage-3 v2 spec"):
        build_successful_correction_dataset_manifest(
            action="forehand_clear",
            expected_stage3_spec=wrapped_lift_spec,
            expected_feed_fingerprint=FEED_FINGERPRINT,
            expected_control_hash=CONTROL_HASH,
            expected_latent_checkpoint_fingerprint=LATENT_FINGERPRINT,
            source_cem_report=fixture["source_report"],
            candidate=fixture["candidate"],
            cpu_audit_report=fixture["audit"],
            cross_backend_seal_report=fixture["seal"],
            correction_dataset=fixture["dataset"],
        )


def test_h3_release_binds_exact_grouped_bounded_residual(tmp_path: Path) -> None:
    fixture = _build_release(
        tmp_path,
        source_bounded_residual_treatment=H3_BOUNDED_RESIDUAL,
    )

    assert fixture["correction"]["control_identity"][
        "bounded_residual_treatment"
    ] == H3_BOUNDED_RESIDUAL
    assert fixture["release"]["control_identity"][
        "bounded_residual_treatment"
    ] == H3_BOUNDED_RESIDUAL
    assert validate_stage3_reachability_release(fixture["release_path"]) == (
        fixture["release"]
    )


@pytest.mark.parametrize(
    "changed_field",
    ["schema_hash", "actuator_names", "alpha"],
)
def test_release_rejects_changed_runtime_bounded_residual_treatment(
    tmp_path: Path,
    changed_field: str,
) -> None:
    changed_runtime = json.loads(json.dumps(H3_BOUNDED_RESIDUAL))
    if changed_field == "schema_hash":
        changed_runtime["schema_sha256"] = "c" * 64
    elif changed_field == "actuator_names":
        changed_runtime["groups"][0]["actuator_names"] = ["SUP", "BRD"]
    else:
        changed_runtime["groups"][0]["alpha"] = 0.04
    fixture = _build_fixture(
        tmp_path,
        source_bounded_residual_treatment=H3_BOUNDED_RESIDUAL,
        runtime_bounded_residual_treatment=changed_runtime,
    )

    with pytest.raises(ValueError, match="changed the source bounded-residual"):
        build_stage3_reachability_release(
            correction_dataset_manifest=fixture["correction_path"],
            short_bc_checkpoint=fixture["short_bc_checkpoint"],
            short_bc_metrics=fixture["short_bc_metrics"],
        )


def test_release_rejects_tampered_bounded_residual_treatment(
    tmp_path: Path,
) -> None:
    fixture = _build_release(
        tmp_path,
        source_bounded_residual_treatment=H3_BOUNDED_RESIDUAL,
    )
    release = json.loads(fixture["release_path"].read_text(encoding="utf-8"))
    release["control_identity"]["bounded_residual_treatment"]["groups"][0][
        "alpha"
    ] = 0.04
    release.pop("release_binding_sha256")
    _write_json(
        fixture["release_path"],
        _with_hash(release, "release_binding_sha256"),
    )

    with pytest.raises(ValueError, match="release is stale"):
        validate_stage3_reachability_release(fixture["release_path"])


def test_static_ppo_entry_rejects_wrong_checkpoint(tmp_path: Path) -> None:
    fixture = _build_release(tmp_path)
    short_bc = fixture["release"]["short_bc"]

    with pytest.raises(ValueError, match="wrong short-BC checkpoint"):
        validate_static_ppo_entry(
            release_path=fixture["release_path"],
            start_checkpoint=fixture["source_checkpoint"],
            teacher_dataset=fixture["dataset"],
            runtime_run_dir=tmp_path / "short_bc",
            runtime_control_manifest=short_bc["runtime_control_manifest"],
            runtime_training_feed_manifest=short_bc[
                "runtime_training_feed_manifest"
            ],
        )


def test_short_bc_to_static_allows_only_release_prerequisite_extension(
    tmp_path: Path,
) -> None:
    fixture = _build_release(tmp_path)
    short_bc = fixture["release"]["short_bc"]
    entry = validate_static_ppo_entry(
        release_path=fixture["release_path"],
        start_checkpoint=fixture["short_bc_checkpoint"],
        teacher_dataset=fixture["dataset"],
        runtime_run_dir=tmp_path / "short_bc",
        runtime_control_manifest=short_bc["runtime_control_manifest"],
        runtime_training_feed_manifest=short_bc["runtime_training_feed_manifest"],
    )
    metadata = json.loads(
        Path(short_bc["checkpoint"]["metadata_path"]).read_text(encoding="utf-8")
    )
    checkpoint_binding = metadata["training_prerequisite_binding"]
    changed_base = dict(checkpoint_binding)
    changed_base.pop("binding_sha256")
    changed_base["fixture_evidence"] = "tampered-base"
    changed_base = _with_hash(changed_base, "binding_sha256")
    runtime_binding = attach_static_ppo_entry_to_prerequisites(
        changed_base,
        entry,
    )

    with pytest.raises(ValueError, match="beyond adding"):
        validate_static_ppo_prerequisite_extension(
            checkpoint_binding=checkpoint_binding,
            runtime_binding=runtime_binding,
            checkpoint_payload_sha256=short_bc["checkpoint"]["payload_sha256"],
        )


def test_release_rejects_missing_correction_manifest(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(FileNotFoundError):
        build_stage3_reachability_release(
            correction_dataset_manifest=tmp_path / "missing-correction.json",
            short_bc_checkpoint=fixture["short_bc_checkpoint"],
            short_bc_metrics=fixture["short_bc_metrics"],
        )


def test_release_rejects_failed_short_bc(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, passing_bc=False)

    with pytest.raises(ValueError, match="loss reduction"):
        build_stage3_reachability_release(
            correction_dataset_manifest=fixture["correction_path"],
            short_bc_checkpoint=fixture["short_bc_checkpoint"],
            short_bc_metrics=fixture["short_bc_metrics"],
        )


def test_zero_ppo_budget_is_reserved_for_fresh_quality_teacher_bc() -> None:
    assert _training_iteration_budget(
        total_env_steps=0,
        steps_per_iteration=64,
        fresh_quality_teacher_bc=True,
    ) == (0, 0, 0)
    with pytest.raises(ValueError, match="reserved"):
        _training_iteration_budget(
            total_env_steps=0,
            steps_per_iteration=64,
            fresh_quality_teacher_bc=False,
        )


def test_train_gpu_fails_before_environment_without_reachability_release() -> None:
    paths = SimpleNamespace(ppo_overrides={})

    with pytest.raises(ValueError, match="stage3-reachability-release"):
        train_gpu(
            paths,  # type: ignore[arg-type]
            total_env_steps=1,
            curriculum_max_stage="C3_static_velocity",
        )


def test_train_gpu_rejects_fresh_post_static_bypass() -> None:
    paths = SimpleNamespace(
        ppo_overrides={},
        task_profile="impact_recovery_v2",
    )

    with pytest.raises(ValueError, match="stage3-reachability-release"):
        train_gpu(
            paths,  # type: ignore[arg-type]
            total_env_steps=1,
            curriculum_max_stage="C7_recovery",
        )


def test_train_gpu_rejects_post_static_without_correction_dataset() -> None:
    paths = SimpleNamespace(
        ppo_overrides={},
        task_profile="impact_recovery_v2",
    )

    with pytest.raises(ValueError, match="same sealed correction dataset"):
        train_gpu(
            paths,  # type: ignore[arg-type]
            total_env_steps=1,
            curriculum_max_stage="C7_recovery",
            stage3_reachability_release="release.json",
            resume_from="completed-c3",
        )


def test_post_static_rejects_changed_correction_dataset(tmp_path: Path) -> None:
    fixture = _build_release(tmp_path)
    _entry, static_checkpoint = _build_completed_static_lineage(
        tmp_path,
        fixture,
    )
    wrong_dataset = tmp_path / "wrong-correction.npz"
    np.savez(wrong_dataset, correction=np.asarray([0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="wrong sealed correction dataset"):
        validate_post_static_ppo_continuation(
            release_path=fixture["release_path"],
            static_checkpoint=static_checkpoint,
            teacher_dataset=wrong_dataset,
            runtime_run_dir=tmp_path / "short_bc",
        )


def test_post_static_rejects_changed_run_root(tmp_path: Path) -> None:
    fixture = _build_release(tmp_path)
    _entry, static_checkpoint = _build_completed_static_lineage(
        tmp_path,
        fixture,
    )
    wrong_root = tmp_path / "different-run-root"
    wrong_root.mkdir()

    with pytest.raises(ValueError, match="short-BC/C3 run root"):
        validate_post_static_ppo_continuation(
            release_path=fixture["release_path"],
            static_checkpoint=static_checkpoint,
            teacher_dataset=fixture["dataset"],
            runtime_run_dir=wrong_root,
        )


def test_post_static_rejects_changed_release(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = _build_release(first_root)
    _entry, static_checkpoint = _build_completed_static_lineage(
        first_root,
        first,
    )
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = _build_release(replacement_root)

    with pytest.raises(ValueError, match="wrong reachability-release lineage"):
        validate_post_static_ppo_continuation(
            release_path=replacement["release_path"],
            static_checkpoint=static_checkpoint,
            teacher_dataset=replacement["dataset"],
            runtime_run_dir=replacement_root / "short_bc",
        )
