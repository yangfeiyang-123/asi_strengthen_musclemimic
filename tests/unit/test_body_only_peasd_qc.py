from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.badminton.scripts.latent_synergy_sweep import (
    _body_only_latent_promotion_payload,
)
from musclemimic.badminton.training_gates import evaluate_promotion
from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.dataset import write_split_shard
from musclemimic.distill.motion_identity import stable_motion_uid
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    physical_signal_metadata,
)
from musclemimic.distill.physical_qc import (
    build_body_only_physical_rollout_metrics,
)
from musclemimic.distill.physical_qc import (
    main as physical_qc_main,
)
from musclemimic.distill.provenance import begin_collection


def _teacher(sha256: str) -> dict:
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": "/fixture/teacher",
        "resolved_path": "/fixture/teacher",
        "sha256": sha256,
        "num_files": 1,
        "num_bytes": 1,
        "files": [{"path": "params", "sha256": "f" * 64, "num_bytes": 1}],
    }


def _write_body_only_dataset(
    root,
    *,
    split: str,
    motion_path: str,
    teacher_sha256: str = "a" * 64,
    with_event_field: bool = False,
):
    sample_count = 128
    names = ["muscle_a", "muscle_b"]
    ctrlrange = np.tile(np.asarray([[0.0, 1.0]]), (2, 1))
    motion_uid = stable_motion_uid(motion_path)
    unit = np.full((sample_count, 2), 0.25, dtype=np.float32)
    arrays = {
        "student_obs": np.zeros((sample_count, 3), dtype=np.float32),
        "teacher_action": np.zeros((sample_count, 2), dtype=np.float32),
        "teacher_ctrl_physical": unit.copy(),
        "muscle_excitation": unit.copy(),
        "muscle_activation": unit.copy(),
        "muscle_force": np.ones((sample_count, 2), dtype=np.float32),
        "muscle_tendon_length": np.ones((sample_count, 2), dtype=np.float32),
        "muscle_tendon_velocity": np.zeros((sample_count, 2), dtype=np.float32),
        "actuator_power": np.zeros((sample_count, 2), dtype=np.float32),
        "qfrc_actuator": np.zeros((sample_count, 4), dtype=np.float32),
        "motion_uid": np.full(sample_count, motion_uid, dtype=np.int64),
        "rollout_uid": np.arange(sample_count, dtype=np.int64),
    }
    if with_event_field:
        arrays["phase_id"] = np.zeros(sample_count, dtype=np.int32)
    teacher = _teacher(teacher_sha256)
    transaction = begin_collection(
        dataset_dir=root,
        teacher_checkpoint=teacher,
        collector="teacher_lookahead_rollout",
        split=split,
        seed=0,
        motion_paths=[motion_path],
        config_payload={"task": "chinajump", "phase_field": None},
        request_payload={"num_transitions": sample_count},
        resume=False,
        allow_test_only_unpromoted_teacher=True,
    )
    staged = write_split_shard(
        transaction.output_dir,
        arrays,
        split=split,
        metadata={
            "actuator_names": names,
            "actuator_ctrlrange": ctrlrange.tolist(),
            "ctrlrange_schema_hash": ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={
                    "actuator_names": names,
                    "ctrlrange": ctrlrange.tolist(),
                },
            ),
            "physical_signal_semantics": physical_signal_metadata(),
            "physical_capture": {
                "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
                "actuator_names": names,
                "activation_valid_mask": [True, True],
                "muscle_channel_contract": {
                    "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
                    "actuator_names": names,
                    "actuator_ids": [0, 1],
                    "actuator_dyntype": ["muscle", "muscle"],
                    "actuator_actnum": [1, 1],
                    "actuator_actadr": [0, 1],
                    "model_na": 2,
                },
            },
            "teacher_checkpoint_fingerprint": teacher_sha256,
            "teacher_checkpoint_content": teacher,
        },
    )
    transaction.commit([staged])


def test_body_only_physical_qc_binds_manifests_teacher_split_and_no_event_claims(
    tmp_path,
):
    train = tmp_path / "train"
    val = tmp_path / "val"
    _write_body_only_dataset(
        train,
        split="train",
        motion_path="ChinaJump/train_jump",
    )
    _write_body_only_dataset(
        val,
        split="val",
        motion_path="ChinaJump/val_jump",
    )

    metrics = build_body_only_physical_rollout_metrics(
        train,
        val,
        teacher_checkpoint_fingerprint="a" * 64,
    )

    assert metrics["schema_version"] == "body_only_physical_rollout_promotion_metrics_v1"
    assert metrics["immutable_manifest_binding_verified"] == 1.0
    assert metrics["checkpoint_binding_verified"] == 1.0
    assert metrics["split_disjoint_verified"] == 1.0
    assert metrics["finite_rate"] == 1.0
    assert metrics["action_saturation_fraction"] == 0.0
    assert "reference_alignment_rate" not in metrics
    assert "event_reference_binding_verified" not in metrics
    assert evaluate_promotion("physical_rollout_body_only_v1", metrics).passed


def test_body_only_physical_qc_rejects_event_fields_and_manifest_tampering(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    _write_body_only_dataset(
        train,
        split="train",
        motion_path="ChinaJump/train_jump",
        with_event_field=True,
    )
    _write_body_only_dataset(
        val,
        split="val",
        motion_path="ChinaJump/val_jump",
    )
    with pytest.raises(ValueError, match="event/phase arrays"):
        build_body_only_physical_rollout_metrics(train, val)

    manifest_path = val / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_uid"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_fingerprint"):
        build_body_only_physical_rollout_metrics(train, val)


def test_body_only_physical_qc_cli_requires_explicit_contract_without_event_metrics(
    tmp_path,
):
    train = tmp_path / "train"
    val = tmp_path / "val"
    output = tmp_path / "metrics.json"
    _write_body_only_dataset(
        train,
        split="train",
        motion_path="ChinaJump/train_jump",
    )
    _write_body_only_dataset(
        val,
        split="val",
        motion_path="ChinaJump/val_jump",
    )
    assert (
        physical_qc_main(
            [
                "--qc-contract",
                "body-only-phase-free",
                "--train",
                str(train),
                "--val",
                str(val),
                "--output",
                str(output),
                "--teacher-checkpoint-fingerprint",
                "a" * 64,
            ]
        )
        == 0
    )
    assert output.is_file()


def _latent_source_metrics() -> dict:
    group = {
        "latent_dim": 8,
        "decoder_type": "fixed_synergy",
        "seed_set": [0, 1, 2],
        "num_seeds": 3,
        "heldout_sample_count_min": 1200,
        "reconstruction_nrmse_mean": 0.08,
        "reconstruction_nrmse_std": 0.01,
        "closed_loop_success_rate_mean": 0.95,
        "closed_loop_success_rate_std": 0.01,
        "residual_energy_ratio_mean": 0.04,
        "residual_energy_ratio_max": 0.06,
        "residual_energy_ratio_ready_max": 0.06,
        "residual_energy_ratio_recovery_max": 0.06,
        "residual_bypass_gate_passed": True,
        "jacobian_projection_score_mean": 0.8,
        "cross_seed_linear_cka_mean": 0.8,
        "cross_seed_jacobian_projection_score_mean": 0.8,
        "offline_intervention_verified": True,
        "causal_rollout_verified": False,
        "phase_residual_gate_applied": False,
        "residual_gate_phase_names": [None, None],
    }
    model = {
        "run_name": "d8_fixed_synergy_seed0",
        "checkpoint_dir": "/checkpoint",
        "checkpoint_fingerprint": "b" * 64,
        "formal_synergy_basis_fingerprint": "c" * 64,
        "runtime_synergy_basis_fingerprint": "d" * 64,
        "runtime_synergy_basis_source_fingerprint": "c" * 64,
        "dataset_fingerprint": "e" * 64,
        "validation_dataset_fingerprint": "f" * 64,
        "motion_split_fingerprint": "1" * 64,
        "latent_dim": 8,
        "decoder_type": "fixed_synergy",
        "seed": 0,
        "deployment_seed_rule": "smallest",
        "checkpoint_binding_verified": 1.0,
    }
    return {
        "heldout_sample_count": 1200,
        "reconstruction_nrmse": 0.08,
        "closed_loop_success_rate": 0.95,
        "residual_energy_ratio": 0.06,
        "residual_energy_ratio_ready": 0.06,
        "residual_energy_ratio_recovery": 0.06,
        "residual_bypass_gate_passed": 1.0,
        "latent_dimension_selected": 1.0,
        "checkpoint_binding_verified": 1.0,
        "analysis_complete": 1.0,
        "cross_seed_stability_verified": 1.0,
        "alignment_evidence_verified": 1.0,
        "offline_intervention_verified": 1.0,
        "causal_rollout_required": False,
        "causal_rollout_verified": 0.0,
        "stage2_diagnostic_outcomes_complete": 0.0,
        "full_matrix_complete": 1.0,
        "selected_group": group,
        "selected_model": model,
        "selected_groups": {"best_synergy": group},
        "selected_models": {"best_synergy": model},
        "group_summaries": [group],
        "selection_rule": {"deployment_seed": "smallest"},
        "metric_mapping": {
            "heldout_sample_count": "heldout",
            "reconstruction_nrmse": "nrmse",
            "closed_loop_success_rate": "closed loop",
            "residual_energy_ratio": "total residual",
            "latent_dimension_selected": "registered matrix",
            "checkpoint_binding_verified": "runtime/checkpoint/dataset",
        },
        "phase_residual_gate_applied": False,
        "residual_gate_phase_names": [None, None],
    }


def test_body_only_latent_projection_omits_phase_and_stage2_causal_claims():
    metrics = _body_only_latent_promotion_payload(_latent_source_metrics())
    encoded = json.dumps(metrics, sort_keys=True)

    assert metrics["schema_version"] == "latent_synergy_body_only_promotion_metrics_v1"
    assert metrics["basis_binding_verified"] == 1.0
    assert "residual_energy_ratio_ready" not in encoded
    assert "residual_energy_ratio_recovery" not in encoded
    assert "causal_rollout_verified" not in encoded
    assert "stage2_diagnostic_outcomes_complete" not in encoded
    assert evaluate_promotion("latent_synergy_body_only_v1", metrics).passed


def test_body_only_latent_gate_rejects_wrong_schema_and_forbidden_claim_field():
    metrics = _body_only_latent_promotion_payload(_latent_source_metrics())
    wrong = dict(metrics)
    wrong["schema_version"] = "latent_synergy_promotion_metrics_v2"
    with pytest.raises(ValueError, match="requires schema_version"):
        evaluate_promotion("latent_synergy_body_only_v1", wrong)

    forbidden = dict(metrics)
    forbidden.pop("promotion_metrics_fingerprint")
    forbidden["residual_energy_ratio_ready"] = 0.01
    from musclemimic.badminton.training_gates import _mapping_sha256

    forbidden["promotion_metrics_fingerprint"] = _mapping_sha256(forbidden)
    with pytest.raises(ValueError, match="claim fields"):
        evaluate_promotion("latent_synergy_body_only_v1", forbidden)


def test_disabled_phase_sweep_selects_body_only_latent_schema():
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _select_promotion_model,
    )

    basis = "b" * 64
    checkpoint = "c" * 64
    plan = {
        "jobs": [{"latent_dim": 8, "decoder_type": "fixed_synergy", "seed": 0}],
        "synergy_basis_fingerprint": basis,
        "phase_contract": {
            "schema_version": "latent_phase_contract_v1",
            "phase_field": None,
            "phases": [],
            "require_all_phases": False,
        },
    }
    records = [
        {
            "run_name": "d8_fixed_synergy_seed0",
            "latent_dim": 8,
            "decoder_type": "fixed_synergy",
            "seed": 0,
            "checkpoint_dir": "/checkpoint",
            "checkpoint_fingerprint": checkpoint,
            "dataset_fingerprint": "d" * 64,
            "validation_dataset_fingerprint": "e" * 64,
            "motion_split_fingerprint": "f" * 64,
            "runtime_synergy_basis_fingerprint": "1" * 64,
            "runtime_synergy_basis_source_fingerprint": basis,
            "synergy_basis_expected_fingerprint": basis,
            "basis_binding_verified": True,
            "analysis_complete": True,
            "metrics": {
                "num_eval_samples": 1200,
                "physical_excitation_mse": 0.0064,
                "residual_energy_ratio": 0.04,
            },
            "closed_loop": {
                "checkpoint_fingerprint": checkpoint,
                "prior_mean_no_fall_rate": 0.95,
                "promotion": {"passed": True},
                "by_lambda": {
                    "lambda_0p000": {"residual_energy_ratio": 0.04}
                },
            },
            "analysis": {
                "jacobian_alignment": {"projection_score_mean": 0.8},
                "intervention": {"num_samples": 1200},
            },
        }
    ]
    cross = [
        {
            "latent_dim": 8,
            "decoder_type": "fixed_synergy",
            "seed_set": ["0"],
            "offline_intervention_verified": True,
            "causal_rollout_verified": False,
            "report": {
                "num_pairs": 1,
                "linear_cka_mean": 0.8,
                "jacobian_projection_score_mean": 0.8,
            },
        }
    ]

    promotion = _select_promotion_model(
        records,
        plan,
        cross_seed_analysis=cross,
        failures=[],
    )

    assert promotion["schema_version"] == "latent_synergy_body_only_promotion_metrics_v1"
    assert evaluate_promotion("latent_synergy_body_only_v1", promotion).passed
