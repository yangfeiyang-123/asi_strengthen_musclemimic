from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
from musclemimic.latent_muscle.analysis_export import ANALYSIS_INPUT_SCHEMA_VERSION
from musclemimic.latent_muscle.causal_rollout_artifact import (
    PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION,
    REQUIRED_OUTCOMES,
    build_causal_rollout_artifact,
    validate_causal_rollout_artifact,
)


def _write_paired_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    n, d, e = 2, 2, 2
    sample_uids = np.asarray(["sample-a", "sample-b"])
    directions = np.eye(d, dtype=np.float32)
    epsilons = np.asarray([-0.5, 0.5], dtype=np.float32)
    analysis_path = tmp_path / "analysis_inputs.npz"
    np.savez_compressed(
        analysis_path,
        sample_uids=sample_uids,
        intervention_directions=directions,
        intervention_epsilons=epsilons,
    )
    analysis_manifest_path = tmp_path / "analysis_inputs.json"
    analysis_manifest = {
        "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
        "npz_sha256": file_sha256(analysis_path),
        "checkpoint_fingerprint": "c" * 64,
        "formal_synergy_basis_fingerprint": "b" * 64,
    }
    analysis_manifest["manifest_fingerprint"] = canonical_json_sha256(analysis_manifest)
    analysis_manifest_path.write_text(json.dumps(analysis_manifest), encoding="utf-8")

    shapes = {
        "muscle_excitation": 2,
        "muscle_activation": 2,
        "joint_position": 3,
        "joint_velocity": 3,
        "trunk_state": 2,
        "racket_state": 2,
        "impact_outcome": 1,
        "landing_outcome": 2,
    }
    baseline = {
        "sample_uids": sample_uids,
        "initial_state_fingerprints": np.asarray(["1" * 64, "2" * 64]),
        "rollout_seeds": np.asarray([101, 202], dtype=np.int64),
    }
    for name, width in shapes.items():
        value = 0.2 if name.startswith("muscle_") else 1.0
        baseline[name] = np.full((n, width), value, dtype=np.float32)
    perturbed = {
        "sample_uids": sample_uids,
        "initial_state_fingerprints": np.broadcast_to(
            baseline["initial_state_fingerprints"][:, None, None], (n, d, e)
        ).copy(),
        "rollout_seeds": np.broadcast_to(baseline["rollout_seeds"][:, None, None], (n, d, e)).copy(),
        "intervention_directions": directions,
        "intervention_epsilons": epsilons,
    }
    for name in REQUIRED_OUTCOMES:
        perturbed[name] = np.broadcast_to(
            baseline[name][:, None, None, :],
            (n, d, e, baseline[name].shape[-1]),
        ).copy()
        perturbed[name] += 0.05

    baseline_path = tmp_path / "baseline.npz"
    perturbed_path = tmp_path / "perturbed.npz"
    np.savez_compressed(baseline_path, **baseline)
    np.savez_compressed(perturbed_path, **perturbed)
    units = {
        "muscle_excitation": ["unit_interval"] * 2,
        "muscle_activation": ["unit_interval"] * 2,
        "joint_position": ["rad"] * 3,
        "joint_velocity": ["rad_s-1"] * 3,
        "trunk_state": ["m", "rad"],
        "racket_state": ["m", "m_s-1"],
        "impact_outcome": ["binary"],
        "landing_outcome": ["m", "m"],
    }
    semantics = {
        "muscle_excitation": "unit_interval_excitation",
        "muscle_activation": "mujoco_unit_interval_activation_state",
    }
    outcome_schemas = {}
    for name, width in shapes.items():
        feature_names = (
            ["muscle_a", "muscle_b"] if name.startswith("muscle_") else [f"{name}_{index}" for index in range(width)]
        )
        outcome_schemas[name] = {
            "feature_names": feature_names,
            "units": units[name],
            "coordinate_frame": "ordered_model_or_world_contract",
            "semantics": semantics.get(name, f"measured_{name}"),
        }
    rollout_manifest_path = tmp_path / "paired_rollout_manifest.json"
    outcome_availability = dict.fromkeys(REQUIRED_OUTCOMES, True)
    rollout_manifest = {
        "schema_version": PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION,
        "evidence_kind": "environment_rollout",
        "checkpoint_fingerprint": "c" * 64,
        "synergy_basis_fingerprint": "b" * 64,
        "analysis_manifest_fingerprint": analysis_manifest["manifest_fingerprint"],
        "baseline_records_sha256": file_sha256(baseline_path),
        "perturbed_records_sha256": file_sha256(perturbed_path),
        "environment_fingerprint": "e" * 64,
        "policy_abi_hash": "a" * 64,
        "rollout_engine": "external_fixed_state_driver_v1",
        "fixed_state_initialization": "exact_snapshot_restore",
        "common_random_numbers": True,
        "physical_signal_semantics": physical_signal_metadata(),
        "activation_valid_mask": [True, True],
        "outcome_schemas": outcome_schemas,
        "outcome_availability": outcome_availability,
        "stage2_diagnostic_outcomes_complete": True,
        "task_outcomes_complete": True,
    }
    rollout_manifest["manifest_fingerprint"] = canonical_json_sha256(rollout_manifest)
    rollout_manifest_path.write_text(json.dumps(rollout_manifest), encoding="utf-8")
    return {
        "analysis_path": analysis_path,
        "analysis_manifest_path": analysis_manifest_path,
        "baseline_path": baseline_path,
        "perturbed_path": perturbed_path,
        "rollout_manifest_path": rollout_manifest_path,
        "baseline": baseline,
        "perturbed": perturbed,
        "rollout_manifest": rollout_manifest,
    }


def _build(fixture, tmp_path):
    return build_causal_rollout_artifact(
        analysis_inputs=fixture["analysis_path"],
        analysis_manifest=fixture["analysis_manifest_path"],
        baseline_records=fixture["baseline_path"],
        perturbed_records=fixture["perturbed_path"],
        rollout_manifest=fixture["rollout_manifest_path"],
        output_npz=tmp_path / "causal_interventions.npz",
    )


def _reseal_records_and_source(fixture):
    np.savez_compressed(fixture["baseline_path"], **fixture["baseline"])
    np.savez_compressed(fixture["perturbed_path"], **fixture["perturbed"])
    manifest = fixture["rollout_manifest"]
    manifest["baseline_records_sha256"] = file_sha256(fixture["baseline_path"])
    manifest["perturbed_records_sha256"] = file_sha256(fixture["perturbed_path"])
    manifest.pop("manifest_fingerprint", None)
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    fixture["rollout_manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")


def _enable_masked_landing_contract(fixture):
    schema = fixture["rollout_manifest"]["outcome_schemas"]["landing_outcome"]
    schema["missing_event_contract"] = {
        "schema_version": "event_presence_masked_zero_sentinel_v1",
        "storage_sentinel": 0.0,
        "effect_policy": ("continuous deltas require both events; stored zero sentinels are never measurements"),
    }
    schema["masked_value_contracts"] = [
        {
            "presence_feature": "landing_outcome_0",
            "value_feature": "landing_outcome_1",
            "missing_sentinel": 0.0,
        }
    ]
    fixture["baseline"]["landing_outcome"][:] = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    fixture["perturbed"]["landing_outcome"] = np.broadcast_to(
        fixture["baseline"]["landing_outcome"][:, None, None, :],
        (2, 2, 2, 2),
    ).copy()
    fixture["perturbed"]["landing_outcome"][0, :, :, 1] += 0.25


def test_builder_only_seals_measured_paired_deltas_and_imports_no_simulator(tmp_path):
    import musclemimic.latent_muscle.causal_rollout_artifact as module

    fixture = _write_paired_fixture(tmp_path)
    manifest = _build(fixture, tmp_path)
    with np.load(tmp_path / "causal_interventions.npz", allow_pickle=False) as data:
        effects = np.asarray(data["causal_effects"])
    assert np.allclose(effects, 0.05)
    assert manifest["fixed_state_initialization"] == "exact_snapshot_restore"
    assert manifest["common_random_numbers"] is True
    source = inspect.getsource(module)
    assert "import mujoco" not in source
    assert "subprocess" not in source


def test_builder_rejects_missing_outcome_and_fixed_state_mismatch(tmp_path):
    fixture = _write_paired_fixture(tmp_path)
    fixture["perturbed"].pop("landing_outcome")
    _reseal_records_and_source(fixture)
    with pytest.raises(ValueError, match="incomplete"):
        _build(fixture, tmp_path)

    fixture = _write_paired_fixture(tmp_path / "state")
    fixture["perturbed"]["initial_state_fingerprints"][0, 0, 0] = "3" * 64
    _reseal_records_and_source(fixture)
    with pytest.raises(ValueError, match="exact baseline state"):
        _build(fixture, tmp_path / "state")


def test_builder_rejects_signed_ctrl_relabel_and_semantic_substitution(tmp_path):
    fixture = _write_paired_fixture(tmp_path)
    fixture["baseline"]["muscle_excitation"][0, 0] = -0.25
    _reseal_records_and_source(fixture)
    with pytest.raises(ValueError, match="signed control"):
        _build(fixture, tmp_path)

    fixture = _write_paired_fixture(tmp_path / "semantics")
    fixture["rollout_manifest"]["physical_signal_semantics"]["muscle_excitation"]["semantics"] = "raw_signed_ctrl"
    manifest = fixture["rollout_manifest"]
    manifest.pop("manifest_fingerprint")
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    fixture["rollout_manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact excitation/activation semantics"):
        _build(fixture, tmp_path / "semantics")


def test_builder_excludes_presence_masked_sentinel_values_from_dense_effects(tmp_path):
    fixture = _write_paired_fixture(tmp_path)
    _enable_masked_landing_contract(fixture)
    _reseal_records_and_source(fixture)
    manifest = _build(fixture, tmp_path)
    with np.load(tmp_path / "causal_interventions.npz", allow_pickle=False) as data:
        names = np.asarray(data["causal_effect_names"]).astype(str).tolist()
    assert "landing_outcome:landing_outcome_0" in names
    assert "landing_outcome:landing_outcome_1" not in names
    layout = manifest["outcome_layout"]["landing_outcome"]
    assert layout["excluded_masked_value_features"] == ["landing_outcome_1"]
    assert layout["effect_names"] == ["landing_outcome:landing_outcome_0"]

    manifest_path = tmp_path / "causal_interventions.json"
    unsafe = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsafe["outcome_layout"]["landing_outcome"]["excluded_masked_value_features"] = []
    unsafe.pop("manifest_fingerprint")
    unsafe["manifest_fingerprint"] = canonical_json_sha256(unsafe)
    manifest_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="mask-safe ordered schemas"):
        validate_causal_rollout_artifact(
            tmp_path / "causal_interventions.npz",
            manifest_path,
        )


def test_builder_rejects_non_sentinel_values_for_absent_events(tmp_path):
    fixture = _write_paired_fixture(tmp_path)
    _enable_masked_landing_contract(fixture)
    fixture["baseline"]["landing_outcome"][1, 1] = 123.0
    _reseal_records_and_source(fixture)
    with pytest.raises(ValueError, match="storage sentinel"):
        _build(fixture, tmp_path)
