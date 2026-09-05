from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

import musclemimic.algorithms.common.env_utils as env_utils
from loco_mujoco.core.utils import Box
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.latent_muscle.decoder_factory import build_decoder_bundle
from musclemimic.latent_muscle.synergy_decoder import LoadedSynergyBasis
from musclemimic.latent_muscle.train_latent import LatentTrainConfig
from musclemimic.synergy.multistage_contract import (
    EXACT_RUNTIME_COMPATIBILITY,
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    FULL_354_MODE,
    PORTABLE_COMPATIBILITY,
    BodySynergyContractV2,
    build_full_354_action_manifest,
    canonical_action_mode,
    canonical_json_sha256,
    load_body_synergy_contract,
    load_compatible_body_synergy_contract,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash

BODY_NAMES = tuple(f"body_muscle_{index:03d}" for index in range(354))
BODY_CTRLRANGE = np.asarray([[0.0, 1.0]] * len(BODY_NAMES), dtype=np.float64)
MODEL_HASH = "a" * 64


class _DirectBodyEnv:
    def __init__(self):
        self.policy_actuator_names = BODY_NAMES
        self.info = SimpleNamespace(
            observation_space=Box(-np.ones(4), np.ones(4)),
            action_space=Box(-np.ones(354), np.ones(354)),
        )
        self.mdp_info = self.info


class _LegacyFullFingerEnv:
    def __init__(self):
        self.policy_actuator_names = tuple(f"legacy_actuator_{index:03d}" for index in range(416))
        self.info = SimpleNamespace(
            observation_space=Box(-np.ones(4), np.ones(4)),
            action_space=Box(-np.ones(416), np.ones(416)),
        )
        self.mdp_info = self.info


class _UnknownActionInterfaceEnv:
    def __init__(self):
        self.info = SimpleNamespace(
            observation_space=Box(-np.ones(4), np.ones(4)),
        )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, FULL_354_MODE),
        ({"enabled": False}, FULL_354_MODE),
        ({"enabled": True}, FIXED_SYNERGY_MODE),
        ({"mode": FIXED_SYNERGY_RESIDUAL_MODE}, FIXED_SYNERGY_RESIDUAL_MODE),
    ],
)
def test_canonical_action_mode_resolves_all_modes_and_legacy_switches(config, expected):
    assert canonical_action_mode(config) == expected


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": False, "mode": FIXED_SYNERGY_MODE},
        {"enabled": True, "mode": FULL_354_MODE},
    ],
)
def test_canonical_action_mode_rejects_conflicting_explicit_switch(config):
    with pytest.raises(ValueError, match="mode and legacy enabled switch conflict"):
        canonical_action_mode(config)


def test_implicit_legacy_416_interface_is_not_mislabeled_full_354():
    config = OmegaConf.create({})
    env = _LegacyFullFingerEnv()

    assert apply_policy_interface_wrappers(env, config) is env
    assert config.get("action_representation") is None
    assert config.get("action_manifest") is None
    assert config.get("body_synergy_contract") is None


def test_implicit_unknown_action_interface_preserves_native_contract():
    config = OmegaConf.create({})
    env = _UnknownActionInterfaceEnv()

    assert apply_policy_interface_wrappers(env, config) is env
    assert config.get("action_representation") is None
    assert config.get("action_manifest") is None
    assert config.get("body_synergy_contract") is None


def test_explicit_full_354_rejects_legacy_416_interface():
    config = OmegaConf.create(
        {"action_representation": {"mode": FULL_354_MODE, "enabled": False}}
    )

    with pytest.raises(ValueError, match="requires exactly 354"):
        apply_policy_interface_wrappers(_LegacyFullFingerEnv(), config)


@pytest.mark.parametrize(
    "action_representation",
    [
        None,
        {"mode": FULL_354_MODE},
        {"enabled": False},
    ],
)
def test_explicit_and_legacy_direct_modes_bind_same_full_354_manifest(
    monkeypatch,
    action_representation,
):
    monkeypatch.setattr(
        env_utils,
        "_resolve_runtime_ctrlrange",
        lambda _env, _names: BODY_CTRLRANGE,
    )
    monkeypatch.setattr(
        env_utils,
        "_resolve_runtime_model_hash",
        lambda _env: MODEL_HASH,
    )
    payload = {} if action_representation is None else {"action_representation": action_representation}
    config = OmegaConf.create(payload)
    env = _DirectBodyEnv()

    result = apply_policy_interface_wrappers(env, config)

    assert result is env
    assert config.action_representation.mode == FULL_354_MODE
    assert config.action_representation.enabled is False
    manifest = OmegaConf.to_container(config.action_manifest, resolve=True)
    assert manifest["mode"] == FULL_354_MODE
    assert manifest["policy_action_dim"] == 354
    assert manifest["body_action_dim"] == 354
    assert manifest["actuator_names"] == list(BODY_NAMES)
    assert manifest["actuator_schema_hash"] == actuator_schema_hash(BODY_NAMES)
    assert manifest["control_range_hash"] == ctrlrange_schema_hash(BODY_NAMES, BODY_CTRLRANGE)
    assert manifest["runtime_model_hash"] == MODEL_HASH
    assert len(manifest["physical_action_interface_hash"]) == 64
    assert "basis" not in manifest
    contract = BodySynergyContractV2.from_manifest(
        OmegaConf.to_container(config.body_synergy_contract, resolve=True)
    )
    assert contract.mode == FULL_354_MODE
    assert contract.policy_action_dim == 354
    assert contract.basis_fingerprint is None
    assert len(contract.portable_decoder_core_fingerprint) == 64
    assert len(contract.stage_runtime_binding_fingerprint) == 64


def test_direct_and_fixed_synergy_contracts_are_incompatible():
    direct = BodySynergyContractV2.from_action_manifest(_direct_manifest())
    synergy = BodySynergyContractV2.from_action_manifest(_fixed_synergy_manifest())

    assert direct.mode == FULL_354_MODE
    assert synergy.mode == FIXED_SYNERGY_MODE
    with pytest.raises(ValueError, match="portable decoder cores"):
        direct.assert_portable_compatible(synergy)
    with pytest.raises(ValueError, match="incompatible BodySynergyContractV2"):
        direct.assert_compatible(synergy)


def test_legacy_action_manifests_are_rejected_after_excitation_v2():
    legacy_direct = _rehash_action_manifest(
        {
            **_direct_manifest(),
            "schema_version": "full_354_action_v1",
        }
    )
    with pytest.raises(ValueError, match="legacy signed-control"):
        BodySynergyContractV2.from_action_manifest(legacy_direct)

    legacy_synergy = _rehash_action_manifest(
        {
            **_fixed_synergy_manifest(),
            "schema_version": "early_synergy_action_v1",
        }
    )
    with pytest.raises(ValueError, match="v1 decoder artifacts"):
        BodySynergyContractV2.from_action_manifest(legacy_synergy)


def test_full_354_physical_exploration_matches_synergy_rms_target(monkeypatch):
    monkeypatch.setattr(
        env_utils,
        "_resolve_runtime_ctrlrange",
        lambda _env, _names: BODY_CTRLRANGE,
    )
    monkeypatch.setattr(env_utils, "_resolve_runtime_model_hash", lambda _env: MODEL_HASH)
    config = OmegaConf.create(
        {
            "action_representation": {
                "mode": FULL_354_MODE,
                "enabled": False,
                "expected_underlying_action_dim": 354,
                "exploration": {
                    "calibrate_in_physical_space": True,
                    "target_initial_excitation_rms": 0.08,
                    "std_mode": "per_dimension",
                },
            }
        }
    )

    apply_policy_interface_wrappers(_DirectBodyEnv(), config)

    std = np.asarray(config.init_std_vector, dtype=np.float64)
    assert std.shape == (354,)
    np.testing.assert_allclose(std, 0.16, rtol=0.0, atol=1e-12)
    exploration = OmegaConf.to_container(config.action_manifest.exploration, resolve=True)
    assert exploration["kind"] == "direct_effective_excitation_jacobian_calibration_v2"
    assert exploration["target_initial_excitation_rms"] == pytest.approx(0.08)
    assert exploration["achieved_initial_excitation_rms"] == pytest.approx(0.08)


def test_full_354_rejects_legacy_signed_runtime_ctrlrange(monkeypatch):
    monkeypatch.setattr(
        env_utils,
        "_resolve_runtime_ctrlrange",
        lambda _env, _names: np.asarray(
            [[-1.0, 1.0]] * len(BODY_NAMES), dtype=np.float64
        ),
    )
    monkeypatch.setattr(env_utils, "_resolve_runtime_model_hash", lambda _env: MODEL_HASH)
    config = OmegaConf.create(
        {
            "action_representation": {
                "mode": FULL_354_MODE,
                "enabled": False,
            }
        }
    )

    with pytest.raises(ValueError, match=r"exactly \[0,1\]"):
        apply_policy_interface_wrappers(_DirectBodyEnv(), config)


def test_contract_roundtrip_and_runtime_or_serialized_drift_fail_closed(tmp_path):
    contract = BodySynergyContractV2.from_action_manifest(_direct_manifest())
    path = contract.save(tmp_path / "body_synergy_contract.json")

    loaded = load_body_synergy_contract(path)
    assert loaded == contract
    assert loaded.contract_fingerprint == contract.contract_fingerprint
    assert (
        loaded.portable_decoder_core_fingerprint
        == contract.portable_decoder_core_fingerprint
    )
    assert (
        loaded.stage_runtime_binding_fingerprint
        == contract.stage_runtime_binding_fingerprint
    )
    loaded.assert_portable_compatible(contract)
    loaded.assert_exact_runtime_compatible(contract, require_complete=True)
    loaded.assert_compatible(contract)
    loaded.validate_runtime(
        actuator_names=BODY_NAMES,
        ctrlrange=BODY_CTRLRANGE,
        runtime_model_hash=MODEL_HASH,
        mode=FULL_354_MODE,
        policy_action_dim=354,
        physical_action_interface_hash=contract.physical_action_interface_hash,
        require_model_hash=True,
    )

    with pytest.raises(ValueError, match="ordered actuator names"):
        loaded.validate_runtime(
            actuator_names=tuple(reversed(BODY_NAMES)),
            ctrlrange=BODY_CTRLRANGE,
            runtime_model_hash=MODEL_HASH,
        )
    with pytest.raises(ValueError, match="model hash"):
        loaded.validate_runtime(
            actuator_names=BODY_NAMES,
            ctrlrange=BODY_CTRLRANGE,
            runtime_model_hash="b" * 64,
        )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["runtime_model_hash"] = "b" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="stage_runtime_binding_fingerprint mismatch"):
        load_body_synergy_contract(path)


def test_portable_core_survives_stage_model_and_coverage_rebinding(tmp_path):
    stage1 = BodySynergyContractV2.from_action_manifest(_fixed_synergy_manifest())
    stage2_manifest = _with_stage_runtime_binding(
        _fixed_synergy_manifest(),
        runtime_model_hash="b" * 64,
        coverage_gate={
            "schema_version": "stage2_mass_100_dynamic_coverage_v1",
            "artifact_fingerprint": "8" * 64,
            "passed": True,
        },
        target_coverage_evidence={"status": "passed_stage2_mass_100"},
    )
    stage2 = BodySynergyContractV2.from_action_manifest(stage2_manifest)

    assert (
        stage1.portable_decoder_core_fingerprint
        == stage2.portable_decoder_core_fingerprint
    )
    assert (
        stage1.stage_runtime_binding_fingerprint
        != stage2.stage_runtime_binding_fingerprint
    )
    assert stage1.contract_fingerprint != stage2.contract_fingerprint
    stage1.assert_portable_compatible(stage2)
    with pytest.raises(ValueError, match="exact runtime bindings"):
        stage1.assert_exact_runtime_compatible(stage2)

    path = stage2.save(tmp_path / "stage2_contract.json")
    assert (
        load_compatible_body_synergy_contract(
            path,
            stage1,
            compatibility=PORTABLE_COMPATIBILITY,
        )
        == stage2
    )
    with pytest.raises(ValueError, match="exact runtime bindings"):
        load_compatible_body_synergy_contract(
            path,
            stage1,
            compatibility=EXACT_RUNTIME_COMPATIBILITY,
        )


def test_decoder_or_source_drift_breaks_portable_compatibility():
    reference = BodySynergyContractV2.from_action_manifest(_fixed_synergy_manifest())
    changed_manifest = _fixed_synergy_manifest()
    changed_manifest["basis_fingerprint"] = "9" * 64
    changed_manifest = _rehash_action_manifest(changed_manifest)
    changed = BodySynergyContractV2.from_action_manifest(changed_manifest)

    assert (
        reference.portable_decoder_core_fingerprint
        != changed.portable_decoder_core_fingerprint
    )
    with pytest.raises(ValueError, match=r"basis_fingerprint"):
        reference.assert_portable_compatible(changed)


def test_serialized_subfingerprints_are_mandatory_and_fail_closed(tmp_path):
    contract = BodySynergyContractV2.from_action_manifest(_fixed_synergy_manifest())
    path = contract.save(tmp_path / "contract.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    missing = dict(manifest)
    missing.pop("portable_decoder_core_fingerprint")
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ from schema"):
        load_body_synergy_contract(path)

    unknown = {**manifest, "future_unvalidated_field": "must-not-be-ignored"}
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ from schema"):
        load_body_synergy_contract(path)

    encoded = json.dumps(manifest)
    duplicate = encoded.replace(
        '"mode": "fixed_synergy"',
        '"mode": "fixed_synergy", "mode": "fixed_synergy"',
        1,
    )
    assert duplicate != encoded
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_body_synergy_contract(path)

    portable_tamper = dict(manifest)
    portable_tamper["basis_fingerprint"] = "9" * 64
    path.write_text(json.dumps(portable_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="portable_decoder_core_fingerprint mismatch"):
        load_body_synergy_contract(path)

    runtime_tamper = dict(manifest)
    runtime_tamper["coverage_binding"] = {
        "coverage_gate": {"passed": True},
        "target_coverage_evidence": {"status": "tampered"},
    }
    path.write_text(json.dumps(runtime_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="stage_runtime_binding_fingerprint mismatch"):
        load_body_synergy_contract(path)


def test_canonical_latent_synergy_disables_full_action_baseline_by_default(monkeypatch):
    config_path = Path("fullbody/config_specific_task/distill/latent_forehandclear_synergy_v3.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["latent_distill"]["synergy_include_baseline"] is False
    assert payload["latent_distill"]["synergy_baseline_l1_weight"] == 0.0
    assert payload["latent_distill"]["synergy_baseline_l2_weight"] == 0.0
    assert LatentTrainConfig(dataset_dir="unused", output_dir="unused").synergy_include_baseline is False

    basis = LoadedSynergyBasis(
        basis=np.asarray([[0.4, 0.1], [0.2, 0.3], [0.1, 0.2]], dtype=np.float32),
        actuator_names=("a", "b", "c"),
        excitation_bounds=np.asarray([[0.0, 1.0]] * 3, dtype=np.float32),
        fingerprint="c" * 64,
        manifest={},
    )
    monkeypatch.setattr(
        "musclemimic.latent_muscle.decoder_factory.load_fixed_synergy_basis",
        lambda *_args, **_kwargs: basis,
    )
    monkeypatch.setattr(
        "musclemimic.latent_muscle.decoder_factory.validate_decoder_synergy_basis",
        lambda *_args, **_kwargs: None,
    )
    base = {
        "decoder_type": FIXED_SYNERGY_MODE,
        "synergy_basis_path": "unused-test-path",
        "legacy_synergy_decoder_ablation": True,
    }
    primary = build_decoder_bundle(
        base,
        action_dim=3,
        hidden_layer_dims=(4,),
        actuator_names=("a", "b", "c"),
    )
    ablation = build_decoder_bundle(
        {**base, "synergy_include_baseline": True},
        action_dim=3,
        hidden_layer_dims=(4,),
        actuator_names=("a", "b", "c"),
    )
    assert primary.module.include_baseline is False
    assert ablation.module.include_baseline is True


def _direct_manifest() -> dict:
    return build_full_354_action_manifest(
        actuator_names=BODY_NAMES,
        ctrlrange=BODY_CTRLRANGE,
        runtime_model_hash=MODEL_HASH,
    )


def _rehash_action_manifest(manifest: dict) -> dict:
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key not in {"physical_action_interface_hash", "exploration"}
    }
    return {
        **unsigned,
        "physical_action_interface_hash": canonical_json_sha256(unsigned),
    }


def _with_stage_runtime_binding(
    manifest: dict,
    *,
    runtime_model_hash: str,
    coverage_gate: dict | None,
    target_coverage_evidence: dict,
) -> dict:
    return _rehash_action_manifest(
        {
            **manifest,
            "runtime_model_hash": runtime_model_hash,
            "coverage_gate": coverage_gate,
            "target_coverage_evidence": target_coverage_evidence,
        }
    )


def _fixed_synergy_manifest() -> dict:
    control_hash = ctrlrange_schema_hash(BODY_NAMES, BODY_CTRLRANGE)
    unsigned = {
        "schema_version": "early_synergy_action_v2",
        "mode": FIXED_SYNERGY_MODE,
        "policy_action_dim": 8,
        "body_action_dim": 354,
        "actuator_names": list(BODY_NAMES),
        "basis_rank": 8,
        "residual_dim": 0,
        "basis_fingerprint": "1" * 64,
        "runtime_basis_fingerprint": "2" * 64,
        "coefficient_transform_fingerprint": "3" * 64,
        "coefficient_statistics_fingerprint": "4" * 64,
        "tonic_baseline_fingerprint": "5" * 64,
        "residual_basis_fingerprint": None,
        "residual_fit_contract_fingerprint": None,
        "residual_allowed_muscle_mask_fingerprint": None,
        "residual_alpha": 0.0,
        "actuator_schema_hash": actuator_schema_hash(BODY_NAMES),
        "control_range_hash": control_hash,
        "runtime_control_range_hash": control_hash,
        "runtime_model_hash": MODEL_HASH,
        "basis_source": {
            "source_dataset_fingerprint": "6" * 64,
            "teacher_checkpoint_fingerprint": "7" * 64,
        },
        "primitive_source_binding": None,
        "coverage_gate": None,
        "target_coverage_evidence": {"status": "not_required_by_config"},
    }
    return {
        **unsigned,
        "physical_action_interface_hash": canonical_json_sha256(unsigned),
    }
