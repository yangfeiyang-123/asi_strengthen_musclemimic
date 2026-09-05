from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash


def _require_deps():
    pytest.importorskip("jax")
    pytest.importorskip("flax")


def _artifact_manifest(*, signal_kind: str, rank: int) -> dict:
    names = ("a", "b")
    ctrlrange = np.asarray([[0.0, 1.0]] * len(names))
    contract = {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(names),
        "actuator_ids": list(range(len(names))),
        "actuator_dyntype": ["muscle"] * len(names),
        "actuator_actnum": [1] * len(names),
        "actuator_actadr": list(range(len(names))),
        "model_na": len(names),
    }
    return {
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "signal_kind": signal_kind,
        "region": "whole_body",
        "rank": rank,
        "normalization": {},
        "source_dataset_fingerprint": "d" * 64,
        "teacher_checkpoint_fingerprint": "c" * 64,
        "fit_seed": 0,
        "transform": {
            "kind": UNIT_EXCITATION_TRANSFORM,
            "raw_signal_kind": "applied_ctrl",
            "formula": MUSCLE_EXCITATION_FORMULA,
            "ctrlrange": ctrlrange.tolist(),
            "actuator_names": list(names),
            "ctrlrange_schema_hash": ctrlrange_schema_hash(names, ctrlrange),
            "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
            "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
            "muscle_channel_contract": contract,
        },
        "split_provenance": {"train": {}, "validation": {}},
        "train_motion_uids": [1, 2],
    }


def test_synergy_baseline_is_state_only_while_coefficients_respond_to_latent():
    _require_deps()
    import jax
    import jax.numpy as jnp

    from musclemimic.latent_muscle.synergy_decoder import LatentSynergyDecoder

    decoder = LatentSynergyDecoder(
        action_dim=3,
        synergy_dim=2,
        hidden_layer_dims=(8,),
        include_baseline=True,
    )
    state = jnp.asarray([[0.25, -0.5, 0.75]], dtype=jnp.float32)
    zero = jnp.zeros((1, 2), dtype=jnp.float32)
    changed = jnp.asarray([[1.5, -0.75]], dtype=jnp.float32)
    basis = jnp.asarray([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]], dtype=jnp.float32)
    bounds = jnp.asarray([[0.0, 1.0]] * 3, dtype=jnp.float32)
    variables = decoder.init(jax.random.PRNGKey(3), state, zero, basis, bounds)

    first = decoder.apply(variables, state, zero, basis, bounds, return_aux=True)
    second = decoder.apply(variables, state, changed, basis, bounds, return_aux=True)

    np.testing.assert_allclose(first.baseline_excitation, second.baseline_excitation, atol=0.0, rtol=0.0)
    assert not np.allclose(first.synergy_coefficients, second.synergy_coefficients)
    assert not np.allclose(first.action, second.action)
    assert np.all(np.asarray(first.physical_excitation) >= 0.0)
    assert np.all(np.asarray(first.physical_excitation) <= 1.0)


def test_synergy_residual_is_hard_masked_to_declared_distal_channel():
    _require_deps()
    import jax
    import jax.numpy as jnp

    from musclemimic.latent_muscle.synergy_decoder import LatentSynergyDecoder

    decoder = LatentSynergyDecoder(
        action_dim=4,
        synergy_dim=2,
        hidden_layer_dims=(8,),
        include_baseline=False,
        residual_indices=(2,),
        residual_alpha=0.05,
    )
    state = jnp.ones((3, 5), dtype=jnp.float32)
    latent = jnp.ones((3, 2), dtype=jnp.float32)
    basis = jnp.asarray([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.8]], dtype=jnp.float32)
    bounds = jnp.asarray([[0.0, 1.0]] * 4, dtype=jnp.float32)
    variables = decoder.init(jax.random.PRNGKey(5), state, latent, basis, bounds)
    output = decoder.apply(variables, state, latent, basis, bounds, return_aux=True)
    residual = np.asarray(output.residual_excitation)
    np.testing.assert_array_equal(residual[:, [0, 1, 3]], 0.0)
    assert np.max(np.abs(residual[:, 2])) <= 0.05 + 1e-7


def test_production_loader_rejects_legacy_npz_without_test_only_flag(tmp_path):
    _require_deps()
    from musclemimic.latent_muscle.synergy_decoder import load_fixed_synergy_basis

    path = tmp_path / "legacy.npz"
    np.savez(path, basis=np.eye(2, dtype=np.float32), actuator_names=np.asarray(["a", "b"]))
    with pytest.raises(ValueError, match="formal basis artifact"):
        load_fixed_synergy_basis(path, expected_actuator_names=["a", "b"])
    loaded = load_fixed_synergy_basis(
        path,
        expected_actuator_names=["a", "b"],
        test_only_allow_legacy=True,
    )
    assert loaded.manifest["noncanonical"] is True


def test_decoder_factory_rejects_activation_basis_and_name_reordering(tmp_path):
    _require_deps()
    from musclemimic.latent_muscle.decoder_factory import build_decoder_bundle
    from musclemimic.synergy.basis_artifact import save_synergy_basis
    from musclemimic.synergy.schema import ACTIVATION_SIGNAL_KIND, EXCITATION_SIGNAL_KIND

    activation = save_synergy_basis(
        tmp_path / "activation",
        basis=np.eye(2),
        muscle_names=("a", "b"),
        manifest=_artifact_manifest(signal_kind=ACTIVATION_SIGNAL_KIND, rank=2),
    )
    with pytest.raises(ValueError, match="physical-excitation"):
        build_decoder_bundle(
            {
                "decoder_type": "fixed_synergy",
                "synergy_basis_path": str(activation.path),
                "legacy_synergy_decoder_ablation": True,
            },
            action_dim=2,
            hidden_layer_dims=(8,),
            actuator_names=("a", "b"),
        )

    excitation = save_synergy_basis(
        tmp_path / "excitation",
        basis=np.eye(2),
        muscle_names=("a", "b"),
        manifest=_artifact_manifest(signal_kind=EXCITATION_SIGNAL_KIND, rank=2),
    )
    with pytest.raises(ValueError, match="names/order"):
        build_decoder_bundle(
            {
                "decoder_type": "fixed_synergy",
                "synergy_basis_path": str(excitation.path),
                "legacy_synergy_decoder_ablation": True,
            },
            action_dim=2,
            hidden_layer_dims=(8,),
            actuator_names=("b", "a"),
        )


def test_decoder_factory_binds_expected_formal_artifact_fingerprint(tmp_path):
    _require_deps()
    from musclemimic.latent_muscle.decoder_factory import build_decoder_bundle
    from musclemimic.synergy.basis_artifact import save_synergy_basis
    from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

    artifact = save_synergy_basis(
        tmp_path / "basis",
        basis=np.eye(2),
        muscle_names=("a", "b"),
        manifest=_artifact_manifest(signal_kind=EXCITATION_SIGNAL_KIND, rank=2),
    )
    with pytest.raises(ValueError, match="expected fingerprint"):
        build_decoder_bundle(
            {
                "decoder_type": "fixed_synergy",
                "synergy_basis_path": str(artifact.path),
                "synergy_basis_expected_fingerprint": "0" * 64,
                "legacy_synergy_decoder_ablation": True,
            },
            action_dim=2,
            hidden_layer_dims=(8,),
            actuator_names=("a", "b"),
        )
    bundle = build_decoder_bundle(
        {
            "decoder_type": "fixed_synergy",
            "synergy_basis_path": str(artifact.path),
            "synergy_basis_expected_fingerprint": artifact.fingerprint,
            "legacy_synergy_decoder_ablation": True,
        },
        action_dim=2,
        hidden_layer_dims=(8,),
        actuator_names=("a", "b"),
    )
    assert bundle.synergy_basis.manifest["source_fingerprint"] == artifact.fingerprint


def test_legacy_json_duplicate_keys_are_rejected(tmp_path):
    _require_deps()
    from musclemimic.latent_muscle.synergy_decoder import load_fixed_synergy_basis

    path = tmp_path / "legacy.json"
    path.write_text(
        '{"basis": [[1]], "basis": [[2]], "actuator_names": ["a"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_fixed_synergy_basis(
            path,
            expected_actuator_names=["a"],
            test_only_allow_legacy=True,
        )


def test_synergy_checkpoint_embeds_fixed_basis_and_roundtrips_without_source(tmp_path):
    _require_deps()
    from musclemimic.latent_muscle.action_mask import ActionMask
    from musclemimic.latent_muscle.checkpoint import (
        build_latent_muscle_action_schema,
        load_latent_checkpoint,
        save_latent_checkpoint,
    )
    from musclemimic.synergy.basis_artifact import save_synergy_basis
    from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

    source = save_synergy_basis(
        tmp_path / "source_basis",
        basis=np.eye(2),
        muscle_names=("a", "b"),
        manifest=_artifact_manifest(signal_kind=EXCITATION_SIGNAL_KIND, rank=2),
    )
    checkpoint = tmp_path / "checkpoint"
    variables = {"params": {"value": np.ones(1, dtype=np.float32)}}
    save_latent_checkpoint(
        checkpoint,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state=variables,
        action_mask=ActionMask.from_correction_actuators(
            all_actuator_names=["a", "b"], correction_actuator_names=[]
        ),
        config={
            "decoder_type": "fixed_synergy",
            "latent_dim": 2,
            "action_dim": 2,
            "legacy_synergy_decoder_ablation": True,
        },
        train_metrics=[],
        eval_metrics={},
        synergy_basis=source,
        action_schema=build_latent_muscle_action_schema(
            ["a", "b"],
            muscle_channel_contract=_artifact_manifest(
                signal_kind=EXCITATION_SIGNAL_KIND,
                rank=2,
            )["transform"]["muscle_channel_contract"],
        ),
    )
    loaded = load_latent_checkpoint(checkpoint)
    assert loaded["synergy_basis"]["actuator_names"] == ["a", "b"]
    assert loaded["config"]["synergy_dim"] == 2
    assert (checkpoint / "synergy_basis.npz").is_file()
