from __future__ import annotations

import json
import shutil
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.dataset import write_distill_shard
from musclemimic.latent_muscle.action_mask import ActionMask
from musclemimic.latent_muscle.checkpoint import (
    load_latent_checkpoint,
    save_latent_checkpoint,
)
from musclemimic.latent_muscle.synergy_decoder import (
    PortableLatentSynergyDecoder,
    portable_decoder_output_from_raw,
)
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoder,
    load_frozen_body_decoder,
)
from musclemimic.synergy.multistage_contract import BodySynergyContractV2
from musclemimic.synergy.schema import ctrlrange_schema_hash

NAMES = ("a", "b", "c")
BOUNDS = np.asarray([[0.0, 1.0]] * 3, dtype=np.float32)


def _contract(
    *,
    names: tuple[str, ...] = NAMES,
    runtime_model_hash: str = "1" * 64,
    physical_interface_hash: str = "2" * 64,
    residual_alpha: float = 0.05,
    transform_fingerprint: str = "5" * 64,
    tonic_fingerprint: str = "7" * 64,
) -> BodySynergyContractV2:
    bounds = np.asarray([[0.0, 1.0]] * len(names), dtype=np.float64)
    control_hash = ctrlrange_schema_hash(names, bounds)
    return BodySynergyContractV2(
        mode="fixed_synergy_residual",
        body_action_dim=len(names),
        policy_action_dim=3,
        actuator_names=names,
        actuator_schema_hash=actuator_schema_hash(names),
        control_range_hash=control_hash,
        runtime_control_range_hash=control_hash,
        runtime_model_hash=runtime_model_hash,
        physical_action_interface_hash=physical_interface_hash,
        basis_fingerprint="3" * 64,
        runtime_basis_fingerprint="4" * 64,
        basis_rank=2,
        coefficient_transform_fingerprint=transform_fingerprint,
        coefficient_statistics_fingerprint="6" * 64,
        tonic_baseline_fingerprint=tonic_fingerprint,
        residual_basis_fingerprint="8" * 64,
        residual_fit_contract_fingerprint="9" * 64,
        residual_allowed_muscle_mask_fingerprint="a" * 64,
        residual_dim=1,
        residual_alpha=residual_alpha,
    )


def _decoder(*, contract: BodySynergyContractV2 | None = None) -> FrozenBodyDecoder:
    return FrozenBodyDecoder(
        body_synergy_contract=_contract() if contract is None else contract,
        basis=np.asarray(
            [[0.45, 0.05], [0.15, 0.40], [0.20, 0.25]],
            dtype=np.float32,
        ),
        excitation_bounds=BOUNDS,
        coefficient_maximum=np.asarray([0.5, 0.7], dtype=np.float32),
        coefficient_center=np.asarray([0.1, 0.2], dtype=np.float32),
        coefficient_temperature=np.asarray([0.8, 1.2], dtype=np.float32),
        tonic_baseline=np.asarray([0.02, 0.03, 0.01], dtype=np.float32),
        residual_basis=np.asarray([[0.2], [-0.1], [0.05]], dtype=np.float32),
    )


def _dataset_metadata(decoder: FrozenBodyDecoder) -> dict:
    contract = decoder.body_synergy_contract
    return {
        "actuator_names": list(decoder.actuator_names),
        "body_synergy_contract": contract.to_manifest(),
        "body_synergy_contract_fingerprint": contract.contract_fingerprint,
        "body_synergy_portable_core_fingerprint": (
            contract.portable_decoder_core_fingerprint
        ),
        "frozen_body_decoder_fingerprint": decoder.artifact_fingerprint,
        "teacher_policy_action_semantics": "clipped_raw_c_rho_coordinates",
        "teacher_policy_action_dim": decoder.policy_action_dim,
    }


def test_early_and_portable_latent_raw_coordinates_decode_identically_under_jit():
    decoder = _decoder()
    raw = jnp.asarray(
        [[-0.4, 0.7, -1.2], [0.1, -0.2, 1.5]], dtype=jnp.float32
    )

    early = decoder.decode(raw)
    latent = portable_decoder_output_from_raw(raw, decoder.jax_params())
    compiled = jax.jit(decoder.decode)(raw)

    np.testing.assert_allclose(early.body_action, latent.action, atol=1e-7)
    np.testing.assert_allclose(
        early.physical_excitation, latent.physical_excitation, atol=1e-7
    )
    np.testing.assert_allclose(
        early.synergy_coefficients, latent.synergy_coefficients, atol=1e-7
    )
    np.testing.assert_allclose(
        early.residual_excitation, latent.residual_excitation, atol=1e-7
    )
    np.testing.assert_allclose(compiled.body_action, early.body_action, atol=1e-7)


def test_portable_latent_network_zero_head_executes_same_frozen_decoder():
    decoder = _decoder()
    module = PortableLatentSynergyDecoder(
        action_dim=3,
        synergy_dim=2,
        residual_dim=1,
        hidden_layer_dims=(8,),
    )
    state = jnp.zeros((2, 4), dtype=jnp.float32)
    latent = jnp.zeros((2, 3), dtype=jnp.float32)
    variables = module.init(
        jax.random.PRNGKey(0), state, latent, decoder.jax_params()
    )
    output = module.apply(
        variables,
        state,
        latent,
        decoder.jax_params(),
        return_aux=True,
    )
    expected = decoder.decode(jnp.zeros((2, 3), dtype=jnp.float32))
    np.testing.assert_allclose(output.action, expected.body_action, atol=1e-7)
    np.testing.assert_allclose(
        output.physical_excitation, expected.physical_excitation, atol=1e-7
    )
    flattened_names = "/".join(
        str(key)
        for path, _value in jax.tree_util.tree_leaves_with_path(variables["params"])
        for key in path
    )
    assert "baseline" not in flattened_names
    assert "raw_synergy_coefficients" in flattened_names
    assert "raw_structured_residual" in flattened_names


def test_frozen_decoder_roundtrip_and_stage_runtime_change_is_portable(tmp_path):
    stage1 = _decoder()
    artifact = stage1.save(tmp_path / "decoder")
    loaded = load_frozen_body_decoder(
        artifact,
        expected_artifact_fingerprint=stage1.artifact_fingerprint,
        expected_portable_decoder_core_fingerprint=(
            stage1.body_synergy_contract.portable_decoder_core_fingerprint
        ),
    )
    assert loaded.artifact_fingerprint == stage1.artifact_fingerprint

    stage2_contract = replace(
        stage1.body_synergy_contract,
        runtime_model_hash="b" * 64,
        physical_action_interface_hash="c" * 64,
    )
    stage2 = _decoder(contract=stage2_contract)
    stage1.body_synergy_contract.assert_portable_compatible(stage2_contract)
    with pytest.raises(ValueError, match="exact runtime"):
        stage1.body_synergy_contract.assert_exact_runtime_compatible(
            stage2_contract
        )
    assert stage2.artifact_fingerprint == stage1.artifact_fingerprint


@pytest.mark.parametrize(
    "field",
    (
        "basis",
        "coefficient_maximum",
        "coefficient_center",
        "coefficient_temperature",
        "tonic_baseline",
        "residual_basis",
    ),
)
def test_frozen_decoder_rejects_tampered_numerical_core(tmp_path, field):
    artifact = _decoder().save(tmp_path / "source")
    target = tmp_path / field
    shutil.copytree(artifact, target)
    array_path = target / "frozen_body_decoder.npz"
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    arrays[field].flat[0] += 0.01
    np.savez_compressed(array_path, **arrays)
    with pytest.raises(ValueError, match="array file fingerprint"):
        load_frozen_body_decoder(target)


def test_frozen_decoder_rejects_tampered_alpha_and_actuator_order(tmp_path):
    source = _decoder().save(tmp_path / "source")
    for name, mutate in (
        (
            "alpha",
            lambda payload: payload.__setitem__(
                "residual_alpha", payload["residual_alpha"] + 0.01
            ),
        ),
        (
            "order",
            lambda payload: payload.__setitem__(
                "actuator_names", list(reversed(payload["actuator_names"]))
            ),
        ),
    ):
        target = tmp_path / name
        shutil.copytree(source, target)
        manifest_path = target / "frozen_body_decoder.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest_fingerprint"):
            load_frozen_body_decoder(target)


def test_each_semantic_core_change_changes_portable_artifact_fingerprint():
    reference = _decoder()
    variants = []
    for field in (
        "basis",
        "coefficient_maximum",
        "coefficient_center",
        "coefficient_temperature",
        "tonic_baseline",
        "residual_basis",
    ):
        values = {
            key: np.asarray(value).copy()
            for key, value in reference._array_payload().items()
        }
        values[field].flat[0] += 0.005
        variants.append(
            FrozenBodyDecoder(
                body_synergy_contract=reference.body_synergy_contract,
                **values,
            )
        )
    alpha_contract = replace(
        reference.body_synergy_contract, residual_alpha=0.06
    )
    variants.append(_decoder(contract=alpha_contract))
    transform_contract = replace(
        reference.body_synergy_contract,
        coefficient_transform_fingerprint="d" * 64,
    )
    variants.append(_decoder(contract=transform_contract))
    tonic_contract = replace(
        reference.body_synergy_contract,
        tonic_baseline_fingerprint="e" * 64,
    )
    variants.append(_decoder(contract=tonic_contract))
    assert all(
        value.artifact_fingerprint != reference.artifact_fingerprint
        for value in variants
    )


def test_mixed_dataset_body_synergy_contract_is_rejected(tmp_path):
    first = _decoder()
    changed_contract = replace(
        first.body_synergy_contract,
        coefficient_transform_fingerprint="d" * 64,
    )
    second = _decoder(contract=changed_contract)
    data = {
        "student_obs": np.zeros((2, 4), dtype=np.float32),
        "teacher_action": np.zeros((2, 3), dtype=np.float32),
    }
    write_distill_shard(
        tmp_path / "shard_000000.npz",
        data,
        metadata=_dataset_metadata(first),
    )
    with pytest.raises(ValueError, match="ABI metadata mismatch"):
        write_distill_shard(
            tmp_path / "shard_000001.npz",
            data,
            metadata=_dataset_metadata(second),
        )


def test_portable_latent_checkpoint_roundtrip_and_tamper_fail_closed(tmp_path):
    decoder = _decoder()
    checkpoint = tmp_path / "checkpoint"
    variables = {"params": {"value": np.ones(1, dtype=np.float32)}}
    save_latent_checkpoint(
        checkpoint,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state=variables,
        action_mask=ActionMask.from_correction_actuators(
            all_actuator_names=list(NAMES), correction_actuator_names=[]
        ),
        config={
            "decoder_type": "synergy_residual",
            "latent_dim": 2,
            "action_dim": 3,
        },
        train_metrics=[],
        eval_metrics={},
        frozen_body_decoder=decoder,
    )
    loaded = load_latent_checkpoint(checkpoint)
    assert loaded["frozen_body_decoder"].artifact_fingerprint == (
        decoder.artifact_fingerprint
    )
    assert loaded["body_synergy_contract_fingerprint"] == (
        decoder.body_synergy_contract.contract_fingerprint
    )
    assert loaded["body_synergy_portable_core_fingerprint"] == (
        decoder.body_synergy_contract.portable_decoder_core_fingerprint
    )
    assert loaded["synergy_basis"] is None

    array_path = checkpoint / "frozen_body_decoder.npz"
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    arrays["basis"][0, 0] += 0.01
    np.savez_compressed(array_path, **arrays)
    with pytest.raises(ValueError, match="array file fingerprint"):
        load_latent_checkpoint(checkpoint)
