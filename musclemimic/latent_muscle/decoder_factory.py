"""Backward-compatible construction and application of latent decoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from musclemimic.latent_muscle.networks import LatentDecoder
from musclemimic.latent_muscle.synergy_decoder import (
    LatentSynergyDecoder,
    LoadedSynergyBasis,
    PortableLatentSynergyDecoder,
    SynergyDecoderOutput,
    coerce_synergy_basis,
    load_fixed_synergy_basis,
    normalized_to_physical,
    validate_decoder_synergy_basis,
)
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoder,
    load_frozen_body_decoder,
)

DIRECT_DECODER = "direct"
FIXED_SYNERGY_DECODER = "fixed_synergy"
SYNERGY_RESIDUAL_DECODER = "synergy_residual"
SUPPORTED_DECODER_TYPES = (
    DIRECT_DECODER,
    FIXED_SYNERGY_DECODER,
    SYNERGY_RESIDUAL_DECODER,
)


@dataclass(frozen=True)
class DecoderBundle:
    decoder_type: str
    module: Any
    excitation_bounds: np.ndarray
    synergy_basis: LoadedSynergyBasis | None = None
    frozen_body_decoder: FrozenBodyDecoder | None = None
    legacy_synergy_ablation: bool = False

    @property
    def is_synergy(self) -> bool:
        return self.synergy_basis is not None or self.frozen_body_decoder is not None


def canonical_decoder_type(value: str | None) -> str:
    """Return a stable type; missing fields preserve every legacy checkpoint."""

    name = DIRECT_DECODER if value is None else str(value).strip().lower()
    aliases = {
        "mlp": DIRECT_DECODER,
        "latent_decoder": DIRECT_DECODER,
        "synergy": FIXED_SYNERGY_DECODER,
        "latent_synergy": FIXED_SYNERGY_DECODER,
        "latent_synergy_residual": SYNERGY_RESIDUAL_DECODER,
    }
    name = aliases.get(name, name)
    if name not in SUPPORTED_DECODER_TYPES:
        raise ValueError(f"unsupported decoder_type={value!r}; expected one of {SUPPORTED_DECODER_TYPES}")
    return name


def build_decoder_bundle(
    config: Mapping[str, Any],
    *,
    action_dim: int,
    hidden_layer_dims: Sequence[int],
    actuator_names: Sequence[str],
    checkpoint_synergy_basis: Mapping[str, Any] | LoadedSynergyBasis | None = None,
    checkpoint_frozen_body_decoder: FrozenBodyDecoder | None = None,
) -> DecoderBundle:
    decoder_type = canonical_decoder_type(config.get("decoder_type"))
    action_dim = int(action_dim)
    bounds = _default_bounds(config, action_dim)
    if decoder_type == DIRECT_DECODER:
        conflicting = {
            "frozen_body_decoder_path": config.get("frozen_body_decoder_path"),
            "synergy_basis_path": config.get("synergy_basis_path"),
            "legacy_synergy_decoder_ablation": config.get("legacy_synergy_decoder_ablation", False),
            "synergy_include_baseline": config.get("synergy_include_baseline", False),
            "synergy_residual_actuator_names": config.get("synergy_residual_actuator_names"),
        }
        enabled = [key for key, value in conflicting.items() if bool(value)]
        if enabled or checkpoint_synergy_basis is not None or checkpoint_frozen_body_decoder is not None:
            raise ValueError(f"direct latent decoder cannot carry synergy/legacy decoder settings: fields={enabled}")
        module = LatentDecoder(
            action_dim=action_dim,
            hidden_layer_dims=tuple(int(value) for value in hidden_layer_dims),
            bounded_action=bool(config.get("bounded_action", True)),
        )
        return DecoderBundle(
            decoder_type=decoder_type,
            module=module,
            excitation_bounds=bounds,
        )

    frozen_decoder = checkpoint_frozen_body_decoder
    frozen_path = config.get("frozen_body_decoder_path")
    if frozen_decoder is None and frozen_path:
        frozen_decoder = load_frozen_body_decoder(
            frozen_path,
            expected_actuator_names=actuator_names,
            expected_artifact_fingerprint=config.get("frozen_body_decoder_expected_fingerprint"),
            expected_portable_decoder_core_fingerprint=config.get("body_synergy_portable_core_expected_fingerprint"),
        )
    if frozen_decoder is not None:
        if bool(config.get("legacy_synergy_decoder_ablation", False)):
            raise ValueError("portable frozen decoder cannot be mixed with legacy_synergy_decoder_ablation")
        if config.get("synergy_basis_path") or checkpoint_synergy_basis is not None:
            raise ValueError("portable frozen decoder cannot be mixed with a standalone W path/payload")
        if bool(config.get("synergy_include_baseline", False)):
            raise ValueError("portable synergy decoder forbids a learned 354-D baseline")
        if config.get("synergy_residual_actuator_names"):
            raise ValueError(
                "portable synergy decoder takes structured R from its contract; "
                "direct residual actuator indices are forbidden"
            )
        if float(config.get("synergy_residual_alpha", 0.0)) != 0.0:
            raise ValueError("portable synergy decoder takes residual alpha from its contract")
        contract = frozen_decoder.body_synergy_contract
        expected_mode = FIXED_SYNERGY_DECODER if decoder_type == FIXED_SYNERGY_DECODER else "fixed_synergy_residual"
        if contract.mode != expected_mode:
            raise ValueError("latent decoder_type differs from frozen BodySynergyContractV2 mode")
        if frozen_decoder.body_action_dim != action_dim:
            raise ValueError("frozen body decoder action dimension differs from latent action dimension")
        if tuple(str(name) for name in actuator_names) != frozen_decoder.actuator_names:
            raise ValueError("frozen body decoder actuator names/order differ from latent body schema")
        expected_contract = config.get("body_synergy_contract_expected_fingerprint")
        if expected_contract not in (None, "") and str(expected_contract) != (contract.contract_fingerprint):
            raise ValueError("latent expected BodySynergyContractV2 fingerprint mismatch")
        basis = LoadedSynergyBasis(
            basis=np.asarray(frozen_decoder.basis, dtype=np.float32),
            actuator_names=frozen_decoder.actuator_names,
            excitation_bounds=np.asarray(frozen_decoder.excitation_bounds, dtype=np.float32),
            fingerprint=str(contract.runtime_basis_fingerprint),
            manifest={
                "source_fingerprint": contract.basis_fingerprint,
                "runtime_fingerprint": contract.runtime_basis_fingerprint,
                "rank": contract.basis_rank,
                "action_dim": contract.body_action_dim,
                "signal_kind": "physical_excitation_unit",
                "portable_decoder_core_fingerprint": (contract.portable_decoder_core_fingerprint),
                "frozen_body_decoder_fingerprint": (frozen_decoder.artifact_fingerprint),
            },
            source_path=(None if frozen_path is None else str(frozen_path)),
        )
        module = PortableLatentSynergyDecoder(
            action_dim=action_dim,
            synergy_dim=frozen_decoder.synergy_dim,
            residual_dim=frozen_decoder.residual_dim,
            hidden_layer_dims=tuple(int(value) for value in hidden_layer_dims),
        )
        return DecoderBundle(
            decoder_type=decoder_type,
            module=module,
            excitation_bounds=np.asarray(frozen_decoder.excitation_bounds, dtype=np.float32),
            synergy_basis=basis,
            frozen_body_decoder=frozen_decoder,
            legacy_synergy_ablation=False,
        )

    if not bool(config.get("legacy_synergy_decoder_ablation", False)):
        raise ValueError(
            f"decoder_type={decoder_type} requires frozen_body_decoder_path (or an "
            "embedded checkpoint artifact); W-only latent decoding is available only "
            "with legacy_synergy_decoder_ablation=true"
        )

    if checkpoint_synergy_basis is not None:
        basis = coerce_synergy_basis(
            checkpoint_synergy_basis,
            expected_actuator_names=actuator_names,
            default_excitation_min=float(config.get("physical_excitation_min", 0.0)),
            default_excitation_max=float(config.get("physical_excitation_max", 1.0)),
        )
    else:
        basis_path = config.get("synergy_basis_path")
        if not basis_path:
            raise ValueError(f"decoder_type={decoder_type} requires synergy_basis_path")
        basis = load_fixed_synergy_basis(
            basis_path,
            expected_actuator_names=actuator_names,
            default_excitation_min=float(config.get("physical_excitation_min", 0.0)),
            default_excitation_max=float(config.get("physical_excitation_max", 1.0)),
            test_only_allow_legacy=bool(config.get("test_only_allow_legacy_synergy_basis", False)),
        )
    if basis.action_dim != action_dim:
        raise ValueError(f"synergy basis action_dim={basis.action_dim} != decoder action_dim={action_dim}")
    validate_decoder_synergy_basis(
        basis,
        allow_noncanonical=bool(config.get("test_only_allow_legacy_synergy_basis", False)),
    )
    expected_fingerprint = config.get("synergy_basis_expected_fingerprint")
    if expected_fingerprint is not None:
        source_fingerprint = basis.manifest.get("source_fingerprint", basis.fingerprint)
        if str(expected_fingerprint) != str(source_fingerprint):
            raise ValueError(
                "synergy basis expected fingerprint mismatch: "
                f"expected={expected_fingerprint} actual={source_fingerprint}"
            )

    residual_names = tuple(str(name) for name in (config.get("synergy_residual_actuator_names") or ()))
    if decoder_type == FIXED_SYNERGY_DECODER and residual_names:
        raise ValueError("fixed_synergy decoder cannot configure residual actuator names")
    if decoder_type == SYNERGY_RESIDUAL_DECODER and not residual_names:
        raise ValueError("synergy_residual decoder requires synergy_residual_actuator_names")
    name_to_index = {str(name): index for index, name in enumerate(actuator_names)}
    missing = [name for name in residual_names if name not in name_to_index]
    if missing:
        raise ValueError(f"synergy residual actuators are absent from body schema: {missing}")
    residual_indices = tuple(name_to_index[name] for name in residual_names)
    residual_alpha = float(config.get("synergy_residual_alpha", 0.0))
    if decoder_type == SYNERGY_RESIDUAL_DECODER and residual_alpha <= 0.0:
        raise ValueError("synergy_residual decoder requires synergy_residual_alpha > 0")

    module = LatentSynergyDecoder(
        action_dim=action_dim,
        synergy_dim=basis.synergy_dim,
        hidden_layer_dims=tuple(int(value) for value in hidden_layer_dims),
        # A learned 354-D state baseline can reconstruct actions outside span(W)
        # and therefore invalidates the primary fixed-synergy comparison.  It is
        # available only through an explicit ablation override.
        include_baseline=bool(config.get("synergy_include_baseline", False)),
        residual_indices=residual_indices,
        residual_alpha=residual_alpha,
        baseline_init=float(config.get("synergy_baseline_init", 0.01)),
    )
    return DecoderBundle(
        decoder_type=decoder_type,
        module=module,
        excitation_bounds=np.asarray(basis.excitation_bounds, dtype=np.float32),
        synergy_basis=basis,
        legacy_synergy_ablation=True,
    )


def init_decoder(bundle: DecoderBundle, rng, state, latent):
    if bundle.frozen_body_decoder is not None:
        return bundle.module.init(
            rng,
            state,
            latent,
            bundle.frozen_body_decoder.jax_params(),
        )
    if bundle.synergy_basis is None:
        return bundle.module.init(rng, state, latent)
    return bundle.module.init(
        rng,
        state,
        latent,
        jnp.asarray(bundle.synergy_basis.basis),
        jnp.asarray(bundle.excitation_bounds),
    )


def apply_decoder(
    bundle: DecoderBundle,
    variables,
    state,
    latent,
    *,
    return_aux: bool = False,
):
    if bundle.frozen_body_decoder is not None:
        return bundle.module.apply(
            variables,
            state,
            latent,
            bundle.frozen_body_decoder.jax_params(),
            return_aux=return_aux,
        )
    if bundle.synergy_basis is not None:
        return bundle.module.apply(
            variables,
            state,
            latent,
            jnp.asarray(bundle.synergy_basis.basis),
            jnp.asarray(bundle.excitation_bounds),
            return_aux=return_aux,
        )
    action = bundle.module.apply(variables, state, latent)
    if not return_aux:
        return action
    action = jnp.asarray(action)
    empty = jnp.zeros((*action.shape[:-1], 0), dtype=action.dtype)
    excitation = normalized_to_physical(action, jnp.asarray(bundle.excitation_bounds))
    return SynergyDecoderOutput(
        action=action,
        physical_excitation=excitation,
        synergy_coefficients=empty,
        baseline_excitation=jnp.zeros_like(action),
        residual_excitation=empty,
    )


def _default_bounds(config: Mapping[str, Any], action_dim: int) -> np.ndarray:
    lower = float(config.get("physical_excitation_min", 0.0))
    upper = float(config.get("physical_excitation_max", 1.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("physical excitation bounds must be finite and increasing")
    if lower < 0.0:
        raise ValueError("physical excitation must be non-negative; convert signed control before using synergy losses")
    return np.tile(np.asarray([[lower, upper]], dtype=np.float32), (int(action_dim), 1))
