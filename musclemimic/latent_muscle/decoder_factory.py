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
    SynergyDecoderOutput,
    coerce_synergy_basis,
    load_fixed_synergy_basis,
    normalized_to_physical,
    validate_decoder_synergy_basis,
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

    @property
    def is_synergy(self) -> bool:
        return self.synergy_basis is not None


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
) -> DecoderBundle:
    decoder_type = canonical_decoder_type(config.get("decoder_type"))
    action_dim = int(action_dim)
    bounds = _default_bounds(config, action_dim)
    if decoder_type == DIRECT_DECODER:
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
        include_baseline=bool(config.get("synergy_include_baseline", True)),
        residual_indices=residual_indices,
        residual_alpha=residual_alpha,
        baseline_init=float(config.get("synergy_baseline_init", 0.01)),
    )
    return DecoderBundle(
        decoder_type=decoder_type,
        module=module,
        excitation_bounds=np.asarray(basis.excitation_bounds, dtype=np.float32),
        synergy_basis=basis,
    )


def init_decoder(bundle: DecoderBundle, rng, state, latent):
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
