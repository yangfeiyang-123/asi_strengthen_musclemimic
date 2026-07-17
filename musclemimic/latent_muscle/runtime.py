"""Frozen prior/decoder runtime with strict checkpoint ABI validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.latent_muscle.action_mask import ActionMask
from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint
from musclemimic.latent_muscle.decoder_factory import (
    apply_decoder,
    build_decoder_bundle,
    canonical_decoder_type,
)
from musclemimic.latent_muscle.losses import positive_sigma
from musclemimic.latent_muscle.networks import ConditionalPrior
from musclemimic.latent_muscle.normalization import ObservationNormalizer


class LatentCheckpointCompatibilityError(ValueError):
    """Raised when runtime state/action contracts differ from training."""


class LatentMuscleRuntime:
    """JIT-compatible frozen conditional prior and 354-D body decoder."""

    def __init__(
        self,
        checkpoint: dict[str, Any],
        *,
        runtime_state_schema: dict[str, Any] | None = None,
        runtime_body_actuator_names: Sequence[str] | None = None,
    ) -> None:
        config = dict(checkpoint["config"])
        self.config = config
        self.checkpoint_dir = checkpoint.get("checkpoint_dir")
        self.latent_dim = int(config["latent_dim"])
        self.state_dim = int(config["student_obs_dim"])
        self.action_dim = int(config["action_dim"])
        self.decoder_type = canonical_decoder_type(config.get("decoder_type"))
        self.sigma_min = float(config.get("sigma_min", 0.05))
        self.sigma_max = float(config.get("sigma_max", 2.0))
        hidden = tuple(int(value) for value in config.get("hidden_layer_dims", (512, 256)))
        self.prior = ConditionalPrior(
            latent_dim=self.latent_dim,
            hidden_layer_dims=hidden,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
        )
        self.prior_variables = checkpoint["prior_variables"]
        self.decoder_variables = checkpoint["decoder_variables"]
        self.normalizer = ObservationNormalizer.from_manifest(checkpoint.get("obs_norm"))
        if self.normalizer.state_dim != self.state_dim:
            raise LatentCheckpointCompatibilityError(
                f"normalizer state_dim={self.normalizer.state_dim} != config state_dim={self.state_dim}"
            )
        self.state_schema = checkpoint.get("state_schema")
        self.body_obs_schema = checkpoint.get("body_obs_schema")
        self.schema_hash = None if self.state_schema is None else self.state_schema.get("schema_hash")
        self.body_obs_schema_hash = (
            None if self.body_obs_schema is None else self.body_obs_schema.get("semantic_hash")
        )
        self.action_schema = checkpoint.get("action_schema")
        self.training_provenance = checkpoint.get("training_provenance")
        self.action_mask = _mask_from_manifest(checkpoint["action_mask"])
        self.body_actuator_names = tuple(self.action_mask.body_actuator_names)
        self.decoder_bundle = build_decoder_bundle(
            config,
            action_dim=self.action_dim,
            hidden_layer_dims=hidden,
            actuator_names=self.body_actuator_names,
            checkpoint_synergy_basis=checkpoint.get("synergy_basis"),
            checkpoint_frozen_body_decoder=checkpoint.get(
                "frozen_body_decoder"
            ),
        )
        # Preserve the historical public attribute used by analysis/tests.
        self.decoder = self.decoder_bundle.module
        self.synergy_basis = self.decoder_bundle.synergy_basis
        self.frozen_body_decoder = self.decoder_bundle.frozen_body_decoder
        self.body_synergy_contract = (
            None
            if self.frozen_body_decoder is None
            else self.frozen_body_decoder.body_synergy_contract
        )
        self.body_synergy_contract_fingerprint = (
            None
            if self.body_synergy_contract is None
            else self.body_synergy_contract.contract_fingerprint
        )
        self.body_synergy_portable_core_fingerprint = (
            None
            if self.body_synergy_contract is None
            else self.body_synergy_contract.portable_decoder_core_fingerprint
        )
        self.excitation_bounds = np.asarray(
            self.decoder_bundle.excitation_bounds, dtype=np.float32
        )
        self.checkpoint_fingerprint = str(checkpoint["checkpoint_fingerprint"])
        if self.action_mask.body_size != self.action_dim:
            raise LatentCheckpointCompatibilityError(
                f"action mask body size={self.action_mask.body_size} != decoder action_dim={self.action_dim}"
            )
        _validate_state_schema(self.state_schema, expected_dim=self.state_dim)
        _validate_action_schema(self.action_schema, self.action_mask.body_actuator_names)
        ctrlrange_payload = None if self.action_schema is None else self.action_schema.get("target_ctrlrange")
        self.body_ctrlrange = (
            None if ctrlrange_payload is None else np.asarray(ctrlrange_payload, dtype=np.float64)
        )
        self.ctrlrange_schema_hash = (
            None if self.action_schema is None else self.action_schema.get("ctrlrange_schema_hash")
        )
        if bool(config.get("strict_motion_identity", False)):
            if self.body_obs_schema is None:
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint is missing body_obs_schema.json"
                )
            if self.body_ctrlrange is None or self.ctrlrange_schema_hash is None:
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint is missing ordered teacher ctrlrange"
                )
        if bool(config.get("require_dataset_provenance", False)):
            if not isinstance(self.training_provenance, dict):
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint is missing training provenance"
                )
            if not self.training_provenance.get("dataset_manifest_fingerprint") or not (
                self.training_provenance.get("teacher_checkpoint") or {}
            ).get("sha256"):
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint has incomplete dataset/teacher provenance"
                )
            promotion = self.training_provenance.get("teacher_promotion") or {}
            if not bool(config.get("test_only_allow_unpromoted_teacher", False)) and (
                not promotion.get("content_sha256")
                or not promotion.get("binding_sha256")
            ):
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint has no verified Stage-2 promotion binding"
                )
            if not bool(config.get("require_closed_loop_metrics", False)):
                raise LatentCheckpointCompatibilityError(
                    "production latent checkpoint disabled strict closed-loop promotion evidence"
                )
        self.control_manifest = {
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "state_schema_hash": self.schema_hash,
            "body_obs_schema_hash": self.body_obs_schema_hash,
            "action_mask_schema_hash": self.action_mask.schema_hash,
            "action_schema_hash": actuator_schema_hash(self.body_actuator_names),
            "ctrlrange_schema_hash": self.ctrlrange_schema_hash,
            "body_actuator_names": list(self.body_actuator_names),
            "body_ctrlrange": None if self.body_ctrlrange is None else self.body_ctrlrange.tolist(),
            "dataset_manifest_fingerprint": (
                None
                if self.training_provenance is None
                else self.training_provenance.get("dataset_manifest_fingerprint")
            ),
            "teacher_checkpoint_sha256": (
                None
                if self.training_provenance is None
                else (self.training_provenance.get("teacher_checkpoint") or {}).get("sha256")
            ),
            "teacher_promotion_manifest_sha256": (
                None
                if self.training_provenance is None
                else (self.training_provenance.get("teacher_promotion") or {}).get(
                    "content_sha256"
                )
            ),
            "teacher_promotion_binding_sha256": (
                None
                if self.training_provenance is None
                else (self.training_provenance.get("teacher_promotion") or {}).get(
                    "binding_sha256"
                )
            ),
        }
        # Do not perturb the historical direct-controller manifest/hash.  The
        # extra contract is present only for opt-in synergy checkpoints.
        if self.synergy_basis is not None:
            self.control_manifest.update(
                {
                    "decoder_type": self.decoder_type,
                    "synergy_basis_fingerprint": self.synergy_basis.fingerprint,
                }
            )
        if self.frozen_body_decoder is not None:
            self.control_manifest.update(
                {
                    "frozen_body_decoder_fingerprint": (
                        self.frozen_body_decoder.artifact_fingerprint
                    ),
                    "body_synergy_contract_fingerprint": (
                        self.body_synergy_contract_fingerprint
                    ),
                    "body_synergy_portable_core_fingerprint": (
                        self.body_synergy_portable_core_fingerprint
                    ),
                }
            )
        if (
            isinstance(self.training_provenance, dict)
            and self.training_provenance.get(
                "validation_dataset_manifest_fingerprint"
            )
            is not None
        ):
            self.control_manifest.update(
                {
                    "validation_dataset_manifest_fingerprint": self.training_provenance.get(
                        "validation_dataset_manifest_fingerprint"
                    ),
                    "motion_split_fingerprint": (
                        None
                        if checkpoint.get("split_manifest") is None
                        else checkpoint["split_manifest"].get("split_fingerprint")
                    ),
                }
            )
        if runtime_state_schema is not None:
            _require_same_schema("state", self.state_schema, runtime_state_schema)
        if runtime_body_actuator_names is not None:
            runtime_names = [str(name) for name in runtime_body_actuator_names]
            if runtime_names != self.action_mask.body_actuator_names:
                raise LatentCheckpointCompatibilityError(
                    "runtime body actuator names/order differ from latent checkpoint"
                )
            runtime_hash = actuator_schema_hash(runtime_names)
            checkpoint_hash = actuator_schema_hash(self.action_mask.body_actuator_names)
            if runtime_hash != checkpoint_hash:
                raise LatentCheckpointCompatibilityError(
                    f"runtime action schema hash={runtime_hash} != checkpoint={checkpoint_hash}"
                )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        **kwargs: Any,
    ) -> LatentMuscleRuntime:
        return cls(load_latent_checkpoint(checkpoint_dir, runtime_only=True), **kwargs)

    def normalize_numpy(self, state: Any) -> np.ndarray:
        return self.normalizer.normalize_numpy(state)

    def normalize_jax(self, state):
        return self.normalizer.normalize_jax(state)

    def prior_raw_jax(self, state, *, normalized: bool = False):
        normalized_state = jnp.asarray(state) if normalized else self.normalize_jax(state)
        return self.prior.apply(self.prior_variables, normalized_state)

    def prior_jax(self, state, *, normalized: bool = False):
        mu, raw_sigma = self.prior_raw_jax(state, normalized=normalized)
        sigma = positive_sigma(raw_sigma, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        return mu, sigma

    def decode_jax(self, state, latent, *, normalized: bool = False):
        normalized_state = jnp.asarray(state) if normalized else self.normalize_jax(state)
        latent = jnp.asarray(latent, dtype=jnp.float32)
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent last dimension must be {self.latent_dim}, got {latent.shape}")
        return apply_decoder(
            self.decoder_bundle,
            self.decoder_variables,
            normalized_state,
            latent,
        )

    def decode_components_jax(self, state, latent, *, normalized: bool = False):
        """Return action plus excitation/synergy/residual diagnostics."""

        normalized_state = jnp.asarray(state) if normalized else self.normalize_jax(state)
        latent = jnp.asarray(latent, dtype=jnp.float32)
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent last dimension must be {self.latent_dim}, got {latent.shape}")
        return apply_decoder(
            self.decoder_bundle,
            self.decoder_variables,
            normalized_state,
            latent,
            return_aux=True,
        )

    def decoder_jax(self, state, latent, *, normalized: bool = False):
        """Compatibility alias used by Stage-3 LAB control."""
        return self.decode_jax(state, latent, normalized=normalized)

    def prior_mean_action_jax(self, state):
        normalized = self.normalize_jax(state)
        prior_mu, _raw_sigma = self.prior_raw_jax(normalized, normalized=True)
        return self.decode_jax(normalized, prior_mu, normalized=True)

    def prior_numpy(self, state: Any, *, normalized: bool = False) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.prior_jax(jnp.asarray(state), normalized=normalized)
        return np.asarray(jax.device_get(mu)), np.asarray(jax.device_get(sigma))

    def prior_raw_numpy(self, state: Any, *, normalized: bool = False) -> tuple[np.ndarray, np.ndarray]:
        mu, raw_sigma = self.prior_raw_jax(jnp.asarray(state), normalized=normalized)
        return np.asarray(jax.device_get(mu)), np.asarray(jax.device_get(raw_sigma))

    def decode_numpy(self, state: Any, latent: Any, *, normalized: bool = False) -> np.ndarray:
        action = self.decode_jax(jnp.asarray(state), jnp.asarray(latent), normalized=normalized)
        return np.asarray(jax.device_get(action))

    def decode_components_numpy(self, state: Any, latent: Any, *, normalized: bool = False):
        output = self.decode_components_jax(
            jnp.asarray(state), jnp.asarray(latent), normalized=normalized
        )
        return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), output)

    def decoder_jacobian_jax(
        self,
        state,
        latent,
        *,
        normalized: bool = False,
        output: str = "physical_excitation",
    ):
        """Differentiate one decoder component with respect to latent input.

        Supports either one ``[state_dim]/[latent_dim]`` pair or aligned batches.
        This is an opt-in analysis interface and is never called by control.
        """

        normalized_state = jnp.asarray(state) if normalized else self.normalize_jax(state)
        latent_value = jnp.asarray(latent, dtype=jnp.float32)
        allowed = {
            "action",
            "physical_excitation",
            "synergy_coefficients",
            "baseline_excitation",
            "residual_excitation",
        }
        if output not in allowed:
            raise ValueError(f"unsupported decoder Jacobian output {output!r}")
        single = normalized_state.ndim == 1
        if single != (latent_value.ndim == 1):
            raise ValueError("state and latent must both be unbatched or both be batched")
        if single:
            normalized_state = normalized_state[None, :]
            latent_value = latent_value[None, :]
        if (
            normalized_state.ndim != 2
            or latent_value.ndim != 2
            or normalized_state.shape[0] != latent_value.shape[0]
            or normalized_state.shape[-1] != self.state_dim
            or latent_value.shape[-1] != self.latent_dim
        ):
            raise ValueError("decoder Jacobian inputs have incompatible state/latent shapes")

        def decode_one(one_state, one_latent):
            components = apply_decoder(
                self.decoder_bundle,
                self.decoder_variables,
                one_state,
                one_latent,
                return_aux=True,
            )
            return getattr(components, output)

        jacobian = jax.vmap(jax.jacrev(decode_one, argnums=1))(
            normalized_state, latent_value
        )
        return jacobian[0] if single else jacobian

    def decoder_jacobian_numpy(
        self,
        state: Any,
        latent: Any,
        *,
        normalized: bool = False,
        output: str = "physical_excitation",
    ) -> np.ndarray:
        return np.asarray(
            jax.device_get(
                self.decoder_jacobian_jax(
                    jnp.asarray(state),
                    jnp.asarray(latent),
                    normalized=normalized,
                    output=output,
                )
            )
        )

    def decoder_numpy(self, state: Any, latent: Any, *, normalized: bool = False) -> np.ndarray:
        """Compatibility alias used by Stage-3 LAB control."""
        return self.decode_numpy(state, latent, normalized=normalized)

    def prior_mean_action_numpy(self, state: Any) -> np.ndarray:
        return np.asarray(jax.device_get(self.prior_mean_action_jax(jnp.asarray(state))))

    def body_action_to_ctrl_jax(self, action):
        """Map normalized decoder output [-1, 1] to teacher MuJoCo ctrlrange."""
        if self.body_ctrlrange is None:
            raise LatentCheckpointCompatibilityError("checkpoint has no teacher body ctrlrange")
        value = jnp.asarray(action, dtype=jnp.float32)
        if value.shape[-1] != self.action_dim:
            raise ValueError(f"body action last dimension must be {self.action_dim}, got {value.shape}")
        limits = jnp.asarray(self.body_ctrlrange, dtype=value.dtype)
        clipped = jnp.clip(value, -1.0, 1.0)
        return limits[:, 0] + 0.5 * (clipped + 1.0) * (limits[:, 1] - limits[:, 0])

    def body_action_to_ctrl_numpy(self, action: Any) -> np.ndarray:
        return np.asarray(jax.device_get(self.body_action_to_ctrl_jax(action)))


def load_latent_runtime(
    checkpoint_dir: str | Path,
    **kwargs: Any,
) -> LatentMuscleRuntime:
    """Load and ABI-validate a frozen latent runtime."""
    return LatentMuscleRuntime.from_checkpoint(checkpoint_dir, **kwargs)


def _mask_from_manifest(payload: dict[str, Any]) -> ActionMask:
    mask = ActionMask.from_partitions(
        all_actuator_names=list(payload["all_actuator_names"]),
        body_actuator_names=list(payload["body_actuator_names"]),
        correction_actuator_names=list(payload["correction_actuator_names"]),
        neutral_actuator_names=list(payload.get("neutral_actuator_names") or []),
        neutral_values=np.asarray(payload.get("neutral_values") or [], dtype=float),
    )
    if str(payload.get("schema_hash")) != mask.schema_hash:
        raise LatentCheckpointCompatibilityError("action_mask.json schema hash mismatch")
    return mask


def _validate_state_schema(schema: dict[str, Any] | None, *, expected_dim: int) -> None:
    if schema is None:
        raise LatentCheckpointCompatibilityError("latent checkpoint is missing state_schema.json")
    if int(schema.get("state_dim", -1)) != int(expected_dim):
        raise LatentCheckpointCompatibilityError(
            f"state schema dimension={schema.get('state_dim')} != expected={expected_dim}"
        )
    supplied = schema.get("schema_hash")
    if supplied is None:
        raise LatentCheckpointCompatibilityError("state schema is missing schema_hash")
    payload = {key: value for key, value in schema.items() if key not in {"schema_hash", "provenance"}}
    actual = ordered_schema_hash(kind="student_state", payload=payload)
    if str(supplied) != actual:
        raise LatentCheckpointCompatibilityError(
            f"state schema hash mismatch: supplied={supplied} computed={actual}"
        )


def _validate_action_schema(schema: dict[str, Any] | None, body_names: Sequence[str]) -> None:
    if schema is None:
        raise LatentCheckpointCompatibilityError("latent checkpoint is missing action_schema.json")
    target_names = list(schema.get("target_actuator_names") or schema.get("actuator_names") or [])
    if target_names != list(body_names):
        raise LatentCheckpointCompatibilityError("action schema target names differ from action mask body names")
    supplied = schema.get("target_schema_hash", schema.get("action_schema_hash"))
    actual = actuator_schema_hash(target_names)
    if str(supplied) != actual:
        raise LatentCheckpointCompatibilityError(
            f"action schema hash mismatch: supplied={supplied} computed={actual}"
        )


def _require_same_schema(kind: str, expected: dict[str, Any] | None, runtime: dict[str, Any]) -> None:
    if expected is None:
        raise LatentCheckpointCompatibilityError(f"checkpoint is missing {kind} schema")
    expected_hash = expected.get("schema_hash")
    runtime_hash = runtime.get("schema_hash")
    if runtime_hash is None:
        payload = {key: value for key, value in runtime.items() if key not in {"schema_hash", "provenance"}}
        runtime_hash = ordered_schema_hash(kind="student_state", payload=payload)
    if str(runtime_hash) != str(expected_hash):
        raise LatentCheckpointCompatibilityError(
            f"runtime {kind} schema hash={runtime_hash} != checkpoint={expected_hash}"
        )
