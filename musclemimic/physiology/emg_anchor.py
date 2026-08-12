"""Partial-EMG anchoring losses over the measured activation subspace.

This module owns the *simulation side* of the PEASD contract and the comparison
between the two sides.  It provides three things:

* :func:`build_emg_observation_projection` materialises the name-safe
  ``M <- 354`` observation operator ``P`` from a reviewed mapping.  ``P`` states
  which simulated muscles an electrode observes; it is not invertible and never
  claims to recover 354 activations from ``M`` channels;
* :func:`tube_distance` penalises only the part of a simulated signal that
  leaves the range real humans produced at the same movement phase, so a policy
  inside natural variability pays nothing;
* :func:`emg_anchor_metrics` and :func:`emg_synergy_metrics` compute the
  activation-anchor and coordination-consistency terms, always returning
  diagnostics whether or not a reward coefficient is armed.

Everything after spec construction is JAX and safe under ``jit``/``vmap``: the
phase lookup is a static gather over a fixed bin count, never data-dependent
control flow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from musclemimic.physiology.emg_reference import (
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    EmgPhaseReferenceTube,
    emg_reference_fingerprint,
    synergy_projection_matrix,
)

EMG_ANCHOR_LOSS_SPEC_SCHEMA_VERSION = "emg_anchor_loss_spec_v1"
"""Schema of the identity block pinned alongside a compiled anchor spec."""

EMG_ANCHOR_LOSS_METHOD = "phase_tube_huber_v1"
"""Outside-tube Huber distance in units of the local robust scale."""

EMG_ANCHOR_SIGNAL = "activation"
"""Compared quantity: ``data.act`` read through ``actuator_actadr``."""

EMG_EXCLUDED_MAPPING_STATUS = "excluded_no_verified_model_homolog"
"""Channels carrying this status are dropped from ``P`` rather than guessed at."""

EMG_ANCHOR_EPS = 1e-8

DEFAULT_TUBE_KAPPA = 1.0
"""Tube half-width in robust scales; deviations below this cost nothing."""

DEFAULT_HUBER_DELTA = 1.0
"""Outside-tube residual where the penalty turns from quadratic to linear."""


class EmgAnchorMetrics(NamedTuple):
    """Activation-anchor aggregate plus always-on diagnostics."""

    loss: jax.Array
    violation_fraction: jax.Array
    mean_abs_deviation: jax.Array
    max_abs_deviation: jax.Array
    pattern_correlation: jax.Array
    valid_channel_fraction: jax.Array
    channel_loss: jax.Array
    projected_activation: jax.Array


class EmgSynergyMetrics(NamedTuple):
    """Coordination-consistency aggregate split into shape and intensity."""

    loss: jax.Array
    shape_loss: jax.Array
    intensity_loss: jax.Array
    shape_cosine: jax.Array
    intensity: jax.Array
    reference_intensity: jax.Array
    coefficients: jax.Array


@struct.dataclass
class EmgAnchorSpec:
    """Static arrays consumed inside JIT-compiled anchoring diagnostics."""

    projection: jax.Array
    activation_addresses: jax.Array
    synergy_projection: jax.Array
    anchor_mean: jax.Array
    anchor_scale: jax.Array
    anchor_valid: jax.Array
    amplitude_confidence: jax.Array
    synergy_mean: jax.Array
    synergy_scale: jax.Array
    synergy_valid: jax.Array
    phase_bin_count: int = struct.field(pytree_node=False, default=1)
    channel_names: tuple[str, ...] = struct.field(pytree_node=False, default=())
    action_ids: tuple[str, ...] = struct.field(pytree_node=False, default=())
    signal: str = struct.field(pytree_node=False, default=EMG_ANCHOR_SIGNAL)

    @property
    def channel_count(self) -> int:
        return int(self.projection.shape[0])

    @property
    def synergy_count(self) -> int:
        return int(self.synergy_projection.shape[0])


@dataclass(frozen=True)
class EmgAnchorSpecIdentity:
    """Fingerprinted description of exactly which tube a runtime spec compiled."""

    schema_version: str
    reference_id: str
    reference_fingerprint: str
    mapping_id: str
    mapping_sha256: str
    actuator_schema_hash: str
    muscle_channel_core_fingerprint: str
    signal: str
    method: str
    channel_names: tuple[str, ...]
    action_ids: tuple[str, ...]
    phase_bin_count: int
    synergy_count: int
    tube_kappa: float
    huber_delta: float
    eps: float
    synergy_shape_weight: float
    synergy_intensity_weight: float
    projection_method: str
    projection_ridge: float
    normalization_schema_version: str
    audit_normalization: str
    model_normalization: str
    loss_spec_fingerprint: str

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "reference_fingerprint": self.reference_fingerprint,
            "mapping_id": self.mapping_id,
            "mapping_sha256": self.mapping_sha256,
            "actuator_schema_hash": self.actuator_schema_hash,
            "muscle_channel_core_fingerprint": self.muscle_channel_core_fingerprint,
            "signal": self.signal,
            "method": self.method,
            "channel_names": list(self.channel_names),
            "action_ids": list(self.action_ids),
            "phase_bin_count": self.phase_bin_count,
            "synergy_count": self.synergy_count,
            "tube_kappa": self.tube_kappa,
            "huber_delta": self.huber_delta,
            "eps": self.eps,
            "synergy_shape_weight": self.synergy_shape_weight,
            "synergy_intensity_weight": self.synergy_intensity_weight,
            "projection_method": self.projection_method,
            "projection_ridge": self.projection_ridge,
            "normalization_schema_version": self.normalization_schema_version,
            "audit_normalization": self.audit_normalization,
            "model_normalization": self.model_normalization,
        }
        payload["loss_spec_fingerprint"] = self.loss_spec_fingerprint
        return payload


def emg_anchor_loss_spec_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint an anchor loss-spec payload, ignoring any embedded digest."""

    material = {str(key): value for key, value in payload.items()}
    material.pop("loss_spec_fingerprint", None)
    return emg_reference_fingerprint(material)


def build_emg_observation_projection(
    mapping: Mapping[str, Any],
    actuator_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Materialise the name-safe ``M x 354`` observation operator ``P``.

    Row ``m`` holds the observation weights of electrode ``m`` over the ordered
    actuator vector, so ``y = P @ a`` projects a simulated activation vector onto
    the measured subspace.  Channels flagged
    ``excluded_no_verified_model_homolog`` are dropped, keeping ``P`` row order
    aligned with the returned channel names.  Unknown actuator names raise
    rather than being silently skipped.
    """

    names = [str(name) for name in actuator_names]
    if len(set(names)) != len(names):
        raise ValueError("actuator_names must be unique to build a name-safe projection")
    index_of = {name: position for position, name in enumerate(names)}

    channels = mapping.get("channels")
    if not isinstance(channels, Sequence) or not channels:
        raise ValueError("EMG mapping must carry a non-empty channels list")

    rows: list[np.ndarray] = []
    channel_names: list[str] = []
    for position, entry in enumerate(channels):
        if not isinstance(entry, Mapping):
            raise ValueError(f"EMG mapping channel {position} must be a mapping")
        if str(entry.get("mapping_status")) == EMG_EXCLUDED_MAPPING_STATUS:
            continue
        channel_name = str(entry.get("emg_channel") or "").strip()
        if not channel_name:
            raise ValueError(f"EMG mapping channel {position} has no emg_channel name")
        actuators = [str(item) for item in entry.get("simulation_actuators", ())]
        weights = np.asarray(entry.get("weights", ()), dtype=np.float64)
        if not actuators:
            raise ValueError(f"channel {channel_name!r} maps to no simulation actuator")
        if weights.shape != (len(actuators),):
            raise ValueError(f"channel {channel_name!r} has {len(actuators)} actuators but {weights.size} weights")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError(f"channel {channel_name!r} weights must be finite and non-negative")
        if abs(float(np.sum(weights)) - 1.0) > 1e-8:
            raise ValueError(f"channel {channel_name!r} weights must sum to one")
        missing = [name for name in actuators if name not in index_of]
        if missing:
            raise ValueError(
                f"channel {channel_name!r} maps simulation actuators absent from the ordered action vector: {missing}"
            )
        if len(set(actuators)) != len(actuators):
            raise ValueError(f"channel {channel_name!r} repeats a simulation actuator")
        row = np.zeros(len(names), dtype=np.float64)
        for actuator, weight in zip(actuators, weights, strict=True):
            row[index_of[actuator]] = weight
        rows.append(row)
        channel_names.append(channel_name)

    if not rows:
        raise ValueError("EMG mapping produced no comparable channel")
    return np.stack(rows, axis=0), channel_names


def build_emg_anchor_spec(
    tube: EmgPhaseReferenceTube,
    mapping: Mapping[str, Any],
    *,
    actuator_names: Sequence[str],
    activation_addresses: Any,
    muscle_channel_core_fingerprint: str,
    tube_kappa: float = DEFAULT_TUBE_KAPPA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    synergy_shape_weight: float = 1.0,
    synergy_intensity_weight: float = 0.25,
    eps: float = EMG_ANCHOR_EPS,
) -> tuple[EmgAnchorSpec, EmgAnchorSpecIdentity]:
    """Compile a tube plus mapping into JIT-ready arrays and a pinned identity.

    The compiled channel order is cross-checked against the tube's own channel
    order; divergence raises rather than silently comparing mismatched
    electrodes.  Follows ``build_continuity_loss_spec``: build, fingerprint,
    re-validate.
    """

    projection, channel_names = build_emg_observation_projection(mapping, actuator_names)
    if tuple(channel_names) != tube.channel_names:
        for index, (compiled, expected) in enumerate(zip(channel_names, tube.channel_names, strict=False)):
            if compiled != expected:
                raise ValueError(
                    "compiled EMG channel order diverges from the reference tube at index "
                    f"{index}: mapping says {compiled!r}, tube says {expected!r}"
                )
        raise ValueError(
            f"compiled EMG mapping yields {len(channel_names)} channels but the tube declares {tube.channel_count}"
        )

    kappa = _positive_float(tube_kappa, field="tube_kappa")
    delta = _positive_float(huber_delta, field="huber_delta")
    shape_weight = _nonnegative_float(synergy_shape_weight, field="synergy_shape_weight")
    intensity_weight = _nonnegative_float(synergy_intensity_weight, field="synergy_intensity_weight")
    if shape_weight <= 0.0 and intensity_weight <= 0.0:
        raise ValueError("synergy consistency requires a positive shape or intensity weight")
    if intensity_weight > shape_weight:
        raise ValueError(
            "synergy intensity weight must not exceed the shape weight: cross-system "
            "amplitude calibration is far less reliable than coordination shape"
        )
    epsilon = _positive_float(eps, field="eps")

    addresses = np.asarray(activation_addresses, dtype=np.int64)
    if addresses.ndim != 1 or addresses.shape[0] != projection.shape[1]:
        raise ValueError(
            f"activation_addresses must have one entry per ordered actuator "
            f"({projection.shape[1]}), found {addresses.shape}"
        )

    ridge = float(tube.synergy_binding["projection_ridge"])
    synergy_projection = synergy_projection_matrix(tube.synergy_basis, ridge=ridge)

    spec = EmgAnchorSpec(
        projection=jnp.asarray(projection, dtype=jnp.float32),
        activation_addresses=jnp.asarray(addresses, dtype=jnp.int32),
        synergy_projection=jnp.asarray(synergy_projection, dtype=jnp.float32),
        anchor_mean=jnp.asarray(tube.anchor_mean, dtype=jnp.float32),
        anchor_scale=jnp.asarray(tube.anchor_scale, dtype=jnp.float32),
        anchor_valid=jnp.asarray(tube.anchor_valid, dtype=jnp.float32),
        amplitude_confidence=jnp.asarray(tube.amplitude_confidence, dtype=jnp.float32),
        synergy_mean=jnp.asarray(tube.synergy_mean, dtype=jnp.float32),
        synergy_scale=jnp.asarray(tube.synergy_scale, dtype=jnp.float32),
        synergy_valid=jnp.asarray(tube.synergy_valid, dtype=jnp.float32),
        phase_bin_count=int(tube.phase_bin_count),
        channel_names=tuple(channel_names),
        action_ids=tube.action_ids,
        signal=EMG_ANCHOR_SIGNAL,
    )

    payload = {
        "schema_version": EMG_ANCHOR_LOSS_SPEC_SCHEMA_VERSION,
        "reference_id": tube.reference_id,
        "reference_fingerprint": tube.reference_fingerprint,
        "mapping_id": str(mapping.get("mapping_id", "")),
        "mapping_sha256": str(tube.mapping_binding["mapping_sha256"]),
        "actuator_schema_hash": str(tube.mapping_binding["actuator_schema_hash"]),
        "muscle_channel_core_fingerprint": str(muscle_channel_core_fingerprint),
        "signal": EMG_ANCHOR_SIGNAL,
        "method": EMG_ANCHOR_LOSS_METHOD,
        "channel_names": list(channel_names),
        "action_ids": list(tube.action_ids),
        "phase_bin_count": int(tube.phase_bin_count),
        "synergy_count": int(tube.synergy_count),
        "tube_kappa": kappa,
        "huber_delta": delta,
        "eps": epsilon,
        "synergy_shape_weight": shape_weight,
        "synergy_intensity_weight": intensity_weight,
        "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
        "projection_ridge": ridge if ridge > 0.0 else EMG_SYNERGY_RIDGE,
        "normalization_schema_version": str(tube.normalization_binding["schema_version"]),
        "audit_normalization": str(tube.normalization_binding["audit_normalization"]),
        "model_normalization": str(tube.normalization_binding["model_normalization"]),
    }
    payload["loss_spec_fingerprint"] = emg_anchor_loss_spec_fingerprint(payload)
    identity = EmgAnchorSpecIdentity(
        schema_version=payload["schema_version"],
        reference_id=payload["reference_id"],
        reference_fingerprint=payload["reference_fingerprint"],
        mapping_id=payload["mapping_id"],
        mapping_sha256=payload["mapping_sha256"],
        actuator_schema_hash=payload["actuator_schema_hash"],
        muscle_channel_core_fingerprint=payload["muscle_channel_core_fingerprint"],
        signal=payload["signal"],
        method=payload["method"],
        channel_names=tuple(channel_names),
        action_ids=tube.action_ids,
        phase_bin_count=payload["phase_bin_count"],
        synergy_count=payload["synergy_count"],
        tube_kappa=kappa,
        huber_delta=delta,
        eps=epsilon,
        synergy_shape_weight=shape_weight,
        synergy_intensity_weight=intensity_weight,
        projection_method=payload["projection_method"],
        projection_ridge=payload["projection_ridge"],
        normalization_schema_version=payload["normalization_schema_version"],
        audit_normalization=payload["audit_normalization"],
        model_normalization=payload["model_normalization"],
        loss_spec_fingerprint=payload["loss_spec_fingerprint"],
    )
    if identity.to_manifest() != payload:
        raise RuntimeError("compiled EMG anchor identity diverges from its fingerprinted payload")
    return spec, identity


def assert_emg_anchor_spec_matches(
    expected: EmgAnchorSpecIdentity,
    actual: EmgAnchorSpecIdentity,
) -> None:
    """Fail closed when a runtime anchor spec differs from a pinned release."""

    if expected.loss_spec_fingerprint != actual.loss_spec_fingerprint:
        raise ValueError(
            "EMG anchor loss spec fingerprint mismatch: expected "
            f"{expected.loss_spec_fingerprint!r}, runtime compiled "
            f"{actual.loss_spec_fingerprint!r}"
        )


def _positive_float(value: Any, *, field: str) -> float:
    result = _nonnegative_float(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def tube_distance(
    value: Any,
    mean: Any,
    scale: Any,
    valid: Any,
    *,
    kappa: float = DEFAULT_TUBE_KAPPA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    eps: float = EMG_ANCHOR_EPS,
) -> jax.Array:
    """Per-element Huber distance outside a ``kappa``-scale uncertainty tube.

    A value within ``kappa * scale`` of the human centre costs exactly zero, so
    the policy is free anywhere inside natural trial-to-trial variability.  Only
    the excess is charged, in units of the local scale, and the Huber transform
    keeps a single bad channel from dominating the term.  Invalid bins
    contribute nothing.
    """

    values = jnp.asarray(value)
    centre = jnp.asarray(mean, dtype=values.dtype)
    width = jnp.maximum(jnp.asarray(scale, dtype=values.dtype), jnp.asarray(eps, values.dtype))
    mask = jnp.asarray(valid, dtype=values.dtype)

    excess = jnp.abs(values - centre) - jnp.asarray(kappa, values.dtype) * width
    outside = jnp.maximum(excess, 0.0) / width
    delta = jnp.asarray(huber_delta, values.dtype)
    quadratic = 0.5 * jnp.square(outside)
    linear = delta * (outside - 0.5 * delta)
    return mask * jnp.where(outside <= delta, quadratic, linear)


def project_ordered_activation(
    ordered_activation: Any,
    spec: EmgAnchorSpec,
) -> jax.Array:
    """Apply ``y = P a`` to an ordered 354-wide activation vector."""

    activation = jnp.asarray(ordered_activation)
    projection = jnp.asarray(spec.projection, dtype=activation.dtype)
    return activation @ projection.T


def phase_bin_index(
    phase: Any,
    spec: EmgAnchorSpec,
    *,
    bin_offset: Any = 0,
) -> jax.Array:
    """Map normalized trajectory progress onto a static circular bin index.

    ``bin_offset`` exists only for the deterministic T4 negative control.  A
    non-zero value circularly shifts the *reference* lookup while leaving the
    real trajectory progress and activation-anchor lookup untouched.
    """

    fraction = jnp.clip(jnp.asarray(phase, dtype=jnp.float32), 0.0, 1.0)
    bins = int(spec.phase_bin_count)
    scaled = jnp.floor(fraction * bins).astype(jnp.int32)
    clipped = jnp.clip(scaled, 0, bins - 1)
    return jnp.mod(clipped + jnp.asarray(bin_offset, dtype=jnp.int32), bins)


def emg_anchor_metrics(
    ordered_activation: Any,
    spec: EmgAnchorSpec,
    *,
    action_index: Any,
    phase: Any,
    kappa: float = DEFAULT_TUBE_KAPPA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    eps: float = EMG_ANCHOR_EPS,
) -> EmgAnchorMetrics:
    """Activation-anchor term over the measured subspace only.

    ``ordered_activation`` is the 354-wide activation vector read via
    ``actuator_actadr``; the term never touches an unmeasured actuator.  All
    diagnostics are computed unconditionally so that turning a reward
    coefficient off does not also turn observability off.
    """

    projected = project_ordered_activation(ordered_activation, spec)
    action = jnp.asarray(action_index, dtype=jnp.int32)
    bin_index = phase_bin_index(phase, spec)

    centre = spec.anchor_mean[action, bin_index]
    width = spec.anchor_scale[action, bin_index]
    mask = spec.anchor_valid[action, bin_index]
    amplitude_confidence = spec.amplitude_confidence[action]
    weighted_mask = mask * amplitude_confidence

    channel_loss = tube_distance(
        projected,
        centre,
        width,
        mask,
        kappa=kappa,
        huber_delta=huber_delta,
        eps=eps,
    )
    valid_count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    weighted_count = jnp.maximum(jnp.sum(weighted_mask, axis=-1), eps)
    deviation = jnp.abs(projected - centre) * mask
    projected_mean = jnp.sum(projected * weighted_mask, axis=-1) / weighted_count
    centre_mean = jnp.sum(centre * weighted_mask, axis=-1) / weighted_count
    correlation_weight = jnp.sqrt(weighted_mask)
    projected_centered = (projected - projected_mean[..., None]) * correlation_weight
    centre_centered = (centre - centre_mean[..., None]) * correlation_weight
    correlation_numerator = jnp.sum(
        projected_centered * centre_centered,
        axis=-1,
    )
    correlation_denominator = jnp.sqrt(
        jnp.sum(jnp.square(projected_centered), axis=-1) * jnp.sum(jnp.square(centre_centered), axis=-1)
    )
    pattern_correlation = jnp.where(
        (valid_count >= 2.0) & (correlation_denominator > eps),
        correlation_numerator / (correlation_denominator + eps),
        0.0,
    )
    return EmgAnchorMetrics(
        loss=jnp.sum(channel_loss * amplitude_confidence, axis=-1) / weighted_count,
        violation_fraction=jnp.sum(weighted_mask * (channel_loss > 0.0), axis=-1) / weighted_count,
        mean_abs_deviation=jnp.sum(jnp.abs(projected - centre) * weighted_mask, axis=-1) / weighted_count,
        max_abs_deviation=jnp.max(deviation, axis=-1),
        pattern_correlation=pattern_correlation,
        valid_channel_fraction=jnp.mean(mask, axis=-1),
        channel_loss=channel_loss,
        projected_activation=projected,
    )


def emg_synergy_metrics(
    ordered_activation: Any,
    spec: EmgAnchorSpec,
    *,
    action_index: Any,
    phase: Any,
    phase_bin_offset: Any = 0,
    shape_weight: float = 1.0,
    intensity_weight: float = 0.25,
    kappa: float = DEFAULT_TUBE_KAPPA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    eps: float = EMG_ANCHOR_EPS,
) -> EmgSynergyMetrics:
    """Coordination consistency, split into shape and intensity.

    The simulated activation is projected onto the fixed human basis via
    ``h = relu(Q y)``.  Shape is compared as a cosine over the intensity
    normalised coefficient vector, and total intensity is charged separately
    through the same tube distance, because cross-system amplitude calibration
    is much weaker than coordination structure.
    """

    projected = project_ordered_activation(ordered_activation, spec)
    action = jnp.asarray(action_index, dtype=jnp.int32)
    bin_index = phase_bin_index(phase, spec, bin_offset=phase_bin_offset)

    basis_projection = jnp.asarray(spec.synergy_projection, dtype=projected.dtype)
    coefficients = jax.nn.relu(projected @ basis_projection.T)

    reference = spec.synergy_mean[action, bin_index]
    width = spec.synergy_scale[action, bin_index]
    mask = spec.synergy_valid[action, bin_index]
    epsilon = jnp.asarray(eps, dtype=projected.dtype)

    intensity = jnp.sum(coefficients * mask, axis=-1)
    reference_intensity = jnp.sum(reference * mask, axis=-1)
    shape = (coefficients * mask) / (intensity[..., None] + epsilon)
    reference_shape = (reference * mask) / (reference_intensity[..., None] + epsilon)

    numerator = jnp.sum(shape * reference_shape, axis=-1)
    denominator = jnp.linalg.norm(shape, axis=-1) * jnp.linalg.norm(reference_shape, axis=-1)
    cosine = numerator / (denominator + epsilon)
    any_valid = (jnp.sum(mask, axis=-1) > 0.0).astype(projected.dtype)
    shape_loss = any_valid * (1.0 - cosine)

    intensity_width = jnp.sqrt(jnp.sum(jnp.square(width * mask), axis=-1)) + epsilon
    intensity_loss = tube_distance(
        intensity,
        reference_intensity,
        intensity_width,
        any_valid,
        kappa=kappa,
        huber_delta=huber_delta,
        eps=eps,
    )

    weighted = jnp.asarray(shape_weight, projected.dtype) * shape_loss + (
        jnp.asarray(intensity_weight, projected.dtype) * intensity_loss
    )
    return EmgSynergyMetrics(
        loss=weighted,
        shape_loss=shape_loss,
        intensity_loss=intensity_loss,
        shape_cosine=cosine,
        intensity=intensity,
        reference_intensity=reference_intensity,
        coefficients=coefficients,
    )


def ordered_activation_from_data(data: Any, spec: EmgAnchorSpec, *, backend=jnp) -> Any:
    """Read the ordered activation vector through validated activation addresses."""

    addresses = backend.asarray(spec.activation_addresses)
    return backend.take(data.act, addresses, axis=-1)


def load_json_mapping(path: Any) -> dict[str, Any]:
    """Read an EMG observation mapping, rejecting duplicate JSON keys."""

    from pathlib import Path

    def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in EMG mapping")
            result[key] = value
        return result

    text = Path(path).expanduser().read_text(encoding="utf-8")
    return json.loads(text, object_pairs_hook=_pairs)
