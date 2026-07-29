"""JAX-compatible intra-muscle consistency diagnostics.

No function in this module adds a reward.  It implements two explicitly
different numerical objects:

* :func:`exact_exo_imr` reproduces Exo-Plore's hard-deadband sum exactly;
* :func:`robust_intra_muscle_consistency` implements the project proposal's
  weighted, group-normalized Huber diagnostic with a stop-gradient activity
  gate.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class IntraMuscleSpec:
    """Static padded arrays consumed inside JIT-compiled diagnostics."""

    group_indices: jax.Array
    member_mask: jax.Array
    member_weights: jax.Array
    group_weights: jax.Array
    deadband: jax.Array
    activity_off: jax.Array
    activity_on: jax.Array
    activation_addresses: jax.Array
    body_actuator_ids: jax.Array
    group_ids: tuple[str, ...] = struct.field(pytree_node=False, default=())
    relationship: str = struct.field(
        pytree_node=False,
        default="hard_line_group",
    )


class ExoImrMetrics(NamedTuple):
    """Exact Exo-Plore IMR value plus non-invasive diagnostics."""

    loss: jax.Array
    violation_fraction: jax.Array
    mean_abs_deviation: jax.Array
    max_abs_deviation: jax.Array
    group_mean: jax.Array
    group_loss: jax.Array


class IntraMuscleMetrics(NamedTuple):
    """Robust project IMR aggregate and per-group diagnostics."""

    loss: jax.Array
    active_group_fraction: jax.Array
    violation_fraction: jax.Array
    mean_abs_deviation: jax.Array
    max_abs_deviation: jax.Array
    group_mean: jax.Array
    group_activity_gate: jax.Array
    group_loss: jax.Array
    group_violation_fraction: jax.Array


def exact_exo_imr(
    ordered_signal: Any,
    spec: IntraMuscleSpec,
    *,
    deadband: float = 0.1,
) -> ExoImrMetrics:
    """Compute Exo-Plore's exact hard-threshold, unnormalized IMR sum.

    For each group, the function uses an unweighted member mean and sums
    ``1[abs(a_i-mean)>deadband] * (a_i-mean)^2`` over all valid members.
    There is deliberately no group-size normalization, Huber transform, or
    activity gate in this exact reference implementation.
    """

    signal = jnp.asarray(ordered_signal)
    _validate_runtime_shapes(signal, spec)
    if spec.group_indices.shape[0] == 0:
        return _empty_exo_metrics(signal.dtype)

    values = jnp.take(signal, spec.group_indices, axis=-1)
    mask = jnp.asarray(spec.member_mask, dtype=signal.dtype)
    member_count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    group_mean = jnp.sum(mask * values, axis=-1) / member_count
    deviation = values - group_mean[:, None]
    abs_deviation = jnp.abs(deviation)
    violations = mask * (abs_deviation > jnp.asarray(deadband, signal.dtype))
    element_loss = violations * jnp.square(deviation)
    group_loss = jnp.sum(element_loss, axis=-1)
    valid_count = jnp.maximum(jnp.sum(mask), 1.0)
    return ExoImrMetrics(
        loss=jnp.sum(group_loss),
        violation_fraction=jnp.sum(violations) / valid_count,
        mean_abs_deviation=jnp.sum(mask * abs_deviation) / valid_count,
        max_abs_deviation=jnp.max(jnp.where(mask > 0.0, abs_deviation, 0.0)),
        group_mean=group_mean,
        group_loss=group_loss,
    )


def robust_intra_muscle_consistency(
    ordered_signal: Any,
    spec: IntraMuscleSpec,
    *,
    scale: float = 0.05,
    huber_delta: float = 1.0,
    eps: float = 1e-8,
) -> IntraMuscleMetrics:
    """Compute robust deadbanded, gated, group-normalized diagnostics."""

    signal = jnp.asarray(ordered_signal)
    _validate_runtime_shapes(signal, spec)
    if spec.group_indices.shape[0] == 0:
        return _empty_robust_metrics(signal.dtype)
    if scale <= 0.0 or huber_delta <= 0.0 or eps <= 0.0:
        raise ValueError("scale, huber_delta and eps must be positive")

    values = jnp.take(signal, spec.group_indices, axis=-1)
    mask = jnp.asarray(spec.member_mask, dtype=signal.dtype)
    weights = jnp.asarray(spec.member_weights, dtype=signal.dtype) * mask
    weight_sum = jnp.maximum(jnp.sum(weights, axis=-1), eps)
    group_mean = jnp.sum(weights * values, axis=-1) / weight_sum
    abs_deviation = jnp.abs(values - group_mean[:, None])
    excess = jax.nn.relu(abs_deviation - jnp.asarray(spec.deadband, dtype=signal.dtype)[:, None])
    element_loss = _huber(excess / scale, delta=huber_delta)
    group_loss = jnp.sum(weights * element_loss, axis=-1) / weight_sum

    activity_off = jnp.asarray(spec.activity_off, dtype=signal.dtype)
    activity_on = jnp.asarray(spec.activity_on, dtype=signal.dtype)
    activity_gate = jnp.clip(
        (group_mean - activity_off) / jnp.maximum(activity_on - activity_off, eps),
        0.0,
        1.0,
    )
    activity_gate = jax.lax.stop_gradient(activity_gate)
    group_weights = jnp.asarray(spec.group_weights, dtype=signal.dtype)
    effective_group_weights = group_weights * activity_gate
    total_weight = jnp.maximum(jnp.sum(effective_group_weights), eps)
    loss = jnp.sum(effective_group_weights * group_loss) / total_weight

    violations = mask * (excess > 0.0)
    valid_count = jnp.maximum(jnp.sum(mask), 1.0)
    member_count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    return IntraMuscleMetrics(
        loss=loss,
        active_group_fraction=jnp.mean(activity_gate > 0.0),
        violation_fraction=jnp.sum(violations) / valid_count,
        mean_abs_deviation=jnp.sum(mask * abs_deviation) / valid_count,
        max_abs_deviation=jnp.max(jnp.where(mask > 0.0, abs_deviation, 0.0)),
        group_mean=group_mean,
        group_activity_gate=activity_gate,
        group_loss=group_loss,
        group_violation_fraction=jnp.sum(violations, axis=-1) / member_count,
    )


def ordered_body_activation(
    data: Any,
    spec: IntraMuscleSpec,
    *,
    backend: Any = jnp,
) -> Any:
    """Read ordered activation through explicit validated activation addresses."""

    return backend.take(
        data.act,
        backend.asarray(spec.activation_addresses),
        axis=-1,
    )


def _huber(value: jax.Array, *, delta: float) -> jax.Array:
    absolute = jnp.abs(value)
    quadratic = jnp.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * jnp.square(quadratic) + delta * linear


def _validate_runtime_shapes(signal: jax.Array, spec: IntraMuscleSpec) -> None:
    if signal.ndim != 1:
        raise ValueError("ordered_signal must be one-dimensional; use jax.vmap for batches")
    indices = jnp.asarray(spec.group_indices)
    mask = jnp.asarray(spec.member_mask)
    weights = jnp.asarray(spec.member_weights)
    if indices.ndim != 2 or mask.shape != indices.shape or weights.shape != indices.shape:
        raise ValueError("group_indices, member_mask and member_weights must share [G,L] shape")
    group_count = indices.shape[0]
    for field_name in (
        "group_weights",
        "deadband",
        "activity_off",
        "activity_on",
    ):
        if jnp.asarray(getattr(spec, field_name)).shape != (group_count,):
            raise ValueError(f"{field_name} must have shape ({group_count},)")
    if len(spec.group_ids) != group_count:
        raise ValueError("group_ids length must equal the static group count")
    if jnp.asarray(spec.activation_addresses).shape != jnp.asarray(spec.body_actuator_ids).shape:
        raise ValueError("activation_addresses and body_actuator_ids must have matching shape")
    if signal.shape[0] != jnp.asarray(spec.activation_addresses).shape[0]:
        raise ValueError("ordered_signal width must match the ordered actuator layout")


def _empty_exo_metrics(dtype: Any) -> ExoImrMetrics:
    zero = jnp.asarray(0.0, dtype=dtype)
    empty = jnp.zeros((0,), dtype=dtype)
    return ExoImrMetrics(
        loss=zero,
        violation_fraction=zero,
        mean_abs_deviation=zero,
        max_abs_deviation=zero,
        group_mean=empty,
        group_loss=empty,
    )


def _empty_robust_metrics(dtype: Any) -> IntraMuscleMetrics:
    zero = jnp.asarray(0.0, dtype=dtype)
    empty = jnp.zeros((0,), dtype=dtype)
    return IntraMuscleMetrics(
        loss=zero,
        active_group_fraction=zero,
        violation_fraction=zero,
        mean_abs_deviation=zero,
        max_abs_deviation=zero,
        group_mean=empty,
        group_activity_gate=empty,
        group_loss=empty,
        group_violation_fraction=empty,
    )
