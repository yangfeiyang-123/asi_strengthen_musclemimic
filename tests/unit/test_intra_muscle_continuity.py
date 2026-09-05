"""Numerical tests for adjacency-based fascicle continuity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.physiology.intra_muscle import (
    FascicleContinuitySpec,
    robust_fascicle_continuity,
)

jax.config.update("jax_platform_name", "cpu")


def _spec(*, padded: bool = False, inactive: bool = False, empty: bool = False):
    if empty:
        return FascicleContinuitySpec(
            edge_indices=jnp.zeros((0, 1, 2), dtype=jnp.int32),
            edge_mask=jnp.zeros((0, 1), dtype=jnp.float32),
            edge_weights=jnp.zeros((0, 1), dtype=jnp.float32),
            member_indices=jnp.zeros((0, 1), dtype=jnp.int32),
            member_mask=jnp.zeros((0, 1), dtype=jnp.float32),
            member_weights=jnp.zeros((0, 1), dtype=jnp.float32),
            chain_weights=jnp.zeros((0,), dtype=jnp.float32),
            deadband=jnp.zeros((0,), dtype=jnp.float32),
            activity_off=jnp.zeros((0,), dtype=jnp.float32),
            activity_on=jnp.zeros((0,), dtype=jnp.float32),
            activation_addresses=jnp.arange(4, dtype=jnp.int32),
            body_actuator_ids=jnp.arange(4, dtype=jnp.int32),
            chain_ids=(),
        )
    edge_indices = [[[0, 1], [1, 2], [0, 0]]] if padded else [[[0, 1], [1, 2]]]
    edge_mask = [[1.0, 1.0, 0.0]] if padded else [[1.0, 1.0]]
    edge_weights = [[1.0, 1.0, 999.0]] if padded else [[1.0, 1.0]]
    member_indices = [[0, 1, 2, 0]] if padded else [[0, 1, 2]]
    member_mask = [[1.0, 1.0, 1.0, 0.0]] if padded else [[1.0, 1.0, 1.0]]
    member_weights = [[1.0, 1.0, 1.0, 999.0]] if padded else [[1.0, 1.0, 1.0]]
    return FascicleContinuitySpec(
        edge_indices=jnp.asarray(edge_indices, dtype=jnp.int32),
        edge_mask=jnp.asarray(edge_mask, dtype=jnp.float32),
        edge_weights=jnp.asarray(edge_weights, dtype=jnp.float32),
        member_indices=jnp.asarray(member_indices, dtype=jnp.int32),
        member_mask=jnp.asarray(member_mask, dtype=jnp.float32),
        member_weights=jnp.asarray(member_weights, dtype=jnp.float32),
        chain_weights=jnp.ones((1,), dtype=jnp.float32),
        deadband=jnp.asarray([0.1], dtype=jnp.float32),
        activity_off=jnp.asarray([0.8 if inactive else 0.0], dtype=jnp.float32),
        activity_on=jnp.asarray([0.9 if inactive else 0.1], dtype=jnp.float32),
        activation_addresses=jnp.arange(4, dtype=jnp.int32),
        body_actuator_ids=jnp.arange(4, dtype=jnp.int32),
        chain_ids=("chain",),
    )


def test_continuity_eager_jit_vmap_padding_and_direction_are_invariant():
    signal = jnp.asarray([0.2, 0.5, 0.8, 0.1], dtype=jnp.float32)
    base = robust_fascicle_continuity(signal, _spec())
    compiled = jax.jit(lambda value: robust_fascicle_continuity(value, _spec()))(signal)
    padded = robust_fascicle_continuity(signal, _spec(padded=True))
    reversed_spec = _spec().replace(edge_indices=jnp.asarray([[[1, 0], [2, 1]]]))
    reversed_metrics = robust_fascicle_continuity(signal, reversed_spec)
    assert float(compiled.loss) == pytest.approx(float(base.loss))
    assert float(padded.loss) == pytest.approx(float(base.loss))
    assert float(reversed_metrics.loss) == pytest.approx(float(base.loss))
    batch = jnp.stack([signal, signal * 0.5])
    vectorized = jax.vmap(lambda value: robust_fascicle_continuity(value, _spec()).loss)(batch)
    loop = jnp.stack([robust_fascicle_continuity(row, _spec()).loss for row in batch])
    np.testing.assert_allclose(vectorized, loop)


def test_continuity_deadband_monotonic_activity_gate_and_gradient():
    within = jnp.asarray([0.20, 0.25, 0.30, 0.0], dtype=jnp.float32)
    moderate = jnp.asarray([0.20, 0.45, 0.70, 0.0], dtype=jnp.float32)
    large = jnp.asarray([0.20, 0.70, 1.00, 0.0], dtype=jnp.float32)
    assert float(robust_fascicle_continuity(within, _spec()).loss) == 0.0
    moderate_loss = float(robust_fascicle_continuity(moderate, _spec()).loss)
    large_loss = float(robust_fascicle_continuity(large, _spec()).loss)
    assert 0.0 < moderate_loss < large_loss
    inactive = robust_fascicle_continuity(moderate, _spec(inactive=True))
    assert float(inactive.loss) == 0.0
    assert float(inactive.active_chain_fraction) == 0.0
    gate_gradient = jax.grad(lambda value: jnp.sum(robust_fascicle_continuity(value, _spec()).chain_activity_gate))(
        moderate
    )
    np.testing.assert_array_equal(gate_gradient, np.zeros(4))
    loss_gradient = jax.grad(lambda value: robust_fascicle_continuity(value, _spec()).loss)(moderate)
    assert np.all(np.isfinite(np.asarray(loss_gradient)))


def test_continuity_deadband_boundary_and_single_active_chain_are_exact():
    boundary = jnp.asarray([0.0, 0.1, 0.0, 0.0], dtype=jnp.float32)
    assert float(robust_fascicle_continuity(boundary, _spec()).loss) == 0.0

    base = _spec()
    two_chain = base.replace(
        edge_indices=jnp.asarray([[[0, 1]], [[2, 3]]], dtype=jnp.int32),
        edge_mask=jnp.ones((2, 1), dtype=jnp.float32),
        edge_weights=jnp.ones((2, 1), dtype=jnp.float32),
        member_indices=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        member_mask=jnp.ones((2, 2), dtype=jnp.float32),
        member_weights=jnp.ones((2, 2), dtype=jnp.float32),
        chain_weights=jnp.ones((2,), dtype=jnp.float32),
        deadband=jnp.asarray([0.1, 0.1], dtype=jnp.float32),
        activity_off=jnp.asarray([0.0, 0.8], dtype=jnp.float32),
        activity_on=jnp.asarray([0.1, 0.9], dtype=jnp.float32),
        chain_ids=("active", "inactive"),
    )
    signal = jnp.asarray([0.2, 0.8, 0.1, 0.2], dtype=jnp.float32)
    metrics = robust_fascicle_continuity(signal, two_chain)
    assert float(metrics.active_chain_fraction) == pytest.approx(0.5)
    assert float(metrics.chain_activity_gate[0]) == pytest.approx(1.0)
    assert float(metrics.chain_activity_gate[1]) == pytest.approx(0.0)
    assert float(metrics.loss) == pytest.approx(float(metrics.chain_loss[0]))


def test_continuity_empty_graph_and_shape_mismatch_fail_closed():
    signal = jnp.asarray([0.2, 0.4, 0.6, 0.8], dtype=jnp.float32)
    metrics = jax.jit(lambda value: robust_fascicle_continuity(value, _spec(empty=True)))(signal)
    assert float(metrics.loss) == 0.0
    assert float(metrics.active_chain_fraction) == 0.0
    assert metrics.chain_loss.shape == (0,)
    with pytest.raises(ValueError, match="ordered_signal width"):
        robust_fascicle_continuity(signal[:3], _spec())
    with pytest.raises(ValueError, match="one-dimensional"):
        robust_fascicle_continuity(signal[None, :], _spec())
