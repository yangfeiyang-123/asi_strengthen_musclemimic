"""End-to-end numerical smoke tests for continuity-aware body actions.

These tests deliberately use deterministic synthetic decoder arrays.  They do
not claim a fitted physiological basis; their purpose is to exercise the exact
354-channel action ABI, MuJoCo muscle dynamics, and the pure-JAX continuity
kernel together before any evidence-gated production artifact is promoted.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    build_fascicle_continuity_spec,
    load_fascicle_continuity_graph,
)
from musclemimic.physiology.intra_muscle import (
    ordered_body_activation,
    robust_fascicle_continuity,
)
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoderJaxParams,
    decode_frozen_body_action,
)

jax.config.update("jax_platform_name", "cpu")

ROOT = Path(__file__).resolve().parents[2]
BODY_DIM = 354
SYNERGY_DIM = 8
RESIDUAL_DIM = 4


@pytest.fixture(scope="module")
def taxonomy():
    return load_anatomical_taxonomy(ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json")


@pytest.fixture(scope="module")
def continuity_spec(taxonomy):
    graph = load_fascicle_continuity_graph(
        ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json",
        taxonomy=taxonomy,
    )
    spec = build_fascicle_continuity_spec(graph, taxonomy)
    assert len(spec.chain_ids) == 28
    assert int(np.sum(np.asarray(spec.edge_mask))) == 140
    return spec


@pytest.fixture(scope="module", params=[MyoFullBody, MyoFullBodyRacket], ids=["bare", "racket"])
def body_env(request):
    return request.param(disable_fingers=True)


def _synthetic_decoder(*, residual: bool) -> FrozenBodyDecoderJaxParams:
    """Build a bounded test-only decoder with safe, non-zero excitation."""

    muscle_phase = np.linspace(0.0, 4.0 * np.pi, BODY_DIM, dtype=np.float32)[:, None]
    component_phase = np.linspace(0.0, np.pi, SYNERGY_DIM, dtype=np.float32)[None, :]
    basis = 0.0125 + 0.0075 * (1.0 + np.sin(muscle_phase + component_phase)) / 2.0
    if residual:
        residual_phase = np.linspace(0.0, 2.0 * np.pi, RESIDUAL_DIM, dtype=np.float32)[None, :]
        residual_basis = 0.01 * np.cos(muscle_phase + residual_phase)
        residual_alpha = np.asarray(0.05, dtype=np.float32)
    else:
        residual_basis = np.zeros((BODY_DIM, 0), dtype=np.float32)
        residual_alpha = np.asarray(0.0, dtype=np.float32)
    return FrozenBodyDecoderJaxParams(
        basis=jnp.asarray(basis, dtype=jnp.float32),
        excitation_bounds=jnp.asarray([[0.0, 1.0]] * BODY_DIM, dtype=jnp.float32),
        coefficient_maximum=jnp.full((SYNERGY_DIM,), 0.5, dtype=jnp.float32),
        coefficient_bias=jnp.zeros((SYNERGY_DIM,), dtype=jnp.float32),
        coefficient_temperature=jnp.ones((SYNERGY_DIM,), dtype=jnp.float32),
        tonic_baseline=jnp.full((BODY_DIM,), 0.05, dtype=jnp.float32),
        residual_basis=jnp.asarray(residual_basis, dtype=jnp.float32),
        residual_alpha=jnp.asarray(residual_alpha, dtype=jnp.float32),
    )


def _body_action(mode: str) -> np.ndarray:
    if mode == "direct":
        phase = np.linspace(0.0, 2.0 * np.pi, BODY_DIM, dtype=np.float32)
        physical_excitation = 0.15 + 0.05 * np.sin(phase)
        return 2.0 * physical_excitation - 1.0
    residual = mode == "fixed_residual"
    decoder = _synthetic_decoder(residual=residual)
    raw_dim = SYNERGY_DIM + (RESIDUAL_DIM if residual else 0)
    raw_action = jnp.linspace(-0.5, 0.5, raw_dim, dtype=jnp.float32)
    return np.asarray(decode_frozen_body_action(raw_action, decoder).body_action)


@pytest.mark.parametrize("mode", ["direct", "fixed", "fixed_residual"])
def test_bare_and_racket_steps_are_finite_for_all_action_modes(body_env, continuity_spec, mode):
    observation = body_env.reset()
    assert np.all(np.isfinite(np.asarray(observation)))
    action = _body_action(mode)
    assert action.shape == (BODY_DIM,)
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= 1.0

    last_metrics = None
    for _ in range(4):
        observation, reward, absorbing, done, _info = body_env.step(action)
        activation = ordered_body_activation(body_env._data, continuity_spec, backend=np)
        last_metrics = robust_fascicle_continuity(jnp.asarray(activation), continuity_spec)
        assert np.all(np.isfinite(np.asarray(observation)))
        assert np.isfinite(float(reward))
        assert isinstance(bool(absorbing), bool)
        assert isinstance(bool(done), bool)
        assert activation.shape == (BODY_DIM,)
        assert np.all(np.isfinite(activation))
        assert all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(last_metrics))

    assert last_metrics is not None
    assert float(last_metrics.active_chain_fraction) > 0.0


def test_decoder_and_continuity_share_one_stable_callback_free_jit(continuity_spec):
    decoder = _synthetic_decoder(residual=True)
    raw_action = jnp.linspace(-0.75, 0.75, SYNERGY_DIM + RESIDUAL_DIM, dtype=jnp.float32)

    def decode_and_measure(raw):
        decoded = decode_frozen_body_action(raw, decoder)
        metrics = robust_fascicle_continuity(decoded.physical_excitation, continuity_spec)
        summary = jnp.stack(
            [
                metrics.loss,
                metrics.active_chain_fraction,
                metrics.violation_fraction,
                metrics.mean_abs_edge_difference,
                metrics.max_abs_edge_difference,
            ]
        )
        return decoded.body_action, summary

    eager_action, eager_metrics = decode_and_measure(raw_action)
    compiled = jax.jit(decode_and_measure)
    cache_before = compiled._cache_size()
    jit_action, jit_metrics = compiled(raw_action)
    jax.block_until_ready((jit_action, jit_metrics))
    cache_after_first_call = compiled._cache_size()
    second_action, second_metrics = compiled(raw_action)
    jax.block_until_ready((second_action, second_metrics))

    np.testing.assert_allclose(jit_action, eager_action, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(jit_metrics, eager_metrics, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(second_action, jit_action)
    np.testing.assert_array_equal(second_metrics, jit_metrics)
    assert cache_after_first_call >= cache_before
    assert compiled._cache_size() == cache_after_first_call

    jaxpr_text = str(jax.make_jaxpr(decode_and_measure)(raw_action)).lower()
    forbidden_callbacks = ("io_callback", "pure_callback", "debug_callback", "host_callback")
    assert not any(token in jaxpr_text for token in forbidden_callbacks)
