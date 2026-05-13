"""Unit tests for adaptive start-state initialization helpers."""

import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.asi import (
    compute_frame_asi_probs,
    create_frame_asi_state,
    make_bucket_start_steps,
    update_frame_asi_state,
)


def test_compute_frame_asi_probs_respects_valid_mask_and_uniform_floor():
    logits = jnp.array([[0.0, 8.0, 1.0]], dtype=jnp.float32)
    valid_mask = jnp.array([[True, False, True]])

    probs = compute_frame_asi_probs(
        logits,
        valid_mask=valid_mask,
        uniform_mix=0.2,
        temperature=1.0,
    )

    np.testing.assert_allclose(float(jnp.sum(probs)), 1.0, rtol=1e-6)
    assert probs[0, 1] == 0.0
    assert probs[0, 0] >= 0.1 - 1e-6
    assert probs[0, 2] >= 0.1 - 1e-6


def test_make_bucket_start_steps_respects_min_remaining_steps():
    split_points = jnp.array([0, 10, 13], dtype=jnp.int32)

    start_steps, valid_mask = make_bucket_start_steps(
        split_points,
        num_buckets=4,
        min_remaining_steps=2,
    )

    np.testing.assert_array_equal(start_steps[0], jnp.array([0, 2, 5, 8], dtype=jnp.int32))
    np.testing.assert_array_equal(start_steps[1], jnp.array([0, 0, 0, 1], dtype=jnp.int32))
    np.testing.assert_array_equal(valid_mask, jnp.ones((2, 4), dtype=bool))


def test_update_frame_asi_state_increases_probability_for_high_score_bucket():
    state = create_frame_asi_state(n_trajectories=1, num_buckets=3)
    probs = compute_frame_asi_probs(state.logits, uniform_mix=0.0)

    new_state = update_frame_asi_state(
        state,
        probs,
        init_traj_ids=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        init_bucket_ids=jnp.array([2, 2, 0, 1], dtype=jnp.int32),
        scores=jnp.array([2.0, 2.0, -1.0, -1.0], dtype=jnp.float32),
        done_mask=jnp.array([True, True, True, True]),
        alpha=0.5,
        baseline_beta=1.0,
        logit_clip=5.0,
    )

    new_probs = compute_frame_asi_probs(new_state.logits, uniform_mix=0.0)

    assert new_probs[0, 2] > probs[0, 2]
    assert new_probs[0, 2] > new_probs[0, 0]
    assert new_probs[0, 2] > new_probs[0, 1]
