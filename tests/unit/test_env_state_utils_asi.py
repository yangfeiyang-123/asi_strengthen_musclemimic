"""Regression tests for ASI carry updates in batched env states."""

import jax.numpy as jnp
from flax import struct

from musclemimic.algorithms.common.env_state_utils import (
    update_carry_asi_normalized,
    update_carry_asi_unnormalized,
)
from musclemimic.core.mujoco_mjx import MjxState
from musclemimic.core.wrappers.mjx import LogEnvState, Metrics, NormalizeVecRewEnvState


@struct.dataclass
class _DummyCarry:
    asi_frame_probs: jnp.ndarray | None
    asi_min_remaining_steps: jnp.ndarray


def _make_mjx_state(num_envs=4):
    carry = _DummyCarry(
        asi_frame_probs=None,
        asi_min_remaining_steps=jnp.ones((num_envs,), dtype=jnp.int32),
    )
    return MjxState(
        data=None,
        observation=jnp.zeros((num_envs, 3), dtype=jnp.float32),
        reward=jnp.zeros((num_envs,), dtype=jnp.float32),
        absorbing=jnp.zeros((num_envs,), dtype=bool),
        done=jnp.zeros((num_envs,), dtype=bool),
        additional_carry=carry,
        info={},
    )


def _make_log_state(num_envs=4):
    metrics = Metrics(
        episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
        episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        returned_episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        timestep=jnp.zeros((num_envs,), dtype=jnp.int32),
        done=jnp.zeros((num_envs,), dtype=bool),
        absorbing=jnp.zeros((num_envs,), dtype=bool),
    )
    return LogEnvState(_make_mjx_state(num_envs), metrics=metrics)


def test_update_carry_asi_unnormalized_preserves_batched_min_remaining_shape():
    num_envs = 4
    frame_probs = jnp.ones((num_envs, 1, 8), dtype=jnp.float32) / 8.0
    env_state = _make_log_state(num_envs)

    updated = update_carry_asi_unnormalized(env_state, frame_probs, 3)

    carry = updated.env_state.additional_carry
    assert carry.asi_frame_probs.shape == (num_envs, 1, 8)
    assert carry.asi_min_remaining_steps.shape == (num_envs,)
    assert jnp.all(carry.asi_min_remaining_steps == 3)


def test_update_carry_asi_normalized_preserves_batched_min_remaining_shape():
    num_envs = 4
    frame_probs = jnp.ones((num_envs, 1, 8), dtype=jnp.float32) / 8.0
    log_state = _make_log_state(num_envs)
    env_state = NormalizeVecRewEnvState(
        env_state=log_state,
        mean=jnp.asarray(0.0, dtype=jnp.float32),
        var=jnp.asarray(1.0, dtype=jnp.float32),
        count=jnp.asarray(1.0, dtype=jnp.float32),
        return_val=jnp.zeros((num_envs,), dtype=jnp.float32),
    )

    updated = update_carry_asi_normalized(env_state, frame_probs, 3)

    carry = updated.env_state.env_state.additional_carry
    assert carry.asi_frame_probs.shape == (num_envs, 1, 8)
    assert carry.asi_min_remaining_steps.shape == (num_envs,)
    assert jnp.all(carry.asi_min_remaining_steps == 3)
