"""Regression tests for ASI-aware trajectory reset sampling."""

import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.trajectory.handler import TrajectoryHandler


@struct.dataclass
class _DummyCarry:
    key: jax.Array
    selected_traj_idx: jax.Array
    sampling_weights: jax.Array | None = None
    asi_frame_probs: jax.Array | None = None
    asi_min_remaining_steps: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(1, dtype=jnp.int32)
    )
    traj_state: object | None = None


class _DummyTrajData:
    split_points = jnp.array([0, 10, 25], dtype=jnp.int32)


class _DummyTraj:
    data = _DummyTrajData()


def _make_handler():
    handler = object.__new__(TrajectoryHandler)
    handler.traj = _DummyTraj()
    handler.random_start = True
    handler.fixed_start_conf = None
    handler.use_fixed_start = False
    handler.start_from_random_step = True
    return handler


def test_disabled_asi_keeps_legacy_random_reset_sampling():
    handler = _make_handler()
    key = jax.random.PRNGKey(7)
    carry = _DummyCarry(key=key, selected_traj_idx=jnp.asarray(-1, dtype=jnp.int32))

    _, new_carry = handler.reset_state(None, None, None, carry, jnp)

    expected_new_key, k1, k2 = jax.random.split(key, 3)
    expected_traj = jax.random.randint(k1, shape=(), minval=0, maxval=handler.n_trajectories)
    expected_step = jax.random.randint(
        k2,
        shape=(),
        minval=0,
        maxval=handler.len_trajectory(expected_traj),
    )

    assert jnp.array_equal(new_carry.key, expected_new_key)
    assert new_carry.traj_state.traj_no == expected_traj
    assert new_carry.traj_state.subtraj_step_no_init == expected_step
    assert new_carry.traj_state.subtraj_bucket_no_init == -1


def test_enabled_asi_uses_bucket_probability_for_start_step():
    handler = _make_handler()
    key = jax.random.PRNGKey(11)
    carry = _DummyCarry(
        key=key,
        selected_traj_idx=jnp.asarray(-1, dtype=jnp.int32),
        asi_frame_probs=jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=jnp.float32),
        asi_min_remaining_steps=jnp.asarray(3, dtype=jnp.int32),
    )

    _, new_carry = handler.reset_state(None, None, None, carry, jnp)

    expected_new_key, k1, _, _ = jax.random.split(key, 4)
    expected_traj = jax.random.randint(k1, shape=(), minval=0, maxval=handler.n_trajectories)
    max_start = jnp.maximum(handler.len_trajectory(expected_traj) - 3, 0)
    expected_step = (max_start * 2) // 3

    assert jnp.array_equal(new_carry.key, expected_new_key)
    assert new_carry.traj_state.traj_no == expected_traj
    assert new_carry.traj_state.subtraj_bucket_no_init == 2
    assert new_carry.traj_state.subtraj_step_no_init == expected_step
