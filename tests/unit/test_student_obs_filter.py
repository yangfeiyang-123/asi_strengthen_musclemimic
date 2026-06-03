"""Tests for student observation filtering used by policy distillation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.core.wrappers.mjx import NStepWrapper
from musclemimic.distill.obs_filter import (
    StudentObservationFilterWrapper,
    build_student_obs_indices,
    filter_student_obs,
)


class MockObsContainer:
    def __init__(self, groups: dict[str, np.ndarray]):
        self._groups = {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}

    def get_obs_ind_by_group(self, group_name: str) -> np.ndarray:
        return self._groups.get(group_name, np.array([], dtype=int))

    def get_all_group_names(self) -> list[str]:
        return list(self._groups)


class MockMDPInfo:
    def __init__(self, obs_dim: int, action_dim: int = 2):
        self.observation_space = Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
        )
        self.action_space = Box(
            low=np.full(action_dim, -1.0, dtype=np.float32),
            high=np.full(action_dim, 1.0, dtype=np.float32),
        )


@struct.dataclass
class MockEnvState:
    step_count: int = 0


class MockEnv:
    def __init__(self, state_dim: int = 6, goal_dim: int = 5, include_goal_group: bool = True):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obs_dim = state_dim + goal_dim
        groups = {"state": np.arange(state_dim, dtype=int)}
        if include_goal_group:
            groups["goal"] = np.arange(state_dim, self.obs_dim, dtype=int)
        self.obs_container = MockObsContainer(groups)
        self.info = MockMDPInfo(self.obs_dim)
        self.mdp_info = self.info
        self.mjx_env = False

    def reset(self, rng_key):
        del rng_key
        state_obs = jnp.arange(1, self.state_dim + 1, dtype=jnp.float32)
        goal_obs = jnp.arange(100, 100 + self.goal_dim, dtype=jnp.float32)
        return jnp.concatenate([state_obs, goal_obs]), MockEnvState(step_count=0)

    def step(self, state: MockEnvState, action):
        del action
        step = state.step_count + 1
        state_obs = jnp.arange(1, self.state_dim + 1, dtype=jnp.float32) + step
        goal_obs = jnp.arange(100, 100 + self.goal_dim, dtype=jnp.float32) + step * 10
        obs = jnp.concatenate([state_obs, goal_obs])
        return obs, 0.0, False, False, {}, MockEnvState(step_count=step)


def test_build_student_obs_indices_keeps_state_and_goal_last_phase():
    env = MockEnv(state_dim=6, goal_dim=5)
    spec = build_student_obs_indices(env, OmegaConf.create({"keep_motion_phase": True}))

    np.testing.assert_array_equal(spec.state_indices, np.arange(6))
    np.testing.assert_array_equal(spec.goal_indices, np.arange(6, 11))
    np.testing.assert_array_equal(spec.student_indices, np.array([0, 1, 2, 3, 4, 5, 10]))
    assert spec.phase_index == 10
    assert spec.student_obs_dim == 7


def test_drop_goal_lookahead_false_rejects_phase_only_student():
    env = MockEnv(state_dim=6, goal_dim=5)

    with pytest.raises(ValueError, match="drop_goal_lookahead"):
        build_student_obs_indices(
            env,
            OmegaConf.create(
                {
                    "drop_goal_lookahead": False,
                    "keep_motion_phase": True,
                }
            ),
        )


def test_filter_student_obs_supports_batched_and_unbatched_arrays():
    env = MockEnv(state_dim=3, goal_dim=4)
    spec = build_student_obs_indices(env, {})
    obs = jnp.array([1, 2, 3, 100, 101, 102, 0.75], dtype=jnp.float32)

    filtered = filter_student_obs(obs, spec)
    assert filtered.shape == (4,)
    assert jnp.allclose(filtered, jnp.array([1, 2, 3, 0.75], dtype=jnp.float32))

    batched = jnp.stack([obs, obs + 1.0])
    filtered_batched = filter_student_obs(batched, spec)
    assert filtered_batched.shape == (2, 4)
    assert jnp.allclose(filtered_batched[:, -1], jnp.array([0.75, 1.75], dtype=jnp.float32))


def test_student_observation_filter_wrapper_updates_obs_and_groups():
    env = MockEnv(state_dim=6, goal_dim=5)
    wrapped = StudentObservationFilterWrapper(env, {"keep_motion_phase": True})

    obs, state = wrapped.reset(jax.random.PRNGKey(0))

    assert obs.shape == (7,)
    assert wrapped.info.observation_space.shape == (7,)
    assert wrapped.mdp_info.observation_space.shape == (7,)
    assert jnp.allclose(obs, jnp.array([1, 2, 3, 4, 5, 6, 104], dtype=jnp.float32))
    np.testing.assert_array_equal(wrapped.obs_container.get_obs_ind_by_group("state"), np.arange(6))
    np.testing.assert_array_equal(wrapped.obs_container.get_obs_ind_by_group("goal"), np.array([6]))
    assert "goal" in wrapped.obs_container
    np.testing.assert_array_equal(wrapped.obs_container["phase"], np.array([6]))
    np.testing.assert_array_equal(wrapped.obs_container.get("missing", np.array([-1])), np.array([-1]))
    assert sorted(name for name, _indices in wrapped.obs_container.items()) == ["goal", "phase", "state"]

    next_obs, *_ = wrapped.step(state, jnp.zeros(2))
    assert jnp.allclose(next_obs, jnp.array([2, 3, 4, 5, 6, 7, 114], dtype=jnp.float32))


def test_student_filter_composes_with_nstep_split_goal():
    env = MockEnv(state_dim=6, goal_dim=5)
    filtered = StudentObservationFilterWrapper(env, {"keep_motion_phase": True})
    wrapped = NStepWrapper(filtered, n_steps=3, split_goal=True)

    obs, _state = wrapped.reset(jax.random.PRNGKey(0))

    assert obs.shape == (3 * 6 + 1,)
    assert jnp.allclose(obs[-1], 104.0)


def test_wrap_env_applies_student_filter_before_nstep_wrapper():
    env = MockEnv(state_dim=6, goal_dim=5)
    config = OmegaConf.create(
        {
            "student_obs_filter": {"enabled": True, "keep_motion_phase": True},
            "len_obs_history": 3,
            "split_goal": True,
            "normalize_env": False,
            "gamma": 0.99,
        }
    )

    wrapped = wrap_env(env, config)
    assert wrapped.info.observation_space.shape == (3 * 6 + 1,)


def test_student_filter_requires_goal_group_by_default():
    env = MockEnv(state_dim=6, goal_dim=0, include_goal_group=False)

    with pytest.raises(ValueError, match="goal"):
        build_student_obs_indices(env, {"require_goal_group": True})
