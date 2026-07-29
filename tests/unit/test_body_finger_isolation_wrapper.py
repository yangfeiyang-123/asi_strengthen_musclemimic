from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.core.wrappers.finger_isolation import BodyFingerIsolationWrapper
from musclemimic.utils.finger_isolation import (
    LEFT_FINGER_ACTUATOR_NAMES,
    RIGHT_FINGER_ACTUATOR_NAMES,
    NamedObservationSchema,
    ObservationField,
)


class _Entry:
    def __init__(self, name: str, indices, group):
        self.name = name
        self.obs_ind = np.asarray(indices, dtype=int)
        self.group = [group]
        self.allow_randomization = True
        self.dim = len(self.obs_ind)


class _MockFingerEnv:
    def __init__(self):
        self.mjx_env = False
        obs_space = Box(-np.ones(9), np.ones(9))
        action_space = Box(-np.ones(416), np.ones(416))
        self.info = SimpleNamespace(
            observation_space=obs_space,
            action_space=action_space,
            gamma=0.99,
            horizon=10,
            dt=0.01,
        )
        self.mdp_info = self.info
        self.obs_container = {
            "root": _Entry("root", [0, 1], "state"),
            "right_finger": _Entry("right_finger", [2], "state"),
            "left_finger": _Entry("left_finger", [3], "state"),
            "body_muscle": _Entry("body_muscle", [4], "state"),
            "right_finger_muscle": _Entry("right_finger_muscle", [5], "state"),
            "left_finger_muscle": _Entry("left_finger_muscle", [6], "state"),
            "goal": _Entry("goal", [7, 8], "goal"),
        }
        self.last_full_action = None

    def reset(self, _key):
        obs = jnp.arange(9, dtype=jnp.float32)
        return obs, SimpleNamespace(step=0)

    def reset_to(self, key, _traj_idx):
        return self.reset(key)

    def step(self, state, action):
        self.last_full_action = action
        obs = jnp.arange(9, dtype=jnp.float32) + 10
        return obs, 1.0, False, False, {}, state


def _action_names() -> tuple[str, ...]:
    body = [f"body_{index:03d}" for index in range(354)]
    return tuple(
        body[:90]
        + list(RIGHT_FINGER_ACTUATOR_NAMES[:12])
        + body[90:230]
        + list(LEFT_FINGER_ACTUATOR_NAMES[:18])
        + list(RIGHT_FINGER_ACTUATOR_NAMES[12:])
        + body[230:]
        + list(LEFT_FINGER_ACTUATOR_NAMES[18:])
    )


def _observation_filter():
    schema = NamedObservationSchema(
        (
            ObservationField("root", width=2),
            ObservationField("right_finger", joint_name="cmc_flexion_r"),
            ObservationField("left_finger", joint_name="cmc_flexion_l"),
            ObservationField("body_muscle", actuator_name="body_000"),
            ObservationField("right_finger_muscle", actuator_name="FDS2"),
            ObservationField("left_finger_muscle", actuator_name="FDS2_left"),
            ObservationField("goal", width=2),
        )
    )
    return schema.without_fingers()


def _patch_runtime_contract(monkeypatch):
    monkeypatch.setattr(
        "musclemimic.core.wrappers.finger_isolation.model_action_names",
        lambda _env: _action_names(),
    )
    monkeypatch.setattr(
        "musclemimic.algorithms.common.env_utils._resolve_runtime_ctrlrange",
        lambda _env, _names: np.tile(
            np.array([[0.0, 1.0]], dtype=np.float64),
            (354, 1),
        ),
    )
    monkeypatch.setattr(
        "musclemimic.core.wrappers.finger_isolation.build_body_observation_filter",
        lambda _env, **_kwargs: _observation_filter(),
    )


def _config():
    return OmegaConf.create(
        {
            "finger_isolation": {
                "enabled": True,
                "expected_partition": [354, 31, 31],
                "expected_removed_observation_dim": 4,
                "expected_policy_observation_dim": 5,
                "right_grip_provider": {"mode": "constant", "value": 0.25},
                "left_neutral_value": 0.0,
            },
            "actor_hidden_layers": [8],
            "critic_hidden_layers": [8],
            "activation": "tanh",
            "init_std": 1.0,
            "learnable_std": True,
            "use_moe": False,
            "use_layernorm": False,
            "use_residual": False,
        }
    )


def test_wrapper_filters_policy_obs_and_routes_three_action_owners(monkeypatch):
    _patch_runtime_contract(monkeypatch)
    env = _MockFingerEnv()
    wrapped = BodyFingerIsolationWrapper(env, _config().finger_isolation)

    assert wrapped.info.observation_space.shape == (5,)
    assert wrapped.info.action_space.shape == (354,)
    assert len(wrapped.policy_actuator_names) == 354
    assert wrapped.policy_interface_schema_hash
    np.testing.assert_array_equal(
        wrapped.obs_container.get_obs_ind_by_group("state"), np.array([0, 1, 2])
    )
    np.testing.assert_array_equal(
        wrapped.obs_container.get_obs_ind_by_group("goal"), np.array([3, 4])
    )

    obs, state = wrapped.reset(jnp.array([0, 1], dtype=jnp.uint32))
    np.testing.assert_array_equal(np.asarray(obs), np.array([0, 1, 4, 7, 8]))

    body_action = jnp.linspace(-0.5, 0.5, 354)
    next_obs, *_ = wrapped.step(state, body_action)
    np.testing.assert_array_equal(np.asarray(next_obs), np.array([10, 11, 14, 17, 18]))
    full = np.asarray(env.last_full_action)
    np.testing.assert_allclose(full[wrapped.partition.body_indices], body_action)
    np.testing.assert_allclose(full[wrapped.partition.right_grip_indices], 0.25)
    np.testing.assert_allclose(full[wrapped.partition.left_neutral_indices], 0.0)


def test_network_and_runtime_share_the_same_354d_interface(monkeypatch):
    _patch_runtime_contract(monkeypatch)
    env = _MockFingerEnv()
    exp = _config()

    interface_env = apply_policy_interface_wrappers(env, exp)
    assert isinstance(interface_env, BodyFingerIsolationWrapper)
    assert interface_env.info.action_space.shape == (354,)
    assert interface_env.info.observation_space.shape == (5,)

    config = OmegaConf.create({"experiment": OmegaConf.to_container(exp, resolve=True)})
    network = PPOJax._create_network(env, config)
    assert network.action_dim == 354
    np.testing.assert_array_equal(np.asarray(network.actor_obs_ind), np.arange(5))
    np.testing.assert_array_equal(np.asarray(network.critic_obs_ind), np.arange(5))


def test_wrapper_rejects_checkpoint_interface_hash_drift(monkeypatch):
    _patch_runtime_contract(monkeypatch)
    cfg = _config().finger_isolation
    cfg.expected_policy_observation_schema_hash = "0" * 64

    with pytest.raises(ValueError, match="policy observation schema hash mismatch"):
        BodyFingerIsolationWrapper(_MockFingerEnv(), cfg)
