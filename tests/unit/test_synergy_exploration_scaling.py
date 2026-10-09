"""Focused tests for physical-space exploration calibration and PPO wiring."""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.utils.env import Box
from musclemimic.algorithms.common.moe_networks import SoftMoEActorCritic
from musclemimic.algorithms.common.networks import ActorCritic
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.synergy.exploration_scaling import (
    SUPPORTED_STD_MODES,
    calibrate_exploration_std,
    physical_exploration_rms,
)


def test_all_modes_are_positive_finite_and_hit_physical_rms_target():
    jacobian = np.array(
        [
            [1.0, 0.8, 0.0, 0.0],
            [0.0, 0.6, 1.0, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )

    for mode in SUPPORTED_STD_MODES:
        std = calibrate_exploration_std(
            jacobian,
            0.2,
            mode=mode,
            min_std=1e-6,
            max_std=5.0,
        )

        assert std.shape == (4,)
        assert np.all(np.isfinite(std))
        assert np.all(std > 0.0)
        assert physical_exploration_rms(jacobian, std) == pytest.approx(0.2)


def test_scalar_calibration_matches_closed_form_without_active_bounds():
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    target = 0.3

    std = calibrate_exploration_std(jacobian, target, mode="scalar_calibrated")

    expected = target * np.sqrt(jacobian.shape[0] / np.sum(jacobian**2))
    np.testing.assert_allclose(std, expected)


def test_per_dimension_equalizes_individual_physical_contributions():
    jacobian = np.diag([1.0, 2.0, 4.0])

    std = calibrate_exploration_std(
        jacobian,
        0.25,
        mode="per_dimension",
        gram_epsilon=1e-12,
    )

    contributions = np.linalg.norm(jacobian, axis=0) * std
    np.testing.assert_allclose(contributions, contributions[0], rtol=1e-9)
    assert physical_exploration_rms(jacobian, std) == pytest.approx(0.25)


def test_gram_whitened_uses_inverse_gram_diagonal_profile():
    jacobian = np.array([[1.0, 0.9], [0.0, 0.2], [0.1, 0.0]])
    epsilon = 1e-5

    std = calibrate_exploration_std(
        jacobian,
        0.15,
        mode="gram_whitened",
        gram_epsilon=epsilon,
    )

    inverse_gram = np.linalg.inv(jacobian.T @ jacobian + epsilon * np.eye(2))
    expected_profile = np.sqrt(np.diag(inverse_gram))
    assert std[0] / std[1] == pytest.approx(expected_profile[0] / expected_profile[1])
    assert physical_exploration_rms(jacobian, std) == pytest.approx(0.15)


def test_residual_scale_applies_to_last_dimensions_before_rms_normalization():
    jacobian = np.eye(4)

    std = calibrate_exploration_std(
        jacobian,
        0.2,
        mode="scalar_calibrated",
        residual_dim=2,
        residual_std_scale=0.25,
    )

    np.testing.assert_allclose(std[:2], std[0])
    np.testing.assert_allclose(std[2:], std[0] * 0.25)
    assert physical_exploration_rms(jacobian, std) == pytest.approx(0.2)


def test_bounds_are_enforced_while_preserving_target_rms():
    # The first dimension reaches max_std while the second remains free, so
    # this exercises the bounded energy solve rather than post-hoc clipping.
    jacobian = np.diag([1.0, 10.0])

    std = calibrate_exploration_std(
        jacobian,
        1.0,
        mode="per_dimension",
        min_std=0.1,
        max_std=0.5,
    )

    assert np.all(std >= 0.1)
    assert np.all(std <= 0.5)
    assert std[0] == pytest.approx(0.5)
    assert physical_exploration_rms(jacobian, std) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_physical_rms": 0.0}, "target_physical_rms"),
        ({"target_physical_rms": 0.1, "mode": "unknown"}, "unsupported"),
        ({"target_physical_rms": 0.1, "residual_dim": 3}, "residual_dim"),
        ({"target_physical_rms": 0.1, "residual_std_scale": np.nan}, "residual_std_scale"),
    ],
)
def test_calibration_rejects_invalid_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        calibrate_exploration_std(np.eye(2), **kwargs)


def test_calibration_rejects_nonfinite_or_locally_constant_decoder():
    with pytest.raises(ValueError, match="finite"):
        calibrate_exploration_std(np.array([[np.nan]]), 0.1)
    with pytest.raises(ValueError, match="non-zero"):
        calibrate_exploration_std(np.zeros((2, 2)), 0.1)


def test_calibration_rejects_target_outside_std_bounds():
    with pytest.raises(ValueError, match="unattainable"):
        calibrate_exploration_std(
            np.eye(2),
            2.0,
            min_std=0.1,
            max_std=0.5,
        )


def _initialized_std(network) -> np.ndarray:
    variables = network.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    return np.exp(np.asarray(variables["params"]["log_std"]))


def test_actor_critic_vector_overrides_scalar_and_scalar_path_is_unchanged():
    vector_network = ActorCritic(
        action_dim=3,
        init_std=9.0,
        init_std_vector=(0.1, 0.2, 0.4),
        hidden_layer_dims=(4,),
    )
    scalar_network = ActorCritic(
        action_dim=3,
        init_std=0.7,
        hidden_layer_dims=(4,),
    )

    np.testing.assert_allclose(_initialized_std(vector_network), [0.1, 0.2, 0.4])
    np.testing.assert_allclose(_initialized_std(scalar_network), [0.7, 0.7, 0.7])


def test_actor_critic_rejects_invalid_std_vector():
    network = ActorCritic(
        action_dim=3,
        init_std_vector=(0.1, -0.2),
        hidden_layer_dims=(4,),
    )

    with pytest.raises(ValueError, match="shape must match"):
        _initialized_std(network)


def test_soft_moe_actor_critic_supports_per_dimension_std():
    network = SoftMoEActorCritic(
        action_dim=3,
        init_std=5.0,
        init_std_vector=(0.15, 0.25, 0.35),
        hidden_layer_dims=(4,),
        num_experts=2,
        moe_at_layers=(),
    )

    np.testing.assert_allclose(_initialized_std(network), [0.15, 0.25, 0.35])


class _NetworkEnv:
    def __init__(self):
        observation_space = Box(low=-np.ones(3), high=np.ones(3))
        action_space = Box(low=-np.ones(2), high=np.ones(2))
        self.info = types.SimpleNamespace(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.mdp_info = types.SimpleNamespace(observation_space=observation_space)


@pytest.mark.parametrize("use_moe", [False, True])
def test_ppo_network_creation_passes_vector_to_both_network_types(use_moe):
    config = OmegaConf.create(
        {
            "experiment": {
                "actor_hidden_layers": [4],
                "critic_hidden_layers": [4],
                "activation": "tanh",
                "init_std": None,
                "init_std_vector": [0.12, 0.34],
                "learnable_std": True,
                "use_moe": use_moe,
                "moe_config": {"num_experts": 2, "moe_at_layers": []},
            }
        }
    )

    network = PPOJax._create_network(_NetworkEnv(), config)

    assert network.init_std == 1.0  # ignored compatibility placeholder
    assert network.init_std_vector == (0.12, 0.34)


def test_ppo_rejects_vector_that_does_not_match_wrapped_action_dim():
    config = OmegaConf.create(
        {
            "experiment": {
                "actor_hidden_layers": [4],
                "critic_hidden_layers": [4],
                "activation": "tanh",
                "init_std": 1.0,
                "init_std_vector": [0.1],
                "learnable_std": True,
            }
        }
    )

    with pytest.raises(ValueError, match="wrapped action dimension"):
        PPOJax._create_network(_NetworkEnv(), config)
