"""Tests for BC trainer validation helpers."""

from typing import ClassVar

import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.distill.train_bc import (
    _validate_gaussian_kl_action_semantics,
    evaluate_bc_loss,
    validate_dataset_matches_student_env,
)


class MockObsContainer:
    def __init__(self):
        self._groups = {
            "state": np.arange(4, dtype=int),
            "goal": np.arange(4, 8, dtype=int),
        }

    def get_obs_ind_by_group(self, group_name):
        return self._groups.get(group_name, np.array([], dtype=int))


class MockInfo:
    def __init__(self):
        self.observation_space = Box(
            low=np.full(8, -np.inf, dtype=np.float32),
            high=np.full(8, np.inf, dtype=np.float32),
        )
        self.action_space = Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
        )


class MockEnv:
    def __init__(self, policy_action_names=None):
        self.obs_container = MockObsContainer()
        self.info = MockInfo()
        self.mdp_info = self.info
        self.mjx_env = False
        if policy_action_names is not None:
            self.policy_action_names = policy_action_names


class MockDataset:
    def __init__(self, student_obs_dim, action_dim=None, actuator_names=None):
        self.student_obs_dim = student_obs_dim
        if action_dim is not None:
            self.action_dim = action_dim
        if actuator_names is not None:
            self.actuator_names = actuator_names


def _config():
    return OmegaConf.create(
        {
            "experiment": {
                "student_obs_filter": {
                    "enabled": True,
                    "drop_goal_lookahead": True,
                    "keep_motion_phase": True,
                    "require_goal_group": True,
                    "require_motion_phase": True,
                },
                "len_obs_history": 1,
                "split_goal": False,
                "normalize_env": False,
                "gamma": 0.99,
            }
        }
    )


def test_validate_dataset_matches_student_env_accepts_expected_dim():
    expected = validate_dataset_matches_student_env(
        dataset=MockDataset(student_obs_dim=5),
        env=MockEnv(),
        config=_config(),
    )

    assert expected == 5


def test_validate_dataset_matches_student_env_rejects_mismatched_dim():
    with pytest.raises(ValueError, match="student_obs_dim"):
        validate_dataset_matches_student_env(
            dataset=MockDataset(student_obs_dim=8),
            env=MockEnv(),
            config=_config(),
        )


def test_validate_dataset_rejects_action_dimension_and_name_order_drift():
    with pytest.raises(ValueError, match="action_dim"):
        validate_dataset_matches_student_env(
            dataset=MockDataset(student_obs_dim=5, action_dim=3),
            env=MockEnv(),
            config=_config(),
        )

    with pytest.raises(ValueError, match="names/order"):
        validate_dataset_matches_student_env(
            dataset=MockDataset(student_obs_dim=5, action_dim=2, actuator_names=["b", "a"]),
            env=MockEnv(policy_action_names=["a", "b"]),
            config=_config(),
        )


def test_evaluate_bc_loss_reports_mse_to_action_and_teacher_mu():
    jnp = pytest.importorskip("jax.numpy")

    class Dist:
        mean = jnp.zeros((2, 2), dtype=jnp.float32)

    class Network:
        def apply(self, _variables, obs):
            return Dist(), jnp.zeros((obs.shape[0],), dtype=jnp.float32)

    class TrainState:
        params: ClassVar[dict] = {}
        run_stats: ClassVar[dict] = {}

    class Dataset:
        def iter_batches(self, batch_size, shuffle=False, repeat=False):
            del batch_size, shuffle, repeat
            yield {
                "student_obs": np.zeros((2, 3), dtype=np.float32),
                "teacher_action": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
                "teacher_mu": np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32),
                "teacher_value": np.zeros((2,), dtype=np.float32),
            }

    metrics = evaluate_bc_loss(TrainState(), Network(), Dataset(), batch_size=2)

    assert metrics["action_mse"] == metrics["mse_to_teacher_action"]
    assert metrics["mse_to_teacher_action"] == 0.5
    assert metrics["deterministic_action_mse"] == 0.5
    assert metrics["mse_to_teacher_mu"] == 2.0


def test_evaluate_bc_loss_convergence_metric_uses_clipped_deterministic_action():
    jnp = pytest.importorskip("jax.numpy")

    class Dist:
        mean = jnp.array([[2.0], [2.0]], dtype=jnp.float32)

    class Network:
        def apply(self, _variables, obs):
            return Dist(), jnp.zeros((obs.shape[0],), dtype=jnp.float32)

    class TrainState:
        params: ClassVar[dict] = {}
        run_stats: ClassVar[dict] = {}

    class Dataset:
        def iter_batches(self, batch_size, shuffle=False, repeat=False):
            del batch_size, shuffle, repeat
            yield {
                "student_obs": np.zeros((2, 3), dtype=np.float32),
                "teacher_action": np.ones((2, 1), dtype=np.float32),
                "teacher_value": np.zeros((2,), dtype=np.float32),
            }

    metrics = evaluate_bc_loss(TrainState(), Network(), Dataset(), batch_size=2)

    assert metrics["action_mse"] == 1.0
    assert metrics["deterministic_action_mse"] == 0.0


def test_decoded_354_targets_cannot_fake_a_diagonal_gaussian_for_kl():
    unavailable = "unavailable_for_nonlinear_decoded_body_action"
    decoded_dataset = type(
        "DecodedDataset",
        (),
        {"metadata": {"teacher_log_std_semantics": unavailable}},
    )()

    with pytest.raises(ValueError, match="no diagonal Gaussian exists"):
        _validate_gaussian_kl_action_semantics(
            train_dataset=decoded_dataset,
            val_dataset=None,
            gaussian_kl_weight=0.1,
        )

    # MSE-only distillation may consume the decoded 354-D target; c/rho keeps
    # the actual policy Gaussian in its separate coordinate schema.
    _validate_gaussian_kl_action_semantics(
        train_dataset=decoded_dataset,
        val_dataset=None,
        gaussian_kl_weight=0.0,
    )
