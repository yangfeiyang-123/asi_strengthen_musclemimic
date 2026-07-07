"""Tests for BC trainer validation helpers."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.distill.train_bc import evaluate_bc_loss, validate_dataset_matches_student_env


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
    def __init__(self):
        self.obs_container = MockObsContainer()
        self.info = MockInfo()
        self.mdp_info = self.info
        self.mjx_env = False


class MockDataset:
    def __init__(self, student_obs_dim):
        self.student_obs_dim = student_obs_dim


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


def test_evaluate_bc_loss_reports_mse_to_action_and_teacher_mu():
    jnp = pytest.importorskip("jax.numpy")

    class Dist:
        mean = jnp.zeros((2, 2), dtype=jnp.float32)

    class Network:
        def apply(self, _variables, obs):
            return Dist(), jnp.zeros((obs.shape[0],), dtype=jnp.float32)

    class TrainState:
        params = {}
        run_stats = {}

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
    assert metrics["mse_to_teacher_mu"] == 2.0
