from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.layered_control import LayeredActuatorRouter
from environment.overall_environment.src.layered_policy import LatentBodyPolicy, LayeredPolicy


class FakePolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def act(self, obs):
        return self.action


def test_layered_policy_merges_frozen_body_and_grip_actions():
    body = FakePolicy([0.1, 0.2])
    grip = FakePolicy([0.7])
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "FDS2", "shoulder"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["FDS2"],
    )
    policy = LayeredPolicy(
        body_policy=body,
        grip_policy=grip,
        router=router,
    )

    output = policy.act(np.zeros(4))

    assert body.eval_called is True
    np.testing.assert_allclose(output.full_action, np.array([0.1, 0.7, 0.2]))
    assert output.body_action.shape == (2,)
    assert output.grip_action.shape == (1,)


def test_layered_policy_accepts_lab_body_policy_for_latent_body_action():
    class HighLevelPolicy(FakePolicy):
        pass

    class LabWrapper:
        def __call__(self, obs, raw_latent):
            np.testing.assert_allclose(obs, np.array([1.0, 2.0, 3.0]))
            np.testing.assert_allclose(raw_latent, np.array([0.25, -0.25]))
            return np.array([0.4, -0.2])

    high_level = HighLevelPolicy([0.25, -0.25])
    body = LatentBodyPolicy(high_level_policy=high_level, lab_wrapper=LabWrapper())
    grip = FakePolicy([0.7])
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "FDS2", "shoulder"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["FDS2"],
    )
    policy = LayeredPolicy(body_policy=body, grip_policy=grip, router=router)

    output = policy.act(np.array([1.0, 2.0, 3.0]))

    assert high_level.eval_called is True
    np.testing.assert_allclose(output.body_action, np.array([0.4, -0.2]))
    np.testing.assert_allclose(output.full_action, np.array([0.4, 0.7, -0.2]))


def test_latent_body_policy_passes_adapted_state_to_lab_wrapper():
    class LabWrapper:
        def __call__(self, state, raw_latent):
            np.testing.assert_allclose(state, np.array([10.0, 20.0]))
            np.testing.assert_allclose(raw_latent, np.array([0.25, -0.25]))
            return np.array([0.4, -0.2])

    def state_adapter(obs):
        np.testing.assert_allclose(obs, np.array([1.0, 10.0, 20.0, 3.0]))
        return obs[1:3]

    high_level = FakePolicy([0.25, -0.25])
    body = LatentBodyPolicy(
        high_level_policy=high_level,
        lab_wrapper=LabWrapper(),
        state_adapter=state_adapter,
        expected_body_size=2,
    )

    np.testing.assert_allclose(body.act(np.array([1.0, 10.0, 20.0, 3.0])), np.array([0.4, -0.2]))


def test_load_grip_policy_rejects_action_size_mismatch(tmp_path: Path):
    torch = pytest.importorskip("torch")

    checkpoint = {
        "metrics": {"obs_size": 4, "action_size": 2},
        "model_state_dict": {},
        "obs_rms": {"mean": [0, 0, 0, 0], "var": [1, 1, 1, 1], "count": 1.0},
    }
    path = tmp_path / "policy_latest.pt"
    torch.save(checkpoint, path)

    from environment.overall_environment.src.layered_policy import load_grip_policy

    with pytest.raises(ValueError, match="action_size"):
        load_grip_policy(path, obs_size=4, action_size=3)


def test_load_grip_policy_can_return_zero_policy_when_checkpoint_missing(tmp_path: Path):
    from environment.overall_environment.src.layered_policy import load_grip_policy

    policy = load_grip_policy(tmp_path / "missing.pt", obs_size=4, action_size=3, allow_random_init=True)

    np.testing.assert_allclose(policy.act(np.zeros(4)), np.zeros(3))
