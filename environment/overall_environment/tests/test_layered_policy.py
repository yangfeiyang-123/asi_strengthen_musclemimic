from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.layered_control import LayeredActuatorRouter
from environment.overall_environment.src.layered_policy import LayeredPolicy


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
