from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.layered_control import LayeredActuatorRouter
from environment.overall_environment.src.layered_policy import LayeredPolicy


class FakePolicy:
    def __init__(self, size):
        self.size = int(size)

    def eval(self):
        return self

    def act(self, obs):
        return np.zeros(self.size, dtype=float)


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return np.zeros(5, dtype=float), {}

    def step(self, action):
        self.steps += 1
        assert action.shape == (3,)
        return np.ones(5, dtype=float), 0.0, False, self.steps >= 100, {}


def test_layered_rollout_100_steps_no_nan():
    env = FakeEnv()
    router = LayeredActuatorRouter(["hip", "FDS2", "shoulder"], ["hip", "shoulder"], ["FDS2"])
    policy = LayeredPolicy(body_policy=FakePolicy(2), grip_policy=FakePolicy(1), router=router)

    obs, _info = env.reset()
    for _ in range(100):
        output = policy.act(obs)
        assert np.isfinite(output.full_action).all()
        obs, reward, terminated, truncated, _info = env.step(output.full_action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break

    assert env.steps == 100
