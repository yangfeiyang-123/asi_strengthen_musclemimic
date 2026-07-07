from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.incoming_shuttle_hit_env import (  # noqa: E402
    IncomingHitState,
    IncomingShuttleHitEnv,
    _validate_reward_weights,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import build_feed_bank  # noqa: E402

SCENE_XML = default_incoming_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)


@pytest.fixture(scope="module")
def feed_bank():
    return build_feed_bank(2, seed=13)


@pytest.fixture(scope="module")
def env(feed_bank) -> IncomingShuttleHitEnv:
    return IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        max_episode_steps=250,
        terminate_on_body_fall=False,
        seed=0,
    )


def test_reset_places_shuttle_on_feed(env: IncomingShuttleHitEnv, feed_bank) -> None:
    obs, info = env.reset(feed_index=0)
    feed = feed_bank[0]
    qadr, dadr = env._shuttle_qadr, env._shuttle_dadr
    np.testing.assert_allclose(env.data.qpos[qadr : qadr + 3], feed.launch_pos, atol=1e-9)
    np.testing.assert_allclose(env.data.qvel[dadr : dadr + 3], feed.launch_vel, atol=1e-9)
    assert info["state"] == IncomingHitState.INCOMING.value
    assert obs.size == env.observation_size
    assert np.isfinite(obs).all()


def test_step_finite_and_shapes(env: IncomingShuttleHitEnv) -> None:
    env.reset(feed_index=0)
    rng = np.random.default_rng(1)
    for _ in range(10):
        action = rng.uniform(-0.2, 0.2, env.action_size)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.size == env.observation_size
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert "reward_terms" in info and "flight" in info and "state" in info
        if terminated or truncated:
            break


def test_zero_action_episode_terminates(env: IncomingShuttleHitEnv) -> None:
    env.reset(feed_index=1)
    zero = np.zeros(env.action_size)
    terminated = truncated = False
    info: dict = {}
    for _ in range(env.max_episode_steps + 1):
        obs, reward, terminated, truncated, info = env.step(zero)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    assert terminated or truncated
    assert info.get("termination_reason") in {"miss", "landed", "time_limit"}


class _StubPhysics:
    """Returns a fake hit diagnostic without stepping MuJoCo."""

    def __init__(self) -> None:
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def substep(self, model, data) -> dict:
        return {
            "aero": None,
            "stringbed": {"active": True, "relative_normal_velocity": -6.0},
            "event_rebound_used": True,
            "event_rebound": None,
            "rebound_cooldown": 0,
        }


def test_state_machine_hit_transition(env: IncomingShuttleHitEnv) -> None:
    env.reset(feed_index=0)
    original_physics = env.physics
    try:
        env.physics = _StubPhysics()
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_size))
        assert info["state"] in {IncomingHitState.HIT.value, IncomingHitState.FLIGHT.value}
        assert info["hit_this_step"] is True
        assert info["reward_terms"]["hit_bonus"] > 0.0
    finally:
        env.physics = original_physics


def test_reward_weights_validation() -> None:
    with pytest.raises(ValueError, match="unknown reward weight keys"):
        _validate_reward_weights({"not_a_term": 1.0})
    merged = _validate_reward_weights({"approach": 2.5})
    assert merged["approach"] == 2.5
    assert merged["hit_bonus"] > 0.0
