"""Tests for the two-player forehand-clear rally environment."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environment.double_play.src.build_double_play_scene import default_double_play_scene_path
from environment.double_play.src.double_play_env import (
    DOUBLE_PLAY_FEED_CONFIG,
    DOUBLE_PLAY_HIT_WINDOW,
    DoublePlayRallyEnv,
    _FlightRecord,
)
from environment.overall_environment.src.shuttle_feeder import sample_feed

SCENE_XML = default_double_play_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="double-play scene XML not built; run environment.double_play.src.build_double_play_scene",
)

CORK_LOCAL_OFFSET = np.array([0.0, 0.0, 0.011154])


@pytest.fixture(scope="module")
def env() -> DoublePlayRallyEnv:
    return DoublePlayRallyEnv(seed=0)


def _zero_actions(env: DoublePlayRallyEnv) -> dict[str, np.ndarray]:
    return {name: np.zeros(env.action_size) for name in ("p1", "p2")}


def _place_cork_on_stringbed(env: DoublePlayRallyEnv, player: str, *, closing_speed: float) -> None:
    binding = env.players[player]
    data, model = env.data, env.model
    origin = np.asarray(data.xpos[binding.racket_body], dtype=float)
    rot = np.asarray(data.xmat[binding.racket_body], dtype=float).reshape(3, 3)
    normal = rot[:, 2]
    stringbed_center = origin + rot @ np.array([0.0, 0.532, 0.0])
    cork_target = stringbed_center + 0.005 * normal
    qadr, dadr = env._shuttle_qadr, env._shuttle_dadr
    data.qpos[qadr : qadr + 3] = cork_target - CORK_LOCAL_OFFSET
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr : dadr + 3] = -closing_speed * normal
    data.qvel[dadr + 3 : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def test_reset_and_step_api(env: DoublePlayRallyEnv) -> None:
    assert env.action_size == 354
    obs, info = env.reset(serve_receiver="p1")
    assert set(obs) == {"p1", "p2"}
    for value in obs.values():
        assert value.shape == (env.observation_size,)
        assert np.isfinite(value).all()
    assert info["serve_receiver"] == "p1"
    assert info["expected_hitter"] == "p1"

    obs, rewards, terminated, truncated, info = env.step(_zero_actions(env))
    assert set(rewards) == {"p1", "p2"}
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for value in obs.values():
        assert np.isfinite(value).all()


def test_seeded_serves_are_deterministic() -> None:
    env_a = DoublePlayRallyEnv(seed=5)
    env_b = DoublePlayRallyEnv(seed=5)
    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    assert info_a["serve_receiver"] == info_b["serve_receiver"]
    np.testing.assert_allclose(obs_a["p1"], obs_b["p1"])
    np.testing.assert_allclose(obs_a["p2"], obs_b["p2"])


def test_observations_are_mirror_symmetric_at_reset() -> None:
    env_a = DoublePlayRallyEnv(seed=11)
    env_b = DoublePlayRallyEnv(seed=11)
    obs_a, _ = env_a.reset(serve_receiver="p1")
    obs_b, _ = env_b.reset(serve_receiver="p2")
    # A(serve to p1) seen by p1 == B(serve to p2) seen by p2, and vice versa
    np.testing.assert_allclose(obs_a["p1"], obs_b["p2"], atol=1e-9)
    np.testing.assert_allclose(obs_a["p2"], obs_b["p1"], atol=1e-9)

    # identical local actions keep rewards mirror-exact (physics trajectories
    # drift statistically through global solver-tolerance coupling)
    action = np.clip(np.random.default_rng(0).normal(0.0, 0.3, env_a.action_size), -1, 1)
    _, rewards_a, *_ = env_a.step({"p1": action, "p2": action})
    _, rewards_b, *_ = env_b.step({"p1": action, "p2": action})
    assert rewards_a["p1"] == pytest.approx(rewards_b["p2"], abs=1e-9)
    assert rewards_a["p2"] == pytest.approx(rewards_b["p1"], abs=1e-9)


def test_serve_feed_reaches_backcourt_hit_window() -> None:
    rng = np.random.default_rng(2)
    for _ in range(5):
        feed = sample_feed(rng, DOUBLE_PLAY_FEED_CONFIG, DOUBLE_PLAY_HIT_WINDOW)
        assert DOUBLE_PLAY_HIT_WINDOW.contains(feed.intercept_point[None, :])[0]
        assert feed.launch_pos[0] > 3.5  # from the opponent backcourt
        assert float(np.max(feed.trajectory[:, 3])) >= 3.0  # cleared like a clear


def test_legal_hit_switches_turn_and_rewards_hitter(env: DoublePlayRallyEnv) -> None:
    env.reset(serve_receiver="p1")
    _place_cork_on_stringbed(env, "p1", closing_speed=8.0)
    _, rewards, terminated, _, info = env.step(_zero_actions(env))
    assert not terminated
    assert env.rally_hits == 1
    assert env.last_hitter == "p1"
    assert env.expected_hitter == "p2"
    assert rewards["p1"] > env.reward_weights["hit_bonus"] - 1.0
    assert len(info["events"]) == 1


def test_wrong_hitter_faults_and_terminates(env: DoublePlayRallyEnv) -> None:
    env.reset(serve_receiver="p2")  # p1 is NOT the expected hitter
    _place_cork_on_stringbed(env, "p1", closing_speed=8.0)
    _, rewards, terminated, _, info = env.step(_zero_actions(env))
    assert terminated
    assert info["termination_reason"] == "wrong_hitter_p1"
    assert rewards["p1"] < -env.reward_weights["wrong_hitter"] + 1.0


def test_receiver_miss_is_penalized(env: DoublePlayRallyEnv) -> None:
    env.reset(serve_receiver="p1")
    # drop the shuttle straight down on p1's half, untouched
    qadr, dadr = env._shuttle_qadr, env._shuttle_dadr
    env.data.qpos[qadr : qadr + 3] = [-4.5, 0.0, 0.3]
    env.data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    terminated = False
    for _ in range(30):
        _, rewards, terminated, truncated, info = env.step(_zero_actions(env))
        if terminated or truncated:
            break
    assert terminated
    assert info["termination_reason"] == "receiver_miss_p1"
    assert rewards["p1"] < -env.reward_weights["miss"] + 1.5


def test_landing_after_hit_scores_deep_clear_region(env: DoublePlayRallyEnv) -> None:
    env.reset(serve_receiver="p1")
    # pretend p1 already hit; the shuttle is dropping deep in p2's backcourt
    env._flight = _FlightRecord(hitter="p1", apex_m=6.0, crossed=True)
    env.last_hitter = "p1"
    env.expected_hitter = "p2"
    env.rally_hits = 1
    qadr, dadr = env._shuttle_qadr, env._shuttle_dadr
    env.data.qpos[qadr : qadr + 3] = [5.8, 0.2, 0.3]
    env.data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    terminated = False
    for _ in range(30):
        _, rewards, terminated, truncated, info = env.step(_zero_actions(env))
        if terminated or truncated:
            break
    assert terminated
    assert info["termination_reason"] == "landed"
    assert info["landing_region"] == "opponent_back"
    # deep landing bonus + apex quality both go to the hitter
    assert rewards["p1"] > env.reward_weights["landing_region"] * 0.9


def test_full_rally_hit_exchange_updates_expected_hitter(env: DoublePlayRallyEnv) -> None:
    env.reset(serve_receiver="p2")
    _place_cork_on_stringbed(env, "p2", closing_speed=8.0)
    _, _, terminated, _, _ = env.step(_zero_actions(env))
    assert not terminated
    assert env.expected_hitter == "p1"
    _place_cork_on_stringbed(env, "p1", closing_speed=8.0)
    _, rewards, terminated, _, _ = env.step(_zero_actions(env))
    assert not terminated
    assert env.rally_hits == 2
    assert env.expected_hitter == "p2"
    # p2's cleared flight was returned: rally_continue settles to p2
    assert rewards["p2"] > env.reward_weights["rally_continue"] - 1.5
