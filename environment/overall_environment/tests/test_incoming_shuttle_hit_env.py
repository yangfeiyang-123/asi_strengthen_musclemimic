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
    _bounded_best_progress,
    _discounted_event_direction_increment,
    _validate_contact_guidance_contract,
    _validate_reward_weights,
    ballistic_return_clearance_score,
    classify_return_net_crossing,
    counterfactual_rebound_guidance_score,
    drag_aware_return_clearance_score,
    inverse_impact_guidance_score,
)
from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    drag_aware_return_clearance_score_jax,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import build_feed_bank  # noqa: E402
from environment.shuttlecock.src.shuttlecock_racket_impact import (  # noqa: E402
    ShuttlecockImpactConfig,
)

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
        assert info["control_finite"] == 1.0
        assert info["normalized_control_energy"] == pytest.approx(float(np.mean(np.square(action))))
        assert info["body_action_saturation_fraction"] == pytest.approx(float(np.mean(np.abs(action) > 0.98)))
        assert info["full_action_saturation_fraction"] == info["body_action_saturation_fraction"]
        assert "raw_latent_saturation" not in info
        assert "lab_state_ood_fraction" not in info
        if terminated or truncated:
            break


def test_task_observation_uses_intercept_aligned_swing_phase(
    env: IncomingShuttleHitEnv,
) -> None:
    env.reset(feed_index=0)
    dt = env.control_substeps * float(env.model.opt.timestep)
    start = float(env.feed.intercept_time_s) - env.contact_phase * env.swing_duration_s
    env.step_index = max(0, int(round((start + 0.4 * env.swing_duration_s) / dt)))

    obs = env._observation()

    assert obs[-1] == pytest.approx(env._swing_phase(), abs=1e-6)
    episode_progress = min(1.0, env.step_index / max(env.max_episode_steps - 1, 1))
    assert abs(float(obs[-1]) - episode_progress) > 1e-3


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

    def __init__(self, *, event_rebound_used: bool = False, cfg=None) -> None:
        self.reset_called = False
        self.event_rebound_used = bool(event_rebound_used)
        if cfg is not None:
            self.cfg = cfg

    def reset(self) -> None:
        self.reset_called = True

    def substep(self, model, data) -> dict:
        return {
            "aero": None,
            # Positive is a valid separating-side high-speed contact after the
            # cork crosses the proxy plane within one substep.
            "stringbed": {
                "active": True,
                "relative_normal_velocity": 6.0,
                "rho2": 0.2,
            },
            "event_rebound_used": self.event_rebound_used,
            "event_rebound": None,
            "event_shuttle_velocity_before_world_m_s": np.array([-8.0, 0.0, -5.0]),
            "event_shuttle_velocity_after_world_m_s": np.array([8.0, 0.0, 5.0]),
            "event_racket_surface_velocity_world_m_s": np.array([5.0, 0.0, 2.0]),
            "event_stringbed_normal_world": np.array([1.0, 0.0, 0.0]),
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
        assert info["reward_terms"]["hit_bonus"] == pytest.approx(env.reward_weights["hit_bonus"])
    finally:
        env.physics = original_physics


def test_event_rebound_mode_does_not_reward_a_soft_or_separating_contact(
    env: IncomingShuttleHitEnv,
) -> None:
    original_physics = env.physics
    original_mode = env.hit_event_mode
    try:
        env.hit_event_mode = "event_rebound"
        env.physics = _StubPhysics(event_rebound_used=False)
        env.reset(feed_index=0)
        _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(env.action_size))
        assert info["stringbed_contact_this_step"] is True
        assert info["event_rebound_this_step"] is False
        assert info["hit_this_step"] is False
        assert info["reward_terms"]["hit_bonus"] == 0.0

        env.physics = _StubPhysics(event_rebound_used=True)
        env.reset(feed_index=0)
        _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(env.action_size))
        assert info["event_rebound_this_step"] is True
        assert info["hit_this_step"] is True
        assert info["reward_terms"]["hit_bonus"] == pytest.approx(env.reward_weights["hit_bonus"])
    finally:
        env.hit_event_mode = original_mode
        env.physics = original_physics


def test_cpu_hit_transition_exports_mjx_aligned_return_quality(feed_bank) -> None:
    cpu_env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        terminate_on_body_fall=False,
        min_return_net_clearance_m=0.20,
        clearance_prediction_mode="quadratic_drag_conservative_v1",
        hit_event_mode="event_rebound",
    )
    original_physics = cpu_env.physics
    try:
        cpu_env.physics = _StubPhysics(
            event_rebound_used=True,
            cfg=original_physics.cfg,
        )
        cpu_env.reset(feed_index=0)
        _obs, _reward, _terminated, _truncated, info = cpu_env.step(np.zeros(cpu_env.action_size))

        assert info["hit_this_step"] is True
        flight = info["flight"]
        velocity = np.asarray(flight["shuttle_velocity"], dtype=float)
        desired = cpu_env._desired_return_direction(flight["shuttle_xyz"])
        expected_direction = float(
            np.clip(
                np.dot(velocity / np.linalg.norm(velocity), desired),
                -1.0,
                1.0,
            )
        )
        expected_clearance = drag_aware_return_clearance_score(
            np.asarray(flight["shuttle_xyz"], dtype=float),
            velocity,
            player_half_sign=cpu_env.player_half_sign,
            net_x_m=cpu_env.return_net_x_m,
            net_height_m=cpu_env.return_net_height_m,
            min_clearance_m=cpu_env.min_return_net_clearance_m,
            terminal_velocity_m_s=original_physics.cfg.aero.terminal_velocity_m_s,
            drag_multiplier=1.0 + original_physics.cfg.aero.angle_drag_gain,
            score_softness_m=cpu_env.ballistic_return_score_softness_m,
        )

        assert info["return_direction_signed_score"] == pytest.approx(expected_direction)
        assert flight["return_direction_signed_score"] == pytest.approx(expected_direction)
        assert info["predicted_net_clearance_m"] == pytest.approx(expected_clearance["predicted_clearance_m"])
        assert flight["predicted_net_clearance_m"] == pytest.approx(expected_clearance["predicted_clearance_m"])
        assert info["return_clearance_score"] == pytest.approx(expected_clearance["score"])
        assert info["valid_net_cross_event"] is bool(flight["valid_net_crossing_event"])
        assert info["crossed_net"] is bool(flight["crossed_net"])
    finally:
        cpu_env.physics = original_physics


def test_reward_weights_validation() -> None:
    with pytest.raises(ValueError, match="unknown reward weight keys"):
        _validate_reward_weights({"not_a_term": 1.0})
    merged = _validate_reward_weights({"approach": 2.5})
    assert merged["approach"] == 2.5
    assert merged["hit_bonus"] > 0.0
    assert "racket_direction" in merged
    assert "return_direction" in merged
    assert "return_clearance" in merged
    assert "invalid_net_crossing" in merged
    assert "miss" in merged


def test_best_progress_guidance_telescopes_and_rejects_profitable_misses() -> None:
    best = 0.0
    increments = []
    for potential in (0.10, 0.40, 0.30, 0.80, 0.75):
        increment, best = _bounded_best_progress(potential, best)
        increments.append(increment)
    assert increments == pytest.approx([0.10, 0.30, 0.0, 0.40, 0.0])
    assert sum(increments) == pytest.approx(0.80)
    assert best == pytest.approx(0.80)

    safe = _validate_reward_weights(
        {
            "shuttle_proximity": 10.0,
            "timed_intercept": 10.0,
            "racket_direction": 5.0,
            "miss": 50.0,
            "hit_bonus": 60.0,
            "invalid_net_crossing": 8.0,
        }
    )
    assert _validate_contact_guidance_contract("best_progress", safe) == "best_progress"
    assert _validate_contact_guidance_contract("event_direction", safe) == "event_direction"
    assert _validate_contact_guidance_contract("potential_event_direction", safe) == "potential_event_direction"
    closest_safe = _validate_reward_weights(
        {
            "shuttle_proximity": 8.0,
            "timed_intercept": 8.0,
            "racket_direction": 120.0,
            "miss": 160.0,
            "hit_bonus": 300.0,
            "return_direction": 40.0,
            "return_clearance": 60.0,
            "invalid_net_crossing": 30.0,
            "landing_region": 20.0,
        }
    )
    assert (
        _validate_contact_guidance_contract("closest_approach_event_direction", closest_safe)
        == "closest_approach_event_direction"
    )
    with pytest.raises(ValueError, match="contact-guidance cap"):
        _validate_contact_guidance_contract(
            "best_progress",
            {**safe, "miss": 25.0},
        )
    with pytest.raises(ValueError, match="contact_guidance_reward_mode"):
        _validate_contact_guidance_contract("repeat_forever", safe)
    with pytest.raises(ValueError, match="pre-contact guidance cap"):
        _validate_contact_guidance_contract(
            "event_direction",
            {**safe, "miss": 20.0},
        )
    with pytest.raises(ValueError, match="terminal-guidance cap"):
        _validate_contact_guidance_contract(
            "closest_approach_event_direction",
            {**closest_safe, "miss": 136.0},
        )
    with pytest.raises(ValueError, match="worst real hit"):
        _validate_contact_guidance_contract(
            "closest_approach_event_direction",
            {**closest_safe, "hit_bonus": 200.0},
        )


def test_discounted_potential_event_direction_telescopes_to_event_only() -> None:
    gamma = 0.99
    reward_0, potential = _discounted_event_direction_increment(
        0.0,
        0.20,
        discount=gamma,
    )
    reward_1, potential = _discounted_event_direction_increment(
        potential,
        0.80,
        discount=gamma,
    )
    reward_2, potential = _discounted_event_direction_increment(
        potential,
        0.0,
        discount=gamma,
        event_score=0.10,
    )
    assert potential == 0.0
    discounted_return = reward_0 + gamma * reward_1 + gamma**2 * reward_2
    assert discounted_return == pytest.approx(gamma**2 * 0.10)

    miss_0, potential = _discounted_event_direction_increment(
        0.0,
        0.70,
        discount=gamma,
    )
    miss_1, potential = _discounted_event_direction_increment(
        potential,
        0.0,
        discount=gamma,
        terminal_without_event=True,
    )
    assert potential == 0.0
    assert miss_0 + gamma * miss_1 == pytest.approx(0.0)


def test_hit_event_mode_validation(feed_bank) -> None:
    with pytest.raises(ValueError, match="hit_event_mode"):
        IncomingShuttleHitEnv(
            SCENE_XML,
            feed_bank=feed_bank,
            hit_event_mode="unphysical_contact",
        )
    with pytest.raises(ValueError, match="racket_guidance_mode"):
        IncomingShuttleHitEnv(
            SCENE_XML,
            feed_bank=feed_bank,
            racket_guidance_mode="reference_pose_copy",
        )
    with pytest.raises(ValueError, match="clearance_reward_mode"):
        IncomingShuttleHitEnv(
            SCENE_XML,
            feed_bank=feed_bank,
            clearance_reward_mode="always_positive",
        )
    with pytest.raises(ValueError, match="requires hit_event_mode=event_rebound"):
        IncomingShuttleHitEnv(
            SCENE_XML,
            feed_bank=feed_bank,
            reward_weights={
                "shuttle_proximity": 8.0,
                "timed_intercept": 8.0,
                "miss": 20.0,
            },
            contact_guidance_reward_mode="event_direction",
            racket_guidance_mode="inverse_impact_decomposed",
        )
    with pytest.raises(ValueError, match="requires hit_event_mode=event_rebound"):
        IncomingShuttleHitEnv(
            SCENE_XML,
            feed_bank=feed_bank,
            reward_weights={
                "shuttle_proximity": 8.0,
                "timed_intercept": 8.0,
                "miss": 20.0,
            },
            contact_guidance_reward_mode="potential_event_direction",
            contact_guidance_discount=0.99,
            racket_guidance_mode="inverse_impact_decomposed",
        )


def test_event_direction_reward_is_zero_before_contact_and_uses_event_snapshot(feed_bank) -> None:
    event_env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        terminate_on_body_fall=False,
        reward_weights={
            "shuttle_proximity": 8.0,
            "timed_intercept": 8.0,
            "racket_direction": 25.0,
            "miss": 70.0,
        },
        contact_guidance_reward_mode="event_direction",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_decomposed",
        racket_velocity_direction_fraction=0.0,
        direction_reward_mode="signed_projection",
    )
    original_physics = event_env.physics
    try:
        event_env.physics = _StubPhysics(event_rebound_used=False)
        event_env.reset(feed_index=0)
        _obs, _reward, _terminated, _truncated, info = event_env.step(np.zeros(event_env.action_size))
        assert info["hit_this_step"] is False
        assert info["reward_terms"]["racket_direction"] == 0.0
        assert info["hit_event_direction_reward_score"] == 0.0

        event_env.physics = _StubPhysics(
            event_rebound_used=True,
            cfg=original_physics.cfg,
        )
        event_env.reset(feed_index=0)
        _obs, _reward, _terminated, _truncated, info = event_env.step(np.zeros(event_env.action_size))
        assert info["hit_this_step"] is True
        event_score = info["hit_event_direction_reward_score"]
        assert -1.0 <= event_score <= 1.0
        assert info["reward_terms"]["racket_direction"] == pytest.approx(
            event_env.reward_weights["racket_direction"] * event_score
        )
    finally:
        event_env.physics = original_physics


def test_closest_approach_direction_is_paid_once_at_miss_terminal(feed_bank) -> None:
    env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        terminate_on_body_fall=False,
        reward_weights={
            "shuttle_proximity": 8.0,
            "timed_intercept": 8.0,
            "racket_direction": 120.0,
            "miss": 160.0,
            "hit_bonus": 300.0,
            "return_direction": 40.0,
            "return_clearance": 60.0,
            "invalid_net_crossing": 30.0,
            "landing_region": 20.0,
        },
        contact_guidance_reward_mode="closest_approach_event_direction",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_decomposed",
        racket_velocity_direction_fraction=0.75,
        direction_reward_mode="signed_projection",
        clearance_reward_mode="signed_centered",
    )
    env.reset(feed_index=0)
    env._closest_racket_distance_m = 0.10
    env._closest_racket_direction_score = 0.40
    env.state = IncomingHitState.DONE
    env.termination_reason = "miss"
    terms = env._reward_terms(
        np.zeros(env.action_size),
        flight=env._flight_info(),
        hit_this_step=False,
        body_fall=False,
    )
    assert terms["racket_direction"] == pytest.approx(48.0)
    assert terms["miss"] == pytest.approx(-160.0)
    assert env._closest_racket_direction_terminal_score == pytest.approx(0.40)


def test_quality_hierarchy_penalizes_miss_and_below_net_return(feed_bank) -> None:
    miss_env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        reward_weights={"miss": 12.0},
    )
    miss_env.reset(feed_index=0)
    miss_env.state = IncomingHitState.DONE
    miss_env.termination_reason = "miss"
    miss_terms = miss_env._reward_terms(
        np.zeros(miss_env.action_size),
        flight=miss_env._flight_info(),
        hit_this_step=False,
        body_fall=False,
    )
    assert miss_terms["miss"] == pytest.approx(-12.0)

    return_env = IncomingShuttleHitEnv(
        SCENE_XML,
        feed_bank=feed_bank,
        min_return_net_clearance_m=0.2,
        clearance_reward_mode="signed_centered",
        reward_weights={"return_clearance": 10.0},
    )
    return_env.reset(feed_index=0)
    low_return = {
        **return_env._flight_info(),
        "shuttle_xyz": np.asarray([-2.5, 0.0, 1.8]),
        "shuttle_velocity": np.asarray([4.0, 0.0, -8.0]),
    }
    return_terms = return_env._reward_terms(
        np.zeros(return_env.action_size),
        flight=low_return,
        hit_this_step=True,
        body_fall=False,
    )
    assert return_terms["return_clearance"] < 0.0


def test_return_net_crossing_requires_player_to_opponent_clearance() -> None:
    low = classify_return_net_crossing(
        np.asarray([-0.2, 0.0, 1.40]),
        np.asarray([0.2, 0.0, 1.50]),
        player_half_sign=-1,
        net_height_m=1.55,
        min_clearance_m=0.20,
    )
    assert low["crossed"] is True
    assert low["valid"] is False
    assert low["crossing_height_m"] == pytest.approx(1.45)

    legal = classify_return_net_crossing(
        np.asarray([-0.2, 0.0, 1.90]),
        np.asarray([0.2, 0.0, 2.10]),
        player_half_sign=-1,
        net_height_m=1.55,
        min_clearance_m=0.20,
    )
    assert legal["crossed"] is True
    assert legal["valid"] is True
    assert legal["clearance_m"] == pytest.approx(0.45)

    incoming = classify_return_net_crossing(
        np.asarray([0.2, 0.0, 2.10]),
        np.asarray([-0.2, 0.0, 1.90]),
        player_half_sign=-1,
        net_height_m=1.55,
        min_clearance_m=0.20,
    )
    assert incoming["crossed"] is False


def test_return_direction_is_task_defined_not_reference_defined(
    env: IncomingShuttleHitEnv,
) -> None:
    direction = env._desired_return_direction(np.asarray([-2.5, 0.4, 2.0]))
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[0] > 0.0
    assert direction[1] < 0.0
    assert direction[2] > 0.0


def test_ballistic_clearance_shaping_uses_real_outgoing_shuttle_state() -> None:
    position = np.asarray([-2.5, 0.0, 2.0])
    high = ballistic_return_clearance_score(
        position,
        np.asarray([9.0, 0.0, 7.0]),
        player_half_sign=-1,
        min_clearance_m=0.20,
    )
    low = ballistic_return_clearance_score(
        position,
        np.asarray([6.0, 0.0, -2.0]),
        player_half_sign=-1,
        min_clearance_m=0.20,
    )
    wrong_way = ballistic_return_clearance_score(
        position,
        np.asarray([-6.0, 0.0, 7.0]),
        player_half_sign=-1,
        min_clearance_m=0.20,
    )

    assert high["predicted_clearance_m"] > 0.20
    assert high["score"] > 0.90
    assert low["predicted_clearance_m"] < 0.0
    assert low["score"] < 0.05
    assert wrong_way["score"] == 0.0


def test_ballistic_clearance_softness_preserves_gradient_for_bad_returns() -> None:
    position = np.asarray([-2.5, 0.0, 2.0])
    velocity = np.asarray([6.0, 0.0, -2.0])
    sharp = ballistic_return_clearance_score(
        position,
        velocity,
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=0.35,
    )
    broad = ballistic_return_clearance_score(
        position,
        velocity,
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=2.0,
    )

    assert broad["predicted_clearance_m"] == pytest.approx(sharp["predicted_clearance_m"])
    assert 0.0 < sharp["score"] < broad["score"] < 0.5


def test_drag_aware_clearance_rejects_vacuum_false_positive_and_accepts_real_reach() -> None:
    position = np.asarray([-3.4, 0.0, 2.35])
    weak_outgoing = np.asarray([6.057, 4.890, 1.397])
    vacuum = ballistic_return_clearance_score(
        position,
        weak_outgoing,
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=0.75,
    )
    drag = drag_aware_return_clearance_score(
        position,
        weak_outgoing,
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=0.75,
    )
    strong = drag_aware_return_clearance_score(
        position,
        np.asarray([12.0, 0.0, 4.0]),
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=0.75,
    )

    assert vacuum["predicted_clearance_m"] > 0.0
    assert drag["predicted_crosses_net"] is False
    assert drag["predicted_clearance_m"] < 0.0
    assert drag["predicted_landing_shortfall_m"] > 0.0
    assert strong["predicted_crosses_net"] is True
    assert strong["predicted_clearance_m"] > 0.20
    assert strong["predicted_landing_shortfall_m"] == pytest.approx(0.0)


def test_drag_aware_clearance_cpu_and_jax_are_batch_equivalent() -> None:
    positions = np.asarray(
        [
            [-3.4, 0.0, 2.35],
            [-3.4, 0.0, 2.35],
            [-2.5, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [6.057, 4.890, 1.397],
            [12.0, 0.0, 4.0],
            [6.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    cpu_rows = [
        drag_aware_return_clearance_score(
            position,
            velocity,
            player_half_sign=-1,
            min_clearance_m=0.20,
            score_softness_m=0.75,
        )
        for position, velocity in zip(positions, velocities, strict=True)
    ]
    jax_rows = drag_aware_return_clearance_score_jax(
        positions,
        velocities,
        player_half_sign=-1,
        min_clearance_m=0.20,
        score_softness_m=0.75,
    )

    for name in (
        "score",
        "predicted_clearance_m",
        "forward_speed_m_s",
        "prediction_time_s",
        "predicted_landing_shortfall_m",
    ):
        np.testing.assert_allclose(
            np.asarray(jax_rows[name]),
            np.asarray([row[name] for row in cpu_rows]),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
    np.testing.assert_array_equal(
        np.asarray(jax_rows["predicted_crosses_net"]),
        np.asarray([row["predicted_crosses_net"] for row in cpu_rows]),
    )


def test_counterfactual_rebound_guidance_prefers_upcourt_physical_impact() -> None:
    position = np.asarray([-2.5, 0.0, 2.0])
    incoming = np.asarray([-5.0, 0.0, -4.0])
    desired = np.asarray([1.0, 0.0, 1.0])
    desired /= np.linalg.norm(desired)
    impact = ShuttlecockImpactConfig()

    useful = counterfactual_rebound_guidance_score(
        position,
        incoming,
        np.asarray([9.0, 0.0, 7.0]),
        desired,
        desired,
        player_half_sign=-1,
        impact_config=impact,
        min_clearance_m=0.20,
        clearance_softness_m=1.0,
    )
    sideways_down = counterfactual_rebound_guidance_score(
        position,
        incoming,
        np.asarray([2.0, 18.0, -7.0]),
        np.asarray([1.0, 0.0, -1.0]),
        desired,
        player_half_sign=-1,
        impact_config=impact,
        min_clearance_m=0.20,
        clearance_softness_m=1.0,
    )

    assert useful["closing_gate"] > 0.95
    assert useful["direction_signed_score"] > sideways_down["direction_signed_score"]
    assert useful["clearance_score"] > sideways_down["clearance_score"]
    assert useful["score"] > sideways_down["score"]
    assert np.asarray(useful["predicted_velocity_m_s"])[2] > 0.0


def test_clearance_priority_removes_wrong_direction_reward_baseline() -> None:
    position = np.asarray([-2.5, 0.0, 2.0])
    incoming = np.asarray([-5.0, 0.0, -4.0])
    desired = np.asarray([1.0, 0.0, 1.0])
    desired /= np.linalg.norm(desired)
    impact = ShuttlecockImpactConfig()
    kwargs = {
        "player_half_sign": -1,
        "impact_config": impact,
        "min_clearance_m": 0.20,
        "clearance_softness_m": 1.0,
    }

    useful = counterfactual_rebound_guidance_score(
        position,
        incoming,
        np.asarray([9.0, 0.0, 7.0]),
        desired,
        desired,
        quality_mode="clearance_priority",
        **kwargs,
    )
    wrong_shifted = counterfactual_rebound_guidance_score(
        position,
        incoming,
        np.asarray([2.0, 18.0, -7.0]),
        np.asarray([1.0, 0.0, -1.0]),
        desired,
        **kwargs,
    )
    wrong_strict = counterfactual_rebound_guidance_score(
        position,
        incoming,
        np.asarray([2.0, 18.0, -7.0]),
        np.asarray([1.0, 0.0, -1.0]),
        desired,
        quality_mode="clearance_priority",
        **kwargs,
    )

    assert wrong_strict["direction_score"] == pytest.approx(max(float(wrong_strict["direction_signed_score"]), 0.0))
    assert wrong_strict["score"] < wrong_shifted["score"]
    assert useful["score"] > 4.0 * wrong_strict["score"]


def test_inverse_impact_guidance_exactly_inverts_event_rebound() -> None:
    incoming = np.asarray([-2.0, 0.2, -6.0])
    desired = np.asarray([1.0, -0.1, 1.0])
    desired /= np.linalg.norm(desired)
    impact = ShuttlecockImpactConfig()
    seed = inverse_impact_guidance_score(
        incoming,
        np.zeros(3),
        desired,
        desired,
        impact_config=impact,
        target_outgoing_speed_m_s=12.0,
        racket_velocity_softness_m_s=6.0,
    )
    exact = inverse_impact_guidance_score(
        incoming,
        np.asarray(seed["target_racket_velocity_m_s"]),
        np.asarray(seed["target_face_normal"]),
        desired,
        impact_config=impact,
        target_outgoing_speed_m_s=12.0,
        racket_velocity_softness_m_s=6.0,
    )
    wrong = inverse_impact_guidance_score(
        incoming,
        np.zeros(3),
        -np.asarray(seed["target_face_normal"]),
        desired,
        impact_config=impact,
        target_outgoing_speed_m_s=12.0,
        racket_velocity_softness_m_s=6.0,
    )
    wrong_with_target_velocity = inverse_impact_guidance_score(
        incoming,
        np.asarray(seed["target_racket_velocity_m_s"]),
        -np.asarray(seed["target_face_normal"]),
        desired,
        impact_config=impact,
        target_outgoing_speed_m_s=12.0,
        racket_velocity_softness_m_s=6.0,
    )

    assert exact["target_closing_speed_m_s"] > impact.min_speed_for_event_m_s
    assert exact["normal_alignment"] == pytest.approx(1.0)
    assert exact["racket_velocity_score"] == pytest.approx(1.0)
    assert exact["score"] == pytest.approx(1.0)
    assert exact["decomposed_score"] == pytest.approx(1.0)
    assert wrong["score"] == pytest.approx(0.0)
    assert wrong["signed_normal_alignment"] == pytest.approx(-1.0)
    assert wrong["decomposed_score"] > 0.0
    # Unlike the legacy product, the decomposed objective still distinguishes
    # velocity improvements while the racket face starts on the wrong side.
    assert wrong_with_target_velocity["score"] == pytest.approx(0.0)
    assert wrong_with_target_velocity["decomposed_score"] > wrong["decomposed_score"]
    np.testing.assert_allclose(
        exact["target_rebound_velocity_m_s"],
        exact["target_outgoing_velocity_m_s"],
        atol=1.0e-10,
    )


def test_landing_region_is_retained_in_terminal_info_after_reward(env: IncomingShuttleHitEnv) -> None:
    env.reset(feed_index=0)
    env.state = IncomingHitState.DONE
    env.termination_reason = "landed"
    env._landing_region = "opponent_back"
    env._hit_rewarded = True
    env._crossed_net_rewarded = True
    flight = env._flight_info() | {"crossed_net": True}

    first = env._reward_terms(
        np.zeros(env.action_size),
        flight=flight,
        hit_this_step=False,
        body_fall=False,
    )
    second = env._reward_terms(
        np.zeros(env.action_size),
        flight=flight,
        hit_this_step=False,
        body_fall=False,
    )

    assert first["landing_region"] > 0.0
    assert second["landing_region"] == 0.0
    assert env._info({})["landing_region"] == "opponent_back"


def test_opponent_landing_without_legal_crossing_is_penalized(
    env: IncomingShuttleHitEnv,
) -> None:
    env.reset(feed_index=0)
    env.state = IncomingHitState.DONE
    env.termination_reason = "landed"
    env._landing_region = "opponent_back"
    env._hit_rewarded = True
    env._crossed_net_rewarded = False

    terms = env._reward_terms(
        np.zeros(env.action_size),
        flight=env._flight_info(),
        hit_this_step=False,
        body_fall=False,
    )

    assert terms["landing_region"] == -env.reward_weights["landing_region"]
