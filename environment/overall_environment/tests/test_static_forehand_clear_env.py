from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.static_forehand_clear_env import (  # noqa: E402
    FlightRegion,
    StaticHitState,
    StaticShuttleTarget,
    classify_landing_region,
    release_condition_met,
    should_transition_to_flight_evaluation,
)


def test_release_condition_requires_active_fast_closing_contact_in_phase_window():
    contact = {
        "active": True,
        "rho2": 0.4,
        "penetration": 0.003,
        "relative_normal_velocity": -6.0,
    }

    assert release_condition_met(contact, phase=0.52, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met(contact, phase=0.80, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met({**contact, "rho2": 1.2}, phase=0.52, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met(
        {**contact, "relative_normal_velocity": 1.0},
        phase=0.52,
        impact_phase=0.50,
        phase_tolerance=0.08,
    )


def test_static_shuttle_target_freeze_writes_qpos_and_qvel():
    qpos = np.zeros(10)
    qvel = np.ones(9)
    target = StaticShuttleTarget(
        qpos_adr=2,
        qvel_adr=3,
        qpos=np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
    )

    target.apply_freeze(qpos, qvel)

    np.testing.assert_allclose(qpos[2:9], target.qpos)
    np.testing.assert_allclose(qvel[3:9], np.zeros(6))


def test_landing_region_classifies_opponent_back_court():
    assert classify_landing_region(
        landing_xy=np.array([5.9, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OPPONENT_BACK
    assert classify_landing_region(
        landing_xy=np.array([-3.0, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OWN_SIDE
    assert classify_landing_region(
        landing_xy=np.array([7.2, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OUT


def test_transition_to_flight_evaluation_after_net_crossing_or_landing():
    assert should_transition_to_flight_evaluation(StaticHitState.IMPACT_RELEASED, crossed_net=True, landed=False)
    assert should_transition_to_flight_evaluation(StaticHitState.IMPACT_RELEASED, crossed_net=False, landed=True)
    assert not should_transition_to_flight_evaluation(
        StaticHitState.PRE_IMPACT_FREEZE,
        crossed_net=True,
        landed=False,
    )


def test_static_hit_reward_positive_for_valid_impact_contact():
    from environment.overall_environment.src.static_forehand_clear_env import compute_static_hit_reward_terms

    terms = compute_static_hit_reward_terms(
        phase=0.5,
        impact_phase=0.5,
        phase_tolerance=0.08,
        contact_info={"active": True, "rho2": 0.1, "penetration": 0.002, "relative_normal_velocity": -8.0},
        flight_info={"region": "opponent_back", "crossed_net": True, "landed": True},
    )

    assert terms["impact"] > 0.0
    assert terms["flight"] > 0.0
    assert sum(terms.values()) > 0.0


def test_static_hit_reward_penalizes_out_of_phase_contact():
    from environment.overall_environment.src.static_forehand_clear_env import compute_static_hit_reward_terms

    terms = compute_static_hit_reward_terms(
        phase=0.1,
        impact_phase=0.5,
        phase_tolerance=0.08,
        contact_info={"active": True, "rho2": 0.1, "penetration": 0.002, "relative_normal_velocity": -8.0},
        flight_info={"region": "own_side", "crossed_net": False, "landed": True},
    )

    assert terms["impact"] == 0.0
    assert terms["flight"] <= 0.0


class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(12)
        self.qvel = np.ones(11)


class _FakeBaseEnv:
    def __init__(self) -> None:
        self.data = _FakeData()
        self.step_count = 0

    def reset(self):
        return np.zeros(3), {"base_reset": True}

    def step(self, ctrl=None, pose_servo=False):
        self.step_count += 1
        return np.array([float(self.step_count)]), {"base_step": self.step_count}


def test_static_env_reset_enters_pre_impact_freeze_and_freezes_shuttle():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
    )

    _obs, info = env.reset()

    assert info["state"] == "PRE_IMPACT_FREEZE"
    np.testing.assert_allclose(base.data.qpos[1:8], target.qpos)
    np.testing.assert_allclose(base.data.qvel[2:8], np.zeros(6))


def test_static_env_step_keeps_shuttle_frozen_before_release():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
    )
    env.reset()
    base.data.qpos[1:8] = 4.0
    base.data.qvel[2:8] = 5.0

    _obs, _reward, terminated, truncated, info = env.step(
        ctrl=None,
        phase=0.1,
        contact_info={"active": False},
    )

    assert not terminated
    assert not truncated
    assert info["state"] == "PRE_IMPACT_FREEZE"
    np.testing.assert_allclose(base.data.qpos[1:8], target.qpos)
    np.testing.assert_allclose(base.data.qvel[2:8], np.zeros(6))


def test_static_env_step_returns_reward_terms_after_release():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(base, target, impact_phase=0.5, phase_tolerance=0.1)
    env.reset()

    _obs, reward, terminated, truncated, info = env.step(
        ctrl=None,
        phase=0.5,
        contact_info={"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -3.0},
    )

    assert reward > 0.0
    assert not terminated
    assert not truncated
    assert "reward_terms" in info


def test_static_env_calls_physics_hooks_after_release_only():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    calls: list[str] = []
    rebound_contacts = []

    def stringbed_hook(model, data):
        calls.append("stringbed")
        return {"active": True, "relative_normal_velocity": -6.0, "normal_world": np.array([0.0, 0.0, 1.0])}

    def rebound_hook(contact_info):
        calls.append("rebound")
        rebound_contacts.append(contact_info)
        return True

    def aero_hook(model, data):
        calls.append("aero")
        return {"speed_m_s": 12.0}

    base = _FakeBaseEnv()
    base.model = object()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
        stringbed_hook=stringbed_hook,
        rebound_hook=rebound_hook,
        aero_hook=aero_hook,
    )
    env.reset()

    env.step(ctrl=None, phase=0.1, contact_info={"active": False})
    assert calls == []

    _obs, _reward, _terminated, _truncated, info = env.step(
        ctrl=None,
        phase=0.5,
        contact_info={"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -3.0},
    )
    assert calls == ["stringbed", "rebound", "aero"]
    assert "normal_world" in rebound_contacts[0]
    assert info["state"] == "IMPACT_RELEASED"
    assert env.release_step == 1


def test_static_env_calls_rebound_hook_after_release_without_stringbed_hook():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    calls: list[str] = []

    def rebound_hook(contact_info):
        calls.append("rebound")
        return True

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
        rebound_hook=rebound_hook,
    )
    env.reset()

    _obs, _reward, _terminated, _truncated, info = env.step(
        ctrl=None,
        phase=0.5,
        contact_info={"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -3.0},
    )

    assert info["event_rebound_used"] is True
    assert calls == ["rebound"]


def test_static_env_step_no_longer_requires_external_phase_or_contact_info():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
        episode_steps=10,
    )
    env.reset()

    _obs, _reward, terminated, truncated, info = env.step(ctrl=None)

    assert not terminated
    assert not truncated
    assert info["phase"] == 0.0
    assert info["contact_info"]["active"] is False
    assert info["state"] == "PRE_IMPACT_FREEZE"


def test_static_env_internal_contact_release_rebound_and_landing_termination():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    contacts = [
        {"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -8.0},
        {"active": False},
    ]

    def detector(_model, _data):
        return contacts.pop(0) if contacts else {"active": False}

    def rebound(contact_info):
        base.data.qvel[2:5] = np.array([20.0, 0.0, 5.0])
        return bool(contact_info["active"])

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.0,
        phase_tolerance=0.05,
        stringbed_hook=detector,
        rebound_hook=rebound,
        episode_steps=20,
    )
    env.reset()

    _obs, reward, terminated, _truncated, info = env.step(ctrl=None)
    assert reward > 0.0
    assert terminated is False
    assert info["state"] == "FLIGHT_EVALUATION"
    assert info["event_rebound_used"] is True
    assert info["flight"]["crossed_net"] is True

    base.data.qpos[1:3] = np.array([5.9, 0.2])
    base.data.qpos[3] = 0.01
    _obs, _reward, terminated, _truncated, info = env.step(ctrl=None)

    assert terminated is True
    assert info["state"] == "TERMINATED"
    assert info["termination_reason"] == "landed"
    assert info["flight"]["region"] == "opponent_back"


def test_static_env_loads_training_scene_and_steps_without_external_phase_or_contact():
    import mujoco

    from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv
    from environment.overall_environment.src.training_scene import default_training_scene_path

    base = OverallBadmintonEnvironment(default_training_scene_path())
    shuttle_joint = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    assert shuttle_joint >= 0
    qpos_adr = int(base.model.jnt_qposadr[shuttle_joint])
    qvel_adr = int(base.model.jnt_dofadr[shuttle_joint])
    target = StaticShuttleTarget(
        qpos_adr=qpos_adr,
        qvel_adr=qvel_adr,
        qpos=np.array([-0.2, 0.0, 1.4, 1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.08,
        episode_steps=30,
    )

    obs, reset_info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(ctrl=np.zeros(base.model.nu, dtype=float))

    assert np.isfinite(obs).all()
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert reset_info["state"] == "PRE_IMPACT_FREEZE"
    assert info["contact_info"]["active"] is False
    assert info["flight"]["landed"] is False
