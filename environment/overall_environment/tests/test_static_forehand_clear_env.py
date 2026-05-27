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


def test_static_env_calls_physics_hooks_after_release_only():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    calls: list[str] = []

    def stringbed_hook(model, data):
        calls.append("stringbed")
        return {"active": True, "relative_normal_velocity": -6.0, "normal_world": np.array([0.0, 0.0, 1.0])}

    def rebound_hook(contact_info):
        calls.append("rebound")
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
    assert info["state"] == "IMPACT_RELEASED"
    assert env.release_step == 1
