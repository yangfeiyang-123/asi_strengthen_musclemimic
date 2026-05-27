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
