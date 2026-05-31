from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.contact_graph import ContactGraphReport
from environment.overall_environment.src.phase_reward import (
    PhaseRewardConfig,
    compute_phase_gated_hit_reward,
)


def test_phase_gated_reward_positive_for_stringbed_contact_in_window():
    contact = ContactGraphReport(
        hand_handle_contacts=3,
        hand_handle_max_penetration=0.001,
        stringbed_shuttle_active=True,
        stringbed_rho2=0.2,
        stringbed_relative_normal_velocity=-8.0,
    )

    terms = compute_phase_gated_hit_reward(
        phase=0.5,
        contact_graph=contact,
        shuttle_state={"crossed_net": True},
        config=PhaseRewardConfig(impact_phase=0.5, phase_tolerance=0.08),
    )

    assert terms.impact > 0.0
    assert terms.post_impact > 0.0


def test_phase_gated_reward_penalizes_out_of_window_stringbed_contact():
    contact = ContactGraphReport(
        hand_handle_contacts=0,
        hand_handle_max_penetration=0.0,
        stringbed_shuttle_active=True,
        stringbed_rho2=0.2,
        stringbed_relative_normal_velocity=-8.0,
    )

    terms = compute_phase_gated_hit_reward(
        phase=0.2,
        contact_graph=contact,
        shuttle_state={"crossed_net": False},
        config=PhaseRewardConfig(impact_phase=0.5, phase_tolerance=0.08),
    )

    assert terms.impact == 0.0
    assert terms.penalties < 0.0
