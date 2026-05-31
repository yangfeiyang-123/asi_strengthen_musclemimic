from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.contact_graph import ContactGraphReport, contact_reward_terms


def test_contact_reward_terms_prefers_handle_and_stringbed_contact():
    report = ContactGraphReport(
        hand_handle_contacts=3,
        hand_handle_max_penetration=0.002,
        stringbed_shuttle_active=True,
        stringbed_rho2=0.2,
        stringbed_relative_normal_velocity=-7.0,
    )

    terms = contact_reward_terms(report)

    assert terms["hand_handle"] > 0.0
    assert terms["stringbed"] > 0.0
