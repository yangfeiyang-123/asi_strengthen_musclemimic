from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactGraphReport:
    hand_handle_contacts: int
    hand_handle_max_penetration: float
    stringbed_shuttle_active: bool
    stringbed_rho2: float
    stringbed_relative_normal_velocity: float


def contact_reward_terms(report: ContactGraphReport) -> dict[str, float]:
    hand_handle = min(1.0, max(0.0, float(report.hand_handle_contacts) / 3.0))
    if float(report.hand_handle_max_penetration) > 0.01:
        hand_handle -= 0.5

    stringbed = 0.0
    if bool(report.stringbed_shuttle_active) and float(report.stringbed_rho2) <= 1.0:
        stringbed = min(1.0, max(0.0, -float(report.stringbed_relative_normal_velocity)) / 8.0)

    return {
        "hand_handle": float(hand_handle),
        "stringbed": float(stringbed),
    }
