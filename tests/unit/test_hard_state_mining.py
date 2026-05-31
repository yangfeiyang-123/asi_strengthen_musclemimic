from __future__ import annotations

from environment.overall_environment.src.hard_state_mining import classify_failure


def test_classify_failure_detects_grip_slip_before_contact():
    failure = classify_failure(
        {
            "grip_slip_m": 0.08,
            "stringbed_contact": False,
            "shuttle_crossed_net": False,
            "fell": False,
        }
    )

    assert failure == "grip_slip"
