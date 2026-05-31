from __future__ import annotations

from environment.overall_environment.src.soft_weld_schedule import soft_weld_schedule


def test_soft_weld_schedule_reduces_weld_and_increases_contact_weight():
    strong = soft_weld_schedule("strong_weld")
    weak = soft_weld_schedule("weak_weld")
    contact_only = soft_weld_schedule("contact_only")

    assert strong.weld_strength > weak.weld_strength > contact_only.weld_strength
    assert strong.reward_weights["contact"] < weak.reward_weights["contact"] < contact_only.reward_weights["contact"]
    assert contact_only.weld_enabled is False
