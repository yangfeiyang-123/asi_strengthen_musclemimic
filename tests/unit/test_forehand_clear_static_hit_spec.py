from __future__ import annotations

from pathlib import Path

import yaml


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml")


def test_static_hit_spec_declares_required_stages_and_checkpoints():
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert data["action"] == "ForehandClearStaticHit"
    assert data["body_policy"]["resume_from"]
    assert data["grip_policy"]["required"] is True
    assert data["grip_policy"]["checkpoint"] == "outputs/right_hand_racket_grip/policy/policy_latest.pt"
    assert [stage["name"] for stage in data["curriculum"]] == [
        "physics_chain_validation",
        "static_grip_stabilizer",
        "swing_disturbance_grip",
        "hit_and_over_net",
        "high_clear_depth",
    ]


def test_static_hit_spec_uses_freeze_release_shuttle_mode():
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert data["shuttle"]["mode"] == "pre_impact_freeze_release"
    assert data["shuttle"]["release"]["require_stringbed_contact"] is True
    assert data["shuttle"]["release"]["phase_tolerance"] == 0.08
