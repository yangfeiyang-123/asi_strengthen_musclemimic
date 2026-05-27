from __future__ import annotations

from pathlib import Path

import yaml

from BadmintonMimic.scripts.run_posttrain_experiment import _posttrain_arms, build_hydra_config, load_spec


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


def test_static_hit_spec_loads_with_posttrain_runner_schema():
    data = load_spec(SPEC)

    assert data["action"] == "ForehandClearStaticHit"
    assert data["arms"]


def test_static_hit_spec_generates_static_hit_hydra_env_params():
    data = load_spec(SPEC)
    arm = next(arm for arm in data["arms"] if arm["id"] == "E4_hit_and_over_net")

    config = build_hydra_config(data, arm)
    env_params = config["experiment"]["env_params"]
    static_hit_params = env_params["static_hit_params"]

    assert env_params["env_name"] == "StaticForehandClearEnv"
    assert env_params["disable_fingers"] is False
    assert static_hit_params["grip_policy"]["checkpoint"] == "outputs/right_hand_racket_grip/policy/policy_latest.pt"
    assert static_hit_params["shuttle"]["mode"] == "pre_impact_freeze_release"
    assert static_hit_params["curriculum_stage"] == "hit_and_over_net"
    assert "E1_physics_chain_validation" not in {arm["id"] for arm in _posttrain_arms(data)}
