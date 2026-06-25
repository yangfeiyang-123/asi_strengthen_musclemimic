from __future__ import annotations

from pathlib import Path

from BadmintonMimic.scripts.run_posttrain_experiment import load_spec, prepare_experiment
from BadmintonMimic.scripts.run_forehand_clear_static_hit import (
    load_static_hit_spec,
    preflight,
    static_hit_acceptance,
)


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml")


def test_static_hit_runner_preflight_requires_training_scene_and_no_pose_servo(tmp_path: Path):
    paths = load_static_hit_spec(SPEC)

    report = preflight(paths, out_dir=tmp_path)

    assert report["runner_type"] == "static_hit"
    assert report["scene_xml"].endswith("overall_badminton_training_scene.xml")
    assert report["scene_exists"] is True
    assert report["actuation_enabled"] is True
    assert report["hand_racket_contact_allowed"] is True
    assert report["pose_servo_allowed"] is False
    assert paths.shuttle_qpos.shape == (7,)
    assert paths.shuttle_qpos[2] > 1.0


def test_static_hit_acceptance_rejects_servo_drop_fall_and_missing_impact():
    report = {
        "finite": True,
        "pose_servo_enabled": True,
        "racket_drop": True,
        "body_fall": True,
        "impact_detected": False,
        "over_net": False,
        "target_landing_region": "opponent_back",
        "landing_region": "own_side",
    }
    validation = {
        "require_finite": True,
        "no_pose_servo": True,
        "no_racket_drop": True,
        "no_fall": True,
        "impact": {"require_detected": True},
        "flight": {"require_over_net": True, "target_landing_region": "opponent_back"},
    }

    verdict = static_hit_acceptance(report, validation)

    assert verdict["passed"] is False
    assert verdict["failures"] == [
        "pose_servo_enabled",
        "racket_drop",
        "body_fall",
        "impact_not_detected",
        "not_over_net",
        "landing_region_mismatch",
    ]


def test_static_hit_prepare_writes_dedicated_runner_commands(tmp_path: Path):
    data = load_spec(SPEC)
    data["output_root"] = str(tmp_path / "outputs" / "posttrain")
    data["hydra_config_root"] = str(tmp_path / "fullbody" / "config_specific_task" / "posttrain")

    result = prepare_experiment(data)

    preflight_cmd = result.output_dir / "commands" / "static_hit_preflight.sh"
    smoke_cmd = result.output_dir / "commands" / "static_hit_physics_smoke.sh"
    assert preflight_cmd.is_file()
    assert smoke_cmd.is_file()
    assert "run_forehand_clear_static_hit.py" in preflight_cmd.read_text(encoding="utf-8")
    assert "--stage preflight" in preflight_cmd.read_text(encoding="utf-8")
    assert "--stage physics-smoke" in smoke_cmd.read_text(encoding="utf-8")
