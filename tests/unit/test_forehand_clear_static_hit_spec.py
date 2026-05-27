from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from BadmintonMimic.scripts.run_posttrain_experiment import (
    _posttrain_arms,
    build_hydra_config,
    load_spec,
    prepare_experiment,
    run_stage,
)


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml")


def test_static_hit_spec_declares_required_stages_and_checkpoints():
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert data["runner_type"] == "static_hit_staging"
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


def test_static_hit_prepare_writes_readme_without_fullbody_command_files(tmp_path: Path):
    data = load_spec(SPEC)
    data["output_root"] = str(tmp_path / "outputs" / "posttrain")
    data["hydra_config_root"] = str(tmp_path / "fullbody" / "config_specific_task" / "posttrain")

    stale_commands = tmp_path / "outputs" / "posttrain" / "ForehandClearStaticHit" / "v1" / "commands"
    stale_commands.mkdir(parents=True)
    (stale_commands / "train_E4_hit_and_over_net.sh").write_text("stale train\n", encoding="utf-8")
    (stale_commands / "eval_E0_baseline.sh").write_text("stale eval\n", encoding="utf-8")
    (stale_commands / "render_E0_baseline.sh").write_text("stale render\n", encoding="utf-8")

    result = prepare_experiment(data)

    readme = result.output_dir / "commands" / "README_static_hit.txt"
    assert readme.exists()
    assert "dedicated static-hit runner" in readme.read_text(encoding="utf-8")
    assert result.generated_configs["E4_hit_and_over_net"].output_copy.exists()
    assert not list((result.output_dir / "commands").glob("train_E4*.sh"))
    assert not list((result.output_dir / "commands").glob("eval_E0*.sh"))
    assert not list((result.output_dir / "commands").glob("render_E0*.sh"))


@pytest.mark.parametrize("stage", ["train", "eval", "render", "all"])
def test_static_hit_non_prepare_stages_fail_fast(tmp_path: Path, stage: str):
    data = load_spec(SPEC)
    data["output_root"] = str(tmp_path / "outputs" / "posttrain")
    data["hydra_config_root"] = str(tmp_path / "fullbody" / "config_specific_task" / "posttrain")

    with pytest.raises(ValueError, match="dedicated static-hit runner"):
        run_stage(data, stage=stage, arm=None, execute=False)
