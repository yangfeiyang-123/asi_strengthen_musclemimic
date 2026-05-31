from pathlib import Path

import pytest
import yaml

from BadmintonMimic.scripts.run_posttrain_experiment import (
    build_eval_command,
    build_train_command,
    load_spec,
    prepare_experiment,
    run_stage,
)


def _write_spec(tmp_path: Path) -> Path:
    spec = {
        "experiment_id": "unit_v1",
        "action": "UnitAction",
        "family": "net_frontcourt",
        "output_root": str(tmp_path / "outputs" / "posttrain"),
        "hydra_config_root": str(tmp_path / "fullbody" / "config_specific_task" / "posttrain"),
        "resume_from": str(tmp_path / "checkpoints" / "base" / "checkpoint_10"),
        "checkpoint_root": str(tmp_path / "outputs" / "checkpoints"),
        "reference": {
            "train": ["UnitAction/best/video01_smpl", "UnitAction/best/video02_smpl.npz"],
            "validation": ["UnitAction/best/video03_smpl"],
            "stress_test": ["UnitAction/best/video04_smpl"],
            "excluded": ["UnitAction/best/video10_smpl"],
        },
        "training": {
            "base_config": "conf_fullbody_gmr",
            "num_envs": 128,
            "total_timesteps": 1024,
            "target_fps": 100,
            "num_steps": 32,
            "update_epochs": 2,
            "num_minibatches": 8,
            "lr": "5e-5",
            "reset_std_on_resume": 0.4,
            "wandb_mode": "disabled",
            "validation_start_from_beginning": True,
        },
        "env": {
            "amass_path": str(tmp_path / "data" / "amass_npz"),
            "converted_amass_path": str(tmp_path / "caches" / "AMASS"),
            "smpl_model_path": str(tmp_path / "smpl_models" / "smplh"),
        },
        "eval": {
            "render_motion": "UnitAction/best/video03_smpl",
            "n_steps": 300,
            "metrics_steps": 1,
            "record": True,
            "mujoco_gl": "osmesa",
            "start_from_beginning": True,
            "stochastic": False,
        },
        "arms": [
            {
                "id": "E0_baseline",
                "type": "baseline",
                "description": "evaluate base checkpoint only",
                "checkpoint": str(tmp_path / "checkpoints" / "base" / "checkpoint_10"),
            },
            {
                "id": "E1_root_hand_focus",
                "type": "posttrain",
                "description": "root and hand focus",
                "training": {"init_std": 0.7, "lr": "3e-5", "reset_std_on_resume": 0.25},
                "policy_anchor": {
                    "enabled": True,
                    "type": "hinge_action_mse",
                    "coeff": 0.003,
                    "margin": 0.02,
                },
                "reward": {
                    "root_pos_w_sum": 0.35,
                    "absolute_site_reward_sites": ["right_hand_mimic"],
                    "absolute_site_w_sum": 0.12,
                },
                "terminal": {"root_deviation_threshold": 0.30},
                "validation_terminal": {"root_deviation_threshold": 0.20},
            },
        ],
    }
    path = tmp_path / "posttrain.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_load_spec_normalizes_motion_paths(tmp_path: Path):
    spec_path = _write_spec(tmp_path)

    spec = load_spec(spec_path)

    assert spec["reference"]["train"] == [
        "UnitAction/best/video01_smpl",
        "UnitAction/best/video02_smpl",
    ]
    assert spec["arms"][0]["id"] == "E0_baseline"


def test_prepare_experiment_writes_hydra_configs_and_report(tmp_path: Path):
    spec = load_spec(_write_spec(tmp_path))

    result = prepare_experiment(spec)

    generated = result.generated_configs["E1_root_hand_focus"]
    assert generated.output_copy.exists()
    assert generated.hydra_config.exists()
    config = yaml.safe_load(generated.output_copy.read_text())
    assert config["experiment"]["resume_from"] == spec["resume_from"]
    assert config["experiment"]["lr"] == "3e-5"
    assert config["experiment"]["reset_std_on_resume"] == 0.25
    assert config["experiment"]["policy_anchor"] == {
        "enabled": True,
        "type": "hinge_action_mse",
        "coeff": 0.003,
        "margin": 0.02,
    }
    assert config["experiment"]["ppo_config"]["init_std"] == 0.7
    assert config["experiment"]["env_params"]["reward_params"]["root_pos_w_sum"] == 0.35
    assert config["experiment"]["env_params"]["terminal_state_params"]["root_deviation_threshold"] == 0.30
    assert config["experiment"]["validation"]["terminal_state_params"]["root_deviation_threshold"] == 0.20
    assert config["experiment"]["validation"]["start_from_beginning"] is True
    assert config["experiment"]["task_factory"]["params"]["amass_dataset_conf"]["rel_dataset_path"] == [
        "UnitAction/best/video01_smpl",
        "UnitAction/best/video02_smpl",
    ]
    assert result.report_path.exists()


def test_build_train_command_uses_generated_config_name(tmp_path: Path):
    spec = load_spec(_write_spec(tmp_path))
    result = prepare_experiment(spec)

    command = build_train_command(spec, "E1_root_hand_focus", result.generated_configs["E1_root_hand_focus"])

    assert command[:3] == ["uv", "run", "fullbody/experiment.py"]
    assert "--config-name=config_specific_task/posttrain/UnitAction/unit_v1/E1_root_hand_focus" in command
    assert 'wandb.mode=disabled' in command


def test_build_eval_command_uses_baseline_checkpoint_and_motion(tmp_path: Path):
    spec = load_spec(_write_spec(tmp_path))

    command = build_eval_command(spec, "E0_baseline", render=True)

    assert command[:3] == ["uv", "run", "fullbody/eval.py"]
    assert "--path" in command
    assert str(tmp_path / "checkpoints" / "base" / "checkpoint_10") in command
    assert "UnitAction/best/video03_smpl" in command
    assert "--start_from_beginning" in command
    assert "--stochastic" not in command
    assert "--record" in command


def test_build_eval_command_uses_latest_posttrain_checkpoint(tmp_path: Path):
    spec = load_spec(_write_spec(tmp_path))
    checkpoint_root = tmp_path / "outputs" / "checkpoints" / "E1_root_hand_focus"
    (checkpoint_root / "checkpoint_10").mkdir(parents=True)
    (checkpoint_root / "checkpoint_20").mkdir(parents=True)

    command = build_eval_command(spec, "E1_root_hand_focus", render=False)

    assert str(checkpoint_root / "checkpoint_20") in command


def test_build_eval_command_uses_latest_checkpoint_under_config_hash(tmp_path: Path):
    spec = load_spec(_write_spec(tmp_path))
    checkpoint_root = tmp_path / "outputs" / "checkpoints" / "E1_root_hand_focus"
    (checkpoint_root / "a8b3a9de7986" / "checkpoint_7907").mkdir(parents=True)
    (checkpoint_root / "a8b3a9de7986" / "checkpoint_7813").mkdir(parents=True)

    command = build_eval_command(spec, "E1_root_hand_focus", render=False)

    assert str(checkpoint_root / "a8b3a9de7986" / "checkpoint_7907") in command


def test_run_stage_rejects_grip_hold_train_stage(tmp_path: Path):
    spec = {
        "experiment_id": "v1",
        "action": "ForehandClearGripHold",
        "output_root": str(tmp_path / "outputs" / "posttrain"),
        "resume_from": "checkpoints/de63059b16c0/checkpoint_7812",
        "runner_type": "forehand_clear_grip_hold",
        "reference": {"train": ["m1"], "validation": ["m2"]},
        "arms": [{"id": "stage1", "description": "grip hold"}],
        "scene": {"xml": "environment/overall_environment/assets/overall_badminton_scene.xml"},
        "grip_seed": {"path": "outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json"},
    }

    with pytest.raises(ValueError, match="dedicated grip-hold runner"):
        run_stage(spec, stage="train", arm=None, execute=False)
