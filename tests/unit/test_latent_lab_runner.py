from __future__ import annotations

import json
from pathlib import Path

import yaml

import fullbody.latent_run_lab_ppo as runner


def test_lab_manifest_launches_standalone_stage3_runner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner,
        "load_latent_checkpoint",
        lambda _path: {
            "eval_metrics": {"promotion": {"passed": True}},
            "action_mask": {
                "decoder_action_dim": 354,
                "correction_action_dim": 31,
                "neutral_action_dim": 31,
                "correction_actuator_names": ["right"] * 31,
            },
        },
    )
    spec = tmp_path / "stage3.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "stage3_lab": {
                    "enabled": True,
                    "curriculum": {"lambda_start": 0.25},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = runner.build_lab_ppo_manifest(
        latent_checkpoint_dir="latent/checkpoint",
        highlevel_config=str(spec),
        output_dir=str(tmp_path / "run"),
        lambda_lab=0.25,
        dry_run=True,
        num_envs=32,
        rollout_steps=8,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "stage3_lab_standalone_v2"
    command = payload["command"]
    assert Path(command[0]).resolve() == Path(runner.sys.executable).resolve()
    assert command[1:3] == [
        "-m",
        "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
    ]
    assert "fullbody/experiment.py" not in command
    assert command[command.index("--stage") + 1] == "train-gpu"
    assert command[command.index("--latent-checkpoint") + 1] == "latent/checkpoint"
