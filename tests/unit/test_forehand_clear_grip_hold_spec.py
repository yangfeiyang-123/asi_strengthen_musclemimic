from __future__ import annotations

from pathlib import Path

import yaml

from BadmintonMimic.scripts.run_posttrain_experiment import (
    load_spec,
    prepare_experiment,
    requires_dedicated_static_hit_runner,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "BadmintonMimic" / "experiments" / "posttrain" / "forehand_clear_grip_hold_v1.yaml"


def test_forehand_clear_grip_hold_spec_uses_existing_local_checkpoint():
    data = load_spec(SPEC)

    assert data["action"] == "ForehandClearGripHold"
    assert data["runner_type"] == "forehand_clear_grip_hold"
    assert data["resume_from"] == str(REPO_ROOT / "checkpoints" / "de63059b16c0" / "checkpoint_7812")
    assert Path(data["resume_from"]).is_dir()
    assert data["shuttle"]["enabled"] is False
    assert data["grip_seed"]["path"] == "outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json"
    assert requires_dedicated_static_hit_runner(data) is False


def test_prepare_writes_forehand_clear_grip_hold_handoff(tmp_path: Path):
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    spec["output_root"] = str(tmp_path / "outputs" / "posttrain")
    spec["hydra_config_root"] = str(tmp_path / "fullbody" / "config_specific_task" / "posttrain")
    local_spec = tmp_path / "spec.yaml"
    local_spec.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    result = prepare_experiment(load_spec(local_spec))

    handoff = result.output_dir / "commands" / "README_forehand_clear_grip_hold.txt"
    assert handoff.is_file()
    text = handoff.read_text(encoding="utf-8")
    assert "forehand_clear_grip_hold" in text
    assert "no shuttle" in text.lower()
    assert not list((result.output_dir / "commands").glob("train_E*.sh"))
