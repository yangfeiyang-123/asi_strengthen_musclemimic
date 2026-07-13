from __future__ import annotations

import json
from pathlib import Path

from fullbody.eval_finger_robustness import (
    _run_eval_once,
    compare_finger_robustness,
)
from musclemimic.badminton.stage1r_artifact import LEGACY_EVIDENCE_KIND


def test_paired_finger_report_enforces_same_seed_and_thresholds():
    clean = {
        "seeds": [1, 2],
        "metrics": {
            "body_site_error": [0.10, 0.10],
            "right_hand_site_error": [0.12, 0.12],
            "racket_head_position_error": [0.20, 0.20],
            "racket_head_rotation_error": [0.30, 0.30],
            "early_termination": [0.0, 0.0],
            "val_max_err_root_xyz": [0.2, 0.2],
            "val_max_err_right_hand_pos": [0.3, 0.3],
            "val_max_err_racket_pos": [0.4, 0.4],
            "val_max_err_racket_rot": [0.5, 0.5],
        },
    }
    perturbed = {
        "seeds": [1, 2],
        "provenance": {"finger_qpos_perturb_scale": 0.03},
        "metrics": {
            "body_site_error": [0.102, 0.103],
            "right_hand_site_error": [0.122, 0.123],
            "racket_head_position_error": [0.205, 0.205],
            "racket_head_rotation_error": [0.305, 0.305],
            "early_termination": [0.0, 0.0],
            "val_max_err_root_xyz": [0.205, 0.205],
            "val_max_err_right_hand_pos": [0.305, 0.305],
            "val_max_err_racket_pos": [0.405, 0.405],
            "val_max_err_racket_rot": [0.505, 0.505],
        },
    }

    report = compare_finger_robustness(clean, perturbed)

    assert report["passed"] is True
    assert report["pair_count"] == 2
    assert report["seed_hash"]
    assert report["new_root_hand_racket_spike_count"] == 0
    assert report["finger_qpos_perturb_scale"] == 0.03
    assert report["evidence_kind"] == LEGACY_EVIDENCE_KIND
    assert report["production_eligible"] is False


def test_paired_finger_report_fails_on_a_new_single_rollout_spike():
    shared = {
        "body_site_error": [0.1],
        "right_hand_site_error": [0.1],
        "racket_head_position_error": [0.1],
        "racket_head_rotation_error": [0.1],
        "early_termination": [0.0],
        "val_max_err_root_xyz": [0.2],
        "val_max_err_right_hand_pos": [0.2],
        "val_max_err_racket_pos": [0.2],
        "val_max_err_racket_rot": [0.2],
    }
    perturbed = {key: list(value) for key, value in shared.items()}
    perturbed["val_max_err_right_hand_pos"] = [0.25]

    report = compare_finger_robustness(
        {"seeds": [7], "metrics": shared},
        {"seeds": [7], "metrics": perturbed},
    )

    assert report["passed"] is False
    assert report["new_root_hand_racket_spike_count"] == 1


def test_stage1r_eval_command_requests_real_racket_hand_and_max_metrics(monkeypatch, tmp_path):
    required = {
        "val_err_rpos": 0.1,
        "val_err_right_hand_pos": 0.1,
        "val_err_racket_pos": 0.1,
        "val_err_racket_rot": 0.1,
        "val_early_termination_rate": 0.0,
        "val_max_err_root_xyz": 0.2,
        "val_max_err_right_hand_pos": 0.2,
        "val_max_err_racket_pos": 0.2,
        "val_max_err_racket_rot": 0.2,
    }
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output = Path(command[command.index("--metrics_output_json") + 1])
        output.write_text(json.dumps(required), encoding="utf-8")

    monkeypatch.setattr("fullbody.eval_finger_robustness.subprocess.run", fake_run)

    result = _run_eval_once(
        checkpoint="/ckpt/stage1r",
        motion_paths=["heldout/video10"],
        seed=11,
        qpos_scale=0.03,
        qvel_scale=0.0,
        metrics_envs=5,
        metrics_steps=500,
    )

    assert result["val_err_right_hand_pos"] == 0.1
    assert "--metrics_deterministic" in seen["command"]
    assert seen["command"][seen["command"].index("--eval_seed") + 1] == "11"
    assert seen["command"][seen["command"].index("--finger_perturb_qpos_scale") + 1] == "0.03"
    assert seen["command"][seen["command"].index("--motion_path") + 1 :] == ["heldout/video10"]
