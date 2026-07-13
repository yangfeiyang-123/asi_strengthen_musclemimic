from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from musclemimic.distill.eval_student import (
    DistillAcceptanceThresholds,
    canonicalize_eval_metrics,
    evaluate_distill_acceptance,
    evaluate_mse_plateau,
    evaluate_temporal_drift,
    run_eval_metrics,
    write_acceptance_outputs,
)
from musclemimic.distill.provenance import validate_direct_acceptance_record


def _good_convergence():
    result = evaluate_mse_plateau(
        [
            {"step": 0, "action_mse": 0.1000},
            {"step": 10, "action_mse": 0.1004},
            {"step": 20, "action_mse": 0.0998},
            {"step": 30, "action_mse": 0.1001},
            {"step": 40, "action_mse": 0.1000},
        ]
    )
    result.update(
        {
            "deterministic": True,
            "split": "val",
            "evaluation_interval_steps": 10,
            "motion_field": "motion_uid",
            "motion_split": {
                "schema_version": "motion_split_v1",
                "mode": "explicit_motion_shards",
                "motion_field": "motion_uid",
                "seed": 0,
                "val_fraction": 0.2,
                "train_motion_ids": [1, 2],
                "val_motion_ids": [3],
                "train_num_samples": 200,
                "val_num_samples": 100,
            },
        }
    )
    return result


def _good_temporal_audit():
    horizon = 30
    rollout_count = 5
    steps = np.tile(np.arange(horizon, dtype=np.int32), rollout_count)
    rollouts = np.repeat(np.arange(rollout_count, dtype=np.int64), horizon)
    motions = np.full(steps.shape, 7, dtype=np.int64)
    teacher = np.stack(
        [np.sin(steps * 0.31), np.cos(steps * 0.17)], axis=-1
    ).astype(np.float32)
    result = evaluate_temporal_drift(
        student_action=teacher,
        teacher_action=teacher,
        motion_uid=motions,
        rollout_uid=rollouts,
        traj_step=steps,
        actuator_names=["muscle_a", "muscle_b"],
        checkpoint_actuator_names=["muscle_a", "muscle_b"],
    )
    result.update(
        {
            "dataset_split": "val",
            "heldout_motion_paths": ["heldout/motion_a.npz"],
            "traj_step_field": "rollout_step",
        }
    )
    return result


def test_canonicalize_validation_metrics_exposes_stable_names():
    metrics = canonicalize_eval_metrics(
        {
            "val_mean_episode_return": 10.0,
            "val_early_termination_rate": 0.1,
            "val_err_rpos": 0.08,
            "val_err_racket_pos": 0.04,
            "val_err_racket_rot": 0.10,
            "val_frame_coverage": 0.92,
        }
    )

    assert metrics["mean_episode_return"] == 10.0
    assert metrics["early_termination_rate"] == 0.1
    assert metrics["err_rpos"] == 0.08
    assert metrics["err_racket_pos"] == 0.04
    assert metrics["err_racket_rot"] == 0.10
    assert metrics["completion_rate"] == 0.92


def test_distill_acceptance_passes_good_racket_student():
    teacher = {
        "mean_episode_return": 100.0,
        "completion_rate": 0.98,
        "early_termination_rate": 0.02,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }
    student = {
        "mean_episode_return": 94.0,
        "completion_rate": 0.94,
        "early_termination_rate": 0.04,
        "err_rpos": 0.084,
        "err_racket_pos": 0.043,
        "err_racket_rot": 0.108,
    }

    result = evaluate_distill_acceptance(
        teacher,
        student,
        convergence=_good_convergence(),
        temporal_audit=_good_temporal_audit(),
    )

    assert result["passed"] is True
    assert result["failed"] == []
    assert result["missing"] == []
    assert validate_direct_acceptance_record(result)["passed"] is True

    forged = dict(result)
    forged["values"] = {"return_ratio": 1.0}
    with pytest.raises(ValueError, match="values are incomplete"):
        validate_direct_acceptance_record(forged)


def test_distill_acceptance_fails_closed_on_bad_or_missing_metrics():
    result = evaluate_distill_acceptance(
        {
            "mean_episode_return": 100.0,
            "completion_rate": 1.0,
            "early_termination_rate": 0.01,
            "err_rpos": 0.08,
        },
        {
            "mean_episode_return": 80.0,
            "completion_rate": 0.85,
            "early_termination_rate": 0.20,
        },
    )

    assert result["passed"] is False
    assert "return_ratio" in result["failed"]
    assert "completion_ratio" in result["failed"]
    assert "early_termination_delta" in result["failed"]
    assert "err_rpos" in result["missing"]
    assert "err_racket_pos" in result["missing"]
    assert "err_racket_rot" in result["missing"]


def test_distill_acceptance_fails_closed_when_new_gate_evidence_is_missing():
    metrics = {
        "mean_episode_return": 100.0,
        "completion_rate": 1.0,
        "early_termination_rate": 0.0,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }

    result = evaluate_distill_acceptance(metrics, metrics)

    assert result["passed"] is False
    assert "mse_plateau" in result["failed"]
    assert "temporal_drift" in result["failed"]
    assert "convergence" in result["missing"]
    assert "temporal_audit" in result["missing"]


def test_distill_acceptance_rejects_unproven_convergence_split():
    metrics = {
        "mean_episode_return": 100.0,
        "completion_rate": 1.0,
        "early_termination_rate": 0.0,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }
    convergence = _good_convergence()
    convergence.pop("motion_split")

    result = evaluate_distill_acceptance(
        metrics,
        metrics,
        convergence=convergence,
        temporal_audit=_good_temporal_audit(),
    )

    assert result["passed"] is False
    assert "mse_plateau" in result["failed"]
    assert "convergence.motion_split" in result["missing"]


def test_write_acceptance_outputs_is_machine_readable(tmp_path):
    path = write_acceptance_outputs(
        {
            "teacher": {
                "mean_episode_return": 10.0,
                "completion_rate": 1.0,
                "early_termination_rate": 0.0,
                "err_rpos": 0.1,
                "err_racket_pos": 0.04,
                "err_racket_rot": 0.10,
            },
            "student_bc": {
                "mean_episode_return": 9.5,
                "completion_rate": 0.95,
                "early_termination_rate": 0.02,
                "err_rpos": 0.105,
                "err_racket_pos": 0.042,
                "err_racket_rot": 0.105,
            },
        },
        tmp_path,
        convergence=_good_convergence(),
        temporal_audits={"student_bc": _good_temporal_audit()},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["student_bc"]["passed"] is True


def test_run_eval_metrics_can_force_deterministic_evaluate_all(monkeypatch):
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = list(command)
        output = command[command.index("--metrics_output_json") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "val_mean_episode_return": 10.0,
                    "val_mean_episode_length": 100.0,
                    "val_early_termination_rate": 0.0,
                    "val_frame_coverage": 1.0,
                    "val_err_rpos": 0.08,
                    "val_err_racket_pos": 0.04,
                    "val_err_racket_rot": 0.10,
                },
                handle,
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("musclemimic.distill.eval_student.subprocess.run", fake_run)
    metrics = run_eval_metrics(
        "/checkpoint",
        motion_paths=["heldout/motion"],
        deterministic=True,
        evaluate_all=True,
    )

    assert "--evaluate_all" in captured["command"]
    assert "--metrics_deterministic" in captured["command"]
    assert captured["command"][captured["command"].index("--motion_path") + 1] == "heldout/motion"
    assert metrics["err_racket_rot"] == 0.10


def test_distill_acceptance_rejects_nonfinite_racket_metric():
    teacher = {
        "mean_episode_return": 100.0,
        "completion_rate": 1.0,
        "early_termination_rate": 0.0,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }
    student = teacher | {"err_racket_rot": float("nan")}

    result = evaluate_distill_acceptance(
        teacher,
        student,
        convergence=_good_convergence(),
        temporal_audit=_good_temporal_audit(),
    )

    assert result["passed"] is False
    assert "err_racket_rot" in result["missing"]


def test_mse_plateau_pass_fail_insufficient_and_nonfinite():
    good = _good_convergence()
    falling = evaluate_mse_plateau(
        [
            {"step": step * 10, "action_mse": value}
            for step, value in enumerate((0.20, 0.18, 0.16, 0.14, 0.12))
        ]
    )
    insufficient = evaluate_mse_plateau(good["history"][:4])
    nonfinite = evaluate_mse_plateau(
        good["history"][:-1] + [{"step": 40, "action_mse": float("nan")}]
    )

    assert good["passed"] is True
    assert falling["passed"] is False
    assert falling["checks"]["normalized_abs_slope"] is False
    assert insufficient["passed"] is False
    assert insufficient["checks"]["enough_validation_points"] is False
    assert nonfinite["passed"] is False
    assert "history[4]_invalid_action_mse" in nonfinite["errors"]
    assert evaluate_mse_plateau(None)["passed"] is False


def test_temporal_drift_passes_aligned_actions_and_rejects_name_drift():
    good = _good_temporal_audit()
    horizon = 30
    steps = np.arange(horizon, dtype=np.int32)
    teacher = np.stack([np.sin(steps), np.cos(steps)], axis=-1)
    mismatched = evaluate_temporal_drift(
        student_action=teacher,
        teacher_action=teacher,
        motion_uid=np.zeros(horizon, dtype=np.int64),
        rollout_uid=np.zeros(horizon, dtype=np.int64),
        traj_step=steps,
        actuator_names=["a", "b"],
        checkpoint_actuator_names=["b", "a"],
        thresholds=DistillAcceptanceThresholds(min_temporal_sequences=1),
    )

    assert good["passed"] is True
    assert good["best_lag_steps"] == 0
    assert mismatched["passed"] is False
    assert "checkpoint_actuator_schema_mismatch" in mismatched["errors"]


def test_temporal_drift_detects_three_step_phase_lag():
    rng = np.random.default_rng(17)
    horizon = 80
    teacher = rng.normal(size=(horizon, 3)).astype(np.float32)
    student = np.empty_like(teacher)
    student[:3] = teacher[0]
    student[3:] = teacher[:-3]
    thresholds = DistillAcceptanceThresholds(
        temporal_search_max_lag_steps=4,
        max_abs_temporal_best_lag_steps=1,
        min_temporal_sequences=1,
    )

    result = evaluate_temporal_drift(
        student_action=student,
        teacher_action=teacher,
        motion_uid=np.zeros(horizon, dtype=np.int64),
        rollout_uid=np.zeros(horizon, dtype=np.int64),
        traj_step=np.arange(horizon, dtype=np.int32),
        actuator_names=["a", "b", "c"],
        checkpoint_actuator_names=["a", "b", "c"],
        thresholds=thresholds,
    )

    assert result["passed"] is False
    assert result["best_lag_steps"] == -3
    assert result["checks"]["global_best_lag"] is False
    assert result["checks"]["lag_mse_improvement"] is False


def test_temporal_drift_nonfinite_actions_fail_with_strict_json():
    teacher = np.zeros((20, 1), dtype=np.float32)
    student = teacher.copy()
    student[3, 0] = np.nan
    result = evaluate_temporal_drift(
        student_action=student,
        teacher_action=teacher,
        motion_uid=np.zeros(20, dtype=np.int64),
        rollout_uid=np.zeros(20, dtype=np.int64),
        traj_step=np.arange(20, dtype=np.int32),
        actuator_names=["a"],
        checkpoint_actuator_names=["a"],
        thresholds=DistillAcceptanceThresholds(min_temporal_sequences=1),
    )

    assert result["passed"] is False
    assert "nonfinite_action" in result["errors"]
    json.dumps(result, allow_nan=False)


def test_distill_acceptance_rechecks_nonfinite_temporal_evidence():
    metrics = {
        "mean_episode_return": 100.0,
        "completion_rate": 1.0,
        "early_termination_rate": 0.0,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }
    audit = _good_temporal_audit()
    audit["lag_mse_improvement_fraction"] = float("nan")

    result = evaluate_distill_acceptance(
        metrics,
        metrics,
        convergence=_good_convergence(),
        temporal_audit=audit,
    )

    assert result["passed"] is False
    assert "temporal_drift" in result["failed"]
    assert "temporal_audit.lag_mse_improvement_fraction" in result["missing"]


def test_distill_compare_forces_deterministic_evaluate_all(monkeypatch, tmp_path):
    from fullbody import distill_compare

    calls = []
    temporal_calls = []
    complete_metrics = {
        "mean_episode_return": 10.0,
        "mean_episode_length": 100.0,
        "completion_rate": 1.0,
        "early_termination_rate": 0.0,
        "frame_coverage": 1.0,
        "err_rpos": 0.08,
        "err_racket_pos": 0.04,
        "err_racket_rot": 0.10,
    }

    def fake_eval(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return dict(complete_metrics)

    def fake_temporal(checkpoint, **kwargs):
        temporal_calls.append((checkpoint, kwargs))
        return _good_temporal_audit()

    monkeypatch.setattr(distill_compare, "run_eval_metrics", fake_eval)
    monkeypatch.setattr(
        distill_compare, "run_checkpoint_temporal_audit", fake_temporal
    )
    convergence_path = tmp_path / "convergence.json"
    convergence_path.write_text(json.dumps(_good_convergence()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distill_compare",
            "--teacher_ckpt",
            "/teacher",
            "--student_ckpt",
            "/student",
            "--output_dir",
            str(tmp_path),
            "--dataset_dir",
            "/dataset",
            "--convergence_metrics",
            str(convergence_path),
            "--motion_path",
            "motion/heldout",
        ],
    )

    assert distill_compare.main() == 0
    assert len(calls) == 2
    assert all(call[1]["evaluate_all"] is True for call in calls)
    assert all(call[1]["deterministic"] is True for call in calls)
    assert all(call[1]["motion_paths"] == ["motion/heldout"] for call in calls)
    assert temporal_calls[0][0] == "/student"
    assert temporal_calls[0][1]["dataset_dir"] == "/dataset"
    assert temporal_calls[0][1]["expected_motion_paths"] == ["motion/heldout"]


def test_distill_compare_require_pass_needs_heldout_motion(monkeypatch, tmp_path):
    from fullbody import distill_compare

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distill_compare",
            "--teacher_ckpt",
            "/teacher",
            "--student_ckpt",
            "/student",
            "--output_dir",
            str(tmp_path),
            "--require_pass",
        ],
    )

    with pytest.raises(SystemExit):
        distill_compare.main()
