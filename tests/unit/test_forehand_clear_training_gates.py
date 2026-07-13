from __future__ import annotations

import json
import sys

import pytest

from musclemimic.badminton import training_gates
from musclemimic.badminton.training_gates import evaluate_promotion


def _stage1_metrics(error: float = 0.08):
    return {
        "val_early_termination_rate": 0.04,
        "val_frame_coverage": 0.96,
        "val_err_rpos": error,
        "val_action_saturation_fraction": 0.04,
        "val_activation_energy": 0.30,
    }


def test_stage1_requires_three_consecutive_complete_validations():
    assert evaluate_promotion("stage1", [_stage1_metrics()] * 2).passed is False
    assert evaluate_promotion("stage1", [_stage1_metrics()] * 3).passed is True
    assert evaluate_promotion("stage1", [_stage1_metrics(), _stage1_metrics(0.10), _stage1_metrics()]).passed is False


def test_stage1_rejects_sustained_action_or_activation_saturation():
    high_action = _stage1_metrics() | {"val_action_saturation_fraction": 0.051}
    high_energy = _stage1_metrics() | {"val_activation_energy": 0.351}

    assert evaluate_promotion("stage1", [high_action] * 3).passed is False
    assert evaluate_promotion("stage1", [high_energy] * 3).passed is False


def test_promotion_fails_closed_when_metric_is_missing_or_nonfinite():
    report = evaluate_promotion("stage1", [{"val_early_termination_rate": 0.0}] * 3)
    assert report.passed is False
    assert any(check.value is None for check in report.evaluations[-1])


def test_promotion_does_not_hide_nonfinite_primary_alias_with_fallback():
    metrics = _stage1_metrics() | {
        "val_err_rpos": float("nan"),
        "err_rpos": 0.01,
    }

    report = evaluate_promotion("stage1", [metrics] * 3)

    assert report.passed is False
    rpos = next(
        check
        for check in report.evaluations[-1]
        if check.name == "relative_site_position_error_m"
    )
    assert rpos.value is None


def test_stage1r_reads_paired_report_and_requires_verified_seed_hash():
    report = {
        "pair_count": 5,
        "seed_hash": "abc123",
        "metrics": {
            "body_site_error": {"relative_degradation": 0.04},
            "right_hand_site_error": {"relative_degradation": 0.03},
            "racket_head_position_error": {"relative_degradation": 0.03},
            "racket_head_rotation_error": {"relative_degradation": 0.03},
            "early_termination": {"absolute_degradation": 0.01},
        },
        "new_root_hand_racket_spike_count": 0,
    }

    assert evaluate_promotion("stage1r", report).passed is True
    report["seed_hash"] = None
    assert evaluate_promotion("stage1r", report).passed is False


def test_stage1r_cli_rejects_legacy_offline_report(tmp_path, monkeypatch):
    report_path = tmp_path / "legacy-stage1r.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "stage1r_paired_robustness_v2",
                "evidence_kind": "legacy_offline_v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_gates",
            "--stage",
            "stage1r",
            "--metrics",
            str(report_path),
            "--checkpoint",
            "/does/not/exist",
            "--finger-perturb-qpos-scale",
            "0.03",
        ],
    )

    with pytest.raises(ValueError, match="legacy/offline"):
        training_gates.main()


def test_stage3_derives_no_fall_rate_and_applies_all_metrics_gates():
    metrics = {
        "evaluated_feed_count": 128,
        "fall_rate": 0.04,
        "hit_rate": 0.91,
        "crossed_net_rate": 0.86,
        "opponent_back_landing_rate": 0.71,
        "racket_head_speed_m_s": 8.1,
        "net_clearance_m": 0.26,
        "control_finite": 1.0,
        "min_root_height_m": 0.8,
        "body_action_saturation_fraction": 0.0,
        "full_action_saturation_fraction": 0.0,
        "normalized_control_energy": 0.1,
        "raw_latent_saturation": 0.0,
        "lab_state_ood_fraction_p95": 0.0,
        "max_attachment_translation_drift_m": 0.0,
        "max_attachment_rotation_drift_rad": 0.0,
        "body_relative_deviation_to_prior": 0.10,
        "right_hand_site_rmse_to_prior_m": 0.05,
        "right_hand_site_relative_deviation_to_prior": 0.10,
        "racket_position_rmse_to_prior_m": 0.05,
        "racket_position_relative_deviation_to_prior": 0.10,
        "racket_rotation_rmse_to_prior_rad": 0.10,
        "racket_rotation_relative_deviation_to_prior": 0.10,
        "prior_vs_direct_body_racket_relative_degradation": 0.05,
        "stage3_vs_direct_naturalness_upper_bound": 0.155,
        "artifact_binding_verified": 1.0,
    }
    assert evaluate_promotion("stage3", metrics).passed is True
    metrics["hit_rate"] = 0.89
    assert evaluate_promotion("stage3", metrics).passed is False


def test_stage3_accepts_real_evaluate_report_aliases_and_requires_128_feeds():
    metrics = {
        "evaluated_feed_count": 128,
        "no_fall_rate": 0.96,
        "hit_rate": 0.91,
        "crossed_net_rate": 0.86,
        "opponent_back_landing_rate": 0.71,
        "mean_racket_head_speed_m_s": 8.1,
        "mean_net_clearance_m": 0.26,
        "control_finite": 1.0,
        "min_root_height_m": 0.8,
        "body_action_saturation_fraction": 0.0,
        "full_action_saturation_fraction": 0.0,
        "normalized_control_energy": 0.1,
        "raw_latent_saturation": 0.0,
        "lab_state_ood_fraction_p95": 0.0,
        "max_attachment_translation_drift_m": 0.0,
        "max_attachment_rotation_drift_rad": 0.0,
        "body_relative_deviation_to_prior": 0.10,
        "right_hand_site_rmse_to_prior_m": 0.05,
        "right_hand_site_relative_deviation_to_prior": 0.10,
        "racket_position_rmse_to_prior_m": 0.05,
        "racket_position_relative_deviation_to_prior": 0.10,
        "racket_rotation_rmse_to_prior_rad": 0.10,
        "racket_rotation_relative_deviation_to_prior": 0.10,
        "prior_vs_direct_body_racket_relative_degradation": 0.05,
        "stage3_vs_direct_naturalness_upper_bound": 0.155,
        "artifact_binding_verified": 1.0,
    }

    assert evaluate_promotion("stage3", metrics).passed is True
    metrics["evaluated_feed_count"] = 127
    assert evaluate_promotion("stage3", metrics).passed is False


def test_stage2_uses_actual_racket_aliases_and_derives_body_degradation():
    baseline = {
        "val_err_rpos": 0.080,
        "val_err_joint_pos": 0.100,
        "val_err_joint_vel": 0.200,
        "val_err_site_abs": 0.120,
    }
    current = {
        "val_early_termination_rate": 0.04,
        "val_frame_coverage": 0.96,
        "val_err_racket_pos": 0.045,
        "val_err_racket_rot": 0.18,
        "val_err_rpos": 0.084,
        "val_err_joint_pos": 0.105,
        "val_err_joint_vel": 0.210,
        "val_err_site_abs": 0.126,
    }

    assert evaluate_promotion(
        "stage2", [current] * 3, baseline_metrics=baseline
    ).passed is True
    current["val_err_rpos"] = 0.10
    assert evaluate_promotion(
        "stage2", [current] * 3, baseline_metrics=baseline
    ).passed is False


def test_stage2_fails_closed_without_body_baseline():
    current = {
        "val_early_termination_rate": 0.04,
        "val_frame_coverage": 0.96,
        "val_err_racket_pos": 0.045,
        "val_err_racket_rot": 0.18,
        "val_err_rpos": 0.084,
    }
    assert evaluate_promotion("stage2", [current] * 3).passed is False


def test_stage2_fails_closed_when_reported_body_metric_is_missing_on_one_side():
    baseline = {
        "val_err_rpos": 0.080,
        "val_err_joint_pos": 0.100,
        "val_err_joint_vel": 0.200,
    }
    current = {
        "val_early_termination_rate": 0.04,
        "val_frame_coverage": 0.96,
        "val_err_racket_pos": 0.045,
        "val_err_racket_rot": 0.18,
        "val_err_rpos": 0.084,
        "val_err_joint_pos": 0.105,
    }

    assert evaluate_promotion(
        "stage2", [current] * 3, baseline_metrics=baseline
    ).passed is False


def test_cli_uses_latest_stage1_validation_as_stage2_baseline(monkeypatch, tmp_path):
    baseline_path = tmp_path / "stage1.json"
    metrics_path = tmp_path / "stage2.json"
    output_path = tmp_path / "gate.json"
    baseline_path.write_text(
        json.dumps(
            {
                "validations": [
                    {"val_err_rpos": 0.20},
                    {"val_err_rpos": 0.080},
                ]
            }
        ),
        encoding="utf-8",
    )
    current = {
        "val_early_termination_rate": 0.04,
        "val_frame_coverage": 0.96,
        "val_err_racket_pos": 0.045,
        "val_err_racket_rot": 0.18,
        "val_err_rpos": 0.084,
    }
    metrics_path.write_text(
        json.dumps({"validations": [current, current, current]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forehand-clear-promotion-gate",
            "--stage",
            "stage2",
            "--metrics",
            str(metrics_path),
            "--baseline-metrics",
            str(baseline_path),
            "--output",
            str(output_path),
            "--require_pass",
        ],
    )

    assert training_gates.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True


def test_online_promotion_history_is_a_first_class_metrics_artifact(monkeypatch, tmp_path):
    metrics_path = tmp_path / "promotion_progress.json"
    output_path = tmp_path / "gate.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "forehand_clear_promotion_progress_v1",
                "history": [
                    {"update_number": index, "metrics": _stage1_metrics()}
                    for index in range(1, 4)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forehand-clear-promotion-gate",
            "--stage",
            "stage1",
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
            "--require_pass",
        ],
    )

    assert training_gates.main() == 0
    assert json.loads(output_path.read_text())["passed"] is True
    assert training_gates.latest_validation_record(
        json.loads(metrics_path.read_text())
    ) == _stage1_metrics()
