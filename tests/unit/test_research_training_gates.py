import inspect
import json
import sys

import pytest

from fullbody.run_forehand_clear_pipeline import _require_passed_report
from musclemimic.badminton.training_gates import evaluate_promotion, main


def test_stage3_v2_gate_is_fail_closed_and_accepts_complete_metrics():
    metrics = {
        "evaluated_feed_count": 128,
        "no_fall_rate": 0.99,
        "hit_rate": 0.95,
        "impact_position_error_m": 0.08,
        "center_hit_rate": 0.8,
        "impact_timing_mae_s": 0.05,
        "stringbed_normal_error_rad": 0.2,
        "racket_linear_velocity_rmse_m_s": 1.0,
        "racket_angular_velocity_rmse_rad_s": 5.0,
        "landing_rmse_m": 0.6,
        "apex_mae_m": 0.3,
        "recovery_ready_rate": 0.9,
        "control_finite": 1.0,
        "artifact_binding_verified": 1.0,
    }
    assert evaluate_promotion("stage3_v2", metrics).passed
    metrics.pop("apex_mae_m")
    assert not evaluate_promotion("stage3_v2", metrics).passed


def test_latent_synergy_gate_rejects_residual_bypass():
    metrics = {
        "heldout_sample_count": 1000,
        "reconstruction_nrmse": 0.10,
        "closed_loop_success_rate": 0.95,
        "residual_energy_ratio": 0.08,
        "residual_energy_ratio_ready": 0.03,
        "residual_energy_ratio_recovery": 0.03,
        "residual_bypass_gate_passed": 1.0,
        "latent_dimension_selected": 1.0,
        "checkpoint_binding_verified": 1.0,
        "causal_rollout_required": True,
        "causal_rollout_verified": 1.0,
        "stage2_diagnostic_outcomes_complete": 1.0,
        "full_matrix_complete": 1.0,
    }
    assert evaluate_promotion("latent_synergy_v2", metrics).passed
    metrics["residual_energy_ratio_ready"] = 0.051
    metrics["residual_bypass_gate_passed"] = 0.0
    report = evaluate_promotion("latent_synergy_v2", metrics)
    assert not report.passed
    failed = {check.name for checks in report.evaluations for check in checks if not check.passed}
    assert failed >= {
        "residual_energy_ratio_ready",
        "residual_bypass_gate_passed",
    }
    metrics["residual_energy_ratio_ready"] = 0.03
    metrics["residual_bypass_gate_passed"] = 1.0
    metrics.pop("causal_rollout_verified")
    assert not evaluate_promotion("latent_synergy_v2", metrics).passed


def test_final_task_causal_gate_requires_full_task_outcomes():
    metrics = {
        "task_causal_complete": 1.0,
        "exact_snapshot_restore": 1.0,
        "common_random_numbers": 1.0,
        "full_intervention_matrix_complete": 1.0,
        "all_task_outcomes_available": 1.0,
        "task_outcomes_complete": 1.0,
        "masked_impact_schema_verified": 1.0,
        "masked_landing_schema_verified": 1.0,
        "missing_event_sentinel_contract_verified": 1.0,
        "masked_event_effects_verified": 1.0,
        "masked_task_values_excluded_from_generic_effects": 1.0,
        "pre_hit_snapshot_verified": 1.0,
        "complete_task_horizon_verified": 1.0,
        "paired_comparison_binding_verified": 1.0,
        "stage3_c7_checkpoint_verified": 1.0,
        "two_branches_complete": 1.0,
        "paired_feed_step_protocol_verified": 1.0,
        "paired_epsilon_protocol_verified": 1.0,
        "symmetric_epsilon_pairs_verified": 1.0,
        "cross_branch_crn_protocol_verified": 1.0,
        "paired_horizon_protocol_verified": 1.0,
        "direct_natural_alignment_branch_complete": 1.0,
        "synergy_constrained_branch_complete": 1.0,
    }
    assert evaluate_promotion("latent_task_causal_v1", metrics).passed
    for required_field in tuple(metrics):
        incomplete = dict(metrics)
        incomplete.pop(required_field)
        assert not evaluate_promotion("latent_task_causal_v1", incomplete).passed


def test_final_task_causal_cli_revalidates_bound_branch_artifacts():
    source = inspect.getsource(main)
    assert '"latent_task_causal_v2"' in source
    assert "validate_task_causal_promotion(args.metrics)" in source


def test_selected_synergy_task_causal_v2_marks_full354_not_applicable():
    metrics = {
        "task_causal_complete": 1.0,
        "fixed_synergy_branch_complete": 1.0,
        "full354_latent_intervention_not_applicable": 1.0,
    }
    assert evaluate_promotion("latent_task_causal_v2", metrics).passed
    for required_field in tuple(metrics):
        incomplete = dict(metrics)
        incomplete.pop(required_field)
        assert not evaluate_promotion("latent_task_causal_v2", incomplete).passed


def test_v3_gate_report_is_bound_to_exact_metrics_content(tmp_path, monkeypatch):
    metrics = tmp_path / "event.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": "event_reference_promotion_metrics_v1",
                "reference_count": 5,
                "event_valid_rate": 1.0,
                "impact_position_uncertainty_m": 0.01,
                "impact_timing_uncertainty_s": 0.01,
                "racket_state_finite_rate": 1.0,
                "artifact_binding_verified": 1.0,
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_gates",
            "--stage",
            "event_reference_v2",
            "--metrics",
            str(metrics),
            "--output",
            str(gate),
            "--require_pass",
        ],
    )
    assert main() == 0
    _require_passed_report(
        gate,
        label="event gate",
        expected_metrics=metrics,
    )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["reference_count"] = 6
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after gate"):
        _require_passed_report(gate, label="event gate", expected_metrics=metrics)
    handwritten = tmp_path / "handwritten.json"
    handwritten.write_text('{"passed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="no source-bound"):
        _require_passed_report(
            handwritten,
            label="event gate",
            expected_metrics=metrics,
        )
