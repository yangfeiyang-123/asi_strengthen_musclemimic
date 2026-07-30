"""Matched-seed success and claim-scope tests for the continuity report."""

from __future__ import annotations

import copy
import hashlib

import pytest

from analysis.physiology_synergy.build_continuity_ablation_report import (
    ABLATION_EVIDENCE_SCHEMA_VERSION,
    ABLATION_REPORT_SCHEMA_VERSION,
    CONDITIONS,
    SEEDS,
    ablation_evidence_fingerprint,
    ablation_report_fingerprint,
    build_continuity_ablation_report,
    validate_ablation_evidence,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metrics(*, reward: bool) -> dict[str, float]:
    if not reward:
        return {
            "early_termination_rate": 0.020,
            "frame_coverage": 0.970,
            "tracking_error": 0.050,
            "relative_site_error": 0.050,
            "right_hand_error": 0.030,
            "action_decoder_saturation_fraction": 0.020,
            "action_preclip_out_of_bounds_fraction": 0.010,
            "action_clip_correction_rms": 0.005,
            "activation_energy": 0.100,
            "residual_energy_fraction": 0.0,
            "synergy_coefficient_effective_dimension": 8.0,
            "activation_continuity_mean": 0.100,
            "activation_continuity_p95": 0.200,
            "activation_continuity_max": 0.400,
            "excitation_continuity_p95": 0.180,
            "continuity_violation_fraction": 0.300,
            "continuity_active_chain_fraction": 0.800,
            "continuity_measured_chain_count": 28,
            "continuity_measured_edge_count": 140,
            "steps_per_second": 100.0,
            "compile_time_seconds": 40.0,
            "gpu_memory_gb": 12.0,
            "update_wall_time_seconds": 1.0,
        }
    return {
        "early_termination_rate": 0.025,
        "frame_coverage": 0.965,
        "tracking_error": 0.051,
        "relative_site_error": 0.051,
        "right_hand_error": 0.031,
        "action_decoder_saturation_fraction": 0.025,
        "action_preclip_out_of_bounds_fraction": 0.012,
        "action_clip_correction_rms": 0.006,
        "activation_energy": 0.098,
        "residual_energy_fraction": 0.0,
        "synergy_coefficient_effective_dimension": 8.0,
        "activation_continuity_mean": 0.080,
        "activation_continuity_p95": 0.160,
        "activation_continuity_max": 0.320,
        "excitation_continuity_p95": 0.160,
        "continuity_violation_fraction": 0.240,
        "continuity_active_chain_fraction": 0.800,
        "continuity_measured_chain_count": 28,
        "continuity_measured_edge_count": 140,
        "steps_per_second": 97.0,
        "compile_time_seconds": 41.0,
        "gpu_memory_gb": 12.2,
        "update_wall_time_seconds": 1.02,
    }


def _run(condition: str, seed: int) -> dict:
    pair = condition[0]
    reward = condition.endswith("1")
    if pair == "A":
        mode, family = "full_354", "direct_354"
        basis = residual = graph = None
    elif pair == "B":
        mode, family = "fixed_synergy", "standard_nmf"
        basis, residual, graph = "b" * 64, None, None
    elif pair == "C":
        mode, family = "fixed_synergy_residual", "standard_nmf_structured_residual"
        basis, residual, graph = "c" * 64, "d" * 64, None
    else:
        mode, family = "fixed_synergy", "graph_nmf"
        basis, residual, graph = "e" * 64, None, "f" * 64
    run_id = f"forehand_clear_continuity_ablation_v1_{condition.lower()}_s{seed}"
    return {
        "condition": condition,
        "seed": seed,
        "run_id": run_id,
        "config_hash": f"config-{condition}-{seed}",
        "checkpoint_fingerprint": _sha(f"checkpoint:{run_id}"),
        "promotion_fingerprint": _sha(f"promotion:{run_id}"),
        "promotion_passed": True,
        "matched_nonreward_contract_fingerprint": _sha(f"matched:{pair}:{seed}"),
        "matched_basis_factor_contract_fingerprint": _sha(
            f"basis-factor:{reward}:{seed}" if pair in {"B", "G"} else f"basis-factor:{condition}:{seed}"
        ),
        "action_mode": mode,
        "basis_family": family,
        "continuity_reward_enabled": reward,
        "graph_regularized_basis": pair == "G",
        "fresh_optimizer": True,
        "resumed": False,
        "total_timesteps": 320_000_000,
        "basis_fingerprint": basis,
        "residual_basis_fingerprint": residual,
        "graph_regularization_lineage_fingerprint": graph,
        "continuity_reward_coefficient": 0.123 if reward else 0.0,
        "continuity_calibration_fingerprint": "9" * 64,
        "continuity_graph_fingerprint": "8" * 64,
        "joint_report_fingerprint": _sha(f"joint:{run_id}"),
        "metrics": _metrics(reward=reward),
    }


def _evidence() -> dict:
    payload = {
        "schema_version": ABLATION_EVIDENCE_SCHEMA_VERSION,
        "study_identity": {
            "branch_commit_sha": "1" * 40,
            "dataset_split_fingerprint": "2" * 64,
            "environment_fingerprint": "3" * 64,
            "validation_motion_fingerprint": "4" * 64,
            "promotion_contract_fingerprint": "5" * 64,
            "racket_curriculum_fingerprint": "6" * 64,
            "taxonomy_fingerprint": "7" * 64,
            "continuity_graph_fingerprint": "8" * 64,
            "calibration_fingerprint": "9" * 64,
            "calibrated_reward_coefficient": 0.123,
            "total_timesteps": 320_000_000,
        },
        "runs": [_run(condition, seed) for condition in CONDITIONS for seed in SEEDS],
    }
    payload["artifact_fingerprint"] = ablation_evidence_fingerprint(payload)
    return payload


def _reseal(payload: dict) -> dict:
    payload["artifact_fingerprint"] = ablation_evidence_fingerprint(payload)
    return payload


def test_all_24_matched_runs_pass_preregistered_task_physiology_and_overhead_gates():
    report = build_continuity_ablation_report(_evidence())

    assert report["schema_version"] == ABLATION_REPORT_SCHEMA_VERSION
    assert report["report_fingerprint"] == ablation_report_fingerprint(report)
    assert len(report["run_inventory"]) == 24
    assert set(report["run_inventory"][0]["metrics"]) == set(_metrics(reward=False))
    assert report["claim_scope"]["overall_better_conditions"] == ["A", "B", "C", "G"]
    assert report["claim_scope"]["pareto_tradeoff_conditions"] == []
    assert report["claim_scope"]["neural_synergy_claim"] is False
    for pair in report["reward_ablation"].values():
        assert pair["status"] == "all_preregistered_gates_passed"
        assert len(pair["seed_comparisons"]) == 3
    assert len(report["graph_nmf_factor"]["standard_vs_graph_without_reward"]) == 3
    assert len(report["graph_nmf_factor"]["standard_vs_graph_with_reward"]) == 3


def test_continuity_gain_with_tracking_regression_is_labeled_pareto_tradeoff():
    evidence = _evidence()
    target = next(run for run in evidence["runs"] if run["condition"] == "A1" and run["seed"] == 1)
    target["metrics"]["tracking_error"] = 0.060
    report = build_continuity_ablation_report(_reseal(evidence))

    assert report["reward_ablation"]["A"]["status"] == "pareto_tradeoff_task_degraded"
    assert "A" in report["claim_scope"]["pareto_tradeoff_conditions"]
    assert "A" not in report["claim_scope"]["overall_better_conditions"]


def test_failed_policy_promotion_cannot_be_reported_as_an_overall_gain():
    evidence = _evidence()
    target = next(run for run in evidence["runs"] if run["condition"] == "B1" and run["seed"] == 1)
    target["promotion_passed"] = False
    report = build_continuity_ablation_report(_reseal(evidence))

    assert report["reward_ablation"]["B"]["status"] == "promotion_gate_not_met"
    assert report["reward_ablation"]["B"]["promotion_passed_all_seeds"] is False
    assert "B" not in report["claim_scope"]["overall_better_conditions"]


def test_report_rejects_missing_seed_and_unmatched_reward_pair():
    missing = _evidence()
    missing["runs"].pop()
    with pytest.raises(ValueError, match="canonical order"):
        validate_ablation_evidence(_reseal(missing))

    unmatched = _evidence()
    target = next(run for run in unmatched["runs"] if run["condition"] == "C1" and run["seed"] == 2)
    target["matched_nonreward_contract_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="non-reward contracts are not matched"):
        validate_ablation_evidence(_reseal(unmatched))


def test_report_rejects_resume_or_stale_evidence_fingerprint():
    resumed = _evidence()
    resumed["runs"][0]["resumed"] = True
    resumed = _reseal(resumed)
    with pytest.raises(ValueError, match="fresh optimizer"):
        validate_ablation_evidence(resumed)

    stale = copy.deepcopy(_evidence())
    stale["runs"][0]["metrics"]["frame_coverage"] = 0.5
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_ablation_evidence(stale)


def test_report_rejects_coefficient_drift_and_unmatched_graph_basis_factor():
    coefficient = _evidence()
    target = next(run for run in coefficient["runs"] if run["condition"] == "B1" and run["seed"] == 0)
    target["continuity_reward_coefficient"] = 0.5
    with pytest.raises(ValueError, match="coefficient differs"):
        validate_ablation_evidence(_reseal(coefficient))

    factor = _evidence()
    target = next(run for run in factor["runs"] if run["condition"] == "G0" and run["seed"] == 2)
    target["matched_basis_factor_contract_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="differ outside the basis factor"):
        validate_ablation_evidence(_reseal(factor))


def test_report_requires_nonzero_continuity_coverage_and_full_metric_binding():
    evidence = _evidence()
    target = evidence["runs"][0]
    target["metrics"]["continuity_measured_edge_count"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        validate_ablation_evidence(_reseal(evidence))

    missing = _evidence()
    del missing["runs"][0]["joint_report_fingerprint"]
    with pytest.raises(ValueError, match="fields differ"):
        validate_ablation_evidence(_reseal(missing))
