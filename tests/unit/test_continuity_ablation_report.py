"""Target/global reporting and fail-closed evidence tests for continuity ablations."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from analysis.physiology_synergy.build_continuity_ablation_evidence import (
    build_continuity_ablation_evidence,
)
from analysis.physiology_synergy.build_continuity_ablation_report import (
    ABLATION_REPORT_SCHEMA_VERSION,
    ablation_evidence_fingerprint,
    ablation_report_fingerprint,
    basis_factor_match_fingerprint,
    build_continuity_ablation_report,
    reward_pair_contract_fingerprint,
    validate_ablation_evidence,
)
from tests.unit.continuity_ablation_evidence_fixtures import build_loaded_ablation_runs


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    runs = build_loaded_ablation_runs(tmp_path_factory.mktemp("continuity-ablation-report"))
    return build_continuity_ablation_evidence(runs)


def _reseal(payload: dict) -> dict:
    payload["artifact_fingerprint"] = ablation_evidence_fingerprint(payload)
    return payload


def _run(payload: dict, condition: str, seed: int = 0) -> dict:
    return next(run for run in payload["runs"] if run["condition"] == condition and run["seed"] == seed)


def test_all_24_matched_runs_report_target_global_and_effective_penalty(evidence):
    report = build_continuity_ablation_report(copy.deepcopy(evidence))

    assert report["schema_version"] == ABLATION_REPORT_SCHEMA_VERSION
    assert report["report_fingerprint"] == ablation_report_fingerprint(report)
    assert len(report["run_inventory"]) == 24
    assert report["claim_scope"]["overall_better_conditions"] == ["A", "B", "C", "G"]
    assert report["claim_scope"]["target_only_improvement_conditions"] == []
    metrics = next(run["metrics"] for run in report["run_inventory"] if run["condition"] == "A1")
    assert metrics["target_activation_continuity_p95"] != metrics["global_activation_continuity_p95"]
    assert metrics["penalty_continuity_effective_after_total_clip_mean"] < 0.0
    assert 0.0 <= metrics["total_clip_masked_fraction"] <= 1.0
    for pair in report["reward_ablation"].values():
        assert pair["status"] == "all_preregistered_gates_passed"
        assert pair["target_continuity_improvement_passed_all_seeds"] is True
        assert pair["global_continuity_degraded_any_seed"] is False
    assert len(report["graph_nmf_factor"]["standard_vs_graph_without_reward"]) == 3


def test_target_gain_with_tracking_regression_is_labeled_pareto_tradeoff(evidence):
    payload = copy.deepcopy(evidence)
    _run(payload, "A1", 1)["metrics"]["tracking_error"] = 0.060
    report = build_continuity_ablation_report(_reseal(payload))

    assert report["reward_ablation"]["A"]["status"] == "pareto_tradeoff_task_degraded"
    assert "A" in report["claim_scope"]["pareto_tradeoff_conditions"]
    assert "A" not in report["claim_scope"]["overall_better_conditions"]


def test_target_gain_with_global_regression_is_not_claimed_as_overall(evidence):
    payload = copy.deepcopy(evidence)
    target = _run(payload, "B1", 2)["metrics"]
    target["global_activation_continuity_p95"] = 0.25
    target["global_violation_fraction"] = 0.35
    report = build_continuity_ablation_report(_reseal(payload))

    assert report["reward_ablation"]["B"]["status"] == "target_improved_global_degraded"
    assert "B" in report["claim_scope"]["target_only_improvement_conditions"]
    assert "B" not in report["claim_scope"]["overall_better_conditions"]


def test_failed_policy_promotion_cannot_be_reported_as_gain(evidence):
    payload = copy.deepcopy(evidence)
    _run(payload, "C1", 1)["promotion_passed"] = False
    report = build_continuity_ablation_report(_reseal(payload))

    assert report["reward_ablation"]["C"]["status"] == "promotion_gate_not_met"
    assert report["reward_ablation"]["C"]["promotion_passed_all_seeds"] is False


def test_report_rejects_missing_seed_and_resolved_pair_drift(evidence):
    missing = copy.deepcopy(evidence)
    missing["runs"].pop()
    with pytest.raises(ValueError, match="canonical order"):
        validate_ablation_evidence(_reseal(missing))

    unmatched = copy.deepcopy(evidence)
    target = _run(unmatched, "C1", 2)
    target["resolved_config"]["experiment"]["action_representation"]["residual"]["alpha"] = 0.05
    target["matched_nonreward_contract_fingerprint"] = reward_pair_contract_fingerprint(target["resolved_config"])
    target["matched_basis_factor_contract_fingerprint"] = basis_factor_match_fingerprint(
        target["resolved_config"],
        target["basis_factor_contract"],
    )
    target["source_artifacts"]["resolved_config"] = hashlib.sha256(
        json.dumps(
            target["resolved_config"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="non-reward contracts are not matched"):
        validate_ablation_evidence(_reseal(unmatched))


def test_report_rejects_resume_stale_release_and_loss_identity(evidence):
    resumed = copy.deepcopy(evidence)
    resumed["runs"][0]["resumed"] = True
    with pytest.raises(ValueError, match="fresh optimizer"):
        validate_ablation_evidence(_reseal(resumed))

    stale = copy.deepcopy(evidence)
    stale["runs"][0]["metrics"]["frame_coverage"] = 0.5
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_ablation_evidence(stale)

    release = copy.deepcopy(evidence)
    _run(release, "A0")["continuity_release_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="differs from study identity"):
        validate_ablation_evidence(_reseal(release))

    loss = copy.deepcopy(evidence)
    _run(loss, "G1")["continuity_loss_spec_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="differs from study identity"):
        validate_ablation_evidence(_reseal(loss))


def test_report_requires_nonzero_effective_penalty_and_complete_target_global_metrics(evidence):
    zero_penalty = copy.deepcopy(evidence)
    target = _run(zero_penalty, "A1")["metrics"]
    target["penalty_continuity_effective_after_total_clip_mean"] = 0.0
    with pytest.raises(ValueError, match="nonzero negative continuity penalties"):
        validate_ablation_evidence(_reseal(zero_penalty))

    missing = copy.deepcopy(evidence)
    del _run(missing, "B0")["metrics"]["global_activation_continuity_p95"]
    with pytest.raises(ValueError, match="fields differ"):
        validate_ablation_evidence(_reseal(missing))

    zero_target = copy.deepcopy(evidence)
    _run(zero_target, "G0")["metrics"]["target_edge_count"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        validate_ablation_evidence(_reseal(zero_target))
