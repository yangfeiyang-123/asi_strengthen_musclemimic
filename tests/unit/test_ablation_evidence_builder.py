"""Artifact-derived matching tests for the 24-run continuity evidence builder."""

from __future__ import annotations

import copy

import pytest

from analysis.physiology_synergy.build_continuity_ablation_evidence import (
    build_continuity_ablation_evidence,
)
from analysis.physiology_synergy.build_continuity_ablation_report import (
    ABLATION_EVIDENCE_SCHEMA_VERSION,
    ablation_evidence_fingerprint,
    validate_ablation_evidence,
)
from musclemimic.synergy.basis_factor_contract import basis_factor_contract_fingerprint
from tests.unit.continuity_ablation_evidence_fixtures import (
    build_loaded_ablation_runs,
    rebind_loaded_run,
    reseal_basis_manifest,
)


def _run(runs: list[dict], condition: str, seed: int = 0) -> dict:
    return next(item for item in runs if item["condition"] == condition and item["seed"] == seed)


def _reseal_factor_run(run: dict) -> None:
    factor = run["basis_artifact"]["basis_factor_contract"]
    factor["basis_factor_contract_fingerprint"] = basis_factor_contract_fingerprint(factor)
    reseal_basis_manifest(run["basis_artifact"])
    rebind_loaded_run(run)


@pytest.fixture(scope="module")
def loaded_runs(tmp_path_factory):
    return build_loaded_ablation_runs(tmp_path_factory.mktemp("continuity-ablation-evidence"))


def test_builder_derives_all_24_match_contracts_from_artifacts(loaded_runs):
    evidence = build_continuity_ablation_evidence(copy.deepcopy(loaded_runs))

    assert evidence["schema_version"] == ABLATION_EVIDENCE_SCHEMA_VERSION
    assert evidence["artifact_fingerprint"] == ablation_evidence_fingerprint(evidence)
    assert len(evidence["runs"]) == 24
    validate_ablation_evidence(evidence)
    by_key = {(run["condition"], run["seed"]): run for run in evidence["runs"]}
    assert by_key[("A0", 0)]["matched_basis_factor_contract_fingerprint"] is None
    assert (
        by_key[("B0", 0)]["matched_basis_factor_contract_fingerprint"]
        == by_key[("G0", 0)]["matched_basis_factor_contract_fingerprint"]
    )
    assert (
        by_key[("C0", 2)]["matched_nonreward_contract_fingerprint"]
        == by_key[("C1", 2)]["matched_nonreward_contract_fingerprint"]
    )


def test_builder_rejects_hand_entered_core_match_hashes(loaded_runs):
    runs = copy.deepcopy(loaded_runs)
    runs[0]["matched_nonreward_contract_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="does not accept hand-entered"):
        build_continuity_ablation_evidence(runs)


@pytest.mark.parametrize(
    "mutation",
    [
        "normalization",
        "selected_rank",
        "train_validation_split",
        "phase_weighting",
        "ppo_learning_rate",
        "network_hidden_sizes",
        "exploration_rms",
        "coefficient_temperature",
        "residual_alpha",
    ],
)
def test_builder_rejects_every_preregistered_matching_drift(loaded_runs, mutation):
    runs = copy.deepcopy(loaded_runs)
    if mutation in {"normalization", "selected_rank", "train_validation_split", "phase_weighting"}:
        target = _run(runs, "G0")
        factor = target["basis_artifact"]["basis_factor_contract"]
        if mutation == "normalization":
            factor["normalization"] = {"normalization": "channel_max", "scales": [2.0] * 4}
        elif mutation == "selected_rank":
            factor["candidate_ranks"] = [2, 3]
            factor["selected_rank"] = 3
            factor["policy_action_dimension"] = 3
        elif mutation == "train_validation_split":
            factor["train_motion_uids"] = [0, 1]
        else:
            factor["phase_weighting"]["acceleration"] = 3.0
        _reseal_factor_run(target)
    elif mutation == "residual_alpha":
        target = _run(runs, "C1")
        target["resolved_config"]["experiment"]["action_representation"]["residual"]["alpha"] = 0.05
        rebind_loaded_run(target)
    else:
        target = _run(runs, "G0")
        experiment = target["resolved_config"]["experiment"]
        if mutation == "ppo_learning_rate":
            experiment["lr"] = 1.0e-4
        elif mutation == "network_hidden_sizes":
            experiment["network"]["hidden_sizes"] = [256, 256]
        elif mutation == "exploration_rms":
            experiment["action_representation"]["exploration"]["target_initial_excitation_rms"] = 0.12
        else:
            experiment["action_representation"]["coefficient_transform"]["temperature"] = 0.5
        rebind_loaded_run(target)

    with pytest.raises(ValueError, match=r"differ|not matched|coefficient transform"):
        build_continuity_ablation_evidence(runs)


def test_validator_recomputes_match_hash_instead_of_trusting_resealed_evidence(loaded_runs):
    evidence = build_continuity_ablation_evidence(copy.deepcopy(loaded_runs))
    tampered = copy.deepcopy(evidence)
    tampered["runs"][0]["matched_nonreward_contract_fingerprint"] = "0" * 64
    tampered["artifact_fingerprint"] = ablation_evidence_fingerprint(tampered)

    with pytest.raises(ValueError, match="was not derived from resolved config"):
        validate_ablation_evidence(tampered)
