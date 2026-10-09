"""Preregistered Graph-NMF lambda selection and matched-factor contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.physiology_synergy.select_graph_nmf_lambda import (
    GRAPH_NMF_LAMBDA_CANDIDATE_INVENTORY_SCHEMA_VERSION,
    build_graph_nmf_lambda_selection,
    candidate_inventory_fingerprint,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import continuity_graph_fingerprint
from musclemimic.synergy.basis_artifact import load_synergy_basis, save_synergy_basis
from musclemimic.synergy.basis_factor_contract import (
    build_basis_factor_contract,
    near_zero_mask_contract,
)
from musclemimic.synergy.graph_nmf import (
    bind_graph_regularization_basis_factor,
    load_verified_graph_regularization,
    validate_formal_graph_nmf_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"


def _verified_graph_path(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    payload["chains"][0]["review_status"] = "verified"
    payload["chains"][0]["training_enabled"] = True
    payload["chains"][0]["provenance"] = [
        {
            "kind": "independent_anatomical_and_baseline_distribution_review",
            "reference": "unit-test fixture only",
        }
    ]
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    path = tmp_path / "candidate_graph.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload["graph_fingerprint"]


def _factor(names: tuple[str, ...]) -> dict:
    normalization = {
        "normalization": "none",
        "scales": [1.0] * len(names),
        "kept_indices": list(range(len(names))),
    }
    return build_basis_factor_contract(
        fit_scope="hybrid_global_regional",
        source_dataset_fingerprint="1" * 64,
        train_motion_uids=[1, 2],
        validation_motion_uids=[3],
        primitive_source_manifest_fingerprint="2" * 64,
        signal_kind="muscle_excitation_v2",
        sample_weighting={"kind": "phase_balanced", "application": "sqrt"},
        phase_weighting={"0": 1.0},
        normalization=normalization,
        near_zero_mask=near_zero_mask_contract(
            channel_count=len(names),
            kept_indices=range(len(names)),
            threshold=1e-8,
        ),
        kept_actuator_indices=range(len(names)),
        candidate_ranks=[2],
        selected_rank=2,
        nmf_initialization_seeds=[0, 1],
        max_iter=100,
        tol=1e-6,
        dynamic_coverage_environment_fingerprint="3" * 64,
        dynamic_coverage_rollout_fingerprint="4" * 64,
    )


def _candidate(tmp_path: Path, *, coefficient: float, passed: bool):
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph_path, graph_fingerprint = _verified_graph_path(tmp_path)
    binding = load_verified_graph_regularization(
        taxonomy_path=TAXONOMY_PATH,
        continuity_path=graph_path,
        expected_taxonomy_fingerprint=taxonomy.fingerprint,
        expected_continuity_fingerprint=graph_fingerprint,
        muscle_names=taxonomy.actuator_names,
        lambda_value=coefficient,
    )
    factor = _factor(taxonomy.actuator_names)
    graph = copy.deepcopy(binding.manifest)
    graph["continuity_release_fingerprint"] = "5" * 64
    graph["continuity_loss_spec_fingerprint"] = "6" * 64
    graph = bind_graph_regularization_basis_factor(
        graph,
        factor["basis_factor_contract_fingerprint"],
    )
    metrics = {
        "all_synergy_gates_passed": passed,
        "dynamic_coverage_required": True,
        "dynamic_coverage_passed": passed,
        "heldout_global_vaf": 0.95,
        "heldout_local_vaf_quantile": 0.80,
        "initialization_stability": 0.90,
        "split_half_stability": 0.89,
        "bootstrap_stability": 0.88,
        "cross_trial_stability": 0.87,
        "basis_condition_number": 2.0,
        "effective_rank_fraction": 1.0,
        "graph_roughness": 0.1,
    }
    normalization = factor["normalization"]
    artifact = save_synergy_basis(
        tmp_path / f"lambda_{coefficient}",
        basis=np.ones((len(taxonomy.actuator_names), 2), dtype=np.float64),
        muscle_names=taxonomy.actuator_names,
        manifest={
            "signal_kind": "muscle_excitation_v2",
            "region": "hybrid_global_regional",
            "rank": 2,
            "normalization": normalization,
            "source_dataset_fingerprint": factor["source_dataset_fingerprint"],
            "teacher_checkpoint_fingerprint": "7" * 64,
            "fit_seed": 0,
            "transform": {"kind": "unit_excitation"},
            "split_provenance": {"train": {}, "validation": {}},
            "train_motion_uids": factor["train_motion_uids"],
            "basis_family": "graph_nmf",
            "basis_artifact_role": "graph_lambda_candidate",
            "basis_factor_contract": factor,
            "basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
            "lambda_selection_metrics": metrics,
            "graph_regularization": graph,
        },
    )
    return artifact


def _inventory(candidates) -> dict:
    payload = {
        "schema_version": GRAPH_NMF_LAMBDA_CANDIDATE_INVENTORY_SCHEMA_VERSION,
        "selection_id": "fixture_lambda_selection",
        "preregistered_lambdas": [item[0] for item in candidates],
        "candidates": [
            {
                "lambda": coefficient,
                "basis_artifact_path": str(artifact.path),
                "expected_basis_artifact_fingerprint": artifact.fingerprint,
            }
            for coefficient, artifact in candidates
        ],
    }
    payload["inventory_fingerprint"] = candidate_inventory_fingerprint(payload)
    return payload


def test_selector_chooses_smallest_positive_lambda_passing_every_non_ppo_gate(tmp_path):
    low = _candidate(tmp_path / "low", coefficient=0.05, passed=False)
    middle = _candidate(tmp_path / "middle", coefficient=0.10, passed=True)
    high = _candidate(tmp_path / "high", coefficient=0.25, passed=True)
    selection = build_graph_nmf_lambda_selection(
        _inventory([(0.05, low), (0.10, middle), (0.25, high)]),
        created_at_utc="2026-07-30T00:00:00Z",
    )

    assert selection["eligible_lambdas"] == [0.10, 0.25]
    assert selection["selected_lambda"] == 0.10
    assert selection["selected_candidate_basis_fingerprint"] == middle.fingerprint

    production_graph = copy.deepcopy(middle.manifest["graph_regularization"])
    production_graph["lambda_selection_fingerprint"] = selection["selection_fingerprint"]
    assert validate_formal_graph_nmf_manifest(production_graph)["requested_lambda"] == 0.10


def test_selector_rejects_unmatched_factor_contract_and_any_ppo_result_field(tmp_path):
    first = _candidate(tmp_path / "first", coefficient=0.10, passed=True)
    second = _candidate(tmp_path / "second", coefficient=0.20, passed=True)
    second_payload = copy.deepcopy(second.manifest)
    factor = copy.deepcopy(second_payload["basis_factor_contract"])
    factor["validation_motion_uids"] = [4]
    factor.pop("basis_factor_contract_fingerprint")
    factor = build_basis_factor_contract(
        **{
            "fit_scope": factor["fit_scope"],
            "source_dataset_fingerprint": factor["source_dataset_fingerprint"],
            "train_motion_uids": factor["train_motion_uids"],
            "validation_motion_uids": factor["validation_motion_uids"],
            "primitive_source_manifest_fingerprint": factor["primitive_source_manifest_fingerprint"],
            "signal_kind": factor["signal_kind"],
            "sample_weighting": factor["sample_weighting"],
            "phase_weighting": factor["phase_weighting"],
            "normalization": factor["normalization"],
            "near_zero_mask": factor["near_zero_mask"],
            "kept_actuator_indices": factor["kept_actuator_indices"],
            "candidate_ranks": factor["candidate_ranks"],
            "selected_rank": factor["selected_rank"],
            "nmf_initialization_seeds": factor["nmf_initialization_seeds"],
            "max_iter": factor["max_iter"],
            "tol": factor["tol"],
            "dynamic_coverage_environment_fingerprint": factor["dynamic_coverage_environment_fingerprint"],
            "dynamic_coverage_rollout_fingerprint": factor["dynamic_coverage_rollout_fingerprint"],
            "coefficient_transform_schema": factor["coefficient_transform_schema"],
            "policy_action_dimension": factor["policy_action_dimension"],
        }
    )
    graph = copy.deepcopy(second_payload["graph_regularization"])
    graph = bind_graph_regularization_basis_factor(
        graph,
        factor["basis_factor_contract_fingerprint"],
    )
    drifted = save_synergy_basis(
        tmp_path / "drifted",
        basis=second.basis,
        muscle_names=second.muscle_names,
        manifest={
            **{
                key: value
                for key, value in second_payload.items()
                if key
                not in {
                    "schema_version",
                    "basis_file",
                    "basis_sha256",
                    "basis_shape",
                    "muscle_names",
                    "muscle_schema_sha256",
                    "artifact_fingerprint",
                }
            },
            "basis_factor_contract": factor,
            "basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
            "graph_regularization": graph,
        },
    )
    with pytest.raises(ValueError, match="not matched"):
        build_graph_nmf_lambda_selection(_inventory([(0.10, first), (0.20, drifted)]))

    forged = _inventory([(0.10, first), (0.20, second)])
    forged["candidates"][0]["ppo_final_return"] = 999.0
    forged["inventory_fingerprint"] = candidate_inventory_fingerprint(forged)
    with pytest.raises(ValueError, match="descriptor fields"):
        build_graph_nmf_lambda_selection(forged)


def test_candidate_basis_artifact_is_not_trainable_before_selection(tmp_path):
    candidate = _candidate(tmp_path, coefficient=0.10, passed=True)
    loaded = load_synergy_basis(candidate.path)
    with pytest.raises(ValueError, match="lambda_selection_fingerprint"):
        validate_formal_graph_nmf_manifest(loaded.manifest["graph_regularization"])
