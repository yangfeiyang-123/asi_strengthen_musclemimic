"""Build a strict paired-seed report for continuity and Graph-NMF ablations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.synergy.basis_factor_contract import validate_basis_factor_contract

ABLATION_EVIDENCE_SCHEMA_VERSION = "forehand_continuity_ablation_evidence_v3"
ABLATION_REPORT_SCHEMA_VERSION = "forehand_continuity_ablation_report_v3"
CONDITIONS = ("A0", "A1", "B0", "B1", "C0", "C1", "G0", "G1")
SEEDS = (0, 1, 2)
THRESHOLDS = {
    "max_early_termination_absolute_increase": 0.02,
    "max_frame_coverage_absolute_degradation": 0.02,
    "max_tracking_error_relative_degradation": 0.05,
    "max_saturation_absolute_increase": 0.01,
    "min_target_continuity_p95_relative_improvement": 0.10,
    "min_target_violation_relative_improvement": 0.10,
    "max_steps_per_second_relative_overhead": 0.05,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_EXPECTED_CONDITION = {
    "A0": ("A", "full_354", "direct_354", False, False),
    "A1": ("A", "full_354", "direct_354", True, False),
    "B0": ("B", "fixed_synergy", "standard_nmf", False, False),
    "B1": ("B", "fixed_synergy", "standard_nmf", True, False),
    "C0": ("C", "fixed_synergy_residual", "standard_nmf_structured_residual", False, False),
    "C1": ("C", "fixed_synergy_residual", "standard_nmf_structured_residual", True, False),
    "G0": ("G", "fixed_synergy", "graph_nmf", False, True),
    "G1": ("G", "fixed_synergy", "graph_nmf", True, True),
}
_SOURCE_ARTIFACT_FIELDS = {
    "resolved_config",
    "run_manifest",
    "checkpoint_metadata",
    "promotion",
    "validation_metrics",
    "continuity_release",
    "basis_artifact",
    "residual_artifact",
    "graph_lineage",
    "joint_report",
    "runtime_environment",
    "performance_profiler",
}
_METRIC_FIELDS = {
    "early_termination_rate",
    "frame_coverage",
    "tracking_error",
    "relative_site_error",
    "right_hand_error",
    "action_decoder_saturation_fraction",
    "action_preclip_out_of_bounds_fraction",
    "action_clip_correction_rms",
    "activation_energy",
    "residual_energy_fraction",
    "synergy_coefficient_effective_dimension",
    "target_activation_continuity_mean",
    "target_activation_continuity_p95",
    "target_activation_continuity_max",
    "target_violation_fraction",
    "target_active_chain_fraction",
    "target_chain_count",
    "target_edge_count",
    "global_activation_continuity_mean",
    "global_activation_continuity_p95",
    "global_activation_continuity_max",
    "global_violation_fraction",
    "global_active_chain_fraction",
    "global_chain_count",
    "global_edge_count",
    "penalty_continuity_raw_mean",
    "penalty_continuity_after_local_clip_mean",
    "penalty_continuity_effective_after_total_clip_mean",
    "total_clip_masked_fraction",
    "steps_per_second",
    "compile_time_seconds",
    "gpu_memory_gb",
    "update_wall_time_seconds",
}


def ablation_evidence_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("artifact_fingerprint", None)
    return _json_sha256(unsigned)


def ablation_report_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("report_fingerprint", None)
    return _json_sha256(unsigned)


def derive_reward_pair_contract(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete resolved training config minus preregistered reward-only fields."""

    contract = copy.deepcopy(_mapping(resolved_config, "resolved config"))
    contract.pop("hydra", None)
    _drop_path(contract, ("wandb", "name"))
    _drop_path(contract, ("wandb", "tags"))
    experiment = _mapping(contract.get("experiment"), "resolved config experiment")
    contract["experiment"] = experiment
    for key in (
        "run_id",
        "checkpoint_dir",
        "validation_video_dir",
        "resume_from",
        "seeds",
        "seed",
        "continuity_training_contract",
    ):
        experiment.pop(key, None)
    validation = experiment.get("validation")
    if isinstance(validation, Mapping):
        validation = copy.deepcopy(dict(validation))
        validation.pop("eval_seed", None)
        experiment["validation"] = validation
    ablation = experiment.get("continuity_ablation")
    if isinstance(ablation, Mapping):
        ablation = copy.deepcopy(dict(ablation))
        for key in ("condition", "matched_pair", "seed", "continuity_reward_enabled"):
            ablation.pop(key, None)
        experiment["continuity_ablation"] = ablation
    continuity_path = (
        "env_params",
        "reward_params",
        "intra_muscle_consistency",
    )
    continuity = _get_path(experiment, continuity_path)
    if isinstance(continuity, Mapping):
        # Different Hydra overlays may spell absent release-only fields as null.
        # Preserve every numerical/method field that is not allowed to vary.
        neutral_fields = (
            "signal",
            "runtime_compatibility",
            "method",
            "scale",
            "huber_delta",
            "raw_penalty_clip",
            "require_verified_training_chains",
        )
        normalized = {key: copy.deepcopy(continuity.get(key)) for key in neutral_fields}
        _set_path(experiment, continuity_path, normalized)
    return _canonical_mapping(contract, "reward-pair contract")


def reward_pair_contract_fingerprint(resolved_config: Mapping[str, Any]) -> str:
    return _json_sha256(derive_reward_pair_contract(resolved_config))


def derive_basis_factor_match_contract(
    resolved_config: Mapping[str, Any],
    basis_factor_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the B/G comparison while excluding only numerical/graph choices."""

    config = copy.deepcopy(_mapping(resolved_config, "resolved config"))
    config.pop("hydra", None)
    _drop_path(config, ("wandb", "name"))
    _drop_path(config, ("wandb", "tags"))
    experiment = _mapping(config.get("experiment"), "resolved config experiment")
    config["experiment"] = experiment
    for key in (
        "run_id",
        "checkpoint_dir",
        "validation_video_dir",
        "resume_from",
        "body_synergy_contract",
        "muscle_control_contract",
    ):
        experiment.pop(key, None)
    ablation = experiment.get("continuity_ablation")
    if isinstance(ablation, Mapping):
        ablation = copy.deepcopy(dict(ablation))
        for key in (
            "condition",
            "matched_pair",
            "basis_family",
            "graph_regularized_basis",
        ):
            ablation.pop(key, None)
        experiment["continuity_ablation"] = ablation
    action = experiment.get("action_representation")
    if not isinstance(action, Mapping):
        raise ValueError("basis-factor comparison requires action_representation")
    action = copy.deepcopy(dict(action))
    for key in (
        "basis_path",
        "expected_basis_fingerprint",
        "required_basis_family",
        "require_graph_regularization",
        "forbid_graph_regularization",
        "expected_graph_regularization_lineage_fingerprint",
        "expected_graph_continuity_fingerprint",
        "expected_graph_regularization_lambda",
        "expected_graph_lambda_selection_fingerprint",
        "expected_graph_continuity_release_fingerprint",
    ):
        action.pop(key, None)
    transform = action.get("coefficient_transform")
    if isinstance(transform, Mapping):
        transform = copy.deepcopy(dict(transform))
        transform.pop("stats_path", None)
        transform.pop("expected_stats_fingerprint", None)
        action["coefficient_transform"] = transform
    experiment["action_representation"] = action
    factor = validate_basis_factor_contract(basis_factor_contract)
    return {
        "schema_version": "continuity_ablation_basis_factor_match_contract_v1",
        "resolved_training_contract": _canonical_mapping(config, "basis-factor resolved contract"),
        "basis_factor_contract": factor,
    }


def basis_factor_match_fingerprint(
    resolved_config: Mapping[str, Any],
    basis_factor_contract: Mapping[str, Any],
) -> str:
    return _json_sha256(
        derive_basis_factor_match_contract(
            resolved_config,
            basis_factor_contract,
        )
    )


def validate_ablation_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        payload,
        {"schema_version", "study_identity", "runs", "artifact_fingerprint"},
        "ablation evidence",
    )
    if payload["schema_version"] != ABLATION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported continuity ablation evidence schema")
    identity = _validate_study_identity(payload["study_identity"])
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list):
        raise ValueError("ablation runs must be a list")
    runs = [_validate_run(item, study_identity=identity) for item in raw_runs]
    keys = [(run["condition"], run["seed"]) for run in runs]
    expected_keys = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    if keys != expected_keys:
        raise ValueError("ablation runs must contain A0..G1 x seeds 0..2 in canonical order")
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("ablation run_id values must be unique")
    if len({run["config_hash"] for run in runs}) != len(runs):
        raise ValueError("ablation config_hash values must be unique")
    by_key = {(run["condition"], run["seed"]): run for run in runs}
    for pair in ("A", "B", "C", "G"):
        for seed in SEEDS:
            baseline = by_key[(f"{pair}0", seed)]
            rewarded = by_key[(f"{pair}1", seed)]
            if baseline["seed"] != rewarded["seed"]:
                raise ValueError(f"{pair} reward pair seed differs")
            if baseline["matched_nonreward_contract_fingerprint"] != rewarded["matched_nonreward_contract_fingerprint"]:
                raise ValueError(f"{pair} seed {seed} non-reward contracts are not matched")
            for field in (
                "basis_fingerprint",
                "residual_basis_fingerprint",
                "graph_regularization_lineage_fingerprint",
            ):
                if baseline[field] != rewarded[field]:
                    raise ValueError(f"{pair} seed {seed} changes {field} across the reward pair")
    for reward_state in (0, 1):
        for seed in SEEDS:
            standard = by_key[(f"B{reward_state}", seed)]
            graph = by_key[(f"G{reward_state}", seed)]
            if (
                standard["matched_basis_factor_contract_fingerprint"]
                != graph["matched_basis_factor_contract_fingerprint"]
            ):
                raise ValueError(f"B{reward_state}/G{reward_state} seed {seed} differ outside the basis factor")
    result = {
        "schema_version": ABLATION_EVIDENCE_SCHEMA_VERSION,
        "study_identity": identity,
        "runs": runs,
    }
    supplied = _sha256(payload["artifact_fingerprint"], "artifact_fingerprint")
    result["artifact_fingerprint"] = supplied
    if ablation_evidence_fingerprint(result) != supplied:
        raise ValueError("continuity ablation evidence fingerprint is stale")
    return result


def build_continuity_ablation_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    evidence = validate_ablation_evidence(payload)
    by_key = {(run["condition"], run["seed"]): run for run in evidence["runs"]}
    reward_pairs: dict[str, Any] = {}
    for pair in ("A", "B", "C", "G"):
        comparisons = [
            _paired_reward_comparison(
                by_key[(f"{pair}0", seed)],
                by_key[(f"{pair}1", seed)],
            )
            for seed in SEEDS
        ]
        promotion_pass = all(item["promotion_passed_both"] for item in comparisons)
        task_pass = all(item["task_preservation_passed"] for item in comparisons)
        physiology_pass = all(item["target_continuity_improvement_passed"] for item in comparisons)
        performance_pass = all(item["performance_overhead_passed"] for item in comparisons)
        global_degraded = any(item["global_continuity_degraded"] for item in comparisons)
        if not promotion_pass:
            status = "promotion_gate_not_met"
        elif task_pass and physiology_pass and performance_pass and not global_degraded:
            status = "all_preregistered_gates_passed"
        elif physiology_pass and not task_pass:
            status = "pareto_tradeoff_task_degraded"
        elif physiology_pass and global_degraded:
            status = "target_improved_global_degraded"
        else:
            status = "preregistered_gates_not_met"
        reward_pairs[pair] = {
            "baseline_condition": f"{pair}0",
            "reward_condition": f"{pair}1",
            "seed_comparisons": comparisons,
            "paired_mean_deltas": _mean_deltas(comparisons),
            "promotion_passed_all_seeds": promotion_pass,
            "task_preservation_passed_all_seeds": task_pass,
            "target_continuity_improvement_passed_all_seeds": physiology_pass,
            "global_continuity_degraded_any_seed": global_degraded,
            "performance_overhead_passed_all_seeds": performance_pass,
            "status": status,
        }

    graph_basis = {
        reward_state: [
            _paired_basis_comparison(
                by_key[(f"B{reward_state}", seed)],
                by_key[(f"G{reward_state}", seed)],
            )
            for seed in SEEDS
        ]
        for reward_state in (0, 1)
    }
    report: dict[str, Any] = {
        "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
        "identity": {
            **copy.deepcopy(evidence["study_identity"]),
            "source_evidence_fingerprint": evidence["artifact_fingerprint"],
        },
        "thresholds": copy.deepcopy(THRESHOLDS),
        "run_inventory": [
            {
                **{
                    key: copy.deepcopy(run[key])
                    for key in (
                        "condition",
                        "seed",
                        "run_id",
                        "config_hash",
                        "checkpoint_fingerprint",
                        "promotion_fingerprint",
                        "promotion_passed",
                        "matched_nonreward_contract_fingerprint",
                        "matched_basis_factor_contract_fingerprint",
                        "basis_fingerprint",
                        "residual_basis_fingerprint",
                        "graph_regularization_lineage_fingerprint",
                        "graph_lambda_selection_fingerprint",
                        "continuity_release_fingerprint",
                        "continuity_loss_spec_fingerprint",
                        "joint_report_fingerprint",
                        "runtime_environment_fingerprint",
                        "fresh_optimizer",
                        "resumed",
                    )
                },
                "metrics": copy.deepcopy(run["metrics"]),
            }
            for run in evidence["runs"]
        ],
        "reward_ablation": reward_pairs,
        "graph_nmf_factor": {
            "standard_vs_graph_without_reward": graph_basis[0],
            "standard_vs_graph_with_reward": graph_basis[1],
            "interpretation": "matched_except_for_basis_numerics_and_preregistered_graph_regularization",
        },
        "claim_scope": {
            "overall_better_conditions": [
                pair for pair, result in reward_pairs.items() if result["status"] == "all_preregistered_gates_passed"
            ],
            "pareto_tradeoff_conditions": [
                pair for pair, result in reward_pairs.items() if result["status"] == "pareto_tradeoff_task_degraded"
            ],
            "target_only_improvement_conditions": [
                pair for pair, result in reward_pairs.items() if result["status"] == "target_improved_global_degraded"
            ],
            "neural_synergy_claim": False,
            "human_physiology_claim": False,
        },
        "limitations": [
            "The thresholds are preregistered engineering gates, not physiological truth.",
            "Target and global continuity are reported separately; target-only gains are not overall physiological gains.",
            "A continuity improvement with failed task gates is a Pareto trade-off, not an overall improvement.",
            "Graph-NMF and online reward are separate factors and do not identify neural modules.",
            "Cross-space EMG claims require a separately bound paired or independent-cohort report.",
        ],
    }
    report["report_fingerprint"] = ablation_report_fingerprint(report)
    return report


def _paired_reward_comparison(baseline: Mapping[str, Any], rewarded: Mapping[str, Any]) -> dict[str, Any]:
    left = baseline["metrics"]
    right = rewarded["metrics"]
    deltas = {
        "early_termination_absolute_increase": right["early_termination_rate"] - left["early_termination_rate"],
        "frame_coverage_absolute_degradation": left["frame_coverage"] - right["frame_coverage"],
        "tracking_error_relative_degradation": _relative_change(left["tracking_error"], right["tracking_error"]),
        "saturation_absolute_increase": right["action_decoder_saturation_fraction"]
        - left["action_decoder_saturation_fraction"],
        "target_continuity_p95_relative_improvement": _relative_improvement(
            left["target_activation_continuity_p95"], right["target_activation_continuity_p95"]
        ),
        "target_violation_relative_improvement": _relative_improvement(
            left["target_violation_fraction"], right["target_violation_fraction"]
        ),
        "global_continuity_p95_relative_improvement": _relative_improvement(
            left["global_activation_continuity_p95"], right["global_activation_continuity_p95"]
        ),
        "global_violation_relative_improvement": _relative_improvement(
            left["global_violation_fraction"], right["global_violation_fraction"]
        ),
        "steps_per_second_relative_overhead": _relative_improvement(
            left["steps_per_second"], right["steps_per_second"]
        ),
    }
    promotion_pass = baseline["promotion_passed"] and rewarded["promotion_passed"]
    task_pass = promotion_pass and (
        deltas["early_termination_absolute_increase"] <= THRESHOLDS["max_early_termination_absolute_increase"]
        and deltas["frame_coverage_absolute_degradation"] <= THRESHOLDS["max_frame_coverage_absolute_degradation"]
        and deltas["tracking_error_relative_degradation"] <= THRESHOLDS["max_tracking_error_relative_degradation"]
        and deltas["saturation_absolute_increase"] <= THRESHOLDS["max_saturation_absolute_increase"]
    )
    target_pass = (
        deltas["target_continuity_p95_relative_improvement"]
        >= THRESHOLDS["min_target_continuity_p95_relative_improvement"]
        and deltas["target_violation_relative_improvement"] >= THRESHOLDS["min_target_violation_relative_improvement"]
    )
    global_degraded = (
        deltas["global_continuity_p95_relative_improvement"] < 0.0
        or deltas["global_violation_relative_improvement"] < 0.0
    )
    performance_pass = (
        deltas["steps_per_second_relative_overhead"] <= THRESHOLDS["max_steps_per_second_relative_overhead"]
    )
    return {
        "seed": baseline["seed"],
        "baseline_run_id": baseline["run_id"],
        "reward_run_id": rewarded["run_id"],
        "promotion_passed_both": promotion_pass,
        "deltas": deltas,
        "task_preservation_passed": task_pass,
        "target_continuity_improvement_passed": target_pass,
        "global_continuity_degraded": global_degraded,
        "performance_overhead_passed": performance_pass,
        "reward_penalty_evidence": {
            "raw_mean": right["penalty_continuity_raw_mean"],
            "after_local_clip_mean": right["penalty_continuity_after_local_clip_mean"],
            "effective_after_total_clip_mean": right["penalty_continuity_effective_after_total_clip_mean"],
            "total_clip_masked_fraction": right["total_clip_masked_fraction"],
        },
    }


def _paired_basis_comparison(standard: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    left = standard["metrics"]
    right = graph["metrics"]
    return {
        "seed": standard["seed"],
        "standard_run_id": standard["run_id"],
        "graph_run_id": graph["run_id"],
        "reward_enabled": standard["continuity_reward_enabled"],
        "matched_basis_factor_contract_fingerprint": standard["matched_basis_factor_contract_fingerprint"],
        "promotion_passed_both": standard["promotion_passed"] and graph["promotion_passed"],
        "target_continuity_p95_relative_improvement": _relative_improvement(
            left["target_activation_continuity_p95"], right["target_activation_continuity_p95"]
        ),
        "global_continuity_p95_relative_improvement": _relative_improvement(
            left["global_activation_continuity_p95"], right["global_activation_continuity_p95"]
        ),
        "tracking_error_relative_degradation": _relative_change(left["tracking_error"], right["tracking_error"]),
        "steps_per_second_relative_overhead": _relative_improvement(
            left["steps_per_second"], right["steps_per_second"]
        ),
    }


def _mean_deltas(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    fields = tuple(comparisons[0]["deltas"])
    return {field: float(np.mean([item["deltas"][field] for item in comparisons])) for field in fields}


def _validate_study_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, "study identity")
    fields = {
        "branch_commit_sha",
        "dataset_split_fingerprint",
        "runtime_environment_fingerprint",
        "validation_motion_fingerprint",
        "promotion_contract_fingerprint",
        "racket_curriculum_fingerprint",
        "taxonomy_fingerprint",
        "continuity_candidate_graph_fingerprint",
        "continuity_loss_spec_fingerprint",
        "continuity_release_fingerprint",
        "calibration_fingerprint",
        "calibrated_reward_coefficient",
        "total_timesteps",
    }
    _exact_keys(identity, fields, "study identity")
    commit = _text(identity["branch_commit_sha"], "branch_commit_sha")
    if _GIT_SHA.fullmatch(commit) is None:
        raise ValueError("branch_commit_sha must be a lowercase 40- or 64-hex git id")
    return {
        "branch_commit_sha": commit,
        **{
            field: _sha256(identity[field], field)
            for field in fields
            - {
                "branch_commit_sha",
                "calibrated_reward_coefficient",
                "total_timesteps",
            }
        },
        "calibrated_reward_coefficient": _positive_finite(
            identity["calibrated_reward_coefficient"],
            "calibrated_reward_coefficient",
        ),
        "total_timesteps": _positive_int(identity["total_timesteps"], "total_timesteps"),
    }


def _validate_run(value: Any, *, study_identity: Mapping[str, Any]) -> dict[str, Any]:
    run = _mapping(value, "ablation run")
    fields = {
        "condition",
        "seed",
        "run_id",
        "config_hash",
        "source_artifacts",
        "resolved_config",
        "checkpoint_fingerprint",
        "promotion_fingerprint",
        "promotion_passed",
        "matched_nonreward_contract_fingerprint",
        "matched_basis_factor_contract_fingerprint",
        "basis_factor_contract",
        "action_mode",
        "basis_family",
        "continuity_reward_enabled",
        "graph_regularized_basis",
        "fresh_optimizer",
        "resumed",
        "total_timesteps",
        "basis_fingerprint",
        "residual_basis_fingerprint",
        "graph_regularization_lineage_fingerprint",
        "graph_lambda_selection_fingerprint",
        "continuity_reward_coefficient",
        "continuity_release_fingerprint",
        "continuity_loss_spec_fingerprint",
        "continuity_calibration_fingerprint",
        "continuity_graph_fingerprint",
        "joint_report_fingerprint",
        "runtime_environment_fingerprint",
        "metrics",
    }
    _exact_keys(run, fields, "ablation run")
    condition = str(run["condition"])
    if condition not in _EXPECTED_CONDITION:
        raise ValueError(f"unsupported ablation condition {condition!r}")
    _pair, mode, family, reward_enabled, graph_enabled = _EXPECTED_CONDITION[condition]
    if (
        run["action_mode"] != mode
        or run["basis_family"] != family
        or run["continuity_reward_enabled"] is not reward_enabled
        or run["graph_regularized_basis"] is not graph_enabled
    ):
        raise ValueError(f"ablation condition {condition} semantics differ from contract")
    if run["fresh_optimizer"] is not True or run["resumed"] is not False:
        raise ValueError("every ablation run requires a fresh optimizer and resumed=false")
    seed = _nonnegative_int(run["seed"], "seed")
    if seed not in SEEDS:
        raise ValueError("ablation seed must be 0, 1, or 2")
    total_timesteps = int(study_identity["total_timesteps"])
    if _positive_int(run["total_timesteps"], "run total_timesteps") != total_timesteps:
        raise ValueError("ablation run total_timesteps differs from study identity")
    sources = _mapping(run["source_artifacts"], "source_artifacts")
    _exact_keys(sources, _SOURCE_ARTIFACT_FIELDS, "source_artifacts")
    source_artifacts = {
        field: _optional_sha256(sources[field], f"source artifact {field}") for field in sorted(sources)
    }
    for required_source in (
        "resolved_config",
        "run_manifest",
        "checkpoint_metadata",
        "promotion",
        "validation_metrics",
        "continuity_release",
        "joint_report",
        "runtime_environment",
        "performance_profiler",
    ):
        if source_artifacts[required_source] is None:
            raise ValueError(f"source artifact {required_source} is required")
    basis = _optional_sha256(run["basis_fingerprint"], "basis_fingerprint")
    residual = _optional_sha256(run["residual_basis_fingerprint"], "residual_basis_fingerprint")
    graph_lineage = _optional_sha256(
        run["graph_regularization_lineage_fingerprint"],
        "graph_regularization_lineage_fingerprint",
    )
    graph_selection = _optional_sha256(
        run["graph_lambda_selection_fingerprint"],
        "graph_lambda_selection_fingerprint",
    )
    if mode == "full_354" and any(item is not None for item in (basis, residual, graph_lineage, graph_selection)):
        raise ValueError("direct-354 condition cannot bind W/R graph artifacts")
    if mode != "full_354" and basis is None:
        raise ValueError("synergy ablation condition requires a basis fingerprint")
    if mode == "fixed_synergy_residual" and residual is None:
        raise ValueError("residual ablation condition requires an R fingerprint")
    if graph_enabled != (graph_lineage is not None and graph_selection is not None):
        raise ValueError("graph-NMF condition graph lineage/selection is missing or unexpected")
    if (basis is None) != (source_artifacts["basis_artifact"] is None):
        raise ValueError("basis source artifact presence differs from basis fingerprint")
    if (residual is None) != (source_artifacts["residual_artifact"] is None):
        raise ValueError("residual source artifact presence differs from residual fingerprint")
    if (graph_lineage is None) != (source_artifacts["graph_lineage"] is None):
        raise ValueError("graph source artifact presence differs from graph lineage")
    release = _sha256(run["continuity_release_fingerprint"], "continuity_release_fingerprint")
    loss_spec = _sha256(run["continuity_loss_spec_fingerprint"], "continuity_loss_spec_fingerprint")
    calibration = _sha256(
        run["continuity_calibration_fingerprint"],
        "continuity_calibration_fingerprint",
    )
    graph_fingerprint = _sha256(run["continuity_graph_fingerprint"], "continuity_graph_fingerprint")
    expected_shared = {
        "continuity_release_fingerprint": study_identity["continuity_release_fingerprint"],
        "continuity_loss_spec_fingerprint": study_identity["continuity_loss_spec_fingerprint"],
        "continuity_calibration_fingerprint": study_identity["calibration_fingerprint"],
        "continuity_graph_fingerprint": study_identity["continuity_candidate_graph_fingerprint"],
    }
    supplied_shared = {
        "continuity_release_fingerprint": release,
        "continuity_loss_spec_fingerprint": loss_spec,
        "continuity_calibration_fingerprint": calibration,
        "continuity_graph_fingerprint": graph_fingerprint,
    }
    for field, expected in expected_shared.items():
        if supplied_shared[field] != expected:
            raise ValueError(f"ablation run {field} differs from study identity")
    continuity_coefficient = _nonnegative_finite(
        run["continuity_reward_coefficient"],
        "continuity_reward_coefficient",
    )
    expected_coefficient = float(study_identity["calibrated_reward_coefficient"]) if reward_enabled else 0.0
    if continuity_coefficient != expected_coefficient:
        raise ValueError("ablation run continuity reward coefficient differs from study calibration")
    resolved_config = _canonical_mapping(run["resolved_config"], "resolved_config")
    if source_artifacts["resolved_config"] != _json_sha256(resolved_config):
        raise ValueError("resolved config source fingerprint is stale")
    derived_reward = reward_pair_contract_fingerprint(resolved_config)
    supplied_reward = _sha256(
        run["matched_nonreward_contract_fingerprint"],
        "matched_nonreward_contract_fingerprint",
    )
    if supplied_reward != derived_reward:
        raise ValueError("matched_nonreward_contract_fingerprint was not derived from resolved config")
    factor_raw = run["basis_factor_contract"]
    factor = None if factor_raw is None else validate_basis_factor_contract(factor_raw)
    if (mode == "full_354") != (factor is None):
        raise ValueError("basis factor contract presence differs from action mode")
    supplied_factor = _optional_sha256(
        run["matched_basis_factor_contract_fingerprint"],
        "matched_basis_factor_contract_fingerprint",
    )
    if factor is None:
        if supplied_factor is not None:
            raise ValueError("direct-354 run cannot declare a basis-factor match")
    else:
        derived_factor = basis_factor_match_fingerprint(resolved_config, factor)
        if supplied_factor != derived_factor:
            raise ValueError("matched_basis_factor_contract_fingerprint was not derived from run artifacts")
    metrics = _validate_metrics(run["metrics"], reward_enabled=reward_enabled)
    runtime_environment = _sha256(
        run["runtime_environment_fingerprint"],
        "runtime_environment_fingerprint",
    )
    if runtime_environment != study_identity["runtime_environment_fingerprint"]:
        raise ValueError("ablation runtime environment differs from study identity")
    return {
        "condition": condition,
        "seed": seed,
        "run_id": _text(run["run_id"], "run_id"),
        "config_hash": _text(run["config_hash"], "config_hash"),
        "source_artifacts": source_artifacts,
        "resolved_config": resolved_config,
        "checkpoint_fingerprint": _sha256(run["checkpoint_fingerprint"], "checkpoint_fingerprint"),
        "promotion_fingerprint": _sha256(run["promotion_fingerprint"], "promotion_fingerprint"),
        "promotion_passed": _boolean(run["promotion_passed"], "promotion_passed"),
        "matched_nonreward_contract_fingerprint": supplied_reward,
        "matched_basis_factor_contract_fingerprint": supplied_factor,
        "basis_factor_contract": factor,
        "action_mode": mode,
        "basis_family": family,
        "continuity_reward_enabled": reward_enabled,
        "graph_regularized_basis": graph_enabled,
        "fresh_optimizer": True,
        "resumed": False,
        "total_timesteps": total_timesteps,
        "basis_fingerprint": basis,
        "residual_basis_fingerprint": residual,
        "graph_regularization_lineage_fingerprint": graph_lineage,
        "graph_lambda_selection_fingerprint": graph_selection,
        "continuity_reward_coefficient": continuity_coefficient,
        "continuity_release_fingerprint": release,
        "continuity_loss_spec_fingerprint": loss_spec,
        "continuity_calibration_fingerprint": calibration,
        "continuity_graph_fingerprint": graph_fingerprint,
        "joint_report_fingerprint": _sha256(run["joint_report_fingerprint"], "joint_report_fingerprint"),
        "runtime_environment_fingerprint": runtime_environment,
        "metrics": metrics,
    }


def _validate_metrics(value: Any, *, reward_enabled: bool) -> dict[str, float | int]:
    metrics = _mapping(value, "ablation metrics")
    _exact_keys(metrics, _METRIC_FIELDS, "ablation metrics")
    result = {field: _finite(metrics[field], field) for field in _METRIC_FIELDS}
    bounded = (
        "early_termination_rate",
        "frame_coverage",
        "action_decoder_saturation_fraction",
        "action_preclip_out_of_bounds_fraction",
        "residual_energy_fraction",
        "target_violation_fraction",
        "target_active_chain_fraction",
        "global_violation_fraction",
        "global_active_chain_fraction",
        "total_clip_masked_fraction",
    )
    for field in bounded:
        if not 0.0 <= result[field] <= 1.0:
            raise ValueError(f"{field} must lie in [0,1]")
    for field in ("target_chain_count", "target_edge_count", "global_chain_count", "global_edge_count"):
        result[field] = _positive_int(result[field], field)
    for field in ("steps_per_second", "compile_time_seconds", "gpu_memory_gb", "update_wall_time_seconds"):
        if result[field] <= 0.0:
            raise ValueError(f"{field} must be positive")
    signed_penalties = {
        "penalty_continuity_raw_mean",
        "penalty_continuity_after_local_clip_mean",
        "penalty_continuity_effective_after_total_clip_mean",
    }
    for field in (
        _METRIC_FIELDS
        - signed_penalties
        - {
            "steps_per_second",
            "compile_time_seconds",
            "gpu_memory_gb",
            "update_wall_time_seconds",
        }
    ):
        if result[field] < 0.0:
            raise ValueError(f"{field} must be non-negative")
    penalties = [result[field] for field in sorted(signed_penalties)]
    if reward_enabled:
        if any(value >= 0.0 for value in penalties):
            raise ValueError("reward-enabled run requires nonzero negative continuity penalties")
        if (
            abs(result["penalty_continuity_effective_after_total_clip_mean"])
            > abs(result["penalty_continuity_after_local_clip_mean"]) + 1e-12
        ):
            raise ValueError("effective continuity penalty exceeds its local-clipped magnitude")
    elif any(value != 0.0 for value in penalties) or result["total_clip_masked_fraction"] != 0.0:
        raise ValueError("reward-disabled run must have zero continuity penalty evidence")
    return {field: result[field] for field in sorted(result)}


def _relative_change(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / max(abs(baseline), 1e-12)


def _relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _canonical_mapping(value: Any, field: str) -> dict[str, Any]:
    mapping = _mapping(value, field)
    # JSON round-trip rejects non-serializable and non-finite config material.
    try:
        return json.loads(json.dumps(mapping, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite JSON") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{field} fields differ from contract")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return result


def _optional_sha256(value: Any, field: str) -> str | None:
    return None if value is None else _sha256(value, field)


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0 or result != float(value):
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    return result


def _nonnegative_finite(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _positive_finite(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _drop_path(root: dict[str, Any], path: Sequence[str]) -> None:
    current: Any = root
    for key in path[:-1]:
        if not isinstance(current, Mapping) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _get_path(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _set_path(root: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current: dict[str, Any] = root
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_continuity_ablation_report(_load_json(args.evidence_json))
    _atomic_write_json(args.output_json, report)
    print(args.output_json.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {value}")),
    )
    return _mapping(payload, str(path))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
