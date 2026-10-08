"""Derive immutable 24-run continuity-ablation evidence from run artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from analysis.physiology_synergy.build_continuity_ablation_report import (
    ABLATION_EVIDENCE_SCHEMA_VERSION,
    CONDITIONS,
    SEEDS,
    ablation_evidence_fingerprint,
    basis_factor_match_fingerprint,
    reward_pair_contract_fingerprint,
    validate_ablation_evidence,
)
from analysis.physiology_synergy.build_joint_report import joint_report_fingerprint
from musclemimic.physiology.release import (
    load_continuity_training_release,
    resolve_continuity_training_release,
    validate_continuity_training_release,
)
from musclemimic.runner.checkpointing import config_hash
from musclemimic.synergy.action_interface import load_structured_residual_basis
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.basis_factor_contract import validate_basis_factor_contract
from musclemimic.synergy.graph_nmf import (
    graph_regularization_lineage_fingerprint,
    validate_formal_graph_nmf_manifest,
)

RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION = "continuity_ablation_run_artifact_inventory_v1"
CHECKPOINT_EVIDENCE_SCHEMA_VERSION = "continuity_ablation_checkpoint_evidence_v1"
PROMOTION_EVIDENCE_SCHEMA_VERSION = "continuity_ablation_promotion_evidence_v1"
VALIDATION_EVIDENCE_SCHEMA_VERSION = "continuity_ablation_validation_metrics_v1"
RUNTIME_ENVIRONMENT_SCHEMA_VERSION = "continuity_ablation_runtime_environment_v1"
PERFORMANCE_PROFILE_SCHEMA_VERSION = "continuity_ablation_performance_profile_v1"

_LOADED_RUN_FIELDS = {
    "condition",
    "seed",
    "resolved_config",
    "run_manifest",
    "checkpoint_metadata",
    "promotion",
    "validation_metrics",
    "continuity_release",
    "basis_artifact",
    "residual_artifact",
    "joint_report",
    "runtime_environment",
    "performance_profiler",
}
_INVENTORY_RUN_FIELDS = {"condition", "seed", "run_directory", "artifacts"}
_ARTIFACT_PATH_FIELDS = {
    "resolved_config",
    "run_manifest",
    "checkpoint_metadata",
    "promotion",
    "validation_metrics",
    "continuity_release",
    "basis_artifact",
    "residual_artifact",
    "joint_report",
    "runtime_environment",
    "performance_profiler",
}
_EXPECTED_CONDITION = {
    "A0": ("full_354", "direct_354", False, False),
    "A1": ("full_354", "direct_354", True, False),
    "B0": ("fixed_synergy", "standard_nmf", False, False),
    "B1": ("fixed_synergy", "standard_nmf", True, False),
    "C0": ("fixed_synergy_residual", "standard_nmf_structured_residual", False, False),
    "C1": ("fixed_synergy_residual", "standard_nmf_structured_residual", True, False),
    "G0": ("fixed_synergy", "graph_nmf", False, True),
    "G1": ("fixed_synergy", "graph_nmf", True, True),
}


def sealed_artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("artifact_fingerprint", None)
    return _json_sha256(unsigned)


def seal_checkpoint_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {"schema_version": CHECKPOINT_EVIDENCE_SCHEMA_VERSION, **copy.deepcopy(dict(payload))}
    result["artifact_fingerprint"] = sealed_artifact_fingerprint(result)
    return _validate_checkpoint(result)


def seal_promotion_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {"schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION, **copy.deepcopy(dict(payload))}
    result["artifact_fingerprint"] = sealed_artifact_fingerprint(result)
    return _validate_promotion(result)


def seal_validation_metrics_evidence(
    *,
    identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "identity": copy.deepcopy(dict(identity)),
        "metrics": copy.deepcopy(dict(metrics)),
    }
    result["artifact_fingerprint"] = sealed_artifact_fingerprint(result)
    return _validate_validation_metrics(result)


def seal_runtime_environment_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {"schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION, **copy.deepcopy(dict(payload))}
    result["artifact_fingerprint"] = sealed_artifact_fingerprint(result)
    return _validate_runtime_environment(result)


def seal_performance_profile(
    *,
    identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
        "identity": copy.deepcopy(dict(identity)),
        "metrics": copy.deepcopy(dict(metrics)),
    }
    result["artifact_fingerprint"] = sealed_artifact_fingerprint(result)
    return _validate_performance_profile(result)


def build_continuity_ablation_evidence(
    loaded_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build evidence from already loaded artifacts; no match hash is accepted."""

    if not isinstance(loaded_runs, Sequence) or isinstance(loaded_runs, str | bytes):
        raise ValueError("loaded runs must be a sequence")
    normalized = [_build_run(entry) for entry in loaded_runs]
    normalized.sort(key=lambda run: (CONDITIONS.index(run["condition"]), run["seed"]))
    keys = [(run["condition"], run["seed"]) for run in normalized]
    expected = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    if keys != expected:
        raise ValueError("evidence builder requires exactly A0..G1 x seeds 0..2")
    identity_fields = {
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
    study_identity = {field: normalized[0].pop(f"__study_{field}") for field in identity_fields}
    for run in normalized[1:]:
        for field in identity_fields:
            value = run.pop(f"__study_{field}")
            if value != study_identity[field]:
                raise ValueError(f"run {run['condition']} seed {run['seed']} differs on study identity {field}")
    payload: dict[str, Any] = {
        "schema_version": ABLATION_EVIDENCE_SCHEMA_VERSION,
        "study_identity": study_identity,
        "runs": normalized,
    }
    payload["artifact_fingerprint"] = ablation_evidence_fingerprint(payload)
    return validate_ablation_evidence(payload)


def build_continuity_ablation_evidence_from_inventory(
    inventory: Mapping[str, Any],
    *,
    inventory_directory: str | Path,
) -> dict[str, Any]:
    source = _mapping(inventory, "run artifact inventory")
    if set(source) != {"schema_version", "runs"}:
        raise ValueError("run artifact inventory fields differ from contract")
    if source["schema_version"] != RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported run artifact inventory schema")
    runs = source["runs"]
    if not isinstance(runs, list):
        raise ValueError("run artifact inventory runs must be a list")
    base = Path(inventory_directory).expanduser().resolve()
    return build_continuity_ablation_evidence([_load_inventory_run(item, base=base) for item in runs])


def _build_run(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(value, "loaded ablation run")
    if set(source) != _LOADED_RUN_FIELDS:
        forbidden = {
            "matched_nonreward_contract_fingerprint",
            "matched_basis_factor_contract_fingerprint",
        } & set(source)
        if forbidden:
            raise ValueError("evidence builder does not accept hand-entered matched fingerprints")
        raise ValueError("loaded ablation run fields differ from contract")
    condition = str(source["condition"])
    if condition not in _EXPECTED_CONDITION:
        raise ValueError(f"unsupported ablation condition {condition!r}")
    seed = _nonnegative_int(source["seed"], "seed")
    if seed not in SEEDS:
        raise ValueError("ablation seed must be 0, 1, or 2")
    mode, family, reward_enabled, graph_enabled = _EXPECTED_CONDITION[condition]
    resolved = _canonical_mapping(source["resolved_config"], "resolved config")
    experiment = _mapping(resolved.get("experiment"), "resolved config experiment")
    ablation = _mapping(experiment.get("continuity_ablation"), "continuity ablation config")
    expected_declaration = {
        "condition": condition,
        "seed": seed,
        "action_mode": mode,
        "basis_family": family,
        "continuity_reward_enabled": reward_enabled,
        "graph_regularized_basis": graph_enabled,
        "fresh_optimizer_required": True,
        "parent_initialization_checkpoint": None,
    }
    for field, expected in expected_declaration.items():
        if ablation.get(field) != expected:
            raise ValueError(f"resolved continuity ablation {field} differs from {condition} contract")
    configured_seeds = experiment.get("seeds")
    if configured_seeds != [seed] or int(experiment.get("n_seeds", 0)) != 1:
        raise ValueError("resolved config does not bind exactly the inventory seed")
    validation_config = _mapping(experiment.get("validation"), "resolved validation config")
    if validation_config.get("eval_seed") != seed:
        raise ValueError("validation seed differs from training seed")
    if experiment.get("auto_resume") is not False or experiment.get("resume_from") is not None:
        raise ValueError("formal ablation requires auto_resume=false and resume_from=null")
    run_id = _text(experiment.get("run_id"), "run_id")
    total_timesteps = _positive_int(experiment.get("total_timesteps"), "total_timesteps")

    manifest = _canonical_mapping(source["run_manifest"], "run manifest")
    manifest_experiment = _mapping(manifest.get("experiment_config"), "run manifest experiment_config")
    if manifest_experiment != experiment:
        raise ValueError("run manifest experiment config differs from resolved config")
    declared_config_hash = _text(manifest.get("config_hash"), "run manifest config_hash")
    if config_hash(experiment) != declared_config_hash:
        raise ValueError("run manifest config_hash differs from resolved config")

    checkpoint = _validate_checkpoint(source["checkpoint_metadata"])
    _assert_run_binding(checkpoint, run_id=run_id, config_hash_value=declared_config_hash, label="checkpoint")
    if checkpoint["fresh_optimizer"] is not True or checkpoint["resumed"] is not False:
        raise ValueError("checkpoint evidence is not a fresh optimizer run")
    if checkpoint["target_global_timestep"] != total_timesteps:
        raise ValueError("checkpoint target global timestep differs from training budget")
    promotion = _validate_promotion(source["promotion"])
    _assert_run_binding(promotion, run_id=run_id, config_hash_value=declared_config_hash, label="promotion")
    if promotion["checkpoint_fingerprint"] != checkpoint["checkpoint_fingerprint"]:
        raise ValueError("promotion belongs to another checkpoint")

    release = validate_continuity_training_release(
        _mapping(source["continuity_release"], "continuity release")
    ).to_manifest()
    release_fingerprint = release["release_fingerprint"]
    loss_fingerprint = release["loss_spec"]["loss_spec_fingerprint"]
    graph_fingerprint = release["candidate_graph"]["graph_fingerprint"]
    calibration_fingerprint = release["calibration"]["calibration_fingerprint"]
    coefficient = float(release["reward"]["coefficient"])
    if mode not in release["allowed_action_modes"]:
        raise ValueError("continuity release does not allow the run action mode")
    expected_release = release_fingerprint if reward_enabled else None
    if checkpoint["continuity_release_fingerprint"] != expected_release:
        raise ValueError("checkpoint continuity release binding differs from reward state")
    continuity_runtime = experiment.get("continuity_training_contract")
    if reward_enabled:
        if not isinstance(continuity_runtime, Mapping):
            raise ValueError("reward-enabled resolved config lacks continuity runtime contract")
        expected_runtime = {
            "release_fingerprint": release_fingerprint,
            "loss_spec_fingerprint": loss_fingerprint,
            "candidate_graph_fingerprint": graph_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "selected_reward_coefficient": coefficient,
            "action_mode": mode,
        }
        for field, expected in expected_runtime.items():
            if continuity_runtime.get(field) != expected:
                raise ValueError(f"continuity runtime contract differs on {field}")
    elif continuity_runtime is not None:
        raise ValueError("reward-disabled config unexpectedly binds a continuity runtime contract")

    basis_manifest, factor, basis_fingerprint, graph_lineage, graph_selection = _basis_contract(
        source["basis_artifact"],
        mode=mode,
        graph_enabled=graph_enabled,
        release_fingerprint=release_fingerprint,
        loss_fingerprint=loss_fingerprint,
        graph_fingerprint=graph_fingerprint,
    )
    action = experiment.get("action_representation", {})
    if mode == "full_354":
        if isinstance(action, Mapping) and action.get("enabled") is True:
            raise ValueError("direct-354 condition unexpectedly enables a synergy action interface")
    else:
        action = _mapping(action, "action representation")
        if action.get("expected_basis_fingerprint") != basis_fingerprint:
            raise ValueError("resolved config basis fingerprint differs from basis artifact")
        if action.get("expected_basis_factor_contract_fingerprint") != factor["basis_factor_contract_fingerprint"]:
            raise ValueError("resolved config factor fingerprint differs from basis artifact")
        transform = _mapping(action.get("coefficient_transform"), "action coefficient transform")
        factor_transform = factor["coefficient_transform_schema"]
        expected_transform = {
            "kind": factor_transform["kind"],
            "max_source": factor_transform["maximum_source"],
            "center_source": factor_transform["center_source"],
            "temperature": factor_transform["temperature"],
        }
        for field, expected in expected_transform.items():
            if transform.get(field) != expected:
                raise ValueError(f"action coefficient transform differs from basis factor on {field}")
    residual_manifest, residual_fingerprint = _residual_contract(
        source["residual_artifact"],
        mode=mode,
        basis_fingerprint=basis_fingerprint,
    )
    if checkpoint["basis_fingerprint"] != basis_fingerprint:
        raise ValueError("checkpoint basis fingerprint differs from basis artifact")
    if checkpoint["residual_basis_fingerprint"] != residual_fingerprint:
        raise ValueError("checkpoint residual fingerprint differs from residual artifact")
    if mode == "fixed_synergy_residual":
        residual_config = _mapping(action.get("residual"), "residual action config")
        if residual_config.get("expected_fingerprint") != residual_fingerprint:
            raise ValueError("resolved residual fingerprint differs from artifact")

    joint = _validate_joint_report(source["joint_report"])
    joint_identity = _mapping(joint.get("identity"), "joint report identity")
    promotion_fingerprint = promotion["artifact_fingerprint"]
    _assert_identity_value(
        joint_identity,
        ("checkpoint_fingerprint",),
        checkpoint["checkpoint_fingerprint"],
        "joint report checkpoint",
    )
    _assert_identity_value(
        joint_identity,
        ("promotion_fingerprint",),
        promotion_fingerprint,
        "joint report promotion",
    )
    validation = _validate_validation_metrics(source["validation_metrics"])
    _assert_metric_identity(
        validation["identity"],
        run_id=run_id,
        config_hash_value=declared_config_hash,
        checkpoint_fingerprint=checkpoint["checkpoint_fingerprint"],
        promotion_fingerprint=promotion_fingerprint,
        joint_report_fingerprint=joint["report_fingerprint"],
        label="validation",
    )
    runtime = _validate_runtime_environment(source["runtime_environment"])
    if promotion["promotion_contract_fingerprint"] != runtime["promotion_contract_fingerprint"]:
        raise ValueError("promotion evidence contract differs from runtime environment")
    manifest_git = str(manifest.get("git_sha", "")).lower()
    if not manifest_git or not runtime["branch_commit_sha"].startswith(manifest_git):
        raise ValueError("run manifest git SHA differs from runtime environment")
    profiler = _validate_performance_profile(source["performance_profiler"])
    _assert_profile_identity(
        profiler["identity"],
        run_id=run_id,
        config_hash_value=declared_config_hash,
        checkpoint_fingerprint=checkpoint["checkpoint_fingerprint"],
    )
    metrics = {**validation["metrics"], **profiler["metrics"]}
    expected_coverage = {
        "target_chain_count": release["loss_spec"]["target_chain_count"],
        "target_edge_count": release["loss_spec"]["target_edge_count"],
        "global_chain_count": release["diagnostic_graph"]["global_chain_count"],
        "global_edge_count": release["diagnostic_graph"]["global_edge_count"],
    }
    for field, expected in expected_coverage.items():
        if metrics.get(field) != expected:
            raise ValueError(f"validation {field} differs from continuity release coverage")

    factor_match = None if factor is None else basis_factor_match_fingerprint(resolved, factor)
    source_fingerprints = {
        "resolved_config": _json_sha256(resolved),
        "run_manifest": _json_sha256(manifest),
        "checkpoint_metadata": checkpoint["artifact_fingerprint"],
        "promotion": promotion["artifact_fingerprint"],
        "validation_metrics": validation["artifact_fingerprint"],
        "continuity_release": release_fingerprint,
        "basis_artifact": None if basis_manifest is None else _json_sha256(basis_manifest),
        "residual_artifact": None if residual_manifest is None else _json_sha256(residual_manifest),
        "graph_lineage": None if graph_lineage is None else _json_sha256(graph_lineage),
        "joint_report": joint["report_fingerprint"],
        "runtime_environment": runtime["artifact_fingerprint"],
        "performance_profiler": profiler["artifact_fingerprint"],
    }
    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "run_id": run_id,
        "config_hash": declared_config_hash,
        "source_artifacts": source_fingerprints,
        "resolved_config": resolved,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "promotion_fingerprint": promotion_fingerprint,
        "promotion_passed": promotion["passed"],
        "matched_nonreward_contract_fingerprint": reward_pair_contract_fingerprint(resolved),
        "matched_basis_factor_contract_fingerprint": factor_match,
        "basis_factor_contract": factor,
        "action_mode": mode,
        "basis_family": family,
        "continuity_reward_enabled": reward_enabled,
        "graph_regularized_basis": graph_enabled,
        "fresh_optimizer": True,
        "resumed": False,
        "total_timesteps": total_timesteps,
        "basis_fingerprint": basis_fingerprint,
        "residual_basis_fingerprint": residual_fingerprint,
        "graph_regularization_lineage_fingerprint": (
            None if graph_lineage is None else graph_regularization_lineage_fingerprint(graph_lineage)
        ),
        "graph_lambda_selection_fingerprint": graph_selection,
        "continuity_reward_coefficient": coefficient if reward_enabled else 0.0,
        "continuity_release_fingerprint": release_fingerprint,
        "continuity_loss_spec_fingerprint": loss_fingerprint,
        "continuity_calibration_fingerprint": calibration_fingerprint,
        "continuity_graph_fingerprint": graph_fingerprint,
        "joint_report_fingerprint": joint["report_fingerprint"],
        "runtime_environment_fingerprint": runtime["environment_fingerprint"],
        "metrics": metrics,
        "__study_branch_commit_sha": runtime["branch_commit_sha"],
        "__study_dataset_split_fingerprint": runtime["dataset_split_fingerprint"],
        "__study_runtime_environment_fingerprint": runtime["environment_fingerprint"],
        "__study_validation_motion_fingerprint": runtime["validation_motion_fingerprint"],
        "__study_promotion_contract_fingerprint": runtime["promotion_contract_fingerprint"],
        "__study_racket_curriculum_fingerprint": runtime["racket_curriculum_fingerprint"],
        "__study_taxonomy_fingerprint": release["taxonomy"]["taxonomy_fingerprint"],
        "__study_continuity_candidate_graph_fingerprint": graph_fingerprint,
        "__study_continuity_loss_spec_fingerprint": loss_fingerprint,
        "__study_continuity_release_fingerprint": release_fingerprint,
        "__study_calibration_fingerprint": calibration_fingerprint,
        "__study_calibrated_reward_coefficient": coefficient,
        "__study_total_timesteps": total_timesteps,
    }
    return result


def _basis_contract(
    value: Any,
    *,
    mode: str,
    graph_enabled: bool,
    release_fingerprint: str,
    loss_fingerprint: str,
    graph_fingerprint: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None, dict[str, Any] | None, str | None]:
    if mode == "full_354":
        if value is not None:
            raise ValueError("direct-354 run cannot supply a basis artifact")
        return None, None, None, None, None
    manifest = _canonical_mapping(value, "basis artifact")
    fingerprint = _require_self_fingerprint(
        manifest,
        "artifact_fingerprint",
        "basis artifact",
        ensure_ascii=False,
    )
    factor = validate_basis_factor_contract(_mapping(manifest.get("basis_factor_contract"), "basis factor contract"))
    if manifest.get("basis_factor_contract_fingerprint") != factor["basis_factor_contract_fingerprint"]:
        raise ValueError("basis artifact factor fingerprint differs from embedded contract")
    expected_family = "graph_nmf" if graph_enabled else "standard_nmf"
    if manifest.get("basis_family") != expected_family:
        raise ValueError("basis artifact family differs from ablation condition")
    graph = manifest.get("graph_regularization")
    if not graph_enabled:
        if graph is not None:
            raise ValueError("standard NMF basis unexpectedly has graph lineage")
        return manifest, factor, fingerprint, None, None
    graph = validate_formal_graph_nmf_manifest(_mapping(graph, "graph regularization lineage"))
    expected = {
        "continuity_release_fingerprint": release_fingerprint,
        "continuity_loss_spec_fingerprint": loss_fingerprint,
        "continuity_graph_fingerprint": graph_fingerprint,
        "basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
    }
    for field, expected_value in expected.items():
        if graph[field] != expected_value:
            raise ValueError(f"Graph-NMF lineage differs from release/factor on {field}")
    return manifest, factor, fingerprint, graph, graph["lambda_selection_fingerprint"]


def _residual_contract(
    value: Any,
    *,
    mode: str,
    basis_fingerprint: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if mode != "fixed_synergy_residual":
        if value is not None:
            raise ValueError("non-residual condition cannot supply a residual artifact")
        return None, None
    manifest = _canonical_mapping(value, "residual artifact")
    fingerprint = _require_self_fingerprint(manifest, "artifact_fingerprint", "residual artifact")
    if manifest.get("source_basis_fingerprint") != basis_fingerprint:
        raise ValueError("residual artifact belongs to another primary basis")
    return manifest, fingerprint


def _validate_checkpoint(value: Any) -> dict[str, Any]:
    payload = _sealed_mapping(value, CHECKPOINT_EVIDENCE_SCHEMA_VERSION, "checkpoint evidence")
    expected = {
        "schema_version",
        "run_id",
        "config_hash",
        "checkpoint_fingerprint",
        "checkpoint_metadata_content_sha256",
        "global_timestep",
        "target_global_timestep",
        "fresh_optimizer",
        "resumed",
        "continuity_release_fingerprint",
        "basis_fingerprint",
        "residual_basis_fingerprint",
        "artifact_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint evidence fields differ from contract")
    result = {
        "schema_version": CHECKPOINT_EVIDENCE_SCHEMA_VERSION,
        "run_id": _text(payload["run_id"], "checkpoint run_id"),
        "config_hash": _text(payload["config_hash"], "checkpoint config_hash"),
        "checkpoint_fingerprint": _sha256(payload["checkpoint_fingerprint"], "checkpoint fingerprint"),
        "checkpoint_metadata_content_sha256": _sha256(
            payload["checkpoint_metadata_content_sha256"], "checkpoint metadata content"
        ),
        "global_timestep": _positive_int(payload["global_timestep"], "checkpoint global timestep"),
        "target_global_timestep": _positive_int(payload["target_global_timestep"], "checkpoint target global timestep"),
        "fresh_optimizer": _boolean(payload["fresh_optimizer"], "fresh_optimizer"),
        "resumed": _boolean(payload["resumed"], "resumed"),
        "continuity_release_fingerprint": _optional_sha256(
            payload["continuity_release_fingerprint"], "checkpoint continuity release"
        ),
        "basis_fingerprint": _optional_sha256(payload["basis_fingerprint"], "checkpoint basis"),
        "residual_basis_fingerprint": _optional_sha256(
            payload["residual_basis_fingerprint"], "checkpoint residual basis"
        ),
        "artifact_fingerprint": _sha256(payload["artifact_fingerprint"], "checkpoint artifact"),
    }
    _check_seal(result, "checkpoint evidence")
    return result


def _validate_promotion(value: Any) -> dict[str, Any]:
    payload = _sealed_mapping(value, PROMOTION_EVIDENCE_SCHEMA_VERSION, "promotion evidence")
    expected = {
        "schema_version",
        "run_id",
        "config_hash",
        "checkpoint_fingerprint",
        "promotion_contract_fingerprint",
        "passed",
        "artifact_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("promotion evidence fields differ from contract")
    result = {
        "schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "run_id": _text(payload["run_id"], "promotion run_id"),
        "config_hash": _text(payload["config_hash"], "promotion config_hash"),
        "checkpoint_fingerprint": _sha256(payload["checkpoint_fingerprint"], "promotion checkpoint"),
        "promotion_contract_fingerprint": _sha256(payload["promotion_contract_fingerprint"], "promotion contract"),
        "passed": _boolean(payload["passed"], "promotion passed"),
        "artifact_fingerprint": _sha256(payload["artifact_fingerprint"], "promotion artifact"),
    }
    _check_seal(result, "promotion evidence")
    return result


def _validate_validation_metrics(value: Any) -> dict[str, Any]:
    payload = _sealed_mapping(value, VALIDATION_EVIDENCE_SCHEMA_VERSION, "validation metrics")
    if set(payload) != {"schema_version", "identity", "metrics", "artifact_fingerprint"}:
        raise ValueError("validation metrics fields differ from contract")
    identity = _mapping(payload["identity"], "validation identity")
    expected_identity = {
        "run_id",
        "config_hash",
        "checkpoint_fingerprint",
        "promotion_fingerprint",
        "joint_report_fingerprint",
    }
    if set(identity) != expected_identity:
        raise ValueError("validation metric identity fields differ from contract")
    canonical_identity = {
        "run_id": _text(identity["run_id"], "validation run_id"),
        "config_hash": _text(identity["config_hash"], "validation config_hash"),
        **{
            field: _sha256(identity[field], f"validation {field}")
            for field in expected_identity - {"run_id", "config_hash"}
        },
    }
    metrics = _numeric_mapping(payload["metrics"], "validation metrics")
    result = {
        "schema_version": VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "identity": canonical_identity,
        "metrics": metrics,
        "artifact_fingerprint": _sha256(payload["artifact_fingerprint"], "validation artifact"),
    }
    _check_seal(result, "validation metrics")
    return result


def _validate_runtime_environment(value: Any) -> dict[str, Any]:
    payload = _sealed_mapping(value, RUNTIME_ENVIRONMENT_SCHEMA_VERSION, "runtime environment")
    expected = {
        "schema_version",
        "branch_commit_sha",
        "dataset_split_fingerprint",
        "environment_fingerprint",
        "validation_motion_fingerprint",
        "promotion_contract_fingerprint",
        "racket_curriculum_fingerprint",
        "artifact_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("runtime environment fields differ from contract")
    commit = _text(payload["branch_commit_sha"], "branch commit SHA")
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("runtime branch commit must be lowercase 40- or 64-hex")
    result = {
        "schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "branch_commit_sha": commit,
        **{
            field: _sha256(payload[field], f"runtime {field}")
            for field in expected - {"schema_version", "branch_commit_sha", "artifact_fingerprint"}
        },
        "artifact_fingerprint": _sha256(payload["artifact_fingerprint"], "runtime artifact"),
    }
    _check_seal(result, "runtime environment")
    return result


def _validate_performance_profile(value: Any) -> dict[str, Any]:
    payload = _sealed_mapping(value, PERFORMANCE_PROFILE_SCHEMA_VERSION, "performance profile")
    if set(payload) != {"schema_version", "identity", "metrics", "artifact_fingerprint"}:
        raise ValueError("performance profile fields differ from contract")
    identity = _mapping(payload["identity"], "performance profile identity")
    if set(identity) != {"run_id", "config_hash", "checkpoint_fingerprint"}:
        raise ValueError("performance profile identity fields differ from contract")
    canonical_identity = {
        "run_id": _text(identity["run_id"], "profile run_id"),
        "config_hash": _text(identity["config_hash"], "profile config_hash"),
        "checkpoint_fingerprint": _sha256(identity["checkpoint_fingerprint"], "profile checkpoint"),
    }
    metrics = _numeric_mapping(payload["metrics"], "performance profile metrics")
    required_metrics = {
        "steps_per_second",
        "compile_time_seconds",
        "gpu_memory_gb",
        "update_wall_time_seconds",
    }
    if set(metrics) != required_metrics or any(value <= 0.0 for value in metrics.values()):
        raise ValueError("performance profile requires four positive preregistered metrics")
    result = {
        "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
        "identity": canonical_identity,
        "metrics": metrics,
        "artifact_fingerprint": _sha256(payload["artifact_fingerprint"], "profile artifact"),
    }
    _check_seal(result, "performance profile")
    return result


def _validate_joint_report(value: Any) -> dict[str, Any]:
    payload = _canonical_mapping(value, "joint report")
    if payload.get("schema_version") != "forehand_physio_synergy_joint_report_v2":
        raise ValueError("ablation evidence requires a v2 physiology/synergy joint report")
    supplied = _sha256(payload.get("report_fingerprint"), "joint report fingerprint")
    if joint_report_fingerprint(payload) != supplied:
        raise ValueError("joint report fingerprint is stale")
    return payload


def _load_inventory_run(value: Any, *, base: Path) -> dict[str, Any]:
    spec = _mapping(value, "inventory run")
    if set(spec) != _INVENTORY_RUN_FIELDS:
        raise ValueError("inventory run fields differ from contract")
    run_dir = _resolve_path(base, spec["run_directory"])
    if not run_dir.is_dir():
        raise ValueError(f"inventory run directory does not exist: {run_dir}")
    paths = _mapping(spec["artifacts"], "inventory artifact paths")
    if set(paths) != _ARTIFACT_PATH_FIELDS:
        raise ValueError("inventory artifact path fields differ from contract")
    resolved_paths = {key: None if paths[key] is None else _resolve_path(run_dir, paths[key]) for key in paths}
    release_path = _required_path(resolved_paths["continuity_release"], "continuity_release")
    release = load_continuity_training_release(release_path)
    resolve_continuity_training_release(release)
    loaded = {
        "condition": spec["condition"],
        "seed": spec["seed"],
        "resolved_config": _load_config(_required_path(resolved_paths["resolved_config"], "resolved_config")),
        "run_manifest": _load_json(_required_path(resolved_paths["run_manifest"], "run_manifest")),
        "checkpoint_metadata": _load_json(_required_path(resolved_paths["checkpoint_metadata"], "checkpoint_metadata")),
        "promotion": _load_json(_required_path(resolved_paths["promotion"], "promotion")),
        "validation_metrics": _load_json(_required_path(resolved_paths["validation_metrics"], "validation_metrics")),
        "continuity_release": release.to_manifest(),
        "basis_artifact": None,
        "residual_artifact": None,
        "joint_report": _load_json(_required_path(resolved_paths["joint_report"], "joint_report")),
        "runtime_environment": _load_json(_required_path(resolved_paths["runtime_environment"], "runtime_environment")),
        "performance_profiler": _load_json(
            _required_path(resolved_paths["performance_profiler"], "performance_profiler")
        ),
    }
    if resolved_paths["basis_artifact"] is not None:
        basis = load_synergy_basis(resolved_paths["basis_artifact"])
        loaded["basis_artifact"] = basis.manifest
        if resolved_paths["residual_artifact"] is not None:
            residual = load_structured_residual_basis(
                resolved_paths["residual_artifact"],
                expected_actuator_names=basis.muscle_names,
                expected_source_basis_fingerprint=basis.fingerprint,
            )
            residual_path = Path(residual.source_path) / "manifest.json"
            loaded["residual_artifact"] = _load_json(residual_path)
    elif resolved_paths["residual_artifact"] is not None:
        raise ValueError("residual artifact path requires a primary basis artifact")
    return loaded


def _assert_run_binding(
    evidence: Mapping[str, Any],
    *,
    run_id: str,
    config_hash_value: str,
    label: str,
) -> None:
    if evidence["run_id"] != run_id or evidence["config_hash"] != config_hash_value:
        raise ValueError(f"{label} identity differs from resolved run")


def _assert_metric_identity(
    identity: Mapping[str, Any],
    *,
    run_id: str,
    config_hash_value: str,
    checkpoint_fingerprint: str,
    promotion_fingerprint: str,
    joint_report_fingerprint: str,
    label: str,
) -> None:
    expected = {
        "run_id": run_id,
        "config_hash": config_hash_value,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "promotion_fingerprint": promotion_fingerprint,
        "joint_report_fingerprint": joint_report_fingerprint,
    }
    if dict(identity) != expected:
        raise ValueError(f"{label} identity differs from run artifacts")


def _assert_profile_identity(
    identity: Mapping[str, Any],
    *,
    run_id: str,
    config_hash_value: str,
    checkpoint_fingerprint: str,
) -> None:
    if dict(identity) != {
        "run_id": run_id,
        "config_hash": config_hash_value,
        "checkpoint_fingerprint": checkpoint_fingerprint,
    }:
        raise ValueError("performance profile identity differs from run artifacts")


def _assert_identity_value(
    identity: Mapping[str, Any],
    keys: Sequence[str],
    expected: str,
    label: str,
) -> None:
    values = [identity[key] for key in keys if key in identity]
    if len(values) != 1 or values[0] != expected:
        raise ValueError(f"{label} differs from run artifacts")


def _sealed_mapping(value: Any, schema: str, label: str) -> dict[str, Any]:
    payload = _canonical_mapping(value, label)
    if payload.get("schema_version") != schema:
        raise ValueError(f"unsupported {label} schema")
    return payload


def _check_seal(payload: Mapping[str, Any], label: str) -> None:
    if sealed_artifact_fingerprint(payload) != payload["artifact_fingerprint"]:
        raise ValueError(f"{label} fingerprint is stale")


def _require_self_fingerprint(
    payload: Mapping[str, Any],
    field: str,
    label: str,
    *,
    ensure_ascii: bool = True,
) -> str:
    supplied = _sha256(payload.get(field), f"{label} {field}")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if _json_sha256(unsigned, ensure_ascii=ensure_ascii) != supplied:
        raise ValueError(f"{label} fingerprint is stale")
    return supplied


def _numeric_mapping(value: Any, label: str) -> dict[str, int | float]:
    payload = _mapping(value, label)
    result: dict[str, int | float] = {}
    for key, raw in payload.items():
        if isinstance(raw, bool):
            raise ValueError(f"{label}.{key} must be numeric")
        number = float(raw)
        if not (-float("inf") < number < float("inf")):
            raise ValueError(f"{label}.{key} must be finite")
        result[str(key)] = raw if isinstance(raw, int) else number
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _canonical_mapping(value: Any, label: str) -> dict[str, Any]:
    payload = _mapping(value, label)
    try:
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return text


def _optional_sha256(value: Any, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0 or result != float(value):
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _json_sha256(value: Any, *, ensure_ascii: bool = True) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {value}")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON artifact {path}") from exc
    return _mapping(payload, str(path))


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    try:
        config = OmegaConf.load(path)
        payload = OmegaConf.to_container(config, resolve=True)
    except Exception as exc:
        raise ValueError(f"cannot resolve Hydra config {path}") from exc
    return _mapping(payload, str(path))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _resolve_path(base: Path, value: Any) -> Path:
    text = _text(value, "artifact path")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _required_path(path: Path | None, label: str) -> Path:
    if path is None or not path.exists():
        raise ValueError(f"required inventory artifact is missing: {label}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inventory-json", type=Path)
    group.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.inventory_json is not None:
        inventory_path = args.inventory_json.expanduser().resolve(strict=True)
        evidence = build_continuity_ablation_evidence_from_inventory(
            _load_json(inventory_path),
            inventory_directory=inventory_path.parent,
        )
    else:
        run_dirs = list(args.run_dir or ())
        if len(run_dirs) != 24:
            raise ValueError("--run-dir requires exactly 24 run directories")
        inventory = {
            "schema_version": RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION,
            "runs": [_discover_run_directory(path) for path in run_dirs],
        }
        evidence = build_continuity_ablation_evidence_from_inventory(
            inventory,
            inventory_directory=Path.cwd(),
        )
    _atomic_write_json(args.output_json, evidence)
    print(args.output_json.resolve())


def _discover_run_directory(value: Path) -> dict[str, Any]:
    run_dir = value.expanduser().resolve(strict=True)
    descriptor_path = run_dir / "continuity_ablation_artifacts.json"
    if not descriptor_path.is_file():
        raise ValueError(
            f"run directory lacks continuity_ablation_artifacts.json: {run_dir}; "
            "use --inventory-json for explicit shared artifact paths"
        )
    descriptor = _load_json(descriptor_path)
    if set(descriptor) != {"condition", "seed", "artifacts"}:
        raise ValueError("run-directory artifact descriptor fields differ from contract")
    return {
        "condition": descriptor["condition"],
        "seed": descriptor["seed"],
        "run_directory": str(run_dir),
        "artifacts": descriptor["artifacts"],
    }


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
