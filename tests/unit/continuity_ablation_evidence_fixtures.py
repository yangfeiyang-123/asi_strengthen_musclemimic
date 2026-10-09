"""Synthetic but fully sealed 24-run inputs for continuity evidence tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from analysis.physiology_synergy.build_continuity_ablation_evidence import (
    seal_checkpoint_evidence,
    seal_performance_profile,
    seal_promotion_evidence,
    seal_runtime_environment_evidence,
    seal_validation_metrics_evidence,
)
from analysis.physiology_synergy.build_joint_report import joint_report_fingerprint
from musclemimic.runner.checkpointing import config_hash
from musclemimic.synergy.basis_factor_contract import (
    build_basis_factor_contract,
    near_zero_mask_contract,
)
from musclemimic.synergy.graph_nmf import (
    GRAPH_NORMALIZATION_SPACE,
    GRAPH_REGULARIZATION_SCHEMA_VERSION,
    LAPLACIAN_NMF_METHOD,
)
from tests.unit.continuity_v3_fixtures import continuity_release_fixture

COMMIT = "1" * 40
DATASET = "2" * 64
ENVIRONMENT = "3" * 64
VALIDATION = "4" * 64
PROMOTION_CONTRACT = "5" * 64
RACKET = "6" * 64


def build_loaded_ablation_runs(root: Path) -> list[dict]:
    release, _release_path, _paths = continuity_release_fixture(root)
    runtime = seal_runtime_environment_evidence(
        {
            "branch_commit_sha": COMMIT,
            "dataset_split_fingerprint": DATASET,
            "environment_fingerprint": ENVIRONMENT,
            "validation_motion_fingerprint": VALIDATION,
            "promotion_contract_fingerprint": PROMOTION_CONTRACT,
            "racket_curriculum_fingerprint": RACKET,
        }
    )
    factor = _factor_contract()
    standard_basis = _basis_manifest("standard_nmf", factor=factor, release=release)
    graph_basis = _basis_manifest("graph_nmf", factor=factor, release=release)
    residual_basis = _residual_manifest(standard_basis["artifact_fingerprint"])
    runs: list[dict] = []
    for condition in ("A0", "A1", "B0", "B1", "C0", "C1", "G0", "G1"):
        for seed in (0, 1, 2):
            if condition.startswith("A"):
                basis = residual = None
            elif condition.startswith("G"):
                basis, residual = graph_basis, None
            elif condition.startswith("C"):
                basis, residual = standard_basis, residual_basis
            else:
                basis, residual = standard_basis, None
            runs.append(
                _run(
                    condition,
                    seed,
                    release=release,
                    runtime=runtime,
                    basis=basis,
                    residual=residual,
                )
            )
    return runs


def rebind_loaded_run(run: dict) -> None:
    """Re-seal every run-local identity after a negative-test config mutation."""

    experiment = run["resolved_config"]["experiment"]
    basis = run["basis_artifact"]
    residual = run["residual_artifact"]
    if basis is not None:
        action = experiment["action_representation"]
        action["expected_basis_fingerprint"] = basis["artifact_fingerprint"]
        action["expected_basis_factor_contract_fingerprint"] = basis["basis_factor_contract_fingerprint"]
    if residual is not None:
        experiment["action_representation"]["residual"]["expected_fingerprint"] = residual["artifact_fingerprint"]
    run_id = experiment["run_id"]
    cfg_hash = config_hash(experiment)
    run["run_manifest"]["config_hash"] = cfg_hash
    run["run_manifest"]["experiment_config"] = copy.deepcopy(experiment)
    checkpoint = run["checkpoint_metadata"]
    checkpoint_basis = None if basis is None else basis["artifact_fingerprint"]
    checkpoint_residual = None if residual is None else residual["artifact_fingerprint"]
    run["checkpoint_metadata"] = seal_checkpoint_evidence(
        {
            key: copy.deepcopy(value)
            for key, value in checkpoint.items()
            if key not in {"schema_version", "artifact_fingerprint", "run_id", "config_hash"}
        }
        | {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "basis_fingerprint": checkpoint_basis,
            "residual_basis_fingerprint": checkpoint_residual,
        }
    )
    checkpoint_fingerprint = run["checkpoint_metadata"]["checkpoint_fingerprint"]
    run["promotion"] = seal_promotion_evidence(
        {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_contract_fingerprint": PROMOTION_CONTRACT,
            "passed": run["promotion"]["passed"],
        }
    )
    promotion_fingerprint = run["promotion"]["artifact_fingerprint"]
    joint = copy.deepcopy(run["joint_report"])
    joint["identity"] = {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "promotion_fingerprint": promotion_fingerprint,
    }
    joint["report_fingerprint"] = joint_report_fingerprint(joint)
    run["joint_report"] = joint
    validation_metrics = run["validation_metrics"]["metrics"]
    run["validation_metrics"] = seal_validation_metrics_evidence(
        identity={
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_fingerprint": promotion_fingerprint,
            "joint_report_fingerprint": joint["report_fingerprint"],
        },
        metrics=validation_metrics,
    )
    profile_metrics = run["performance_profiler"]["metrics"]
    run["performance_profiler"] = seal_performance_profile(
        identity={
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
        },
        metrics=profile_metrics,
    )


def reseal_basis_manifest(manifest: dict) -> None:
    manifest["basis_factor_contract_fingerprint"] = manifest["basis_factor_contract"][
        "basis_factor_contract_fingerprint"
    ]
    graph = manifest.get("graph_regularization")
    if graph is not None:
        graph["basis_factor_contract_fingerprint"] = manifest["basis_factor_contract_fingerprint"]
    manifest["artifact_fingerprint"] = _manifest_fingerprint(manifest, ensure_ascii=False)


def _run(
    condition: str,
    seed: int,
    *,
    release: dict,
    runtime: dict,
    basis: dict | None,
    residual: dict | None,
) -> dict:
    reward_enabled = condition.endswith("1")
    prefix = condition[0]
    mode = {
        "A": "full_354",
        "B": "fixed_synergy",
        "C": "fixed_synergy_residual",
        "G": "fixed_synergy",
    }[prefix]
    family = {
        "A": "direct_354",
        "B": "standard_nmf",
        "C": "standard_nmf_structured_residual",
        "G": "graph_nmf",
    }[prefix]
    graph_enabled = prefix == "G"
    run_id = f"forehand_clear_continuity_ablation_v1_{condition.lower()}_s{seed}"
    action = _action_config(
        mode=mode,
        graph_enabled=graph_enabled,
        basis=basis,
        residual=residual,
        release=release,
    )
    continuity = {
        "mode": "reward" if reward_enabled else "off",
        "signal": "activation",
        "release_path": "/sealed/release.json" if reward_enabled else None,
        "expected_release_fingerprint": release["release_fingerprint"] if reward_enabled else None,
        "runtime_compatibility": "portable_muscle_channel_abi",
        "method": "robust_fascicle_continuity_v1",
        "scale": 0.05,
        "huber_delta": 1.0,
        "coefficient": 0.0,
        "raw_penalty_clip": None,
        "require_verified_training_chains": True,
    }
    experiment = {
        "run_id": run_id,
        "auto_resume": False,
        "resume_from": None,
        "n_seeds": 1,
        "seeds": [seed],
        "total_timesteps": 320_000_000,
        "lr": 2.0e-4,
        "network": {"hidden_sizes": [512, 512, 256]},
        "training_source": {"split_fingerprint": DATASET},
        "validation": {"eval_seed": seed, "motion_fingerprint": VALIDATION},
        "promotion": {"contract_fingerprint": PROMOTION_CONTRACT},
        "racket_mass_curriculum": {"fingerprint": RACKET},
        "action_representation": action,
        "env_params": {"reward_params": {"intra_muscle_consistency": continuity}},
        "continuity_ablation": {
            "schema_version": "forehand_continuity_matched_ablation_v1",
            "condition": condition,
            "matched_pair": prefix,
            "seed": seed,
            "action_mode": mode,
            "basis_family": family,
            "continuity_reward_enabled": reward_enabled,
            "graph_regularized_basis": graph_enabled,
            "fresh_optimizer_required": True,
            "parent_initialization_checkpoint": None,
        },
    }
    if reward_enabled:
        experiment["continuity_training_contract"] = {
            "release_fingerprint": release["release_fingerprint"],
            "loss_spec_fingerprint": release["loss_spec"]["loss_spec_fingerprint"],
            "candidate_graph_fingerprint": release["candidate_graph"]["graph_fingerprint"],
            "calibration_fingerprint": release["calibration"]["calibration_fingerprint"],
            "selected_reward_coefficient": release["reward"]["coefficient"],
            "action_mode": mode,
        }
    resolved = {
        "wandb": {"project": "fixture", "name": run_id, "tags": [condition, f"seed-{seed}"]},
        "experiment": experiment,
    }
    cfg_hash = config_hash(experiment)
    checkpoint_fingerprint = hashlib.sha256(f"checkpoint:{run_id}".encode()).hexdigest()
    basis_fingerprint = None if basis is None else basis["artifact_fingerprint"]
    residual_fingerprint = None if residual is None else residual["artifact_fingerprint"]
    checkpoint = seal_checkpoint_evidence(
        {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "checkpoint_metadata_content_sha256": hashlib.sha256(f"metadata:{run_id}".encode()).hexdigest(),
            "global_timestep": 320_000_000,
            "target_global_timestep": 320_000_000,
            "fresh_optimizer": True,
            "resumed": False,
            "continuity_release_fingerprint": release["release_fingerprint"] if reward_enabled else None,
            "basis_fingerprint": basis_fingerprint,
            "residual_basis_fingerprint": residual_fingerprint,
        }
    )
    promotion = seal_promotion_evidence(
        {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_contract_fingerprint": PROMOTION_CONTRACT,
            "passed": True,
        }
    )
    joint = {
        "schema_version": "forehand_physio_synergy_joint_report_v2",
        "identity": {
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_fingerprint": promotion["artifact_fingerprint"],
        },
        "claim_scope": {"fixture": True},
    }
    joint["report_fingerprint"] = joint_report_fingerprint(joint)
    validation = seal_validation_metrics_evidence(
        identity={
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_fingerprint": promotion["artifact_fingerprint"],
            "joint_report_fingerprint": joint["report_fingerprint"],
        },
        metrics=_validation_metrics(reward=reward_enabled),
    )
    profiler = seal_performance_profile(
        identity={
            "run_id": run_id,
            "config_hash": cfg_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
        },
        metrics={
            "steps_per_second": 97.0 if reward_enabled else 100.0,
            "compile_time_seconds": 41.0 if reward_enabled else 40.0,
            "gpu_memory_gb": 12.2 if reward_enabled else 12.0,
            "update_wall_time_seconds": 1.02 if reward_enabled else 1.0,
        },
    )
    return {
        "condition": condition,
        "seed": seed,
        "resolved_config": resolved,
        "run_manifest": {
            "config_hash": cfg_hash,
            "git_sha": COMMIT,
            "experiment_config": copy.deepcopy(experiment),
        },
        "checkpoint_metadata": checkpoint,
        "promotion": promotion,
        "validation_metrics": validation,
        "continuity_release": copy.deepcopy(release),
        "basis_artifact": copy.deepcopy(basis),
        "residual_artifact": copy.deepcopy(residual),
        "joint_report": joint,
        "runtime_environment": copy.deepcopy(runtime),
        "performance_profiler": profiler,
    }


def _action_config(
    *,
    mode: str,
    graph_enabled: bool,
    basis: dict | None,
    residual: dict | None,
    release: dict,
) -> dict:
    if mode == "full_354":
        return {"enabled": False, "mode": "full_354"}
    factor = basis["basis_factor_contract"]
    action = {
        "enabled": True,
        "schema_version": "early_synergy_action_v2",
        "mode": mode,
        "basis_path": f"/sealed/{basis['basis_family']}",
        "expected_basis_fingerprint": basis["artifact_fingerprint"],
        "require_basis_factor_contract": True,
        "expected_basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
        "required_basis_family": basis["basis_family"],
        "require_raw_unit_basis_factor": True,
        "require_graph_regularization": graph_enabled,
        "forbid_graph_regularization": not graph_enabled,
        "coefficient_transform": {
            "kind": "bounded_sigmoid",
            "stats_path": f"/sealed/{basis['basis_family']}-stats.json",
            "expected_stats_fingerprint": ("7" if graph_enabled else "8") * 64,
            "max_source": "train_q99_times_1p2",
            "center_source": "train_q50",
            "temperature": 1.0,
        },
        "exploration": {
            "calibrate_in_physical_space": True,
            "target_initial_excitation_rms": 0.08,
            "std_mode": "per_dimension",
            "residual_std_scale": 0.25,
        },
        "residual": {
            "enabled": residual is not None,
            "basis_path": None if residual is None else "/sealed/residual",
            "expected_fingerprint": None if residual is None else residual["artifact_fingerprint"],
            "alpha": 0.03 if residual is not None else 0.0,
        },
    }
    if graph_enabled:
        graph = basis["graph_regularization"]
        action.update(
            expected_graph_regularization_lineage_fingerprint="9" * 64,
            expected_graph_continuity_fingerprint=release["candidate_graph"]["graph_fingerprint"],
            expected_graph_regularization_lambda=graph["lambda"],
            expected_graph_lambda_selection_fingerprint=graph["lambda_selection_fingerprint"],
            expected_graph_continuity_release_fingerprint=release["release_fingerprint"],
        )
    return action


def _factor_contract() -> dict:
    return build_basis_factor_contract(
        fit_scope="whole_body",
        source_dataset_fingerprint=DATASET,
        train_motion_uids=[0, 1, 2],
        validation_motion_uids=[3, 4],
        primitive_source_manifest_fingerprint="a" * 64,
        signal_kind="physical_excitation_raw_units",
        sample_weighting={"kind": "per_motion_equal"},
        phase_weighting={"preparation": 1.0, "acceleration": 2.0, "follow_through": 1.0},
        normalization={"normalization": "none", "scales": [1.0, 1.0, 1.0, 1.0]},
        near_zero_mask=near_zero_mask_contract(channel_count=4, kept_indices=[0, 1, 2, 3], threshold=1e-8),
        kept_actuator_indices=[0, 1, 2, 3],
        candidate_ranks=[2],
        selected_rank=2,
        nmf_initialization_seeds=[0, 1, 2],
        max_iter=1000,
        tol=1e-6,
        dynamic_coverage_environment_fingerprint=ENVIRONMENT,
        dynamic_coverage_rollout_fingerprint="b" * 64,
    )


def _basis_manifest(family: str, *, factor: dict, release: dict) -> dict:
    graph = None
    if family == "graph_nmf":
        graph = {
            "schema_version": GRAPH_REGULARIZATION_SCHEMA_VERSION,
            "enabled": True,
            "method": LAPLACIAN_NMF_METHOD,
            "lambda": 0.1,
            "requested_lambda": 0.1,
            "continuity_graph_id": release["candidate_graph"]["graph_id"],
            "continuity_graph_fingerprint": release["candidate_graph"]["graph_fingerprint"],
            "taxonomy_id": release["taxonomy"]["taxonomy_id"],
            "taxonomy_fingerprint": release["taxonomy"]["taxonomy_fingerprint"],
            "edge_count": 1,
            "full_graph_edge_count": 1,
            "normalization_space": GRAPH_NORMALIZATION_SPACE,
            "training_enabled_only": True,
            "chain_ids": ["fixture-chain"],
            "ordered_muscle_schema_sha256": "c" * 64,
            "edge_set_fingerprint": "d" * 64,
            "scope": "whole_body",
            "continuity_release_fingerprint": release["release_fingerprint"],
            "continuity_loss_spec_fingerprint": release["loss_spec"]["loss_spec_fingerprint"],
            "lambda_selection_fingerprint": "e" * 64,
            "basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
        }
    manifest = {
        "schema_version": "forehand_clear_synergy_basis_v1",
        "basis_family": family,
        "basis_artifact_role": "production_basis",
        "basis_factor_contract": copy.deepcopy(factor),
        "basis_factor_contract_fingerprint": factor["basis_factor_contract_fingerprint"],
        "graph_regularization": graph,
        "numerical_basis_id": "graph" if family == "graph_nmf" else "standard",
    }
    manifest["artifact_fingerprint"] = _manifest_fingerprint(manifest, ensure_ascii=False)
    return manifest


def _residual_manifest(source_basis_fingerprint: str) -> dict:
    manifest = {
        "schema_version": "early_synergy_residual_basis_v3",
        "source_basis_fingerprint": source_basis_fingerprint,
        "alpha_reference": 0.03,
        "numerical_residual_id": "fixture-residual",
    }
    manifest["artifact_fingerprint"] = _manifest_fingerprint(manifest)
    return manifest


def _validation_metrics(*, reward: bool) -> dict[str, float]:
    target_scale = 0.8 if reward else 1.0
    global_scale = 0.9 if reward else 1.0
    penalties = {
        "penalty_continuity_raw_mean": -0.010 if reward else 0.0,
        "penalty_continuity_after_local_clip_mean": -0.009 if reward else 0.0,
        "penalty_continuity_effective_after_total_clip_mean": -0.008 if reward else 0.0,
        "total_clip_masked_fraction": 0.1 if reward else 0.0,
    }
    return {
        "early_termination_rate": 0.025 if reward else 0.020,
        "frame_coverage": 0.965 if reward else 0.970,
        "tracking_error": 0.051 if reward else 0.050,
        "relative_site_error": 0.051 if reward else 0.050,
        "right_hand_error": 0.031 if reward else 0.030,
        "action_decoder_saturation_fraction": 0.025 if reward else 0.020,
        "action_preclip_out_of_bounds_fraction": 0.012 if reward else 0.010,
        "action_clip_correction_rms": 0.006 if reward else 0.005,
        "activation_energy": 0.098 if reward else 0.100,
        "residual_energy_fraction": 0.0,
        "synergy_coefficient_effective_dimension": 8.0,
        "target_activation_continuity_mean": 0.100 * target_scale,
        "target_activation_continuity_p95": 0.200 * target_scale,
        "target_activation_continuity_max": 0.400 * target_scale,
        "target_violation_fraction": 0.300 * target_scale,
        "target_active_chain_fraction": 0.800,
        "target_chain_count": 4,
        "target_edge_count": 20,
        "global_activation_continuity_mean": 0.110 * global_scale,
        "global_activation_continuity_p95": 0.220 * global_scale,
        "global_activation_continuity_max": 0.440 * global_scale,
        "global_violation_fraction": 0.320 * global_scale,
        "global_active_chain_fraction": 0.780,
        "global_chain_count": 28,
        "global_edge_count": 140,
        **penalties,
    }


def _manifest_fingerprint(value: dict, *, ensure_ascii: bool = True) -> str:
    unsigned = {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
