"""Independent-cohort sEMG/simulation synergy evaluation.

This module is intentionally separate from :mod:`musclemimic.evaluation.emg_eval`.
It accepts only explicitly unpaired action cohorts and therefore never computes
trial-wise envelope, timing, phase, or NMF-coefficient comparisons.  Each cohort
is impact-aligned and factorized independently; only channel-space NMF basis
geometry and within-cohort reconstruction quality are compared.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.evaluation.emg_eval import (
    EMG_OBSERVATION_MAPPING_SCHEMA_VERSION,
    PREPROCESSED_NORMALIZED_ENVELOPE_KIND,
    WHOLE_BODY_15_OF_16_SCOPE,
    _as_trial_time_channel,
    _exact_identity_scalar,
    _file_sha256,
    _identity_scalar,
    _identity_vector,
    _integer_scalar,
    _nonempty_string_array,
    _resolve_expected_policy_bindings,
    _scalar,
    _sha256_scalar,
    _string_vector,
    _validate_preprocessed_emg_contract,
    impact_aligned_resample,
    map_simulation_activation,
    match_synergy_bases,
    validate_emg_mapping,
    validate_simulation_activation_contract,
    validate_simulation_policy_evidence,
)
from musclemimic.synergy.metrics import (
    basis_condition_number,
    global_vaf,
    local_vaf,
    reconstruction_rmse,
)
from musclemimic.synergy.nmf import fit_best_initialization

EMG_COHORT_REPORT_SCHEMA_VERSION = "emg_unpaired_action_cohort_validation_v1"
UNPAIRED_COMPARISON_DESIGN = "unpaired_action_cohort_v1"
JIDIAN_IMPORT_SCHEMA_VERSION = "jidian_emg_import_v1"

COHORT_CLAIM_LIMITATIONS = (
    "The simulation and sEMG trials are independent action cohorts, not paired trials.",
    "NMF basis similarity is descriptive and does not establish trial-wise temporal agreement.",
    "Reported VAF is an in-cohort reconstruction diagnostic, not held-out generalization evidence.",
    "Surface-EMG observation mappings and MVC normalization remain measurement-model assumptions.",
    "Population confidence intervals are unavailable when the measured cohort contains one subject.",
)

_IDENTITY_FIELDS = {
    "trial_uid",
    "subject_uid",
    "session_uid",
    "dataset_split",
    "training_session_uid",
}

_SIMULATION_REQUIRED_FIELDS = {
    "muscle_activation",
    "actuator_names",
    "physical_signal_schema_version",
    "muscle_activation_source",
    "muscle_activation_semantics",
    "muscle_activation_roundoff_policy",
    "activation_valid_mask",
    "sampling_rate_hz",
    "impact_frame",
    "policy_decoder_type",
    "policy_checkpoint_fingerprint",
    "policy_promotion_fingerprint",
    "formal_synergy_basis_fingerprint",
    "analysis_synergy_basis_fingerprint",
    "model_taxonomy_id",
    "model_taxonomy_fingerprint",
    "runtime_model_hash",
    "actuator_schema_hash",
    "comparison_design",
    "comparison_set_uid",
    "action_id",
    "handedness",
    *_IDENTITY_FIELDS,
}

_EMG_REQUIRED_FIELDS = {
    "import_schema_version",
    "emg",
    "emg_signal_kind",
    "channel_names",
    "stream_channel_ids",
    "sides",
    "muscle_slugs",
    "sampling_rate_hz",
    "impact_frame",
    "processing_manifest_schema_version",
    "processing_manifest_sha256",
    "source_provenance_sha256",
    "selection_manifest_sha256",
    "channel_profile_id",
    "channel_profile_version",
    "channel_profile_sha256",
    "normalization_method",
    "processing_fallback_method",
    "acquired_channel_count",
    "comparable_channel_count",
    "excluded_sensor_ids",
    "comparison_design",
    "comparison_set_uid",
    "action_id",
    "handedness",
    *_IDENTITY_FIELDS,
}

_UNAVAILABLE_PAIRED_METRICS = {
    "envelope_correlation": ("Unavailable by design: independent cohorts have no one-to-one reference-trial pairing."),
    "normalized_dtw": ("Unavailable by design: DTW between arbitrary trials would fabricate a pairing."),
    "onset_error_s": ("Unavailable by design: onset error requires paired trials on a shared event reference."),
    "peak_timing_error_s": (
        "Unavailable by design: peak timing error requires paired trials on a shared event reference."
    ),
    "nmf_coefficient_h_correlation": (
        "Unavailable by design: independently fitted H rows belong to different trials and sample populations."
    ),
    "shared_phase_comparison": ("Unavailable by design: the two cohorts have no shared per-trial phase clock."),
}


def evaluate_emg_cohort_validation(
    *,
    simulation_npz: str | Path,
    emg_npz: str | Path,
    mapping_json: str | Path,
    expected_policy_checkpoint_fingerprint: str,
    expected_policy_promotion_fingerprint: str,
    expected_formal_synergy_basis_fingerprint: str,
    synergy_rank: int,
    initialization_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    pre_impact_s: float = 0.5,
    post_impact_s: float = 0.8,
    output_samples: int = 261,
    max_iter: int = 1000,
    tol: float = 1e-6,
    allow_provisional_mapping: bool = False,
) -> dict[str, Any]:
    """Evaluate independent simulation and measured-EMG action cohorts."""

    simulation_path = Path(simulation_npz)
    emg_path = Path(emg_npz)
    mapping_path = Path(mapping_json)
    mapping_raw = load_json_strict(mapping_path)
    if mapping_raw.get("schema_version") != EMG_OBSERVATION_MAPPING_SCHEMA_VERSION:
        raise ValueError("unpaired cohort evaluation requires emg_observation_mapping_v2")
    with np.load(simulation_path, allow_pickle=False) as source:
        simulation = {name: np.asarray(source[name]) for name in source.files}
    with np.load(emg_path, allow_pickle=False) as source:
        emg_data = {name: np.asarray(source[name]) for name in source.files}
    if missing := sorted(_SIMULATION_REQUIRED_FIELDS - set(simulation)):
        raise ValueError(f"simulation NPZ is missing cohort fields: {missing}")
    if missing := sorted(_EMG_REQUIRED_FIELDS - set(emg_data)):
        raise ValueError(f"EMG NPZ is missing strict-import/cohort fields: {missing}")
    if "reference_trial_fingerprint" in simulation or "reference_trial_fingerprint" in emg_data:
        raise ValueError("unpaired cohort inputs must not claim reference_trial_fingerprint pairing evidence")

    actuator_names = _string_vector(simulation["actuator_names"], "actuator_names")
    emg_names = _string_vector(emg_data["channel_names"], "channel_names")
    emg_values = _as_trial_time_channel(emg_data["emg"], field_name="emg")
    if emg_values.shape[2] != len(emg_names):
        raise ValueError("EMG array width does not match channel_names")
    mapping = validate_emg_mapping(
        mapping_raw,
        emg_channel_names=emg_names,
        actuator_names=actuator_names,
        allow_provisional_mapping=allow_provisional_mapping,
    )
    if mapping["validation_scope"] != WHOLE_BODY_15_OF_16_SCOPE:
        raise ValueError("unpaired cohort evaluation requires the 16-acquired/15-comparable scope")
    runtime_binding = _validate_unpaired_runtime_bindings(
        simulation,
        emg_data,
        mapping=mapping,
        emg_channel_names=emg_names,
    )
    preprocessing_contract = _validate_preprocessed_emg_contract(
        emg_data,
        mapping=mapping,
    )
    activation_contract = validate_simulation_activation_contract(
        simulation,
        actuator_names=actuator_names,
        mapping=mapping,
    )
    policy_evidence = validate_simulation_policy_evidence(
        simulation,
        expected_policy_checkpoint_fingerprint=expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=expected_formal_synergy_basis_fingerprint,
    )

    simulation_values = _as_trial_time_channel(
        simulation["muscle_activation"],
        field_name="muscle_activation",
    )
    cohort_contract = _validate_unpaired_cohort_contract(
        simulation,
        emg_data,
        simulation_trial_count=simulation_values.shape[0],
        emg_trial_count=emg_values.shape[0],
    )
    mapped_simulation, channel_names = map_simulation_activation(
        simulation_values,
        actuator_names=actuator_names,
        mapping=mapping,
        allow_provisional_mapping=allow_provisional_mapping,
    )
    emg_indices = [emg_names.index(name) for name in channel_names]
    comparable_emg = emg_values[:, :, emg_indices]
    if len(channel_names) != 15 or comparable_emg.shape[2] != 15:
        raise ValueError("unpaired cohort projection must produce exactly 15 channels")

    sim_fs = _scalar(simulation["sampling_rate_hz"], "simulation sampling_rate_hz")
    emg_fs = _scalar(emg_data["sampling_rate_hz"], "EMG sampling_rate_hz")
    aligned_simulation, simulation_time = impact_aligned_resample(
        mapped_simulation,
        simulation["impact_frame"],
        sampling_rate_hz=sim_fs,
        pre_impact_s=pre_impact_s,
        post_impact_s=post_impact_s,
        output_samples=output_samples,
    )
    aligned_emg, emg_time = impact_aligned_resample(
        comparable_emg,
        emg_data["impact_frame"],
        sampling_rate_hz=emg_fs,
        pre_impact_s=pre_impact_s,
        post_impact_s=post_impact_s,
        output_samples=output_samples,
    )
    np.testing.assert_allclose(simulation_time, emg_time, rtol=0.0, atol=1e-12)

    rank = _positive_integer(synergy_rank, "synergy_rank")
    if rank > len(channel_names):
        raise ValueError("synergy_rank cannot exceed the 15-channel comparison width")
    seeds = _initialization_seed_contract(initialization_seeds)
    iterations = _positive_integer(max_iter, "max_iter")
    tolerance = _nonnegative_finite_float(tol, "tol")
    simulation_matrix = aligned_simulation.reshape(-1, aligned_simulation.shape[2])
    emg_matrix = aligned_emg.reshape(-1, aligned_emg.shape[2])
    simulation_nmf = _fit_cohort_nmf(
        simulation_matrix,
        channel_names=channel_names,
        rank=rank,
        seeds=seeds,
        max_iter=iterations,
        tol=tolerance,
    )
    emg_nmf = _fit_cohort_nmf(
        emg_matrix,
        channel_names=channel_names,
        rank=rank,
        seeds=seeds,
        max_iter=iterations,
        tol=tolerance,
    )
    basis_similarity = match_synergy_bases(
        np.asarray(simulation_nmf["basis"], dtype=np.float64),
        np.asarray(emg_nmf["basis"], dtype=np.float64),
    )

    emg_subject_count = int(cohort_contract["emg_subject_count"])
    population_reason = (
        "The measured cohort contains one subject; trials are repeated measures and cannot provide a population CI."
        if emg_subject_count == 1
        else "No preregistered independent-subject resampling procedure was supplied; metrics remain descriptive."
    )
    return {
        "schema_version": EMG_COHORT_REPORT_SCHEMA_VERSION,
        "comparison_design": UNPAIRED_COMPARISON_DESIGN,
        "claim_scope": mapping["validation_scope"],
        "claim_limitations": list(COHORT_CLAIM_LIMITATIONS),
        "exploratory_only": bool(mapping.get("exploratory_only", False)),
        "cohort_contract": cohort_contract,
        "activation_contract": activation_contract,
        "policy_evidence": policy_evidence,
        "mapping_runtime_binding": runtime_binding,
        "mapping": mapping,
        "impact_alignment": {
            "mode": "independent_measured_impact_frame_per_cohort",
            "pairing_performed": False,
            "pre_impact_s": float(pre_impact_s),
            "post_impact_s": float(post_impact_s),
            "output_samples": int(output_samples),
            "simulation_sampling_rate_hz": sim_fs,
            "emg_sampling_rate_hz": emg_fs,
        },
        "preprocessing": {
            "emg_signal_kind": PREPROCESSED_NORMALIZED_ENVELOPE_KIND,
            "evaluator_filter_applied": False,
            "evaluator_emg_normalization_applied": False,
            "simulation_normalization": "none_unit_mujoco_activation",
            "measurement_processing_contract": preprocessing_contract,
        },
        "rank_contract": {
            "source": "explicit_pre_registered_argument",
            "common_rank": rank,
            "initialization_seeds": seeds,
            "selection_rule": "minimum_reconstruction_mse_then_lowest_seed_order",
            "max_iter": iterations,
            "convergence_tolerance": tolerance,
        },
        "nmf": {
            "matrix_orientation": "sample_by_mapped_channel",
            "simulation": simulation_nmf,
            "emg": emg_nmf,
            "hungarian_basis_similarity": basis_similarity,
        },
        "metric_availability": {
            name: {
                "status": "unavailable",
                "available": False,
                "reason": reason,
            }
            for name, reason in _UNAVAILABLE_PAIRED_METRICS.items()
        },
        "uncertainty": {
            "confidence_intervals_computed": False,
            "population_inference_available": False,
            "single_measured_subject_limitation": emg_subject_count == 1,
            "reason": population_reason,
        },
        "input_fingerprints": {
            "simulation_npz_sha256": _file_sha256(simulation_path),
            "emg_npz_sha256": _file_sha256(emg_path),
            "mapping_json_sha256": _file_sha256(mapping_path),
        },
    }


def _validate_unpaired_runtime_bindings(
    simulation: Mapping[str, np.ndarray],
    emg: Mapping[str, np.ndarray],
    *,
    mapping: Mapping[str, Any],
    emg_channel_names: Sequence[str],
) -> dict[str, Any]:
    import_schema = _identity_scalar(emg["import_schema_version"], "import_schema_version")
    if import_schema != JIDIAN_IMPORT_SCHEMA_VERSION:
        raise ValueError(f"EMG cohort input must come from strict importer {JIDIAN_IMPORT_SCHEMA_VERSION!r}")
    selection_manifest_sha256 = _sha256_scalar(
        emg["selection_manifest_sha256"],
        "selection_manifest_sha256",
    )
    profile = mapping["profile_binding"]
    actual_profile = {
        "profile_id": _identity_scalar(emg["channel_profile_id"], "channel_profile_id"),
        "profile_version": _integer_scalar(
            emg["channel_profile_version"],
            "channel_profile_version",
            minimum=1,
        ),
        "profile_sha256": _sha256_scalar(
            emg["channel_profile_sha256"],
            "channel_profile_sha256",
        ),
        "handedness": _identity_scalar(emg["handedness"], "EMG handedness"),
    }
    expected_profile = {
        "profile_id": str(profile["profile_id"]).lower(),
        "profile_version": int(profile["profile_version"]),
        "profile_sha256": str(profile["profile_sha256"]),
        "handedness": str(profile["intended_handedness"]),
    }
    if actual_profile != expected_profile:
        raise ValueError("EMG profile/handedness differs from the v2 mapping binding")
    if _identity_scalar(simulation["handedness"], "simulation handedness") != expected_profile["handedness"]:
        raise ValueError("simulation handedness differs from the v2 mapping binding")

    model = mapping["model_binding"]
    actual_model = {
        "taxonomy_id": _identity_scalar(
            simulation["model_taxonomy_id"],
            "model_taxonomy_id",
        ),
        "taxonomy_fingerprint": _sha256_scalar(
            simulation["model_taxonomy_fingerprint"],
            "model_taxonomy_fingerprint",
        ),
        "runtime_model_hash": _sha256_scalar(
            simulation["runtime_model_hash"],
            "runtime_model_hash",
        ),
        "actuator_schema_hash": _sha256_scalar(
            simulation["actuator_schema_hash"],
            "actuator_schema_hash",
        ),
    }
    expected_model = {
        "taxonomy_id": str(model["taxonomy_id"]).lower(),
        "taxonomy_fingerprint": str(model["taxonomy_fingerprint"]),
        "runtime_model_hash": str(model["runtime_model_hash"]),
        "actuator_schema_hash": str(model["actuator_schema_hash"]),
    }
    if actual_model != expected_model:
        raise ValueError("simulation model/taxonomy differs from the v2 mapping binding")

    acquired_count = _integer_scalar(
        emg["acquired_channel_count"],
        "acquired_channel_count",
        minimum=1,
    )
    comparable_count = _integer_scalar(
        emg["comparable_channel_count"],
        "comparable_channel_count",
        minimum=1,
    )
    excluded = np.asarray(emg["excluded_sensor_ids"])
    if acquired_count != 16 or comparable_count != 15:
        raise ValueError("strict EMG import must declare exactly 16 acquired and 15 comparable channels")
    if excluded.shape != (1,) or not np.issubdtype(excluded.dtype, np.integer) or excluded.astype(int).tolist() != [1]:
        raise ValueError("strict EMG import must explicitly exclude only sensor 1")
    sensor_ids = np.asarray(emg["stream_channel_ids"])
    if (
        sensor_ids.shape != (16,)
        or not np.issubdtype(sensor_ids.dtype, np.integer)
        or sensor_ids.astype(int).tolist() != list(range(1, 17))
    ):
        raise ValueError("EMG stream_channel_ids must be exact ordered integers 1..16")
    sides = _nonempty_string_array(emg["sides"], "sides", expected=16)
    muscle_slugs = _nonempty_string_array(emg["muscle_slugs"], "muscle_slugs", expected=16)
    expected_channels = mapping["channels"]
    if list(emg_channel_names) != [entry["emg_channel"] for entry in expected_channels]:
        raise ValueError("EMG channel_names differ from the mapping profile order")
    if sides != [entry["side"] for entry in expected_channels]:
        raise ValueError("EMG sides differ from the mapping profile order")
    if muscle_slugs != [entry["muscle_slug"] for entry in expected_channels]:
        raise ValueError("EMG muscle_slugs differ from the mapping profile order")
    return {
        "strict_import_verified": 1.0,
        "import_schema_version": import_schema,
        "selection_manifest_sha256": selection_manifest_sha256,
        "profile_binding_verified": 1.0,
        "model_binding_verified": 1.0,
        "acquired_channel_count": acquired_count,
        "comparable_channel_count": comparable_count,
        "excluded_sensor_ids": [1],
        "profile": actual_profile,
        "model": actual_model,
    }


def _validate_unpaired_cohort_contract(
    simulation: Mapping[str, np.ndarray],
    emg: Mapping[str, np.ndarray],
    *,
    simulation_trial_count: int,
    emg_trial_count: int,
) -> dict[str, Any]:
    simulation_design = _identity_scalar(
        simulation["comparison_design"],
        "simulation comparison_design",
    )
    emg_design = _identity_scalar(emg["comparison_design"], "EMG comparison_design")
    if simulation_design != UNPAIRED_COMPARISON_DESIGN or emg_design != UNPAIRED_COMPARISON_DESIGN:
        raise ValueError(f"both cohort inputs must declare comparison_design={UNPAIRED_COMPARISON_DESIGN!r}")
    scalar_bindings = {}
    for field in ("action_id", "comparison_set_uid"):
        simulation_value = _exact_identity_scalar(simulation[field], f"simulation {field}")
        emg_value = _exact_identity_scalar(emg[field], f"EMG {field}")
        if simulation_value != emg_value:
            raise ValueError(f"simulation/EMG cohort {field} values differ")
        scalar_bindings[field] = simulation_value
    for field in ("handedness", "dataset_split"):
        simulation_value = _identity_scalar(simulation[field], f"simulation {field}")
        emg_value = _identity_scalar(emg[field], f"EMG {field}")
        if simulation_value != emg_value:
            raise ValueError(f"simulation/EMG cohort {field} values differ")
        scalar_bindings[field] = simulation_value
    if scalar_bindings["dataset_split"] not in {"heldout", "validation", "test"}:
        raise ValueError("unpaired cohort dataset_split must be heldout, validation, or test")
    if scalar_bindings["handedness"] not in {"right", "left"}:
        raise ValueError("unpaired cohort handedness must be explicitly right or left")

    sim_trials = _identity_vector(
        simulation["trial_uid"],
        "simulation trial_uid",
        expected=simulation_trial_count,
    )
    emg_trials = _identity_vector(
        emg["trial_uid"],
        "EMG trial_uid",
        expected=emg_trial_count,
    )
    sim_subjects = _identity_vector(
        simulation["subject_uid"],
        "simulation subject_uid",
        expected=simulation_trial_count,
    )
    emg_subjects = _identity_vector(
        emg["subject_uid"],
        "EMG subject_uid",
        expected=emg_trial_count,
    )
    sim_sessions = _identity_vector(
        simulation["session_uid"],
        "simulation session_uid",
        expected=simulation_trial_count,
    )
    emg_sessions = _identity_vector(
        emg["session_uid"],
        "EMG session_uid",
        expected=emg_trial_count,
    )
    sim_training = _identity_vector(
        simulation["training_session_uid"],
        "simulation training_session_uid",
    )
    emg_training = _identity_vector(
        emg["training_session_uid"],
        "EMG training_session_uid",
    )
    if len(set(sim_training)) != len(sim_training) or len(set(emg_training)) != len(emg_training):
        raise ValueError("training_session_uid inventories must be unique")
    if set(sim_training) != set(emg_training):
        raise ValueError("simulation/EMG training_session_uid inventories differ")
    leakage = sorted((set(sim_sessions) | set(emg_sessions)) & set(sim_training))
    if leakage:
        raise ValueError(f"unpaired held-out cohort leaks policy training sessions: {leakage}")
    return {
        "binding_verified": 1.0,
        "pairing_performed": False,
        "trial_uid_used_for_pairing": False,
        **scalar_bindings,
        "simulation_trial_count": simulation_trial_count,
        "emg_trial_count": emg_trial_count,
        "different_trial_counts_allowed": True,
        "simulation_subject_count": len(set(sim_subjects)),
        "emg_subject_count": len(set(emg_subjects)),
        "simulation_session_count": len(set(sim_sessions)),
        "emg_session_count": len(set(emg_sessions)),
        "trial_uid_overlap_count": len(set(sim_trials) & set(emg_trials)),
        "training_session_uids": sorted(set(sim_training)),
    }


def _fit_cohort_nmf(
    matrix: np.ndarray,
    *,
    channel_names: Sequence[str],
    rank: int,
    seeds: Sequence[int],
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    best, initializations = fit_best_initialization(
        matrix,
        rank=rank,
        seeds=seeds,
        max_iter=max_iter,
        tol=tol,
    )
    global_score = global_vaf(matrix, best.reconstruction)
    local_scores = local_vaf(matrix, best.reconstruction)
    condition = basis_condition_number(best.basis)
    return {
        "trial_samples": int(matrix.shape[0]),
        "channel_count": int(matrix.shape[1]),
        "rank": rank,
        "best_seed": int(best.seed),
        "best_reconstruction_mse": float(best.loss),
        "best_n_iter": int(best.n_iter),
        "global_vaf": float(global_score),
        "per_channel_vaf": {
            name: None if not np.isfinite(local_scores[index]) else float(local_scores[index])
            for index, name in enumerate(channel_names)
        },
        "reconstruction_rmse": reconstruction_rmse(matrix, best.reconstruction),
        "basis_condition_number": None if not np.isfinite(condition) else float(condition),
        "basis_condition_number_nonfinite": not np.isfinite(condition),
        "basis": best.basis.tolist(),
        "initializations": [
            {
                "seed": int(result.seed),
                "reconstruction_mse": float(result.loss),
                "n_iter": int(result.n_iter),
            }
            for result in initializations
        ],
        "vaf_definition": "one_minus_sse_over_zero_baseline_sum_squares",
    }


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{field_name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _initialization_seed_contract(values: Sequence[int]) -> list[int]:
    seeds = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError("initialization_seeds must contain only non-negative integers")
        seed = int(value)
        if seed < 0:
            raise ValueError("initialization_seeds must contain only non-negative integers")
        seeds.append(seed)
    seeds = sorted(set(seeds))
    if len(seeds) < 2:
        raise ValueError("unpaired cohort NMF requires at least two distinct initialization seeds")
    return seeds


def _nonnegative_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-npz")
    parser.add_argument("--emg-npz")
    parser.add_argument("--mapping-json")
    parser.add_argument("--output-json")
    parser.add_argument("--policy-evidence-json")
    parser.add_argument("--expected-policy-checkpoint-fingerprint")
    parser.add_argument("--expected-policy-promotion-fingerprint")
    parser.add_argument("--expected-formal-synergy-basis-fingerprint")
    parser.add_argument("--synergy-rank", type=int)
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pre-impact-s", type=float, default=0.5)
    parser.add_argument("--post-impact-s", type=float, default=0.8)
    parser.add_argument("--output-samples", type=int, default=261)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--allow-provisional-mapping", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": EMG_COHORT_REPORT_SCHEMA_VERSION,
                    "comparison_design": UNPAIRED_COMPARISON_DESIGN,
                    "mapping_schema_version": EMG_OBSERVATION_MAPPING_SCHEMA_VERSION,
                    "emg_signal_kind": PREPROCESSED_NORMALIZED_ENVELOPE_KIND,
                    "required_simulation_fields": sorted(_SIMULATION_REQUIRED_FIELDS),
                    "required_emg_fields": sorted(_EMG_REQUIRED_FIELDS),
                    "unavailable_paired_metrics": sorted(_UNAVAILABLE_PAIRED_METRICS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    required = {
        "--simulation-npz": args.simulation_npz,
        "--emg-npz": args.emg_npz,
        "--mapping-json": args.mapping_json,
        "--output-json": args.output_json,
        "--synergy-rank": args.synergy_rank,
    }
    if missing := [flag for flag, value in required.items() if value is None]:
        raise SystemExit(f"missing required arguments outside --dry-run: {', '.join(missing)}")
    expected_policy = _resolve_expected_policy_bindings(
        policy_evidence_json=args.policy_evidence_json,
        expected_policy_checkpoint_fingerprint=args.expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=args.expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=args.expected_formal_synergy_basis_fingerprint,
    )
    report = evaluate_emg_cohort_validation(
        simulation_npz=args.simulation_npz,
        emg_npz=args.emg_npz,
        mapping_json=args.mapping_json,
        expected_policy_checkpoint_fingerprint=expected_policy["policy_checkpoint_fingerprint"],
        expected_policy_promotion_fingerprint=expected_policy["policy_promotion_fingerprint"],
        expected_formal_synergy_basis_fingerprint=expected_policy["formal_synergy_basis_fingerprint"],
        synergy_rank=args.synergy_rank,
        initialization_seeds=args.initialization_seeds,
        pre_impact_s=args.pre_impact_s,
        post_impact_s=args.post_impact_s,
        output_samples=args.output_samples,
        max_iter=args.max_iter,
        tol=args.tol,
        allow_provisional_mapping=args.allow_provisional_mapping,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
