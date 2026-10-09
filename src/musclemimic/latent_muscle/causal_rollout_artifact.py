"""Seal paired fixed-state environment rollouts as causal latent evidence.

This command does not run a simulator and never synthesizes outcomes.  It
accepts baseline and perturbed records produced by an external rollout driver,
verifies exact sample/direction/epsilon pairing and common-random-number state
restoration, then stores only measured outcome deltas in the analysis artifact.

Example::

    python -m musclemimic.latent_muscle.causal_rollout_artifact \
      --analysis-inputs run/analysis_inputs.npz \
      --analysis-manifest run/analysis_inputs.json \
      --baseline-records run/rollouts/baseline.npz \
      --perturbed-records run/rollouts/perturbed.npz \
      --rollout-manifest run/rollouts/paired_rollout_manifest.json \
      --output-npz run/causal_interventions.npz
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.physical import (
    physical_signal_metadata,
    validate_activation_valid_mask,
    validate_physical_signal_semantics,
)
from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
from musclemimic.latent_muscle.analysis_export import (
    ANALYSIS_INPUT_SCHEMA_VERSION,
    CAUSAL_EVIDENCE_SCHEMA_VERSION,
)

PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION = "latent_synergy_paired_rollout_source_v1"
CAUSAL_ROLLOUT_PRODUCER_SCHEMA_VERSION = "latent_synergy_causal_rollout_builder_v2"
REQUIRED_OUTCOMES = (
    "muscle_excitation",
    "muscle_activation",
    "joint_position",
    "joint_velocity",
    "trunk_state",
    "racket_state",
    "impact_outcome",
    "landing_outcome",
)
_HEX = frozenset("0123456789abcdef")


def build_causal_rollout_artifact(
    *,
    analysis_inputs: str | Path,
    analysis_manifest: str | Path,
    baseline_records: str | Path,
    perturbed_records: str | Path,
    rollout_manifest: str | Path,
    output_npz: str | Path,
    output_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and seal real paired rollout records; never execute rollouts."""

    analysis_path = Path(analysis_inputs)
    analysis_manifest_path = Path(analysis_manifest)
    baseline_path = Path(baseline_records)
    perturbed_path = Path(perturbed_records)
    source_manifest_path = Path(rollout_manifest)
    for label, path in (
        ("analysis inputs", analysis_path),
        ("analysis manifest", analysis_manifest_path),
        ("baseline rollout records", baseline_path),
        ("perturbed rollout records", perturbed_path),
        ("paired rollout source manifest", source_manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    analysis_sidecar = _load_self_fingerprinted_json(
        analysis_manifest_path,
        fingerprint_field="manifest_fingerprint",
    )
    if analysis_sidecar.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        raise ValueError("causal rollout builder requires analysis_inputs_v2")
    if analysis_sidecar.get("npz_sha256") != file_sha256(analysis_path):
        raise ValueError("analysis input NPZ differs from its sidecar")
    source_manifest = _load_self_fingerprinted_json(
        source_manifest_path,
        fingerprint_field="manifest_fingerprint",
    )
    _validate_source_manifest(
        source_manifest,
        baseline_path=baseline_path,
        perturbed_path=perturbed_path,
        analysis_sidecar=analysis_sidecar,
    )

    with np.load(analysis_path, allow_pickle=False) as analysis:
        required = {
            "sample_uids",
            "intervention_directions",
            "intervention_epsilons",
        }
        missing = sorted(required - set(analysis.files))
        if missing:
            raise ValueError(f"analysis inputs are missing causal bindings: {missing}")
        sample_uids = np.asarray(analysis["sample_uids"]).astype(str)
        directions = np.asarray(analysis["intervention_directions"], dtype=np.float64)
        epsilons = np.asarray(analysis["intervention_epsilons"], dtype=np.float64)
    if (
        sample_uids.ndim != 1
        or len(set(sample_uids.tolist())) != len(sample_uids)
        or directions.ndim != 2
        or directions.shape[0] <= 0
        or epsilons.ndim != 1
        or epsilons.shape[0] <= 0
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(epsilons))
        or np.any(epsilons == 0.0)
    ):
        raise ValueError("analysis sample/direction/epsilon bindings are malformed")

    with np.load(baseline_path, allow_pickle=False) as raw_baseline:
        baseline = {name: np.asarray(raw_baseline[name]) for name in raw_baseline.files}
    with np.load(perturbed_path, allow_pickle=False) as raw_perturbed:
        perturbed = {name: np.asarray(raw_perturbed[name]) for name in raw_perturbed.files}
    outcome_availability = _validated_outcome_availability(source_manifest.get("outcome_availability"))
    stage2_diagnostic_complete, task_outcomes_complete = _outcome_completeness(outcome_availability)
    _validate_pairing(
        baseline,
        perturbed,
        sample_uids=sample_uids,
        directions=directions,
        epsilons=epsilons,
        outcome_availability=outcome_availability,
    )
    outcome_schemas = _validated_outcome_schemas(
        source_manifest.get("outcome_schemas"),
        baseline=baseline,
        activation_valid_mask=source_manifest.get("activation_valid_mask"),
        outcome_availability=source_manifest.get("outcome_availability"),
    )
    _validate_masked_value_records(
        baseline,
        perturbed,
        outcome_schemas=outcome_schemas,
        outcome_availability=outcome_availability,
    )

    effect_parts: list[np.ndarray] = []
    effect_names: list[str] = []
    outcome_layout: dict[str, Any] = {}
    offset = 0
    for name in REQUIRED_OUTCOMES:
        base = np.asarray(baseline[name], dtype=np.float64)
        changed = np.asarray(perturbed[name], dtype=np.float64)
        trailing_shape = tuple(int(value) for value in base.shape[1:])
        width = int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
        feature_names = outcome_schemas[name]["feature_names"]
        excluded = _excluded_masked_value_features(outcome_schemas[name])
        included_indices = [index for index, feature in enumerate(feature_names) if feature not in excluded]
        names = [f"{name}:{feature_names[index]}" for index in included_indices]
        available = outcome_availability[name]
        if available and included_indices:
            base_flat = base.reshape((base.shape[0], width))
            changed_flat = changed.reshape((*changed.shape[:3], width))
            effect_parts.append(changed_flat[..., included_indices] - base_flat[:, None, None, included_indices])
            effect_names.extend(names)
        outcome_layout[name] = {
            "source_shape_per_sample": list(trailing_shape),
            "flat_start": offset,
            "flat_stop": offset + (len(included_indices) if available else 0),
            "effect_names": names if available else [],
            "excluded_masked_value_features": excluded if available else [],
            "available": available,
            "schema": outcome_schemas[name],
        }
        if available:
            offset += len(included_indices)
    if not effect_parts:
        raise ValueError("causal rollout contains no available measured outcomes")
    causal_effects = np.concatenate(effect_parts, axis=-1)
    if not np.all(np.isfinite(causal_effects)):
        raise ValueError("paired rollout deltas contain non-finite values")

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed_directions = directions.astype(np.float32)
    sealed_epsilons = epsilons.astype(np.float32)
    np.savez_compressed(
        output,
        causal_effects=causal_effects.astype(np.float32),
        causal_effect_names=np.asarray(effect_names, dtype=np.str_),
        sample_uids=np.asarray(sample_uids, dtype=np.str_),
        intervention_directions=sealed_directions,
        intervention_epsilons=sealed_epsilons,
    )
    manifest_path = Path(output_manifest) if output_manifest is not None else output.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": CAUSAL_EVIDENCE_SCHEMA_VERSION,
        "producer_schema_version": CAUSAL_ROLLOUT_PRODUCER_SCHEMA_VERSION,
        "evidence_kind": "environment_rollout",
        "npz_path": str(output.resolve()),
        "npz_sha256": file_sha256(output),
        "checkpoint_fingerprint": analysis_sidecar["checkpoint_fingerprint"],
        "synergy_basis_fingerprint": analysis_sidecar["formal_synergy_basis_fingerprint"],
        "analysis_inputs_sha256": file_sha256(analysis_path),
        "analysis_manifest_fingerprint": analysis_sidecar["manifest_fingerprint"],
        "sample_uid_fingerprint": canonical_json_sha256(sample_uids.tolist()),
        "intervention_direction_fingerprint": canonical_json_sha256(sealed_directions.astype(np.float64).tolist()),
        "intervention_epsilon_fingerprint": canonical_json_sha256(sealed_epsilons.astype(np.float64).tolist()),
        "paired_rollout_source_manifest_path": str(source_manifest_path.resolve()),
        "paired_rollout_source_manifest_fingerprint": source_manifest["manifest_fingerprint"],
        "baseline_records_sha256": file_sha256(baseline_path),
        "perturbed_records_sha256": file_sha256(perturbed_path),
        "environment_fingerprint": source_manifest["environment_fingerprint"],
        "policy_abi_hash": source_manifest["policy_abi_hash"],
        "rollout_engine": source_manifest["rollout_engine"],
        "fixed_state_initialization": "exact_snapshot_restore",
        "common_random_numbers": True,
        "physical_signal_semantics": source_manifest["physical_signal_semantics"],
        "physical_signal_semantics_fingerprint": canonical_json_sha256(source_manifest["physical_signal_semantics"]),
        "activation_valid_mask": source_manifest["activation_valid_mask"],
        "activation_valid_mask_fingerprint": canonical_json_sha256(source_manifest["activation_valid_mask"]),
        "outcome_schemas": outcome_schemas,
        "outcome_availability": outcome_availability,
        "stage2_diagnostic_outcomes_complete": stage2_diagnostic_complete,
        "task_outcomes_complete": task_outcomes_complete,
        "outcome_schemas_fingerprint": canonical_json_sha256(outcome_schemas),
        "required_outcomes": list(REQUIRED_OUTCOMES),
        "outcome_layout": outcome_layout,
        "num_samples": int(sample_uids.shape[0]),
        "num_directions": int(directions.shape[0]),
        "num_epsilons": int(epsilons.shape[0]),
        "num_flat_outcomes": int(causal_effects.shape[-1]),
        "limitations": (
            "This artifact seals supplied paired simulator records; it does not run "
            "the environment or independently prove that producer signal labels are correct."
        ),
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_causal_rollout_artifact(output, manifest_path)
    return manifest


def validate_causal_rollout_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate that a causal artifact came through the strict paired builder."""

    source = Path(npz_path)
    manifest_file = Path(manifest_path)
    if not source.is_file() or not manifest_file.is_file():
        raise ValueError("causal rollout artifact or manifest is missing")
    manifest = _load_self_fingerprinted_json(
        manifest_file,
        fingerprint_field="manifest_fingerprint",
    )
    if manifest.get("evidence_kind") != "environment_rollout":
        raise ValueError("causal effects must come from environment_rollout evidence")
    if (
        manifest.get("schema_version") != CAUSAL_EVIDENCE_SCHEMA_VERSION
        or manifest.get("producer_schema_version") != CAUSAL_ROLLOUT_PRODUCER_SCHEMA_VERSION
    ):
        raise ValueError("causal artifact was not produced by the paired rollout builder")
    if manifest.get("npz_sha256") != file_sha256(source):
        raise ValueError("causal rollout NPZ hash mismatch")
    for key in (
        "checkpoint_fingerprint",
        "synergy_basis_fingerprint",
        "analysis_inputs_sha256",
        "analysis_manifest_fingerprint",
        "paired_rollout_source_manifest_fingerprint",
        "baseline_records_sha256",
        "perturbed_records_sha256",
        "environment_fingerprint",
        "policy_abi_hash",
    ):
        _require_hex64(key, manifest.get(key))
    if (
        manifest.get("fixed_state_initialization") != "exact_snapshot_restore"
        or manifest.get("common_random_numbers") is not True
        or manifest.get("required_outcomes") != list(REQUIRED_OUTCOMES)
    ):
        raise ValueError("causal rollout artifact lacks exact fixed-state paired evidence")
    semantics = validate_physical_signal_semantics(manifest.get("physical_signal_semantics"))
    if semantics != physical_signal_metadata() or canonical_json_sha256(semantics) != manifest.get(
        "physical_signal_semantics_fingerprint"
    ):
        raise ValueError("causal rollout physical signal semantics are not exact")
    outcome_schemas = manifest.get("outcome_schemas")
    outcome_availability = _validated_outcome_availability(manifest.get("outcome_availability"))
    diagnostic_complete, task_complete = _outcome_completeness(outcome_availability)
    if (
        manifest.get("stage2_diagnostic_outcomes_complete") is not diagnostic_complete
        or manifest.get("task_outcomes_complete") is not task_complete
    ):
        raise ValueError("causal rollout outcome-completeness flags are inconsistent")
    if (
        not isinstance(outcome_schemas, dict)
        or set(outcome_schemas) != set(REQUIRED_OUTCOMES)
        or canonical_json_sha256(outcome_schemas) != manifest.get("outcome_schemas_fingerprint")
    ):
        raise ValueError("causal rollout outcome schemas are absent or changed")
    muscle_width = len(outcome_schemas["muscle_activation"].get("feature_names", []))
    activation_mask = validate_activation_valid_mask(manifest.get("activation_valid_mask"), expected_width=muscle_width)
    if canonical_json_sha256(activation_mask.tolist()) != manifest.get("activation_valid_mask_fingerprint"):
        raise ValueError("causal rollout activation-valid mask fingerprint mismatch")
    dummy_baseline: dict[str, np.ndarray] = {}
    for outcome in REQUIRED_OUTCOMES:
        shape = outcome_schemas[outcome].get("source_shape_per_sample")
        if not isinstance(shape, list) or any(not isinstance(item, int) or item < 0 for item in shape):
            if shape != []:
                raise ValueError(f"causal outcome schema {outcome!r} has an invalid shape")
        if outcome_availability[outcome] != (int(np.prod(shape, dtype=np.int64)) > 0):
            raise ValueError(f"causal outcome schema {outcome!r} availability/shape mismatch")
        dummy_baseline[outcome] = np.zeros((1, *shape), dtype=np.float32)
    canonical_schemas = _validated_outcome_schemas(
        outcome_schemas,
        baseline=dummy_baseline,
        activation_valid_mask=activation_mask,
        outcome_availability=outcome_availability,
    )
    if canonical_schemas != outcome_schemas:
        raise ValueError("causal rollout outcome schemas are not canonical")
    expected_layout = _expected_outcome_layout(
        outcome_schemas,
        outcome_availability=outcome_availability,
    )
    if manifest.get("outcome_layout") != expected_layout:
        raise ValueError("causal rollout outcome_layout differs from the mask-safe ordered schemas")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "causal_effects",
            "causal_effect_names",
            "sample_uids",
            "intervention_directions",
            "intervention_epsilons",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"causal rollout NPZ is missing {missing}")
        effects = np.asarray(data["causal_effects"])
        names = np.asarray(data["causal_effect_names"]).astype(str)
        uids = np.asarray(data["sample_uids"]).astype(str)
        directions = np.asarray(data["intervention_directions"])
        epsilons = np.asarray(data["intervention_epsilons"])
    expected_shape = (
        int(manifest["num_samples"]),
        int(manifest["num_directions"]),
        int(manifest["num_epsilons"]),
        int(manifest["num_flat_outcomes"]),
    )
    if effects.shape != expected_shape or not np.all(np.isfinite(effects)):
        raise ValueError("causal rollout effect tensor differs from its manifest")
    if names.shape != (expected_shape[-1],) or len(set(names.tolist())) != len(names):
        raise ValueError("causal rollout effect names are incomplete or duplicated")
    expected_names = [
        effect_name for outcome in REQUIRED_OUTCOMES for effect_name in expected_layout[outcome]["effect_names"]
    ]
    if names.tolist() != expected_names:
        raise ValueError("causal rollout effect names differ from ordered outcome schemas")
    if canonical_json_sha256(uids.tolist()) != manifest.get("sample_uid_fingerprint"):
        raise ValueError("causal rollout sample UID fingerprint mismatch")
    if canonical_json_sha256(np.asarray(directions, dtype=np.float64).tolist()) != manifest.get(
        "intervention_direction_fingerprint"
    ):
        raise ValueError("causal rollout direction fingerprint mismatch")
    if canonical_json_sha256(np.asarray(epsilons, dtype=np.float64).tolist()) != manifest.get(
        "intervention_epsilon_fingerprint"
    ):
        raise ValueError("causal rollout epsilon fingerprint mismatch")
    return manifest


def _validate_source_manifest(
    manifest: dict[str, Any],
    *,
    baseline_path: Path,
    perturbed_path: Path,
    analysis_sidecar: dict[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION
        or manifest.get("evidence_kind") != "environment_rollout"
    ):
        raise ValueError("paired source manifest is not environment rollout evidence")
    expected = {
        "checkpoint_fingerprint": analysis_sidecar.get("checkpoint_fingerprint"),
        "synergy_basis_fingerprint": analysis_sidecar.get("formal_synergy_basis_fingerprint"),
        "analysis_manifest_fingerprint": analysis_sidecar.get("manifest_fingerprint"),
        "baseline_records_sha256": file_sha256(baseline_path),
        "perturbed_records_sha256": file_sha256(perturbed_path),
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(f"paired rollout source binding mismatch: {mismatch}")
    if (
        manifest.get("fixed_state_initialization") != "exact_snapshot_restore"
        or manifest.get("common_random_numbers") is not True
        or not str(manifest.get("rollout_engine", "")).strip()
    ):
        raise ValueError("paired rollouts require snapshot restore and common random numbers")
    semantics = validate_physical_signal_semantics(manifest.get("physical_signal_semantics"))
    if semantics != physical_signal_metadata():
        raise ValueError("paired rollout source must declare the exact excitation/activation semantics")
    if not isinstance(manifest.get("outcome_schemas"), dict):
        raise ValueError("paired rollout source lacks ordered outcome_schemas")
    availability = _validated_outcome_availability(manifest.get("outcome_availability"))
    diagnostic_complete, task_complete = _outcome_completeness(availability)
    if (
        manifest.get("stage2_diagnostic_outcomes_complete") is not diagnostic_complete
        or manifest.get("task_outcomes_complete") is not task_complete
    ):
        raise ValueError("paired rollout outcome-completeness flags do not match availability")
    for key in (
        "checkpoint_fingerprint",
        "synergy_basis_fingerprint",
        "analysis_manifest_fingerprint",
        "baseline_records_sha256",
        "perturbed_records_sha256",
        "environment_fingerprint",
        "policy_abi_hash",
    ):
        _require_hex64(key, manifest.get(key))


def _validate_pairing(
    baseline: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    *,
    sample_uids: np.ndarray,
    directions: np.ndarray,
    epsilons: np.ndarray,
    outcome_availability: dict[str, bool],
) -> None:
    required_common = {"sample_uids", "initial_state_fingerprints", "rollout_seeds"}
    missing_base = sorted((required_common | set(REQUIRED_OUTCOMES)) - set(baseline))
    missing_changed = sorted(
        (required_common | set(REQUIRED_OUTCOMES) | {"intervention_directions", "intervention_epsilons"})
        - set(perturbed)
    )
    if missing_base or missing_changed:
        raise ValueError(f"paired rollout records are incomplete: baseline={missing_base}, perturbed={missing_changed}")
    n, d, e = len(sample_uids), directions.shape[0], len(epsilons)
    baseline_uids = np.asarray(baseline["sample_uids"]).astype(str)
    changed_uids = np.asarray(perturbed["sample_uids"]).astype(str)
    if not np.array_equal(baseline_uids, sample_uids) or not np.array_equal(changed_uids, sample_uids):
        raise ValueError("paired rollout sample_uids differ from analysis inputs")
    if not np.array_equal(
        np.asarray(perturbed["intervention_directions"], dtype=np.float64),
        directions,
    ) or not np.array_equal(
        np.asarray(perturbed["intervention_epsilons"], dtype=np.float64),
        epsilons,
    ):
        raise ValueError("paired rollout directions or epsilons are not exact analysis bindings")
    base_state = np.asarray(baseline["initial_state_fingerprints"]).astype(str)
    changed_state = np.asarray(perturbed["initial_state_fingerprints"]).astype(str)
    if base_state.shape != (n,) or changed_state.shape != (n, d, e):
        raise ValueError("paired rollout initial-state fingerprint shapes are invalid")
    for value in base_state.tolist():
        _require_hex64("initial_state_fingerprint", value)
    if not np.array_equal(changed_state, np.broadcast_to(base_state[:, None, None], (n, d, e))):
        raise ValueError("perturbed rollouts did not restore the exact baseline state")
    base_seed = np.asarray(baseline["rollout_seeds"])
    changed_seed = np.asarray(perturbed["rollout_seeds"])
    if (
        base_seed.shape != (n,)
        or changed_seed.shape != (n, d, e)
        or not np.issubdtype(base_seed.dtype, np.integer)
        or not np.issubdtype(changed_seed.dtype, np.integer)
        or np.any(base_seed < 0)
        or not np.array_equal(changed_seed, np.broadcast_to(base_seed[:, None, None], (n, d, e)))
    ):
        raise ValueError("paired rollouts do not use exact common random-number seeds")
    for name in REQUIRED_OUTCOMES:
        base = np.asarray(baseline[name])
        changed = np.asarray(perturbed[name])
        if base.shape[0:1] != (n,) or changed.shape[:3] != (n, d, e):
            raise ValueError(f"paired outcome {name!r} has invalid leading shape")
        if changed.shape[3:] != base.shape[1:]:
            raise ValueError(f"paired outcome {name!r} trailing shapes differ")
        available = outcome_availability[name]
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(changed)):
            raise ValueError(f"paired outcome {name!r} is non-finite")
        if available and base.size == 0:
            raise ValueError(f"available paired outcome {name!r} is empty")
        if not available and (base.shape[1:] != (0,) or changed.shape[3:] != (0,)):
            raise ValueError(
                f"unavailable paired outcome {name!r} must use an empty trailing vector, not a placeholder"
            )
        if name in {"muscle_excitation", "muscle_activation"} and (
            np.any(base < -1e-6) or np.any(base > 1.0 + 1e-6) or np.any(changed < -1e-6) or np.any(changed > 1.0 + 1e-6)
        ):
            raise ValueError(
                f"paired outcome {name!r} must be unit-interval physical evidence; "
                "signed control cannot be relabeled as excitation or activation"
            )
        if name in {"muscle_excitation", "muscle_activation"} and (base.ndim != 2 or changed.ndim != 4):
            raise ValueError(f"paired outcome {name!r} must be ordered [sample,muscle] evidence")
    if baseline["muscle_excitation"].shape != baseline["muscle_activation"].shape:
        raise ValueError("paired muscle excitation and activation must share one ordered muscle ABI")


def _validated_outcome_schemas(
    value: Any,
    *,
    baseline: dict[str, np.ndarray],
    activation_valid_mask: Any,
    outcome_availability: Any,
) -> dict[str, dict[str, Any]]:
    """Require ordered, unit- and frame-aware names for every outcome axis."""

    if not isinstance(value, dict) or set(value) != set(REQUIRED_OUTCOMES):
        raise ValueError("outcome_schemas must contain exactly the required muscle/joint/task outcomes")
    availability = _validated_outcome_availability(outcome_availability)
    result: dict[str, dict[str, Any]] = {}
    for outcome in REQUIRED_OUTCOMES:
        schema = value[outcome]
        if not isinstance(schema, dict):
            raise ValueError(f"outcome schema {outcome!r} must be an object")
        trailing = tuple(int(item) for item in np.asarray(baseline[outcome]).shape[1:])
        width = int(np.prod(trailing, dtype=np.int64)) if trailing else 1
        feature_names = [str(item) for item in schema.get("feature_names") or []]
        units = [str(item) for item in schema.get("units") or []]
        coordinate_frame = str(schema.get("coordinate_frame", "")).strip()
        semantics = str(schema.get("semantics", "")).strip()
        available = availability[outcome]
        if (
            len(feature_names) != width
            or len(set(feature_names)) != width
            or any(not name.strip() for name in feature_names)
            or len(units) != width
            or any(not unit.strip() for unit in units)
            or not coordinate_frame
            or not semantics
            or available != (width > 0)
        ):
            raise ValueError(
                f"outcome schema {outcome!r} must bind every flattened value to a "
                "unique name, unit, coordinate frame, and semantics"
            )
        canonical_schema = {
            "feature_names": feature_names,
            "units": units,
            "coordinate_frame": coordinate_frame,
            "semantics": semantics,
            "source_shape_per_sample": list(trailing),
            "available": available,
        }
        contracts, missing_event_contract = _validated_masked_value_contracts(
            schema,
            outcome=outcome,
            feature_names=feature_names,
            available=available,
        )
        if contracts:
            canonical_schema["missing_event_contract"] = missing_event_contract
            canonical_schema["masked_value_contracts"] = contracts
        result[outcome] = canonical_schema
    excitation_names = result["muscle_excitation"]["feature_names"]
    activation_names = result["muscle_activation"]["feature_names"]
    if excitation_names != activation_names:
        raise ValueError("muscle excitation and activation schemas must use the same ordered muscle names")
    if (
        result["muscle_excitation"]["units"] != ["unit_interval"] * len(excitation_names)
        or result["muscle_activation"]["units"] != ["unit_interval"] * len(activation_names)
        or result["muscle_excitation"]["semantics"] != "unit_interval_excitation"
        or result["muscle_activation"]["semantics"] != "mujoco_unit_interval_activation_state"
    ):
        raise ValueError("muscle outcome schemas must declare unit excitation and MuJoCo unit activation")
    mask = validate_activation_valid_mask(
        activation_valid_mask,
        expected_width=len(activation_names),
    )
    if not np.any(mask):
        raise ValueError("muscle activation outcome has no valid name-aligned channels")
    return result


def _validated_masked_value_contracts(
    schema: Mapping[str, Any],
    *,
    outcome: str,
    feature_names: list[str],
    available: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raw_contracts = schema.get("masked_value_contracts")
    raw_policy = schema.get("missing_event_contract")
    if raw_contracts is None and raw_policy is None:
        return [], None
    if not available:
        raise ValueError(f"unavailable outcome {outcome!r} cannot declare masked value contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise ValueError(f"outcome {outcome!r} masked_value_contracts must be a non-empty list")
    if not isinstance(raw_policy, Mapping) or set(raw_policy) != {
        "schema_version",
        "storage_sentinel",
        "effect_policy",
    }:
        raise ValueError(f"outcome {outcome!r} lacks a canonical missing-event contract")
    if raw_policy.get("schema_version") != "event_presence_masked_zero_sentinel_v1":
        raise ValueError(f"outcome {outcome!r} uses an unsupported missing-event contract")
    storage_sentinel = raw_policy.get("storage_sentinel")
    if (
        isinstance(storage_sentinel, bool)
        or not isinstance(storage_sentinel, int | float)
        or not np.isfinite(float(storage_sentinel))
        or float(storage_sentinel) != 0.0
        or "never measurements" not in str(raw_policy.get("effect_policy", ""))
    ):
        raise ValueError(f"outcome {outcome!r} missing-event storage sentinel policy is unsafe")

    canonical: list[dict[str, Any]] = []
    value_features: set[str] = set()
    for item in raw_contracts:
        if not isinstance(item, Mapping) or set(item) != {
            "presence_feature",
            "value_feature",
            "missing_sentinel",
        }:
            raise ValueError(f"outcome {outcome!r} has a malformed masked value contract")
        presence = str(item.get("presence_feature", "")).strip()
        value = str(item.get("value_feature", "")).strip()
        sentinel = item.get("missing_sentinel")
        if (
            not presence
            or not value
            or presence == value
            or presence not in feature_names
            or value not in feature_names
            or value in value_features
            or isinstance(sentinel, bool)
            or not isinstance(sentinel, int | float)
            or not np.isfinite(float(sentinel))
            or float(sentinel) != float(storage_sentinel)
        ):
            raise ValueError(f"outcome {outcome!r} has an unsafe masked value contract")
        value_features.add(value)
        canonical.append(
            {
                "presence_feature": presence,
                "value_feature": value,
                "missing_sentinel": float(sentinel),
            }
        )
    if value_features & {item["presence_feature"] for item in canonical}:
        raise ValueError(f"outcome {outcome!r} cannot reuse a masked value as an event-presence feature")
    policy = {
        "schema_version": "event_presence_masked_zero_sentinel_v1",
        "storage_sentinel": 0.0,
        "effect_policy": str(raw_policy["effect_policy"]),
    }
    return canonical, policy


def _excluded_masked_value_features(schema: Mapping[str, Any]) -> list[str]:
    excluded = {str(contract["value_feature"]) for contract in schema.get("masked_value_contracts", ())}
    return [str(feature) for feature in schema.get("feature_names", ()) if str(feature) in excluded]


def _validate_masked_value_records(
    baseline: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    outcome_schemas: Mapping[str, Mapping[str, Any]],
    outcome_availability: Mapping[str, bool],
) -> None:
    for outcome in REQUIRED_OUTCOMES:
        schema = outcome_schemas[outcome]
        contracts = schema.get("masked_value_contracts", ())
        if not contracts or not outcome_availability[outcome]:
            continue
        names = list(schema["feature_names"])
        width = len(names)
        base = np.asarray(baseline[outcome]).reshape((-1, width))
        changed = np.asarray(perturbed[outcome]).reshape((-1, width))
        for contract in contracts:
            presence_index = names.index(contract["presence_feature"])
            value_index = names.index(contract["value_feature"])
            sentinel = float(contract["missing_sentinel"])
            for label, rows in (("baseline", base), ("perturbed", changed)):
                presence = rows[:, presence_index]
                values = rows[:, value_index]
                if np.any((presence != 0.0) & (presence != 1.0)):
                    raise ValueError(f"{label} {outcome!r} event presence must be exact binary values")
                if np.any(values[presence == 0.0] != sentinel):
                    raise ValueError(f"{label} {outcome!r} missing event value differs from its storage sentinel")


def _expected_outcome_layout(
    outcome_schemas: Mapping[str, Mapping[str, Any]],
    *,
    outcome_availability: Mapping[str, bool],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    offset = 0
    for outcome in REQUIRED_OUTCOMES:
        schema = outcome_schemas[outcome]
        feature_names = list(schema["feature_names"])
        excluded = _excluded_masked_value_features(schema)
        included = [feature for feature in feature_names if feature not in excluded]
        available = outcome_availability[outcome]
        effect_names = [f"{outcome}:{feature}" for feature in included] if available else []
        width = len(effect_names)
        result[outcome] = {
            "source_shape_per_sample": list(schema["source_shape_per_sample"]),
            "flat_start": offset,
            "flat_stop": offset + width,
            "effect_names": effect_names,
            "excluded_masked_value_features": excluded if available else [],
            "available": available,
            "schema": dict(schema),
        }
        offset += width
    return result


def _validated_outcome_availability(value: Any) -> dict[str, bool]:
    if (
        not isinstance(value, dict)
        or set(value) != set(REQUIRED_OUTCOMES)
        or any(type(value[name]) is not bool for name in REQUIRED_OUTCOMES)
    ):
        raise ValueError("outcome_availability must contain exact boolean entries for every required outcome")
    result = {name: bool(value[name]) for name in REQUIRED_OUTCOMES}
    if not result["muscle_excitation"] or not result["muscle_activation"]:
        raise ValueError("causal evidence must include measured excitation and activation")
    return result


def _outcome_completeness(availability: Mapping[str, bool]) -> tuple[bool, bool]:
    diagnostic = (
        "muscle_excitation",
        "muscle_activation",
        "joint_position",
        "joint_velocity",
        "trunk_state",
        "racket_state",
    )
    return (
        all(availability[name] for name in diagnostic),
        all(availability[name] for name in REQUIRED_OUTCOMES),
    )


def _load_self_fingerprinted_json(
    path: Path,
    *,
    fingerprint_field: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied = payload.get(fingerprint_field)
    content = {key: value for key, value in payload.items() if key != fingerprint_field}
    if supplied != canonical_json_sha256(content):
        raise ValueError(f"JSON fingerprint mismatch: {path}")
    return payload


def _require_hex64(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be lowercase 64-hex")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-inputs", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--perturbed-records", type=Path, required=True)
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_causal_rollout_artifact(
        analysis_inputs=args.analysis_inputs,
        analysis_manifest=args.analysis_manifest,
        baseline_records=args.baseline_records,
        perturbed_records=args.perturbed_records,
        rollout_manifest=args.rollout_manifest,
        output_npz=args.output_npz,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
