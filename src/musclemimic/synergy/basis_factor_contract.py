"""Matched-data contract for standard and graph-regularized NMF factors."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BASIS_FACTOR_CONTRACT_SCHEMA_VERSION = "basis_factor_contract_v1"
COEFFICIENT_TRANSFORM_FACTOR_SCHEMA_VERSION = "bounded_sigmoid_coefficient_transform_factor_v1"
NEAR_ZERO_MASK_SCHEMA_VERSION = "near_zero_channel_mask_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {
    "schema_version",
    "fit_scope",
    "source_dataset_fingerprint",
    "train_motion_uids",
    "validation_motion_uids",
    "primitive_source_manifest_fingerprint",
    "signal_kind",
    "sample_weighting",
    "phase_weighting",
    "normalization",
    "near_zero_mask",
    "kept_actuator_indices",
    "candidate_ranks",
    "selected_rank",
    "nmf_initialization_seeds",
    "max_iter",
    "tol",
    "dynamic_coverage_environment_fingerprint",
    "dynamic_coverage_rollout_fingerprint",
    "coefficient_transform_schema",
    "policy_action_dimension",
    "basis_factor_contract_fingerprint",
}


def default_coefficient_transform_factor() -> dict[str, Any]:
    """Return the preregistered transform shared by the B/G action arms."""

    return {
        "schema_version": COEFFICIENT_TRANSFORM_FACTOR_SCHEMA_VERSION,
        "kind": "bounded_sigmoid",
        "maximum_source": "train_q99_times_1p2",
        "center_source": "train_q50",
        "temperature": 1.0,
    }


def near_zero_mask_contract(
    *,
    channel_count: int,
    kept_indices: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    count = _positive_int(channel_count, "near-zero channel_count")
    kept = _ordered_indices(kept_indices, width=count)
    checked_threshold = _nonnegative_finite(threshold, "near-zero threshold")
    mask = np.zeros(count, dtype=np.uint8)
    mask[np.asarray(kept, dtype=np.int64)] = 1
    return {
        "schema_version": NEAR_ZERO_MASK_SCHEMA_VERSION,
        "threshold": checked_threshold,
        "channel_count": count,
        "kept_mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
    }


def build_basis_factor_contract(
    *,
    fit_scope: str,
    source_dataset_fingerprint: str,
    train_motion_uids: Sequence[int],
    validation_motion_uids: Sequence[int],
    primitive_source_manifest_fingerprint: str | None,
    signal_kind: str,
    sample_weighting: Mapping[str, Any],
    phase_weighting: Mapping[str, Any],
    normalization: Mapping[str, Any],
    near_zero_mask: Mapping[str, Any],
    kept_actuator_indices: Sequence[int],
    candidate_ranks: Sequence[int] | Mapping[str, Any],
    selected_rank: int,
    nmf_initialization_seeds: Sequence[int],
    max_iter: int,
    tol: float,
    dynamic_coverage_environment_fingerprint: str | None,
    dynamic_coverage_rollout_fingerprint: str | None,
    coefficient_transform_schema: Mapping[str, Any] | None = None,
    policy_action_dimension: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": BASIS_FACTOR_CONTRACT_SCHEMA_VERSION,
        "fit_scope": str(fit_scope),
        "source_dataset_fingerprint": str(source_dataset_fingerprint),
        "train_motion_uids": [int(value) for value in train_motion_uids],
        "validation_motion_uids": [int(value) for value in validation_motion_uids],
        "primitive_source_manifest_fingerprint": primitive_source_manifest_fingerprint,
        "signal_kind": str(signal_kind),
        "sample_weighting": copy.deepcopy(dict(sample_weighting)),
        "phase_weighting": copy.deepcopy(dict(phase_weighting)),
        "normalization": copy.deepcopy(dict(normalization)),
        "near_zero_mask": copy.deepcopy(dict(near_zero_mask)),
        "kept_actuator_indices": [int(value) for value in kept_actuator_indices],
        "candidate_ranks": copy.deepcopy(candidate_ranks),
        "selected_rank": int(selected_rank),
        "nmf_initialization_seeds": [int(value) for value in nmf_initialization_seeds],
        "max_iter": int(max_iter),
        "tol": float(tol),
        "dynamic_coverage_environment_fingerprint": dynamic_coverage_environment_fingerprint,
        "dynamic_coverage_rollout_fingerprint": dynamic_coverage_rollout_fingerprint,
        "coefficient_transform_schema": copy.deepcopy(
            dict(coefficient_transform_schema or default_coefficient_transform_factor())
        ),
        "policy_action_dimension": int(policy_action_dimension or selected_rank),
    }
    payload["basis_factor_contract_fingerprint"] = basis_factor_contract_fingerprint(payload)
    return validate_basis_factor_contract(payload)


def validate_basis_factor_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("basis factor contract fields differ from basis_factor_contract_v1")
    if value["schema_version"] != BASIS_FACTOR_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported basis factor contract schema")
    train_uids = _uids(value["train_motion_uids"], "train_motion_uids")
    validation_uids = _uids(value["validation_motion_uids"], "validation_motion_uids")
    if set(train_uids) & set(validation_uids):
        raise ValueError("basis factor train/validation motion UIDs overlap")
    primitive = value["primitive_source_manifest_fingerprint"]
    if primitive is not None:
        primitive = _sha256(primitive, "primitive source manifest fingerprint")
    environment = value["dynamic_coverage_environment_fingerprint"]
    rollout = value["dynamic_coverage_rollout_fingerprint"]
    if (environment is None) != (rollout is None):
        raise ValueError("dynamic coverage environment and rollout fingerprints must be paired")
    if environment is not None:
        environment = _sha256(environment, "dynamic coverage environment fingerprint")
        rollout = _sha256(rollout, "dynamic coverage rollout fingerprint")
    normalization = _mapping(value["normalization"], "basis factor normalization")
    sample_weighting = _mapping(value["sample_weighting"], "basis factor sample weighting")
    phase_weighting = _mapping(value["phase_weighting"], "basis factor phase weighting")
    near_zero = _validate_near_zero_mask(value["near_zero_mask"])
    kept = _ordered_indices(
        value["kept_actuator_indices"],
        width=near_zero["channel_count"],
    )
    candidate_ranks = _candidate_ranks(value["candidate_ranks"])
    selected_rank = _positive_int(value["selected_rank"], "selected_rank")
    if selected_rank not in _flatten_candidate_ranks(candidate_ranks) and not isinstance(candidate_ranks, dict):
        raise ValueError("selected rank is absent from candidate ranks")
    seeds = _nonnegative_unique_ints(value["nmf_initialization_seeds"], "NMF initialization seeds")
    transform = _validate_coefficient_transform(value["coefficient_transform_schema"])
    result = {
        "schema_version": BASIS_FACTOR_CONTRACT_SCHEMA_VERSION,
        "fit_scope": _text(value["fit_scope"], "fit_scope"),
        "source_dataset_fingerprint": _sha256(
            value["source_dataset_fingerprint"],
            "source dataset fingerprint",
        ),
        "train_motion_uids": train_uids,
        "validation_motion_uids": validation_uids,
        "primitive_source_manifest_fingerprint": primitive,
        "signal_kind": _text(value["signal_kind"], "signal_kind"),
        "sample_weighting": sample_weighting,
        "phase_weighting": phase_weighting,
        "normalization": normalization,
        "near_zero_mask": near_zero,
        "kept_actuator_indices": kept,
        "candidate_ranks": candidate_ranks,
        "selected_rank": selected_rank,
        "nmf_initialization_seeds": seeds,
        "max_iter": _positive_int(value["max_iter"], "max_iter"),
        "tol": _nonnegative_finite(value["tol"], "tol"),
        "dynamic_coverage_environment_fingerprint": environment,
        "dynamic_coverage_rollout_fingerprint": rollout,
        "coefficient_transform_schema": transform,
        "policy_action_dimension": _positive_int(
            value["policy_action_dimension"],
            "policy_action_dimension",
        ),
        "basis_factor_contract_fingerprint": _sha256(
            value["basis_factor_contract_fingerprint"],
            "basis factor contract fingerprint",
        ),
    }
    if result["policy_action_dimension"] != selected_rank:
        raise ValueError("basis factor policy action dimension must equal selected rank")
    if basis_factor_contract_fingerprint(result) != result["basis_factor_contract_fingerprint"]:
        raise ValueError("basis factor contract fingerprint is stale")
    return result


def basis_factor_contract_fingerprint(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("basis_factor_contract_fingerprint", None)
    return _json_sha256(payload)


def assert_matched_basis_factor_contracts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> str:
    left_contract = validate_basis_factor_contract(left)
    right_contract = validate_basis_factor_contract(right)
    left_fingerprint = left_contract["basis_factor_contract_fingerprint"]
    if right_contract["basis_factor_contract_fingerprint"] != left_fingerprint:
        raise ValueError("standard and Graph-NMF basis factor contracts are not matched")
    if left_contract != right_contract:
        raise ValueError("equal basis factor fingerprints do not resolve to equal contracts")
    return left_fingerprint


def compose_basis_factor_contract(
    source_contracts: Mapping[str, Mapping[str, Any]],
    *,
    fit_scope: str,
    normalization: Mapping[str, Any],
    channel_count: int,
    kept_actuator_indices: Sequence[int],
    selected_rank: int,
) -> dict[str, Any]:
    """Compose regional/hybrid factors without admitting graph-specific fields."""

    if not isinstance(source_contracts, Mapping) or not source_contracts:
        raise ValueError("composite basis factor requires source contracts")
    validated = {
        _text(label, "source factor label"): validate_basis_factor_contract(contract)
        for label, contract in sorted(source_contracts.items())
    }
    first = next(iter(validated.values()))
    invariant_fields = (
        "source_dataset_fingerprint",
        "train_motion_uids",
        "validation_motion_uids",
        "primitive_source_manifest_fingerprint",
        "signal_kind",
        "sample_weighting",
        "phase_weighting",
        "nmf_initialization_seeds",
        "max_iter",
        "tol",
        "dynamic_coverage_environment_fingerprint",
        "dynamic_coverage_rollout_fingerprint",
        "coefficient_transform_schema",
    )
    for label, contract in validated.items():
        for field in invariant_fields:
            if contract[field] != first[field]:
                raise ValueError(f"source factor {label!r} differs on shared field {field}")
    kept = _ordered_indices(kept_actuator_indices, width=_positive_int(channel_count, "channel_count"))
    thresholds = {contract["near_zero_mask"]["threshold"] for contract in validated.values()}
    if len(thresholds) != 1:
        raise ValueError("source factor near-zero thresholds differ")
    candidate_ranks = {label: contract["candidate_ranks"] for label, contract in validated.items()}
    return build_basis_factor_contract(
        fit_scope=fit_scope,
        source_dataset_fingerprint=first["source_dataset_fingerprint"],
        train_motion_uids=first["train_motion_uids"],
        validation_motion_uids=first["validation_motion_uids"],
        primitive_source_manifest_fingerprint=first["primitive_source_manifest_fingerprint"],
        signal_kind=first["signal_kind"],
        sample_weighting=first["sample_weighting"],
        phase_weighting=first["phase_weighting"],
        normalization=normalization,
        near_zero_mask=near_zero_mask_contract(
            channel_count=channel_count,
            kept_indices=kept,
            threshold=next(iter(thresholds)),
        ),
        kept_actuator_indices=kept,
        candidate_ranks=candidate_ranks,
        selected_rank=selected_rank,
        nmf_initialization_seeds=first["nmf_initialization_seeds"],
        max_iter=first["max_iter"],
        tol=first["tol"],
        dynamic_coverage_environment_fingerprint=first["dynamic_coverage_environment_fingerprint"],
        dynamic_coverage_rollout_fingerprint=first["dynamic_coverage_rollout_fingerprint"],
        coefficient_transform_schema=first["coefficient_transform_schema"],
        policy_action_dimension=selected_rank,
    )


def _validate_near_zero_mask(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "threshold",
        "channel_count",
        "kept_mask_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("near-zero mask fields differ from contract")
    if value["schema_version"] != NEAR_ZERO_MASK_SCHEMA_VERSION:
        raise ValueError("unsupported near-zero mask schema")
    return {
        "schema_version": NEAR_ZERO_MASK_SCHEMA_VERSION,
        "threshold": _nonnegative_finite(value["threshold"], "near-zero threshold"),
        "channel_count": _positive_int(value["channel_count"], "near-zero channel_count"),
        "kept_mask_sha256": _sha256(value["kept_mask_sha256"], "near-zero kept mask fingerprint"),
    }


def _validate_coefficient_transform(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version": COEFFICIENT_TRANSFORM_FACTOR_SCHEMA_VERSION,
        "kind": "bounded_sigmoid",
        "maximum_source": "train_q99_times_1p2",
        "center_source": "train_q50",
        "temperature": 1.0,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("unsupported basis-factor coefficient transform schema")
    return expected


def _candidate_ranks(value: Any) -> list[int] | dict[str, Any]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("candidate rank mapping must be non-empty")
        return {_text(key, "candidate rank scope"): _candidate_ranks(ranks) for key, ranks in sorted(value.items())}
    return _positive_unique_ints(value, "candidate ranks")


def _flatten_candidate_ranks(value: list[int] | dict[str, Any]) -> list[int]:
    if isinstance(value, list):
        return list(value)
    return [rank for child in value.values() for rank in _flatten_candidate_ranks(child)]


def _uids(value: Any, field: str) -> list[int]:
    values = _nonnegative_unique_ints(value, field)
    if not values:
        raise ValueError(f"{field} must be non-empty")
    return sorted(values)


def _positive_unique_ints(value: Any, field: str) -> list[int]:
    values = _int_sequence(value, field)
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique positive integers")
    return sorted(values)


def _nonnegative_unique_ints(value: Any, field: str) -> list[int]:
    values = _int_sequence(value, field)
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique non-negative integers")
    return sorted(values)


def _int_sequence(value: Any, field: str) -> list[int]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{field} must be one-dimensional")
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{field} must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int | np.integer) for item in value):
        raise ValueError(f"{field} must contain integers")
    return [int(item) for item in value]


def _ordered_indices(value: Any, *, width: int) -> list[int]:
    indices = _nonnegative_unique_ints(value, "kept actuator indices")
    if indices != sorted(indices) or any(index >= width for index in indices):
        raise ValueError("kept actuator indices must be sorted and within the channel schema")
    return indices


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    payload = copy.deepcopy(dict(value))
    _json_sha256(payload)
    return payload


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return text


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
