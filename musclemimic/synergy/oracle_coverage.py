"""Fail-closed static proxy coverage gate for an early-synergy action basis.

The evidence produced here is deliberately narrow.  It answers whether a
formal physical-excitation basis can reconstruct a supplied target excitation
proxy with bounded, non-negative coefficients.  It does *not* execute the
simulator, optimize a trajectory, or provide short-horizon dynamics evidence.

The report is self-fingerprinted and binds the formal basis artifact, the
ordered muscle schema, the exact proxy/phase content, coefficient bounds, and
every promotion threshold.  Training code should load it through
``load_static_proxy_coverage_gate`` with ``require_passed=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.physical import UNIT_INTERVAL_TOLERANCE
from musclemimic.synergy.basis_artifact import (
    BASIS_ARTIFACT_SCHEMA_VERSION,
    SynergyBasisArtifact,
    load_synergy_basis,
)
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

STATIC_PROXY_COVERAGE_SCHEMA_VERSION = "chinajump_static_proxy_coverage_gate_v3"
FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION = "chinajump_static_proxy_coverage_gate_v4"
STATIC_PROXY_EVIDENCE_KIND = "static_proxy_excitation_reconstruction"
STATIC_PROXY_PHASE_SCHEMA_VERSION = "chinajump_coverage_phase_schema_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_kind",
        "limitations",
        "formal_basis_binding",
        "proxy_binding",
        "proxy_fingerprint",
        "solver",
        "metrics",
        "thresholds",
        "checks",
        "passed",
        "artifact_fingerprint",
    }
)
_BASIS_BINDING_FIELDS = frozenset(
    {
        "artifact_schema_version",
        "artifact_fingerprint",
        "signal_kind",
        "basis_shape",
        "muscle_names",
        "muscle_schema_sha256",
    }
)
_PROXY_BINDING_FIELDS = frozenset(
    {
        "signal_kind",
        "content_fingerprint",
        "shape",
        "muscle_schema_sha256",
        "phase_id_included",
        "observed_phase_ids",
        "phase_schema",
        "phase_schema_fingerprint",
    }
)
_FORMAL_PROXY_BINDING_FIELDS = _PROXY_BINDING_FIELDS | {"producer_binding"}
_PRODUCER_BINDING_FIELDS = frozenset(
    {
        "producer_manifest_schema_version",
        "producer_manifest_fingerprint",
        "producer_artifact_kind",
        "source_kind",
        "source_manifest_fingerprint",
        "source_qc_fingerprint",
        "proxy_content_fingerprint",
        "phase_schema_fingerprint",
        "required_phase_ids",
        "min_phase_samples",
        "per_phase_sample_counts",
    }
)
_PHASE_SCHEMA_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "target_skill_id",
        "phase_field",
        "producer_contract",
        "phases",
    }
)
_PHASE_SCHEMA_FIELDS = _PHASE_SCHEMA_SOURCE_FIELDS | {"phase_schema_fingerprint"}
_PHASE_DEFINITION_FIELDS = frozenset({"id", "name", "definition"})
_SOLVER_FIELDS = frozenset(
    {
        "objective",
        "algorithm",
        "coefficient_lower_bound",
        "coefficient_upper_bounds",
        "solver_tolerance",
        "solver_max_iterations",
        "normalization_epsilon",
        "saturation_tolerance",
        "effective_rank_relative_tolerance",
        "solved_frame_count",
        "coefficient_solution_fingerprint",
        "decoded_solution_fingerprint",
    }
)
_METRICS_FIELDS = frozenset(
    {
        "global",
        "per_phase",
        "proxy_active_muscle_fraction",
        "decoded_saturation_fraction",
        "decoded_preclip_upper_violation_fraction",
        "coefficient_upper_saturation_fraction",
        "basis_condition_number",
        "basis_effective_rank",
        "basis_effective_rank_fraction",
        "basis_singular_values",
    }
)
_ERROR_METRIC_FIELDS = frozenset(
    {
        "sample_count",
        "target_l2_norm",
        "residual_l2_norm",
        "rmse",
        "target_rms",
        "relative_l2_nrmse",
        "relative_squared_loss",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {
        "min_proxy_target_rms",
        "min_active_muscle_fraction",
        "active_muscle_rms_threshold",
        "max_global_relative_l2_nrmse",
        "max_phase_relative_l2_nrmse",
        "max_decoded_saturation_fraction",
        "max_basis_condition_number",
        "min_effective_rank_fraction",
        "required_phase_ids",
    }
)
_CHECK_FIELDS = frozenset(
    {
        "proxy_target_rms",
        "proxy_active_muscle_fraction",
        "global_relative_l2_nrmse",
        "all_observed_phases_relative_l2_nrmse",
        "required_phase_presence",
        "decoded_saturation_fraction",
        "basis_condition_number",
        "basis_effective_rank_fraction",
        "per_phase_relative_l2_nrmse",
        "required_phase_target_rms",
    }
)
_LIMITATIONS = [
    "static_proxy_only",
    "no_simulator_transition_or_short_horizon_dynamics_evaluated",
    "proxy_quality_is_external_to_this_gate",
]


@dataclass(frozen=True)
class StaticProxyCoverageThresholds:
    """Promotion thresholds for static proxy reconstruction evidence."""

    min_proxy_target_rms: float = 1e-4
    min_active_muscle_fraction: float = 0.05
    active_muscle_rms_threshold: float = 1e-4
    max_global_relative_l2_nrmse: float = 0.15
    max_phase_relative_l2_nrmse: float = 0.25
    max_decoded_saturation_fraction: float = 0.05
    max_basis_condition_number: float = 100.0
    min_effective_rank_fraction: float = 0.80
    required_phase_ids: tuple[int, ...] = ()

    def validated(self) -> StaticProxyCoverageThresholds:
        minimums = {
            "min_proxy_target_rms": self.min_proxy_target_rms,
            "min_active_muscle_fraction": self.min_active_muscle_fraction,
            "active_muscle_rms_threshold": self.active_muscle_rms_threshold,
        }
        for name, raw in minimums.items():
            value = _finite_float(raw, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.min_proxy_target_rms) <= 0.0:
            raise ValueError("min_proxy_target_rms must be positive")
        if float(self.active_muscle_rms_threshold) <= 0.0:
            raise ValueError("active_muscle_rms_threshold must be positive")
        if not 0.0 < float(self.min_active_muscle_fraction) <= 1.0:
            raise ValueError("min_active_muscle_fraction must lie in (0,1]")
        maximums = {
            "max_global_relative_l2_nrmse": self.max_global_relative_l2_nrmse,
            "max_phase_relative_l2_nrmse": self.max_phase_relative_l2_nrmse,
            "max_decoded_saturation_fraction": self.max_decoded_saturation_fraction,
            "max_basis_condition_number": self.max_basis_condition_number,
        }
        for name, raw in maximums.items():
            value = _finite_float(raw, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.max_decoded_saturation_fraction) > 1.0:
            raise ValueError("max_decoded_saturation_fraction must lie in [0,1]")
        effective_rank = _finite_float(self.min_effective_rank_fraction, "min_effective_rank_fraction")
        if not 0.0 <= effective_rank <= 1.0:
            raise ValueError("min_effective_rank_fraction must lie in [0,1]")
        phase_ids = tuple(_strict_int(value, "required_phase_ids") for value in self.required_phase_ids)
        if any(value < 0 for value in phase_ids):
            raise ValueError("required_phase_ids must be non-negative")
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("required_phase_ids must be unique")
        return StaticProxyCoverageThresholds(
            min_proxy_target_rms=float(self.min_proxy_target_rms),
            min_active_muscle_fraction=float(self.min_active_muscle_fraction),
            active_muscle_rms_threshold=float(self.active_muscle_rms_threshold),
            max_global_relative_l2_nrmse=float(self.max_global_relative_l2_nrmse),
            max_phase_relative_l2_nrmse=float(self.max_phase_relative_l2_nrmse),
            max_decoded_saturation_fraction=float(self.max_decoded_saturation_fraction),
            max_basis_condition_number=float(self.max_basis_condition_number),
            min_effective_rank_fraction=float(self.min_effective_rank_fraction),
            required_phase_ids=tuple(sorted(phase_ids)),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | StaticProxyCoverageThresholds | None,
    ) -> StaticProxyCoverageThresholds:
        if payload is None:
            return cls().validated()
        if isinstance(payload, cls):
            return payload.validated()
        if not isinstance(payload, Mapping):
            raise TypeError("thresholds must be StaticProxyCoverageThresholds or a mapping")
        unknown = sorted(set(payload) - _THRESHOLD_FIELDS)
        if unknown:
            raise ValueError(f"unknown static proxy threshold fields: {unknown}")
        values = dict(payload)
        if "required_phase_ids" in values:
            raw_phase_ids = values["required_phase_ids"]
            if not isinstance(raw_phase_ids, list | tuple):
                raise ValueError("required_phase_ids must be a list or tuple")
            values["required_phase_ids"] = tuple(raw_phase_ids)
        return cls(**values).validated()


def canonicalize_static_proxy_phase_schema(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and fingerprint the semantic meaning of every proxy phase id.

    Source JSON omits ``phase_schema_fingerprint``; embedded gate contracts may
    include it, in which case it must exactly match the canonical content.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("static proxy phase schema must be a JSON object")
    fields = set(payload)
    if fields not in (set(_PHASE_SCHEMA_SOURCE_FIELDS), set(_PHASE_SCHEMA_FIELDS)):
        raise ValueError(f"static proxy phase schema fields differ from contract (actual={sorted(fields)})")
    if payload.get("schema_version") != STATIC_PROXY_PHASE_SCHEMA_VERSION:
        raise ValueError("unsupported static proxy phase schema")
    target_skill_id = payload.get("target_skill_id")
    producer_contract = payload.get("producer_contract")
    if not isinstance(target_skill_id, str) or not target_skill_id.strip():
        raise ValueError("static proxy phase schema target_skill_id must be non-empty")
    if payload.get("phase_field") != "phase_id":
        raise ValueError("static proxy phase schema phase_field must be 'phase_id'")
    if not isinstance(producer_contract, str) or not producer_contract.strip():
        raise ValueError("static proxy phase schema producer_contract must be non-empty")
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("static proxy phase schema requires at least one phase")
    phases: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, Mapping) or set(raw_phase) != _PHASE_DEFINITION_FIELDS:
            raise ValueError("every static proxy phase requires exactly id/name/definition")
        phase_id = _strict_int(raw_phase.get("id"), "static proxy phase id")
        name = raw_phase.get("name")
        definition = raw_phase.get("definition")
        if phase_id < 0 or phase_id in seen_ids:
            raise ValueError("static proxy phase ids must be unique and non-negative")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise ValueError("static proxy phase names must be unique and non-empty")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("static proxy phase definitions must be non-empty")
        seen_ids.add(phase_id)
        seen_names.add(name)
        phases.append(
            {
                "id": phase_id,
                "name": name,
                "definition": definition,
            }
        )
    if [phase["id"] for phase in phases] != sorted(seen_ids):
        raise ValueError("static proxy phases must be sorted by id")
    unsigned = {
        "schema_version": STATIC_PROXY_PHASE_SCHEMA_VERSION,
        "target_skill_id": target_skill_id,
        "phase_field": "phase_id",
        "producer_contract": producer_contract,
        "phases": phases,
    }
    fingerprint = _json_sha256(unsigned)
    supplied_fingerprint = payload.get("phase_schema_fingerprint")
    if supplied_fingerprint is not None:
        _require_sha256(supplied_fingerprint, "phase_schema_fingerprint")
        if supplied_fingerprint != fingerprint:
            raise ValueError("static proxy phase schema fingerprint mismatch")
    return {**unsigned, "phase_schema_fingerprint": fingerprint}


def load_static_proxy_phase_schema(path: str | Path) -> dict[str, Any]:
    """Load a semantic phase schema used by a proxy producer and gate."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"static proxy phase schema does not exist: {source}")
    payload = load_json_strict(source)
    if not isinstance(payload, dict):
        raise ValueError("static proxy phase schema must contain a JSON object")
    return canonicalize_static_proxy_phase_schema(payload)


def _phase_schema_for_phase_ids(
    phase_schema: Mapping[str, Any] | None,
    *,
    phase_id: np.ndarray | None,
) -> dict[str, Any] | None:
    if phase_id is None:
        if phase_schema is not None:
            raise ValueError("phase_schema cannot be supplied without phase_id")
        return None
    if phase_schema is None:
        raise ValueError("phase_id requires a semantic phase_schema; integer labels alone are ambiguous")
    canonical = canonicalize_static_proxy_phase_schema(phase_schema)
    declared_ids = {int(phase["id"]) for phase in canonical["phases"]}
    observed_ids = {int(value) for value in np.unique(phase_id).tolist()}
    unknown = sorted(observed_ids - declared_ids)
    if unknown:
        raise ValueError(f"proxy contains phase ids absent from semantic phase_schema: {unknown}")
    return canonical


def proxy_content_fingerprint(
    proxy_excitation: np.ndarray,
    *,
    muscle_names: Sequence[str],
    phase_id: np.ndarray | None = None,
    phase_schema: Mapping[str, Any] | None = None,
) -> str:
    """Hash proxy values, ordered muscles, phase ids, and their semantics."""

    names = tuple(str(name) for name in muscle_names)
    proxy = _validate_proxy_excitation(proxy_excitation, expected_width=len(names))
    if len(names) != proxy.shape[1] or len(set(names)) != len(names):
        raise ValueError("muscle_names must uniquely match proxy columns")
    phases = _validate_phase_id(phase_id, sample_count=proxy.shape[0])
    canonical_phase_schema = _phase_schema_for_phase_ids(
        phase_schema,
        phase_id=phases,
    )
    digest = hashlib.sha256()
    header = {
        "schema_version": "static_proxy_content_fingerprint_v2",
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "shape": list(proxy.shape),
        "canonical_dtype": "little_endian_float64",
        "muscle_names": list(names),
        "phase_id_included": phases is not None,
        "phase_dtype": None if phases is None else "little_endian_int64",
        "phase_schema_fingerprint": (
            None if canonical_phase_schema is None else canonical_phase_schema["phase_schema_fingerprint"]
        ),
    }
    digest.update(_canonical_json_bytes(header))
    digest.update(np.ascontiguousarray(proxy, dtype="<f8").tobytes(order="C"))
    if phases is not None:
        digest.update(np.ascontiguousarray(phases, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def evaluate_static_proxy_coverage(
    formal_basis: str | Path | SynergyBasisArtifact,
    proxy_excitation: np.ndarray,
    *,
    phase_id: np.ndarray | None = None,
    phase_schema: Mapping[str, Any] | None = None,
    coefficient_upper_bounds: float | Sequence[float] | np.ndarray = 1.0,
    thresholds: Mapping[str, Any] | StaticProxyCoverageThresholds | None = None,
    proxy_muscle_names: Sequence[str] | None = None,
    solver_tolerance: float = 1e-10,
    solver_max_iterations: int = 500,
    normalization_epsilon: float = 1e-12,
    saturation_tolerance: float = 1e-6,
    effective_rank_relative_tolerance: float = 1e-8,
    proxy_producer_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate and seal bounded static-proxy excitation reconstruction.

    Args:
        formal_basis: A strict physical-excitation synergy basis artifact or its
            directory/manifest path.  Raw matrices are intentionally rejected.
        proxy_excitation: Target unit physical excitation with shape ``[T, M]``.
        phase_id: Optional integer phase id for every proxy frame.  When
            supplied, ``phase_schema`` is mandatory.
        phase_schema: Semantic id/name/definition contract for ``phase_id``.
        coefficient_upper_bounds: One finite positive upper bound or one per
            basis coefficient.  Coefficient lower bounds are exactly zero.
        thresholds: Promotion thresholds; every observed phase is gated when
            ``phase_id`` is supplied.  ``required_phase_ids`` additionally makes
            missing critical phases fail closed.
        proxy_muscle_names: Optional ordered proxy schema.  If supplied, it must
            exactly equal the formal basis order.

    Returns:
        A self-fingerprinted JSON-compatible ``static_proxy`` gate report.
    """

    if isinstance(formal_basis, SynergyBasisArtifact):
        artifact = load_synergy_basis(formal_basis.path)
        if artifact.fingerprint != formal_basis.fingerprint:
            raise ValueError("supplied formal basis object differs from its persisted artifact")
    else:
        artifact = load_synergy_basis(formal_basis)
    _validate_formal_excitation_basis(artifact)
    basis = np.asarray(artifact.basis, dtype=np.float64)
    proxy = _validate_proxy_excitation(proxy_excitation, expected_width=basis.shape[0])
    phases = _validate_phase_id(phase_id, sample_count=proxy.shape[0])
    canonical_phase_schema = _phase_schema_for_phase_ids(
        phase_schema,
        phase_id=phases,
    )
    if proxy_muscle_names is not None and tuple(str(name) for name in proxy_muscle_names) != artifact.muscle_names:
        raise ValueError("proxy muscle names/order differ from formal basis")

    upper = _coefficient_upper_bounds(coefficient_upper_bounds, rank=basis.shape[1])
    gate_thresholds = StaticProxyCoverageThresholds.from_mapping(thresholds)
    if gate_thresholds.required_phase_ids and phases is None:
        raise ValueError("required_phase_ids cannot be checked without phase_id")
    if canonical_phase_schema is not None:
        declared_phase_ids = {int(phase["id"]) for phase in canonical_phase_schema["phases"]}
        unknown_required = sorted(set(gate_thresholds.required_phase_ids) - declared_phase_ids)
        if unknown_required:
            raise ValueError(f"required phase ids are absent from semantic phase_schema: {unknown_required}")
    solver_tol = _positive_finite(solver_tolerance, "solver_tolerance")
    max_iter = _strict_int(solver_max_iterations, "solver_max_iterations")
    if max_iter <= 0:
        raise ValueError("solver_max_iterations must be positive")
    norm_eps = _positive_finite(normalization_epsilon, "normalization_epsilon")
    saturation_tol = _finite_float(saturation_tolerance, "saturation_tolerance")
    if not 0.0 < saturation_tol < 1.0:
        raise ValueError("saturation_tolerance must lie in (0,1)")
    rank_tol = _positive_finite(effective_rank_relative_tolerance, "effective_rank_relative_tolerance")
    if rank_tol >= 1.0:
        raise ValueError("effective_rank_relative_tolerance must be less than one")

    coefficients = np.empty((proxy.shape[0], basis.shape[1]), dtype=np.float64)
    for frame_index, target in enumerate(proxy):
        result = lsq_linear(
            basis,
            target,
            bounds=(np.zeros_like(upper), upper),
            method="trf",
            lsq_solver="exact",
            tol=solver_tol,
            max_iter=max_iter,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(
                f"bounded nonnegative least squares failed for proxy frame {frame_index}: {result.message}"
            )
        coefficients[frame_index] = result.x

    decoded_preclip = coefficients @ basis.T
    if not np.all(np.isfinite(decoded_preclip)) or np.min(decoded_preclip) < -UNIT_INTERVAL_TOLERANCE:
        raise RuntimeError("bounded nonnegative decoding produced invalid excitation")
    decoded = np.clip(decoded_preclip, 0.0, 1.0)
    metrics = _coverage_metrics(
        proxy,
        decoded,
        decoded_preclip=decoded_preclip,
        coefficients=coefficients,
        coefficient_upper_bounds=upper,
        phase_id=phases,
        normalization_epsilon=norm_eps,
        saturation_tolerance=saturation_tol,
        basis=basis,
        effective_rank_relative_tolerance=rank_tol,
        active_muscle_rms_threshold=gate_thresholds.active_muscle_rms_threshold,
    )
    threshold_payload = asdict(gate_thresholds)
    threshold_payload["required_phase_ids"] = list(gate_thresholds.required_phase_ids)
    checks = _gate_checks(metrics, threshold_payload, phases_present=phases is not None)
    muscle_schema_sha256 = _json_sha256({"muscle_names": list(artifact.muscle_names)})
    proxy_fingerprint = proxy_content_fingerprint(
        proxy,
        muscle_names=artifact.muscle_names,
        phase_id=phases,
        phase_schema=canonical_phase_schema,
    )
    producer_binding = _validate_proxy_producer_binding(
        proxy_producer_binding,
        proxy_fingerprint=proxy_fingerprint,
        phase_schema_fingerprint=(
            None if canonical_phase_schema is None else canonical_phase_schema["phase_schema_fingerprint"]
        ),
        observed_phase_ids=([] if phases is None else [int(value) for value in np.unique(phases).tolist()]),
        sample_count=int(proxy.shape[0]),
    )
    proxy_binding: dict[str, Any] = {
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "content_fingerprint": proxy_fingerprint,
        "shape": list(proxy.shape),
        "muscle_schema_sha256": muscle_schema_sha256,
        "phase_id_included": phases is not None,
        "observed_phase_ids": [] if phases is None else [int(value) for value in np.unique(phases).tolist()],
        "phase_schema": canonical_phase_schema,
        "phase_schema_fingerprint": (
            None if canonical_phase_schema is None else canonical_phase_schema["phase_schema_fingerprint"]
        ),
    }
    if producer_binding is not None:
        proxy_binding["producer_binding"] = producer_binding
    report: dict[str, Any] = {
        "schema_version": (
            STATIC_PROXY_COVERAGE_SCHEMA_VERSION
            if producer_binding is None
            else FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION
        ),
        "evidence_kind": STATIC_PROXY_EVIDENCE_KIND,
        "limitations": list(_LIMITATIONS),
        "formal_basis_binding": {
            "artifact_schema_version": BASIS_ARTIFACT_SCHEMA_VERSION,
            "artifact_fingerprint": artifact.fingerprint,
            "signal_kind": EXCITATION_SIGNAL_KIND,
            "basis_shape": list(basis.shape),
            "muscle_names": list(artifact.muscle_names),
            "muscle_schema_sha256": muscle_schema_sha256,
        },
        "proxy_binding": proxy_binding,
        "proxy_fingerprint": proxy_fingerprint,
        "solver": {
            "objective": "per_frame_bounded_nonnegative_least_squares",
            "algorithm": "scipy.optimize.lsq_linear_trf_exact",
            "coefficient_lower_bound": 0.0,
            "coefficient_upper_bounds": upper.tolist(),
            "solver_tolerance": solver_tol,
            "solver_max_iterations": max_iter,
            "normalization_epsilon": norm_eps,
            "saturation_tolerance": saturation_tol,
            "effective_rank_relative_tolerance": rank_tol,
            "solved_frame_count": int(proxy.shape[0]),
            "coefficient_solution_fingerprint": _array_fingerprint(coefficients),
            "decoded_solution_fingerprint": _array_fingerprint(decoded),
        },
        "metrics": metrics,
        "thresholds": threshold_payload,
        "checks": checks,
        "passed": _checks_passed(checks),
    }
    report["artifact_fingerprint"] = _report_fingerprint(report)
    return _validate_static_proxy_report(report, expected_basis_fingerprint=artifact.fingerprint, require_passed=False)


def write_static_proxy_coverage_gate(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and atomically write a static-proxy gate artifact."""

    validated = _validate_static_proxy_report(dict(report), expected_basis_fingerprint=None, require_passed=False)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return load_static_proxy_coverage_gate(output, require_passed=False)


def load_static_proxy_coverage_gate(
    path: str | Path,
    expected_basis_fingerprint: str | None = None,
    require_passed: bool = True,
) -> dict[str, Any]:
    """Load a strict, self-fingerprinted static-proxy coverage artifact.

    ``require_passed=True`` is the production default: a valid report that did
    not meet every bound still raises instead of being treated as permission to
    start an early-synergy training run.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"static proxy coverage gate does not exist: {source}")
    payload = load_json_strict(source)
    if not isinstance(payload, dict):
        raise ValueError("static proxy coverage gate must contain a JSON object")
    return _validate_static_proxy_report(
        payload,
        expected_basis_fingerprint=expected_basis_fingerprint,
        require_passed=require_passed,
    )


# Explicit aliases keep call sites readable without changing the evidence kind.
build_static_proxy_coverage_gate = evaluate_static_proxy_coverage
save_static_proxy_coverage_gate = write_static_proxy_coverage_gate


def _coverage_metrics(
    proxy: np.ndarray,
    decoded: np.ndarray,
    *,
    decoded_preclip: np.ndarray,
    coefficients: np.ndarray,
    coefficient_upper_bounds: np.ndarray,
    phase_id: np.ndarray | None,
    normalization_epsilon: float,
    saturation_tolerance: float,
    basis: np.ndarray,
    effective_rank_relative_tolerance: float,
    active_muscle_rms_threshold: float,
) -> dict[str, Any]:
    singular_values = np.linalg.svd(basis, compute_uv=False)
    largest = float(singular_values[0])
    effective_rank = int(np.count_nonzero(singular_values > largest * effective_rank_relative_tolerance))
    rank_fraction = float(effective_rank / basis.shape[1])
    condition = None
    if effective_rank == basis.shape[1] and float(singular_values[-1]) > 0.0:
        candidate = largest / float(singular_values[-1])
        if math.isfinite(candidate):
            condition = float(candidate)
    per_phase: dict[str, Any] = {}
    if phase_id is not None:
        for phase in np.unique(phase_id):
            mask = phase_id == phase
            per_phase[str(int(phase))] = _error_metrics(
                proxy[mask],
                decoded[mask],
                normalization_epsilon=normalization_epsilon,
            )
    return {
        "global": _error_metrics(proxy, decoded, normalization_epsilon=normalization_epsilon),
        "per_phase": per_phase,
        "proxy_active_muscle_fraction": float(
            np.mean(np.sqrt(np.mean(np.square(proxy), axis=0)) >= active_muscle_rms_threshold)
        ),
        "decoded_saturation_fraction": float(np.mean(decoded >= 1.0 - saturation_tolerance)),
        "decoded_preclip_upper_violation_fraction": float(np.mean(decoded_preclip > 1.0 + UNIT_INTERVAL_TOLERANCE)),
        "coefficient_upper_saturation_fraction": float(
            np.mean(coefficients >= coefficient_upper_bounds[None, :] - saturation_tolerance)
        ),
        "basis_condition_number": condition,
        "basis_effective_rank": effective_rank,
        "basis_effective_rank_fraction": rank_fraction,
        "basis_singular_values": [float(value) for value in singular_values.tolist()],
    }


def _error_metrics(
    target: np.ndarray,
    decoded: np.ndarray,
    *,
    normalization_epsilon: float,
) -> dict[str, Any]:
    residual = target - decoded
    target_energy = float(np.sum(np.square(target)))
    residual_energy = float(np.sum(np.square(residual)))
    denominator = target_energy + normalization_epsilon
    squared_loss = residual_energy / denominator
    return {
        "sample_count": int(target.shape[0]),
        "target_l2_norm": float(math.sqrt(target_energy)),
        "residual_l2_norm": float(math.sqrt(residual_energy)),
        "rmse": float(math.sqrt(np.mean(np.square(residual)))),
        "target_rms": float(math.sqrt(np.mean(np.square(target)))),
        "relative_l2_nrmse": float(math.sqrt(squared_loss)),
        "relative_squared_loss": float(squared_loss),
    }


def _gate_checks(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    phases_present: bool,
) -> dict[str, Any]:
    per_phase_metrics = metrics["per_phase"]
    phase_checks = {
        phase: bool(values["relative_l2_nrmse"] <= thresholds["max_phase_relative_l2_nrmse"])
        for phase, values in per_phase_metrics.items()
    }
    observed = {int(value) for value in per_phase_metrics}
    required = {int(value) for value in thresholds["required_phase_ids"]}
    condition = metrics["basis_condition_number"]
    return {
        "proxy_target_rms": bool(metrics["global"]["target_rms"] >= thresholds["min_proxy_target_rms"]),
        "proxy_active_muscle_fraction": bool(
            metrics["proxy_active_muscle_fraction"] >= thresholds["min_active_muscle_fraction"]
        ),
        "global_relative_l2_nrmse": bool(
            metrics["global"]["relative_l2_nrmse"] <= thresholds["max_global_relative_l2_nrmse"]
        ),
        "all_observed_phases_relative_l2_nrmse": bool(not phases_present or all(phase_checks.values())),
        "required_phase_presence": bool(required.issubset(observed)),
        "decoded_saturation_fraction": bool(
            metrics["decoded_saturation_fraction"] <= thresholds["max_decoded_saturation_fraction"]
        ),
        "basis_condition_number": bool(condition is not None and condition <= thresholds["max_basis_condition_number"]),
        "basis_effective_rank_fraction": bool(
            metrics["basis_effective_rank_fraction"] >= thresholds["min_effective_rank_fraction"]
        ),
        "per_phase_relative_l2_nrmse": phase_checks,
        "required_phase_target_rms": {
            str(phase): bool(
                str(phase) in per_phase_metrics
                and per_phase_metrics[str(phase)]["target_rms"] >= thresholds["min_proxy_target_rms"]
            )
            for phase in sorted(required)
        },
    }


def _checks_passed(checks: Mapping[str, Any]) -> bool:
    mapping_keys = {"per_phase_relative_l2_nrmse", "required_phase_target_rms"}
    scalar = [value for key, value in checks.items() if key not in mapping_keys]
    mappings = [checks[key] for key in mapping_keys]
    return bool(
        all(value is True for value in scalar)
        and all(value is True for mapping in mappings for value in mapping.values())
    )


def _validate_proxy_producer_binding(
    value: Mapping[str, Any] | None,
    *,
    proxy_fingerprint: str,
    phase_schema_fingerprint: str | None,
    observed_phase_ids: Sequence[int],
    sample_count: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    binding = _require_mapping(value, "proxy producer binding")
    _exact_fields(binding, _PRODUCER_BINDING_FIELDS, "proxy producer binding")
    if binding["producer_manifest_schema_version"] != "chinajump_excitation_proxy_manifest_v2":
        raise ValueError("unsupported coverage proxy producer manifest schema")
    if binding["producer_artifact_kind"] != "chinajump_target_physical_excitation_proxy":
        raise ValueError("coverage proxy producer artifact kind is unsupported")
    if binding["source_kind"] not in {"full_action_teacher", "trajectory_optimizer"}:
        raise ValueError("coverage proxy producer must bind an independent full-action source")
    for field in (
        "producer_manifest_fingerprint",
        "source_manifest_fingerprint",
        "source_qc_fingerprint",
        "proxy_content_fingerprint",
    ):
        _require_sha256(binding[field], field)
    if binding["proxy_content_fingerprint"] != proxy_fingerprint:
        raise ValueError("coverage proxy producer content fingerprint differs from supplied proxy")
    bound_phase = binding["phase_schema_fingerprint"]
    if phase_schema_fingerprint is None:
        raise ValueError("formal coverage proxy producer binding requires phase-conditioned evidence")
    if _require_sha256(bound_phase, "producer phase_schema_fingerprint") != phase_schema_fingerprint:
        raise ValueError("coverage proxy producer phase schema differs from supplied proxy")
    required_phases = _int_list(
        binding["required_phase_ids"],
        "producer required_phase_ids",
        unique=True,
        sorted_values=True,
    )
    if not required_phases or any(value < 0 for value in required_phases):
        raise ValueError("coverage proxy producer required phases must be non-empty and non-negative")
    min_phase_samples = _strict_int(binding["min_phase_samples"], "producer min_phase_samples")
    if min_phase_samples <= 0:
        raise ValueError("coverage proxy producer min_phase_samples must be positive")
    raw_counts = _require_mapping(binding["per_phase_sample_counts"], "producer per_phase_sample_counts")
    counts: dict[int, int] = {}
    for key, value in raw_counts.items():
        phase = _strict_int_string(key, "producer phase count key")
        count = _strict_int(value, f"producer phase count {key}")
        if count <= 0:
            raise ValueError("coverage proxy producer phase counts must be positive")
        counts[phase] = count
    observed = [int(value) for value in observed_phase_ids]
    if sorted(counts) != observed or sum(counts.values()) != int(sample_count):
        raise ValueError("coverage proxy producer phase counts differ from supplied proxy")
    if any(counts.get(phase, 0) < min_phase_samples for phase in required_phases):
        raise ValueError("coverage proxy producer required phase sample floor is not satisfied")
    return dict(binding)


def _validate_static_proxy_report(
    payload: dict[str, Any],
    *,
    expected_basis_fingerprint: str | None,
    require_passed: bool,
) -> dict[str, Any]:
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "static proxy coverage gate")
    schema_version = payload["schema_version"]
    if schema_version not in {
        STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
        FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported static proxy coverage gate schema")
    if payload["evidence_kind"] != STATIC_PROXY_EVIDENCE_KIND:
        raise ValueError("coverage artifact is not static_proxy excitation reconstruction evidence")
    if payload["limitations"] != _LIMITATIONS:
        raise ValueError("static proxy coverage limitations are missing or changed")

    basis = _require_mapping(payload["formal_basis_binding"], "formal_basis_binding")
    _exact_fields(basis, _BASIS_BINDING_FIELDS, "formal_basis_binding")
    if basis["artifact_schema_version"] != BASIS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("static proxy gate binds an unsupported basis artifact schema")
    basis_fingerprint = _require_sha256(basis["artifact_fingerprint"], "formal basis artifact fingerprint")
    if expected_basis_fingerprint is not None:
        expected = _require_sha256(expected_basis_fingerprint, "expected_basis_fingerprint")
        if basis_fingerprint != expected:
            raise ValueError("static proxy coverage basis fingerprint differs from expected formal basis")
    if basis["signal_kind"] != EXCITATION_SIGNAL_KIND:
        raise ValueError("static proxy gate requires a formal physical-excitation basis")
    shape = _shape2(basis["basis_shape"], "formal basis shape")
    names = basis["muscle_names"]
    if (
        not isinstance(names, list)
        or len(names) != shape[0]
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("formal basis muscle_names do not uniquely match basis rows")
    muscle_schema = _require_sha256(basis["muscle_schema_sha256"], "basis muscle schema fingerprint")
    if muscle_schema != _json_sha256({"muscle_names": names}):
        raise ValueError("formal basis muscle schema fingerprint mismatch")

    proxy = _require_mapping(payload["proxy_binding"], "proxy_binding")
    expected_proxy_fields = (
        _PROXY_BINDING_FIELDS
        if schema_version == STATIC_PROXY_COVERAGE_SCHEMA_VERSION
        else _FORMAL_PROXY_BINDING_FIELDS
    )
    _exact_fields(proxy, expected_proxy_fields, "proxy_binding")
    if proxy["signal_kind"] != EXCITATION_SIGNAL_KIND:
        raise ValueError("static proxy signal_kind must be physical excitation")
    _require_sha256(proxy["content_fingerprint"], "proxy content fingerprint")
    if _require_sha256(payload["proxy_fingerprint"], "proxy_fingerprint") != proxy["content_fingerprint"]:
        raise ValueError("top-level proxy_fingerprint differs from proxy binding")
    proxy_shape = _shape2(proxy["shape"], "proxy shape")
    if proxy_shape[1] != shape[0]:
        raise ValueError("proxy width differs from formal basis muscle rows")
    if _require_sha256(proxy["muscle_schema_sha256"], "proxy muscle schema fingerprint") != muscle_schema:
        raise ValueError("proxy and formal basis muscle schemas differ")
    phase_included = _strict_bool(proxy["phase_id_included"], "phase_id_included")
    observed_phases = _int_list(proxy["observed_phase_ids"], "observed_phase_ids", unique=True, sorted_values=True)
    if any(value < 0 for value in observed_phases):
        raise ValueError("observed_phase_ids must be non-negative")
    if phase_included != bool(observed_phases):
        raise ValueError("proxy phase_id_included and observed_phase_ids are inconsistent")
    phase_schema_raw = proxy["phase_schema"]
    phase_schema_fingerprint_raw = proxy["phase_schema_fingerprint"]
    if phase_included:
        if not isinstance(phase_schema_raw, Mapping):
            raise ValueError("phase-conditioned proxy requires a semantic phase_schema")
        phase_schema = canonicalize_static_proxy_phase_schema(phase_schema_raw)
        phase_schema_fingerprint = _require_sha256(
            phase_schema_fingerprint_raw,
            "proxy phase_schema_fingerprint",
        )
        if phase_schema_fingerprint != phase_schema["phase_schema_fingerprint"]:
            raise ValueError("proxy phase schema fingerprint differs from embedded schema")
        declared_phase_ids = {int(item["id"]) for item in phase_schema["phases"]}
        if not set(observed_phases).issubset(declared_phase_ids):
            raise ValueError("observed proxy phases are absent from semantic phase_schema")
    else:
        if phase_schema_raw is not None or phase_schema_fingerprint_raw is not None:
            raise ValueError("proxy without phase ids cannot bind a phase_schema")
        phase_schema = None

    if schema_version == FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION:
        _validate_proxy_producer_binding(
            proxy["producer_binding"],
            proxy_fingerprint=proxy["content_fingerprint"],
            phase_schema_fingerprint=proxy["phase_schema_fingerprint"],
            observed_phase_ids=observed_phases,
            sample_count=proxy_shape[0],
        )

    solver = _require_mapping(payload["solver"], "solver")
    _exact_fields(solver, _SOLVER_FIELDS, "solver")
    if (
        solver["objective"] != "per_frame_bounded_nonnegative_least_squares"
        or solver["algorithm"] != "scipy.optimize.lsq_linear_trf_exact"
        or _finite_float(solver["coefficient_lower_bound"], "coefficient_lower_bound") != 0.0
    ):
        raise ValueError("static proxy solver contract is unsupported")
    upper = _float_list(solver["coefficient_upper_bounds"], "coefficient_upper_bounds")
    if len(upper) != shape[1] or any(value <= 0.0 for value in upper):
        raise ValueError("coefficient upper bounds must be finite, positive, and match basis rank")
    _positive_finite(solver["solver_tolerance"], "solver_tolerance")
    if _strict_int(solver["solver_max_iterations"], "solver_max_iterations") <= 0:
        raise ValueError("solver_max_iterations must be positive")
    _positive_finite(solver["normalization_epsilon"], "normalization_epsilon")
    saturation_tolerance = _positive_finite(solver["saturation_tolerance"], "saturation_tolerance")
    rank_tolerance = _positive_finite(solver["effective_rank_relative_tolerance"], "effective_rank_relative_tolerance")
    if saturation_tolerance >= 1.0 or rank_tolerance >= 1.0:
        raise ValueError("saturation/effective-rank tolerances must be less than one")
    if _strict_int(solver["solved_frame_count"], "solved_frame_count") != proxy_shape[0]:
        raise ValueError("solver frame count differs from proxy shape")
    _require_sha256(solver["coefficient_solution_fingerprint"], "coefficient solution fingerprint")
    _require_sha256(solver["decoded_solution_fingerprint"], "decoded solution fingerprint")

    metrics = _require_mapping(payload["metrics"], "metrics")
    _exact_fields(metrics, _METRICS_FIELDS, "metrics")
    global_metrics = _validate_error_metrics(metrics["global"], "metrics.global")
    if global_metrics["sample_count"] != proxy_shape[0]:
        raise ValueError("global metric sample count differs from proxy")
    per_phase = _require_mapping(metrics["per_phase"], "metrics.per_phase")
    phase_sample_count = 0
    for phase, values in per_phase.items():
        if str(_strict_int_string(phase, "per-phase metric key")) != phase:
            raise ValueError("per-phase metric keys must be canonical integer strings")
        phase_sample_count += _validate_error_metrics(values, f"metrics.per_phase[{phase}]")["sample_count"]
    if set(per_phase) != {str(value) for value in observed_phases}:
        raise ValueError("per-phase metrics differ from observed proxy phases")
    if phase_included and phase_sample_count != proxy_shape[0]:
        raise ValueError("per-phase sample counts do not cover the proxy")
    for name in (
        "proxy_active_muscle_fraction",
        "decoded_saturation_fraction",
        "decoded_preclip_upper_violation_fraction",
        "coefficient_upper_saturation_fraction",
        "basis_effective_rank_fraction",
    ):
        value = _finite_float(metrics[name], name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    condition = metrics["basis_condition_number"]
    if condition is not None and _positive_finite(condition, "basis_condition_number") < 1.0:
        raise ValueError("basis_condition_number cannot be less than one")
    effective_rank = _strict_int(metrics["basis_effective_rank"], "basis_effective_rank")
    if not 0 <= effective_rank <= shape[1]:
        raise ValueError("basis_effective_rank is outside [0,basis rank]")
    expected_rank_fraction = effective_rank / shape[1]
    if not math.isclose(metrics["basis_effective_rank_fraction"], expected_rank_fraction, abs_tol=1e-12):
        raise ValueError("basis effective rank fraction is inconsistent")
    singular_values = _float_list(metrics["basis_singular_values"], "basis_singular_values")
    if len(singular_values) != min(shape) or any(value < 0.0 for value in singular_values):
        raise ValueError("basis singular values do not match basis shape")
    if any(first < second for first, second in pairwise(singular_values)):
        raise ValueError("basis singular values must be descending")
    expected_effective_rank = sum(value > singular_values[0] * rank_tolerance for value in singular_values)
    if effective_rank != expected_effective_rank:
        raise ValueError("basis effective rank differs from the bound singular values")
    expected_condition = None
    if expected_effective_rank == shape[1] and singular_values[-1] > 0.0:
        candidate = singular_values[0] / singular_values[-1]
        if math.isfinite(candidate):
            expected_condition = candidate
    if (condition is None) != (expected_condition is None) or (
        condition is not None
        and expected_condition is not None
        and not math.isclose(condition, expected_condition, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("basis condition number differs from the bound singular values")

    thresholds = _require_mapping(payload["thresholds"], "thresholds")
    _exact_fields(thresholds, _THRESHOLD_FIELDS, "thresholds")
    canonical_thresholds = StaticProxyCoverageThresholds.from_mapping(thresholds)
    if dict(thresholds) != {
        **asdict(canonical_thresholds),
        "required_phase_ids": list(canonical_thresholds.required_phase_ids),
    }:
        raise ValueError("static proxy thresholds are not canonical")
    if phase_schema is not None:
        declared_phase_ids = {int(item["id"]) for item in phase_schema["phases"]}
        if not set(canonical_thresholds.required_phase_ids).issubset(declared_phase_ids):
            raise ValueError("required coverage phases are absent from semantic phase_schema")

    checks = _require_mapping(payload["checks"], "checks")
    _exact_fields(checks, _CHECK_FIELDS, "checks")
    phase_checks = _require_mapping(checks["per_phase_relative_l2_nrmse"], "per-phase checks")
    if set(phase_checks) != set(per_phase) or any(type(value) is not bool for value in phase_checks.values()):
        raise ValueError("per-phase checks do not exactly cover proxy phases")
    activity_checks = _require_mapping(
        checks["required_phase_target_rms"],
        "required-phase target activity checks",
    )
    if set(activity_checks) != {str(value) for value in canonical_thresholds.required_phase_ids} or any(
        type(value) is not bool for value in activity_checks.values()
    ):
        raise ValueError("required-phase target activity checks differ from thresholds")
    for name in _CHECK_FIELDS - {
        "per_phase_relative_l2_nrmse",
        "required_phase_target_rms",
    }:
        _strict_bool(checks[name], f"checks.{name}")
    recomputed_checks = _gate_checks(metrics, thresholds, phases_present=phase_included)
    if checks != recomputed_checks:
        raise ValueError("static proxy coverage checks are stale or inconsistent")
    passed = _strict_bool(payload["passed"], "passed")
    if passed != _checks_passed(checks):
        raise ValueError("static proxy coverage passed flag is stale or inconsistent")
    supplied_fingerprint = _require_sha256(payload["artifact_fingerprint"], "artifact_fingerprint")
    if supplied_fingerprint != _report_fingerprint(payload):
        raise ValueError("static proxy coverage artifact_fingerprint mismatch")
    if require_passed and not passed:
        raise ValueError("static proxy coverage gate did not pass")
    return payload


def _validate_error_metrics(value: Any, field: str) -> Mapping[str, Any]:
    metrics = _require_mapping(value, field)
    _exact_fields(metrics, _ERROR_METRIC_FIELDS, field)
    if _strict_int(metrics["sample_count"], f"{field}.sample_count") <= 0:
        raise ValueError(f"{field}.sample_count must be positive")
    for name in _ERROR_METRIC_FIELDS - {"sample_count"}:
        if _finite_float(metrics[name], f"{field}.{name}") < 0.0:
            raise ValueError(f"{field}.{name} must be non-negative")
    if not math.isclose(
        metrics["relative_l2_nrmse"] ** 2,
        metrics["relative_squared_loss"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{field} relative error metrics are inconsistent")
    return metrics


def _validate_formal_excitation_basis(artifact: SynergyBasisArtifact) -> None:
    if artifact.manifest.get("schema_version") != BASIS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("formal basis has an unsupported artifact schema")
    if artifact.manifest.get("signal_kind") != EXCITATION_SIGNAL_KIND:
        raise ValueError("static proxy coverage requires a formal physical-excitation basis")
    _require_sha256(artifact.fingerprint, "formal basis fingerprint")


def _validate_proxy_excitation(values: np.ndarray, *, expected_width: int) -> np.ndarray:
    proxy = np.asarray(values, dtype=np.float64)
    if proxy.ndim != 2 or proxy.shape[0] <= 0 or proxy.shape[1] != expected_width:
        raise ValueError(f"proxy_excitation must have shape [T,{expected_width}] with T > 0")
    if not np.all(np.isfinite(proxy)):
        raise ValueError("proxy_excitation contains NaN/Inf")
    if np.min(proxy) < -UNIT_INTERVAL_TOLERANCE or np.max(proxy) > 1.0 + UNIT_INTERVAL_TOLERANCE:
        raise ValueError("proxy_excitation must be unit physical excitation in [0,1]")
    return np.clip(proxy, 0.0, 1.0)


def _validate_phase_id(values: np.ndarray | None, *, sample_count: int) -> np.ndarray | None:
    if values is None:
        return None
    phases = np.asarray(values)
    if phases.ndim != 1 or phases.shape[0] != sample_count:
        raise ValueError("phase_id must have shape [T] matching proxy_excitation")
    if not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("phase_id must contain integers; float truncation is forbidden")
    result = phases.astype(np.int64)
    if np.any(result < 0):
        raise ValueError("phase_id must be non-negative")
    return result


def _coefficient_upper_bounds(values: float | Sequence[float] | np.ndarray, *, rank: int) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.full(rank, float(raw), dtype=np.float64)
    if raw.shape != (rank,) or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(f"coefficient_upper_bounds must be one positive finite value or shape [{rank}]")
    return raw


def _array_fingerprint(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"shape": list(values.shape), "dtype": "little_endian_float64"}))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _report_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {str(key): value for key, value in payload.items() if key != "artifact_fingerprint"}
    return _json_sha256(canonical)


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise ValueError(f"{field} fields differ from schema (missing={missing}, unknown={unknown})")


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_finite(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _strict_int_string(value: Any, field: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{field} must be a non-negative canonical integer string")
    return int(value)


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _shape2(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain two dimensions")
    shape = tuple(_strict_int(item, field) for item in value)
    if min(shape) <= 0:
        raise ValueError(f"{field} dimensions must be positive")
    return shape


def _float_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [_finite_float(item, field) for item in value]


def _int_list(
    value: Any,
    field: str,
    *,
    unique: bool,
    sorted_values: bool,
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = [_strict_int(item, field) for item in value]
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique values")
    if sorted_values and result != sorted(result):
        raise ValueError(f"{field} must be sorted")
    return result


def _parse_coefficient_upper_bounds(text: str, *, rank: int) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--coefficient-upper-bounds must contain comma-separated numbers") from exc
    if not values:
        raise ValueError("--coefficient-upper-bounds cannot be empty")
    return _coefficient_upper_bounds(values[0] if len(values) == 1 else values, rank=rank)


def _load_proxy_npz(
    path: str | Path,
    *,
    proxy_field: str,
    phase_field: str | None,
    formal_muscle_names: tuple[str, ...],
    assume_basis_order: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"proxy NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        if proxy_field not in data.files:
            raise ValueError(f"proxy NPZ is missing field {proxy_field!r}")
        proxy = np.asarray(data[proxy_field])
        phase = None if phase_field is None else np.asarray(data[phase_field]) if phase_field in data.files else None
        name_field = next((name for name in ("actuator_names", "muscle_names") if name in data.files), None)
        if name_field is None:
            if not assume_basis_order:
                raise ValueError(
                    "proxy NPZ lacks actuator_names/muscle_names; pass --assume-basis-order only after external schema verification"
                )
        else:
            proxy_names = tuple(np.asarray(data[name_field]).astype(str).tolist())
            if proxy_names != formal_muscle_names:
                raise ValueError("proxy NPZ muscle names/order differ from formal basis")
        if phase_field is not None and phase is None:
            raise ValueError(f"proxy NPZ is missing requested phase field {phase_field!r}")
    return proxy, phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bounded static-proxy excitation coverage for a formal physical-excitation synergy basis. "
            "This is not a short-horizon dynamics oracle."
        )
    )
    parser.add_argument("--basis-artifact", required=True, help="formal synergy basis directory or manifest.json")
    parser.add_argument("--proxy-npz", default=None, help="NPZ containing a target physical excitation proxy")
    parser.add_argument(
        "--proxy-manifest",
        default=None,
        help=(
            "self-fingerprinted chinajump_excitation_proxy_manifest_v2; formal "
            "coverage gates bind this producer provenance"
        ),
    )
    parser.add_argument("--output", required=True, help="output static_proxy gate JSON")
    parser.add_argument("--proxy-field", default="physical_excitation")
    parser.add_argument("--phase-field", default="phase_id")
    parser.add_argument(
        "--phase-schema",
        default=None,
        help=("JSON binding each phase id to a semantic name/definition; required unless --without-phase-id is used"),
    )
    parser.add_argument("--without-phase-id", action="store_true")
    parser.add_argument(
        "--assume-basis-order",
        action="store_true",
        help="explicitly accept a proxy NPZ without embedded ordered actuator names",
    )
    parser.add_argument(
        "--allow-unbound-proxy",
        action="store_true",
        help="legacy analysis only: permit a proxy that has no producer manifest",
    )
    parser.add_argument(
        "--coefficient-upper-bounds",
        default=None,
        help="one value or comma-separated K values; defaults to 1.0 when no stats artifact is supplied",
    )
    parser.add_argument(
        "--coefficient-stats",
        default=None,
        help="early-synergy coefficient_stats.npz bound to the formal basis",
    )
    parser.add_argument(
        "--coefficient-max-source",
        choices=("train_q99_times_1p2", "train_q99"),
        default="train_q99_times_1p2",
        help="runtime coefficient cap to derive from --coefficient-stats",
    )
    parser.add_argument("--max-global-relative-l2-nrmse", type=float, default=0.15)
    parser.add_argument("--min-proxy-target-rms", type=float, default=1e-4)
    parser.add_argument("--min-active-muscle-fraction", type=float, default=0.05)
    parser.add_argument("--active-muscle-rms-threshold", type=float, default=1e-4)
    parser.add_argument("--max-phase-relative-l2-nrmse", type=float, default=0.25)
    parser.add_argument("--max-decoded-saturation-fraction", type=float, default=0.05)
    parser.add_argument("--max-basis-condition-number", type=float, default=100.0)
    parser.add_argument("--min-effective-rank-fraction", type=float, default=0.80)
    parser.add_argument("--required-phase-id", type=int, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; writes the report even when the gate fails."""

    args = _build_parser().parse_args(argv)
    artifact = load_synergy_basis(args.basis_artifact)
    _validate_formal_excitation_basis(artifact)
    proxy_artifact = None
    proxy_path = None if args.proxy_npz is None else Path(args.proxy_npz)
    manifest_path = None if args.proxy_manifest is None else Path(args.proxy_manifest)
    if manifest_path is None and proxy_path is not None:
        sibling_manifest = proxy_path.with_name("proxy_manifest.json")
        if sibling_manifest.is_file():
            manifest_path = sibling_manifest
    if manifest_path is not None:
        from musclemimic.synergy.coverage_proxy import load_coverage_proxy_artifact

        proxy_artifact = load_coverage_proxy_artifact(manifest_path)
        if proxy_path is not None and proxy_path.resolve() != proxy_artifact.npz_path:
            raise ValueError("--proxy-npz differs from the NPZ bound by --proxy-manifest")
        proxy_path = proxy_artifact.npz_path
    if proxy_path is None:
        raise ValueError("static coverage requires --proxy-manifest or --proxy-npz")
    if args.assume_basis_order and not args.allow_unbound_proxy:
        raise ValueError(
            "--assume-basis-order is forbidden in formal/default coverage mode; use an ordered producer artifact"
        )
    if proxy_artifact is not None and args.allow_unbound_proxy:
        raise ValueError("--allow-unbound-proxy cannot discard an available producer manifest")
    proxy, phase = _load_proxy_npz(
        proxy_path,
        proxy_field=args.proxy_field,
        phase_field=None if args.without_phase_id else args.phase_field,
        formal_muscle_names=artifact.muscle_names,
        assume_basis_order=bool(args.assume_basis_order),
    )
    if args.without_phase_id:
        if args.phase_schema is not None:
            raise ValueError("--phase-schema cannot be used with --without-phase-id")
        phase_schema = None
    else:
        if args.phase_schema is None:
            raise ValueError("phase-conditioned coverage requires --phase-schema")
        phase_schema = load_static_proxy_phase_schema(args.phase_schema)
    if proxy_artifact is not None:
        manifest_phase = proxy_artifact.manifest["phase_binding"]
        if phase_schema is None or (
            phase_schema["phase_schema_fingerprint"] != manifest_phase["phase_schema_fingerprint"]
        ):
            raise ValueError("coverage CLI phase schema differs from producer manifest")
    if args.coefficient_stats is not None:
        if args.coefficient_upper_bounds is not None:
            raise ValueError("use either --coefficient-stats or --coefficient-upper-bounds, not both")
        from musclemimic.synergy.action_interface import load_coefficient_statistics

        stats = load_coefficient_statistics(
            args.coefficient_stats,
            expected_basis_fingerprint=artifact.fingerprint,
            expected_rank=artifact.basis.shape[1],
        )
        multiplier = 1.2 if args.coefficient_max_source == "train_q99_times_1p2" else 1.0
        upper = multiplier * np.asarray(stats["coefficient_q99"], dtype=np.float64)
    else:
        upper = _parse_coefficient_upper_bounds(
            "1.0" if args.coefficient_upper_bounds is None else args.coefficient_upper_bounds,
            rank=artifact.basis.shape[1],
        )
    thresholds = StaticProxyCoverageThresholds(
        min_proxy_target_rms=args.min_proxy_target_rms,
        min_active_muscle_fraction=args.min_active_muscle_fraction,
        active_muscle_rms_threshold=args.active_muscle_rms_threshold,
        max_global_relative_l2_nrmse=args.max_global_relative_l2_nrmse,
        max_phase_relative_l2_nrmse=args.max_phase_relative_l2_nrmse,
        max_decoded_saturation_fraction=args.max_decoded_saturation_fraction,
        max_basis_condition_number=args.max_basis_condition_number,
        min_effective_rank_fraction=args.min_effective_rank_fraction,
        required_phase_ids=tuple(args.required_phase_id),
    ).validated()
    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        phase_id=phase,
        phase_schema=phase_schema,
        coefficient_upper_bounds=upper,
        thresholds=thresholds,
        proxy_muscle_names=artifact.muscle_names,
        proxy_producer_binding=(None if proxy_artifact is None else proxy_artifact.oracle_binding),
    )
    write_static_proxy_coverage_gate(args.output, report)
    print(
        json.dumps(
            {
                "evidence_kind": report["evidence_kind"],
                "passed": report["passed"],
                "artifact_fingerprint": report["artifact_fingerprint"],
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI entry point
    raise SystemExit(main())
