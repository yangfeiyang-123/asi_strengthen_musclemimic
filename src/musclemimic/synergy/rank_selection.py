"""Fail-closed rank-selection contracts for regional synergy bases.

The helpers in this module are intentionally independent of NMF fitting.  A
candidate rank is deployable only when every required gate explicitly passes;
there is no "best available" fallback.  Optional dynamic evidence is kept
separate from static reconstruction metrics so a static proxy can never be
mislabelled as simulator-rollout coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from musclemimic.synergy.graph_nmf import validate_graph_regularization_manifest

DYNAMIC_COVERAGE_SCHEMA_VERSION = "synergy_rank_dynamic_coverage_gate_v1"
DYNAMIC_COVERAGE_EVIDENCE_KIND = "environment_rollout_dynamic_coverage"
DYNAMIC_COVERAGE_REQUIREMENT_SCHEMA_VERSION = "synergy_rank_dynamic_coverage_requirement_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_COVERAGE_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_kind",
        "signal_kind",
        "region",
        "rank",
        "candidate_basis_fingerprint",
        "rollout_manifest_fingerprint",
        "environment_fingerprint",
        "metrics",
        "thresholds",
        "checks",
        "passed",
        "artifact_fingerprint",
    }
)


class BasisNotEligibleForEarlyControl(ValueError):  # noqa: N818
    """Raised when no candidate satisfies every required deployment gate."""


def dynamic_coverage_requirement(
    *,
    required: bool,
    max_mean_dynamic_gap: float,
    max_key_phase_dynamic_gap: float,
    expected_environment_fingerprint: str | None,
    expected_rollout_manifest_fingerprint: str | None,
) -> dict[str, Any]:
    """Build the canonical selection-manifest requirement for dynamic coverage."""

    payload = {
        "schema_version": DYNAMIC_COVERAGE_REQUIREMENT_SCHEMA_VERSION,
        "required": required,
        "evidence_kind": DYNAMIC_COVERAGE_EVIDENCE_KIND,
        "max_mean_dynamic_gap": max_mean_dynamic_gap,
        "max_key_phase_dynamic_gap": max_key_phase_dynamic_gap,
        "expected_environment_fingerprint": expected_environment_fingerprint,
        "expected_rollout_manifest_fingerprint": expected_rollout_manifest_fingerprint,
    }
    return validate_dynamic_coverage_requirement(payload)


def validate_dynamic_coverage_requirement(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pinned dynamic gate without inferring evidence or a pass."""

    expected_fields = {
        "schema_version",
        "required",
        "evidence_kind",
        "max_mean_dynamic_gap",
        "max_key_phase_dynamic_gap",
        "expected_environment_fingerprint",
        "expected_rollout_manifest_fingerprint",
    }
    if not isinstance(report, Mapping) or set(report) != expected_fields:
        raise ValueError("dynamic coverage requirement fields differ from contract")
    if report.get("schema_version") != DYNAMIC_COVERAGE_REQUIREMENT_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic coverage requirement schema")
    if type(report.get("required")) is not bool:
        raise ValueError("dynamic coverage requirement required must be boolean")
    if report.get("evidence_kind") != DYNAMIC_COVERAGE_EVIDENCE_KIND:
        raise ValueError("dynamic coverage requirement must name environment-rollout evidence")
    expected_environment = report.get("expected_environment_fingerprint")
    expected_rollout = report.get("expected_rollout_manifest_fingerprint")
    if (expected_environment is None) != (expected_rollout is None):
        raise ValueError("dynamic coverage requirement must pin environment and rollout fingerprints together")
    if report["required"] is True and expected_environment is None:
        raise ValueError("required dynamic coverage must pin expected environment and rollout fingerprints")
    if expected_environment is not None:
        expected_environment = _require_sha256(
            expected_environment,
            "expected environment fingerprint",
        )
        expected_rollout = _require_sha256(
            expected_rollout,
            "expected rollout manifest fingerprint",
        )
    return {
        "schema_version": DYNAMIC_COVERAGE_REQUIREMENT_SCHEMA_VERSION,
        "required": report["required"],
        "evidence_kind": DYNAMIC_COVERAGE_EVIDENCE_KIND,
        "max_mean_dynamic_gap": _finite_nonnegative(
            report["max_mean_dynamic_gap"],
            "max_mean_dynamic_gap",
        ),
        "max_key_phase_dynamic_gap": _finite_nonnegative(
            report["max_key_phase_dynamic_gap"],
            "max_key_phase_dynamic_gap",
        ),
        "expected_environment_fingerprint": expected_environment,
        "expected_rollout_manifest_fingerprint": expected_rollout,
    }


def canonical_candidate_ranks(values: Sequence[int], *, label: str = "ranks") -> tuple[int, ...]:
    """Return sorted unique positive ranks without accepting boolean aliases."""

    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of positive integers")
    ranks: list[int] = []
    for raw in values:
        if isinstance(raw, bool | np.bool_) or not isinstance(raw, int | np.integer):
            raise ValueError(f"{label} must contain only positive integers")
        rank = int(raw)
        if rank <= 0:
            raise ValueError(f"{label} must contain only positive integers")
        ranks.append(rank)
    canonical = tuple(sorted(set(ranks)))
    if not canonical:
        raise ValueError(f"{label} must contain at least one positive integer")
    return canonical


def canonical_region_candidate_ranks(
    value: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    """Validate optional per-region rank grids while preserving region labels."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("region_ranks must be a mapping from region name to candidate ranks")
    result: dict[str, tuple[int, ...]] = {}
    for raw_region, raw_ranks in value.items():
        if not isinstance(raw_region, str) or not raw_region.strip() or raw_region != raw_region.strip():
            raise ValueError("region_ranks keys must be non-empty canonical strings")
        if raw_region in result:
            raise ValueError(f"duplicate region_ranks entry for {raw_region!r}")
        result[raw_region] = canonical_candidate_ranks(
            raw_ranks,
            label=f"region_ranks[{raw_region!r}]",
        )
    return result


def candidate_ranks_for_region(
    default_ranks: Sequence[int],
    region_ranks: Mapping[str, Sequence[int]] | None,
    *,
    region: str,
) -> tuple[int, ...]:
    """Resolve one region's grid, falling back only to the configured default grid."""

    default = canonical_candidate_ranks(default_ranks)
    overrides = canonical_region_candidate_ranks(region_ranks)
    return overrides.get(str(region), default)


def select_smallest_eligible_rank(
    rank_reports: Mapping[int, Mapping[str, Any]],
    *,
    region: str,
) -> int:
    """Select the smallest explicitly eligible rank or fail closed."""

    if not isinstance(rank_reports, Mapping) or not rank_reports:
        raise BasisNotEligibleForEarlyControl(f"region {region!r} has no evaluated candidate ranks")
    eligible: list[int] = []
    for raw_rank, report in rank_reports.items():
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise ValueError("rank report keys must be integer ranks")
        if not isinstance(report, Mapping) or type(report.get("eligible")) is not bool:
            raise ValueError(f"rank report {raw_rank} lacks an explicit boolean eligible gate")
        if report["eligible"] is True:
            eligible.append(int(raw_rank))
    if not eligible:
        raise BasisNotEligibleForEarlyControl(f"region {region!r} has no rank passing every required gate")
    return min(eligible)


def enforce_total_rank_budget(
    regional_ranks: Mapping[str, int],
    *,
    total_rank_budget: int | None,
) -> int:
    """Validate a composite rank sum; never truncate components to fit a budget."""

    if not isinstance(regional_ranks, Mapping) or not regional_ranks:
        raise ValueError("regional_ranks must contain at least one component")
    canonical: dict[str, int] = {}
    for region, raw_rank in regional_ranks.items():
        if not isinstance(region, str) or not region.strip():
            raise ValueError("regional_ranks keys must be non-empty strings")
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int | np.integer) or int(raw_rank) <= 0:
            raise ValueError(f"regional rank for {region!r} must be a positive integer")
        canonical[region] = int(raw_rank)
    total = sum(canonical.values())
    if total_rank_budget is None:
        return total
    if (
        isinstance(total_rank_budget, bool)
        or not isinstance(total_rank_budget, int | np.integer)
        or int(total_rank_budget) <= 0
    ):
        raise ValueError("total_rank_budget must be a positive integer or null")
    budget = int(total_rank_budget)
    if total > budget:
        detail = ", ".join(f"{region}={rank}" for region, rank in canonical.items())
        raise BasisNotEligibleForEarlyControl(
            f"regional composite rank {total} exceeds explicit total_rank_budget={budget} ({detail}); "
            "component truncation is forbidden"
        )
    return total


def candidate_basis_fingerprint(
    basis: np.ndarray,
    *,
    muscle_names: Sequence[str],
    signal_kind: str,
    region: str,
    graph_regularization: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint the exact candidate matrix consumed by a dynamic gate."""

    matrix = np.asarray(basis, dtype=np.float64)
    names = tuple(str(name) for name in muscle_names)
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or not np.all(np.isfinite(matrix)) or np.min(matrix) < -1e-10:
        raise ValueError("candidate basis must be a finite non-negative non-empty matrix")
    if matrix.shape[0] != len(names) or len(set(names)) != len(names):
        raise ValueError("candidate basis rows must match unique ordered muscle_names")
    stored_matrix = np.ascontiguousarray(matrix.astype("<f4"))
    if not np.all(np.isfinite(stored_matrix)):
        raise ValueError("candidate basis cannot be represented by the persisted float32 artifact")
    payload = {
        "schema_version": "synergy_rank_candidate_basis_v1",
        "signal_kind": str(signal_kind),
        "region": str(region),
        "muscle_names": list(names),
        "shape": list(matrix.shape),
        # Basis artifacts are persisted as float32.  Dynamic rollouts must bind
        # those exact stored values, not higher-precision transient fit values.
        "float32_c_order_sha256": hashlib.sha256(stored_matrix.tobytes()).hexdigest(),
    }
    if graph_regularization is not None:
        payload["graph_regularization"] = validate_graph_regularization_manifest(graph_regularization)
    return _json_sha256(payload)


def validate_dynamic_coverage_gate(
    report: Mapping[str, Any],
    *,
    region: str,
    rank: int,
    candidate_fingerprint: str,
    signal_kind: str,
    max_mean_dynamic_gap: float,
    max_key_phase_dynamic_gap: float,
    expected_environment_fingerprint: str | None,
    expected_rollout_manifest_fingerprint: str | None,
) -> dict[str, Any]:
    """Validate separately produced simulator-rollout coverage evidence.

    Static proxy reports are rejected by the exact ``evidence_kind`` check.
    The validator does not synthesize missing metrics or infer a pass from a
    reconstruction score.
    """

    if not isinstance(report, Mapping):
        raise ValueError("dynamic coverage evidence must be a mapping")
    fields = set(report)
    missing = sorted(_DYNAMIC_COVERAGE_FIELDS - fields)
    unknown = sorted(fields - _DYNAMIC_COVERAGE_FIELDS)
    if missing or unknown:
        raise ValueError(f"dynamic coverage fields differ from contract (missing={missing}, unknown={unknown})")
    if report.get("schema_version") != DYNAMIC_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic coverage schema")
    if report.get("evidence_kind") != DYNAMIC_COVERAGE_EVIDENCE_KIND:
        raise ValueError("rank dynamic coverage requires environment-rollout evidence; static proxy is insufficient")
    if report.get("signal_kind") != str(signal_kind):
        raise ValueError("dynamic coverage signal-kind binding mismatch")
    report_rank = report.get("rank")
    if report.get("region") != str(region) or type(report_rank) is not int or report_rank != int(rank):
        raise ValueError("dynamic coverage region/rank binding mismatch")
    expected_candidate = _require_sha256(candidate_fingerprint, "candidate fingerprint")
    if report.get("candidate_basis_fingerprint") != expected_candidate:
        raise ValueError("dynamic coverage candidate basis fingerprint mismatch")
    report_rollout = _require_sha256(
        report.get("rollout_manifest_fingerprint"),
        "rollout manifest fingerprint",
    )
    report_environment = _require_sha256(
        report.get("environment_fingerprint"),
        "environment fingerprint",
    )
    if (expected_environment_fingerprint is None) != (expected_rollout_manifest_fingerprint is None):
        raise ValueError("expected environment and rollout manifest fingerprints must be pinned together")
    if expected_environment_fingerprint is not None:
        expected_environment = _require_sha256(
            expected_environment_fingerprint,
            "expected environment fingerprint",
        )
        expected_rollout = _require_sha256(
            expected_rollout_manifest_fingerprint,
            "expected rollout manifest fingerprint",
        )
        if report_environment != expected_environment:
            raise ValueError("dynamic coverage environment fingerprint mismatch")
        if report_rollout != expected_rollout:
            raise ValueError("dynamic coverage rollout manifest fingerprint mismatch")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "mean_dynamic_gap",
        "max_key_phase_dynamic_gap",
        "rollout_count",
        "key_phase_count",
        "horizon_steps",
    }:
        raise ValueError("dynamic coverage metrics differ from the short-horizon contract")
    mean_gap = _finite_nonnegative(metrics["mean_dynamic_gap"], "mean_dynamic_gap")
    phase_gap = _finite_nonnegative(
        metrics["max_key_phase_dynamic_gap"],
        "max_key_phase_dynamic_gap",
    )
    for field in ("rollout_count", "key_phase_count", "horizon_steps"):
        value = metrics[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"dynamic coverage {field} must be a positive integer")
    expected_thresholds = {
        "max_mean_dynamic_gap": _finite_nonnegative(
            max_mean_dynamic_gap,
            "configured max_mean_dynamic_gap",
        ),
        "max_key_phase_dynamic_gap": _finite_nonnegative(
            max_key_phase_dynamic_gap,
            "configured max_key_phase_dynamic_gap",
        ),
    }
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(expected_thresholds):
        raise ValueError("dynamic coverage thresholds differ from contract")
    supplied_thresholds = {key: _finite_nonnegative(thresholds[key], key) for key in expected_thresholds}
    if supplied_thresholds != expected_thresholds:
        raise ValueError("dynamic coverage thresholds differ from configured promotion thresholds")
    expected_checks = {
        "mean_dynamic_gap": mean_gap <= expected_thresholds["max_mean_dynamic_gap"],
        "key_phase_dynamic_gap": phase_gap <= expected_thresholds["max_key_phase_dynamic_gap"],
        "nonempty_rollout_evidence": True,
    }
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(expected_checks)
        or any(type(value) is not bool for value in checks.values())
    ):
        raise ValueError("dynamic coverage checks differ from the short-horizon contract")
    if dict(checks) != expected_checks:
        raise ValueError("dynamic coverage checks are stale or inconsistent with metrics")
    passed = report.get("passed")
    if type(passed) is not bool or passed is not all(checks.values()):
        raise ValueError("dynamic coverage passed flag is stale or inconsistent with checks")
    expected_artifact = dynamic_coverage_artifact_fingerprint(report)
    if report.get("artifact_fingerprint") != expected_artifact:
        raise ValueError("dynamic coverage artifact_fingerprint mismatch")
    return {str(key): value for key, value in report.items()}


def dynamic_coverage_artifact_fingerprint(report: Mapping[str, Any]) -> str:
    """Compute the canonical self-fingerprint used by an external gate producer."""

    if not isinstance(report, Mapping):
        raise TypeError("dynamic coverage report must be a mapping")
    return _json_sha256({str(key): value for key, value in report.items() if key != "artifact_fingerprint"})


def dynamic_coverage_report_for_rank(
    reports: Mapping[int | str, Mapping[str, Any]] | None,
    *,
    rank: int,
) -> Mapping[str, Any] | None:
    """Resolve a rank-keyed report without accepting ambiguous JSON/Python keys."""

    if reports is None:
        return None
    if not isinstance(reports, Mapping):
        raise TypeError("dynamic coverage reports must be a rank-keyed mapping")
    matches = [reports[key] for key in (rank, str(rank)) if key in reports]
    if len(matches) > 1:
        raise ValueError(f"dynamic coverage reports contain ambiguous keys for rank {rank}")
    return None if not matches else matches[0]


def validate_dynamic_coverage_rank_inventory(
    reports: Mapping[int | str, Mapping[str, Any]] | None,
    *,
    candidate_ranks: Sequence[int],
    label: str = "dynamic coverage inventory",
) -> None:
    """Reject ambiguous, stale, or otherwise silently ignored rank reports."""

    if reports is None:
        return
    if not isinstance(reports, Mapping):
        raise TypeError(f"{label} must be a rank-keyed mapping")
    expected = set(canonical_candidate_ranks(candidate_ranks, label="candidate_ranks"))
    observed: set[int] = set()
    for raw_rank in reports:
        if isinstance(raw_rank, bool):
            raise ValueError(f"{label} rank keys cannot be boolean")
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} rank keys must be positive integers") from exc
        if rank <= 0 or str(rank) != str(raw_rank):
            raise ValueError(f"{label} rank keys must use canonical positive integers")
        if rank in observed:
            raise ValueError(f"{label} has ambiguous duplicate rank {rank}")
        if rank not in expected:
            raise ValueError(f"{label} contains unconfigured or unevaluated rank {rank}")
        observed.add(rank)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return digest


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result
