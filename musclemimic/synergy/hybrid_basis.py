"""Fail-closed construction of a hybrid regional/global synergy basis.

The hybrid basis always starts with the complete regional basis.  A global
component may only be appended as its original non-negative column: signed
projection residuals are diagnostics and are never used as basis columns.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from musclemimic.synergy.basis_artifact import SynergyBasisArtifact, save_synergy_basis
from musclemimic.synergy.metrics import global_vaf, local_vaf
from musclemimic.synergy.nmf import transform_nmf

HYBRID_BASIS_SCHEMA_VERSION = "hybrid_global_regional_basis_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COLUMN_EPSILON = 1e-12


@dataclass(frozen=True)
class HybridBasisConfig:
    """Selection thresholds and mandatory held-out promotion gates."""

    novelty_residual_ratio: float = 0.15
    duplicate_cosine_similarity: float = 0.95
    min_heldout_global_vaf_marginal_gain: float = 1e-6
    max_total_rank: int = 64
    min_heldout_global_vaf: float = 0.90
    local_vaf_quantile: float = 0.10
    min_heldout_local_vaf_quantile: float = 0.70
    max_basis_condition_number: float = 100.0
    min_effective_rank_fraction: float = 0.80
    effective_rank_relative_tolerance: float = 1e-8

    def validated(self) -> HybridBasisConfig:
        _validate_config(self)
        return self


@dataclass(frozen=True)
class HybridBasisResult:
    """A promoted hybrid basis and its content-bound construction manifest."""

    basis: np.ndarray
    muscle_names: tuple[str, ...]
    manifest: dict[str, Any]


class HybridBasisGateError(ValueError):
    """Raised when construction is valid but any mandatory gate fails."""

    def __init__(self, message: str, *, gate_report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.gate_report = dict(gate_report)


def build_hybrid_basis(
    regional_basis: np.ndarray,
    global_basis: np.ndarray,
    *,
    regional_muscle_names: Sequence[str],
    global_muscle_names: Sequence[str],
    heldout_values: np.ndarray,
    regional_source_fingerprint: str,
    global_source_fingerprint: str,
    config: HybridBasisConfig | None = None,
) -> HybridBasisResult:
    """Build ``[W_reg, selected original columns of W_global]``.

    Selection first removes global columns already represented by the
    non-negative cone of ``W_reg``.  Remaining columns are greedily ordered by
    marginal held-out VAF; cosine duplicates are rejected against both the
    regional basis and previously retained global columns.  A column is kept
    only when its marginal gain strictly exceeds the configured minimum.  No
    rank truncation or quality-gate fallback is permitted.
    """

    cfg = config or HybridBasisConfig()
    _validate_config(cfg)
    regional, regional_names = _validate_basis(
        regional_basis,
        regional_muscle_names,
        label="regional_basis",
    )
    global_matrix, global_names = _validate_basis(
        global_basis,
        global_muscle_names,
        label="global_basis",
    )
    if regional_names != global_names:
        raise ValueError("regional and global muscle names/order must match exactly")
    heldout = _validate_heldout(heldout_values, muscle_count=regional.shape[0])
    regional_fingerprint = _validate_sha256(regional_source_fingerprint, label="regional_source_fingerprint")
    global_fingerprint = _validate_sha256(global_source_fingerprint, label="global_source_fingerprint")

    candidate_decisions = _screen_global_candidates(global_matrix, regional, cfg)
    _, regional_reconstruction = transform_nmf(heldout, regional)
    current_vaf = global_vaf(heldout, regional_reconstruction)
    selected_indices: list[int] = []
    current = regional.copy()
    pending = {int(item["global_column_index"]) for item in candidate_decisions if item["status"] == "pending"}

    while pending:
        for index in sorted(pending):
            if not selected_indices:
                continue
            selected = global_matrix[:, selected_indices]
            similarity, selected_position = _max_cosine(global_matrix[:, index], selected)
            if similarity >= cfg.duplicate_cosine_similarity:
                duplicate_index = selected_indices[selected_position]
                decision = candidate_decisions[index]
                rejection_reasons = list(decision["rejection_reasons"])
                rejection_reasons.append("cosine_duplicate_of_retained_global_column")
                decision.update(
                    {
                        "status": "rejected",
                        "decision_reason": "cosine_duplicate_of_retained_global_column",
                        "rejection_reasons": rejection_reasons,
                        "duplicate_cosine_similarity": similarity,
                        "duplicate_of_global_column_index": duplicate_index,
                    }
                )
                pending.remove(index)
        if not pending:
            break

        scored: list[tuple[float, int, float]] = []
        for index in sorted(pending):
            trial = np.column_stack((current, global_matrix[:, index]))
            _, reconstruction = transform_nmf(heldout, trial)
            trial_vaf = global_vaf(heldout, reconstruction)
            scored.append((trial_vaf - current_vaf, index, trial_vaf))
        marginal_gain, selected_index, selected_vaf = min(scored, key=lambda item: (-item[0], item[1]))
        if marginal_gain <= cfg.min_heldout_global_vaf_marginal_gain:
            for candidate_gain, index, candidate_vaf in sorted(scored, key=lambda item: item[1]):
                decision = candidate_decisions[index]
                rejection_reasons = list(decision["rejection_reasons"])
                rejection_reasons.append("insufficient_heldout_global_vaf_marginal_gain")
                decision.update(
                    {
                        "status": "rejected",
                        "decision_reason": "insufficient_heldout_global_vaf_marginal_gain",
                        "rejection_reasons": rejection_reasons,
                        "heldout_global_vaf_marginal_gain": candidate_gain,
                        "heldout_global_vaf_if_appended": candidate_vaf,
                    }
                )
            pending.clear()
            break
        selected_indices.append(selected_index)
        pending.remove(selected_index)
        current = np.column_stack((current, global_matrix[:, selected_index]))
        current_vaf = selected_vaf
        candidate_decisions[selected_index].update(
            {
                "status": "retained",
                "decision_reason": "retained_novel_original_global_column",
                "selection_order": len(selected_indices) - 1,
                "heldout_global_vaf_marginal_gain": marginal_gain,
                "heldout_global_vaf_after_selection": selected_vaf,
            }
        )

    result_basis = np.asarray(current, dtype=np.float64)
    _, reconstruction = transform_nmf(heldout, result_basis)
    gate_report = _gate_report(
        basis=result_basis,
        heldout=heldout,
        reconstruction=reconstruction,
        config=cfg,
    )
    manifest = _construction_manifest(
        basis=result_basis,
        regional_basis=regional,
        global_basis=global_matrix,
        muscle_names=regional_names,
        heldout=heldout,
        regional_source_fingerprint=regional_fingerprint,
        global_source_fingerprint=global_fingerprint,
        selected_indices=selected_indices,
        candidate_decisions=candidate_decisions,
        gate_report=gate_report,
        config=cfg,
    )
    manifest["construction_fingerprint"] = _json_sha256(manifest)
    _assert_json_safe(manifest)
    if not gate_report["all_passed"]:
        failures = [str(item["name"]) for item in gate_report["gates"] if not item["passed"]]
        raise HybridBasisGateError(
            f"hybrid synergy basis failed mandatory gates: {', '.join(failures)}",
            gate_report={"failures": failures, "construction_manifest": manifest},
        )
    return HybridBasisResult(
        basis=result_basis.copy(),
        muscle_names=regional_names,
        manifest=manifest,
    )


def save_hybrid_basis_artifact(
    path: str | Path,
    result: HybridBasisResult,
    *,
    signal_kind: str,
    source_dataset_fingerprint: str,
    teacher_checkpoint_fingerprint: str,
    normalization: Mapping[str, Any],
    fit_seed: int,
    transform: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    train_motion_uids: Sequence[int],
    primitive_source_binding: Mapping[str, Any] | None,
    source_components: Mapping[str, Mapping[str, Any]],
    artifact_role: str,
    dynamic_coverage_requirement: Mapping[str, Any],
    dynamic_coverage: Mapping[str, Any] | None,
    candidate_basis_fingerprint: str,
) -> SynergyBasisArtifact:
    """Persist a validated hybrid result through the formal basis artifact API."""

    matrix, names, construction = _validate_result(result)
    _validate_artifact_metadata(
        signal_kind=signal_kind,
        source_dataset_fingerprint=source_dataset_fingerprint,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
        normalization=normalization,
        fit_seed=fit_seed,
        transform=transform,
        split_provenance=split_provenance,
        train_motion_uids=train_motion_uids,
        primitive_source_binding=primitive_source_binding,
        source_components=source_components,
        artifact_role=artifact_role,
        dynamic_coverage_requirement=dynamic_coverage_requirement,
        dynamic_coverage=dynamic_coverage,
        candidate_basis_fingerprint=candidate_basis_fingerprint,
    )
    if source_components["regional"]["artifact_fingerprint"] != construction.get(
        "regional_source_fingerprint"
    ) or source_components["global"]["artifact_fingerprint"] != construction.get("global_source_fingerprint"):
        raise ValueError("hybrid source component fingerprints differ from construction manifest")
    manifest = {
        "signal_kind": str(signal_kind),
        "region": "hybrid_global_regional",
        "rank": int(matrix.shape[1]),
        "normalization": dict(normalization),
        "source_dataset_fingerprint": str(source_dataset_fingerprint),
        "teacher_checkpoint_fingerprint": str(teacher_checkpoint_fingerprint),
        "fit_seed": int(fit_seed),
        "transform": dict(transform),
        "split_provenance": dict(split_provenance),
        "train_motion_uids": [int(value) for value in train_motion_uids],
        "primitive_source_binding": (None if primitive_source_binding is None else dict(primitive_source_binding)),
        "artifact_role": str(artifact_role),
        "hybrid_schema_version": HYBRID_BASIS_SCHEMA_VERSION,
        "hybrid_matrix_content_sha256": construction["matrix_content_sha256"],
        "hybrid_construction": construction,
        "source_basis_fingerprints": {
            "regional": construction["regional_source_fingerprint"],
            "global": construction["global_source_fingerprint"],
        },
        "source_components": {label: dict(source_components[label]) for label in ("regional", "global")},
        "hybrid_dynamic_coverage": {
            "requirement": dict(dynamic_coverage_requirement),
            "candidate_basis_fingerprint": str(candidate_basis_fingerprint),
            "evidence": None if dynamic_coverage is None else dict(dynamic_coverage),
        },
    }
    _assert_json_safe(manifest)
    artifact = save_synergy_basis(path, basis=matrix, muscle_names=names, manifest=manifest)
    if _matrix_sha256(artifact.basis, dtype="<f4") != construction["matrix_content_sha256"]:
        raise RuntimeError("saved hybrid basis differs from the promoted matrix content hash")
    if artifact.manifest.get("hybrid_construction") != construction:
        raise RuntimeError("saved hybrid construction manifest differs from the promoted manifest")
    return artifact


def validate_hybrid_basis_result(
    result: HybridBasisResult,
    *,
    regional_basis: np.ndarray | None = None,
    global_basis: np.ndarray | None = None,
) -> HybridBasisResult:
    """Replay content, gate, and optional source-column bindings.

    Held-out samples are intentionally not embedded in production artifacts.
    Their content hash and scalar VAF evidence are construction-fingerprint
    bound; this validator recomputes every gate decision from those bound
    scalars and recomputes numerical rank/conditioning from the saved matrix.
    """

    matrix, names, construction = _validate_result(result)
    _validate_gate_evidence(matrix, construction)
    if (regional_basis is None) != (global_basis is None):
        raise ValueError("regional_basis and global_basis must be supplied together")
    if regional_basis is not None and global_basis is not None:
        _validate_source_columns(
            matrix,
            construction,
            regional_basis=np.asarray(regional_basis, dtype=np.float64),
            global_basis=np.asarray(global_basis, dtype=np.float64),
        )
    return HybridBasisResult(matrix.copy(), names, construction)


def _validate_config(config: HybridBasisConfig) -> None:
    unit_interval_fields = {
        "novelty_residual_ratio": config.novelty_residual_ratio,
        "duplicate_cosine_similarity": config.duplicate_cosine_similarity,
        "min_heldout_global_vaf_marginal_gain": config.min_heldout_global_vaf_marginal_gain,
        "min_heldout_global_vaf": config.min_heldout_global_vaf,
        "local_vaf_quantile": config.local_vaf_quantile,
        "min_heldout_local_vaf_quantile": config.min_heldout_local_vaf_quantile,
        "min_effective_rank_fraction": config.min_effective_rank_fraction,
    }
    for name, value in unit_interval_fields.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    if isinstance(config.max_total_rank, bool) or int(config.max_total_rank) != config.max_total_rank:
        raise ValueError("max_total_rank must be a positive integer")
    if int(config.max_total_rank) <= 0:
        raise ValueError("max_total_rank must be a positive integer")
    if not math.isfinite(float(config.max_basis_condition_number)) or config.max_basis_condition_number <= 0.0:
        raise ValueError("max_basis_condition_number must be finite and positive")
    if (
        not math.isfinite(float(config.effective_rank_relative_tolerance))
        or config.effective_rank_relative_tolerance <= 0.0
    ):
        raise ValueError("effective_rank_relative_tolerance must be finite and positive")


def _validate_basis(
    basis: np.ndarray,
    muscle_names: Sequence[str],
    *,
    label: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(basis, dtype=np.float64)
    names = tuple(muscle_names)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError(f"{label} must be a non-empty [muscles,rank] matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    if len(names) != matrix.shape[0]:
        raise ValueError(f"{label} muscle names must match its rows")
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{label} muscle names must be non-empty and unique")
    if np.any(np.linalg.norm(matrix, axis=0) <= _COLUMN_EPSILON):
        raise ValueError(f"{label} contains an empty component")
    return matrix.copy(), names


def _validate_heldout(values: np.ndarray, *, muscle_count: int) -> np.ndarray:
    heldout = np.asarray(values, dtype=np.float64)
    if heldout.ndim != 2 or min(heldout.shape) <= 0 or heldout.shape[1] != muscle_count:
        raise ValueError("heldout_values must be a non-empty [samples,muscles] matrix aligned to both bases")
    if not np.all(np.isfinite(heldout)) or np.any(heldout < 0.0):
        raise ValueError("heldout_values must be finite and non-negative")
    if np.sum(np.square(heldout)) <= _COLUMN_EPSILON:
        raise ValueError("heldout_values must contain non-zero signal energy")
    return heldout.copy()


def _validate_sha256(value: str, *, label: str) -> str:
    fingerprint = str(value)
    if _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"{label} must be lowercase 64-hex")
    return fingerprint


def _screen_global_candidates(
    global_basis: np.ndarray,
    regional_basis: np.ndarray,
    config: HybridBasisConfig,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for index in range(global_basis.shape[1]):
        column = global_basis[:, index]
        coefficients, residual_norm = nnls(regional_basis, column)
        residual_ratio = float(residual_norm / np.linalg.norm(column))
        regional_similarity, regional_index = _max_cosine(column, regional_basis)
        rejection_reasons: list[str] = []
        if residual_ratio <= config.novelty_residual_ratio:
            rejection_reasons.append("represented_by_regional_nonnegative_cone")
        if regional_similarity >= config.duplicate_cosine_similarity:
            rejection_reasons.append("cosine_duplicate_of_regional_column")
        decision: dict[str, Any] = {
            "global_column_index": index,
            "original_column_content_sha256": _matrix_sha256(column[:, None], dtype="<f8"),
            "regional_cone_projection_residual_ratio": residual_ratio,
            "regional_cone_projection_coefficient_count": int(np.count_nonzero(coefficients > _COLUMN_EPSILON)),
            "max_regional_cosine_similarity": regional_similarity,
            "max_regional_cosine_column_index": regional_index,
            "status": "rejected" if rejection_reasons else "pending",
            "decision_reason": rejection_reasons[0] if rejection_reasons else "pending_heldout_marginal_gain_ordering",
            "rejection_reasons": rejection_reasons,
        }
        decisions.append(decision)
    return decisions


def _max_cosine(column: np.ndarray, references: np.ndarray) -> tuple[float, int]:
    column_norm = float(np.linalg.norm(column))
    reference_norms = np.linalg.norm(references, axis=0)
    similarities = (references.T @ column) / (reference_norms * column_norm)
    best_index = int(np.argmax(similarities))
    return float(np.clip(similarities[best_index], -1.0, 1.0)), best_index


def _gate_report(
    *,
    basis: np.ndarray,
    heldout: np.ndarray,
    reconstruction: np.ndarray,
    config: HybridBasisConfig,
) -> dict[str, Any]:
    heldout_global_vaf = global_vaf(heldout, reconstruction)
    heldout_local_vaf = local_vaf(heldout, reconstruction)
    active_mask = np.sum(np.square(heldout), axis=0) > _COLUMN_EPSILON
    active_local = heldout_local_vaf[active_mask]
    local_quantile_value = float(np.quantile(active_local, config.local_vaf_quantile))
    singular_values = np.linalg.svd(basis, compute_uv=False)
    cutoff = float(singular_values[0] * config.effective_rank_relative_tolerance)
    effective_rank = int(np.count_nonzero(singular_values > cutoff))
    effective_rank_fraction = float(effective_rank / basis.shape[1])
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if effective_rank == basis.shape[1] and singular_values[-1] > 0.0
        else math.inf
    )
    gates = [
        _gate("rank_budget", basis.shape[1] <= config.max_total_rank, basis.shape[1], "<=", config.max_total_rank),
        _gate(
            "heldout_global_vaf",
            heldout_global_vaf >= config.min_heldout_global_vaf,
            heldout_global_vaf,
            ">=",
            config.min_heldout_global_vaf,
        ),
        _gate(
            "heldout_local_vaf_quantile",
            local_quantile_value >= config.min_heldout_local_vaf_quantile,
            local_quantile_value,
            ">=",
            config.min_heldout_local_vaf_quantile,
        ),
        _gate(
            "basis_condition_number",
            math.isfinite(condition_number) and condition_number <= config.max_basis_condition_number,
            condition_number if math.isfinite(condition_number) else None,
            "<=",
            config.max_basis_condition_number,
        ),
        _gate(
            "effective_rank_fraction",
            effective_rank_fraction >= config.min_effective_rank_fraction,
            effective_rank_fraction,
            ">=",
            config.min_effective_rank_fraction,
        ),
    ]
    return {
        "all_passed": all(bool(item["passed"]) for item in gates),
        "gates": gates,
        "heldout_global_vaf": heldout_global_vaf,
        "heldout_local_vaf": [float(value) if np.isfinite(value) else None for value in heldout_local_vaf],
        "heldout_active_channel_count": int(np.count_nonzero(active_mask)),
        "heldout_local_vaf_quantile_probability": float(config.local_vaf_quantile),
        "heldout_local_vaf_quantile_value": local_quantile_value,
        "basis_condition_number": condition_number if math.isfinite(condition_number) else None,
        "basis_condition_number_nonfinite": not math.isfinite(condition_number),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank_fraction,
        "effective_rank_relative_tolerance": float(config.effective_rank_relative_tolerance),
    }


def _gate(name: str, passed: bool, value: int | float | None, operator: str, threshold: int | float) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "operator": operator,
        "threshold": threshold,
    }


def _construction_manifest(
    *,
    basis: np.ndarray,
    regional_basis: np.ndarray,
    global_basis: np.ndarray,
    muscle_names: tuple[str, ...],
    heldout: np.ndarray,
    regional_source_fingerprint: str,
    global_source_fingerprint: str,
    selected_indices: Sequence[int],
    candidate_decisions: Sequence[Mapping[str, Any]],
    gate_report: Mapping[str, Any],
    config: HybridBasisConfig,
) -> dict[str, Any]:
    regional_rank = int(regional_basis.shape[1])
    selected = [int(value) for value in selected_indices]
    output_columns = [
        {
            "output_column_index": index,
            "source": "regional",
            "source_column_index": index,
            "column_content_sha256": _matrix_sha256(regional_basis[:, index : index + 1], dtype="<f8"),
        }
        for index in range(regional_rank)
    ]
    output_columns.extend(
        {
            "output_column_index": regional_rank + order,
            "source": "global_original_nonnegative_column",
            "source_column_index": source_index,
            "column_content_sha256": _matrix_sha256(global_basis[:, source_index : source_index + 1], dtype="<f8"),
        }
        for order, source_index in enumerate(selected)
    )
    return {
        "hybrid_schema_version": HYBRID_BASIS_SCHEMA_VERSION,
        "construction_mode": (
            "regional_plus_retained_original_global_columns" if selected else "regional_only_no_novel_global_columns"
        ),
        "column_policy": "append_original_nonnegative_global_columns_only_never_signed_projection_residuals",
        "selection_policy": ("greedy_heldout_global_vaf_marginal_gain_strictly_above_minimum_then_global_column_index"),
        "regional_source_fingerprint": regional_source_fingerprint,
        "global_source_fingerprint": global_source_fingerprint,
        "muscle_names": list(muscle_names),
        "muscle_schema_sha256": _json_sha256({"muscle_names": list(muscle_names)}),
        "regional_rank": regional_rank,
        "global_candidate_rank": int(global_basis.shape[1]),
        "retained_global_rank": len(selected),
        "total_rank": int(basis.shape[1]),
        "retained_global_column_indices_in_output_order": selected,
        "output_columns": output_columns,
        "candidate_decisions": [dict(item) for item in candidate_decisions],
        "thresholds": {
            "novelty_residual_ratio_strictly_greater_than": float(config.novelty_residual_ratio),
            "duplicate_cosine_similarity_reject_at_or_above": float(config.duplicate_cosine_similarity),
            "heldout_global_vaf_marginal_gain_retain_strictly_greater_than": float(
                config.min_heldout_global_vaf_marginal_gain
            ),
            "max_total_rank": int(config.max_total_rank),
            "min_heldout_global_vaf": float(config.min_heldout_global_vaf),
            "local_vaf_quantile": float(config.local_vaf_quantile),
            "min_heldout_local_vaf_quantile": float(config.min_heldout_local_vaf_quantile),
            "max_basis_condition_number": float(config.max_basis_condition_number),
            "min_effective_rank_fraction": float(config.min_effective_rank_fraction),
            "effective_rank_relative_tolerance": float(config.effective_rank_relative_tolerance),
        },
        "heldout_evaluation": dict(gate_report),
        "heldout_shape": list(heldout.shape),
        "heldout_content_sha256": _matrix_sha256(heldout, dtype="<f8"),
        "matrix_shape": list(basis.shape),
        "matrix_storage_dtype": "float32-little-endian",
        "matrix_content_sha256": _matrix_sha256(basis, dtype="<f4"),
    }


def _validate_result(result: HybridBasisResult) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    if not isinstance(result, HybridBasisResult):
        raise TypeError("result must be a HybridBasisResult")
    matrix, names = _validate_basis(result.basis, result.muscle_names, label="result.basis")
    if not isinstance(result.manifest, Mapping):
        raise ValueError("result manifest must be an object")
    construction = dict(result.manifest)
    supplied_fingerprint = construction.pop("construction_fingerprint", None)
    if supplied_fingerprint != _json_sha256(construction):
        raise ValueError("hybrid construction_fingerprint mismatch")
    construction["construction_fingerprint"] = supplied_fingerprint
    if construction.get("hybrid_schema_version") != HYBRID_BASIS_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid construction schema")
    if construction.get("muscle_names") != list(names):
        raise ValueError("hybrid construction muscle names/order differ from result")
    if construction.get("muscle_schema_sha256") != _json_sha256({"muscle_names": list(names)}):
        raise ValueError("hybrid construction muscle schema hash mismatch")
    if construction.get("matrix_shape") != list(matrix.shape):
        raise ValueError("hybrid construction matrix shape differs from result")
    if construction.get("matrix_content_sha256") != _matrix_sha256(matrix, dtype="<f4"):
        raise ValueError("hybrid construction matrix content hash differs from result")
    _validate_sha256(construction.get("regional_source_fingerprint", ""), label="regional_source_fingerprint")
    _validate_sha256(construction.get("global_source_fingerprint", ""), label="global_source_fingerprint")
    if construction.get("matrix_storage_dtype") != "float32-little-endian":
        raise ValueError("hybrid construction matrix storage dtype is unsupported")
    if construction.get("total_rank") != matrix.shape[1]:
        raise ValueError("hybrid construction total rank differs from result")
    regional_rank = construction.get("regional_rank")
    retained_rank = construction.get("retained_global_rank")
    if (
        not isinstance(regional_rank, int)
        or not isinstance(retained_rank, int)
        or regional_rank <= 0
        or retained_rank < 0
        or regional_rank + retained_rank != matrix.shape[1]
    ):
        raise ValueError("hybrid construction regional/retained ranks are inconsistent")
    output_columns = construction.get("output_columns")
    candidate_decisions = construction.get("candidate_decisions")
    if not isinstance(output_columns, list) or len(output_columns) != matrix.shape[1]:
        raise ValueError("hybrid construction output-column provenance is incomplete")
    if not isinstance(candidate_decisions, list) or len(candidate_decisions) != construction.get(
        "global_candidate_rank"
    ):
        raise ValueError("hybrid construction candidate decisions are incomplete")
    heldout_evaluation = construction.get("heldout_evaluation")
    if not isinstance(heldout_evaluation, Mapping) or heldout_evaluation.get("all_passed") is not True:
        raise ValueError("hybrid construction did not pass all mandatory gates")
    gates = heldout_evaluation.get("gates")
    expected_gate_names = {
        "rank_budget",
        "heldout_global_vaf",
        "heldout_local_vaf_quantile",
        "basis_condition_number",
        "effective_rank_fraction",
    }
    if (
        not isinstance(gates, list)
        or {item.get("name") for item in gates if isinstance(item, Mapping)} != expected_gate_names
        or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in gates)
    ):
        raise ValueError("hybrid construction mandatory gate evidence is incomplete")
    _assert_json_safe(construction)
    return matrix, names, construction


def _validate_gate_evidence(matrix: np.ndarray, construction: Mapping[str, Any]) -> None:
    thresholds = construction.get("thresholds")
    expected_threshold_fields = {
        "novelty_residual_ratio_strictly_greater_than",
        "duplicate_cosine_similarity_reject_at_or_above",
        "heldout_global_vaf_marginal_gain_retain_strictly_greater_than",
        "max_total_rank",
        "min_heldout_global_vaf",
        "local_vaf_quantile",
        "min_heldout_local_vaf_quantile",
        "max_basis_condition_number",
        "min_effective_rank_fraction",
        "effective_rank_relative_tolerance",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_threshold_fields:
        raise ValueError("hybrid construction threshold fields differ from contract")
    max_total_rank = _strict_positive_int(thresholds["max_total_rank"], "hybrid max_total_rank")
    bounded = {
        name: _bounded_float(thresholds[name], name)
        for name in (
            "novelty_residual_ratio_strictly_greater_than",
            "duplicate_cosine_similarity_reject_at_or_above",
            "heldout_global_vaf_marginal_gain_retain_strictly_greater_than",
            "min_heldout_global_vaf",
            "local_vaf_quantile",
            "min_heldout_local_vaf_quantile",
            "min_effective_rank_fraction",
        )
    }
    max_condition = _positive_float(
        thresholds["max_basis_condition_number"],
        "max_basis_condition_number",
    )
    rank_tolerance = _positive_float(
        thresholds["effective_rank_relative_tolerance"],
        "effective_rank_relative_tolerance",
    )
    evaluation = construction.get("heldout_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("hybrid construction heldout evaluation is absent")
    heldout_global_vaf = _finite_float(evaluation.get("heldout_global_vaf"), "heldout_global_vaf")
    local_quantile = _finite_float(
        evaluation.get("heldout_local_vaf_quantile_value"),
        "heldout_local_vaf_quantile_value",
    )
    if not np.isclose(
        _finite_float(
            evaluation.get("heldout_local_vaf_quantile_probability"),
            "heldout_local_vaf_quantile_probability",
        ),
        bounded["local_vaf_quantile"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("hybrid local VAF quantile probability differs from threshold")
    if not np.isclose(
        _finite_float(
            evaluation.get("effective_rank_relative_tolerance"),
            "effective_rank_relative_tolerance",
        ),
        rank_tolerance,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("hybrid effective-rank tolerance differs from threshold")
    local_values = evaluation.get("heldout_local_vaf")
    if not isinstance(local_values, list) or len(local_values) != matrix.shape[0]:
        raise ValueError("hybrid heldout local VAF evidence is not muscle-aligned")
    if any(value is not None and not _is_finite_number(value) for value in local_values):
        raise ValueError("hybrid heldout local VAF evidence contains invalid values")
    active_count = _strict_positive_int(
        evaluation.get("heldout_active_channel_count"),
        "heldout_active_channel_count",
    )
    if active_count > matrix.shape[0] or active_count != sum(value is not None for value in local_values):
        raise ValueError("hybrid heldout active-channel evidence is inconsistent")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    cutoff = float(singular_values[0] * rank_tolerance)
    effective_rank = int(np.count_nonzero(singular_values > cutoff))
    effective_rank_fraction = float(effective_rank / matrix.shape[1])
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if effective_rank == matrix.shape[1] and singular_values[-1] > 0.0
        else math.inf
    )
    if evaluation.get("effective_rank") != effective_rank or not np.isclose(
        _finite_float(evaluation.get("effective_rank_fraction"), "effective_rank_fraction"),
        effective_rank_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("hybrid effective-rank evidence differs from saved matrix")
    declared_condition = evaluation.get("basis_condition_number")
    if math.isfinite(condition_number):
        if (
            not np.isclose(
                _finite_float(declared_condition, "basis_condition_number"),
                condition_number,
                rtol=1e-6,
                atol=1e-9,
            )
            or evaluation.get("basis_condition_number_nonfinite") is not False
        ):
            raise ValueError("hybrid condition-number evidence differs from saved matrix")
    elif declared_condition is not None or evaluation.get("basis_condition_number_nonfinite") is not True:
        raise ValueError("hybrid nonfinite condition-number evidence differs from saved matrix")

    expected_gates = {
        "rank_budget": (matrix.shape[1] <= max_total_rank, matrix.shape[1], "<=", max_total_rank),
        "heldout_global_vaf": (
            heldout_global_vaf >= bounded["min_heldout_global_vaf"],
            heldout_global_vaf,
            ">=",
            bounded["min_heldout_global_vaf"],
        ),
        "heldout_local_vaf_quantile": (
            local_quantile >= bounded["min_heldout_local_vaf_quantile"],
            local_quantile,
            ">=",
            bounded["min_heldout_local_vaf_quantile"],
        ),
        "basis_condition_number": (
            math.isfinite(condition_number) and condition_number <= max_condition,
            condition_number if math.isfinite(condition_number) else None,
            "<=",
            max_condition,
        ),
        "effective_rank_fraction": (
            effective_rank_fraction >= bounded["min_effective_rank_fraction"],
            effective_rank_fraction,
            ">=",
            bounded["min_effective_rank_fraction"],
        ),
    }
    gates = evaluation.get("gates")
    if not isinstance(gates, list) or len(gates) != len(expected_gates):
        raise ValueError("hybrid mandatory gate evidence is incomplete")
    seen: set[str] = set()
    for gate in gates:
        if not isinstance(gate, Mapping) or set(gate) != {"name", "passed", "value", "operator", "threshold"}:
            raise ValueError("hybrid mandatory gate descriptor differs from contract")
        name = str(gate["name"])
        if name not in expected_gates or name in seen:
            raise ValueError("hybrid mandatory gate names are invalid or duplicated")
        seen.add(name)
        passed, value, operator, threshold = expected_gates[name]
        if gate["passed"] is not bool(passed) or gate["operator"] != operator:
            raise ValueError(f"hybrid {name} decision differs from recomputed gate")
        _require_equal_number(gate["value"], value, label=f"hybrid {name} value")
        _require_equal_number(gate["threshold"], threshold, label=f"hybrid {name} threshold")
    if evaluation.get("all_passed") is not all(value[0] for value in expected_gates.values()):
        raise ValueError("hybrid all_passed differs from recomputed gates")


def _validate_source_columns(
    matrix: np.ndarray,
    construction: Mapping[str, Any],
    *,
    regional_basis: np.ndarray,
    global_basis: np.ndarray,
) -> None:
    if regional_basis.ndim != 2 or global_basis.ndim != 2:
        raise ValueError("hybrid source bases must be matrices")
    if regional_basis.shape[0] != matrix.shape[0] or global_basis.shape[0] != matrix.shape[0]:
        raise ValueError("hybrid source basis rows differ from the saved matrix")
    regional_rank = construction.get("regional_rank")
    global_rank = construction.get("global_candidate_rank")
    if regional_rank != regional_basis.shape[1] or global_rank != global_basis.shape[1]:
        raise ValueError("hybrid source basis ranks differ from construction manifest")
    selected = construction.get("retained_global_column_indices_in_output_order")
    if (
        not isinstance(selected, list)
        or any(type(index) is not int or index < 0 or index >= global_basis.shape[1] for index in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("hybrid retained global source-column indices are invalid")
    expected = np.column_stack((regional_basis, global_basis[:, selected])) if selected else regional_basis
    if not np.array_equal(matrix.astype("<f4"), np.asarray(expected, dtype="<f4")):
        raise ValueError("hybrid matrix is not the regional prefix plus specified original global columns")
    decisions = construction.get("candidate_decisions")
    if not isinstance(decisions, list) or [item.get("global_column_index") for item in decisions] != list(
        range(global_basis.shape[1])
    ):
        raise ValueError("hybrid candidate decisions are not in original global-column order")
    thresholds = construction["thresholds"]
    novelty_threshold = float(thresholds["novelty_residual_ratio_strictly_greater_than"])
    duplicate_threshold = float(thresholds["duplicate_cosine_similarity_reject_at_or_above"])
    marginal_threshold = float(thresholds["heldout_global_vaf_marginal_gain_retain_strictly_greater_than"])
    selected_order = {int(source_index): order for order, source_index in enumerate(selected)}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping) or decision.get("status") not in {"retained", "rejected"}:
            raise ValueError("hybrid candidate decision status is invalid")
        column = global_basis[:, index]
        _, residual_norm = nnls(regional_basis, column)
        residual_ratio = float(residual_norm / np.linalg.norm(column))
        regional_cosine, regional_index = _max_cosine(column, regional_basis)
        if (
            not np.isclose(
                _finite_float(
                    decision.get("regional_cone_projection_residual_ratio"),
                    "regional cone residual ratio",
                ),
                residual_ratio,
                rtol=1e-7,
                atol=1e-10,
            )
            or not np.isclose(
                _finite_float(
                    decision.get("max_regional_cosine_similarity"),
                    "max regional cosine similarity",
                ),
                regional_cosine,
                rtol=1e-7,
                atol=1e-10,
            )
            or decision.get("max_regional_cosine_column_index") != regional_index
        ):
            raise ValueError("hybrid candidate regional novelty evidence differs from source matrices")
        if decision.get("original_column_content_sha256") != _matrix_sha256(column[:, None], dtype="<f8"):
            raise ValueError("hybrid original global-column content hash mismatch")
        reason = decision.get("decision_reason")
        if decision["status"] == "retained":
            gain = _finite_float(
                decision.get("heldout_global_vaf_marginal_gain"),
                "retained hybrid marginal VAF gain",
            )
            if reason != "retained_novel_original_global_column" or gain <= marginal_threshold:
                raise ValueError("hybrid retained a global column without sufficient heldout gain")
        elif reason == "represented_by_regional_nonnegative_cone" and residual_ratio > novelty_threshold:
            raise ValueError("hybrid regional-cone rejection differs from recomputed novelty")
        elif reason == "cosine_duplicate_of_regional_column" and regional_cosine < duplicate_threshold:
            raise ValueError("hybrid regional-cosine rejection differs from recomputed similarity")
        elif reason == "insufficient_heldout_global_vaf_marginal_gain":
            gain = _finite_float(
                decision.get("heldout_global_vaf_marginal_gain"),
                "rejected hybrid marginal VAF gain",
            )
            if gain > marginal_threshold:
                raise ValueError("hybrid heldout-gain rejection differs from configured threshold")
        elif reason == "cosine_duplicate_of_retained_global_column":
            duplicate_index = decision.get("duplicate_of_global_column_index")
            if duplicate_index not in selected_order:
                raise ValueError("hybrid duplicate rejection does not name a retained global column")
            cosine, _ = _max_cosine(column, global_basis[:, [duplicate_index]])
            if cosine < duplicate_threshold:
                raise ValueError("hybrid retained-global duplicate rejection differs from source matrices")
        elif reason not in {
            "represented_by_regional_nonnegative_cone",
            "cosine_duplicate_of_regional_column",
        }:
            raise ValueError("hybrid rejected candidate has an unsupported reason")
    retained = sorted(
        (
            (item.get("selection_order"), item.get("global_column_index"))
            for item in decisions
            if isinstance(item, Mapping) and item.get("status") == "retained"
        ),
        key=lambda item: item[0] if type(item[0]) is int else -1,
    )
    if retained != list(enumerate(selected)):
        raise ValueError("hybrid retained decisions differ from specified global output columns")


def _validate_artifact_metadata(
    *,
    signal_kind: str,
    source_dataset_fingerprint: str,
    teacher_checkpoint_fingerprint: str,
    normalization: Mapping[str, Any],
    fit_seed: int,
    transform: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    train_motion_uids: Sequence[int],
    primitive_source_binding: Mapping[str, Any] | None,
    source_components: Mapping[str, Mapping[str, Any]],
    artifact_role: str,
    dynamic_coverage_requirement: Mapping[str, Any],
    dynamic_coverage: Mapping[str, Any] | None,
    candidate_basis_fingerprint: str,
) -> None:
    if not str(signal_kind).strip() or not str(source_dataset_fingerprint).strip():
        raise ValueError("signal_kind and source_dataset_fingerprint must be non-empty")
    _validate_sha256(teacher_checkpoint_fingerprint, label="teacher_checkpoint_fingerprint")
    for label, value in (
        ("normalization", normalization),
        ("transform", transform),
        ("split_provenance", split_provenance),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
    if isinstance(fit_seed, bool) or int(fit_seed) != fit_seed:
        raise ValueError("fit_seed must be an integer")
    if not isinstance(train_motion_uids, Sequence) or isinstance(train_motion_uids, str | bytes):
        raise ValueError("train_motion_uids must be a sequence of integers")
    if any(isinstance(value, bool) or not isinstance(value, int | np.integer) for value in train_motion_uids):
        raise ValueError("train_motion_uids must be a sequence of integers")
    if primitive_source_binding is not None and not isinstance(primitive_source_binding, Mapping):
        raise ValueError("primitive_source_binding must be an object or null")
    if not isinstance(source_components, Mapping) or set(source_components) != {"regional", "global"}:
        raise ValueError("source_components must contain exactly regional and global descriptors")
    expected_regions = {"regional": "regional_composite", "global": "whole_body"}
    for label, expected_region in expected_regions.items():
        descriptor = source_components[label]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "region",
            "artifact_path",
            "artifact_fingerprint",
        }:
            raise ValueError(f"hybrid {label} source descriptor fields differ from contract")
        if descriptor.get("region") != expected_region:
            raise ValueError(f"hybrid {label} source descriptor region mismatch")
        source_path = Path(str(descriptor.get("artifact_path", "")))
        if not source_path.is_absolute():
            raise ValueError(f"hybrid {label} source artifact_path must be absolute")
        _validate_sha256(
            descriptor.get("artifact_fingerprint", ""),
            label=f"hybrid {label} source artifact_fingerprint",
        )
    if artifact_role not in {
        "primary_hybrid_global_regional",
        "dynamic_coverage_rollout_candidate",
    }:
        raise ValueError("unsupported hybrid artifact_role")
    if not isinstance(dynamic_coverage_requirement, Mapping):
        raise ValueError("dynamic_coverage_requirement must be an object")
    if dynamic_coverage is not None and not isinstance(dynamic_coverage, Mapping):
        raise ValueError("dynamic_coverage must be an object or null")
    _validate_sha256(candidate_basis_fingerprint, label="candidate_basis_fingerprint")
    if (
        artifact_role == "primary_hybrid_global_regional"
        and dynamic_coverage_requirement.get("required") is True
        and dynamic_coverage is None
    ):
        raise ValueError("required hybrid dynamic coverage evidence is absent from primary artifact")
    _assert_json_safe(
        {
            "normalization": dict(normalization),
            "transform": dict(transform),
            "split_provenance": dict(split_provenance),
            "train_motion_uids": [int(value) for value in train_motion_uids],
            "primitive_source_binding": (None if primitive_source_binding is None else dict(primitive_source_binding)),
            "source_components": {label: dict(source_components[label]) for label in ("regional", "global")},
            "dynamic_coverage_requirement": dict(dynamic_coverage_requirement),
            "dynamic_coverage": None if dynamic_coverage is None else dict(dynamic_coverage),
        }
    )


def _matrix_sha256(matrix: np.ndarray, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(matrix, dtype=np.dtype(dtype)))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_json_safe(payload: Any) -> None:
    try:
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("hybrid manifest must be strict-JSON serializable") from exc


def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _is_finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _finite_float(value: Any, label: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _bounded_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_equal_number(actual: Any, expected: Any, *, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} differs from recomputed evidence")
        return
    if not np.isclose(
        _finite_float(actual, label),
        _finite_float(expected, label),
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(f"{label} differs from recomputed evidence")
