"""Fit a bounded, phase-structured residual basis from primitive excitation.

The primary primitive basis remains frozen.  This builder first projects the
primitive train/validation excitation through the exact bounded coefficient
contract used by Stage-1, then fits small signed SVD directions to the
unexplained train residual.  Every direction is restricted by an explicit
task/phase and allowed-muscle mask; held-out validation must pass before an
artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.synergy.action_interface import (
    load_coefficient_statistics,
    residual_matrix_fingerprint,
    save_structured_residual_basis,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.fit import LoadedSynergySplit, load_synergy_split
from musclemimic.synergy.primitive_manifest import load_primitive_source_manifest
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

RESIDUAL_MASK_SCHEMA_VERSION = "early_synergy_residual_mask_v1"
RESIDUAL_FIT_SCHEMA_VERSION = "early_synergy_residual_fit_v1"
RESIDUAL_FIT_REPORT_SCHEMA_VERSION = "early_synergy_residual_fit_report_v1"
_DERIVATION = "phase_grouped_unexplained_excitation_svd_v1"
_BALANCING = "equal_task_phase_then_trial_mean_quality_then_frame_quality"
_SOLVER = "scipy_lsq_linear_bounded_exact"


@dataclass(frozen=True)
class StructuredResidualFitConfig:
    """Frozen choices that define the Phase-A residual artifact."""

    alpha: float = 0.03
    min_dimension: int = 4
    max_dimension: int = 12
    max_row_l1_norm: float = 2.0
    min_validation_residual_energy_reduction: float = 0.01
    min_group_validation_residual_energy_reduction: float = 0.01
    max_validation_coordinate_saturation_fraction: float = 0.75
    solver_tolerance: float = 1e-10
    solver_max_iterations: int = 500
    energy_epsilon: float = 1e-12

    def validated(self) -> StructuredResidualFitConfig:
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("residual alpha must lie in (0,1]")
        if not 1 <= self.min_dimension <= self.max_dimension <= 32:
            raise ValueError("residual dimension bounds must satisfy 1 <= min <= max <= 32")
        if not np.isfinite(self.max_row_l1_norm) or self.max_row_l1_norm <= 0.0:
            raise ValueError("max_row_l1_norm must be finite and positive")
        for field in (
            "min_validation_residual_energy_reduction",
            "min_group_validation_residual_energy_reduction",
            "max_validation_coordinate_saturation_fraction",
        ):
            value = float(getattr(self, field))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [0,1]")
        if (
            not np.isfinite(self.solver_tolerance)
            or self.solver_tolerance <= 0.0
            or self.solver_max_iterations <= 0
            or not np.isfinite(self.energy_epsilon)
            or self.energy_epsilon <= 0.0
        ):
            raise ValueError("solver tolerance/iterations and energy epsilon must be positive")
        return self


def load_residual_mask_contract(
    path: str | Path,
    *,
    expected_actuator_names: Sequence[str],
    primitive_required_phase_ids: Mapping[str, Sequence[int]],
    min_dimension: int = 4,
    max_dimension: int = 12,
) -> dict[str, Any]:
    """Load one exact mask that covers every declared primitive task/phase cell."""

    source = Path(path)
    payload = load_json_strict(source)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "actuator_names",
        "groups",
    }:
        raise ValueError("residual mask JSON requires exactly schema_version, actuator_names, and groups")
    if payload.get("schema_version") != RESIDUAL_MASK_SCHEMA_VERSION:
        raise ValueError("unsupported structured residual mask schema")
    names = tuple(str(name) for name in expected_actuator_names)
    supplied_names = payload.get("actuator_names")
    if not isinstance(supplied_names, list) or supplied_names != list(names):
        raise ValueError("residual mask actuator names/order differ from primary basis")
    if not names or len(set(names)) != len(names):
        raise ValueError("primary actuator schema must be non-empty and unique")

    expected_cells = {
        (str(task), int(phase)) for task, phases in primitive_required_phase_ids.items() for phase in phases
    }
    if not expected_cells:
        raise ValueError("primitive required-phase inventory cannot be empty")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("residual mask requires at least one group")
    groups: list[dict[str, Any]] = []
    seen_group_names: set[str] = set()
    seen_cells: set[tuple[str, int]] = set()
    total_rank = 0
    union_mask = np.zeros(len(names), dtype=bool)
    name_to_index = {name: index for index, name in enumerate(names)}
    for raw_group in raw_groups:
        required_fields = {
            "name",
            "task_phase_selectors",
            "allowed_muscle_names",
            "rank",
        }
        if not isinstance(raw_group, Mapping) or set(raw_group) != required_fields:
            raise ValueError("every residual mask group requires exactly name/selectors/muscles/rank")
        group_name = raw_group.get("name")
        if not isinstance(group_name, str) or not group_name.strip() or group_name in seen_group_names:
            raise ValueError("residual mask group names must be unique and non-empty")
        seen_group_names.add(group_name)
        selectors = _canonical_task_phase_selectors(raw_group.get("task_phase_selectors"))
        for task, phases in selectors.items():
            for phase in phases:
                cell = (task, phase)
                if cell not in expected_cells:
                    raise ValueError(f"residual mask selector {cell!r} is absent from primitive required phases")
                if cell in seen_cells:
                    raise ValueError("residual mask task-phase selectors must not overlap")
                seen_cells.add(cell)
        allowed_names = raw_group.get("allowed_muscle_names")
        if not isinstance(allowed_names, list) or not allowed_names:
            raise ValueError("residual mask group requires allowed_muscle_names")
        if any(not isinstance(name, str) or name not in name_to_index for name in allowed_names):
            raise ValueError("residual mask group contains an unknown muscle")
        rows = [name_to_index[name] for name in allowed_names]
        if len(rows) != len(set(rows)) or rows != sorted(rows):
            raise ValueError("residual allowed_muscle_names must be unique and follow actuator order")
        rank = raw_group.get("rank")
        if type(rank) is not int or rank <= 0 or rank > len(rows):
            raise ValueError("residual group rank must be positive and no larger than its mask")
        union_mask[np.asarray(rows, dtype=np.int64)] = True
        total_rank += rank
        groups.append(
            {
                "name": group_name,
                "task_phase_selectors": selectors,
                "allowed_muscle_names": list(allowed_names),
                "allowed_row_indices": rows,
                "rank": rank,
            }
        )
    if seen_cells != expected_cells:
        missing = sorted(expected_cells - seen_cells)
        raise ValueError(f"residual mask does not cover every primitive required phase: {missing}")
    if not min_dimension <= total_rank <= max_dimension:
        raise ValueError(f"residual mask total rank {total_rank} lies outside [{min_dimension},{max_dimension}]")
    unsigned = {
        "schema_version": RESIDUAL_MASK_SCHEMA_VERSION,
        "actuator_schema_hash": actuator_schema_hash(names),
        "groups": groups,
        "union_allowed_muscle_names": [names[index] for index in np.flatnonzero(union_mask).tolist()],
        "union_allowed_muscle_mask": union_mask.tolist(),
    }
    return {**unsigned, "fingerprint": _json_sha256(unsigned)}


def fit_structured_residual_basis(
    train_source: str | Path,
    validation_source: str | Path,
    *,
    primary_basis_path: str | Path,
    coefficient_statistics_path: str | Path,
    primitive_source_manifest_path: str | Path,
    expected_primitive_source_manifest_fingerprint: str,
    residual_mask_path: str | Path,
    output_path: str | Path,
    config: StructuredResidualFitConfig | None = None,
) -> dict[str, Any]:
    """Fit and seal a structured residual artifact, or fail without writing one."""

    cfg = (config or StructuredResidualFitConfig()).validated()
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            "structured residual output must be a fresh path so a failed refit cannot "
            f"leave a stale runtime artifact: {destination}"
        )
    primary = load_synergy_basis(primary_basis_path)
    if primary.manifest.get("signal_kind") != EXCITATION_SIGNAL_KIND:
        raise ValueError("structured residual primary basis must represent unit excitation")
    source = load_primitive_source_manifest(
        primitive_source_manifest_path,
        expected_fingerprint=expected_primitive_source_manifest_fingerprint,
    )
    binding = primary.manifest.get("primitive_source_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("structured residual primary basis lacks primitive source binding")
    if (
        binding.get("manifest_fingerprint") != source.fingerprint
        or binding.get("source_dataset_fingerprint") != source.manifest["source_dataset_fingerprint"]
    ):
        raise ValueError("primary basis and primitive source manifest provenance differ")

    train = load_synergy_split(train_source, split="train")
    validation = load_synergy_split(validation_source, split="val")
    _validate_source_splits(train, validation, primary=primary, source_manifest=source.manifest)
    train_signal = train.signal(EXCITATION_SIGNAL_KIND)
    validation_signal = validation.signal(EXCITATION_SIGNAL_KIND)
    stats = load_coefficient_statistics(
        coefficient_statistics_path,
        expected_basis_fingerprint=primary.fingerprint,
        expected_rank=primary.basis.shape[1],
    )
    upper = 1.2 * np.asarray(stats["coefficient_q99"], dtype=np.float64)
    if np.any(upper <= 0.0) or not np.all(np.isfinite(upper)):
        raise ValueError("runtime coefficient upper bounds derived from q99 are invalid")
    mask_contract = load_residual_mask_contract(
        residual_mask_path,
        expected_actuator_names=primary.muscle_names,
        primitive_required_phase_ids=source.manifest["primitive_required_phase_ids"],
        min_dimension=cfg.min_dimension,
        max_dimension=cfg.max_dimension,
    )

    train_primary_preclip = _bounded_primary_preclip(
        train_signal.values,
        primary.basis,
        upper,
        config=cfg,
    )
    validation_primary_preclip = _bounded_primary_preclip(
        validation_signal.values,
        primary.basis,
        upper,
        config=cfg,
    )
    train_unexplained = np.asarray(train_signal.values, dtype=np.float64) - train_primary_preclip

    matrix = np.zeros(
        (len(primary.muscle_names), sum(group["rank"] for group in mask_contract["groups"])),
        dtype=np.float64,
    )
    fitted_groups: list[dict[str, Any]] = []
    start = 0
    for group in mask_contract["groups"]:
        rank = int(group["rank"])
        stop = start + rank
        rows = np.asarray(group["allowed_row_indices"], dtype=np.int64)
        sample_mask = _selector_mask(train.arrays, group["task_phase_selectors"])
        sample_count = int(np.count_nonzero(sample_mask))
        if sample_count < rank:
            raise ValueError(f"residual group {group['name']!r} has {sample_count} train samples for rank {rank}")
        weights = _balanced_selector_weights(train.arrays, sample_mask)
        selected = train_unexplained[sample_mask][:, rows]
        weighted = selected * np.sqrt(weights)[:, None]
        _left, singular_values, right = np.linalg.svd(weighted, full_matrices=False)
        if singular_values.size < rank or singular_values[rank - 1] <= cfg.energy_epsilon:
            raise ValueError(f"residual group {group['name']!r} has fewer than {rank} nonzero directions")
        directions = _canonicalize_column_signs(right[:rank].T)
        matrix[np.ix_(rows, np.arange(start, stop, dtype=np.int64))] = directions
        total_energy = float(np.sum(np.square(singular_values)))
        directional_fraction = float(np.sum(np.square(singular_values[:rank])) / max(total_energy, cfg.energy_epsilon))
        fitted_groups.append(
            {
                **group,
                "column_start": start,
                "column_stop": stop,
                "train_sample_count": sample_count,
                "weighted_singular_values": singular_values.tolist(),
                "weighted_directional_energy_fraction": directional_fraction,
            }
        )
        start = stop
    row_l1 = float(np.max(np.sum(np.abs(matrix), axis=1)))
    if row_l1 > cfg.max_row_l1_norm + 1e-12:
        raise ValueError(f"fitted residual row-L1 norm {row_l1:.6g} exceeds {cfg.max_row_l1_norm:.6g}")

    train_metrics = _residual_metrics(
        np.asarray(train_signal.values, dtype=np.float64),
        train_primary_preclip,
        matrix,
        alpha=cfg.alpha,
        config=cfg,
    )
    validation_metrics = _residual_metrics(
        np.asarray(validation_signal.values, dtype=np.float64),
        validation_primary_preclip,
        matrix,
        alpha=cfg.alpha,
        config=cfg,
    )
    per_validation_group: dict[str, dict[str, Any]] = {}
    for group in fitted_groups:
        sample_mask = _selector_mask(validation.arrays, group["task_phase_selectors"])
        if not np.any(sample_mask):
            raise ValueError(f"residual group {group['name']!r} has no held-out validation samples")
        start = int(group["column_start"])
        stop = int(group["column_stop"])
        per_validation_group[str(group["name"])] = _residual_metrics(
            np.asarray(validation_signal.values, dtype=np.float64)[sample_mask],
            validation_primary_preclip[sample_mask],
            matrix[:, start:stop],
            alpha=cfg.alpha,
            config=cfg,
        )
    thresholds = {
        "min_validation_residual_energy_reduction": cfg.min_validation_residual_energy_reduction,
        "min_group_validation_residual_energy_reduction": (cfg.min_group_validation_residual_energy_reduction),
        "max_validation_coordinate_saturation_fraction": (cfg.max_validation_coordinate_saturation_fraction),
    }
    passed = bool(
        validation_metrics["residual_energy_reduction"] >= cfg.min_validation_residual_energy_reduction
        and validation_metrics["coordinate_saturation_fraction"] <= cfg.max_validation_coordinate_saturation_fraction
        and all(
            metrics["residual_energy_reduction"] >= cfg.min_group_validation_residual_energy_reduction
            and metrics["coordinate_saturation_fraction"] <= cfg.max_validation_coordinate_saturation_fraction
            for metrics in per_validation_group.values()
        )
    )
    if not passed:
        raise ValueError(
            "structured residual held-out gates failed; no runtime artifact was written: "
            f"validation={validation_metrics}, per_group={per_validation_group}"
        )

    contract_without_fingerprint: dict[str, Any] = {
        "schema_version": RESIDUAL_FIT_SCHEMA_VERSION,
        "derivation": _DERIVATION,
        "fit_scope": "train_only_validation_held_out",
        "passed": passed,
        "source_basis_fingerprint": primary.fingerprint,
        "source_dataset_fingerprint": source.manifest["source_dataset_fingerprint"],
        "primitive_source_manifest_fingerprint": source.fingerprint,
        "coefficient_statistics_fingerprint": stats["stats_fingerprint"],
        "coefficient_upper_bounds": upper.tolist(),
        "reference_alpha": cfg.alpha,
        "train_source_fingerprint": train.content_fingerprint,
        "validation_source_fingerprint": validation.content_fingerprint,
        "residual_matrix_fingerprint": residual_matrix_fingerprint(
            matrix,
            primary.muscle_names,
        ),
        "mask_contract": mask_contract,
        "groups": fitted_groups,
        "sample_balancing": _BALANCING,
        "projection_solver": _SOLVER,
        "projection_solver_parameters": {
            "solver_tolerance": cfg.solver_tolerance,
            "solver_max_iterations": cfg.solver_max_iterations,
            "energy_epsilon": cfg.energy_epsilon,
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "per_validation_group": per_validation_group,
        },
        "thresholds": thresholds,
    }
    fit_contract = {
        **contract_without_fingerprint,
        "fit_contract_fingerprint": _json_sha256(contract_without_fingerprint),
    }
    artifact = save_structured_residual_basis(
        output_path,
        basis=matrix,
        actuator_names=primary.muscle_names,
        source_basis_fingerprint=primary.fingerprint,
        source_description=(
            "train-only phase-grouped SVD of primitive excitation unexplained by "
            "the exact bounded primary Stage-1 decoder"
        ),
        allowed_muscle_mask=np.asarray(
            mask_contract["union_allowed_muscle_mask"],
            dtype=bool,
        ),
        fit_contract=fit_contract,
    )
    report = {
        "schema_version": RESIDUAL_FIT_REPORT_SCHEMA_VERSION,
        "artifact_path": artifact.source_path,
        "artifact_fingerprint": artifact.fingerprint,
        "source_basis_path": str(primary.path.resolve()),
        "source_basis_fingerprint": primary.fingerprint,
        "primitive_source_manifest_path": str(source.path.resolve()),
        "primitive_source_manifest_fingerprint": source.fingerprint,
        "coefficient_statistics_path": str(Path(coefficient_statistics_path).resolve()),
        "residual_mask_path": str(Path(residual_mask_path).resolve()),
        "fit_config": asdict(cfg),
        "fit_contract": fit_contract,
    }
    report_path = Path(artifact.source_path) / "fit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "fit_report_path": str(report_path.resolve())}


def _validate_source_splits(
    train: LoadedSynergySplit,
    validation: LoadedSynergySplit,
    *,
    primary: Any,
    source_manifest: Mapping[str, Any],
) -> None:
    if train.muscle_names != validation.muscle_names or train.muscle_names != primary.muscle_names:
        raise ValueError("residual train/validation/primary actuator schemas differ")
    if set(train.source_files) & set(validation.source_files):
        raise ValueError("residual train/validation shard files overlap")
    split_provenance = {
        "train": train.provenance(),
        "validation": validation.provenance(),
    }
    dataset_fingerprint = _json_sha256(split_provenance)
    if (
        dataset_fingerprint != source_manifest["source_dataset_fingerprint"]
        or dataset_fingerprint != primary.manifest.get("source_dataset_fingerprint")
        or primary.manifest.get("split_provenance") != split_provenance
    ):
        raise ValueError("residual source shard content differs from primary primitive basis")
    for split_name, split in (("train", train), ("validation", validation)):
        required = {"task_id", "trial_id", "phase_id", "quality_weight"}
        missing = sorted(required - set(split.arrays))
        if missing:
            raise ValueError(f"residual {split_name} source lacks fields: {missing}")
        count = split.phase_id.shape[0]
        for field in required:
            if np.asarray(split.arrays[field]).shape != (count,):
                raise ValueError(f"residual {split_name} {field} must have shape [{count}]")
        for field in ("task_id", "trial_id"):
            if np.asarray(split.arrays[field]).dtype.kind not in {"U", "S"}:
                raise ValueError(f"residual {split_name} {field} must use string dtype")
        quality = np.asarray(split.arrays["quality_weight"], dtype=np.float64)
        if not np.all(np.isfinite(quality)) or np.any(quality <= 0.0):
            raise ValueError(f"residual {split_name} quality weights must be positive")


def _canonical_task_phase_selectors(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("residual task_phase_selectors must be a non-empty object")
    result: dict[str, list[int]] = {}
    for task in sorted(value):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("residual task selector names must be non-empty strings")
        phases = value[task]
        if not isinstance(phases, list) or not phases:
            raise ValueError("residual task selector phases must be a non-empty array")
        if any(type(phase) is not int or phase < 0 for phase in phases):
            raise ValueError("residual selector phase ids must be non-negative integers")
        if phases != sorted(set(phases)):
            raise ValueError("residual selector phase ids must be sorted and unique")
        result[task] = list(phases)
    return result


def _selector_mask(
    arrays: Mapping[str, np.ndarray],
    selectors: Mapping[str, Sequence[int]],
) -> np.ndarray:
    tasks = np.asarray(arrays["task_id"])
    phases = np.asarray(arrays["phase_id"])
    mask = np.zeros(tasks.shape[0], dtype=bool)
    for task, phase_ids in selectors.items():
        mask |= (tasks == task) & np.isin(phases, np.asarray(phase_ids, dtype=phases.dtype))
    return mask


def _balanced_selector_weights(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> np.ndarray:
    tasks = np.asarray(arrays["task_id"])[selected]
    phases = np.asarray(arrays["phase_id"])[selected]
    trials = np.asarray(arrays["trial_id"])[selected]
    quality = np.asarray(arrays["quality_weight"], dtype=np.float64)[selected]
    if tasks.size == 0:
        raise ValueError("residual selector has no train samples")
    result = np.zeros(tasks.shape[0], dtype=np.float64)
    cells = sorted({(str(task), int(phase)) for task, phase in zip(tasks, phases, strict=True)})
    cell_total = 1.0 / len(cells)
    for task, phase in cells:
        cell = (tasks == task) & (phases == phase)
        cell_trials = np.unique(trials[cell])
        trial_quality = np.asarray(
            [float(np.mean(quality[cell & (trials == trial)])) for trial in cell_trials],
            dtype=np.float64,
        )
        trial_totals = cell_total * trial_quality / np.sum(trial_quality)
        for trial, trial_total in zip(cell_trials, trial_totals, strict=True):
            trial_mask = cell & (trials == trial)
            frame_quality = quality[trial_mask]
            result[trial_mask] = trial_total * frame_quality / np.sum(frame_quality)
    result /= np.mean(result)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("residual sample balancing produced invalid weights")
    return result


def _bounded_primary_preclip(
    targets: np.ndarray,
    basis: np.ndarray,
    upper: np.ndarray,
    *,
    config: StructuredResidualFitConfig,
) -> np.ndarray:
    coefficients = _solve_bounded(
        targets,
        basis,
        lower=np.zeros_like(upper),
        upper=upper,
        config=config,
    )
    decoded_preclip = coefficients @ np.asarray(basis, dtype=np.float64).T
    if not np.all(np.isfinite(decoded_preclip)) or np.any(decoded_preclip < -1e-10):
        raise RuntimeError("bounded primary projection produced invalid preclip excitation")
    return np.maximum(decoded_preclip, 0.0)


def _solve_bounded(
    targets: np.ndarray,
    basis: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    config: StructuredResidualFitConfig,
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    matrix = np.asarray(basis, dtype=np.float64)
    result = np.empty((values.shape[0], matrix.shape[1]), dtype=np.float64)
    for index, target in enumerate(values):
        solved = lsq_linear(
            matrix,
            target,
            bounds=(lower, upper),
            method="trf",
            lsq_solver="exact",
            tol=config.solver_tolerance,
            max_iter=config.solver_max_iterations,
        )
        if not solved.success or not np.all(np.isfinite(solved.x)):
            raise RuntimeError(f"bounded residual projection failed at frame {index}: {solved.message}")
        result[index] = solved.x
    return result


def _residual_metrics(
    targets: np.ndarray,
    primary_preclip: np.ndarray,
    residual_basis: np.ndarray,
    *,
    alpha: float,
    config: StructuredResidualFitConfig,
) -> dict[str, Any]:
    target_values = np.asarray(targets, dtype=np.float64)
    preclip = np.asarray(primary_preclip, dtype=np.float64)
    primary_decoded = np.clip(preclip, 0.0, 1.0)
    primary_residual = target_values - primary_decoded
    # Runtime applies one final clip after adding R rho.  Solve against the
    # un-clipped Wc so a negative residual can actually undo primary overshoot.
    correction_target = target_values - preclip
    dimension = residual_basis.shape[1]
    lower = np.full(dimension, -alpha, dtype=np.float64)
    upper = np.full(dimension, alpha, dtype=np.float64)
    coordinates = _solve_bounded(
        correction_target,
        residual_basis,
        lower=lower,
        upper=upper,
        config=config,
    )
    augmented = np.clip(
        preclip + coordinates @ np.asarray(residual_basis, dtype=np.float64).T,
        0.0,
        1.0,
    )
    augmented_residual = target_values - augmented
    primary_energy = float(np.sum(np.square(primary_residual)))
    augmented_energy = float(np.sum(np.square(augmented_residual)))
    if primary_energy <= config.energy_epsilon:
        # No unexplained primary energy means there is no held-out residual
        # demand to demonstrate.  Treating 0/0 as 100% improvement would let a
        # semantically empty task/phase group pass the promotion gate.
        reduction = 0.0
    else:
        reduction = float(np.clip(1.0 - augmented_energy / primary_energy, 0.0, 1.0))
    saturation = float(np.mean(np.abs(coordinates) >= alpha - max(1e-9, alpha * 1e-6)))
    return {
        "sample_count": int(np.asarray(targets).shape[0]),
        "primary_residual_energy": primary_energy,
        "augmented_residual_energy": augmented_energy,
        "residual_energy_reduction": reduction,
        "coordinate_saturation_fraction": saturation,
    }


def _canonicalize_column_signs(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0.0:
            result[:, column] *= -1.0
    return result


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--primary-basis", required=True)
    parser.add_argument("--coefficient-stats", required=True)
    parser.add_argument("--primitive-source-manifest", required=True)
    parser.add_argument("--expected-primitive-source-fingerprint", required=True)
    parser.add_argument("--residual-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--min-dimension", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=12)
    parser.add_argument("--max-row-l1-norm", type=float, default=2.0)
    parser.add_argument("--min-validation-residual-energy-reduction", type=float, default=0.01)
    parser.add_argument("--min-group-validation-residual-energy-reduction", type=float, default=0.01)
    parser.add_argument("--max-validation-coordinate-saturation-fraction", type=float, default=0.75)
    parser.add_argument("--solver-tolerance", type=float, default=1e-10)
    parser.add_argument("--solver-max-iterations", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StructuredResidualFitConfig(
        alpha=args.alpha,
        min_dimension=args.min_dimension,
        max_dimension=args.max_dimension,
        max_row_l1_norm=args.max_row_l1_norm,
        min_validation_residual_energy_reduction=(args.min_validation_residual_energy_reduction),
        min_group_validation_residual_energy_reduction=(args.min_group_validation_residual_energy_reduction),
        max_validation_coordinate_saturation_fraction=(args.max_validation_coordinate_saturation_fraction),
        solver_tolerance=args.solver_tolerance,
        solver_max_iterations=args.solver_max_iterations,
    )
    report = fit_structured_residual_basis(
        args.train,
        args.val,
        primary_basis_path=args.primary_basis,
        coefficient_statistics_path=args.coefficient_stats,
        primitive_source_manifest_path=args.primitive_source_manifest,
        expected_primitive_source_manifest_fingerprint=(args.expected_primitive_source_fingerprint),
        residual_mask_path=args.residual_mask,
        output_path=args.output,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
