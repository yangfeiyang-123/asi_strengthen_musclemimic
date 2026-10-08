"""Compare decoder-Jacobian and fixed-synergy subspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles

from musclemimic.latent_muscle.phase_contract import (
    FOREHAND_PHASE_NAMES,
    phase_items,
)

PHASE_NAMES = FOREHAND_PHASE_NAMES


def subspace_alignment(jacobian: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    j = _finite_matrix("jacobian", jacobian)
    w = _finite_matrix("basis", basis)
    if j.shape[0] != w.shape[0]:
        raise ValueError("jacobian and synergy basis must share the muscle dimension")
    qj = _orthonormal_span(j)
    qw = _orthonormal_span(w)
    angles = subspace_angles(qj, qw)
    correlations = np.cos(angles)
    projected = qw @ (qw.T @ j)
    projection_score = float(np.sum(projected * projected) / max(float(np.sum(j * j)), 1e-12))
    return {
        "principal_angles_radians": angles.tolist(),
        "principal_angles_degrees": np.degrees(angles).tolist(),
        "canonical_correlations": correlations.tolist(),
        "mean_canonical_correlation": float(np.mean(correlations)),
        "minimum_canonical_correlation": float(np.min(correlations)),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "projection_score": projection_score,
        "jacobian_rank": int(qj.shape[1]),
        "synergy_rank": int(qw.shape[1]),
    }


def jacobian_alignment_report(
    jacobians: np.ndarray,
    basis: np.ndarray,
    *,
    phase_ids: np.ndarray | None = None,
    require_all_phases: bool = False,
    phase_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    array = np.asarray(jacobians, dtype=np.float64)
    w = _finite_matrix("basis", basis)
    if array.ndim != 3 or array.shape[0] <= 0 or array.shape[1] != w.shape[0]:
        raise ValueError("jacobians must be non-empty [samples, muscles, latent_dim]")
    if not np.all(np.isfinite(array)):
        raise ValueError("jacobians contain non-finite values")
    samples = [subspace_alignment(item, w) for item in array]
    report: dict[str, Any] = {
        "schema_version": "latent_synergy_jacobian_alignment_v1",
        "num_samples": int(array.shape[0]),
        "muscle_dim": int(array.shape[1]),
        "latent_dim": int(array.shape[2]),
        "synergy_rank": int(w.shape[1]),
        "projection_score_mean": float(np.mean([item["projection_score"] for item in samples])),
        "projection_score_std": float(np.std([item["projection_score"] for item in samples])),
        "grassmann_distance_mean": float(np.mean([item["grassmann_distance"] for item in samples])),
        "grassmann_distance_std": float(np.std([item["grassmann_distance"] for item in samples])),
        "mean_canonical_correlation": float(np.mean([item["mean_canonical_correlation"] for item in samples])),
        "per_sample": samples,
    }
    if phase_ids is not None:
        phases = np.asarray(phase_ids)
        if phases.shape != (array.shape[0],) or not np.all(np.isfinite(phases)):
            raise ValueError("phase_ids must be a finite vector matching jacobian samples")
        if not np.all(phases == np.floor(phases)):
            raise ValueError("phase_ids must contain integers")
        phases = phases.astype(np.int64)
        phases_and_names = phase_items(phase_contract)
        known_ids = {phase_id for phase_id, _name in phases_and_names}
        unknown = sorted(set(phases.tolist()) - known_ids)
        if unknown:
            raise ValueError(f"phase_ids contain unknown values: {unknown}")
        by_phase: dict[str, Any] = {}
        missing: list[str] = []
        for phase_id, name in phases_and_names:
            selected = np.flatnonzero(phases == phase_id)
            if selected.size == 0:
                missing.append(name)
                continue
            phase_samples = [samples[int(index)] for index in selected]
            by_phase[name] = {
                "num_samples": int(selected.size),
                "projection_score_mean": float(np.mean([item["projection_score"] for item in phase_samples])),
                "grassmann_distance_mean": float(np.mean([item["grassmann_distance"] for item in phase_samples])),
                "mean_canonical_correlation": float(
                    np.mean([item["mean_canonical_correlation"] for item in phase_samples])
                ),
            }
        if require_all_phases and missing:
            raise ValueError(f"phase-conditioned alignment is missing phases: {missing}")
        report["by_phase"] = by_phase
        report["missing_phases"] = missing
    elif require_all_phases:
        raise ValueError("require_all_phases=True requires phase_ids")
    return report


def _orthonormal_span(matrix: np.ndarray) -> np.ndarray:
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        raise ValueError("subspace matrix has zero numerical rank")
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    return u[:, :rank]


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty rank-2 matrix")
    return array


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--basis-npy", type=Path, required=True)
    parser.add_argument("--jacobian-key", default="jacobians")
    parser.add_argument("--phase-key", default=None)
    parser.add_argument("--require-all-phases", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.input_npz, allow_pickle=False) as data:
        if args.jacobian_key not in data.files:
            raise ValueError(f"input is missing jacobian key {args.jacobian_key!r}")
        phases = None if args.phase_key is None else data[args.phase_key]
        report = jacobian_alignment_report(
            data[args.jacobian_key],
            np.load(args.basis_npy, allow_pickle=False),
            phase_ids=phases,
            require_all_phases=bool(args.require_all_phases),
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
