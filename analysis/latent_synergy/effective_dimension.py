"""Measure configured versus actually used latent dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def participation_ratio(eigenvalues: np.ndarray, *, epsilon: float = 1e-12) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < -epsilon):
        raise ValueError("covariance eigenvalues must be finite, non-negative, and non-empty")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= epsilon:
        return 0.0
    return float(total * total / max(float(np.sum(values * values)), epsilon))


def effective_rank(singular_values: np.ndarray, *, relative_threshold: float = 1e-3) -> int:
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular values must be finite, non-negative, and non-empty")
    if not 0.0 <= float(relative_threshold) <= 1.0:
        raise ValueError("relative_threshold must lie in [0,1]")
    largest = float(np.max(values))
    if largest <= 0.0:
        return 0
    return int(np.sum(values > float(relative_threshold) * largest))


def effective_dimension_report(
    latents: np.ndarray,
    *,
    per_dimension_kl: np.ndarray | None = None,
    decoder_jacobians: np.ndarray | None = None,
    active_std_threshold: float = 1e-3,
    jacobian_relative_threshold: float = 1e-3,
) -> dict[str, Any]:
    z = _finite_matrix("latents", latents, min_rows=2)
    if float(active_std_threshold) < 0.0:
        raise ValueError("active_std_threshold must be non-negative")
    centered = z - np.mean(z, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(z.shape[0] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    standard_deviation = np.std(z, axis=0, ddof=1)
    positive = eigenvalues[eigenvalues > 1e-12]
    condition_number = None if positive.size == 0 else float(positive[0] / positive[-1])
    report: dict[str, Any] = {
        "schema_version": "latent_effective_dimension_v1",
        "num_samples": int(z.shape[0]),
        "configured_dimension": int(z.shape[1]),
        "participation_ratio_dimension": participation_ratio(eigenvalues),
        "active_unit_count": int(np.sum(standard_deviation > float(active_std_threshold))),
        "active_unit_fraction": float(np.mean(standard_deviation > float(active_std_threshold))),
        "latent_covariance_eigenvalues": eigenvalues.tolist(),
        "latent_standard_deviation": standard_deviation.tolist(),
        "latent_covariance_condition_number": condition_number,
    }
    if per_dimension_kl is not None:
        kl = np.asarray(per_dimension_kl, dtype=np.float64)
        if kl.ndim == 2:
            if kl.shape != z.shape:
                raise ValueError("per-sample KL must have the same shape as latents")
            kl = np.mean(kl, axis=0)
        kl = kl.reshape(-1)
        if kl.shape != (z.shape[1],) or not np.all(np.isfinite(kl)) or np.any(kl < -1e-10):
            raise ValueError("per_dimension_kl must be finite, non-negative, and match latent_dim")
        report["per_dimension_kl"] = np.maximum(kl, 0.0).tolist()
        report["total_mean_kl"] = float(np.sum(np.maximum(kl, 0.0)))
    if decoder_jacobians is not None:
        jacobians = np.asarray(decoder_jacobians, dtype=np.float64)
        if (
            jacobians.ndim != 3
            or jacobians.shape[0] != z.shape[0]
            or jacobians.shape[2] != z.shape[1]
            or not np.all(np.isfinite(jacobians))
        ):
            raise ValueError("decoder_jacobians must be finite [samples, muscles, latent_dim]")
        singular = np.linalg.svd(jacobians, compute_uv=False)
        ranks = np.asarray(
            [effective_rank(row, relative_threshold=jacobian_relative_threshold) for row in singular],
            dtype=np.float64,
        )
        report.update(
            {
                "jacobian_effective_rank_mean": float(np.mean(ranks)),
                "jacobian_effective_rank_std": float(np.std(ranks)),
                "jacobian_effective_rank_by_sample": ranks.astype(int).tolist(),
                "jacobian_singular_values_mean": np.mean(singular, axis=0).tolist(),
            }
        )
    return report


def _finite_matrix(name: str, value: np.ndarray, *, min_rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] <= 0:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix with >= {min_rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--latent-key", default="latents")
    parser.add_argument("--kl-key", default=None)
    parser.add_argument("--jacobian-key", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.input_npz, allow_pickle=False) as data:
        if args.latent_key not in data.files:
            raise ValueError(f"input is missing latent key {args.latent_key!r}")
        report = effective_dimension_report(
            data[args.latent_key],
            per_dimension_kl=None if args.kl_key is None else data[args.kl_key],
            decoder_jacobians=(None if args.jacobian_key is None else data[args.jacobian_key]),
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
