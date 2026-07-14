"""Quantify latent-to-synergy predictability and representation geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PHASE_NAMES = ("ready", "backswing", "acceleration", "impact", "followthrough", "recovery")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _paired_matrices(x, y)
    xc = x - np.mean(x, axis=0, keepdims=True)
    yc = y - np.mean(y, axis=0, keepdims=True)
    cross = xc.T @ yc
    denominator = np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro")
    if denominator <= 1e-12:
        raise ValueError("linear CKA is undefined for a zero-variance representation")
    return float(np.sum(cross * cross) / denominator)


def canonical_correlations(
    x: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float = 1e-6,
) -> np.ndarray:
    x, y = _paired_matrices(x, y)
    if float(regularization) <= 0.0:
        raise ValueError("CCA regularization must be positive")
    xc = x - np.mean(x, axis=0, keepdims=True)
    yc = y - np.mean(y, axis=0, keepdims=True)
    scale = max(x.shape[0] - 1, 1)
    cxx = xc.T @ xc / scale + float(regularization) * np.eye(x.shape[1])
    cyy = yc.T @ yc / scale + float(regularization) * np.eye(y.shape[1])
    cxy = xc.T @ yc / scale
    wx = _inverse_sqrt_psd(cxx)
    wy = _inverse_sqrt_psd(cyy)
    values = np.linalg.svd(wx @ cxy @ wy, compute_uv=False)
    return np.clip(values, 0.0, 1.0)


def procrustes_aligned_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Rotation-invariant normalized nuclear cross-covariance similarity."""

    x, y = _paired_matrices(x, y)
    xc = x - np.mean(x, axis=0, keepdims=True)
    yc = y - np.mean(y, axis=0, keepdims=True)
    denominator = np.linalg.norm(xc, "fro") * np.linalg.norm(yc, "fro")
    if denominator <= 1e-12:
        raise ValueError("Procrustes similarity is undefined for zero-variance data")
    return float(np.sum(np.linalg.svd(xc.T @ yc, compute_uv=False)) / denominator)


def ridge_predictability(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float = 1e-3,
) -> dict[str, Any]:
    x, y = _paired_matrices(x, y)
    train, test = _validate_split(train_indices, test_indices, x.shape[0])
    if float(alpha) < 0.0:
        raise ValueError("ridge alpha must be non-negative")
    x_mean = np.mean(x[train], axis=0, keepdims=True)
    x_scale = np.std(x[train], axis=0, keepdims=True)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_mean = np.mean(y[train], axis=0, keepdims=True)
    design_train = np.concatenate([np.ones((train.size, 1)), (x[train] - x_mean) / x_scale], axis=1)
    design_test = np.concatenate([np.ones((test.size, 1)), (x[test] - x_mean) / x_scale], axis=1)
    penalty = float(alpha) * np.eye(design_train.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design_train.T @ design_train + penalty,
        design_train.T @ (y[train] - y_mean),
    )
    predicted = design_test @ coefficients + y_mean
    return _prediction_report(y[test], predicted)


def rbf_kernel_ridge_predictability(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float = 1e-3,
    gamma: float | None = None,
) -> dict[str, Any]:
    x, y = _paired_matrices(x, y)
    train, test = _validate_split(train_indices, test_indices, x.shape[0])
    if float(alpha) <= 0.0:
        raise ValueError("kernel-ridge alpha must be positive")
    x_train = x[train]
    distances = _squared_distances(x_train, x_train)
    if gamma is None:
        positive = distances[distances > 1e-12]
        if positive.size == 0:
            raise ValueError("RBF bandwidth is undefined for duplicate/constant latent samples")
        gamma = 1.0 / float(np.median(positive))
    if not np.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("RBF gamma must be finite and positive")
    kernel_train = np.exp(-float(gamma) * distances)
    dual = np.linalg.solve(
        kernel_train + float(alpha) * np.eye(train.size),
        y[train],
    )
    predicted = np.exp(-float(gamma) * _squared_distances(x[test], x_train)) @ dual
    return _prediction_report(y[test], predicted) | {"gamma": float(gamma)}


def representation_report(
    latents: np.ndarray,
    synergy_coefficients: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    phase_ids: np.ndarray | None = None,
    seed: int = 0,
    test_fraction: float = 0.2,
    ridge_alpha: float = 1e-3,
    nonlinear_alpha: float = 1e-3,
    require_all_phases: bool = False,
) -> dict[str, Any]:
    z, c = _paired_matrices(latents, synergy_coefficients)
    if z.shape[0] < 5:
        raise ValueError("representation analysis requires at least five paired samples")
    train, test = _resolve_split(z.shape[0], train_mask=train_mask, seed=int(seed), test_fraction=float(test_fraction))
    correlations = canonical_correlations(z, c)
    report: dict[str, Any] = {
        "schema_version": "latent_synergy_representation_v1",
        "num_samples": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "synergy_dim": int(c.shape[1]),
        "num_train": int(train.size),
        "num_test": int(test.size),
        "ridge": ridge_predictability(z, c, train_indices=train, test_indices=test, alpha=float(ridge_alpha)),
        "nonlinear_rbf_kernel_ridge": rbf_kernel_ridge_predictability(
            z, c, train_indices=train, test_indices=test, alpha=float(nonlinear_alpha)
        ),
        "canonical_correlations": correlations.tolist(),
        "mean_canonical_correlation": float(np.mean(correlations)),
        "linear_cka": linear_cka(z, c),
        "procrustes_aligned_similarity": procrustes_aligned_similarity(z, c),
    }
    if phase_ids is not None:
        phases = np.asarray(phase_ids)
        if phases.shape != (z.shape[0],) or not np.all(np.isfinite(phases)):
            raise ValueError("phase_ids must be a finite vector matching paired samples")
        if not np.all(phases == np.floor(phases)):
            raise ValueError("phase_ids must contain integer values")
        phases = phases.astype(np.int64)
        unknown = sorted(set(phases.tolist()) - set(range(len(PHASE_NAMES))))
        if unknown:
            raise ValueError(f"phase_ids contain unknown values: {unknown}")
        by_phase: dict[str, Any] = {}
        missing: list[str] = []
        for phase_id, name in enumerate(PHASE_NAMES):
            selected = np.flatnonzero(phases == phase_id)
            if selected.size < 2:
                missing.append(name)
                continue
            by_phase[name] = {
                "num_samples": int(selected.size),
                "linear_cka": linear_cka(z[selected], c[selected]),
                "mean_canonical_correlation": float(np.mean(canonical_correlations(z[selected], c[selected]))),
                "procrustes_aligned_similarity": procrustes_aligned_similarity(z[selected], c[selected]),
            }
        if require_all_phases and missing:
            raise ValueError(f"phase-conditioned representation is missing phases: {missing}")
        report["by_phase"] = by_phase
        report["missing_phases"] = missing
    elif require_all_phases:
        raise ValueError("require_all_phases=True requires phase_ids")
    return report


def _prediction_report(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if target.shape != predicted.shape or target.shape[0] == 0:
        raise ValueError("prediction target/output shapes are inconsistent or empty")
    residual_sum = np.sum(np.square(target - predicted), axis=0)
    centered_sum = np.sum(np.square(target - np.mean(target, axis=0, keepdims=True)), axis=0)
    valid = centered_sum > 1e-12
    if not np.any(valid):
        raise ValueError("held-out synergy coefficients have zero variance")
    per_output = np.full(target.shape[1], np.nan, dtype=np.float64)
    per_output[valid] = 1.0 - residual_sum[valid] / centered_sum[valid]
    aggregate = 1.0 - float(np.sum(residual_sum[valid])) / float(np.sum(centered_sum[valid]))
    return {
        "r2": float(aggregate),
        "r2_per_synergy": [None if not valid[index] else float(per_output[index]) for index in range(target.shape[1])],
        "mse": float(np.mean(np.square(target - predicted))),
        "num_test": int(target.shape[0]),
    }


def _inverse_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0.0):
        raise ValueError("regularized covariance is not positive definite")
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True) + np.sum(y * y, axis=1, keepdims=True).T - 2.0 * (x @ y.T),
        0.0,
    )


def _paired_matrices(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] != right.shape[0]
        or left.shape[0] <= 1
        or left.shape[1] <= 0
        or right.shape[1] <= 0
    ):
        raise ValueError("representations must be paired, non-empty rank-2 matrices")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("representations contain non-finite values")
    return left, right


def _resolve_split(
    size: int,
    *,
    train_mask: np.ndarray | None,
    seed: int,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if train_mask is not None:
        mask = np.asarray(train_mask)
        if mask.shape != (size,):
            raise ValueError("train_mask must match sample count")
        if mask.dtype != np.bool_:
            if not np.all(np.isin(mask, [0, 1])):
                raise ValueError("train_mask must be boolean or 0/1")
            mask = mask.astype(bool)
        return _validate_split(np.flatnonzero(mask), np.flatnonzero(~mask), size)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    order = np.random.default_rng(seed).permutation(size)
    test_size = min(size - 2, max(2, round(size * test_fraction)))
    return _validate_split(order[test_size:], order[:test_size], size)


def _validate_split(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    test = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    if train.size < 2 or test.size < 2:
        raise ValueError("representation split requires at least two train and two test samples")
    if (
        np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= size)
        or np.any(test >= size)
        or len(set(train.tolist())) != train.size
        or len(set(test.tolist())) != test.size
        or set(train.tolist()) & set(test.tolist())
    ):
        raise ValueError("train/test indices must be unique, disjoint, and in range")
    return train, test


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--latent-key", default="latents")
    parser.add_argument("--coefficient-key", default="synergy_coefficients")
    parser.add_argument("--train-mask-key", default=None)
    parser.add_argument("--phase-key", default=None)
    parser.add_argument("--require-all-phases", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.input_npz, allow_pickle=False) as data:
        for key in (args.latent_key, args.coefficient_key):
            if key not in data.files:
                raise ValueError(f"input is missing key {key!r}")
        report = representation_report(
            data[args.latent_key],
            data[args.coefficient_key],
            train_mask=None if args.train_mask_key is None else data[args.train_mask_key],
            phase_ids=None if args.phase_key is None else data[args.phase_key],
            seed=int(args.seed),
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
