"""Reconstruction and conditioning metrics for muscle synergies."""

from __future__ import annotations

import numpy as np


def global_vaf(values: np.ndarray, reconstruction: np.ndarray, *, epsilon: float = 1e-12) -> float:
    """Variance accounted for using the zero-baseline convention for activations."""

    values, reconstruction = _paired(values, reconstruction)
    denominator = float(np.sum(np.square(values)))
    if denominator <= float(epsilon):
        raise ValueError("global VAF is undefined for an all-zero signal")
    return float(1.0 - np.sum(np.square(values - reconstruction)) / denominator)


def local_vaf(values: np.ndarray, reconstruction: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    values, reconstruction = _paired(values, reconstruction)
    denominator = np.sum(np.square(values), axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = denominator > float(epsilon)
    result[valid] = 1.0 - np.sum(np.square(values[:, valid] - reconstruction[:, valid]), axis=0) / denominator[valid]
    return result


def reconstruction_rmse(values: np.ndarray, reconstruction: np.ndarray) -> float:
    values, reconstruction = _paired(values, reconstruction)
    return float(np.sqrt(np.mean(np.square(values - reconstruction))))


def basis_condition_number(basis: np.ndarray) -> float:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("basis must be a finite matrix")
    return float(np.linalg.cond(matrix))


def reconstruction_metrics(values: np.ndarray, reconstruction: np.ndarray) -> dict:
    return {
        "global_vaf": global_vaf(values, reconstruction),
        "local_vaf": local_vaf(values, reconstruction).tolist(),
        "rmse": reconstruction_rmse(values, reconstruction),
    }


def _paired(values: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(values, dtype=np.float64)
    second = np.asarray(reconstruction, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("values and reconstruction must be same-shape matrices")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("values/reconstruction contain NaN/Inf")
    return first, second
