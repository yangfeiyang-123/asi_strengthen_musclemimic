"""Small deterministic NMF implementation for offline synergy extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class NMFResult:
    basis: np.ndarray
    coefficients: np.ndarray
    reconstruction: np.ndarray
    loss: float
    n_iter: int
    rank: int
    seed: int


def fit_nmf(
    values: np.ndarray,
    *,
    rank: int,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    epsilon: float = 1e-10,
) -> NMFResult:
    """Fit X ~= C @ W.T with non-negative C and column-normalized W."""

    x = _validate_matrix(values)
    n_samples, n_muscles = x.shape
    rank = int(rank)
    if rank <= 0 or rank > min(n_samples, n_muscles):
        raise ValueError("NMF rank must be in [1,min(samples,muscles)]")
    if int(max_iter) <= 0 or float(tol) < 0.0:
        raise ValueError("max_iter must be positive and tol non-negative")
    rng = np.random.default_rng(int(seed))
    coefficients = rng.random((n_samples, rank)) + 0.05
    basis_t = rng.random((rank, n_muscles)) + 0.05
    previous = np.inf
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        coefficients *= (x @ basis_t.T) / (coefficients @ basis_t @ basis_t.T + epsilon)
        basis_t *= (coefficients.T @ x) / (coefficients.T @ coefficients @ basis_t + epsilon)
        if iteration == 1 or iteration % 10 == 0 or iteration == int(max_iter):
            reconstruction = coefficients @ basis_t
            loss = float(np.mean(np.square(x - reconstruction)))
            if np.isfinite(previous) and abs(previous - loss) <= float(tol) * max(previous, epsilon):
                break
            previous = loss
    basis = basis_t.T
    norms = np.linalg.norm(basis, axis=0)
    if np.any(norms <= epsilon):
        raise RuntimeError("NMF produced an empty synergy component")
    basis = basis / norms
    coefficients = coefficients * norms
    reconstruction = coefficients @ basis.T
    return NMFResult(
        basis=basis.astype(np.float64),
        coefficients=coefficients.astype(np.float64),
        reconstruction=reconstruction.astype(np.float64),
        loss=float(np.mean(np.square(x - reconstruction))),
        n_iter=iteration,
        rank=rank,
        seed=int(seed),
    )


def transform_nmf(values: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _validate_matrix(values)
    w = np.asarray(basis, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != x.shape[1] or np.min(w) < 0.0 or not np.all(np.isfinite(w)):
        raise ValueError("basis must be finite non-negative [muscles,rank]")
    coefficients = np.stack([nnls(w, row)[0] for row in x], axis=0)
    return coefficients, coefficients @ w.T


def fit_best_initialization(
    values: np.ndarray,
    *,
    rank: int,
    seeds: Iterable[int],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[NMFResult, list[NMFResult]]:
    results = [fit_nmf(values, rank=rank, seed=int(seed), max_iter=max_iter, tol=tol) for seed in seeds]
    if not results:
        raise ValueError("at least one NMF initialization seed is required")
    return min(results, key=lambda item: item.loss), results


def rank_scan(
    values: np.ndarray,
    *,
    ranks: Iterable[int],
    seeds: Iterable[int],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[int, tuple[NMFResult, list[NMFResult]]]:
    result: dict[int, tuple[NMFResult, list[NMFResult]]] = {}
    for rank in sorted({int(value) for value in ranks}):
        result[rank] = fit_best_initialization(
            values,
            rank=rank,
            seeds=seeds,
            max_iter=max_iter,
            tol=tol,
        )
    if not result:
        raise ValueError("rank scan requires at least one rank")
    return result


def _validate_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("NMF input must be a non-empty matrix")
    if not np.all(np.isfinite(array)) or np.min(array) < -1e-10:
        raise ValueError("NMF input must be finite and non-negative")
    return np.maximum(array, 0.0)
