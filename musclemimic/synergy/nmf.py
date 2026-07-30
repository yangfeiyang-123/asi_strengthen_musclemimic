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
    graph_penalty: float = 0.0
    objective: float = 0.0
    graph_lambda: float = 0.0


def fit_nmf(
    values: np.ndarray,
    *,
    rank: int,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    epsilon: float = 1e-10,
    graph_adjacency: np.ndarray | None = None,
    graph_lambda: float = 0.0,
) -> NMFResult:
    """Fit X ~= C @ W.T with optional verified Laplacian regularization.

    ``graph_lambda=0`` follows the historical code path exactly.  A positive
    coefficient applies the multiplicative update from the physiology guide
    and constrains each basis column to unit L2 norm during optimization to
    remove the otherwise unbounded C/W scaling ambiguity.
    """

    x = _validate_matrix(values)
    n_samples, n_muscles = x.shape
    rank = int(rank)
    if rank <= 0 or rank > min(n_samples, n_muscles):
        raise ValueError("NMF rank must be in [1,min(samples,muscles)]")
    if int(max_iter) <= 0 or float(tol) < 0.0 or float(epsilon) <= 0.0:
        raise ValueError("max_iter must be positive and tol non-negative")
    adjacency, degree, coefficient = _validate_graph_regularization(
        graph_adjacency,
        width=n_muscles,
        graph_lambda=graph_lambda,
    )
    rng = np.random.default_rng(int(seed))
    coefficients = rng.random((n_samples, rank)) + 0.05
    basis_t = rng.random((rank, n_muscles)) + 0.05
    previous = np.inf
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        coefficients *= (x @ basis_t.T) / (coefficients @ basis_t @ basis_t.T + epsilon)
        if coefficient > 0.0:
            basis_t *= (coefficients.T @ x + coefficient * basis_t @ adjacency) / (
                coefficients.T @ coefficients @ basis_t + coefficient * basis_t * degree[None, :] + epsilon
            )
            norms = np.linalg.norm(basis_t, axis=1)
            if np.any(norms <= epsilon) or not np.all(np.isfinite(norms)):
                raise RuntimeError("graph-NMF produced an empty synergy component")
            basis_t = basis_t / norms[:, None]
            coefficients = coefficients * norms[None, :]
        else:
            basis_t *= (coefficients.T @ x) / (coefficients.T @ coefficients @ basis_t + epsilon)
        if iteration == 1 or iteration % 10 == 0 or iteration == int(max_iter):
            reconstruction = coefficients @ basis_t
            loss = float(np.mean(np.square(x - reconstruction)))
            convergence_value = (
                loss
                if coefficient == 0.0
                else _graph_objective(
                    x,
                    reconstruction,
                    basis_t,
                    adjacency,
                    degree,
                    coefficient,
                )[1]
            )
            if np.isfinite(previous) and abs(previous - convergence_value) <= float(tol) * max(
                previous,
                epsilon,
            ):
                break
            previous = convergence_value
    basis = basis_t.T
    norms = np.linalg.norm(basis, axis=0)
    if np.any(norms <= epsilon):
        raise RuntimeError("NMF produced an empty synergy component")
    basis = basis / norms
    coefficients = coefficients * norms
    reconstruction = coefficients @ basis.T
    graph_penalty, objective = _graph_objective(
        x,
        reconstruction,
        basis.T,
        adjacency,
        degree,
        coefficient,
    )
    return NMFResult(
        basis=basis.astype(np.float64),
        coefficients=coefficients.astype(np.float64),
        reconstruction=reconstruction.astype(np.float64),
        loss=float(np.mean(np.square(x - reconstruction))),
        n_iter=iteration,
        rank=rank,
        seed=int(seed),
        graph_penalty=graph_penalty,
        objective=objective,
        graph_lambda=coefficient,
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
    graph_adjacency: np.ndarray | None = None,
    graph_lambda: float = 0.0,
) -> tuple[NMFResult, list[NMFResult]]:
    results = [
        fit_nmf(
            values,
            rank=rank,
            seed=int(seed),
            max_iter=max_iter,
            tol=tol,
            graph_adjacency=graph_adjacency,
            graph_lambda=graph_lambda,
        )
        for seed in seeds
    ]
    if not results:
        raise ValueError("at least one NMF initialization seed is required")
    key = (lambda item: item.objective) if float(graph_lambda) > 0.0 else (lambda item: item.loss)
    return min(results, key=key), results


def rank_scan(
    values: np.ndarray,
    *,
    ranks: Iterable[int],
    seeds: Iterable[int],
    max_iter: int = 1000,
    tol: float = 1e-6,
    graph_adjacency: np.ndarray | None = None,
    graph_lambda: float = 0.0,
) -> dict[int, tuple[NMFResult, list[NMFResult]]]:
    result: dict[int, tuple[NMFResult, list[NMFResult]]] = {}
    for rank in sorted({int(value) for value in ranks}):
        result[rank] = fit_best_initialization(
            values,
            rank=rank,
            seeds=seeds,
            max_iter=max_iter,
            tol=tol,
            graph_adjacency=graph_adjacency,
            graph_lambda=graph_lambda,
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


def _validate_graph_regularization(
    adjacency: np.ndarray | None,
    *,
    width: int,
    graph_lambda: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(graph_lambda, bool):
        raise ValueError("graph_lambda must be finite and non-negative")
    coefficient = float(graph_lambda)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("graph_lambda must be finite and non-negative")
    if coefficient == 0.0:
        if adjacency is not None:
            raise ValueError("graph_adjacency must be omitted when graph_lambda is zero")
        empty = np.zeros((width, width), dtype=np.float64)
        return empty, np.zeros((width,), dtype=np.float64), coefficient
    if adjacency is None:
        raise ValueError("positive graph_lambda requires graph_adjacency")
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.shape != (width, width):
        raise ValueError(f"graph_adjacency must have shape ({width}, {width})")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("graph_adjacency must be finite and non-negative")
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0.0):
        raise ValueError("graph_adjacency must be symmetric with a zero diagonal")
    if np.count_nonzero(np.triu(matrix, k=1)) == 0:
        raise ValueError("positive graph_lambda requires at least one graph edge")
    return matrix, np.sum(matrix, axis=1), coefficient


def _graph_objective(
    values: np.ndarray,
    reconstruction: np.ndarray,
    basis_t: np.ndarray,
    adjacency: np.ndarray,
    degree: np.ndarray,
    coefficient: float,
) -> tuple[float, float]:
    if coefficient == 0.0:
        loss = float(np.mean(np.square(values - reconstruction)))
        return 0.0, loss
    laplacian = np.diag(degree) - adjacency
    penalty = float(np.trace(basis_t @ laplacian @ basis_t.T))
    penalty = max(penalty, 0.0)
    objective = float(np.sum(np.square(values - reconstruction)) + coefficient * penalty)
    if not np.isfinite(penalty) or not np.isfinite(objective):
        raise RuntimeError("graph-NMF objective became non-finite")
    return penalty, objective
