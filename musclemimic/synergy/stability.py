"""Permutation-invariant stability diagnostics for fitted synergy bases."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from musclemimic.synergy.nmf import NMFResult, fit_nmf


def match_bases(reference: np.ndarray, candidate: np.ndarray) -> dict:
    first = _normalized_basis(reference)
    second = _normalized_basis(candidate)
    if first.shape != second.shape:
        raise ValueError(f"basis shapes differ: {first.shape} vs {second.shape}")
    similarity = first.T @ second
    rows, columns = linear_sum_assignment(-similarity)
    order = np.argsort(rows)
    rows = rows[order]
    columns = columns[order]
    matched = similarity[rows, columns]
    return {
        "mean_similarity": float(np.mean(matched)),
        "min_similarity": float(np.min(matched)),
        "matched_similarity": matched.astype(float).tolist(),
        "candidate_permutation": columns.astype(int).tolist(),
    }


def initialization_stability(results: Iterable[NMFResult]) -> dict:
    items = list(results)
    if len(items) < 2:
        return {"mean_similarity": 1.0, "min_similarity": 1.0, "pair_count": 0}
    scores = []
    for i, first in enumerate(items):
        for second in items[i + 1 :]:
            scores.append(match_bases(first.basis, second.basis)["mean_similarity"])
    return {
        "mean_similarity": float(np.mean(scores)),
        "min_similarity": float(np.min(scores)),
        "pair_count": len(scores),
    }


def split_half_stability(
    values: np.ndarray,
    *,
    rank: int,
    repeats: int = 5,
    seed: int = 0,
    max_iter: int = 500,
) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < max(4, 2 * int(rank)):
        raise ValueError("split-half stability needs at least max(4,2*rank) samples")
    rng = np.random.default_rng(int(seed))
    scores = []
    for repeat in range(int(repeats)):
        order = rng.permutation(x.shape[0])
        midpoint = x.shape[0] // 2
        first = fit_nmf(x[order[:midpoint]], rank=rank, seed=seed + repeat * 2, max_iter=max_iter)
        second = fit_nmf(x[order[midpoint:]], rank=rank, seed=seed + repeat * 2 + 1, max_iter=max_iter)
        scores.append(match_bases(first.basis, second.basis)["mean_similarity"])
    return _score_summary(scores, repeats=int(repeats), seed=int(seed))


def bootstrap_stability(
    values: np.ndarray,
    *,
    reference_basis: np.ndarray,
    rank: int,
    repeats: int = 10,
    seed: int = 0,
    max_iter: int = 500,
) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < int(rank):
        raise ValueError("bootstrap stability input is too small")
    rng = np.random.default_rng(int(seed))
    scores = []
    for repeat in range(int(repeats)):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        fitted = fit_nmf(x[indices], rank=rank, seed=seed + repeat, max_iter=max_iter)
        scores.append(match_bases(reference_basis, fitted.basis)["mean_similarity"])
    return _score_summary(scores, repeats=int(repeats), seed=int(seed))


def cross_trial_stability(bases: Iterable[np.ndarray]) -> dict:
    items = list(bases)
    if len(items) < 2:
        return {"mean_similarity": 1.0, "min_similarity": 1.0, "pair_count": 0}
    scores = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            scores.append(match_bases(first, second)["mean_similarity"])
    return {
        "mean_similarity": float(np.mean(scores)),
        "min_similarity": float(np.min(scores)),
        "pair_count": len(scores),
    }


def _normalized_basis(value: np.ndarray) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.ndim != 2 or np.min(basis) < 0.0 or not np.all(np.isfinite(basis)):
        raise ValueError("basis must be a finite non-negative matrix")
    norm = np.linalg.norm(basis, axis=0)
    if np.any(norm <= 1e-12):
        raise ValueError("basis contains an empty component")
    return basis / norm


def _score_summary(scores: list[float], *, repeats: int, seed: int) -> dict:
    if not scores:
        raise ValueError("stability repeats must be positive")
    return {
        "mean_similarity": float(np.mean(scores)),
        "min_similarity": float(np.min(scores)),
        "std_similarity": float(np.std(scores)),
        "scores": [float(value) for value in scores],
        "repeats": repeats,
        "seed": seed,
    }
