from __future__ import annotations

import math

import numpy as np


def normalized(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    values = np.asarray(vec, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"expected a 1D vector, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector values must be finite")

    norm = float(np.linalg.norm(values))
    if norm < eps:
        return np.zeros_like(values, dtype=float)
    return values / norm


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    unit_a = normalized(a)
    unit_b = normalized(b)
    if unit_a.shape != unit_b.shape:
        raise ValueError(f"vector shapes must match, got {unit_a.shape} and {unit_b.shape}")

    cosine = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))
