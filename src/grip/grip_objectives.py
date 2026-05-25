from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def weighted_site_target_residuals(
    current_sites: Mapping[str, np.ndarray],
    target_sites: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    residuals = []
    for site_name in sorted(target_sites):
        current = _site_vector(current_sites, site_name, "current_sites")
        target = _site_vector(target_sites, site_name, "target_sites")
        weight = _finite_float(weights.get(site_name, 1.0), f"weights[{site_name!r}]")
        residuals.append((current - target) * weight)

    if not residuals:
        return np.array([], dtype=float)
    return np.concatenate(residuals)


def mean_site_error(
    current_sites: Mapping[str, np.ndarray],
    target_sites: Mapping[str, np.ndarray],
) -> float:
    if not target_sites:
        return 0.0

    errors = [
        float(np.linalg.norm(_site_vector(current_sites, site_name, "current_sites") - _site_vector(target_sites, site_name, "target_sites")))
        for site_name in sorted(target_sites)
    ]
    return float(np.mean(errors))


def joint_limit_margin_cost(qpos_values: np.ndarray, ranges: np.ndarray, margin: float = 0.03) -> float:
    qpos = np.asarray(qpos_values, dtype=float)
    limits = np.asarray(ranges, dtype=float)
    if limits.size == 0:
        return 0.0

    margin = _finite_float(margin, "margin")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if qpos.ndim != 1:
        raise ValueError(f"qpos_values must be 1D, got shape {qpos.shape}")
    if limits.shape != (qpos.shape[0], 2):
        raise ValueError(f"ranges must have shape ({qpos.shape[0]}, 2), got {limits.shape}")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(limits)):
        raise ValueError("qpos_values and ranges must be finite")

    lower = limits[:, 0]
    upper = limits[:, 1]
    if np.any(lower > upper):
        raise ValueError("each joint range lower bound must be <= upper bound")

    lower_margin = np.maximum(0.0, margin - (qpos - lower))
    upper_margin = np.maximum(0.0, margin - (upper - qpos))
    return float(np.sum(lower_margin**2 + upper_margin**2))


def _site_vector(sites: Mapping[str, np.ndarray], site_name: str, context: str) -> np.ndarray:
    if site_name not in sites:
        raise KeyError(f"{context} missing site {site_name!r}")

    value = np.asarray(sites[site_name], dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{context}[{site_name!r}] must have shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{context}[{site_name!r}] must be finite")
    return value


def _finite_float(value: object, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return number
