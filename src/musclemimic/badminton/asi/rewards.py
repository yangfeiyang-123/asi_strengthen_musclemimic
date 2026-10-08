from __future__ import annotations

from typing import Any

import numpy as np


def compute_body_graph_laplacian_error(actual_laplacian: np.ndarray, reference_laplacian: np.ndarray) -> float:
    actual = np.asarray(actual_laplacian, dtype=np.float32)
    reference = np.asarray(reference_laplacian, dtype=np.float32)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[-1] != 3:
        raise ValueError("actual_laplacian and reference_laplian must both have shape [K, 3]")
    return float(np.mean(np.linalg.norm(actual - reference, axis=-1)))


def compute_contact_tracking_components(
    *,
    actual_body_laplacian: np.ndarray | None = None,
    reference_body_laplacian: np.ndarray | None = None,
    actual_foot_points: np.ndarray,
    reference_foot_points: np.ndarray,
    stance_mask: np.ndarray,
    fps: float,
    body_graph_scale: float = 20.0,
    foot_height_scale: float = 80.0,
    foot_velocity_scale: float = 8.0,
) -> dict[str, Any]:
    actual_feet = np.asarray(actual_foot_points, dtype=np.float32)
    reference_feet = np.asarray(reference_foot_points, dtype=np.float32)
    stance = np.asarray(stance_mask, dtype=np.bool_)
    if actual_feet.shape != reference_feet.shape or actual_feet.ndim != 3 or actual_feet.shape[-1] != 3:
        raise ValueError("actual_foot_points and reference_foot_points must both have shape [T, K, 3]")
    if stance.shape != actual_feet.shape[:2]:
        raise ValueError("stance_mask must match foot point frame/foot dimensions")
    if float(fps) <= 0:
        raise ValueError("fps must be positive")

    body_graph_error = 0.0
    if actual_body_laplacian is not None and reference_body_laplacian is not None:
        body_graph_error = compute_body_graph_laplacian_error(actual_body_laplacian, reference_body_laplacian)

    if np.any(stance):
        height_error = float(np.mean(np.abs(actual_feet[..., 2][stance] - reference_feet[..., 2][stance])))
    else:
        height_error = 0.0

    speed = np.zeros(actual_feet.shape[:2], dtype=np.float32)
    if actual_feet.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(actual_feet, axis=0), axis=-1) * float(fps)
    stance_speed = speed[stance]
    foot_speed_mps = float(np.mean(stance_speed)) if stance_speed.size else 0.0

    return {
        "body_graph_error": body_graph_error,
        "foot_contact_height_error": height_error,
        "foot_contact_speed_mps": foot_speed_mps,
        "reward_body_graph": _exp_reward(body_graph_error, body_graph_scale),
        "reward_foot_contact_height": _exp_reward(height_error, foot_height_scale),
        "reward_foot_contact_velocity": _exp_reward(foot_speed_mps, foot_velocity_scale),
    }


def _exp_reward(error: float, scale: float) -> float:
    return float(np.exp(-float(scale) * max(0.0, float(error))))
