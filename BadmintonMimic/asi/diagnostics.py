from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_tracking_diagnostics(
    *,
    reference_trans: np.ndarray,
    actual_trans: np.ndarray,
    actual_foot_points: np.ndarray | None = None,
    stance_mask: np.ndarray | None = None,
    fps: float,
    root_error_threshold_m: float = 0.75,
    stance_speed_threshold_mps: float = 1.0,
) -> dict[str, Any]:
    reference = np.asarray(reference_trans, dtype=np.float32)
    actual = np.asarray(actual_trans, dtype=np.float32)
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_trans and actual_trans must both have shape [T, 3]")
    if float(fps) <= 0:
        raise ValueError("fps must be positive")

    failed: dict[int, list[dict[str, Any]]] = {}
    root_error = np.linalg.norm(actual - reference, axis=1)
    for frame in np.where(root_error > float(root_error_threshold_m))[0]:
        _append_failure(
            failed,
            int(frame),
            "root_tracking_error",
            float(root_error[frame]),
            float(root_error_threshold_m),
        )

    max_stance_speed = 0.0
    if actual_foot_points is not None and stance_mask is not None:
        feet = np.asarray(actual_foot_points, dtype=np.float32)
        stance = np.asarray(stance_mask, dtype=np.bool_)
        if feet.ndim != 3 or feet.shape[-1] != 3:
            raise ValueError("actual_foot_points must have shape [T, K, 3]")
        if stance.shape != feet.shape[:2]:
            raise ValueError("stance_mask must match actual_foot_points frame/foot dimensions")
        if feet.shape[0] != reference.shape[0]:
            raise ValueError("actual_foot_points frame count must match reference_trans")
        if feet.shape[0] > 1:
            speeds = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * float(fps)
            active = stance[1:]
            if np.any(active):
                max_stance_speed = float(np.max(speeds[active]))
            for rel_frame, foot_index in np.argwhere((speeds > float(stance_speed_threshold_mps)) & active):
                frame = int(rel_frame) + 1
                _append_failure(
                    failed,
                    frame,
                    "stance_foot_slip",
                    float(speeds[rel_frame, foot_index]),
                    float(stance_speed_threshold_mps),
                    foot_index=int(foot_index),
                )

    failed_frames = [
        {"frame": frame, **failure}
        for frame in sorted(failed)
        for failure in failed[frame]
    ]
    return {
        "failed_frames": failed_frames,
        "summary": {
            "num_frames": int(reference.shape[0]),
            "num_failed_frames": int(len(failed)),
            "max_root_error_m": float(np.max(root_error)) if root_error.size else 0.0,
            "max_stance_foot_speed_mps": max_stance_speed,
        },
    }


def _append_failure(
    failed: dict[int, list[dict[str, Any]]],
    frame: int,
    name: str,
    value: float,
    threshold: float,
    **extra: Any,
) -> None:
    failed.setdefault(int(frame), []).append(
        {
            "failure": name,
            "value": float(value),
            "threshold": float(threshold),
            **extra,
        }
    )

