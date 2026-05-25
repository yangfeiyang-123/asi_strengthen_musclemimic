"""Pure NumPy helpers for root tracking diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def select_latest_checkpoint(checkpoint_root: str | Path) -> Path:
    """Return the checkpoint_* directory with the largest numeric suffix."""
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint root is not a directory: {root}")

    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        suffix = child.name.removeprefix("checkpoint_")
        if suffix.isdigit():
            candidates.append((int(suffix), child))

    if not candidates:
        raise ValueError(f"no checkpoint_* directories found in checkpoint root: {root}")

    return max(candidates, key=lambda item: item[0])[1]


def compute_root_reference_metrics(
    *,
    qpos,
    qvel=None,
    site_xpos=None,
    right_hand_site_index=None,
    frequency=None,
) -> dict[str, float]:
    qpos_array = _as_2d_array(qpos, name="qpos")
    root_xy = _root_xy(qpos_array, name="qpos")

    speed = _root_xy_speed(root_xy=root_xy, qvel=qvel, frequency=frequency)
    yaw = _root_yaw(qpos_array)
    right_hand_path_length = _right_hand_world_path_length(
        site_xpos=site_xpos,
        right_hand_site_index=right_hand_site_index,
    )
    total_displacement = _total_displacement(root_xy)
    path_length = _path_length(root_xy)

    return {
        "reference_root_xy_total_displacement": total_displacement,
        "reference_root_xy_path_length": path_length,
        "reference_root_xy_peak_speed": float(np.max(speed)) if speed.size else 0.0,
        "reference_root_yaw_change": _yaw_change(yaw),
        "right_hand_world_path_length": right_hand_path_length,
        "reference_frame_count": float(qpos_array.shape[0]),
        "reference_frequency": float(frequency) if frequency is not None else 0.0,
        "reference_root_xy_path_displacement_ratio": _safe_ratio(path_length, total_displacement),
        "reference_qpos_max_abs_step": _max_abs_step(qpos_array),
        "reference_qvel_max_abs": _max_abs_value(qvel),
        "reference_site_xpos_max_step": _site_xpos_max_step(site_xpos),
    }


def compute_rollout_root_metrics(
    *,
    reference_qpos,
    rollout_qpos,
    reference_qvel=None,
    rollout_qvel=None,
    frequency=None,
) -> dict[str, float]:
    reference_qpos_array = _as_2d_array(reference_qpos, name="reference_qpos")
    rollout_qpos_array = _as_2d_array(rollout_qpos, name="rollout_qpos")
    frame_count = min(reference_qpos_array.shape[0], rollout_qpos_array.shape[0])
    reference_qpos_array = reference_qpos_array[:frame_count]
    rollout_qpos_array = rollout_qpos_array[:frame_count]
    reference_qvel_array = _prefix_qvel(reference_qvel, frame_count=frame_count, name="reference_qvel")
    rollout_qvel_array = _prefix_qvel(rollout_qvel, frame_count=frame_count, name="rollout_qvel")

    reference_root_xy = _root_xy(reference_qpos_array, name="reference_qpos")
    rollout_root_xy = _root_xy(rollout_qpos_array, name="rollout_qpos")

    metrics = compute_root_reference_metrics(
        qpos=reference_qpos_array,
        qvel=reference_qvel_array,
        frequency=frequency,
    )

    reference_displacement = metrics["reference_root_xy_total_displacement"]
    rollout_displacement = _total_displacement(rollout_root_xy)
    xy_error = rollout_root_xy - reference_root_xy
    reference_speed = _root_xy_speed(
        root_xy=reference_root_xy,
        qvel=reference_qvel_array,
        frequency=frequency,
    )
    rollout_speed = _root_xy_speed(
        root_xy=rollout_root_xy,
        qvel=rollout_qvel_array,
        frequency=frequency,
    )

    metrics.update(
        {
            "rollout_root_xy_total_displacement": rollout_displacement,
            "root_displacement_ratio": _safe_ratio(rollout_displacement, reference_displacement),
            "root_xy_rmse": _rmse(np.linalg.norm(xy_error, axis=1)),
            "root_xy_final_error": float(np.linalg.norm(xy_error[-1])) if xy_error.size else 0.0,
            "root_speed_rmse": _paired_rmse(rollout_speed, reference_speed),
        }
    )
    return metrics


def _as_2d_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    return array


def _prefix_qvel(qvel, *, frame_count: int, name: str) -> np.ndarray | None:
    if qvel is None:
        return None

    qvel_array = _as_2d_array(qvel, name=name)
    if qvel_array.shape[0] < frame_count:
        raise ValueError(f"{name} must contain at least {frame_count} frames")
    return qvel_array[:frame_count]


def _root_xy(qpos: np.ndarray, *, name: str) -> np.ndarray:
    if qpos.shape[1] < 2:
        raise ValueError(f"{name} must contain root x/y columns")
    return qpos[:, :2]


def _root_xy_speed(*, root_xy: np.ndarray, qvel, frequency) -> np.ndarray:
    if qvel is not None:
        qvel_array = _as_2d_array(qvel, name="qvel")
        if qvel_array.shape[0] != root_xy.shape[0]:
            raise ValueError("qvel must have the same frame count as qpos")
        if qvel_array.shape[1] < 2:
            raise ValueError("qvel must contain root x/y velocity columns")
        return np.linalg.norm(qvel_array[:, :2], axis=1)

    if frequency is None:
        return np.zeros(root_xy.shape[0], dtype=np.float64)
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if root_xy.shape[0] < 2:
        return np.zeros(root_xy.shape[0], dtype=np.float64)

    frame_distances = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) * float(frequency)
    return np.concatenate(([0.0], frame_distances))


def _root_yaw(qpos: np.ndarray) -> np.ndarray | None:
    if qpos.shape[1] < 7:
        return None

    quat = qpos[:, 3:7]
    norm = np.linalg.norm(quat, axis=1)
    valid = norm > 0.0
    if not np.all(valid):
        quat = quat.copy()
        quat[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        norm = np.linalg.norm(quat, axis=1)
    quat = quat / norm[:, None]

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_change(yaw: np.ndarray | None) -> float:
    if yaw is None or yaw.size < 2:
        return 0.0
    delta = (yaw[-1] - yaw[0] + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(delta))


def _right_hand_world_path_length(*, site_xpos, right_hand_site_index) -> float:
    if site_xpos is None or right_hand_site_index is None:
        return 0.0

    site_xpos_array = np.asarray(site_xpos, dtype=np.float64)
    if site_xpos_array.ndim != 3:
        raise ValueError("site_xpos must be a 3D array of shape (frames, sites, xyz)")
    if site_xpos_array.shape[2] < 3:
        raise ValueError("site_xpos must contain xyz coordinates")

    site_index = _validate_site_index(
        right_hand_site_index,
        site_count=site_xpos_array.shape[1],
    )
    path = site_xpos_array[:, site_index, :3]
    return _path_length(path)


def _validate_site_index(index, *, site_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("right_hand_site_index must be an integer")
    if index < 0:
        raise ValueError("right_hand_site_index must be non-negative")
    if index >= site_count:
        raise IndexError(
            f"right_hand_site_index out of range: {index} for {site_count} sites"
        )
    return int(index)


def _max_abs_step(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0))))


def _max_abs_value(values) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _site_xpos_max_step(site_xpos) -> float:
    if site_xpos is None:
        return 0.0

    site_xpos_array = np.asarray(site_xpos, dtype=np.float64)
    if site_xpos_array.ndim != 3:
        raise ValueError("site_xpos must be a 3D array of shape (frames, sites, xyz)")
    if site_xpos_array.shape[2] < 3:
        raise ValueError("site_xpos must contain xyz coordinates")
    if site_xpos_array.shape[0] < 2:
        return 0.0

    frame_steps = np.linalg.norm(np.diff(site_xpos_array[:, :, :3], axis=0), axis=2)
    return float(np.max(frame_steps))


def _total_displacement(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _paired_rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("speed arrays must have the same frame count")
    return _rmse(lhs - rhs)


def _safe_ratio(numerator: float, denominator: float) -> float:
    eps = 1e-8
    if denominator < eps:
        if numerator < eps:
            return 1.0
        return float("inf")
    return float(numerator / denominator)
