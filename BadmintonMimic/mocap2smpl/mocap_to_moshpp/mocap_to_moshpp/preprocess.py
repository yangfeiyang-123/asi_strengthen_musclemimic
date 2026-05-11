from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_marker_npz, write_json, write_marker_npz


def interpolate_short_gaps(markers: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, dict[str, int]]:
    out = markers.copy()
    stats = {"filled_values": 0, "filled_marker_frames": 0}
    t, n, _ = out.shape
    for mi in range(n):
        marker_filled_frames = set()
        for ci in range(3):
            y = out[:, mi, ci]
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue
            invalid_idx = np.where(~valid)[0]
            if invalid_idx.size == 0:
                continue
            start = 0
            while start < invalid_idx.size:
                end = start
                while end + 1 < invalid_idx.size and invalid_idx[end + 1] == invalid_idx[end] + 1:
                    end += 1
                gap = invalid_idx[start : end + 1]
                left = gap[0] - 1
                right = gap[-1] + 1
                if len(gap) <= max_gap and left >= 0 and right < t and valid[left] and valid[right]:
                    y[gap] = np.interp(gap, [left, right], [y[left], y[right]])
                    stats["filled_values"] += int(len(gap))
                    marker_filled_frames.update(int(x) for x in gap)
                start = end + 1
        stats["filled_marker_frames"] += len(marker_filled_frames)
    return out, stats


def detect_speed_outliers(markers: np.ndarray, fps: float, percentile: float = 99.5) -> tuple[np.ndarray, float, np.ndarray]:
    deltas = np.linalg.norm(np.diff(markers, axis=0), axis=2) * float(fps)
    valid = np.isfinite(deltas)
    if not np.any(valid):
        return np.zeros(markers.shape[:2], dtype=bool), float("nan"), deltas
    threshold = float(np.nanpercentile(deltas[valid], percentile))
    outlier_pairs = deltas > threshold
    outliers = np.zeros(markers.shape[:2], dtype=bool)
    outliers[1:, :] |= outlier_pairs
    return outliers, threshold, deltas


def moving_average(markers: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return markers
    pad = window // 2
    out = markers.copy()
    for mi in range(markers.shape[1]):
        for ci in range(3):
            y = markers[:, mi, ci]
            for ti in range(markers.shape[0]):
                lo = max(0, ti - pad)
                hi = min(markers.shape[0], ti + pad + 1)
                vals = y[lo:hi]
                if np.isfinite(vals).any():
                    out[ti, mi, ci] = np.nanmean(vals)
    return out


def preprocess_marker_npz(
    in_npz: str | Path,
    out_npz: str | Path,
    report_json: str | Path,
    max_gap: int = 10,
    outlier_speed_percentile: float = 99.5,
    outlier_action: str = "report",
    smooth: bool = False,
    smooth_window: int = 5,
) -> dict[str, Any]:
    data = read_marker_npz(in_npz)
    markers = data["markers"].astype(np.float32)
    mask_before = np.isfinite(markers).all(axis=2)
    clean, interp_stats = interpolate_short_gaps(markers, max_gap=max_gap)
    outliers, speed_threshold, speeds = detect_speed_outliers(clean, float(data["fps"]), outlier_speed_percentile)
    if outlier_action == "nan":
        clean[outliers] = np.nan
    if smooth:
        clean = moving_average(clean, smooth_window)
    mask_after = np.isfinite(clean).all(axis=2)
    meta = data.get("meta", {})
    meta["preprocess"] = {
        "max_gap": max_gap,
        "outlier_speed_percentile": outlier_speed_percentile,
        "outlier_action": outlier_action,
        "smooth": smooth,
        "smooth_window": smooth_window,
    }
    write_marker_npz(
        out_npz,
        markers=clean.astype(np.float32),
        mask=mask_after,
        marker_names=data["marker_names"],
        fps=float(data["fps"]),
        frame_ids=data["frame_ids"],
        time=data["time"],
        units=str(data["units"]),
        coordinate_space=str(data["coordinate_space"]),
        meta_json=json.dumps(meta, ensure_ascii=False),
    )
    per_marker_outliers = outliers.sum(axis=0).astype(int)
    report = {
        "input": str(in_npz),
        "output": str(out_npz),
        "frames": int(clean.shape[0]),
        "markers": int(clean.shape[1]),
        "missing_rate_before": float(1.0 - mask_before.mean()) if mask_before.size else 0.0,
        "missing_rate_after": float(1.0 - mask_after.mean()) if mask_after.size else 0.0,
        "interpolation": interp_stats,
        "speed_threshold": speed_threshold,
        "outlier_frames_total": int(outliers.sum()),
        "per_marker_outliers": {name: int(per_marker_outliers[i]) for i, name in enumerate(data["marker_names"])},
        "speed_units_per_second": f"{data['units']}/s",
    }
    write_json(report_json, report)
    return report
