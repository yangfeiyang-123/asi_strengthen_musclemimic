from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import finite_range, import_ezc3d, read_marker_npz, write_json


def _point_param(c3d: Any, key: str, default: Any = None) -> Any:
    try:
        return c3d["parameters"]["POINT"][key]["value"]
    except Exception:
        return default


def read_c3d_points(c3d_path: str | Path) -> dict[str, Any]:
    ezc3d = import_ezc3d()
    c3d = ezc3d.c3d(str(c3d_path))
    points = c3d["data"]["points"]
    labels = [str(x) for x in _point_param(c3d, "LABELS", [])]
    rate = float(_point_param(c3d, "RATE", [0])[0])
    units_value = _point_param(c3d, "UNITS", [""])
    units = str(units_value[0] if isinstance(units_value, list) else units_value)
    xyz = np.transpose(points[:3, :, :], (2, 1, 0))
    residuals = c3d["data"].get("meta_points", {}).get("residuals")
    if residuals is not None and getattr(residuals, "size", 0):
        residual = np.transpose(residuals[0, :, :], (1, 0))
    else:
        residual = np.transpose(points[3, :, :], (1, 0))
    mask = residual >= 0
    xyz = xyz.astype(np.float32)
    xyz[~mask] = np.nan
    return {"markers": xyz, "mask": mask, "labels": labels, "fps": rate, "units": units, "raw_points": points}


def validate_c3d(c3d_path: str | Path, ref_npz: str | Path | None, out_json: str | Path) -> dict[str, Any]:
    c3d_data = read_c3d_points(c3d_path)
    markers = c3d_data["markers"]
    report: dict[str, Any] = {
        "c3d": str(c3d_path),
        "frame_rate": c3d_data["fps"],
        "frame_count": int(markers.shape[0]),
        "marker_count": int(markers.shape[1]),
        "units": c3d_data["units"],
        "first_20_labels": c3d_data["labels"][:20],
        "coordinate_range": finite_range(markers),
    }
    if ref_npz:
        ref = read_marker_npz(ref_npz)
        ref_markers = ref["markers"].astype(np.float32)
        ref_mask = np.isfinite(ref_markers).all(axis=2)
        c3d_mask = np.isfinite(markers).all(axis=2)
        common = ref_mask & c3d_mask
        diffs = np.abs(ref_markers - markers)
        report.update(
            {
                "ref_npz": str(ref_npz),
                "shape_match": list(ref_markers.shape) == list(markers.shape),
                "labels_match": list(ref["marker_names"]) == c3d_data["labels"],
                "mask_match": bool(np.array_equal(ref_mask, c3d_mask)),
                "max_abs_error": float(np.nanmax(diffs[common])) if np.any(common) else None,
                "mean_abs_error": float(np.nanmean(diffs[common])) if np.any(common) else None,
            }
        )
    write_json(out_json, report)
    return report
