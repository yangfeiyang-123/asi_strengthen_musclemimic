from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_parent, import_ezc3d, read_marker_npz, write_json


def write_c3d_from_npz(in_npz: str | Path, out_c3d: str | Path, units: str = "mm") -> dict[str, Any]:
    ezc3d = import_ezc3d()
    data = read_marker_npz(in_npz)
    markers = data["markers"].astype(float)
    mask = np.isfinite(markers).all(axis=2)
    t, n, _ = markers.shape
    points = np.zeros((4, n, t), dtype=float)
    points[:3, :, :] = np.nan_to_num(np.transpose(markers, (2, 1, 0)), nan=0.0)
    points[3, :, :] = np.transpose(np.where(mask, 1.0, -1.0), (1, 0))

    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["USED"]["value"] = [n]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [float(data["fps"])]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = [str(x) for x in data["marker_names"]]
    c3d["parameters"]["POINT"]["UNITS"]["value"] = [units]
    c3d["data"]["points"] = points
    c3d["data"]["meta_points"]["residuals"] = np.where(mask.T[None, :, :], 0.0, -1.0)
    c3d["data"]["meta_points"]["camera_masks"] = np.zeros((7, n, t), dtype=bool)
    c3d["data"]["analogs"] = np.zeros((1, 0, t))
    out_c3d = ensure_parent(out_c3d)
    c3d.write(str(out_c3d))

    sidecar = out_c3d.with_suffix(".missing_mask.npz")
    np.savez_compressed(sidecar, mask=mask, marker_names=np.asarray(data["marker_names"], dtype=object))

    read_back = ezc3d.c3d(str(out_c3d))
    rb_points = read_back["data"]["points"]
    rb_labels = [str(x) for x in read_back["parameters"]["POINT"]["LABELS"]["value"]]
    report = {
        "out_c3d": str(out_c3d),
        "missing_mask": str(sidecar),
        "frames": int(t),
        "markers": int(n),
        "fps": float(data["fps"]),
        "units": units,
        "readback_shape": list(rb_points.shape),
        "readback_labels_match": rb_labels == data["marker_names"],
        "missing_rate": float(1.0 - mask.mean()) if mask.size else 0.0,
    }
    write_json(out_c3d.with_suffix(".write_report.json"), report)
    return report
