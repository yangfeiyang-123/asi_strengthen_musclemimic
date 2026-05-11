from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: str | Path, data: Any) -> Path:
    path = ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return path


def read_marker_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {k: data[k] for k in data.files}
    if "marker_names" in out:
        out["marker_names"] = [str(x) for x in out["marker_names"].tolist()]
    if "meta_json" in out:
        out["meta"] = json.loads(str(out["meta_json"]))
    return out


def write_marker_npz(path: str | Path, **kwargs: Any) -> Path:
    path = ensure_parent(path)
    if "marker_names" in kwargs:
        kwargs["marker_names"] = np.asarray(kwargs["marker_names"], dtype=object)
    np.savez_compressed(path, **kwargs)
    return path


def finite_range(values: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(values)
    if not np.any(valid):
        return {"min": None, "max": None, "mean": None}
    vv = values[valid]
    return {"min": float(np.min(vv)), "max": float(np.max(vv)), "mean": float(np.mean(vv))}


def import_ezc3d():
    try:
        import ezc3d  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "ezc3d is required for C3D read/write. Install it in this environment, "
            "for example: pip install ezc3d"
        ) from exc
    return ezc3d
