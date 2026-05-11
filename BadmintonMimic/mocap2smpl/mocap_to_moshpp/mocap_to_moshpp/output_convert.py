from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_parent, write_json


POSE_KEYS = ("poses", "pose", "fullpose", "full_pose", "theta")
TRANS_KEYS = ("trans", "transl", "translation")
BETAS_KEYS = ("betas", "shape", "shapes")


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def _load_any(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=True)
            return {k: z[k] for k in z.files}
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                obj = pickle.load(f, encoding="latin1")
            return obj if isinstance(obj, dict) else {"object": obj}
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lower = {k.lower(): k for k in data.keys()}
    for key in keys:
        if key in data:
            return data[key]
        if key.lower() in lower:
            return data[lower[key.lower()]]
    return None


def convert_moshpp_output(moshpp_out: str | Path, out_npz: str | Path) -> dict[str, Any]:
    root = Path(moshpp_out)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".pkl", ".npz", ".yaml", ".yml", ".json"}])
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        if path.suffix.lower() in {".yaml", ".yml"}:
            continue
        data = _load_any(path)
        if data is not None:
            loaded.append((path, data))

    selected: tuple[Path, dict[str, Any]] | None = None
    for item in loaded:
        _, data = item
        if _first(data, POSE_KEYS) is not None or _first(data, TRANS_KEYS) is not None:
            selected = item
            break
    if selected is None:
        keys = {str(path): list(data.keys()) for path, data in loaded}
        write_json(ensure_parent(out_npz).with_suffix(".debug_keys.json"), {"files": [str(f) for f in files], "keys": keys})
        raise RuntimeError("Could not find MoSh++ pose output. Debug keys were written next to the requested output.")

    source, data = selected
    poses = _first(data, POSE_KEYS)
    trans = _first(data, TRANS_KEYS)
    betas = _first(data, BETAS_KEYS)
    if poses is None:
        if trans is None:
            raise RuntimeError(f"Selected {source} but found neither poses nor trans.")
        poses = np.zeros((len(_as_array(trans)), 72), dtype=np.float32)
    poses = _as_array(poses).astype(np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    t = poses.shape[0]
    trans_arr = _as_array(trans).astype(np.float32) if trans is not None else np.zeros((t, 3), dtype=np.float32)
    if trans_arr.ndim == 1:
        trans_arr = np.tile(trans_arr[None, :], (t, 1))
    betas_arr = _as_array(betas).astype(np.float32).reshape(-1) if betas is not None else np.zeros(10, dtype=np.float32)

    out_npz = ensure_parent(out_npz)
    payload = {
        "poses": poses,
        "trans": trans_arr,
        "betas": betas_arr,
        "gender": str(data.get("gender", "unknown")),
        "mocap_framerate": float(data.get("mocap_framerate", data.get("fps", 120.0))),
        "model_type": str(data.get("model_type", "unknown")),
        "marker_labels": np.asarray(data.get("marker_labels", []), dtype=object),
        "source_c3d": str(data.get("source_c3d", "")),
        "source_csv": str(data.get("source_csv", "")),
    }
    if "dmpls" in data:
        payload["dmpls"] = _as_array(data["dmpls"]).astype(np.float32)
    np.savez_compressed(out_npz, **payload)
    report = {"out_npz": str(out_npz), "source": str(source), "keys": list(data.keys()), "frames": int(t)}
    write_json(out_npz.with_suffix(".convert_report.json"), report)
    return report
