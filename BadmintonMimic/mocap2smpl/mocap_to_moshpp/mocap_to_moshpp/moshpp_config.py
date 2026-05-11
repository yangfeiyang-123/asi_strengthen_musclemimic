from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .c3d_validator import read_c3d_points
from .utils import ensure_dir, write_json


def discover_moshpp_templates(moshpp_dir: str | Path) -> list[str]:
    root = Path(moshpp_dir)
    if not root.exists():
        return []
    patterns = ["*.yaml", "*.yml", "*.json", "*.cfg", "*.toml"]
    hits: list[str] = []
    for pat in patterns:
        for path in root.rglob(pat):
            rel = str(path.relative_to(root))
            lower = rel.lower()
            if any(k in lower for k in ("mosh", "fit", "subject", "config", "tutorial", "example")):
                hits.append(rel)
    return sorted(hits)[:100]


def prepare_moshpp_config(
    c3d: str | Path,
    marker_name_map: str | Path,
    moshpp_dir: str | Path,
    body_model_dir: str | Path,
    model_type: str,
    gender: str,
    out_dir: str | Path,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    c3d_data = read_c3d_points(c3d)
    labels = c3d_data["labels"]
    if len(labels) < 10:
        raise ValueError(f"Only {len(labels)} markers found. MoSh++ needs a fuller marker layout.")

    templates = discover_moshpp_templates(moshpp_dir)
    map_data: dict[str, Any] = {}
    map_path = Path(marker_name_map)
    if map_path.exists():
        map_data = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}

    marker_layout = {
        "description": "TODO: fill vertex_id/body_region/weight from the official MoSh++ marker layout for this marker set.",
        "source_c3d": str(c3d),
        "markers": [
            {"label": label, "vertex_id": None, "body_region": None, "weight": 1.0, "note": "TODO correspondence"}
            for label in labels
        ],
    }
    (out / "marker_layout.yaml").write_text(
        yaml.safe_dump(marker_layout, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    subject = {
        "todo": "Review against the official MoSh++ example config in your checked-out repository.",
        "subject_id": Path(c3d).stem,
        "gender": gender,
        "model_type": model_type,
        "body_model_dir": str(body_model_dir),
        "marker_layout": str(out / "marker_layout.yaml"),
    }
    fit_config = {
        "todo": "This is a conservative template. Replace field names if your MoSh++ checkout uses different names.",
        "mocap_file": str(c3d),
        "subject_config": str(out / "subject.yaml"),
        "marker_layout": str(out / "marker_layout.yaml"),
        "output_dir": str(out.parent / "moshpp_run"),
        "model_type": model_type,
        "gender": gender,
        "body_model_dir": str(body_model_dir),
    }
    if templates:
        subject["official_template_candidates"] = templates[:20]
        fit_config["official_template_candidates"] = templates[:20]
    else:
        fit_config["warning"] = "No official MoSh++ config templates were found; fill this file from MoSh++ examples."

    (out / "subject.yaml").write_text(yaml.safe_dump(subject, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "fit_config.yaml").write_text(yaml.safe_dump(fit_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    run_info = {
        "c3d": str(c3d),
        "marker_name_map": str(marker_name_map),
        "marker_count": len(labels),
        "fps": c3d_data["fps"],
        "model_type": model_type,
        "gender": gender,
        "body_model_dir": str(body_model_dir),
        "moshpp_dir": str(moshpp_dir),
        "official_template_candidates": templates,
        "warnings": [
            "All marker vertex_id fields are placeholders until marker-to-SMPL surface correspondences are filled.",
            "Do not expect stable MoSh++ fitting before marker_layout.yaml is completed.",
        ],
        "name_map_loaded": bool(map_data),
    }
    write_json(out / "run_info.json", run_info)
    return run_info
