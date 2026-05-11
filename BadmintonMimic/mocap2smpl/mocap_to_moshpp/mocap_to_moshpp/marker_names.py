from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .utils import ensure_parent, read_marker_npz, write_marker_npz


KNOWN_ANATOMY = {
    "右肩峰": "R_ACR",
    "左肩峰": "L_ACR",
    "右髂前上棘": "R_ASIS",
    "左髂前上棘": "L_ASIS",
    "右髂后上棘": "R_PSIS",
    "左髂后上棘": "L_PSIS",
    "胸骨柄": "STRN",
    "胸骨": "STRN",
    "第七颈椎": "C7",
    "颈椎7": "C7",
    "右膝": "R_KNE",
    "左膝": "L_KNE",
    "右踝": "R_ANK",
    "左踝": "L_ANK",
    "右腕": "R_WRA",
    "左腕": "L_WRA",
}


def _heuristic_chinese_label(name: str) -> str | None:
    side = "R" if "右" in name else "L" if "左" in name else ""
    upper = name.upper()
    if "C7" in upper or "脊椎上" in name:
        return "C7"
    if "肩峰" in name and side:
        return f"{side}_ACR"
    if ("髂前上棘" in name or "ASIS" in upper) and side:
        return f"{side}_ASIS"
    if ("髂后上棘" in name or "PSIS" in upper or "髂嵴" in name) and side:
        return f"{side}_PSIS"
    if ("胸" in name or "两胸" in name) and not side:
        return "STRN"
    if "膝" in name and side:
        return f"{side}_KNE"
    if "脚后跟" in name and side:
        return f"{side}_HEE"
    if ("脚最前端" in name or "大拇指" in name) and side:
        return f"{side}_TOE"
    if ("脚背" in name or "小腿骨连接" in name) and side:
        return f"{side}_ANK"
    if "手腕" in name and side:
        return f"{side}_WRA"
    if "手肘" in name and side:
        return f"{side}_ELB"
    if "肩胛上" in name and side:
        return f"{side}_SCAP_SUP"
    if "肩胛下" in name and side:
        return f"{side}_SCAP_INF"
    if "头顶" in name:
        return "HEAD_TOP"
    if "额头" in name:
        return "HEAD_FRONT"
    if "肚子" in name:
        return "BELL"
    return None


def safe_label(raw_name: str, marker_id: str | None = None, max_len: int = 48) -> str:
    name = str(raw_name or "").strip()
    if name in KNOWN_ANATOMY:
        base = KNOWN_ANATOMY[name]
    else:
        base = _heuristic_chinese_label(name) or ""
        normalized = unicodedata.normalize("NFKD", name)
        ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
        if not base:
            base = re.sub(r"[^A-Za-z0-9_]+", "_", ascii_name).strip("_")
        if not base:
            mid = re.sub(r"[^A-Za-z0-9_]+", "_", str(marker_id or "")).strip("_")
            base = f"M_{mid}" if mid else "M"
    if re.match(r"^\d", base):
        base = f"M_{base}"
    base = re.sub(r"_+", "_", base)[:max_len].strip("_")
    return base or "M"


def make_unique(labels: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    out: list[str] = []
    for label in labels:
        counts[label] += 1
        out.append(label if counts[label] == 1 else f"{label}_{counts[label]:02d}")
    return out


def normalize_names(marker_names: list[str], marker_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    bases = []
    records = marker_records or [{} for _ in marker_names]
    for name, rec in zip(marker_names, records):
        bases.append(safe_label(name, str(rec.get("id", ""))))
    safe = make_unique(bases)
    entries = []
    for i, (orig, safe_name, rec) in enumerate(zip(marker_names, safe, records)):
        entries.append(
            {
                "index": i,
                "original_name": orig,
                "original_id": str(rec.get("id", "")),
                "type": str(rec.get("type", "")),
                "safe_label": safe_name,
            }
        )
    return {
        "original_to_safe": {e["original_name"]: e["safe_label"] for e in entries},
        "safe_to_original": {e["safe_label"]: e["original_name"] for e in entries},
        "markers": entries,
    }


def write_name_map(path: str | Path, mapping: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def normalize_marker_npz(markers_npz: str | Path, out_map: str | Path, out_npz: str | Path) -> dict[str, Any]:
    data = read_marker_npz(markers_npz)
    meta = data.get("meta", {})
    records = meta.get("marker_records", [])
    mapping = normalize_names(data["marker_names"], records)
    write_name_map(out_map, mapping)
    meta["marker_name_map"] = mapping
    write_marker_npz(
        out_npz,
        markers=data["markers"].astype("float32"),
        mask=data["mask"].astype(bool),
        marker_names=[m["safe_label"] for m in mapping["markers"]],
        fps=float(data["fps"]),
        frame_ids=data["frame_ids"],
        time=data["time"],
        units=str(data["units"]),
        coordinate_space=str(data["coordinate_space"]),
        meta_json=__import__("json").dumps(meta, ensure_ascii=False),
    )
    return mapping
