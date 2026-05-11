from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .marker_names import write_name_map
from .utils import ensure_parent, write_marker_npz


HEADER_KEYS = {
    "type": "Type",
    "name": "Name",
    "id": "ID",
}


def _parse_meta(row: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for i in range(0, len(row) - 1, 2):
        key = row[i].strip()
        if key:
            meta[key] = row[i + 1].strip()
    for key in ("FrameRate", "FrameCount"):
        if key in meta:
            try:
                meta[key] = int(meta[key]) if key == "FrameCount" else float(meta[key])
            except ValueError:
                pass
    return meta


def _to_float(value: str) -> float:
    try:
        if value is None:
            return float("nan")
        value = str(value).strip()
        if value == "" or value.lower() in {"nan", "none", "null"}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _find_header_rows(rows: list[list[str]]) -> tuple[dict[str, int], int]:
    found: dict[str, int] = {}
    xyz_row = None
    for i, row in enumerate(rows[:30]):
        normalized = [c.strip() for c in row[:10]]
        for key, label in HEADER_KEYS.items():
            if label in normalized:
                found[key] = i
        if "Frame" in normalized and "Time" in normalized and {"X", "Y", "Z"}.issubset(set(row)):
            xyz_row = i
    if xyz_row is None:
        raise ValueError("Could not find CSV component row containing Frame/Time and X/Y/Z.")
    if not all(k in found for k in ("type", "name", "id")):
        raise ValueError(f"Could not find required Type/Name/ID header rows. Found: {found}")
    return {**found, "field": xyz_row - 1, "component": xyz_row}, xyz_row + 1


def _cell(row: list[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) else ""


def _unique_marker_names(records: list[dict[str, str]]) -> list[str]:
    names = [r["name"] or f"M_{r['id']}" for r in records]
    name_counts = Counter(names)
    prelim = []
    for r, name in zip(records, names):
        if name_counts[name] > 1:
            prelim.append(f"{name}_{r['id']}" if r["id"] else name)
        else:
            prelim.append(name)
    counts: Counter[str] = Counter()
    out: list[str] = []
    for i, name in enumerate(prelim):
        counts[name] += 1
        out.append(name if counts[name] == 1 else f"{name}_{i:03d}")
    return out


def _valid_run(mask: np.ndarray) -> tuple[int | None, int | None, int]:
    valid = np.where(mask)[0]
    if valid.size == 0:
        return None, None, 0
    return int(valid[0]), int(valid[-1]), int(valid.size)


def _sort_start(mask: np.ndarray) -> int:
    start, _, _ = _valid_run(mask)
    return 10**9 if start is None else start


def _merge_duplicate_marker_tracks(
    markers: np.ndarray,
    records: list[dict[str, str]],
    overlap_distance_mm: float = 50.0,
) -> tuple[np.ndarray, list[dict[str, str]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec["type"], rec["name"], rec["field"])].append(i)

    merged_markers: list[np.ndarray] = []
    merged_records: list[dict[str, str]] = []
    merge_report: list[dict[str, Any]] = []
    consumed: set[int] = set()

    for indices in groups.values():
        valid_indices = [idx for idx in indices if _valid_run(np.isfinite(markers[:, idx, :]).all(axis=1))[2] > 0]
        if len(valid_indices) <= 1:
            continue
        first = records[valid_indices[0]]
        # Only same-name physical marker tracks are merged. Rigid-body and skeleton
        # outputs are derived data and should not be stitched as surface markers.
        if first["type"] != "Marker" or first["field"] != "Position" or not first["name"]:
            continue

        ordered = sorted(valid_indices, key=lambda idx: (_sort_start(np.isfinite(markers[:, idx, :]).all(axis=1)), idx))
        merged = markers[:, ordered[0], :].copy()
        merged_mask = np.isfinite(merged).all(axis=1)
        pieces = []
        overlap_distances: list[float] = []
        skipped: list[str] = []
        merged_indices = [ordered[0]]

        for idx in ordered:
            cur = markers[:, idx, :]
            cur_mask = np.isfinite(cur).all(axis=1)
            start, end, count = _valid_run(cur_mask)
            pieces.append({"id": records[idx]["id"], "start": start, "end": end, "valid_frames": count})
            if idx == ordered[0]:
                continue
            overlap = merged_mask & cur_mask
            if np.any(overlap):
                dists = np.linalg.norm(merged[overlap] - cur[overlap], axis=1)
                median_dist = float(np.nanmedian(dists))
                overlap_distances.append(median_dist)
                if median_dist > overlap_distance_mm:
                    skipped.append(records[idx]["id"])
                    continue
                merged[overlap] = 0.5 * (merged[overlap] + cur[overlap])
            fill = ~merged_mask & cur_mask
            merged[fill] = cur[fill]
            merged_mask = np.isfinite(merged).all(axis=1)
            merged_indices.append(idx)

        merged_record = dict(first)
        merged_record["id"] = "+".join(records[idx]["id"] for idx in merged_indices if records[idx]["id"])
        merged_markers.append(merged)
        merged_records.append(merged_record)
        consumed.update(merged_indices)
        merge_report.append(
            {
                "name": first["name"],
                "type": first["type"],
                "field": first["field"],
                "merged_ids": [records[idx]["id"] for idx in merged_indices],
                "pieces": pieces,
                "overlap_frames": int(sum(np.sum(np.isfinite(markers[:, a, :]).all(axis=1) & np.isfinite(markers[:, b, :]).all(axis=1)) for a in ordered for b in ordered if a < b)),
                "overlap_distance_mm_median": overlap_distances,
                "skipped_ids": skipped,
            }
        )

    out_markers: list[np.ndarray] = []
    out_records: list[dict[str, str]] = []
    merge_by_name = {(r["type"], r["name"], r["field"]): (m, r) for m, r in zip(merged_markers, merged_records)}
    emitted_names: set[tuple[str, str, str]] = set()
    for i, rec in enumerate(records):
        key = (rec["type"], rec["name"], rec["field"])
        if i in consumed:
            if key not in emitted_names:
                merged, merged_rec = merge_by_name[key]
                out_markers.append(merged)
                out_records.append(merged_rec)
                emitted_names.add(key)
            continue
        out_markers.append(markers[:, i, :])
        out_records.append(rec)

    if not merge_report:
        return markers, records, []
    return np.stack(out_markers, axis=1).astype(np.float32), out_records, merge_report


def parse_mocap_csv(
    csv_path: str | Path,
    out_npz: str | Path,
    name_map: str | Path | None = None,
    meters_npz: str | Path | None = None,
    allow_any_position: bool = True,
    min_valid_frames: int = 10,
    merge_duplicate_names: bool = True,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    meta = _parse_meta(rows[0])
    header_rows, data_start = _find_header_rows(rows)
    type_row = rows[header_rows["type"]]
    name_row = rows[header_rows["name"]]
    id_row = rows[header_rows["id"]]
    field_row = rows[header_rows["field"]]
    comp_row = rows[header_rows["component"]]

    groups: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
    group_meta: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for col in range(2, len(comp_row)):
        component = _cell(comp_row, col)
        if component not in {"X", "Y", "Z"}:
            continue
        field = _cell(field_row, col)
        if field != "Position":
            continue
        typ = _cell(type_row, col)
        if not allow_any_position and typ not in {"RigidBodyMarker", "Marker", "SkeletonMarker"}:
            continue
        name = _cell(name_row, col)
        marker_id = _cell(id_row, col)
        key = (typ, name, marker_id, field)
        groups[key][component] = col
        group_meta[key] = {"type": typ, "name": name, "id": marker_id, "field": field}

    complete_keys = [k for k, comps in groups.items() if {"X", "Y", "Z"}.issubset(comps)]
    complete_keys.sort(key=lambda k: min(groups[k].values()))
    records = [group_meta[k] for k in complete_keys]
    marker_names = _unique_marker_names(records)

    data_rows = [r for r in rows[data_start:] if len(r) >= 2 and _cell(r, 0) not in {"", "Frame"}]
    t = len(data_rows)
    n = len(complete_keys)
    markers = np.full((t, n, 3), np.nan, dtype=np.float32)
    frame_ids = np.zeros(t, dtype=np.int64)
    times = np.zeros(t, dtype=np.float64)

    for ti, row in enumerate(data_rows):
        frame_ids[ti] = int(_to_float(_cell(row, 0))) if np.isfinite(_to_float(_cell(row, 0))) else ti + 1
        times[ti] = _to_float(_cell(row, 1))
        for mi, key in enumerate(complete_keys):
            comps = groups[key]
            markers[ti, mi, 0] = _to_float(_cell(row, comps["X"]))
            markers[ti, mi, 1] = _to_float(_cell(row, comps["Y"]))
            markers[ti, mi, 2] = _to_float(_cell(row, comps["Z"]))

    duplicate_merge_report: list[dict[str, Any]] = []
    if merge_duplicate_names:
        markers, records, duplicate_merge_report = _merge_duplicate_marker_tracks(markers, records)
        marker_names = _unique_marker_names(records)

    mask = np.isfinite(markers).all(axis=2)
    valid_counts = mask.sum(axis=0)
    keep = valid_counts >= int(min_valid_frames)
    dropped_all_invalid = int((~keep).sum())
    if not np.all(keep):
        markers = markers[:, keep, :]
        mask = mask[:, keep]
        records = [rec for rec, ok in zip(records, keep) if ok]
        marker_names = _unique_marker_names(records)
    fps = float(meta.get("FrameRate", 0) or 0)
    if fps <= 0 and len(times) > 1:
        fps = 1.0 / float(np.nanmedian(np.diff(times)))

    meta_out = {
        **meta,
        "source_csv": str(csv_path),
        "header_rows": header_rows,
        "data_start_row": data_start,
        "marker_records": records,
        "raw_marker_names": marker_names,
        "duplicate_marker_merge": duplicate_merge_report,
        "merge_duplicate_names": bool(merge_duplicate_names),
        "dropped_markers_below_min_valid_frames": dropped_all_invalid,
        "min_valid_frames": int(min_valid_frames),
    }
    out_npz = write_marker_npz(
        out_npz,
        markers=markers,
        mask=mask,
        marker_names=marker_names,
        fps=fps,
        frame_ids=frame_ids,
        time=times,
        units=str(meta.get("LengthUnits", "mm")),
        coordinate_space=str(meta.get("CoordinateSpace", "")),
        meta_json=json.dumps(meta_out, ensure_ascii=False),
    )
    if meters_npz:
        scale = 0.001 if str(meta.get("LengthUnits", "mm")).lower() == "mm" else 1.0
        write_marker_npz(
            meters_npz,
            markers=(markers * scale).astype(np.float32),
            mask=mask,
            marker_names=marker_names,
            fps=fps,
            frame_ids=frame_ids,
            time=times,
            units="m",
            coordinate_space=str(meta.get("CoordinateSpace", "")),
            meta_json=json.dumps(meta_out, ensure_ascii=False),
        )
    if name_map:
        mapping = {
            "source_csv": str(csv_path),
            "markers": [
                {
                    "index": i,
                    "original_name": rec["name"],
                    "original_id": rec["id"],
                    "type": rec["type"],
                    "final_name": marker_names[i],
                }
                for i, rec in enumerate(records)
            ],
        }
        path = ensure_parent(name_map)
        path.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "out_npz": str(out_npz),
        "frames": int(markers.shape[0]),
        "markers": int(markers.shape[1]),
        "fps": fps,
        "missing_rate": float(1.0 - mask.mean()) if mask.size else 0.0,
    }
