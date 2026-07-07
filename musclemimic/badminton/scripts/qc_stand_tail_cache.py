#!/usr/bin/env python3
"""QC stand-tail MyoFullBody caches before post-training."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REQUIRED_SITES = (
    "pelvis_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
)


@dataclass
class StandTailQcRow:
    motion: str
    frames: int
    frequency: float
    original_frames: int
    settle_frames: int
    hold_frames: int
    hold_seconds: float
    tail_qvel_max: float
    tail_qpos_step_max: float
    settle_qvel_max: float
    root_height_min_hold: float
    root_height_max_hold: float
    com_support_margin_max_m: float
    com_support_margin_mean_m: float
    support_area_xy_m2: float
    foot_span_xy_m: float
    passed: bool
    failures: str


def _load_manifest(path: Path) -> list[str]:
    motions: list[str] = []
    for line in path.read_text().splitlines():
        motion = line.strip()
        if not motion or motion.startswith("#"):
            continue
        motions.append(motion.removesuffix(".npz"))
    if not motions:
        raise ValueError(f"empty manifest: {path}")
    return motions


def _metadata_dict(value: np.ndarray) -> dict:
    item = value.item()
    if item is None:
        return {}
    if not isinstance(item, dict):
        raise ValueError(f"metadata must be dict or None, got {type(item)!r}")
    return item


def _site_index(site_names: np.ndarray, site_name: str) -> int:
    matches = np.where(site_names == site_name)[0]
    if matches.size == 0:
        raise ValueError(f"missing required site {site_name!r}")
    return int(matches[0])


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)
    lower: list[tuple[float, float]] = []
    for point in pts:
        p = np.asarray(point)
        while len(lower) >= 2 and _cross(np.asarray(lower[-2]), np.asarray(lower[-1]), p) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        p = np.asarray(point)
        while len(upper) >= 2 and _cross(np.asarray(upper[-2]), np.asarray(upper[-1]), p) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_in_convex_polygon(point: np.ndarray, poly: np.ndarray, eps: float = 1e-9) -> bool:
    if len(poly) < 3:
        return False
    signs = []
    for idx in range(len(poly)):
        signs.append(_cross(poly[idx], poly[(idx + 1) % len(poly)], point))
    signs_arr = np.asarray(signs)
    return bool(np.all(signs_arr >= -eps) or np.all(signs_arr <= eps))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _support_margin(point_xy: np.ndarray, support_xy: np.ndarray) -> float:
    """Distance outside the foot support proxy; 0 means inside the proxy."""
    hull = _convex_hull(support_xy)
    if len(hull) == 0:
        return float("inf")
    if len(hull) == 1:
        return float(np.linalg.norm(point_xy - hull[0]))
    if len(hull) >= 3 and _point_in_convex_polygon(point_xy, hull):
        return 0.0
    distances = [
        _point_segment_distance(point_xy, hull[idx], hull[(idx + 1) % len(hull)])
        for idx in range(len(hull))
    ]
    return float(min(distances))


def _support_area_and_span(support_xy: np.ndarray) -> tuple[float, float]:
    hull = _convex_hull(support_xy)
    area = _polygon_area(hull)
    span = 0.0
    for i in range(len(support_xy)):
        for j in range(i + 1, len(support_xy)):
            span = max(span, float(np.linalg.norm(support_xy[i] - support_xy[j])))
    return area, span


def qc_cache(
    cache_path: Path,
    motion: str,
    *,
    max_tail_qvel: float,
    max_tail_qpos_step: float,
    min_root_height: float,
    max_com_support_margin: float,
) -> StandTailQcRow:
    with np.load(cache_path, allow_pickle=True) as data:
        metadata = _metadata_dict(data["metadata"]) if "metadata" in data else {}
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        site_xpos = np.asarray(data["site_xpos"], dtype=np.float64)
        site_names = np.asarray(data["site_names"]).astype(str)
        subtree_com = np.asarray(data["subtree_com"], dtype=np.float64)
        frequency = float(np.asarray(data["frequency"]).reshape(-1)[0])

        original_frames = int(metadata.get("original_frames", 0))
        settle_frames = int(metadata.get("settle_frames", 0))
        hold_frames = int(metadata.get("hold_frames", 0))
        if original_frames <= 0 or hold_frames <= 0:
            raise ValueError(f"{cache_path} is missing stand-tail metadata")
        if qpos.shape[0] != qvel.shape[0] or qpos.shape[0] != site_xpos.shape[0]:
            raise ValueError(f"{cache_path} has inconsistent frame counts")

        hold_start = qpos.shape[0] - hold_frames
        settle_start = max(original_frames, hold_start - settle_frames)
        hold_slice = slice(hold_start, qpos.shape[0])
        settle_slice = slice(settle_start, hold_start)

        tail_qvel_max = float(np.max(np.abs(qvel[hold_slice]))) if hold_frames else float("inf")
        tail_qpos_step_max = (
            float(np.max(np.abs(np.diff(qpos[hold_slice], axis=0)))) if hold_frames > 1 else 0.0
        )
        settle_qvel_max = float(np.max(np.abs(qvel[settle_slice]))) if settle_frames > 0 else 0.0

        root_height = qpos[hold_slice, 2]
        root_height_min = float(np.min(root_height))
        root_height_max = float(np.max(root_height))

        indices = {name: _site_index(site_names, name) for name in REQUIRED_SITES}
        support_ids = [
            indices["left_ankle_mimic"],
            indices["left_toes_mimic"],
            indices["right_ankle_mimic"],
            indices["right_toes_mimic"],
        ]
        support_xy = np.mean(site_xpos[hold_slice, support_ids, :2], axis=0)
        support_area, foot_span = _support_area_and_span(support_xy)
        com_xy = subtree_com[hold_slice, 1, :2]
        support_margins = np.asarray([_support_margin(point, support_xy) for point in com_xy])
        support_margin_max = float(np.max(support_margins))
        support_margin_mean = float(np.mean(support_margins))

    failures: list[str] = []
    if tail_qvel_max > max_tail_qvel:
        failures.append(f"tail_qvel_max>{max_tail_qvel:g}")
    if tail_qpos_step_max > max_tail_qpos_step:
        failures.append(f"tail_qpos_step_max>{max_tail_qpos_step:g}")
    if root_height_min < min_root_height:
        failures.append(f"root_height_min<{min_root_height:g}")
    if support_margin_max > max_com_support_margin:
        failures.append(f"com_support_margin_max>{max_com_support_margin:g}")
    if foot_span < 0.05:
        failures.append("foot_span_too_small")

    return StandTailQcRow(
        motion=motion,
        frames=int(qpos.shape[0]),
        frequency=frequency,
        original_frames=original_frames,
        settle_frames=settle_frames,
        hold_frames=hold_frames,
        hold_seconds=float(hold_frames / frequency),
        tail_qvel_max=tail_qvel_max,
        tail_qpos_step_max=tail_qpos_step_max,
        settle_qvel_max=settle_qvel_max,
        root_height_min_hold=root_height_min,
        root_height_max_hold=root_height_max,
        com_support_margin_max_m=support_margin_max,
        com_support_margin_mean_m=support_margin_mean,
        support_area_xy_m2=support_area,
        foot_span_xy_m=foot_span,
        passed=not failures,
        failures=";".join(failures),
    )


def _write_csv(path: Path, rows: list[StandTailQcRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summary(rows: list[StandTailQcRow]) -> dict:
    failed = [row for row in rows if not row.passed]
    return {
        "count": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "max_tail_qvel": max(row.tail_qvel_max for row in rows),
        "max_tail_qpos_step": max(row.tail_qpos_step_max for row in rows),
        "max_settle_qvel": max(row.settle_qvel_max for row in rows),
        "min_root_height_hold": min(row.root_height_min_hold for row in rows),
        "max_com_support_margin_m": max(row.com_support_margin_max_m for row in rows),
        "failed_motions": [{"motion": row.motion, "failures": row.failures} for row in failed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/10trajectories_smooth_27_stand_tail_list.txt"))
    parser.add_argument("--cache-root", type=Path, default=Path("caches/AMASS/MyoFullBody/gmr"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/qc/stand_tail_qc.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/qc/stand_tail_qc_summary.json"))
    parser.add_argument("--max-tail-qvel", type=float, default=1e-6)
    parser.add_argument("--max-tail-qpos-step", type=float, default=1e-6)
    parser.add_argument("--min-root-height", type=float, default=0.60)
    parser.add_argument(
        "--max-com-support-margin",
        type=float,
        default=0.20,
        help="Max allowed COM xy distance outside the ankle/toe support proxy.",
    )
    args = parser.parse_args()

    rows = [
        qc_cache(
            args.cache_root / f"{motion}.npz",
            motion,
            max_tail_qvel=args.max_tail_qvel,
            max_tail_qpos_step=args.max_tail_qpos_step,
            min_root_height=args.min_root_height,
            max_com_support_margin=args.max_com_support_margin,
        )
        for motion in _load_manifest(args.manifest)
    ]
    _write_csv(args.output_csv, rows)
    summary = _summary(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
