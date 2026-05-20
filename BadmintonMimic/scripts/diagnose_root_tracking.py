#!/usr/bin/env python3
"""Diagnose reference root-tracking metrics from retarget cache files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_TRACKING = REPO_ROOT / "musclemimic" / "utils" / "root_tracking.py"
EXPECTED_USER_ERRORS = (FileNotFoundError, KeyError, ValueError, IndexError)


def _load_compute_root_reference_metrics():
    spec = importlib.util.spec_from_file_location(
        "_diagnose_root_tracking_root_tracking",
        ROOT_TRACKING,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load root tracking module: {ROOT_TRACKING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_root_reference_metrics


compute_root_reference_metrics = _load_compute_root_reference_metrics()


def _resolve_cache_file(cache_root: Path, motion: str) -> Path:
    motion_path = Path(motion)
    if motion_path.is_absolute():
        raise ValueError(f"motion path must be relative to cache root: {motion}")
    if motion_path.suffix == "":
        motion_path = motion_path.with_suffix(".npz")

    cache_file = cache_root / motion_path
    if not cache_file.exists():
        raise FileNotFoundError(f"cache file does not exist: {cache_file}")
    if not cache_file.is_file():
        raise FileNotFoundError(f"cache path is not a file: {cache_file}")
    return cache_file


def _float_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _diagnose_cache_file(cache_file: Path, right_hand_site_index: int | None) -> dict[str, Any]:
    with np.load(cache_file) as data:
        if "qpos" not in data:
            raise KeyError(f"cache file is missing required qpos array: {cache_file}")

        qpos = data["qpos"]
        qvel = data["qvel"] if "qvel" in data else None
        site_xpos = data["site_xpos"] if "site_xpos" in data else None
        frequency = _float_scalar(data["frequency"]) if "frequency" in data else 0.0

        metrics = compute_root_reference_metrics(
            qpos=qpos,
            qvel=qvel,
            site_xpos=site_xpos,
            right_hand_site_index=right_hand_site_index,
            frequency=frequency if "frequency" in data else None,
        )

        row: dict[str, Any] = {
            "motion": cache_file.stem,
            "path": str(cache_file),
            "frames": int(qpos.shape[0]),
            "frequency": frequency,
        }
        row.update(metrics)
        return row


def _write_json_report(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose reference root-tracking metrics from retarget cache files.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("caches/AMASS/MyoFullBody/gmr"),
        help="Root directory containing retarget cache .npz files.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        required=True,
        help="Motion path under cache root. The .npz suffix is optional.",
    )
    parser.add_argument(
        "--right-hand-site-index",
        type=int,
        default=8,
        help="Site index used for right-hand world path length when site_xpos is present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path.",
    )
    return parser


def _format_summary(row: dict[str, Any]) -> str:
    return (
        f"{row['motion']}: "
        f"frames={row['frames']} "
        f"frequency={row['frequency']:.3f} "
        f"root_xy_displacement={row['reference_root_xy_total_displacement']:.3f}m "
        f"root_xy_path={row['reference_root_xy_path_length']:.3f}m "
        f"root_xy_peak_speed={row['reference_root_xy_peak_speed']:.3f}m/s "
        f"yaw_change={row['reference_root_yaw_change']:.3f}rad "
        f"right_hand_path={row['right_hand_world_path_length']:.3f}m"
    )


def _error_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and len(exc.args) == 1:
        return str(exc.args[0])
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        rows = []
        for motion in args.motion:
            cache_file = _resolve_cache_file(args.cache_root, motion)
            row = _diagnose_cache_file(
                cache_file,
                right_hand_site_index=args.right_hand_site_index,
            )
            rows.append(row)
            print(_format_summary(row))

        if args.output is not None:
            _write_json_report(args.output, rows)
            print(f"wrote JSON report: {args.output}")
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
