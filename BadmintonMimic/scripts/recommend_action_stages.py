#!/usr/bin/env python3
"""Recommend base/posttrain/repair/exclude stages for badminton motion manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_TRACKING = REPO_ROOT / "musclemimic" / "utils" / "root_tracking.py"
ACTION_STAGE = REPO_ROOT / "musclemimic" / "utils" / "action_stage.py"
EXPECTED_USER_ERRORS = (FileNotFoundError, KeyError, ValueError, IndexError)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_root_tracking = _load_module(ROOT_TRACKING, "_recommend_action_stages_root_tracking")
_action_stage = _load_module(ACTION_STAGE, "_recommend_action_stages_action_stage")

compute_root_reference_metrics = _root_tracking.compute_root_reference_metrics
MotionHints = _action_stage.MotionHints
classify_motion_stage = _action_stage.classify_motion_stage


class HintTable:
    def __init__(
        self,
        defaults: dict[str, dict[str, Any]] | None = None,
        motions: dict[str, dict[str, Any]] | None = None,
    ):
        self._defaults = defaults or {}
        self._motions = motions or {}

    def for_motion(self, motion: str):
        action_label = motion.split("/", 1)[0]
        values: dict[str, Any] = {"action_label": action_label}
        values.update(self._defaults.get(action_label, {}))
        values.update(self._motions.get(motion, {}))
        return MotionHints(**values)


def _read_manifest(path: Path) -> list[str]:
    motions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        motions.append(line.removesuffix(".npz"))
    return motions


def _load_hints(path: Path | None) -> HintTable:
    if path is None:
        return HintTable()

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("hints YAML must contain a mapping")
    defaults = data.get("defaults", {})
    motions = data.get("motions", {})
    if not isinstance(defaults, dict):
        raise ValueError("hints defaults must contain a mapping")
    if not isinstance(motions, dict):
        raise ValueError("hints motions must contain a mapping")
    return HintTable(defaults=defaults, motions=motions)


def _resolve_cache_file(cache_root: Path, motion: str) -> Path:
    motion_path = Path(motion)
    if motion_path.is_absolute():
        raise ValueError(f"motion path must be relative to cache root: {motion}")
    if motion_path.suffix == "":
        motion_path = motion_path.with_suffix(".npz")

    resolved_cache_root = cache_root.resolve()
    cache_file = (cache_root / motion_path).resolve()
    if cache_file != resolved_cache_root and resolved_cache_root not in cache_file.parents:
        raise ValueError(f"motion path must stay under cache root: {motion}")
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


def _metrics_for_cache(cache_file: Path, right_hand_site_index: int | None) -> dict[str, float]:
    with np.load(cache_file) as data:
        if "qpos" not in data:
            raise KeyError(f"cache file is missing required qpos array: {cache_file}")

        qpos = data["qpos"]
        qvel = data["qvel"] if "qvel" in data else None
        site_xpos = data["site_xpos"] if "site_xpos" in data else None
        frequency = _float_scalar(data["frequency"]) if "frequency" in data else None

        return compute_root_reference_metrics(
            qpos=qpos,
            qvel=qvel,
            site_xpos=site_xpos,
            right_hand_site_index=right_hand_site_index,
            frequency=frequency,
        )


def _recommend_motion(
    cache_root: Path,
    motion: str,
    hints,
    right_hand_site_index: int | None,
) -> dict[str, Any]:
    cache_file = _resolve_cache_file(cache_root, motion)
    metrics = _metrics_for_cache(cache_file, right_hand_site_index=right_hand_site_index)
    decision = classify_motion_stage(metrics, hints)
    return {
        "motion": motion,
        "cache_file": str(cache_file),
        "stage": decision.stage,
        "family": decision.family,
        "reasons": list(decision.reasons),
        "hints": asdict(hints),
        "metrics": metrics,
    }


def _write_json_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("caches/AMASS/MyoFullBody/gmr"),
        help="Root directory containing retarget cache .npz files.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Motion manifest. May be passed more than once.",
    )
    parser.add_argument(
        "--hints",
        type=Path,
        help="Optional YAML file containing default and per-motion action-stage hints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON recommendation report path.",
    )
    parser.add_argument(
        "--right-hand-site-index",
        type=int,
        default=8,
        help="Site index used for right-hand world path length when site_xpos is present.",
    )
    return parser


def _format_summary(row: dict[str, Any]) -> str:
    reasons = ",".join(row["reasons"])
    return (
        f"{row['motion']}: "
        f"stage={row['stage']} "
        f"family={row['family']} "
        f"reasons={reasons}"
    )


def _error_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and len(exc.args) == 1:
        return str(exc.args[0])
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        hint_table = _load_hints(args.hints)
        rows: list[dict[str, Any]] = []
        for manifest in args.manifest:
            for motion in _read_manifest(manifest):
                row = _recommend_motion(
                    args.cache_root,
                    motion,
                    hint_table.for_motion(motion),
                    right_hand_site_index=args.right_hand_site_index,
                )
                rows.append(row)
                print(_format_summary(row))

        _write_json_report(args.output, rows)
        print(f"wrote JSON report: {args.output}")
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
