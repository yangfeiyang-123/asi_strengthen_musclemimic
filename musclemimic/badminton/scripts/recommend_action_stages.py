#!/usr/bin/env python3
"""Recommend base/posttrain/repair/exclude stages for badminton motion manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
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
VALID_HINT_KEYS = frozenset(MotionHints.__dataclass_fields__)
VALID_TOP_LEVEL_HINT_KEYS = frozenset({"defaults", "motions"})
BOOLEAN_HINT_KEYS = frozenset(
    {
        "expected_large_motion",
        "has_jump_or_lunge",
        "contact_unreliable",
        "endpoint_unreliable",
        "fine_hand_dominant",
    }
)


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

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse hints YAML: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ValueError("hints YAML must contain a mapping")
    for key in data:
        if key not in VALID_TOP_LEVEL_HINT_KEYS:
            raise ValueError(f"invalid top-level hints key {key!r}")
    defaults = data.get("defaults", {})
    motions = data.get("motions", {})
    return HintTable(
        defaults=_validate_hint_section("defaults", defaults),
        motions=_validate_hint_section("motions", motions),
    )


def _validate_hint_section(section_name: str, values: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise ValueError(f"hints {section_name} must contain a mapping")

    validated: dict[str, dict[str, Any]] = {}
    for source, hints in values.items():
        if not isinstance(hints, Mapping):
            raise ValueError(f"hints {section_name} entry for {source} must contain a mapping")
        for key in hints:
            if key not in VALID_HINT_KEYS:
                raise ValueError(f"invalid hint key {key!r} for {section_name} {source}")
            _validate_hint_value(section_name, source, key, hints[key])
        validated[str(source)] = dict(hints)
    return validated


def _validate_hint_value(section_name: str, source: Any, key: str, value: Any) -> None:
    if key in BOOLEAN_HINT_KEYS and type(value) is not bool:
        raise ValueError(f"invalid value for hint key {key!r} in {section_name} {source}: expected bool")
    if key == "action_label" and not isinstance(value, str):
        raise ValueError(f"invalid value for hint key {key!r} in {section_name} {source}: expected str")


def _resolve_cache_file(cache_root: Path, motion: str) -> Path:
    motion_path = Path(motion)
    if motion_path.is_absolute():
        raise ValueError(f"motion path must be relative to cache root: {motion}")
    if ".." in motion_path.parts:
        raise ValueError(f"motion path must not contain parent traversal: {motion}")
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
    hints: MotionHints,
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
        "confidence": decision.confidence,
        "failure_modes": list(decision.failure_modes),
        "review_required": decision.review_required,
        "required_action": decision.required_action,
        "hints": asdict(hints),
        "metrics": metrics,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = Counter(row["stage"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    required_action_counts = Counter(row["required_action"] for row in rows)
    failure_mode_counts = Counter(
        failure_mode
        for row in rows
        for failure_mode in row.get("failure_modes", [])
    )
    return {
        "total_motions": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "required_action_counts": dict(sorted(required_action_counts.items())),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "review_required_count": sum(1 for row in rows if row.get("review_required") is True),
    }


def _write_json_report(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr"),
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
        "--summary-output",
        type=Path,
        help="Optional JSON path for aggregate validation counts.",
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
    failures = ",".join(row["failure_modes"])
    return (
        f"{row['motion']}: "
        f"stage={row['stage']} "
        f"family={row['family']} "
        f"confidence={row['confidence']} "
        f"required_action={row['required_action']} "
        f"review_required={row['review_required']} "
        f"reasons={reasons} "
        f"failure_modes={failures}"
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
        if args.summary_output is not None:
            _write_json_report(args.summary_output, _summarize_rows(rows))
            print(f"wrote summary JSON report: {args.summary_output}")
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
