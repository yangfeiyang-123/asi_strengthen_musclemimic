#!/usr/bin/env python3
"""Build stage-specific motion manifests from action-stage recommendations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_USER_ERRORS = (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError)
FAMILY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
GENERATED_MANIFEST_NAMES = ("base_general_list.txt", "repair_list.txt", "exclude_list.txt")
VALID_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
VALID_REQUIRED_ACTION_VALUES = frozenset({"train", "posttrain", "repair_first", "exclude", "manual_review"})


def _load_rows(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("recommendations JSON must contain a list")
    return data


def _require_non_empty_string(row: Mapping[str, Any], field: str, row_index: int) -> str:
    if field not in row:
        raise ValueError(f"row {row_index} missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str):
        raise ValueError(f"row {row_index} field {field!r} must be a non-empty string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"row {row_index} field {field!r} must be a non-empty string")
    if field == "motion" and stripped.splitlines() != [stripped]:
        raise ValueError(f"row {row_index} field 'motion' must not contain line separators")
    return stripped


def _validate_family_name(family: str, row_index: int) -> None:
    if "/" in family or "\\" in family or ".." in family or FAMILY_NAME_PATTERN.fullmatch(family) is None:
        raise ValueError(
            f"row {row_index} field 'family' must only contain letters, numbers, underscores, or hyphens"
        )


def _validate_optional_validation_fields(row: Mapping[str, Any], row_index: int) -> None:
    if "confidence" in row:
        confidence = row["confidence"]
        if confidence not in VALID_CONFIDENCE_VALUES:
            raise ValueError(f"row {row_index} field 'confidence' must be high, medium, or low")
    if "required_action" in row:
        required_action = row["required_action"]
        if required_action not in VALID_REQUIRED_ACTION_VALUES:
            raise ValueError(f"row {row_index} field 'required_action' is invalid")
    if "review_required" in row and type(row["review_required"]) is not bool:
        raise ValueError(f"row {row_index} field 'review_required' must be a boolean")
    if "failure_modes" in row:
        failure_modes = row["failure_modes"]
        if not isinstance(failure_modes, list) or not all(isinstance(item, str) for item in failure_modes):
            raise ValueError(f"row {row_index} field 'failure_modes' must be a list of strings")


def _manifest_name(stage: str, family: str) -> str:
    if stage == "base":
        return "base_general_list.txt"
    if stage == "posttrain":
        return f"posttrain_{family}_list.txt"
    if stage == "repair":
        return "repair_list.txt"
    if stage == "exclude":
        return "exclude_list.txt"
    raise ValueError(f"unsupported stage: {stage}")


def _group_rows(rows: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_index} must be a mapping")

        _validate_optional_validation_fields(row, row_index)
        motion = _require_non_empty_string(row, "motion", row_index)
        stage = _require_non_empty_string(row, "stage", row_index)
        family = "general"
        if stage == "posttrain":
            family = _require_non_empty_string(row, "family", row_index)
            _validate_family_name(family, row_index)
        grouped[_manifest_name(stage, family)].append(motion)
    return dict(grouped)


def _remove_stale_generated_manifests(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for name in GENERATED_MANIFEST_NAMES:
        stale_file = output_dir / name
        if stale_file.is_file():
            stale_file.unlink()
    for stale_file in output_dir.glob("posttrain_*_list.txt"):
        if stale_file.is_file():
            stale_file.unlink()


def _write_manifests(output_dir: Path, grouped: dict[str, list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_generated_manifests(output_dir)
    for name, motions in sorted(grouped.items()):
        unique_sorted = sorted(dict.fromkeys(motions))
        content = "".join(f"{motion}\n" for motion in unique_sorted)
        (output_dir / name).write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows = _load_rows(args.recommendations)
        grouped = _group_rows(rows)
        _write_manifests(args.output_dir, grouped)
        for name, motions in sorted(grouped.items()):
            print(f"{name}: {len(set(motions))} motions")
        return 0
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
