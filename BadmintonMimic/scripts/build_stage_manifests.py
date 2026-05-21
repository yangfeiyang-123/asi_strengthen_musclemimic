#!/usr/bin/env python3
"""Build stage-specific motion manifests from action-stage recommendations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("recommendations JSON must contain a list")
    return data


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


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        motion = str(row["motion"])
        stage = str(row["stage"])
        family = str(row.get("family", "general"))
        grouped[_manifest_name(stage, family)].append(motion)
    return dict(grouped)


def _write_manifests(output_dir: Path, grouped: dict[str, list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    rows = _load_rows(args.recommendations)
    grouped = _group_rows(rows)
    _write_manifests(args.output_dir, grouped)
    for name, motions in sorted(grouped.items()):
        print(f"{name}: {len(set(motions))} motions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
