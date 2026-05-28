#!/usr/bin/env python3
"""Run ForehandClear grip-hold diagnostics and training stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GripHoldPaths:
    spec_path: Path
    runner_type: str
    resume_from: Path
    scene_xml: Path
    grip_seed: Path
    output_dir: Path


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_grip_hold_spec(spec_path: str | Path) -> GripHoldPaths:
    resolved_spec = _resolve(spec_path)
    data = yaml.safe_load(resolved_spec.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_spec} must contain a mapping")
    if data.get("runner_type") != "forehand_clear_grip_hold":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")

    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    return GripHoldPaths(
        spec_path=resolved_spec,
        runner_type=str(data["runner_type"]),
        resume_from=_resolve(data["resume_from"]),
        scene_xml=_resolve(data["scene"]["xml"]),
        grip_seed=_resolve(data["grip_seed"]["path"]),
        output_dir=output_dir,
    )


def preflight(paths: GripHoldPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "runner_type": paths.runner_type,
        "spec_path": str(paths.spec_path),
        "resume_from": str(paths.resume_from),
        "scene_xml": str(paths.scene_xml),
        "grip_seed": str(paths.grip_seed),
        "output_dir": str(out_path),
        "checkpoint_exists": paths.resume_from.is_dir(),
        "scene_exists": paths.scene_xml.is_file(),
        "grip_seed_exists": paths.grip_seed.is_file(),
    }
    (out_path / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    parser.add_argument("--stage", choices=("preflight",), default="preflight")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    paths = load_grip_hold_spec(args.spec)
    report = preflight(paths, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
