#!/usr/bin/env python3
"""Write a claim-to-evidence template for badminton validation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_USER_ERRORS = (OSError, ValueError)


def _template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "required_experiments": [
            "all_mix",
            "action_name_grouping",
            "metric_gated_staging",
        ],
        "required_ablations": [
            "no_repair_gate",
            "no_rotation_speed_gate",
            "no_posttrain_root_focus",
        ],
        "metrics": [
            "root_xy_rmse",
            "root_displacement_ratio",
            "root_speed_rmse",
            "heading_yaw_error",
            "right_hand_position_error",
            "relative_body_pose_error",
            "early_termination_rate",
            "foot_slip_proxy",
            "control_action_rate_cost",
        ],
        "claims": [
            {
                "id": "staging_improves_stability",
                "claim": "Staging improves training stability.",
                "required_evidence": [
                    "metric_gated_staging has lower early termination or fewer catastrophic failures than all_mix",
                    "metric_gated_staging does not regress aggregate tracking metrics relative to action_name_grouping",
                ],
                "status": "not_evaluated",
                "decision_rule": "support only if stability improves without a severe tracking regression",
            },
            {
                "id": "posttrain_helps_large_motion",
                "claim": "Posttraining helps movement-heavy badminton actions.",
                "required_evidence": [
                    "posttrain actions improve root displacement ratio or root_xy_rmse",
                    "improvement appears on held-out or repeated movement-heavy clips",
                ],
                "status": "not_evaluated",
                "decision_rule": "downgrade to action-specific fine-tuning if only one clip improves",
            },
            {
                "id": "repair_gate_protects_training",
                "claim": "Repair/exclusion prevents corrupted references from hurting training.",
                "required_evidence": [
                    "no_repair_gate performs worse on reliability or tracking metrics",
                    "flagged repair clips show concrete data-quality failure modes",
                ],
                "status": "not_evaluated",
                "decision_rule": "treat as data hygiene if performance evidence is neutral",
            },
            {
                "id": "metric_gated_beats_action_name_grouping",
                "claim": "Metric-gated staging is better than action-name grouping.",
                "required_evidence": [
                    "metric_gated_staging beats action_name_grouping on at least one aggregate metric",
                    "metric_gated_staging does not create a severe family-level regression",
                ],
                "status": "not_evaluated",
                "decision_rule": "reframe as diagnostics if action-name grouping is equal or better",
            },
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _write_json(args.output, _template())
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote claim evidence template: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
