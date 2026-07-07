from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    config: str
    total_steps: int


def build_stage_command(stage: CurriculumStage) -> list[str]:
    return [
        "python",
        "musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py",
        "--stage-name",
        stage.name,
        "--config",
        stage.config,
        "--total-steps",
        str(stage.total_steps),
    ]


def default_curriculum_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            "strong_weld_grip",
            "experiments/posttrain/forehand_clear_grip_hold_v1.yaml",
            50_000,
        ),
        CurriculumStage(
            "medium_weld_swing",
            "experiments/posttrain/forehand_clear_grip_hold_v1.yaml",
            100_000,
        ),
        CurriculumStage(
            "static_hit",
            "experiments/posttrain/forehand_clear_static_hit_v1.yaml",
            200_000,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForehandClear racket curriculum stages.")
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    for stage in default_curriculum_stages():
        print(" ".join(build_stage_command(stage)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
