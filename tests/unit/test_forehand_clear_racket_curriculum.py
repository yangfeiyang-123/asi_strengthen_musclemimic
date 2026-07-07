from __future__ import annotations

from musclemimic.badminton.scripts.run_forehand_clear_racket_curriculum import (
    CurriculumStage,
    build_stage_command,
)


def test_build_stage_command_contains_stage_and_config():
    stage = CurriculumStage(
        name="soft_weld_medium",
        config="experiments/posttrain/forehand_clear_grip_hold_v1.yaml",
        total_steps=1000,
    )

    command = build_stage_command(stage)

    assert "--stage-name" in command
    assert "soft_weld_medium" in command
    assert "--config" in command
