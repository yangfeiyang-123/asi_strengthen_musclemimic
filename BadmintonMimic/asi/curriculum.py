from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactTrackingCurriculumStage:
    name: str
    start_update: int
    max_quality_tier: str
    min_frames: int
    max_frames: int | None
    reward_terms: tuple[str, ...]


def build_default_contact_tracking_curriculum() -> tuple[ContactTrackingCurriculumStage, ...]:
    return (
        ContactTrackingCurriculumStage(
            name="short_clean",
            start_update=0,
            max_quality_tier="A",
            min_frames=30,
            max_frames=120,
            reward_terms=("root_pos", "root_rot", "body_pos"),
        ),
        ContactTrackingCurriculumStage(
            name="joint_tracking",
            start_update=2_000,
            max_quality_tier="A",
            min_frames=30,
            max_frames=180,
            reward_terms=("root_pos", "root_rot", "body_pos", "joint_pos", "joint_vel"),
        ),
        ContactTrackingCurriculumStage(
            name="contact_tracking",
            start_update=5_000,
            max_quality_tier="B",
            min_frames=45,
            max_frames=240,
            reward_terms=(
                "root_pos",
                "root_rot",
                "body_pos",
                "joint_pos",
                "joint_vel",
                "foot_contact_height",
                "foot_contact_velocity",
            ),
        ),
        ContactTrackingCurriculumStage(
            name="long_clips",
            start_update=10_000,
            max_quality_tier="B",
            min_frames=60,
            max_frames=None,
            reward_terms=(
                "root_pos",
                "root_rot",
                "body_pos",
                "joint_pos",
                "joint_vel",
                "body_graph",
                "foot_contact_height",
                "foot_contact_velocity",
            ),
        ),
        ContactTrackingCurriculumStage(
            name="full_finetune",
            start_update=20_000,
            max_quality_tier="C",
            min_frames=60,
            max_frames=None,
            reward_terms=(
                "root_pos",
                "root_rot",
                "body_pos",
                "body_rot",
                "joint_pos",
                "joint_vel",
                "body_graph",
                "foot_contact_height",
                "foot_contact_velocity",
                "muscle_effort",
                "action_rate",
            ),
        ),
    )


def stage_for_update(
    stages: tuple[ContactTrackingCurriculumStage, ...] | list[ContactTrackingCurriculumStage],
    update: int,
) -> ContactTrackingCurriculumStage:
    if not stages:
        raise ValueError("stages must not be empty")
    selected = stages[0]
    for stage in stages:
        if int(update) >= int(stage.start_update):
            selected = stage
        else:
            break
    return selected

