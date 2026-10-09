"""Claim-driven static-to-flight Stage-3 v2 curriculum contract.

The stage description is deliberately simulator independent.  CPU and MJX
environments consume the same integer environment mode and reward mask, while
the training loop checkpoints the selected stage by name.  This keeps a C3
static-target checkpoint resumable into the C4--C7 flight curriculum without
changing the observation or control ABI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

V2_REWARD_TERM_ORDER = (
    "impact_position",
    "impact_center",
    "impact_time",
    "impact_normal",
    "impact_linear_velocity",
    "impact_angular_velocity",
    "precise_landing",
    "apex",
    "recovery_ready",
    "recovery_balance",
    "recovery_deceleration",
)

ENVIRONMENT_MODES = (
    "no_shuttle",
    "virtual_target",
    "fixed_shuttle",
    "dynamic_feed",
)
ENVIRONMENT_MODE_CODE = {name: index for index, name in enumerate(ENVIRONMENT_MODES)}


@dataclass(frozen=True)
class Stage3V2CurriculumStage:
    name: str
    start_steps: int
    feed_fraction: float
    environment_mode: str
    active_reward_terms: tuple[str, ...]
    promotion_thresholds: tuple[tuple[str, str, float], ...]

    def __post_init__(self) -> None:
        if int(self.start_steps) < 0:
            raise ValueError("Stage-3 v2 stage start_steps must be non-negative")
        if not 0.0 <= float(self.feed_fraction) <= 1.0:
            raise ValueError("Stage-3 v2 feed_fraction must lie in [0, 1]")
        if self.environment_mode not in ENVIRONMENT_MODE_CODE:
            raise ValueError(f"unknown Stage-3 v2 environment mode: {self.environment_mode}")
        unknown = set(self.active_reward_terms) - set(V2_REWARD_TERM_ORDER)
        if unknown:
            raise ValueError(f"unknown Stage-3 v2 reward terms: {sorted(unknown)}")

    @property
    def environment_mode_code(self) -> int:
        return ENVIRONMENT_MODE_CODE[self.environment_mode]

    @property
    def reward_mask(self) -> tuple[float, ...]:
        active = set(self.active_reward_terms)
        return tuple(float(name in active) for name in V2_REWARD_TERM_ORDER)


@dataclass(frozen=True)
class Stage3V2RuntimeValues:
    stage_index: int
    stage_name: str
    feed_fraction: float
    active_feed_count: int
    environment_mode: str
    environment_mode_code: int
    reward_mask: tuple[float, ...]


def canonical_stage3_v2_curriculum() -> tuple[Stage3V2CurriculumStage, ...]:
    return (
        Stage3V2CurriculumStage(
            "C0_ready_pose",
            0,
            0.0,
            "no_shuttle",
            ("recovery_ready", "recovery_balance"),
            (("ready_pose_error", "<=", 0.15), ("no_fall_rate", ">=", 0.99)),
        ),
        Stage3V2CurriculumStage(
            "C1_static_center_time",
            1_000_000,
            0.0,
            "virtual_target",
            ("impact_position", "impact_center", "impact_time"),
            (("impact_position_error_m", "<=", 0.15), ("impact_timing_mae_s", "<=", 0.10)),
        ),
        Stage3V2CurriculumStage(
            "C2_static_normal",
            2_000_000,
            0.0,
            "fixed_shuttle",
            ("impact_position", "impact_center", "impact_time", "impact_normal"),
            (("stringbed_normal_error_rad", "<=", 0.35),),
        ),
        Stage3V2CurriculumStage(
            "C3_static_velocity",
            4_000_000,
            0.0,
            "fixed_shuttle",
            (
                "impact_position",
                "impact_center",
                "impact_time",
                "impact_normal",
                "impact_linear_velocity",
                "impact_angular_velocity",
            ),
            (("racket_linear_velocity_rmse_m_s", "<=", 2.0), ("racket_angular_velocity_rmse_rad_s", "<=", 8.0)),
        ),
        Stage3V2CurriculumStage(
            "C4_deterministic_feed",
            6_000_000,
            0.05,
            "dynamic_feed",
            (
                "impact_position",
                "impact_center",
                "impact_time",
                "impact_normal",
                "impact_linear_velocity",
                "impact_angular_velocity",
                "precise_landing",
            ),
            (("hit_rate", ">=", 0.70), ("landing_rmse_m", "<=", 1.25)),
        ),
        Stage3V2CurriculumStage(
            "C5_feed_jitter",
            9_000_000,
            0.35,
            "dynamic_feed",
            (
                "impact_position",
                "impact_center",
                "impact_time",
                "impact_normal",
                "impact_linear_velocity",
                "impact_angular_velocity",
                "precise_landing",
                "apex",
            ),
            (("hit_rate", ">=", 0.80), ("crossed_net_rate", ">=", 0.70)),
        ),
        Stage3V2CurriculumStage(
            "C6_full_flight",
            13_000_000,
            1.0,
            "dynamic_feed",
            (
                "impact_position",
                "impact_center",
                "impact_time",
                "impact_normal",
                "impact_linear_velocity",
                "impact_angular_velocity",
                "precise_landing",
                "apex",
            ),
            (("landing_rmse_m", "<=", 0.85), ("apex_mae_m", "<=", 0.40)),
        ),
        Stage3V2CurriculumStage(
            "C7_recovery",
            18_000_000,
            1.0,
            "dynamic_feed",
            (
                "impact_position",
                "impact_center",
                "impact_time",
                "impact_normal",
                "impact_linear_velocity",
                "impact_angular_velocity",
                "precise_landing",
                "apex",
                "recovery_ready",
                "recovery_balance",
                "recovery_deceleration",
            ),
            (("recovery_ready_rate", ">=", 0.85), ("no_fall_rate", ">=", 0.98)),
        ),
    )


def stage_by_name(name: str) -> Stage3V2CurriculumStage:
    for stage in canonical_stage3_v2_curriculum():
        if stage.name == str(name):
            return stage
    valid = [stage.name for stage in canonical_stage3_v2_curriculum()]
    raise ValueError(f"unknown Stage-3 v2 curriculum stage {name!r}; expected one of {valid}")


def stage_for_steps(steps: int, *, max_stage: str | Stage3V2CurriculumStage | None = None) -> Stage3V2CurriculumStage:
    selected = canonical_stage3_v2_curriculum()[0]
    for stage in canonical_stage3_v2_curriculum():
        if int(steps) >= stage.start_steps:
            selected = stage
    if max_stage is not None:
        maximum = stage_by_name(max_stage) if isinstance(max_stage, str) else max_stage
        stages = canonical_stage3_v2_curriculum()
        selected_index = stages.index(selected)
        maximum_index = stages.index(maximum)
        selected = stages[min(selected_index, maximum_index)]
    return selected


def runtime_values(
    steps: int,
    *,
    feed_bank_size: int,
    max_stage: str | Stage3V2CurriculumStage | None = None,
) -> Stage3V2RuntimeValues:
    if int(feed_bank_size) <= 0:
        raise ValueError("Stage-3 v2 curriculum requires a non-empty feed bank")
    stages = canonical_stage3_v2_curriculum()
    stage = stage_for_steps(steps, max_stage=max_stage)
    active_feed_count = max(1, math.ceil(float(stage.feed_fraction) * int(feed_bank_size)))
    return Stage3V2RuntimeValues(
        stage_index=stages.index(stage),
        stage_name=stage.name,
        feed_fraction=float(stage.feed_fraction),
        active_feed_count=active_feed_count,
        environment_mode=stage.environment_mode,
        environment_mode_code=stage.environment_mode_code,
        reward_mask=stage.reward_mask,
    )


def runtime_values_for_stage(
    stage_index: int,
    *,
    feed_bank_size: int,
    max_stage: str | Stage3V2CurriculumStage,
) -> Stage3V2RuntimeValues:
    stages = canonical_stage3_v2_curriculum()
    maximum = stage_by_name(max_stage) if isinstance(max_stage, str) else max_stage
    maximum_index = stages.index(maximum)
    index = int(stage_index)
    if not 0 <= index <= maximum_index:
        raise ValueError(f"Stage-3 v2 stage index {index} is outside [0, {maximum_index}]")
    stage = stages[index]
    return runtime_values(
        stage.start_steps,
        feed_bank_size=feed_bank_size,
        max_stage=stage,
    )


def curriculum_complete(steps: int, *, max_stage: str) -> bool:
    return int(steps) >= stage_by_name(max_stage).start_steps


def promotion_failures(stage: Stage3V2CurriculumStage, metrics: Mapping[str, float]) -> tuple[str, ...]:
    failures: list[str] = []
    for name, operator, threshold in stage.promotion_thresholds:
        if name not in metrics:
            failures.append(f"missing:{name}")
            continue
        value = float(metrics[name])
        passed = value <= threshold if operator == "<=" else value >= threshold
        if not passed:
            failures.append(f"{name}{operator}{threshold:g} (got {value:g})")
    return tuple(failures)
