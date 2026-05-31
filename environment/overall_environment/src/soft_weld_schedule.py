from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftWeldSchedule:
    stage: str
    weld_enabled: bool
    weld_strength: float
    solref: tuple[float, float]
    solimp: tuple[float, float, float]
    reward_weights: dict[str, float]


def soft_weld_schedule(stage: str) -> SoftWeldSchedule:
    schedules = {
        "strong_weld": SoftWeldSchedule(
            stage="strong_weld",
            weld_enabled=True,
            weld_strength=1.0,
            solref=(0.002, 1.0),
            solimp=(0.95, 0.99, 0.001),
            reward_weights={"racket_pose": 8.0, "hand_pose": 4.0, "contact": 0.5, "no_slip": 0.5},
        ),
        "medium_weld": SoftWeldSchedule(
            stage="medium_weld",
            weld_enabled=True,
            weld_strength=0.5,
            solref=(0.006, 1.0),
            solimp=(0.9, 0.97, 0.002),
            reward_weights={"racket_pose": 4.0, "hand_pose": 3.0, "contact": 2.0, "no_slip": 2.0},
        ),
        "weak_weld": SoftWeldSchedule(
            stage="weak_weld",
            weld_enabled=True,
            weld_strength=0.2,
            solref=(0.015, 1.0),
            solimp=(0.8, 0.95, 0.004),
            reward_weights={"racket_pose": 2.0, "hand_pose": 2.0, "contact": 4.0, "no_slip": 6.0},
        ),
        "contact_only": SoftWeldSchedule(
            stage="contact_only",
            weld_enabled=False,
            weld_strength=0.0,
            solref=(0.0, 0.0),
            solimp=(0.0, 0.0, 0.0),
            reward_weights={"racket_pose": 1.0, "hand_pose": 1.0, "contact": 6.0, "no_slip": 8.0},
        ),
    }
    if stage not in schedules:
        raise ValueError(f"unknown soft-weld stage: {stage}")
    return schedules[stage]
