"""Classify badminton motions into training stages from reference metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


STAGE_BASE = "base"
STAGE_POSTTRAIN = "posttrain"
STAGE_REPAIR = "repair"
STAGE_EXCLUDE = "exclude"


@dataclass(frozen=True)
class MotionHints:
    action_label: str = ""
    expected_large_motion: bool = False
    has_jump_or_lunge: bool = False
    contact_unreliable: bool = False
    endpoint_unreliable: bool = False
    fine_hand_dominant: bool = False


@dataclass(frozen=True)
class StageDecision:
    stage: str
    family: str
    reasons: tuple[str, ...]


def _metric(metrics: Mapping[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(name)
    return float(metrics[name])


def _label_family(label: str) -> str:
    normalized = label.lower()
    if "net" in normalized or "lift" in normalized or "front" in normalized:
        return "net_frontcourt"
    if "smash" in normalized:
        return "smash"
    if "backhand" in normalized or "turn" in normalized or "rotation" in normalized:
        return "rotation"
    if "footwork" in normalized or "step" in normalized or "lunge" in normalized:
        return "footwork"
    if "tumble" in normalized or "slice" in normalized or "wrist" in normalized:
        return "fine_hand"
    return "general"


def _posttrain_family(label_family: str, reasons: list[str]) -> str:
    if "large_yaw_change" in reasons:
        return "rotation"
    if label_family in {"net_frontcourt", "smash", "rotation", "footwork"}:
        return label_family
    return "general"


def classify_motion_stage(metrics: Mapping[str, float], hints: MotionHints | None = None) -> StageDecision:
    """Return a stage recommendation for one retargeted badminton motion.

    The thresholds are intentionally simple and conservative. They match the
    first design pass and should be tuned after real training outcomes exist.
    """

    hints = hints or MotionHints()
    root_disp = _metric(metrics, "reference_root_xy_total_displacement")
    root_peak_speed = _metric(metrics, "reference_root_xy_peak_speed")
    yaw_change = abs(_metric(metrics, "reference_root_yaw_change"))
    _right_hand_path_length = _metric(metrics, "right_hand_world_path_length")

    label_family = _label_family(hints.action_label)
    reasons: list[str] = []

    if hints.fine_hand_dominant:
        return StageDecision(
            stage=STAGE_EXCLUDE,
            family="fine_hand",
            reasons=("fine_hand_dominant",),
        )

    if hints.contact_unreliable:
        reasons.append("contact_unreliable")
    if hints.endpoint_unreliable:
        reasons.append("endpoint_unreliable")
    if hints.expected_large_motion and root_disp < 0.25:
        reasons.append("expected_large_motion_but_root_is_small")

    repair_reasons = (
        "contact_unreliable",
        "endpoint_unreliable",
        "expected_large_motion_but_root_is_small",
    )
    if any(reason in reasons for reason in repair_reasons):
        return StageDecision(
            stage=STAGE_REPAIR,
            family=label_family,
            reasons=tuple(reasons),
        )

    if root_disp < 0.25:
        reasons.append("stationary_or_small_step")
    elif root_disp <= 0.60:
        reasons.append("medium_root_displacement")
    else:
        reasons.append("large_root_displacement")

    if root_peak_speed > 1.20:
        reasons.append("high_root_peak_speed")
    if yaw_change > 0.80:
        reasons.append("large_yaw_change")
    if hints.has_jump_or_lunge:
        reasons.append("jump_or_lunge_hint")

    posttrain_reasons = {
        "large_root_displacement",
        "high_root_peak_speed",
        "large_yaw_change",
        "jump_or_lunge_hint",
    }
    if posttrain_reasons.intersection(reasons):
        return StageDecision(
            stage=STAGE_POSTTRAIN,
            family=_posttrain_family(label_family, reasons),
            reasons=tuple(reasons),
        )

    return StageDecision(
        stage=STAGE_BASE,
        family="general" if label_family == "fine_hand" else label_family,
        reasons=tuple(reasons),
    )
