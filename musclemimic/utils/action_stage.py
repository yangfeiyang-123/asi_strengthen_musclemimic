"""Classify badminton motions into training stages from reference metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


STAGE_BASE = "base"
STAGE_POSTTRAIN = "posttrain"
STAGE_REPAIR = "repair"
STAGE_EXCLUDE = "exclude"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

REQUIRED_TRAIN = "train"
REQUIRED_POSTTRAIN = "posttrain"
REQUIRED_REPAIR_FIRST = "repair_first"
REQUIRED_EXCLUDE = "exclude"
REQUIRED_MANUAL_REVIEW = "manual_review"

ROOT_SMALL_DISPLACEMENT = 0.25
ROOT_LARGE_DISPLACEMENT = 0.60
ROOT_HIGH_PEAK_SPEED = 1.20
ROOT_LARGE_YAW_CHANGE = 0.80

ROOT_DISPLACEMENT_MARGIN = 0.05
ROOT_PEAK_SPEED_MARGIN = 0.10
ROOT_YAW_MARGIN = 0.05


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
    confidence: str = CONFIDENCE_HIGH
    failure_modes: tuple[str, ...] = ()
    review_required: bool = False
    required_action: str = REQUIRED_TRAIN


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


def _near(value: float, threshold: float, margin: float) -> bool:
    return abs(value - threshold) <= margin


def _borderline_failure_modes(root_disp: float, root_peak_speed: float, yaw_change: float) -> list[str]:
    failures: list[str] = []
    if _near(root_disp, ROOT_SMALL_DISPLACEMENT, ROOT_DISPLACEMENT_MARGIN):
        failures.append("borderline_small_root_displacement")
    if _near(root_disp, ROOT_LARGE_DISPLACEMENT, ROOT_DISPLACEMENT_MARGIN):
        failures.append("borderline_large_root_displacement")
    if _near(root_peak_speed, ROOT_HIGH_PEAK_SPEED, ROOT_PEAK_SPEED_MARGIN):
        failures.append("borderline_root_peak_speed")
    if _near(yaw_change, ROOT_LARGE_YAW_CHANGE, ROOT_YAW_MARGIN):
        failures.append("borderline_root_yaw_change")
    return failures


def _confidence_from_failures(failure_modes: list[str]) -> tuple[str, bool]:
    if any(not failure.startswith("borderline_") for failure in failure_modes):
        return CONFIDENCE_LOW, True
    if failure_modes:
        return CONFIDENCE_MEDIUM, True
    return CONFIDENCE_HIGH, False


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
    borderline_failures = _borderline_failure_modes(root_disp, root_peak_speed, yaw_change)

    label_family = _label_family(hints.action_label)
    reasons: list[str] = []

    if hints.fine_hand_dominant:
        return StageDecision(
            stage=STAGE_EXCLUDE,
            family="fine_hand",
            reasons=("fine_hand_dominant",),
            confidence=CONFIDENCE_LOW,
            failure_modes=("fine_hand_dominant",),
            review_required=True,
            required_action=REQUIRED_EXCLUDE,
        )

    if hints.contact_unreliable:
        reasons.append("contact_unreliable")
    if hints.endpoint_unreliable:
        reasons.append("endpoint_unreliable")
    if hints.expected_large_motion and root_disp < ROOT_SMALL_DISPLACEMENT:
        reasons.append("expected_large_motion_but_root_is_small")

    repair_reasons = (
        "contact_unreliable",
        "endpoint_unreliable",
        "expected_large_motion_but_root_is_small",
    )
    if any(reason in reasons for reason in repair_reasons):
        failure_modes = list(dict.fromkeys(reasons + borderline_failures))
        confidence, review_required = _confidence_from_failures(failure_modes)
        return StageDecision(
            stage=STAGE_REPAIR,
            family=label_family,
            reasons=tuple(reasons),
            confidence=confidence,
            failure_modes=tuple(failure_modes),
            review_required=review_required,
            required_action=REQUIRED_REPAIR_FIRST,
        )

    if root_disp < ROOT_SMALL_DISPLACEMENT:
        reasons.append("stationary_or_small_step")
    elif root_disp <= ROOT_LARGE_DISPLACEMENT:
        reasons.append("medium_root_displacement")
    else:
        reasons.append("large_root_displacement")

    if root_peak_speed > ROOT_HIGH_PEAK_SPEED:
        reasons.append("high_root_peak_speed")
    if yaw_change > ROOT_LARGE_YAW_CHANGE:
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
        confidence, review_required = _confidence_from_failures(borderline_failures)
        return StageDecision(
            stage=STAGE_POSTTRAIN,
            family=_posttrain_family(label_family, reasons),
            reasons=tuple(reasons),
            confidence=confidence,
            failure_modes=tuple(borderline_failures),
            review_required=review_required,
            required_action=REQUIRED_POSTTRAIN,
        )

    confidence, review_required = _confidence_from_failures(borderline_failures)
    return StageDecision(
        stage=STAGE_BASE,
        family="general" if label_family == "fine_hand" else label_family,
        reasons=tuple(reasons),
        confidence=confidence,
        failure_modes=tuple(borderline_failures),
        review_required=review_required,
        required_action=REQUIRED_TRAIN,
    )
