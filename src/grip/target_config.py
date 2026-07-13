from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.grip.paths import target_config_path

_REQUIRED_TOP_LEVEL_KEYS = ("handle_radius_m", "contact_clearance_m", "target_points_racket_local")
_REQUIRED_TARGET_NAMES = {"palm", "thumb", "index", "middle", "ring", "pinky"}
_REQUIRED_POINT_FIELDS = ("y", "theta_deg", "weight")


@dataclass(frozen=True)
class GripTargetPoint:
    y: float
    theta_deg: float
    weight: float
    radial_offset_m: float = 0.0


@dataclass(frozen=True)
class ForehandVTargets:
    """Standard right-hand forehand grip orientation targets.

    The tiger mouth (thumb-index web) rests on the +45 deg diagonal bevel, the
    thumb lies on the +Z wide face, the palm faces the handle from the -135 deg
    bevel, and the palm center keeps a small radial clearance (slightly hollow).
    """

    v_bisector_theta_deg: float = 45.0
    palm_theta_deg: float = -135.0
    thumb_theta_deg: float = 90.0
    palm_min_clearance_m: float = 0.003
    max_thumb_index_y_gap_m: float = 0.015


@dataclass(frozen=True)
class GripTargetConfig:
    handle_radius_m: float
    contact_clearance_m: float
    target_points_racket_local: dict[str, GripTargetPoint]
    forehand_v: ForehandVTargets
    raw: dict[str, Any]

    def target_xyz(self, name: str) -> np.ndarray:
        point = self.target_points_racket_local[name]
        theta = math.radians(point.theta_deg)
        radius = self.handle_radius_m + point.radial_offset_m
        return np.array(
            [
                radius * math.cos(theta),
                point.y,
                radius * math.sin(theta),
            ],
            dtype=float,
        )

    def target_weight(self, name: str) -> float:
        return self.target_points_racket_local[name].weight


def load_grip_target_config(path: str | Path | None = None) -> GripTargetConfig:
    config_path = Path(path) if path is not None else target_config_path()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    _require_object(raw, "config root")
    _validate_top_level_keys(raw)

    target_points_raw = raw["target_points_racket_local"]
    _require_object(target_points_raw, "target_points_racket_local")
    missing_targets = sorted(_REQUIRED_TARGET_NAMES.difference(target_points_raw))
    if missing_targets:
        raise ValueError(f"missing required target(s): {missing_targets}")

    handle_radius_m = _finite_float(raw["handle_radius_m"], "handle_radius_m")
    if handle_radius_m <= 0:
        raise ValueError(f"handle_radius_m must be > 0, got {handle_radius_m}")

    contact_clearance_m = _finite_float(raw["contact_clearance_m"], "contact_clearance_m")
    if contact_clearance_m < 0:
        raise ValueError(f"contact_clearance_m must be >= 0, got {contact_clearance_m}")

    points = {name: _parse_target_point(name, value) for name, value in target_points_raw.items()}
    forehand_v = _parse_forehand_v(raw.get("forehand_v"))

    return GripTargetConfig(
        handle_radius_m=handle_radius_m,
        contact_clearance_m=contact_clearance_m,
        target_points_racket_local=points,
        forehand_v=forehand_v,
        raw=raw,
    )


def _require_object(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")


def _validate_top_level_keys(raw: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing top-level key(s): {missing}")


def _parse_target_point(name: str, value: Any) -> GripTargetPoint:
    _require_object(value, f"target point {name}")
    missing_fields = [field for field in _REQUIRED_POINT_FIELDS if field not in value]
    if missing_fields:
        raise ValueError(f"target point {name} missing field(s): {missing_fields}")

    weight = _finite_float(value["weight"], f"{name}.weight")
    if weight <= 0:
        raise ValueError(f"{name}.weight must be > 0, got {weight}")

    radial_offset_m = 0.0
    if "radial_offset_m" in value:
        radial_offset_m = _finite_float(value["radial_offset_m"], f"{name}.radial_offset_m")
        if radial_offset_m < 0:
            raise ValueError(f"{name}.radial_offset_m must be >= 0, got {radial_offset_m}")

    return GripTargetPoint(
        y=_finite_float(value["y"], f"{name}.y"),
        theta_deg=_finite_float(value["theta_deg"], f"{name}.theta_deg"),
        weight=weight,
        radial_offset_m=radial_offset_m,
    )


def _parse_forehand_v(value: Any) -> ForehandVTargets:
    if value is None:
        return ForehandVTargets()
    _require_object(value, "forehand_v")
    defaults = ForehandVTargets()
    kwargs = {}
    for field in (
        "v_bisector_theta_deg",
        "palm_theta_deg",
        "thumb_theta_deg",
        "palm_min_clearance_m",
        "max_thumb_index_y_gap_m",
    ):
        if field in value:
            kwargs[field] = _finite_float(value[field], f"forehand_v.{field}")
        else:
            kwargs[field] = getattr(defaults, field)
    if kwargs["palm_min_clearance_m"] < 0:
        raise ValueError(f"forehand_v.palm_min_clearance_m must be >= 0, got {kwargs['palm_min_clearance_m']}")
    if kwargs["max_thumb_index_y_gap_m"] <= 0:
        raise ValueError(f"forehand_v.max_thumb_index_y_gap_m must be > 0, got {kwargs['max_thumb_index_y_gap_m']}")
    return ForehandVTargets(**kwargs)


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a JSON number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return number
