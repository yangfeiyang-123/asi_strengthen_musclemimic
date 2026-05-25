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


@dataclass(frozen=True)
class GripTargetConfig:
    handle_radius_m: float
    contact_clearance_m: float
    target_points_racket_local: dict[str, GripTargetPoint]
    raw: dict[str, Any]

    def target_xyz(self, name: str) -> np.ndarray:
        point = self.target_points_racket_local[name]
        theta = math.radians(point.theta_deg)
        return np.array(
            [
                self.handle_radius_m * math.cos(theta),
                point.y,
                self.handle_radius_m * math.sin(theta),
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

    return GripTargetConfig(
        handle_radius_m=handle_radius_m,
        contact_clearance_m=contact_clearance_m,
        target_points_racket_local=points,
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

    return GripTargetPoint(
        y=_finite_float(value["y"], f"{name}.y"),
        theta_deg=_finite_float(value["theta_deg"], f"{name}.theta_deg"),
        weight=weight,
    )


def _finite_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return number
