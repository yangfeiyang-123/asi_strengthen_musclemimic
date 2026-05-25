from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.grip.paths import target_config_path


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

    points = {
        name: GripTargetPoint(
            y=float(value["y"]),
            theta_deg=float(value["theta_deg"]),
            weight=float(value["weight"]),
        )
        for name, value in raw["target_points_racket_local"].items()
    }
    required = {"palm", "thumb", "index", "middle", "ring", "pinky"}
    missing = required.difference(points)
    if missing:
        raise ValueError(f"missing grip target points: {sorted(missing)}")

    return GripTargetConfig(
        handle_radius_m=float(raw["handle_radius_m"]),
        contact_clearance_m=float(raw["contact_clearance_m"]),
        target_points_racket_local=points,
        raw=raw,
    )
