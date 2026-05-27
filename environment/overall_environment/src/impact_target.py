from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyScale:
    shoulder_height_m: float
    arm_reach_up_m: float
    racket_effective_length_m: float


@dataclass(frozen=True)
class ImpactTargetConfig:
    min_forward_offset_m: float = 0.28
    max_forward_offset_m: float = 0.90
    min_racket_side_offset_m: float = 0.16
    max_racket_side_offset_m: float = 0.70
    reach_alpha: float = 0.78
    racket_beta: float = 0.82
    min_height_margin_m: float = -0.08
    max_height_margin_m: float = 0.08


@dataclass(frozen=True)
class ImpactTarget:
    impact_frame: int
    impact_phase: float
    position_root_local: np.ndarray
    racket_head_velocity_dir: np.ndarray
    racket_normal_hint: np.ndarray


def _as_array(name: str, value: np.ndarray, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float).copy()
    return np.asarray(vec, dtype=float) / norm


def extract_impact_target_from_sites(
    *,
    right_hand_pos: np.ndarray,
    root_pos: np.ndarray,
    forward_axis: np.ndarray,
    right_axis: np.ndarray,
    dt: float,
    racket_length_m: float,
) -> ImpactTarget:
    right_hand_pos = _as_array("right_hand_pos", right_hand_pos, 3)
    root_pos = _as_array("root_pos", root_pos, 3)
    forward_axis = _as_array("forward_axis", forward_axis, 3)
    right_axis = _as_array("right_axis", right_axis, 3)
    if right_hand_pos.shape[0] < 3:
        raise ValueError("at least three frames are required")
    if root_pos.shape != right_hand_pos.shape:
        raise ValueError("root_pos must match right_hand_pos shape")
    if forward_axis.shape != right_hand_pos.shape:
        raise ValueError("forward_axis must match right_hand_pos shape")
    if right_axis.shape != right_hand_pos.shape:
        raise ValueError("right_axis must match right_hand_pos shape")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if racket_length_m <= 0.0:
        raise ValueError("racket_length_m must be positive")

    forward_unit = np.vstack([_normalize(v, np.array([1.0, 0.0, 0.0])) for v in forward_axis])
    right_unit = np.vstack([_normalize(v, np.array([0.0, 1.0, 0.0])) for v in right_axis])
    virtual_head = right_hand_pos + racket_length_m * forward_unit
    velocity = np.gradient(virtual_head, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    rel = virtual_head - root_pos
    forward_offset = np.einsum("ij,ij->i", rel, forward_unit)
    side_offset = np.einsum("ij,ij->i", rel, right_unit)
    candidate = (forward_offset > 0.0) & (side_offset > 0.0)
    if np.any(candidate):
        masked_speed = np.where(candidate, speed, -np.inf)
        max_candidate_speed = float(np.max(masked_speed))
        fast_candidate = candidate & (speed >= 0.5 * max_candidate_speed)
        if np.any(fast_candidate):
            masked_height = np.where(fast_candidate, rel[:, 2], -np.inf)
            impact_frame = int(np.argmax(masked_height))
        else:
            impact_frame = int(np.argmax(masked_speed))
    else:
        impact_frame = int(np.argmax(speed))

    position_root_local = np.array(
        [
            forward_offset[impact_frame],
            side_offset[impact_frame],
            rel[impact_frame, 2],
        ],
        dtype=float,
    )
    phase_denominator = max(1, right_hand_pos.shape[0] - 1)
    impact_phase = float(impact_frame / phase_denominator)
    velocity_dir = _normalize(velocity[impact_frame], forward_unit[impact_frame])
    racket_normal_hint = _normalize(
        np.cross(velocity_dir, right_unit[impact_frame]),
        np.array([0.0, 0.0, 1.0]),
    )
    return ImpactTarget(
        impact_frame=impact_frame,
        impact_phase=impact_phase,
        position_root_local=position_root_local,
        racket_head_velocity_dir=velocity_dir,
        racket_normal_hint=racket_normal_hint,
    )


def regularize_impact_target(
    position_root_local: np.ndarray,
    body_scale: BodyScale,
    config: ImpactTargetConfig = ImpactTargetConfig(),
) -> np.ndarray:
    position = np.asarray(position_root_local, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"position_root_local must have shape (3,), got {position.shape}")
    if not np.isfinite(position).all():
        raise ValueError("position_root_local contains non-finite values")

    nominal_height = (
        body_scale.shoulder_height_m
        + body_scale.arm_reach_up_m * config.reach_alpha
        + body_scale.racket_effective_length_m * config.racket_beta
    )
    min_height = nominal_height + config.min_height_margin_m
    max_height = nominal_height + config.max_height_margin_m

    return np.array(
        [
            np.clip(position[0], config.min_forward_offset_m, config.max_forward_offset_m),
            np.clip(position[1], config.min_racket_side_offset_m, config.max_racket_side_offset_m),
            np.clip(position[2], min_height, max_height),
        ],
        dtype=float,
    )
