from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.overall_environment.src.ghost_racket import (
    GhostRacketFrame,
    GhostRacketTrajectory,
    interpolate_ghost,
)


@dataclass(frozen=True)
class GhostRacketTeacherTrajectory:
    trajectory: GhostRacketTrajectory
    impact_index: int
    impact_phase: float
    head_speed: np.ndarray

    @property
    def phase(self) -> np.ndarray:
        return self.trajectory.phase

    @property
    def frames(self) -> list[GhostRacketFrame]:
        return self.trajectory.frames


def build_ghost_racket_trajectory(
    right_hand_pos: np.ndarray,
    *,
    fps: float,
    racket_offset: np.ndarray | None = None,
) -> GhostRacketTeacherTrajectory:
    hand = np.asarray(right_hand_pos, dtype=float)
    if hand.ndim != 2 or hand.shape[1] != 3:
        raise ValueError(f"right_hand_pos must have shape (T, 3), got {hand.shape}")
    if hand.shape[0] < 2:
        raise ValueError("ghost racket trajectory requires at least two frames")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not np.isfinite(hand).all():
        raise ValueError("right_hand_pos contains non-finite values")

    offset = np.asarray(racket_offset if racket_offset is not None else [0.0, 0.55, 0.0], dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"racket_offset must have shape (3,), got {offset.shape}")
    pos = hand + offset
    dt = 1.0 / float(fps)
    velocity = _finite_difference_velocity(pos, dt)
    head_speed = np.linalg.norm(velocity, axis=1)
    impact_index = int(np.argmax(head_speed))
    phase = np.linspace(0.0, 1.0, hand.shape[0], dtype=float)
    frames = [
        GhostRacketFrame(pos=pos[index], xmat=_orientation_from_velocity(velocity[index]), velocity=velocity[index])
        for index in range(hand.shape[0])
    ]
    return GhostRacketTeacherTrajectory(
        trajectory=GhostRacketTrajectory(phase=phase, frames=frames),
        impact_index=impact_index,
        impact_phase=float(phase[impact_index]),
        head_speed=head_speed,
    )


def sample_ghost_at_phase(teacher: GhostRacketTeacherTrajectory, phase: float) -> GhostRacketFrame:
    return interpolate_ghost(teacher.trajectory, phase)


def compute_ghost_reward(
    *,
    racket_pos: np.ndarray,
    racket_xmat: np.ndarray,
    racket_velocity: np.ndarray,
    ghost_frame: GhostRacketFrame,
    pos_sigma: float = 0.15,
    vel_sigma: float = 6.0,
) -> dict[str, float]:
    pos = np.asarray(racket_pos, dtype=float)
    xmat = np.asarray(racket_xmat, dtype=float)
    vel = np.asarray(racket_velocity, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"racket_pos must have shape (3,), got {pos.shape}")
    if xmat.shape != (3, 3):
        raise ValueError(f"racket_xmat must have shape (3, 3), got {xmat.shape}")
    if vel.shape != (3,):
        raise ValueError(f"racket_velocity must have shape (3,), got {vel.shape}")

    pos_error = float(np.linalg.norm(pos - np.asarray(ghost_frame.pos, dtype=float)))
    vel_error = float(np.linalg.norm(vel - np.asarray(ghost_frame.velocity, dtype=float)))
    orient_error = _orientation_error(xmat, np.asarray(ghost_frame.xmat, dtype=float))
    terms = {
        "pos": float(np.exp(-pos_error / max(pos_sigma, 1e-9))),
        "orient": float(np.exp(-orient_error)),
        "velocity": float(np.exp(-vel_error / max(vel_sigma, 1e-9))),
    }
    terms["total"] = float(0.5 * terms["pos"] + 0.25 * terms["orient"] + 0.25 * terms["velocity"])
    return terms


def _finite_difference_velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros_like(pos)
    velocity[0] = (pos[1] - pos[0]) / dt
    for index in range(1, pos.shape[0]):
        velocity[index] = (pos[index] - pos[index - 1]) / dt
    return velocity


def _orientation_from_velocity(velocity: np.ndarray) -> np.ndarray:
    forward = np.asarray(velocity, dtype=float)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = forward / norm
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    side = np.cross(up, forward)
    if float(np.linalg.norm(side)) < 1e-9:
        side = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        side = side / np.linalg.norm(side)
    normal = np.cross(forward, side)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    return np.column_stack([forward, side, normal])


def _orientation_error(a: np.ndarray, b: np.ndarray) -> float:
    rotation = a @ b.T
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(np.arccos(max(-1.0, min(1.0, cosine))))
