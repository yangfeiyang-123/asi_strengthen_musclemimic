"""Event-style racket impact helpers for the shuttlecock model."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass(frozen=True)
class ShuttlecockImpactConfig:
    event_restitution_normal: float = 0.5
    event_tangential_velocity_scale: float = 0.85
    min_speed_for_event_m_s: float = 5.0
    max_rebound_speed_m_s: float = 100.0


@dataclass(frozen=True)
class ShuttlecockImpactDiagnostics:
    relative_normal_velocity_m_s: float
    rebound_speed_m_s: float
    rebound_clipped: bool


def _normalized(vec: np.ndarray, name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return vec / norm


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0:
        return vec * (max_norm / norm)
    return vec


def compute_event_rebound_velocity(
    *,
    shuttle_velocity_world: np.ndarray,
    racket_surface_velocity_world: np.ndarray,
    normal_world: np.ndarray,
    cfg: ShuttlecockImpactConfig,
) -> np.ndarray:
    """Return shuttlecock velocity after an event impact with a racket surface."""
    return compute_event_rebound(
        shuttle_velocity_world=shuttle_velocity_world,
        racket_surface_velocity_world=racket_surface_velocity_world,
        normal_world=normal_world,
        cfg=cfg,
    )[0]


def compute_event_rebound(
    *,
    shuttle_velocity_world: np.ndarray,
    racket_surface_velocity_world: np.ndarray,
    normal_world: np.ndarray,
    cfg: ShuttlecockImpactConfig,
) -> tuple[np.ndarray, ShuttlecockImpactDiagnostics]:
    """Return event impact velocity and diagnostics for a racket surface."""
    shuttle_velocity_world = np.asarray(shuttle_velocity_world, dtype=float)
    racket_surface_velocity_world = np.asarray(racket_surface_velocity_world, dtype=float)
    normal = _normalized(normal_world, "normal_world")

    relative_velocity = shuttle_velocity_world - racket_surface_velocity_world
    relative_normal_speed = float(np.dot(relative_velocity, normal))
    if relative_normal_speed >= 0.0:
        rebound_velocity = shuttle_velocity_world.copy()
        return rebound_velocity, ShuttlecockImpactDiagnostics(
            relative_normal_velocity_m_s=relative_normal_speed,
            rebound_speed_m_s=float(np.linalg.norm(rebound_velocity)),
            rebound_clipped=False,
        )

    relative_normal_velocity = relative_normal_speed * normal
    relative_tangential_velocity = relative_velocity - relative_normal_velocity
    rebound_relative_velocity = (
        cfg.event_tangential_velocity_scale * relative_tangential_velocity
        - cfg.event_restitution_normal * relative_normal_velocity
    )
    rebound_velocity = racket_surface_velocity_world + rebound_relative_velocity
    rebound_speed = float(np.linalg.norm(rebound_velocity))
    rebound_clipped = rebound_speed > cfg.max_rebound_speed_m_s > 0
    rebound_velocity = _clip_norm(rebound_velocity, cfg.max_rebound_speed_m_s)
    return rebound_velocity, ShuttlecockImpactDiagnostics(
        relative_normal_velocity_m_s=relative_normal_speed,
        rebound_speed_m_s=float(np.linalg.norm(rebound_velocity)),
        rebound_clipped=rebound_clipped,
    )


def should_apply_event_rebound(contact: Mapping[str, object], cfg: ShuttlecockImpactConfig) -> bool:
    """Return whether an active contact is closing fast enough for an event rebound."""
    if not bool(contact.get("active", False)):
        return False
    relative_normal_velocity = float(contact.get("relative_normal_velocity", 0.0))
    return relative_normal_velocity < -cfg.min_speed_for_event_m_s


def compute_equal_opposite_event_impulses(
    *,
    shuttle_mass_kg: float,
    velocity_before_world: np.ndarray,
    velocity_after_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the event impulse on the shuttle and its racket reaction.

    The event model prescribes the shuttle COM velocity discontinuity.  Its
    linear impulse is therefore ``m * (v_after - v_before)``; the rigid racket
    receives the exact opposite impulse at the cork/contact point.
    """

    mass = float(shuttle_mass_kg)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("shuttle_mass_kg must be finite and positive")
    before = np.asarray(velocity_before_world, dtype=float)
    after = np.asarray(velocity_after_world, dtype=float)
    if before.shape != (3,) or after.shape != (3,):
        raise ValueError("event impulse velocities must be three-vectors")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("event impulse velocities must be finite")
    impulse_on_shuttle = mass * (after - before)
    return impulse_on_shuttle, -impulse_on_shuttle


def _body_id(model, body_name: str) -> int:
    body_name_to_id = getattr(model, "body_name_to_id", None)
    if body_name_to_id is not None:
        try:
            return int(body_name_to_id[body_name])
        except KeyError as exc:
            raise ValueError(f"Body not found: {body_name!r}") from exc

    if mujoco is None:  # pragma: no cover
        raise RuntimeError(f"mujoco Python package is not available: {_MUJOCO_IMPORT_ERROR}")

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name!r}")
    return body_id


def set_freejoint_linear_velocity(
    model,
    data,
    *,
    body_name: str,
    velocity_world: np.ndarray,
    free_joint_type_value: int | None = None,
) -> None:
    """Set the first three velocity DoFs of a body's free joint."""
    if free_joint_type_value is None:
        free_joint_type_value = 0
        if mujoco is not None:  # pragma: no cover
            free_joint_type_value = int(mujoco.mjtJoint.mjJNT_FREE)

    body_id = _body_id(model, body_name)
    joint_id = int(np.asarray(model.body_jntadr)[body_id])
    if joint_id < 0:
        raise ValueError(f"Body has no joint: {body_name!r}")

    joint_type = int(np.asarray(model.jnt_type)[joint_id])
    if joint_type != int(free_joint_type_value):
        raise ValueError(f"Body joint is not a free joint: {body_name!r}")

    dof_start = int(np.asarray(model.jnt_dofadr)[joint_id])
    data.qvel[dof_start : dof_start + 3] = np.asarray(velocity_world, dtype=float)
