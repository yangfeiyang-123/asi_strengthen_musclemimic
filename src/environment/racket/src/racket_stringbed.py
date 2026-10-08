"""String-bed proxy forces for a MuJoCo badminton racket.

This module is intentionally independent of a full finite-element string model. It
implements a stable, tunable membrane/contact proxy that is appropriate for
robotics, SMPL, and reinforcement-learning simulations.

Coordinate convention of the racket body:
    +X: lateral direction across the string bed
    +Y: butt cap -> head tip
    +Z: normal to the string-bed plane

Typical use inside a MuJoCo step loop:
    data.qfrc_applied[:] = 0
    apply_stringbed_force(model, data, racket_body_name="racket", shuttle_body_name="shuttle")
    mujoco.mj_step(model, data)

For high-speed impacts, combine this force proxy with the event-based rebound
helper below, or reduce timestep to 0.0005 s or below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - lets Codex import the math helpers without MuJoCo installed.
    mujoco = None  # type: ignore


@dataclass
class RacketGeometry:
    """Geometric parameters for the string-bed plane."""

    stringbed_center_y: float = 0.532
    stringbed_half_width: float = 0.094
    stringbed_half_length: float = 0.1265
    stringbed_proxy_thickness: float = 0.0015


@dataclass
class StringbedParams:
    """Membrane/contact parameters for the simplified string bed."""

    # Static stiffness target: around 48 N at 5 mm center displacement => 9600 N/m.
    center_normal_stiffness: float = 9600.0
    edge_gain: float = 1.20
    max_stiffness_multiplier: float = 2.50
    normal_damping: float = 3.0
    tangential_damping: float = 0.15
    tangential_mu: float = 0.08
    cork_radius: float = 0.0135
    min_speed_for_event: float = 5.0
    event_restitution_normal: float = 0.50
    event_tangential_velocity_scale: float = 0.85


def _require_mujoco() -> None:
    if mujoco is None:
        raise ImportError("mujoco is required for MuJoCo force application functions")


def _body_id(model, name: str) -> int:
    _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {name!r}")
    return int(body_id)


def _site_id(model, name: str) -> int:
    _require_mujoco()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return int(site_id)


def _body_pose(data, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return body origin position and rotation matrix in world frame."""
    pos = np.array(data.xpos[body_id], dtype=float).copy()
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return pos, rot


def _body_velocity(model, data, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return angular and linear velocity of a body in world frame.

    MuJoCo's mj_objectVelocity returns [angular(3), linear(3)] when flg_local=0.
    """
    _require_mujoco()
    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
    return vel6[:3].copy(), vel6[3:].copy()


def _point_velocity(model, data, body_id: int, point_world: np.ndarray) -> np.ndarray:
    """Approximate world velocity of a point fixed to the body."""
    omega, v_origin = _body_velocity(model, data, body_id)
    origin = np.array(data.xpos[body_id], dtype=float)
    return v_origin + np.cross(omega, point_world - origin)


def _shuttle_contact_point(model, data, shuttle_body_id: int, shuttle_site_name: Optional[str]) -> np.ndarray:
    """Return the cork/contact point position.

    Prefer a cork site if your shuttlecock model defines one. Otherwise this uses
    the shuttle body origin/COM approximation, which is acceptable for early tests
    but should be replaced by the cork/base site for accurate off-axis torque.
    """
    if shuttle_site_name:
        site_id = _site_id(model, shuttle_site_name)
        if site_id >= 0:
            return np.array(data.site_xpos[site_id], dtype=float).copy()
    # xipos is the inertial-frame origin, usually COM. xpos is body frame origin.
    return np.array(data.xipos[shuttle_body_id], dtype=float).copy()


def local_impact_coordinates(
    racket_origin_world: np.ndarray,
    racket_rot_world: np.ndarray,
    point_world: np.ndarray,
    geom: RacketGeometry = RacketGeometry(),
) -> Tuple[np.ndarray, float]:
    """Return local point coordinates and normalized elliptical radius squared.

    rho2 <= 1 means the point projects inside the elliptical string-bed area.
    """
    p_local = racket_rot_world.T @ (point_world - racket_origin_world)
    dx = p_local[0] / geom.stringbed_half_width
    dy = (p_local[1] - geom.stringbed_center_y) / geom.stringbed_half_length
    return p_local, float(dx * dx + dy * dy)


def stringbed_stiffness_at(rho2: float, params: StringbedParams = StringbedParams()) -> float:
    """Return normal stiffness for an impact position.

    The edge multiplier is a stable heuristic: near-edge hits should feel stiffer
    because effective string spans are shorter. The multiplier is clamped to avoid
    numerical blow-ups close to the frame.
    """
    rho2_clamped = min(max(rho2, 0.0), 1.0)
    multiplier = 1.0 + params.edge_gain * rho2_clamped
    multiplier = min(multiplier, params.max_stiffness_multiplier)
    return params.center_normal_stiffness * multiplier


def stringbed_rebound_velocity(
    shuttle_velocity_world: np.ndarray,
    racket_surface_velocity_world: np.ndarray,
    normal_world: np.ndarray,
    restitution_normal: float = 0.50,
    tangential_velocity_scale: float = 0.85,
) -> np.ndarray:
    """Compute an event-style post-impact shuttle velocity.

    This does not mutate MuJoCo state. It is useful when pure soft contact misses
    a high-speed impact due to timestep limitations.
    """
    n = np.asarray(normal_world, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("normal_world must be non-zero")
    n = n / n_norm

    v_rel = np.asarray(shuttle_velocity_world, dtype=float) - np.asarray(racket_surface_velocity_world, dtype=float)
    v_n = float(np.dot(v_rel, n))
    v_t = v_rel - v_n * n

    # If v_n is positive, shuttle is already moving away along this normal.
    if v_n >= 0.0:
        return np.asarray(shuttle_velocity_world, dtype=float).copy()

    v_rel_after = (-restitution_normal * v_n) * n + tangential_velocity_scale * v_t
    return np.asarray(racket_surface_velocity_world, dtype=float) + v_rel_after


def apply_stringbed_force(
    model,
    data,
    racket_body_name: str = "racket",
    shuttle_body_name: str = "shuttle",
    shuttle_contact_site_name: Optional[str] = None,
    geom: RacketGeometry = RacketGeometry(),
    params: StringbedParams = StringbedParams(),
    apply_forces: bool = True,
) -> Dict[str, object]:
    """Apply equal/opposite string-bed contact proxy forces to racket and shuttle.

    Returns a dictionary with contact diagnostics. The returned ``active`` flag is
    False when the shuttle is outside the string bed or not penetrating the proxy
    plane. Forces are written to ``data.qfrc_applied`` using ``mj_applyFT``.
    ``apply_forces=False`` computes the same diagnostics without touching
    ``qfrc_applied`` (used while an event-rebound cooldown suppresses the spring).
    """
    _require_mujoco()
    racket_id = _body_id(model, racket_body_name)
    shuttle_id = _body_id(model, shuttle_body_name)

    racket_origin, racket_rot = _body_pose(data, racket_id)
    contact_point = _shuttle_contact_point(model, data, shuttle_id, shuttle_contact_site_name)
    p_local, rho2 = local_impact_coordinates(racket_origin, racket_rot, contact_point, geom)

    if rho2 > 1.0:
        return {"active": False, "reason": "outside_stringbed", "rho2": rho2, "p_local": p_local}

    signed_z = float(p_local[2])
    side = 1.0 if signed_z >= 0.0 else -1.0
    penetration = params.cork_radius + geom.stringbed_proxy_thickness - abs(signed_z)
    if penetration <= 0.0:
        return {
            "active": False,
            "reason": "no_penetration",
            "rho2": rho2,
            "p_local": p_local,
            "signed_z": signed_z,
            "penetration": penetration,
        }

    normal_world = side * racket_rot[:, 2]
    v_shuttle = _point_velocity(model, data, shuttle_id, contact_point)
    v_racket = _point_velocity(model, data, racket_id, contact_point)
    v_rel = v_shuttle - v_racket
    v_n = float(np.dot(v_rel, normal_world))
    v_t = v_rel - v_n * normal_world

    k_n = stringbed_stiffness_at(rho2, params)
    # Closing velocity has negative v_n, so damping increases normal force.
    f_n_mag = k_n * penetration - params.normal_damping * v_n
    f_n_mag = max(0.0, f_n_mag)

    tangential_force = -params.tangential_damping * v_t
    tangential_norm = float(np.linalg.norm(tangential_force))
    tangential_limit = params.tangential_mu * f_n_mag
    if tangential_norm > tangential_limit > 0.0:
        tangential_force *= tangential_limit / tangential_norm

    force_on_shuttle = f_n_mag * normal_world + tangential_force

    # Apply force at the cork/contact point. This creates torque if the point is off COM.
    if apply_forces:
        zero_torque = np.zeros(3, dtype=float)
        mujoco.mj_applyFT(model, data, force_on_shuttle, zero_torque, contact_point, shuttle_id, data.qfrc_applied)
        mujoco.mj_applyFT(model, data, -force_on_shuttle, zero_torque, contact_point, racket_id, data.qfrc_applied)

    return {
        "active": True,
        "rho2": rho2,
        "p_local": p_local,
        "signed_z": signed_z,
        "penetration": float(penetration),
        "normal_world": normal_world,
        "relative_normal_velocity": v_n,
        "normal_stiffness": k_n,
        "force_on_shuttle_world": force_on_shuttle,
        "normal_force_n": f_n_mag,
        "tangential_force_n": tangential_force,
    }


def should_use_event_rebound(contact_info: Dict[str, object], params: StringbedParams = StringbedParams()) -> bool:
    """Heuristic: decide whether a high-speed collision should use event rebound."""
    if not bool(contact_info.get("active", False)):
        return False
    vn = float(contact_info.get("relative_normal_velocity", 0.0))
    return vn < -params.min_speed_for_event


def swept_stringbed_crossing(
    prev_local: np.ndarray,
    curr_local: np.ndarray,
    geom: RacketGeometry = RacketGeometry(),
) -> Dict[str, object]:
    """Detect a cork path that crossed the stringbed plane between substeps.

    The penalty/event contact tests only instantaneous cork positions, so a
    closing speed above ``(2*(cork_radius+thickness))/timestep`` (about 30 m/s
    at a 1 ms step) can jump the whole capture band in one substep and tunnel
    through the bed.  This helper takes the cork position in the racket frame
    at two consecutive substeps and reports whether the straight segment
    between them pierced the elliptical bed.  The caller should then resolve
    the missed impact with the event-rebound model, using
    ``side_from * racket_z_axis`` as the impact normal (the side the cork came
    from).  The few-millimetre position overshoot left behind is not corrected;
    the reflected velocity carries the cork back out of the band.
    """
    prev_local = np.asarray(prev_local, dtype=float)
    curr_local = np.asarray(curr_local, dtype=float)
    if prev_local.shape != (3,) or curr_local.shape != (3,):
        raise ValueError("swept crossing endpoints must be three-vectors")
    z0 = float(prev_local[2])
    z1 = float(curr_local[2])
    if z0 == 0.0 or (z0 > 0.0) == (z1 > 0.0):
        return {"crossed": False, "reason": "same_side"}
    fraction = z0 / (z0 - z1)
    crossing_local = prev_local + fraction * (curr_local - prev_local)
    dx = crossing_local[0] / geom.stringbed_half_width
    dy = (crossing_local[1] - geom.stringbed_center_y) / geom.stringbed_half_length
    rho2 = float(dx * dx + dy * dy)
    if rho2 > 1.0:
        return {"crossed": False, "reason": "outside_stringbed", "rho2": rho2}
    return {
        "crossed": True,
        "fraction": float(fraction),
        "crossing_local": crossing_local,
        "rho2": rho2,
        "side_from": 1.0 if z0 > 0.0 else -1.0,
    }
