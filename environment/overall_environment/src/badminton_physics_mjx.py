"""JAX/MJX port of the badminton substep physics for GPU-parallel training.

Formula-for-formula port of ``badminton_physics.BadmintonPhysics.substep``:

1. shuttle aerodynamics (quadratic drag anchored on terminal velocity,
   angle-of-attack drag gain, pressure-center righting torque, angular damping)
2. stringbed contact force (elliptical bed, edge-stiffened normal spring with
   damping, friction-capped tangential damping) applied equal/opposite
3. event rebound for fast impacts (restitution 0.50 normal / 0.85 tangential)
   with a substep cooldown; the triggering substep cancels the stringbed force
   and keeps only aero, exactly like the CPU path
4. ``mjx.step``

All branches are ``jnp.where`` so the whole substep is jit/vmap/scan friendly.
Numerical parity with the numpy implementation is covered by
``tests/test_badminton_physics_mjx.py``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import support as mjx_support

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics import BadmintonPhysicsConfig


@dataclass(frozen=True)
class BadmintonMjxParams:
    """Static scalars baked into the jitted substep (from the numpy configs)."""

    shuttle_mass_kg: float
    gravity_m_s2: float
    terminal_velocity_m_s: float
    center_of_pressure_offset_m: float
    angle_drag_gain: float
    angular_damping: float
    max_force_n: float
    max_torque_nm: float
    # stringbed
    stringbed_center_y: float
    stringbed_half_width: float
    stringbed_half_length: float
    stringbed_proxy_thickness: float
    center_normal_stiffness: float
    edge_gain: float
    max_stiffness_multiplier: float
    normal_damping: float
    tangential_damping: float
    tangential_mu: float
    cork_radius: float
    # event rebound
    min_speed_for_event: float
    event_restitution_normal: float
    event_tangential_velocity_scale: float
    max_rebound_speed: float
    rebound_cooldown_substeps: int


@dataclass(frozen=True)
class BadmintonMjxIds:
    shuttle_body: int
    racket_body: int
    cork_site: int
    shuttle_root: int
    racket_root: int
    shuttle_dofadr: int


def make_params(model: mujoco.MjModel, cfg: BadmintonPhysicsConfig | None = None) -> BadmintonMjxParams:
    cfg = cfg if cfg is not None else BadmintonPhysicsConfig()
    shuttle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.shuttle_body_name)
    if shuttle_body < 0:
        raise ValueError(f"missing body {cfg.shuttle_body_name!r}")
    return BadmintonMjxParams(
        shuttle_mass_kg=float(model.body_mass[shuttle_body]),
        gravity_m_s2=float(np.linalg.norm(model.opt.gravity)) or 9.81,
        terminal_velocity_m_s=cfg.aero.terminal_velocity_m_s,
        center_of_pressure_offset_m=cfg.aero.center_of_pressure_offset_m,
        angle_drag_gain=cfg.aero.angle_drag_gain,
        angular_damping=cfg.aero.angular_damping_nms_per_rad,
        max_force_n=cfg.aero.max_force_n,
        max_torque_nm=cfg.aero.max_torque_nm,
        stringbed_center_y=cfg.stringbed_geom.stringbed_center_y,
        stringbed_half_width=cfg.stringbed_geom.stringbed_half_width,
        stringbed_half_length=cfg.stringbed_geom.stringbed_half_length,
        stringbed_proxy_thickness=cfg.stringbed_geom.stringbed_proxy_thickness,
        center_normal_stiffness=cfg.stringbed_params.center_normal_stiffness,
        edge_gain=cfg.stringbed_params.edge_gain,
        max_stiffness_multiplier=cfg.stringbed_params.max_stiffness_multiplier,
        normal_damping=cfg.stringbed_params.normal_damping,
        tangential_damping=cfg.stringbed_params.tangential_damping,
        tangential_mu=cfg.stringbed_params.tangential_mu,
        cork_radius=cfg.stringbed_params.cork_radius,
        min_speed_for_event=cfg.impact.min_speed_for_event_m_s,
        event_restitution_normal=cfg.impact.event_restitution_normal,
        event_tangential_velocity_scale=cfg.impact.event_tangential_velocity_scale,
        max_rebound_speed=cfg.impact.max_rebound_speed_m_s,
        rebound_cooldown_substeps=int(cfg.rebound_cooldown_substeps),
    )


def make_ids(model: mujoco.MjModel, cfg: BadmintonPhysicsConfig | None = None) -> BadmintonMjxIds:
    cfg = cfg if cfg is not None else BadmintonPhysicsConfig()
    shuttle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.shuttle_body_name)
    racket_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.racket_body_name)
    cork_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.shuttle_contact_site_name)
    if min(shuttle_body, racket_body, cork_site) < 0:
        raise ValueError("missing shuttle/racket body or cork site")
    shuttle_joint = int(model.body_jntadr[shuttle_body])
    return BadmintonMjxIds(
        shuttle_body=int(shuttle_body),
        racket_body=int(racket_body),
        cork_site=int(cork_site),
        shuttle_root=int(model.body_rootid[shuttle_body]),
        racket_root=int(model.body_rootid[racket_body]),
        shuttle_dofadr=int(model.jnt_dofadr[shuttle_joint]),
    )


def _clip_norm(vec: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    norm = jnp.linalg.norm(vec)
    scale = jnp.where(norm > max_norm, max_norm / jnp.maximum(norm, 1e-12), 1.0)
    return vec * scale


def aero_force_torque(
    p: BadmintonMjxParams,
    *,
    v_world: jnp.ndarray,
    omega_world: jnp.ndarray,
    nose_axis_world: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aero force and total torque about the COM (world frame), wind-free."""
    k = p.shuttle_mass_kg * p.gravity_m_s2 / (p.terminal_velocity_m_s**2)
    speed = jnp.linalg.norm(v_world)
    safe_speed = jnp.maximum(speed, 1e-12)
    v_hat = v_world / safe_speed
    cos_alpha = jnp.clip(jnp.dot(nose_axis_world, v_hat), -1.0, 1.0)
    sin2_alpha = jnp.maximum(0.0, 1.0 - cos_alpha * cos_alpha)
    k_eff = k * (1.0 + p.angle_drag_gain * sin2_alpha)

    force = -k_eff * speed * v_world
    force = _clip_norm(force, p.max_force_n)

    cp_offset = -p.center_of_pressure_offset_m * nose_axis_world  # cp - com
    torque = jnp.cross(cp_offset, force) - p.angular_damping * omega_world
    torque = _clip_norm(torque, p.max_torque_nm)

    moving = speed >= 1e-8
    return jnp.where(moving, force, 0.0), jnp.where(moving, torque, 0.0)


def stringbed_contact(
    p: BadmintonMjxParams,
    *,
    racket_origin: jnp.ndarray,
    racket_rot: jnp.ndarray,
    contact_point: jnp.ndarray,
    v_shuttle_point: jnp.ndarray,
    v_racket_point: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Stringbed force on the shuttle plus diagnostics (all-branchless)."""
    p_local = racket_rot.T @ (contact_point - racket_origin)
    dx = p_local[0] / p.stringbed_half_width
    dy = (p_local[1] - p.stringbed_center_y) / p.stringbed_half_length
    rho2 = dx * dx + dy * dy

    signed_z = p_local[2]
    side = jnp.where(signed_z >= 0.0, 1.0, -1.0)
    penetration = p.cork_radius + p.stringbed_proxy_thickness - jnp.abs(signed_z)
    active = (rho2 <= 1.0) & (penetration > 0.0)

    normal_world = side * racket_rot[:, 2]
    v_rel = v_shuttle_point - v_racket_point
    v_n = jnp.dot(v_rel, normal_world)
    v_t = v_rel - v_n * normal_world

    multiplier = jnp.minimum(1.0 + p.edge_gain * jnp.clip(rho2, 0.0, 1.0), p.max_stiffness_multiplier)
    k_n = p.center_normal_stiffness * multiplier
    f_n_mag = jnp.maximum(0.0, k_n * penetration - p.normal_damping * v_n)

    tangential_force = -p.tangential_damping * v_t
    tangential_norm = jnp.linalg.norm(tangential_force)
    tangential_limit = p.tangential_mu * f_n_mag
    tangential_scale = jnp.where(
        (tangential_norm > tangential_limit) & (tangential_limit > 0.0),
        tangential_limit / jnp.maximum(tangential_norm, 1e-12),
        jnp.where(tangential_limit > 0.0, 1.0, 0.0),
    )
    force_on_shuttle = f_n_mag * normal_world + tangential_force * tangential_scale
    force_on_shuttle = jnp.where(active, force_on_shuttle, 0.0)

    return {
        "active": active,
        "rho2": rho2,
        "penetration": penetration,
        "normal_world": normal_world,
        "relative_normal_velocity": v_n,
        "force_on_shuttle": force_on_shuttle,
    }


def event_rebound_velocity(
    p: BadmintonMjxParams,
    *,
    shuttle_velocity: jnp.ndarray,
    racket_surface_velocity: jnp.ndarray,
    normal_world: jnp.ndarray,
) -> jnp.ndarray:
    v_rel = shuttle_velocity - racket_surface_velocity
    v_n = jnp.dot(v_rel, normal_world)
    v_n_vec = v_n * normal_world
    v_t_vec = v_rel - v_n_vec
    rebound_rel = (
        p.event_tangential_velocity_scale * v_t_vec - p.event_restitution_normal * v_n_vec
    )
    rebound = racket_surface_velocity + rebound_rel
    rebound = _clip_norm(rebound, p.max_rebound_speed)
    return jnp.where(v_n >= 0.0, shuttle_velocity, rebound)


def _point_velocity(d: Any, body_id: int, root_id: int, point: jnp.ndarray) -> jnp.ndarray:
    """World velocity of a body-fixed point from MJX cvel (see mjx.support.jac_dot)."""
    cvel = d.cvel[body_id]
    offset = point - d.subtree_com[root_id]
    return cvel[3:] + jnp.cross(cvel[:3], offset)


def make_substep_fn(mx: Any, ids: BadmintonMjxIds, p: BadmintonMjxParams):
    """Return substep(d, cooldown) -> (d, cooldown, diag), suitable for jit/scan/vmap.

    Mirrors BadmintonPhysics.substep: aero + stringbed forces enter
    ``qfrc_applied``; a fast closing impact overrides the shuttle's linear
    velocity, cancels the stringbed force for that substep, and arms a cooldown.
    """
    def substep(d: Any, cooldown: jnp.ndarray):
        shuttle_com = d.xipos[ids.shuttle_body]
        shuttle_rot = d.xmat[ids.shuttle_body].reshape(3, 3)
        nose_axis = shuttle_rot[:, 2]
        omega = d.cvel[ids.shuttle_body][:3]
        v_com = _point_velocity(d, ids.shuttle_body, ids.shuttle_root, shuttle_com)

        aero_f, aero_t = aero_force_torque(
            p, v_world=v_com, omega_world=omega, nose_axis_world=nose_axis
        )
        qfrc_aero = mjx_support.apply_ft(
            mx, d, aero_f, aero_t, shuttle_com, jnp.asarray(ids.shuttle_body)
        )

        contact_point = d.site_xpos[ids.cork_site]
        v_shuttle_pt = _point_velocity(d, ids.shuttle_body, ids.shuttle_root, contact_point)
        v_racket_pt = _point_velocity(d, ids.racket_body, ids.racket_root, contact_point)
        contact = stringbed_contact(
            p,
            racket_origin=d.xpos[ids.racket_body],
            racket_rot=d.xmat[ids.racket_body].reshape(3, 3),
            contact_point=contact_point,
            v_shuttle_point=v_shuttle_pt,
            v_racket_point=v_racket_pt,
        )
        zero3 = jnp.zeros(3)
        qfrc_bed = mjx_support.apply_ft(
            mx, d, contact["force_on_shuttle"], zero3, contact_point, jnp.asarray(ids.shuttle_body)
        ) + mjx_support.apply_ft(
            mx, d, -contact["force_on_shuttle"], zero3, contact_point, jnp.asarray(ids.racket_body)
        )

        trigger = (
            (cooldown <= 0)
            & contact["active"]
            & (contact["relative_normal_velocity"] < -p.min_speed_for_event)
        )

        dadr = ids.shuttle_dofadr
        shuttle_qvel = jax.lax.dynamic_slice(d.qvel, (dadr,), (3,))
        new_qvel3 = event_rebound_velocity(
            p,
            shuttle_velocity=shuttle_qvel,
            racket_surface_velocity=v_racket_pt,
            normal_world=contact["normal_world"],
        )
        qvel = jnp.where(
            trigger,
            jax.lax.dynamic_update_slice(d.qvel, new_qvel3, (dadr,)),
            d.qvel,
        )
        # the rebound substep keeps only aero, exactly like the CPU path
        qfrc = jnp.where(trigger, qfrc_aero, qfrc_aero + qfrc_bed)
        cooldown = jnp.where(
            trigger,
            jnp.asarray(p.rebound_cooldown_substeps, dtype=cooldown.dtype),
            jnp.maximum(cooldown - 1, 0),
        )

        d = d.replace(qvel=qvel, qfrc_applied=qfrc)
        d = mjx.step(mx, d)
        diag = {
            "stringbed_active": contact["active"],
            "relative_normal_velocity": contact["relative_normal_velocity"],
            "event_rebound_used": trigger,
        }
        return d, cooldown, diag

    return substep


def make_batched_substep_fn(
    mx: Any,
    ids: BadmintonMjxIds,
    p: BadmintonMjxParams,
    model: mujoco.MjModel,
    *,
    vmap_mjx_step: bool = False,
):
    """Batch-native substep for semi-batched Data (MJX-Warp implicit vmap).

    All custom physics runs as plain jnp math over the leading world dimension;
    ``mjx.step`` is called directly (Warp kernels vmap over worlds internally).
    Set ``vmap_mjx_step=True`` for the classic JAX impl where Data is fully
    batched and ``mjx.step`` must be vmapped explicitly.
    """
    dof_mask_shuttle = jnp.asarray(np.asarray(model.dof_bodyid) == ids.shuttle_body)
    dof_mask_racket = jnp.asarray(np.asarray(model.dof_bodyid) == ids.racket_body)
    step_all = jax.vmap(lambda d: mjx.step(mx, d)) if vmap_mjx_step else (lambda d: mjx.step(mx, d))

    def _qfrc_from_point_force(cdof, subtree_com_root, dof_mask, force, torque, point):
        offset = point - subtree_com_root
        jacp = cdof[:, 3:] + jnp.cross(cdof[:, :3], jnp.broadcast_to(offset, cdof[:, :3].shape))
        jacr = cdof[:, :3]
        return dof_mask * (jacp @ force + jacr @ torque)

    def _point_velocity(cvel, subtree_com_root, point):
        return cvel[3:] + jnp.cross(cvel[:3], point - subtree_com_root)

    def substep(d: Any, cooldown: jnp.ndarray):
        shuttle_com = d.xipos[:, ids.shuttle_body]
        shuttle_rot = d.xmat[:, ids.shuttle_body].reshape(-1, 3, 3)
        nose_axis = shuttle_rot[:, :, 2]
        omega = d.cvel[:, ids.shuttle_body, :3]
        com_root = d.subtree_com[:, ids.shuttle_root]
        racket_root_com = d.subtree_com[:, ids.racket_root]
        v_com = jax.vmap(_point_velocity)(d.cvel[:, ids.shuttle_body], com_root, shuttle_com)

        aero_f, aero_t = jax.vmap(
            lambda v, o, n: aero_force_torque(p, v_world=v, omega_world=o, nose_axis_world=n)
        )(v_com, omega, nose_axis)

        contact_point = d.site_xpos[:, ids.cork_site]
        v_shuttle_pt = jax.vmap(_point_velocity)(d.cvel[:, ids.shuttle_body], com_root, contact_point)
        v_racket_pt = jax.vmap(_point_velocity)(d.cvel[:, ids.racket_body], racket_root_com, contact_point)
        contact = jax.vmap(
            lambda ro, rr, cp, vs, vr: stringbed_contact(
                p,
                racket_origin=ro,
                racket_rot=rr,
                contact_point=cp,
                v_shuttle_point=vs,
                v_racket_point=vr,
            )
        )(
            d.xpos[:, ids.racket_body],
            d.xmat[:, ids.racket_body].reshape(-1, 3, 3),
            contact_point,
            v_shuttle_pt,
            v_racket_pt,
        )

        cdof = d._impl.cdof
        qfrc_aero = jax.vmap(
            lambda c, s, f, t, pt: _qfrc_from_point_force(c, s, dof_mask_shuttle, f, t, pt)
        )(cdof, com_root, aero_f, aero_t, shuttle_com)
        zero3 = jnp.zeros_like(aero_f)
        qfrc_bed = jax.vmap(
            lambda c, s, f, t, pt: _qfrc_from_point_force(c, s, dof_mask_shuttle, f, t, pt)
        )(cdof, com_root, contact["force_on_shuttle"], zero3, contact_point) + jax.vmap(
            lambda c, s, f, t, pt: _qfrc_from_point_force(c, s, dof_mask_racket, f, t, pt)
        )(cdof, racket_root_com, -contact["force_on_shuttle"], zero3, contact_point)

        trigger = (
            (cooldown <= 0)
            & contact["active"]
            & (contact["relative_normal_velocity"] < -p.min_speed_for_event)
        )
        dadr = ids.shuttle_dofadr
        new_vel = jax.vmap(
            lambda sv, rv, n: event_rebound_velocity(
                p, shuttle_velocity=sv, racket_surface_velocity=rv, normal_world=n
            )
        )(d.qvel[:, dadr : dadr + 3], v_racket_pt, contact["normal_world"])
        qvel = jnp.where(
            trigger[:, None],
            d.qvel.at[:, dadr : dadr + 3].set(new_vel),
            d.qvel,
        )
        qfrc = jnp.where(trigger[:, None], qfrc_aero, qfrc_aero + qfrc_bed)
        cooldown = jnp.where(
            trigger,
            jnp.asarray(p.rebound_cooldown_substeps, cooldown.dtype),
            jnp.maximum(cooldown - 1, 0),
        )
        d = d.replace(qvel=qvel, qfrc_applied=qfrc)
        d = step_all(d)
        diag = {
            "stringbed_active": contact["active"],
            "relative_normal_velocity": contact["relative_normal_velocity"],
            "event_rebound_used": trigger,
        }
        return d, cooldown, diag

    return substep
