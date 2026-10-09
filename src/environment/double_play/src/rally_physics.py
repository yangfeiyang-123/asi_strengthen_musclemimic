"""Two-racket badminton substep physics for the double-play rally scene.

Same substep contract as ``BadmintonPhysics`` (aero -> stringbed -> event
rebound -> ``mj_step``) generalized to one shuttle and N rackets, each with
its own event-rebound cooldown and swept-crossing history.  The v2 physics is
on by default here: skirt cross-flow aero with fin damping, restored shuttle
inertia semantics (see ``restore_shuttle_inertia``), swept plane-crossing
detection against tunneling, speed-dependent restitution, and the eccentric
cork angular-impulse closure.

Only one racket can geometrically reach the shuttle at a time (the players
stand in opposite backcourts), so at most one event rebound is resolved per
substep; the first triggering racket in declaration order wins.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.racket.src.racket_stringbed import (
    RacketGeometry,
    StringbedParams,
    apply_stringbed_force,
    swept_stringbed_crossing,
)
from environment.shuttlecock.src.shuttlecock_aero import (
    ShuttlecockAeroConfig,
    apply_shuttlecock_aero,
    shuttlecock_aero_config_v2,
)
from environment.shuttlecock.src.shuttlecock_racket_impact import (
    ShuttlecockImpactConfig,
    compute_cork_angular_impulse_omega,
    compute_equal_opposite_event_impulses,
    compute_event_rebound,
    set_freejoint_angular_velocity,
    set_freejoint_linear_velocity,
    should_apply_event_rebound,
)

SHUTTLE_BODY_NAME = "overall_shuttle"
SHUTTLE_CONTACT_SITE_NAME = "overall_cork_contact_site"
DEFAULT_RACKET_BODY_NAMES = ("overall_racket", "p2_overall_racket")


def default_rally_aero_config() -> ShuttlecockAeroConfig:
    return shuttlecock_aero_config_v2(body_name=SHUTTLE_BODY_NAME)


def default_rally_impact_config() -> ShuttlecockImpactConfig:
    # v2 preset from params/shuttlecock_nominal.json (racket_impact.v2)
    return ShuttlecockImpactConfig(
        restitution_speed_slope_per_m_s=0.005,
        restitution_reference_speed_m_s=10.0,
        min_restitution=0.30,
    )


@dataclass
class RallyPhysicsConfig:
    aero: ShuttlecockAeroConfig = field(default_factory=default_rally_aero_config)
    stringbed_geom: RacketGeometry = field(default_factory=RacketGeometry)
    stringbed_params: StringbedParams = field(default_factory=StringbedParams)
    impact: ShuttlecockImpactConfig = field(default_factory=default_rally_impact_config)
    shuttle_body_name: str = SHUTTLE_BODY_NAME
    shuttle_contact_site_name: str = SHUTTLE_CONTACT_SITE_NAME
    racket_body_names: tuple[str, ...] = DEFAULT_RACKET_BODY_NAMES
    rebound_cooldown_substeps: int = 20
    enable_swept_crossing_detection: bool = True
    apply_cork_angular_impulse: bool = True


@dataclass
class _RacketState:
    body_id: int
    cooldown: int = 0
    prev_cork_local: np.ndarray | None = None


def _point_velocity_world(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point_world: np.ndarray
) -> np.ndarray:
    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
    omega, v_origin = vel6[:3], vel6[3:]
    origin = np.array(data.xpos[body_id], dtype=float)
    return v_origin + np.cross(omega, np.asarray(point_world, dtype=float) - origin)


def _apply_racket_event_reaction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    racket_body_id: int,
    contact_point_world: np.ndarray,
    impulse_on_racket_world: np.ndarray,
) -> np.ndarray:
    timestep = float(model.opt.timestep)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError(f"MuJoCo timestep must be positive, got {timestep}")
    generalized_force = np.zeros(model.nv, dtype=float)
    mujoco.mj_applyFT(
        model,
        data,
        np.asarray(impulse_on_racket_world, dtype=float) / timestep,
        np.zeros(3, dtype=float),
        np.asarray(contact_point_world, dtype=float),
        int(racket_body_id),
        generalized_force,
    )
    data.qfrc_applied[:] += generalized_force
    return generalized_force * timestep


class RallyBadmintonPhysics:
    """Stateful multi-racket substep pipeline (per-racket cooldown/history)."""

    def __init__(self, cfg: RallyPhysicsConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else RallyPhysicsConfig()
        if not self.cfg.racket_body_names:
            raise ValueError("RallyPhysicsConfig needs at least one racket body")
        self._shuttle_body: int | None = None
        self._cork_site: int | None = None
        self._rackets: dict[str, _RacketState] | None = None

    def reset(self) -> None:
        if self._rackets is not None:
            for state in self._rackets.values():
                state.cooldown = 0
                state.prev_cork_local = None

    def _resolve_ids(self, model: mujoco.MjModel) -> None:
        if self._rackets is not None:
            return
        shuttle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.shuttle_body_name)
        if shuttle < 0:
            raise ValueError(f"missing body {self.cfg.shuttle_body_name!r}")
        cork = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.shuttle_contact_site_name
        )
        if cork < 0:
            raise ValueError(f"missing site {self.cfg.shuttle_contact_site_name!r}")
        rackets: dict[str, _RacketState] = {}
        for name in self.cfg.racket_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"missing racket body {name!r}")
            rackets[name] = _RacketState(body_id=int(body_id))
        self._shuttle_body = int(shuttle)
        self._cork_site = int(cork)
        self._rackets = rackets

    def substep(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        self._resolve_ids(model)
        assert self._rackets is not None and self._shuttle_body is not None
        cfg = self.cfg

        data.qfrc_applied[:] = 0.0
        aero_diag = apply_shuttlecock_aero(model, data, cfg.aero)

        contact_point = np.array(data.site_xpos[self._cork_site], dtype=float)
        vel6 = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self._shuttle_body, vel6, 0
        )
        shuttle_omega = np.asarray(vel6[:3], dtype=float)
        shuttle_velocity = np.asarray(vel6[3:], dtype=float)

        racket_diags: dict[str, dict[str, Any]] = {}
        event_racket: str | None = None
        event_normal: np.ndarray | None = None
        event_swept = False

        for name, state in self._rackets.items():
            in_cooldown = state.cooldown > 0
            contact = apply_stringbed_force(
                model,
                data,
                racket_body_name=name,
                shuttle_body_name=cfg.shuttle_body_name,
                shuttle_contact_site_name=cfg.shuttle_contact_site_name,
                geom=cfg.stringbed_geom,
                params=cfg.stringbed_params,
                apply_forces=not in_cooldown,
            )
            curr_local = np.asarray(contact["p_local"], dtype=float).copy()
            diag: dict[str, Any] = {
                "stringbed": contact,
                "event_rebound_used": False,
                "swept_crossing_used": False,
                "stringbed_force_suppressed": in_cooldown,
                "cooldown": state.cooldown,
            }
            if in_cooldown:
                state.cooldown -= 1
            elif event_racket is None:
                if should_apply_event_rebound(contact, cfg.impact):
                    event_racket = name
                    event_normal = np.asarray(contact["normal_world"], dtype=float)
                elif cfg.enable_swept_crossing_detection and state.prev_cork_local is not None:
                    crossing = swept_stringbed_crossing(
                        state.prev_cork_local, curr_local, cfg.stringbed_geom
                    )
                    if bool(crossing.get("crossed", False)):
                        racket_rot = np.array(data.xmat[state.body_id], dtype=float).reshape(3, 3)
                        candidate = float(crossing["side_from"]) * racket_rot[:, 2]
                        surface_velocity = _point_velocity_world(
                            model, data, state.body_id, contact_point
                        )
                        closing = float(np.dot(shuttle_velocity - surface_velocity, candidate))
                        if closing < -cfg.impact.min_speed_for_event_m_s:
                            event_racket = name
                            event_normal = candidate
                            event_swept = True
            state.prev_cork_local = curr_local
            racket_diags[name] = diag

        event_diag: dict[str, Any] | None = None
        if event_racket is not None and event_normal is not None:
            state = self._rackets[event_racket]
            surface_velocity = _point_velocity_world(model, data, state.body_id, contact_point)
            new_velocity, rebound_diag = compute_event_rebound(
                shuttle_velocity_world=shuttle_velocity,
                racket_surface_velocity_world=surface_velocity,
                normal_world=event_normal,
                cfg=cfg.impact,
            )
            # Undo every continuous force this substep except aero, then apply
            # the event pair (velocity rewrite + opposite racket point impulse).
            data.qfrc_applied[:] = 0.0
            apply_shuttlecock_aero(model, data, cfg.aero)
            impulse_on_shuttle, impulse_on_racket = compute_equal_opposite_event_impulses(
                shuttle_mass_kg=float(model.body_mass[self._shuttle_body]),
                velocity_before_world=shuttle_velocity,
                velocity_after_world=new_velocity,
            )
            generalized_impulse = _apply_racket_event_reaction(
                model,
                data,
                racket_body_id=state.body_id,
                contact_point_world=contact_point,
                impulse_on_racket_world=impulse_on_racket,
            )
            set_freejoint_linear_velocity(
                model,
                data,
                body_name=cfg.shuttle_body_name,
                velocity_world=new_velocity,
            )
            omega_after = shuttle_omega
            if cfg.apply_cork_angular_impulse:
                omega_after = compute_cork_angular_impulse_omega(
                    omega_before_world=shuttle_omega,
                    impulse_on_shuttle_world=impulse_on_shuttle,
                    contact_point_world=contact_point,
                    com_world=np.array(data.xipos[self._shuttle_body], dtype=float),
                    inertia_diag_body_kg_m2=np.array(
                        model.body_inertia[self._shuttle_body], dtype=float
                    ),
                    inertia_rot_world=np.array(data.ximat[self._shuttle_body], dtype=float),
                    cfg=cfg.impact,
                )
                set_freejoint_angular_velocity(
                    model,
                    data,
                    body_name=cfg.shuttle_body_name,
                    omega_world=omega_after,
                )
            state.cooldown = int(cfg.rebound_cooldown_substeps)
            racket_diags[event_racket].update(
                {
                    "event_rebound_used": True,
                    "swept_crossing_used": event_swept,
                    "stringbed_force_suppressed": True,
                    "cooldown": state.cooldown,
                }
            )
            event_diag = {
                "racket": event_racket,
                "rebound": rebound_diag,
                "normal_world": np.asarray(event_normal, dtype=float).copy(),
                "shuttle_velocity_before_world_m_s": shuttle_velocity.copy(),
                "shuttle_velocity_after_world_m_s": np.asarray(new_velocity, dtype=float).copy(),
                "shuttle_omega_after_world_rad_s": np.asarray(omega_after, dtype=float).copy(),
                "racket_surface_velocity_world_m_s": np.asarray(surface_velocity, dtype=float).copy(),
                "impulse_on_shuttle_world_ns": impulse_on_shuttle,
                "impulse_on_racket_world_ns": impulse_on_racket,
                "racket_generalized_impulse_ns": generalized_impulse,
                "swept_crossing_used": event_swept,
            }

        mujoco.mj_step(model, data)
        return {
            "aero": aero_diag,
            "rackets": racket_diags,
            "event": event_diag,
            "event_racket": event_racket,
        }
