"""Per-substep badminton physics pipeline for the incoming-hit environment.

Each ``substep`` applies the full realistic force set before advancing MuJoCo
by one ``mj_step``:

1. clear ``data.qfrc_applied``
2. shuttle aerodynamics (quadratic drag, angle-of-attack gain, righting moment)
3. stringbed contact force model (low-speed racket-shuttle interaction)
4. event rebound (high-speed impacts, restitution model) with a cooldown so a
   single hit is not applied repeatedly on consecutive substeps; continuous
   stringbed force stays suppressed throughout that cooldown so the same impact
   cannot be counted once as an event and again as a penalty-spring impulse;
   the prescribed shuttle impulse is paired with an equal/opposite point impulse
   on the rigid racket and therefore propagates into its exact-child ancestor DOFs
5. ``mujoco.mj_step``

Ground/net/cork contacts remain native MuJoCo contacts from the scene XML.
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
)
from environment.shuttlecock.src.shuttlecock_aero import (
    ShuttlecockAeroConfig,
    apply_shuttlecock_aero,
)
from environment.shuttlecock.src.shuttlecock_racket_impact import (
    ShuttlecockImpactConfig,
    compute_equal_opposite_event_impulses,
    compute_event_rebound,
    set_freejoint_linear_velocity,
    should_apply_event_rebound,
)

SHUTTLE_BODY_NAME = "overall_shuttle"
RACKET_BODY_NAME = "overall_racket"
SHUTTLE_CONTACT_SITE_NAME = "overall_cork_contact_site"


def default_aero_config() -> ShuttlecockAeroConfig:
    return ShuttlecockAeroConfig(body_name=SHUTTLE_BODY_NAME)


@dataclass
class BadmintonPhysicsConfig:
    aero: ShuttlecockAeroConfig = field(default_factory=default_aero_config)
    stringbed_geom: RacketGeometry = field(default_factory=RacketGeometry)
    stringbed_params: StringbedParams = field(default_factory=StringbedParams)
    impact: ShuttlecockImpactConfig = field(default_factory=ShuttlecockImpactConfig)
    racket_body_name: str = RACKET_BODY_NAME
    shuttle_body_name: str = SHUTTLE_BODY_NAME
    shuttle_contact_site_name: str = SHUTTLE_CONTACT_SITE_NAME
    rebound_cooldown_substeps: int = 20


def _point_velocity_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point_world: np.ndarray) -> np.ndarray:
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
    """Integrate a point impulse through one MuJoCo substep.

    MuJoCo exposes continuous generalized forces rather than an impulse input.
    Dividing by the exact physics timestep gives an equivalent one-substep
    generalized impulse. ``mj_applyFT`` maps it through every ancestor of a
    jointless exact-child racket.
    """

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


class BadmintonPhysics:
    """Stateful substep pipeline (tracks the event-rebound cooldown)."""

    def __init__(self, cfg: BadmintonPhysicsConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else BadmintonPhysicsConfig()
        self._cooldown = 0
        self._body_ids: dict[str, int] | None = None

    def reset(self) -> None:
        self._cooldown = 0

    def _resolve_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        if self._body_ids is None:
            ids = {}
            for key, name in (
                ("shuttle", self.cfg.shuttle_body_name),
                ("racket", self.cfg.racket_body_name),
            ):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError(f"missing body {name!r}")
                ids[key] = int(body_id)
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.shuttle_contact_site_name
            )
            if site_id < 0:
                raise ValueError(f"missing site {self.cfg.shuttle_contact_site_name!r}")
            ids["cork_site"] = int(site_id)
            self._body_ids = ids
        return self._body_ids

    def substep(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        ids = self._resolve_ids(model)
        data.qfrc_applied[:] = 0.0
        aero_diag = apply_shuttlecock_aero(model, data, self.cfg.aero)
        contact = apply_stringbed_force(
            model,
            data,
            racket_body_name=self.cfg.racket_body_name,
            shuttle_body_name=self.cfg.shuttle_body_name,
            shuttle_contact_site_name=self.cfg.shuttle_contact_site_name,
            geom=self.cfg.stringbed_geom,
            params=self.cfg.stringbed_params,
        )

        event_rebound_used = False
        rebound_diag = None
        impulse_on_shuttle = np.zeros(3, dtype=float)
        impulse_on_racket = np.zeros(3, dtype=float)
        racket_generalized_impulse = np.zeros(model.nv, dtype=float)
        event_velocity_before = np.zeros(3, dtype=float)
        event_velocity_after = np.zeros(3, dtype=float)
        event_racket_surface_velocity = np.zeros(3, dtype=float)
        event_normal = np.zeros(3, dtype=float)
        stringbed_force_suppressed = False
        if self._cooldown > 0:
            # The restitution event already resolves the high-speed impact.
            # The cork can remain inside the finite proxy thickness for several
            # integration substeps; retaining the penalty spring here would
            # apply a second, timestep-dependent impulse from the same hit.
            data.qfrc_applied[:] = 0.0
            apply_shuttlecock_aero(model, data, self.cfg.aero)
            stringbed_force_suppressed = True
            self._cooldown -= 1
        elif should_apply_event_rebound(contact, self.cfg.impact):
            contact_point = np.array(data.site_xpos[ids["cork_site"]], dtype=float)
            racket_surface_velocity = _point_velocity_world(model, data, ids["racket"], contact_point)
            vel6 = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ids["shuttle"], vel6, 0)
            shuttle_velocity = vel6[3:]
            event_velocity_before = np.asarray(shuttle_velocity, dtype=float).copy()
            event_racket_surface_velocity = np.asarray(racket_surface_velocity, dtype=float).copy()
            event_normal = np.asarray(contact["normal_world"], dtype=float).copy()
            new_velocity, rebound_diag = compute_event_rebound(
                shuttle_velocity_world=shuttle_velocity,
                racket_surface_velocity_world=racket_surface_velocity,
                normal_world=np.asarray(contact["normal_world"], dtype=float),
                cfg=self.cfg.impact,
            )
            event_velocity_after = np.asarray(new_velocity, dtype=float).copy()
            # Undo this substep's stringbed force so the impulse is not applied
            # twice (velocity reset + integrated force); re-apply only aero.
            data.qfrc_applied[:] = 0.0
            apply_shuttlecock_aero(model, data, self.cfg.aero)
            impulse_on_shuttle, impulse_on_racket = compute_equal_opposite_event_impulses(
                shuttle_mass_kg=float(model.body_mass[ids["shuttle"]]),
                velocity_before_world=shuttle_velocity,
                velocity_after_world=new_velocity,
            )
            racket_generalized_impulse = _apply_racket_event_reaction(
                model,
                data,
                racket_body_id=ids["racket"],
                contact_point_world=contact_point,
                impulse_on_racket_world=impulse_on_racket,
            )
            set_freejoint_linear_velocity(
                model,
                data,
                body_name=self.cfg.shuttle_body_name,
                velocity_world=new_velocity,
            )
            event_rebound_used = True
            stringbed_force_suppressed = True
            self._cooldown = int(self.cfg.rebound_cooldown_substeps)

        mujoco.mj_step(model, data)
        return {
            "aero": aero_diag,
            "stringbed": contact,
            "event_rebound_used": event_rebound_used,
            "event_rebound": rebound_diag,
            "event_impulse_on_shuttle_world_ns": impulse_on_shuttle,
            "event_impulse_on_racket_world_ns": impulse_on_racket,
            "event_reaction_generalized_impulse_ns": racket_generalized_impulse,
            "event_shuttle_velocity_before_world_m_s": event_velocity_before,
            "event_shuttle_velocity_after_world_m_s": event_velocity_after,
            "event_racket_surface_velocity_world_m_s": event_racket_surface_velocity,
            "event_stringbed_normal_world": event_normal,
            "event_stringbed_force_suppressed": bool(stringbed_force_suppressed),
            "rebound_cooldown": self._cooldown,
        }
