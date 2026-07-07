"""Per-substep badminton physics pipeline for the incoming-hit environment.

Each ``substep`` applies the full realistic force set before advancing MuJoCo
by one ``mj_step``:

1. clear ``data.qfrc_applied``
2. shuttle aerodynamics (quadratic drag, angle-of-attack gain, righting moment)
3. stringbed contact force model (low-speed racket-shuttle interaction)
4. event rebound (high-speed impacts, restitution model) with a cooldown so a
   single hit is not applied repeatedly on consecutive substeps
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
        if self._cooldown > 0:
            self._cooldown -= 1
        elif should_apply_event_rebound(contact, self.cfg.impact):
            contact_point = np.array(data.site_xpos[ids["cork_site"]], dtype=float)
            racket_surface_velocity = _point_velocity_world(model, data, ids["racket"], contact_point)
            vel6 = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ids["shuttle"], vel6, 0)
            shuttle_velocity = vel6[3:]
            new_velocity, rebound_diag = compute_event_rebound(
                shuttle_velocity_world=shuttle_velocity,
                racket_surface_velocity_world=racket_surface_velocity,
                normal_world=np.asarray(contact["normal_world"], dtype=float),
                cfg=self.cfg.impact,
            )
            # Undo this substep's stringbed force so the impulse is not applied
            # twice (velocity reset + integrated force); re-apply only aero.
            data.qfrc_applied[:] = 0.0
            apply_shuttlecock_aero(model, data, self.cfg.aero)
            set_freejoint_linear_velocity(
                model,
                data,
                body_name=self.cfg.shuttle_body_name,
                velocity_world=new_velocity,
            )
            event_rebound_used = True
            self._cooldown = int(self.cfg.rebound_cooldown_substeps)

        mujoco.mj_step(model, data)
        return {
            "aero": aero_diag,
            "stringbed": contact,
            "event_rebound_used": event_rebound_used,
            "event_rebound": rebound_diag,
            "rebound_cooldown": self._cooldown,
        }
