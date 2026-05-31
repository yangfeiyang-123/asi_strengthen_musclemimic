from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np


class StaticHitState(str, Enum):
    RESET = "RESET"
    PRE_IMPACT_FREEZE = "PRE_IMPACT_FREEZE"
    IMPACT_RELEASED = "IMPACT_RELEASED"
    FLIGHT_EVALUATION = "FLIGHT_EVALUATION"
    TERMINATED = "TERMINATED"


class FlightRegion(str, Enum):
    OWN_SIDE = "own_side"
    NET_FRONT = "net_front"
    OPPONENT_MID = "opponent_mid"
    OPPONENT_BACK = "opponent_back"
    OUT = "out"


@dataclass(frozen=True)
class StaticShuttleTarget:
    qpos_adr: int
    qvel_adr: int
    qpos: np.ndarray

    def apply_freeze(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        target_qpos = np.asarray(self.qpos, dtype=float)
        if target_qpos.shape != (7,):
            raise ValueError(f"static shuttle qpos must have shape (7,), got {target_qpos.shape}")
        qpos[self.qpos_adr : self.qpos_adr + 7] = target_qpos
        qvel[self.qvel_adr : self.qvel_adr + 6] = 0.0


def release_condition_met(
    contact: Mapping[str, object],
    *,
    phase: float,
    impact_phase: float,
    phase_tolerance: float,
) -> bool:
    if not bool(contact.get("active", False)):
        return False
    if abs(float(phase) - float(impact_phase)) > float(phase_tolerance):
        return False
    if float(contact.get("rho2", 2.0)) > 1.0:
        return False
    if float(contact.get("penetration", 0.0)) <= 0.0:
        return False
    return float(contact.get("relative_normal_velocity", 0.0)) < 0.0


def should_transition_to_flight_evaluation(
    state: StaticHitState,
    *,
    crossed_net: bool,
    landed: bool,
) -> bool:
    return state == StaticHitState.IMPACT_RELEASED and (crossed_net or landed)


def compute_static_hit_reward_terms(
    *,
    phase: float,
    impact_phase: float,
    phase_tolerance: float,
    contact_info: Mapping[str, object],
    flight_info: Mapping[str, object],
) -> dict[str, float]:
    in_phase = abs(float(phase) - float(impact_phase)) <= float(phase_tolerance)
    active = bool(contact_info.get("active", False))
    rho2 = float(contact_info.get("rho2", 2.0))
    closing_speed = max(0.0, -float(contact_info.get("relative_normal_velocity", 0.0)))
    penetration = max(0.0, float(contact_info.get("penetration", 0.0)))

    impact = 0.0
    if in_phase and active and rho2 <= 1.0 and penetration > 0.0:
        impact = min(1.0, closing_speed / 8.0) + min(0.5, penetration * 100.0)

    region = str(flight_info.get("region", FlightRegion.OWN_SIDE.value))
    crossed_net = bool(flight_info.get("crossed_net", False))
    flight = 0.0
    if region == FlightRegion.OPPONENT_BACK.value:
        flight += 1.0
    elif region == FlightRegion.OPPONENT_MID.value:
        flight += 0.5
    elif region == FlightRegion.OUT.value:
        flight -= 1.0
    elif region == FlightRegion.OWN_SIDE.value:
        flight -= 0.5
    if crossed_net:
        flight += 0.25

    early_contact_penalty = -0.25 if active and not in_phase else 0.0
    return {
        "pre_impact": 0.0,
        "impact": float(impact),
        "flight": float(flight),
        "penalty": float(early_contact_penalty),
    }


def classify_landing_region(
    landing_xy: np.ndarray,
    *,
    player_half_sign: int,
    singles: bool,
) -> FlightRegion:
    xy = np.asarray(landing_xy, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"landing_xy must have shape (2,), got {xy.shape}")
    x, y = float(xy[0]), float(xy[1])
    half_width = 2.59 if singles else 3.05
    if abs(x) > 6.70 or abs(y) > half_width:
        return FlightRegion.OUT
    if np.sign(x) == player_half_sign or abs(x) < 1e-9:
        return FlightRegion.OWN_SIDE
    opponent_depth = abs(x)
    if opponent_depth < 2.0:
        return FlightRegion.NET_FRONT
    if opponent_depth >= 5.35:
        return FlightRegion.OPPONENT_BACK
    return FlightRegion.OPPONENT_MID


class StaticForehandClearEnv:
    def __init__(
        self,
        base_env: Any,
        shuttle_target: StaticShuttleTarget,
        impact_phase: float,
        phase_tolerance: float,
        stringbed_hook: Callable[[Any, Any], Mapping[str, object]] | None = None,
        rebound_hook: Callable[[Mapping[str, object]], bool] | None = None,
        aero_hook: Callable[[Any, Any], Mapping[str, object]] | None = None,
    ) -> None:
        self.base_env = base_env
        self.shuttle_target = shuttle_target
        self.impact_phase = float(impact_phase)
        self.phase_tolerance = float(phase_tolerance)
        self.stringbed_hook = stringbed_hook
        self.rebound_hook = rebound_hook
        self.aero_hook = aero_hook
        self.state = StaticHitState.RESET
        self.release_step: int | None = None
        self.step_index = 0

    def reset(self):
        obs, base_info = self.base_env.reset()
        self.state = StaticHitState.PRE_IMPACT_FREEZE
        self.release_step = None
        self.step_index = 0
        self._freeze_shuttle()
        info = dict(base_info)
        info["state"] = self.state.value
        return obs, info

    def step(self, ctrl=None, *, phase: float, contact_info: Mapping[str, object] | None = None):
        if self.state == StaticHitState.PRE_IMPACT_FREEZE:
            self._freeze_shuttle()
            if release_condition_met(
                contact_info or {},
                phase=phase,
                impact_phase=self.impact_phase,
                phase_tolerance=self.phase_tolerance,
            ):
                self.state = StaticHitState.IMPACT_RELEASED
                self.release_step = self.step_index

        diagnostics: dict[str, object] = {}
        if self.state == StaticHitState.IMPACT_RELEASED:
            diagnostics = self._apply_released_physics(contact_info or {})

        obs, base_info = self.base_env.step(ctrl)

        if self.state == StaticHitState.PRE_IMPACT_FREEZE:
            self._freeze_shuttle()

        self.step_index += 1
        info = dict(base_info)
        info.update(diagnostics)
        info["state"] = self.state.value
        flight_info = diagnostics.get("flight", {})
        reward_terms = compute_static_hit_reward_terms(
            phase=phase,
            impact_phase=self.impact_phase,
            phase_tolerance=self.phase_tolerance,
            contact_info=contact_info or {},
            flight_info=flight_info if isinstance(flight_info, Mapping) else {},
        )
        reward = float(sum(reward_terms.values()))
        terminated = self.state == StaticHitState.TERMINATED
        truncated = False
        info["reward_terms"] = reward_terms
        return obs, reward, terminated, truncated, info

    def _freeze_shuttle(self) -> None:
        self.shuttle_target.apply_freeze(self.base_env.data.qpos, self.base_env.data.qvel)

    def _apply_released_physics(self, fallback_contact_info: Mapping[str, object]) -> dict[str, object]:
        diagnostics: dict[str, object] = {}
        rebound_contact_info = fallback_contact_info

        if self.stringbed_hook is not None:
            stringbed_info = self.stringbed_hook(self.base_env.model, self.base_env.data)
            diagnostics["stringbed"] = stringbed_info
            rebound_contact_info = stringbed_info

        if self.rebound_hook is not None:
            diagnostics["event_rebound_used"] = bool(self.rebound_hook(rebound_contact_info))

        if self.aero_hook is not None:
            diagnostics["aero"] = self.aero_hook(self.base_env.model, self.base_env.data)

        return diagnostics
