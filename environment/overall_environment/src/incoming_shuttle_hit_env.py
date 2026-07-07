"""Incoming-shuttle hit RL environment.

A feeder launches a shuttle from the opposite half court toward the player,
who stands at the center of their own half with the racket welded to the right
hand. The policy drives the full muscle actuator set and is rewarded for
intercepting the shuttle with the string bed and returning it over the net
into the opponent court.

This environment is intentionally independent from the musclemimic trajectory
tracking pipeline: it owns its MuJoCo model/data and runs the badminton
physics substep loop (aero + stringbed + event rebound) itself.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics import (
    BadmintonPhysics,
    BadmintonPhysicsConfig,
)
from environment.overall_environment.src.control_scaling import normalized_action_to_model_ctrl
from environment.overall_environment.src.shuttle_feeder import (
    FeedConfig,
    FeedSample,
    HitWindow,
    launch_quat_from_velocity,
    sample_feed,
)
from environment.overall_environment.src.static_forehand_clear_env import (
    FlightRegion,
    classify_landing_region,
)

READY_KEYFRAME = "overall_ready"
HUMAN_ROOT_FREEJOINT = "root"
SHUTTLE_FREEJOINT = "overall_shuttle_free"
RACKET_FREEJOINT = "overall_racket_free"
STRINGBED_CENTER_SITE = "overall_stringbed_center_site"
PALM_SITE = "rh_palm_grip_site"
GROUND_REST_HEIGHT_M = 0.035
BODY_FALL_ROOT_HEIGHT_M = 0.55

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "approach": 1.0,
    "hit_bonus": 5.0,
    "crossed_net": 2.0,
    "landing_region": 4.0,
    "effort": 0.01,
    "posture": 0.5,
    "body_fall": 10.0,
    # residual-mode extra: penalty on deviation from the frozen base swing
    "residual": 0.0,
}

REGION_SCORES: dict[str, float] = {
    FlightRegion.OPPONENT_BACK.value: 1.0,
    FlightRegion.OPPONENT_MID.value: 0.5,
    FlightRegion.NET_FRONT.value: 0.2,
    FlightRegion.OWN_SIDE.value: -0.5,
    FlightRegion.OUT.value: -1.0,
}


class IncomingHitState(str, Enum):
    INCOMING = "INCOMING"
    HIT = "HIT"
    FLIGHT = "FLIGHT"
    DONE = "DONE"


def _validate_reward_weights(weights: dict[str, float]) -> dict[str, float]:
    merged = dict(DEFAULT_REWARD_WEIGHTS)
    unknown = set(weights) - set(DEFAULT_REWARD_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown reward weight keys: {sorted(unknown)}")
    for key, value in weights.items():
        merged[key] = float(value)
    return merged


class IncomingShuttleHitEnv:
    def __init__(
        self,
        xml: str | Path,
        *,
        feed_bank: list[FeedSample] | None = None,
        feed_config: FeedConfig | None = None,
        hit_window: HitWindow | None = None,
        physics_config: BadmintonPhysicsConfig | None = None,
        control_substeps: int = 10,
        max_episode_steps: int = 300,
        reward_weights: dict[str, float] | None = None,
        player_half_sign: int = -1,
        singles: bool = True,
        terminate_on_body_fall: bool = True,
        base_policy_artifact: str | Path | None = None,
        residual_scale: float = 0.3,
        base_skill: str | None = None,
        seed: int = 0,
    ) -> None:
        self.xml_path = Path(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.physics = BadmintonPhysics(physics_config)
        self.feed_bank = feed_bank
        self.feed_config = feed_config if feed_config is not None else FeedConfig()
        self.hit_window = hit_window if hit_window is not None else HitWindow()
        self.control_substeps = int(control_substeps)
        self.max_episode_steps = int(max_episode_steps)
        if self.control_substeps <= 0:
            raise ValueError(f"control_substeps must be positive, got {control_substeps}")
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")
        self.reward_weights = _validate_reward_weights(reward_weights or {})
        self.player_half_sign = int(player_half_sign)
        self.singles = bool(singles)
        self.terminate_on_body_fall = bool(terminate_on_body_fall)
        self.rng = np.random.default_rng(seed)

        self.base_bridge = None
        if base_policy_artifact is not None:
            from environment.overall_environment.src.base_swing_bridge import BaseSwingBridge

            self.base_bridge = BaseSwingBridge(
                base_policy_artifact, self.model, residual_scale=residual_scale, skill=base_skill
            )

        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, READY_KEYFRAME)
        if self.keyframe_id < 0:
            raise ValueError(f"missing keyframe {READY_KEYFRAME!r} in {self.xml_path}")
        self._root_qadr = self._joint_qposadr(HUMAN_ROOT_FREEJOINT)
        self._shuttle_qadr = self._joint_qposadr(SHUTTLE_FREEJOINT)
        self._shuttle_dadr = self._joint_dofadr(SHUTTLE_FREEJOINT)
        self._stringbed_site = self._site_id(STRINGBED_CENTER_SITE)
        self._palm_site = self._site_id(PALM_SITE)
        self._racket_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
        if self._racket_body < 0:
            raise ValueError("missing body 'overall_racket'")
        self._qpos_obs_index = self._build_qpos_obs_index()
        self._qvel_obs_index = self._build_qvel_obs_index()

        self.state = IncomingHitState.INCOMING
        self.step_index = 0
        self.termination_reason: str | None = None
        self.feed: FeedSample | None = None
        self._hit_closing_speed = 0.0
        self._hit_rewarded = False
        self._crossed_net_rewarded = False
        self._landing_region: str | None = None

    # ---- id helpers -----------------------------------------------------

    def _joint_qposadr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name!r}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dofadr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name!r}")
        return int(self.model.jnt_dofadr[joint_id])

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"missing site {name!r}")
        return int(site_id)

    def _build_qpos_obs_index(self) -> np.ndarray:
        keep = np.ones(self.model.nq, dtype=bool)
        keep[self._shuttle_qadr : self._shuttle_qadr + 7] = False  # replaced by relative features
        keep[self._root_qadr : self._root_qadr + 2] = False  # drop absolute root x/y
        return np.nonzero(keep)[0]

    def _build_qvel_obs_index(self) -> np.ndarray:
        keep = np.ones(self.model.nv, dtype=bool)
        keep[self._shuttle_dadr : self._shuttle_dadr + 6] = False
        return np.nonzero(keep)[0]

    # ---- spaces ----------------------------------------------------------

    @property
    def action_size(self) -> int:
        return int(self.model.nu)

    @property
    def observation_size(self) -> int:
        return int(self._qpos_obs_index.size + self._qvel_obs_index.size + 12 + 9 + 8)

    # ---- core API --------------------------------------------------------

    def reset(self, *, feed_index: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        if self.feed_bank:
            if feed_index is None:
                feed_index = int(self.rng.integers(len(self.feed_bank)))
            self.feed = self.feed_bank[feed_index % len(self.feed_bank)]
        else:
            self.feed = sample_feed(self.rng, self.feed_config, self.hit_window)

        qadr, dadr = self._shuttle_qadr, self._shuttle_dadr
        self.data.qpos[qadr : qadr + 3] = self.feed.launch_pos
        self.data.qpos[qadr + 3 : qadr + 7] = launch_quat_from_velocity(self.feed.launch_vel)
        self.data.qvel[dadr : dadr + 3] = self.feed.launch_vel
        self.data.qvel[dadr + 3 : dadr + 6] = 0.0

        self.physics.reset()
        self.state = IncomingHitState.INCOMING
        self.step_index = 0
        self.termination_reason = None
        self._hit_closing_speed = 0.0
        self._hit_rewarded = False
        self._crossed_net_rewarded = False
        self._landing_region = None

        mujoco.mj_forward(self.model, self.data)
        obs = self._observation()
        return obs, self._info({})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.feed is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=float)
        swing_phase = 0.0
        if self.base_bridge is not None:
            elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
            swing_phase = self.base_bridge.phase_config.phase_at(
                elapsed, float(self.feed.intercept_time_s)
            )
            combined, _base = self.base_bridge.combined_action(
                self.model, self.data, action, phase=swing_phase
            )
            ctrl = normalized_action_to_model_ctrl(self.model, combined)
        else:
            ctrl = normalized_action_to_model_ctrl(self.model, action)
        self.data.ctrl[:] = ctrl

        hit_this_step = False
        rebound_this_step = False
        max_closing_speed = 0.0
        for _ in range(self.control_substeps):
            diag = self.physics.substep(self.model, self.data)
            contact = diag["stringbed"]
            closing = max(0.0, -float(contact.get("relative_normal_velocity", 0.0)))
            if bool(diag["event_rebound_used"]):
                rebound_this_step = True
                hit_this_step = True
                max_closing_speed = max(max_closing_speed, closing)
            elif bool(contact.get("active", False)) and closing > 0.0:
                hit_this_step = True
                max_closing_speed = max(max_closing_speed, closing)

        self.step_index += 1
        if self.state == IncomingHitState.INCOMING and hit_this_step:
            self.state = IncomingHitState.HIT
            self._hit_closing_speed = max_closing_speed

        flight = self._flight_info()
        if self.state == IncomingHitState.HIT and bool(flight["crossed_net"]):
            self.state = IncomingHitState.FLIGHT

        terminated = False
        if bool(flight["landed"]):
            terminated = True
            if self.state == IncomingHitState.INCOMING:
                self.termination_reason = "miss"
            else:
                self.termination_reason = "landed"
                self._landing_region = str(flight["region"])
            self.state = IncomingHitState.DONE

        body_fall = self._root_height() < BODY_FALL_ROOT_HEIGHT_M
        if body_fall and self.terminate_on_body_fall and not terminated:
            terminated = True
            self.termination_reason = "body_fall"
            self.state = IncomingHitState.DONE

        obs = self._observation()
        if not np.isfinite(obs).all():
            terminated = True
            self.termination_reason = "non_finite"
            self.state = IncomingHitState.DONE
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        truncated = False
        if not terminated and self.step_index >= self.max_episode_steps:
            truncated = True
            self.termination_reason = "time_limit"
            self.state = IncomingHitState.DONE

        reward_terms = self._reward_terms(
            action,
            flight=flight,
            hit_this_step=hit_this_step,
            body_fall=body_fall,
        )
        reward = float(sum(reward_terms.values()))

        info = self._info(
            {
                "reward_terms": reward_terms,
                "flight": flight,
                "hit_this_step": hit_this_step,
                "event_rebound_this_step": rebound_this_step,
                "hit_closing_speed_m_s": self._hit_closing_speed,
                "body_fall": bool(body_fall),
                "landing_region": self._landing_region,
                "swing_phase": swing_phase,
            }
        )
        return obs, reward, terminated, truncated, info

    # ---- observation -----------------------------------------------------

    def _observation(self) -> np.ndarray:
        data = self.data
        qpos = np.asarray(data.qpos, dtype=float)[self._qpos_obs_index]
        qvel = np.asarray(data.qvel, dtype=float)[self._qvel_obs_index]

        root_pos = np.asarray(data.qpos[self._root_qadr : self._root_qadr + 3], dtype=float)
        shuttle_pos = np.asarray(data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3], dtype=float)
        shuttle_vel = np.asarray(data.qvel[self._shuttle_dadr : self._shuttle_dadr + 3], dtype=float)
        stringbed_pos = np.asarray(data.site_xpos[self._stringbed_site], dtype=float)
        stringbed_mat = np.asarray(data.site_xmat[self._stringbed_site], dtype=float).reshape(3, 3)
        face_normal = stringbed_mat[:, 2]
        face_vel = self._stringbed_velocity()

        shuttle_features = np.concatenate(
            [
                shuttle_pos - root_pos,
                shuttle_vel,
                shuttle_pos - stringbed_pos,
                shuttle_vel - face_vel,
            ]
        )
        racket_features = np.concatenate([stringbed_pos - root_pos, face_normal, face_vel])

        intercept = np.asarray(self.feed.intercept_point, dtype=float) if self.feed is not None else np.zeros(3)
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        time_to_intercept = max(0.0, (self.feed.intercept_time_s if self.feed is not None else 0.0) - elapsed)
        phase = min(1.0, self.step_index / max(self.max_episode_steps - 1, 1))
        task_features = np.concatenate(
            [
                intercept - stringbed_pos,
                intercept - root_pos,
                [time_to_intercept, phase],
            ]
        )
        return np.concatenate([qpos, qvel, shuttle_features, racket_features, task_features])

    def _stringbed_velocity(self) -> np.ndarray:
        vel6 = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._racket_body, vel6, 0
        )
        omega, v_origin = vel6[:3], vel6[3:]
        origin = np.asarray(self.data.xpos[self._racket_body], dtype=float)
        point = np.asarray(self.data.site_xpos[self._stringbed_site], dtype=float)
        return v_origin + np.cross(omega, point - origin)

    # ---- reward / termination helpers -------------------------------------

    def _reward_terms(
        self,
        action: np.ndarray,
        *,
        flight: dict[str, Any],
        hit_this_step: bool,
        body_fall: bool,
    ) -> dict[str, float]:
        w = self.reward_weights
        terms = {key: 0.0 for key in DEFAULT_REWARD_WEIGHTS}

        if self.state == IncomingHitState.INCOMING and self.feed is not None:
            stringbed_pos = np.asarray(self.data.site_xpos[self._stringbed_site], dtype=float)
            dist = float(np.linalg.norm(self.feed.intercept_point - stringbed_pos))
            terms["approach"] = w["approach"] * float(np.exp(-2.0 * dist))

        if hit_this_step and not self._hit_rewarded:
            self._hit_rewarded = True
            terms["hit_bonus"] = w["hit_bonus"] * min(1.0, self._hit_closing_speed / 8.0)

        if (
            self.state in (IncomingHitState.FLIGHT, IncomingHitState.DONE)
            and self._hit_rewarded
            and bool(flight["crossed_net"])
            and not self._crossed_net_rewarded
        ):
            self._crossed_net_rewarded = True
            terms["crossed_net"] = w["crossed_net"]

        if self._landing_region is not None and self.termination_reason == "landed" and self._hit_rewarded:
            terms["landing_region"] = w["landing_region"] * REGION_SCORES.get(self._landing_region, 0.0)
            self._landing_region = None  # score once

        terms["effort"] = -w["effort"] * float(np.mean(np.square(action)))
        if self.base_bridge is not None and w.get("residual", 0.0) != 0.0:
            terms["residual"] = -w["residual"] * float(np.mean(np.square(action)))
        terms["posture"] = -w["posture"] * max(0.0, 0.85 - self._root_height())
        if body_fall:
            terms["body_fall"] = -w["body_fall"]
        return terms

    def _root_height(self) -> float:
        return float(self.data.qpos[self._root_qadr + 2])

    def _flight_info(self) -> dict[str, Any]:
        shuttle_pos = np.asarray(
            self.data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3], dtype=float
        )
        shuttle_vel = np.asarray(
            self.data.qvel[self._shuttle_dadr : self._shuttle_dadr + 3], dtype=float
        )
        landed = bool(shuttle_pos[2] <= GROUND_REST_HEIGHT_M)
        crossed_net = bool(
            np.sign(shuttle_pos[0]) == self.player_half_sign * -1 and abs(shuttle_pos[0]) > 1e-9
        )
        region = classify_landing_region(
            shuttle_pos[:2],
            player_half_sign=self.player_half_sign,
            singles=self.singles,
        )
        return {
            "shuttle_xyz": shuttle_pos.copy(),
            "shuttle_velocity": shuttle_vel.copy(),
            "crossed_net": crossed_net,
            "landed": landed,
            "region": region.value,
        }

    def _info(self, extra: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state": self.state.value,
            "step_count": self.step_index,
            "feed_intercept_point": None
            if self.feed is None
            else np.asarray(self.feed.intercept_point, dtype=float).copy(),
            "feed_intercept_time_s": None if self.feed is None else float(self.feed.intercept_time_s),
        }
        if self.termination_reason is not None:
            info["termination_reason"] = self.termination_reason
        info.update(extra)
        return info
