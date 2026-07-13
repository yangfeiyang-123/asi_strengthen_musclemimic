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

import hashlib
import json
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
        lab_controller: Any | None = None,
        lab_state_builder: Any | None = None,
        curriculum: Any | None = None,
        filter_finger_observation: bool | None = None,
        swing_duration_s: float = 1.2,
        contact_phase: float = 0.55,
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

        if lab_controller is not None and base_policy_artifact is not None:
            raise ValueError("LAB control and legacy full-action residual control are mutually exclusive")
        if (lab_controller is None) != (lab_state_builder is None):
            raise ValueError("lab_controller and lab_state_builder must be provided together")
        self.lab_controller = lab_controller
        self.lab_state_builder = lab_state_builder
        self.curriculum = curriculum
        self.filter_finger_observation = bool(
            lab_controller is not None
            if filter_finger_observation is None
            else filter_finger_observation
        )
        self.swing_duration_s = float(swing_duration_s)
        self.contact_phase = float(contact_phase)
        if self.swing_duration_s <= 0.0:
            raise ValueError("swing_duration_s must be positive")
        if not 0.0 <= self.contact_phase <= 1.0:
            raise ValueError("contact_phase must lie in [0, 1]")
        self._effective_ctrlrange_hash: str | None = None
        self._control_manifest_cache: dict[str, Any] | None = None
        if self.lab_controller is not None:
            if int(self.lab_controller.router.full_size) != int(self.model.nu):
                raise ValueError("Stage-3 LAB router does not cover every model actuator")
            if int(self.lab_state_builder.expected_state_dim) != int(
                self.lab_controller.lab_state_size
            ):
                raise ValueError("LAB state builder and latent runtime dimensions differ")
            from environment.overall_environment.src.stage3_lab import (
                apply_teacher_body_ctrlrange,
            )

            self._effective_ctrlrange_hash = apply_teacher_body_ctrlrange(
                self.model, self.lab_controller
            )

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
        self._landing_rewarded = False
        self.lab_state: np.ndarray | None = None
        self._last_lab_output: Any | None = None
        self._last_lab_input_state: np.ndarray | None = None
        self._previous_raw_latent: np.ndarray | None = None

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
        if self.filter_finger_observation:
            self._remove_finger_joint_coordinates(keep, qpos=True)
        return np.nonzero(keep)[0]

    def _build_qvel_obs_index(self) -> np.ndarray:
        keep = np.ones(self.model.nv, dtype=bool)
        keep[self._shuttle_dadr : self._shuttle_dadr + 6] = False
        if self.filter_finger_observation:
            self._remove_finger_joint_coordinates(keep, qpos=False)
        return np.nonzero(keep)[0]

    def _remove_finger_joint_coordinates(self, keep: np.ndarray, *, qpos: bool) -> None:
        from musclemimic.utils.finger_isolation import finger_joint_side

        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if finger_joint_side(name) is None:
                continue
            joint_type = int(self.model.jnt_type[joint_id])
            if qpos:
                address = int(self.model.jnt_qposadr[joint_id])
                width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (
                    4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
                )
            else:
                address = int(self.model.jnt_dofadr[joint_id])
                width = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (
                    3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
                )
            keep[address : address + width] = False

    # ---- spaces ----------------------------------------------------------

    @property
    def action_size(self) -> int:
        if self.lab_controller is not None:
            return int(self.lab_controller.task_action_size)
        return int(self.model.nu)

    @property
    def full_action_size(self) -> int:
        return int(self.model.nu)

    @property
    def expects_raw_latent(self) -> bool:
        return self.lab_controller is not None

    @property
    def control_manifest(self) -> dict[str, Any]:
        if self.lab_controller is None:
            return {"schema_version": "incoming_hit_direct_action_v1"}
        if self._control_manifest_cache is not None:
            return self._control_manifest_cache
        from environment.overall_environment.src.stage3_lab import (
            stage3_attachment_report,
        )

        payload = dict(self.lab_controller.control_manifest)
        payload["lab_state_schema_hash"] = self.lab_state_builder.schema_hash
        payload["filter_finger_observation"] = self.filter_finger_observation
        payload["racket_attachment"] = stage3_attachment_report(
            self.model, self.xml_path
        )
        payload["environment_abi"] = {
            "schema_version": "incoming_hit_environment_v1",
            "scene_sha256": hashlib.sha256(self.xml_path.read_bytes()).hexdigest(),
            "effective_ctrlrange_hash": self._effective_ctrlrange_hash,
            "full_action_size": self.full_action_size,
            "control_substeps": self.control_substeps,
            "max_episode_steps": self.max_episode_steps,
            "reward_weights": self.reward_weights,
            "player_half_sign": self.player_half_sign,
            "singles": self.singles,
            "terminate_on_body_fall": self.terminate_on_body_fall,
            "swing_duration_s": self.swing_duration_s,
            "contact_phase": self.contact_phase,
        }
        payload["curriculum"] = (
            None
            if self.curriculum is None
            else dict(vars(self.curriculum))
        )
        payload_without_hash = dict(payload)
        payload_without_hash.pop("control_hash", None)
        payload["control_hash"] = hashlib.sha256(
            json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        self._control_manifest_cache = payload
        return payload

    @property
    def control_hash(self) -> str | None:
        return self.control_manifest.get("control_hash")

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
        self._landing_rewarded = False

        mujoco.mj_forward(self.model, self.data)
        if self.lab_controller is not None:
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=self._swing_phase(),
            )
            self._last_lab_output = None
            self._last_lab_input_state = None
            self._previous_raw_latent = None
        obs = self._observation()
        return obs, self._info({})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.feed is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        swing_phase = 0.0
        applied_action = action
        if self.lab_controller is not None:
            swing_phase = self._swing_phase()
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=swing_phase,
            )
            self._last_lab_input_state = np.asarray(self.lab_state, dtype=float).copy()
            self._last_lab_output = self.lab_controller.decode_task_numpy(
                lab_state=self.lab_state,
                task_action=action,
            )
            applied_action = np.asarray(self._last_lab_output.full_action, dtype=float)
            ctrl = normalized_action_to_model_ctrl(self.model, applied_action)
        elif self.base_bridge is not None:
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
        if self.lab_controller is not None:
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=self._swing_phase(),
            )
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
            applied_action,
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
                **self._lab_diagnostics(action),
            }
        )
        return obs, reward, terminated, truncated, info

    def _swing_phase(self) -> float:
        if self.feed is None:
            return 0.0
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        start = (
            float(self.feed.intercept_time_s)
            - self.contact_phase * self.swing_duration_s
        )
        return float(np.clip((elapsed - start) / self.swing_duration_s, 0.0, 1.0))

    def _lab_diagnostics(self, raw_latent: np.ndarray) -> dict[str, Any]:
        if self._last_lab_output is None or self._last_lab_input_state is None:
            return {}
        output = self._last_lab_output
        normalizer = getattr(self.lab_controller.runtime, "normalizer", None)
        if normalizer is None:
            # Lightweight test runtimes may omit normalization.  Emit
            # non-finite diagnostics so any production promotion report fails
            # closed instead of inventing an in-distribution score.
            unclipped_state_z = np.full_like(
                np.asarray(self._last_lab_input_state, dtype=float), np.nan
            )
        else:
            unclipped_state_z = (
                np.asarray(self._last_lab_input_state, dtype=float)
                - np.asarray(normalizer.mean, dtype=float)
            ) / np.asarray(normalizer.std, dtype=float)
        task_action = np.asarray(raw_latent, dtype=float)
        raw = np.asarray(output.raw_latent, dtype=float)
        raw_rate = (
            0.0
            if self._previous_raw_latent is None
            else float(
                np.sqrt(np.mean(np.square(task_action - self._previous_raw_latent)))
            )
        )
        self._previous_raw_latent = task_action.copy()
        diagnostics = {
            "control_finite": float(
                np.all(np.isfinite(np.asarray(output.full_action, dtype=float)))
            ),
            "raw_latent_rms": float(np.sqrt(np.mean(np.square(raw)))),
            "raw_latent_saturation": float(np.mean(np.abs(raw) > 2.0)),
            "latent_norm": float(np.linalg.norm(output.latent)),
            "prior_sigma_mean": float(np.mean(output.prior_sigma)),
            "lab_state_unclipped_z_rms": float(
                np.sqrt(np.mean(np.square(unclipped_state_z)))
            ),
            "lab_state_ood_fraction": float(np.mean(np.abs(unclipped_state_z) > 5.0)),
            "body_action_rms": float(np.sqrt(np.mean(np.square(output.body_action)))),
            "right_grip_action_rms": float(
                np.sqrt(np.mean(np.square(output.right_grip_action)))
            ),
            "lambda_lab": float(output.lambda_lab),
            "raw_action_rate_rms": raw_rate,
            "normalized_control_energy": float(
                np.mean(np.square(np.asarray(output.full_action, dtype=float)))
            ),
            "body_action_saturation_fraction": float(
                np.mean(np.abs(np.asarray(output.body_action, dtype=float)) > 0.98)
            ),
            "full_action_saturation_fraction": float(
                np.mean(np.abs(np.asarray(output.full_action, dtype=float)) > 0.98)
            ),
            "muscle_power_abs_mean": float(
                np.mean(
                    np.abs(
                        np.asarray(self.data.actuator_force, dtype=float)
                        * np.asarray(self.data.actuator_velocity, dtype=float)
                    )
                )
            ),
            "lab_state_schema_hash": self.lab_state_builder.schema_hash,
            "control_hash": self.control_hash,
        }
        if output.raw_bounded_residual is not None:
            diagnostics["bounded_residual_rms"] = float(
                np.sqrt(np.mean(np.square(output.raw_bounded_residual)))
            )
        return diagnostics

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
        # Task policy and frozen LAB prior must share the same intercept-aligned
        # swing clock.  Episode progress is not a swing phase when feed timing
        # varies and previously made the task policy condition on a conflicting
        # signal.
        phase = self._swing_phase()
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

        if (
            self._landing_region is not None
            and self.termination_reason == "landed"
            and self._hit_rewarded
            and not self._landing_rewarded
        ):
            terms["landing_region"] = w["landing_region"] * REGION_SCORES.get(self._landing_region, 0.0)
            self._landing_rewarded = True

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
        if self._landing_region is not None:
            info["landing_region"] = self._landing_region
        info.update(extra)
        return info
