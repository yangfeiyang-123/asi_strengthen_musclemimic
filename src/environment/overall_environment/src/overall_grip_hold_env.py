from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter
from environment.overall_environment.src.action_manifest import reconstruct_action_manifest
from environment.overall_environment.src.body_obs_adapter import BodyObsAdapter
from environment.overall_environment.src.body_tracking_reward import (
    BodyTrackingReport,
    reset_snapshot_body_tracking_error,
    trajectory_body_tracking_error,
)
from environment.overall_environment.src.control_scaling import (
    CtrlRangeOverrideReport,
    apply_checkpoint_ctrl_ranges_to_model,
    normalized_action_to_model_ctrl,
)
from environment.overall_environment.src.frozen_body_policy import (
    FrozenBodyPolicy,
    load_frozen_body_policy_manifest,
)
from environment.overall_environment.src.layered_control import resolve_actuator_groups
from environment.overall_environment.src.layered_control import actuator_names_from_model
from environment.overall_environment.src.musclemimic_scene_env import MuscleMimicSceneEnv
from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider
from environment.overall_environment.src.training_scene import (
    build_training_scene_report,
    validate_training_scene_report,
)


DEFAULT_REWARD_WEIGHTS = {
    "mimic_body": 0.2,
    "grip_site": 8.0,
    "contact": 1.0,
    "no_slip": 8.0,
    "no_penetration": 10.0,
    "racket_hand_pose": 4.0,
    "residual_effort": 0.01,
}
DEFAULT_BODY_CONTROL_SUBSTEPS = 10
REWARD_TERM_NAMES = (
    "r_mimic_body",
    "r_grip_site",
    "r_contact",
    "r_no_slip",
    "r_no_penetration",
    "r_racket_hand_pose",
    "r_residual_effort",
)


class OverallGripHoldEnv:
    """No-shuttle grip-hold environment on the overall training scene.

    Without a frozen body policy artifact, the base body action is zero and the
    residual action controls the configured right-hand actuator groups.
    """

    def __init__(
        self,
        xml: str | Path,
        *,
        residual_groups: list[str] | tuple[str, ...] = ("right_hand_fingers",),
        control_substeps: int = DEFAULT_BODY_CONTROL_SUBSTEPS,
        max_episode_steps: int = 200,
        reward_weights: dict[str, float] | None = None,
        body_policy_artifact: str | Path | None = None,
        body_checkpoint: str | Path | None = None,
        body_goal_obs: np.ndarray | None = None,
        pose_servo: bool = False,
        servo_scope: str = "none",
        base_runtime: str = "musclemimic",
    ) -> None:
        self.xml_path = Path(xml)
        report = build_training_scene_report(self.xml_path)
        validate_training_scene_report(report)
        preloaded_manifest = None
        base_body_checkpoint = Path(body_checkpoint) if body_checkpoint is not None else None
        if base_body_checkpoint is None and body_policy_artifact is not None:
            preloaded_manifest = load_frozen_body_policy_manifest(Path(body_policy_artifact))
            base_body_checkpoint = Path(preloaded_manifest.source_checkpoint)
        if base_runtime not in {"musclemimic", "raw_mujoco"}:
            raise ValueError(f"unsupported base_runtime: {base_runtime!r}")
        self.base_runtime = base_runtime
        if base_runtime == "musclemimic":
            self.base_env = MuscleMimicSceneEnv(
                self.xml_path,
                disable_fingers=False,
                body_checkpoint=base_body_checkpoint,
            )
        else:
            self.base_env = OverallBadmintonEnvironment(self.xml_path)
        self.model = self.base_env.model
        self.data = self.base_env.data
        if base_runtime == "musclemimic":
            self.control_substeps = int(self.base_env.control_substeps)
        else:
            self.control_substeps = _positive_int(control_substeps, "control_substeps")
        self.max_episode_steps = _positive_int(max_episode_steps, "max_episode_steps")
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        _validate_reward_weights(self.reward_weights)
        self.pose_servo = bool(pose_servo)
        self.servo_scope = str(servo_scope)
        if self.servo_scope not in {"none", "all_debug"}:
            raise ValueError(f"unsupported servo_scope: {self.servo_scope!r}")
        if self.pose_servo and self.servo_scope == "none":
            self.servo_scope = "all_debug"
        if not self.pose_servo and self.servo_scope != "none":
            raise ValueError("servo_scope must be 'none' when pose_servo is disabled")

        self.residual_actuator_names = resolve_actuator_groups(self.model, list(residual_groups))
        if not self.residual_actuator_names:
            raise ValueError(f"residual_groups produced no actuators: {list(residual_groups)}")
        self.residual_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.residual_actuator_names
            ],
            dtype=int,
        )
        if np.any(self.residual_actuator_ids < 0):
            raise ValueError(f"missing residual actuators: {self.residual_actuator_names}")

        self.palm_site_id = _site_id(self.model, "rh_palm_grip_site")
        self.grip_site_id = _site_id(self.model, "overall_grip_pose_site")
        self.racket_body_id = _body_id(self.model, "overall_racket")
        self.root_joint_id = _joint_id(self.model, "root")
        self.handle_geom_ids = _handle_geom_ids(self.model)
        self.right_hand_geom_ids = _right_hand_geom_ids(self.model)
        if not self.handle_geom_ids:
            raise ValueError("overall scene has no handle contact geoms")
        if not self.right_hand_geom_ids:
            raise ValueError("overall scene has no right-hand contact geoms")
        self._ready_grip_to_palm = _ready_grip_to_palm_vector(self.xml_path)
        self._ready_object_free_joint_states = _ready_object_free_joint_states(self.xml_path)
        self._ready_root_xy = _ready_free_joint_xy(self.xml_path, "root")

        self._step_count = 0
        self._reference_qpos = np.array(self.model.qpos0, dtype=float)
        self._reference_grip_to_palm = np.zeros(3, dtype=float)
        self._reference_palm_to_grip_distance = 0.0
        self._last_body_obs_size = 0
        self._last_body_action_size = 0
        self._last_raw_body_action_max_abs = 0.0
        self._last_clipped_full_ctrl_max_abs = 0.0
        self._last_servo_force_norm_max = 0.0
        self.body_ctrl_range_report = CtrlRangeOverrideReport(0, 0, 0, ())
        self._last_reset_mode = "keyframe"
        self._last_reset_traj_step = 0

        self.body_policy_source = "fake_zero"
        self.body_policy: FrozenBodyPolicy | None = None
        self.body_obs_adapter: BodyObsAdapter | None = None
        self.body_action_adapter: CheckpointToFullActionAdapter | None = None
        self.body_goal_provider: TrajectoryGoalProvider | None = None
        self.body_goal_obs = np.zeros(0, dtype=float)
        self.body_goal_obs_source = "none"
        if body_policy_artifact is not None:
            artifact_path = Path(body_policy_artifact)
            manifest = preloaded_manifest or load_frozen_body_policy_manifest(artifact_path)
            checkpoint_path = Path(body_checkpoint) if body_checkpoint is not None else Path(manifest.source_checkpoint)
            self.body_policy = FrozenBodyPolicy.load_from_export(artifact_path)
            self.body_obs_adapter = BodyObsAdapter.from_checkpoint(checkpoint_path)
            action_manifest = reconstruct_action_manifest(checkpoint_path)
            self.body_ctrl_range_report = apply_checkpoint_ctrl_ranges_to_model(self.model, checkpoint_path)
            self.body_action_adapter = CheckpointToFullActionAdapter(
                list(action_manifest.actuator_names),
                actuator_names_from_model(self.model),
            )
            goal_size = int(self.body_obs_adapter.schema.goal_size) if self.body_obs_adapter.schema is not None else 0
            if body_goal_obs is None:
                self.body_goal_provider = TrajectoryGoalProvider.from_checkpoint(checkpoint_path)
                self.body_goal_obs = np.zeros(goal_size, dtype=float)
                self.body_goal_obs_source = self.body_goal_provider.source
            else:
                goal = np.asarray(body_goal_obs, dtype=float)
                if goal.shape != (goal_size,):
                    raise ValueError(f"body_goal_obs must have shape ({goal_size},), got {goal.shape}")
                if not np.isfinite(goal).all():
                    raise ValueError("body_goal_obs must be finite")
                self.body_goal_obs = goal.copy()
                self.body_goal_obs_source = "provided"
            self.body_policy_source = "frozen_artifact"

    @property
    def action_size(self) -> int:
        return int(self.residual_actuator_ids.size)

    def reset(self, *, traj_step: int = 0, reset_mode: str = "auto") -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        if reset_mode not in {"auto", "keyframe", "trajectory"}:
            raise ValueError(f"unsupported reset_mode: {reset_mode!r}")
        obs, base_info = self.base_env.reset()
        keyframe_grip_to_palm = self._ready_grip_to_palm
        restore_object_free_joints(self.model, self.data, self._ready_object_free_joint_states)
        use_trajectory = reset_mode == "trajectory" or (reset_mode == "auto" and self.body_goal_provider is not None)
        self._last_reset_mode = "trajectory" if use_trajectory else "keyframe"
        self._last_reset_traj_step = 0
        if self.body_goal_provider is not None:
            self.body_goal_provider.reset(traj_step=traj_step if use_trajectory else 0)
            if use_trajectory:
                reference = self.body_goal_provider.apply_reference_state_to(
                    self.model,
                    self.data,
                    traj_step=traj_step,
                )
                set_free_joint_xy(self.model, self.data, "root", self._ready_root_xy)
                restore_object_free_joints(self.model, self.data, self._ready_object_free_joint_states)
                align_racket_grip_to_palm(
                    self.model,
                    self.data,
                    reference_grip_to_palm=keyframe_grip_to_palm,
                )
                self._last_reset_traj_step = int(reference.traj_step)
                if self.base_runtime == "musclemimic":
                    obs = self.base_env.sync_observation()
                    base_info = self.base_env._info(obs)
                else:
                    self.base_env._servo_qpos = np.array(self.data.qpos, dtype=float)
                    obs = np.concatenate([np.array(self.data.qpos, dtype=float), np.array(self.data.qvel, dtype=float)])
                    base_info = self.base_env._info()
        self._reference_qpos = np.array(self.data.qpos, dtype=float)
        self._reference_grip_to_palm = self._grip_to_palm_vector()
        self._reference_palm_to_grip_distance = float(np.linalg.norm(self._reference_grip_to_palm))
        info = self._info()
        info.update(base_info)
        info["scene_action_size"] = int(self.model.nu)
        info.update(self._body_policy_info())
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("action must be finite")

        clipped_action = np.clip(action_array, -1.0, 1.0)
        full_action = self._body_policy_action()
        full_action[self.residual_actuator_ids] += clipped_action
        full_action = np.clip(full_action, -1.0, 1.0)
        self._last_clipped_full_ctrl_max_abs = float(np.max(np.abs(full_action))) if full_action.size else 0.0
        obs = np.zeros(self.model.nq + self.model.nv, dtype=float)
        base_info: dict[str, Any] = {}
        self._last_servo_force_norm_max = 0.0
        if self.base_runtime == "musclemimic":
            obs, _, base_terminated, base_truncated, base_info = self.base_env.step_action(full_action)
            self._last_servo_force_norm_max = 0.0
        else:
            full_ctrl = normalized_action_to_model_ctrl(self.model, full_action)
            self._last_clipped_full_ctrl_max_abs = float(np.max(np.abs(full_ctrl))) if full_ctrl.size else 0.0
            base_terminated = False
            base_truncated = False
            for _ in range(self.control_substeps):
                obs, base_info = self.base_env.step(ctrl=full_ctrl, pose_servo=self.pose_servo)
                self._last_servo_force_norm_max = max(
                    self._last_servo_force_norm_max,
                    float(np.linalg.norm(self.data.qfrc_applied)),
                )
        if self.body_goal_provider is not None:
            self.body_goal_provider.advance()
        self._step_count += 1

        info = self._info()
        info.update(base_info)
        info["scene_action_size"] = int(self.model.nu)
        info.update(self._body_policy_info())
        reward_terms = self._reward_terms(clipped_action, info)
        reward = float(sum(reward_terms.values()))
        if not math.isfinite(reward):
            raise FloatingPointError(f"non-finite reward from terms: {reward_terms}")
        info["reward_terms"] = reward_terms

        finite = bool(np.isfinite(obs).all())
        terminated = bool(base_terminated) or (not finite) or bool(info["racket_drop"]) or bool(info["body_fall"])
        truncated = bool(base_truncated) or self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _body_policy_action(self) -> np.ndarray:
        if self.body_policy is None:
            self._last_body_obs_size = 0
            self._last_body_action_size = 0
            self._last_raw_body_action_max_abs = 0.0
            return np.zeros(self.model.nu, dtype=float)
        if self.body_obs_adapter is None or self.body_action_adapter is None:
            raise RuntimeError("frozen body policy is missing adapter state")

        goal_obs = self.body_goal_obs
        if self.body_goal_provider is not None:
            goal_obs = self.body_goal_provider.build(self.model, self.data)
        body_obs = self.body_obs_adapter.build_from_mujoco(
            self.model,
            self.data,
            goal_obs=goal_obs,
        )
        body_action = self.body_policy.act(body_obs)
        full_action = self.body_action_adapter.adapt(body_action)
        self._last_body_obs_size = int(body_obs.size)
        self._last_body_action_size = int(body_action.size)
        self._last_raw_body_action_max_abs = float(np.max(np.abs(body_action))) if body_action.size else 0.0
        return full_action

    def _body_policy_info(self) -> dict[str, Any]:
        return {
            "body_policy_source": self.body_policy_source,
            "body_obs_size": int(self._last_body_obs_size),
            "body_action_size": int(self._last_body_action_size),
            "body_goal_obs_size": int(self.body_goal_provider.goal_size)
            if self.body_goal_provider is not None
            else int(self.body_goal_obs.size),
            "body_goal_obs_source": self.body_goal_obs_source,
            "body_goal_motion_path": self.body_goal_provider.motion_path if self.body_goal_provider is not None else None,
            "body_goal_traj_step": self.body_goal_provider.last_built_step if self.body_goal_provider is not None else 0,
            "body_goal_next_traj_step": self.body_goal_provider.traj_step if self.body_goal_provider is not None else 0,
            "body_goal_traj_len": self.body_goal_provider.traj_len if self.body_goal_provider is not None else 0,
            "raw_body_action_max_abs": float(self._last_raw_body_action_max_abs),
            "clipped_full_ctrl_max_abs": float(self._last_clipped_full_ctrl_max_abs),
            "body_ctrl_range_overrides": asdict(self.body_ctrl_range_report),
            "pose_servo_enabled": bool(self.pose_servo),
            "servo_scope": self.servo_scope,
            "servo_force_norm_max": float(self._last_servo_force_norm_max),
            "reset_mode": self._last_reset_mode,
            "reset_traj_step": int(self._last_reset_traj_step),
            "control_substeps": int(self.control_substeps),
            "physics_timestep_s": float(self.model.opt.timestep),
            "policy_control_dt_s": float(self.model.opt.timestep * self.control_substeps),
            "base_runtime": self.base_runtime,
        }

    def _info(self) -> dict[str, Any]:
        grip_slip = self._grip_slip()
        grip_site_error = self._grip_site_error()
        palm_to_grip = self._palm_to_grip_distance()
        contact_report = self._hand_handle_contact_report()
        body_report = self._body_mimic_report()
        root_z = self._root_height()
        return {
            "body_mimic_error": body_report.error,
            "body_mimic_qpos_mse": body_report.qpos_mse,
            "body_mimic_qvel_mse": body_report.qvel_mse,
            "body_mimic_reference": body_report.reference,
            "body_mimic_tracked_joint_count": body_report.tracked_joint_count,
            "body_mimic_tracked_qpos_count": body_report.tracked_qpos_count,
            "body_mimic_tracked_qvel_count": body_report.tracked_qvel_count,
            "grip_site_error_m": grip_site_error,
            "racket_hand_pose_error_m": grip_site_error,
            "grip_slip_m": grip_slip,
            "palm_to_grip_m": palm_to_grip,
            "hand_handle_contact_count": contact_report["hand_handle_contact_count"],
            "illegal_handle_contact_count": contact_report["illegal_handle_contact_count"],
            "max_handle_penetration_m": contact_report["max_handle_penetration_m"],
            "racket_drop": bool(palm_to_grip > 0.25),
            "body_fall": bool(root_z < 0.55),
            "root_height_m": root_z,
            "step_count": int(self._step_count),
        }

    def _reward_terms(self, action: np.ndarray, info: dict[str, Any]) -> dict[str, float]:
        contact_reward = min(float(info["hand_handle_contact_count"]) / 4.0, 1.0)
        effort = float(np.mean(np.square(action))) if action.size else 0.0
        terms = {
            "r_mimic_body": -self.reward_weights["mimic_body"] * float(info["body_mimic_error"]),
            "r_grip_site": -self.reward_weights["grip_site"] * float(info["grip_site_error_m"]),
            "r_contact": self.reward_weights["contact"] * contact_reward,
            "r_no_slip": -self.reward_weights["no_slip"] * float(info["grip_slip_m"]),
            "r_no_penetration": -self.reward_weights["no_penetration"] * float(info["max_handle_penetration_m"]),
            "r_racket_hand_pose": -self.reward_weights["racket_hand_pose"] * float(info["racket_hand_pose_error_m"]),
            "r_residual_effort": -self.reward_weights["residual_effort"] * effort,
        }
        return {name: float(terms[name]) for name in REWARD_TERM_NAMES}

    def _grip_to_palm_vector(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.grip_site_id] - self.data.site_xpos[self.palm_site_id], dtype=float)

    def _grip_slip(self) -> float:
        return float(np.linalg.norm(self._grip_to_palm_vector() - self._reference_grip_to_palm))

    def _grip_site_error(self) -> float:
        return float(np.linalg.norm(self._grip_to_palm_vector() - self._reference_grip_to_palm))

    def _palm_to_grip_distance(self) -> float:
        return float(np.linalg.norm(self._grip_to_palm_vector()))

    def _body_mimic_error(self) -> float:
        return self._body_mimic_report().error

    def _body_mimic_report(self) -> BodyTrackingReport:
        if self.body_goal_provider is None:
            return reset_snapshot_body_tracking_error(self.data.qpos, self._reference_qpos)
        reference = self.body_goal_provider.reference_state(self.body_goal_provider.traj_step)
        return trajectory_body_tracking_error(
            model=self.model,
            data=self.data,
            reference_model=self.body_goal_provider._legacy_model,
            reference_qpos=reference.qpos,
            reference_qvel=reference.qvel,
        )

    def _root_height(self) -> float:
        qadr = int(self.model.jnt_qposadr[self.root_joint_id])
        return float(self.data.qpos[qadr + 2])

    def _hand_handle_contact_report(self) -> dict[str, int | float]:
        hand_handle_geoms: set[int] = set()
        illegal_count = 0
        max_penetration = 0.0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            has_handle = geom1 in self.handle_geom_ids or geom2 in self.handle_geom_ids
            if not has_handle:
                continue
            max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
            if geom1 in self.right_hand_geom_ids and geom2 in self.handle_geom_ids:
                hand_handle_geoms.add(geom1)
            elif geom2 in self.right_hand_geom_ids and geom1 in self.handle_geom_ids:
                hand_handle_geoms.add(geom2)
            else:
                illegal_count += 1
        return {
            "hand_handle_contact_count": len(hand_handle_geoms),
            "illegal_handle_contact_count": illegal_count,
            "max_handle_penetration_m": max_penetration,
        }


def _validate_reward_weights(weights: dict[str, float]) -> None:
    unknown = sorted(set(weights) - set(DEFAULT_REWARD_WEIGHTS))
    if unknown:
        raise ValueError(f"unknown grip-hold reward weights: {unknown}")
    for name, value in weights.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"reward weight {name!r} must be finite")


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"missing site {name!r}")
    return int(site_id)


def align_racket_grip_to_palm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    reference_grip_to_palm: np.ndarray,
) -> None:
    racket_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_racket_free")
    palm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    if racket_joint_id < 0 or palm_site_id < 0 or grip_site_id < 0:
        raise ValueError("cannot align racket grip: missing racket joint or grip sites")
    if int(model.jnt_type[racket_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("overall_racket_free must be a free joint")
    ref = np.asarray(reference_grip_to_palm, dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all():
        raise ValueError(f"reference_grip_to_palm must be finite shape (3,), got {ref.shape}")
    qadr = int(model.jnt_qposadr[racket_joint_id])
    desired_grip = np.asarray(data.site_xpos[palm_site_id], dtype=float) + ref
    delta = desired_grip - np.asarray(data.site_xpos[grip_site_id], dtype=float)
    data.qpos[qadr : qadr + 3] += delta
    mujoco.mj_forward(model, data)


def restore_object_free_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    states: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for joint_name, (qpos, qvel) in states.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"object joint {joint_name!r} is not a free joint")
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        data.qpos[qadr : qadr + 7] = qpos
        data.qvel[dadr : dadr + 6] = qvel
    mujoco.mj_forward(model, data)


def set_free_joint_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    xy: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"missing free joint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"joint {joint_name!r} is not a free joint")
    value = np.asarray(xy, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(f"xy must be finite shape (2,), got {value.shape}")
    qadr = int(model.jnt_qposadr[joint_id])
    data.qpos[qadr : qadr + 2] = value
    mujoco.mj_forward(model, data)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"missing body {name!r}")
    return int(body_id)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"missing joint {name!r}")
    return int(joint_id)


def _ready_grip_to_palm_vector(xml_path: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    palm_site_id = _site_id(model, "rh_palm_grip_site")
    grip_site_id = _site_id(model, "overall_grip_pose_site")
    return np.array(data.site_xpos[grip_site_id] - data.site_xpos[palm_site_id], dtype=float)


def _ready_object_free_joint_states(xml_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for joint_name in ("overall_racket_free", "overall_shuttle_free"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"object joint {joint_name!r} is not a free joint")
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        qpos = np.asarray(data.qpos[qadr : qadr + 7], dtype=float).copy()
        qvel = np.asarray(data.qvel[dadr : dadr + 6], dtype=float).copy()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError(f"ready state for {joint_name!r} is non-finite")
        if not np.isclose(np.linalg.norm(qpos[3:7]), 1.0, atol=1e-6):
            raise ValueError(f"ready state for {joint_name!r} has invalid quaternion {qpos[3:7]}")
        states[joint_name] = (qpos, qvel)
    return states


def _ready_free_joint_xy(xml_path: Path, joint_name: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"missing free joint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"joint {joint_name!r} is not a free joint")
    qadr = int(model.jnt_qposadr[joint_id])
    return np.asarray(data.qpos[qadr : qadr + 2], dtype=float).copy()


def _handle_geom_ids(model: mujoco.MjModel) -> set[int]:
    result: set[int] = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("overall_handle_bevel_") or name == "overall_handle_grip":
            result.add(int(geom_id))
    return result


def _right_hand_geom_ids(model: mujoco.MjModel) -> set[int]:
    roots = {
        "lunate_r",
        "distal_thumb_r",
        "2distph_r",
        "3distph_r",
        "4distph_r",
        "5distph_r",
    }
    root_ids = {
        int(body_id)
        for name in roots
        if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    }
    descendant_ids = _descendant_body_ids(model, root_ids)
    return {
        int(geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendant_ids
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    }


def _descendant_body_ids(model: mujoco.MjModel, root_body_ids: set[int]) -> set[int]:
    children_by_parent: dict[int, list[int]] = {body_id: [] for body_id in range(model.nbody)}
    for body_id in range(1, model.nbody):
        children_by_parent[int(model.body_parentid[body_id])].append(body_id)
    descendants: set[int] = set()
    stack = list(root_body_ids)
    while stack:
        body_id = stack.pop()
        if body_id in descendants:
            continue
        descendants.add(body_id)
        stack.extend(children_by_parent[body_id])
    return descendants
