from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter
from environment.overall_environment.src.action_manifest import reconstruct_action_manifest
from environment.overall_environment.src.body_obs_adapter import BodyObsAdapter
from environment.overall_environment.src.frozen_body_policy import (
    FrozenBodyPolicy,
    load_frozen_body_policy_manifest,
)
from environment.overall_environment.src.layered_control import resolve_actuator_groups
from environment.overall_environment.src.layered_control import actuator_names_from_model
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
        control_substeps: int = 1,
        max_episode_steps: int = 200,
        reward_weights: dict[str, float] | None = None,
        body_policy_artifact: str | Path | None = None,
        body_checkpoint: str | Path | None = None,
        body_goal_obs: np.ndarray | None = None,
    ) -> None:
        self.xml_path = Path(xml)
        report = build_training_scene_report(self.xml_path)
        validate_training_scene_report(report)
        self.base_env = OverallBadmintonEnvironment(self.xml_path)
        self.model = self.base_env.model
        self.data = self.base_env.data
        self.control_substeps = _positive_int(control_substeps, "control_substeps")
        self.max_episode_steps = _positive_int(max_episode_steps, "max_episode_steps")
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        _validate_reward_weights(self.reward_weights)

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

        self._step_count = 0
        self._reference_qpos = np.array(self.model.qpos0, dtype=float)
        self._reference_grip_to_palm = np.zeros(3, dtype=float)
        self._reference_palm_to_grip_distance = 0.0
        self._last_body_obs_size = 0
        self._last_body_action_size = 0
        self._last_raw_body_action_max_abs = 0.0
        self._last_clipped_full_ctrl_max_abs = 0.0

        self.body_policy_source = "fake_zero"
        self.body_policy: FrozenBodyPolicy | None = None
        self.body_obs_adapter: BodyObsAdapter | None = None
        self.body_action_adapter: CheckpointToFullActionAdapter | None = None
        self.body_goal_provider: TrajectoryGoalProvider | None = None
        self.body_goal_obs = np.zeros(0, dtype=float)
        self.body_goal_obs_source = "none"
        if body_policy_artifact is not None:
            artifact_path = Path(body_policy_artifact)
            manifest = load_frozen_body_policy_manifest(artifact_path)
            checkpoint_path = Path(body_checkpoint) if body_checkpoint is not None else Path(manifest.source_checkpoint)
            self.body_policy = FrozenBodyPolicy.load_from_export(artifact_path)
            self.body_obs_adapter = BodyObsAdapter.from_checkpoint(checkpoint_path)
            action_manifest = reconstruct_action_manifest(checkpoint_path)
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

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        if self.body_goal_provider is not None:
            self.body_goal_provider.reset()
        obs, base_info = self.base_env.reset()
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
        full_ctrl = self._body_policy_ctrl()
        full_ctrl[self.residual_actuator_ids] += clipped_action
        full_ctrl = _clip_ctrl_to_model_range(self.model, full_ctrl)
        self._last_clipped_full_ctrl_max_abs = float(np.max(np.abs(full_ctrl))) if full_ctrl.size else 0.0
        obs = np.zeros(self.model.nq + self.model.nv, dtype=float)
        base_info: dict[str, Any] = {}
        for _ in range(self.control_substeps):
            obs, base_info = self.base_env.step(ctrl=full_ctrl, pose_servo=True)
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
        terminated = (not finite) or bool(info["racket_drop"]) or bool(info["body_fall"])
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _body_policy_ctrl(self) -> np.ndarray:
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
        full_ctrl = self.body_action_adapter.adapt(body_action)
        self._last_body_obs_size = int(body_obs.size)
        self._last_body_action_size = int(body_action.size)
        self._last_raw_body_action_max_abs = float(np.max(np.abs(body_action))) if body_action.size else 0.0
        return full_ctrl

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
        }

    def _info(self) -> dict[str, Any]:
        grip_slip = self._grip_slip()
        palm_to_grip = self._palm_to_grip_distance()
        contact_report = self._hand_handle_contact_report()
        body_error = self._body_mimic_error()
        root_z = self._root_height()
        return {
            "body_mimic_error": body_error,
            "grip_site_error_m": abs(palm_to_grip - self._reference_palm_to_grip_distance),
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
            "r_racket_hand_pose": -self.reward_weights["racket_hand_pose"] * float(info["palm_to_grip_m"]),
            "r_residual_effort": -self.reward_weights["residual_effort"] * effort,
        }
        return {name: float(terms[name]) for name in REWARD_TERM_NAMES}

    def _grip_to_palm_vector(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.grip_site_id] - self.data.site_xpos[self.palm_site_id], dtype=float)

    def _grip_slip(self) -> float:
        return float(np.linalg.norm(self._grip_to_palm_vector() - self._reference_grip_to_palm))

    def _palm_to_grip_distance(self) -> float:
        return float(np.linalg.norm(self._grip_to_palm_vector()))

    def _body_mimic_error(self) -> float:
        delta = np.asarray(self.data.qpos, dtype=float) - self._reference_qpos
        if delta.size >= 7:
            delta[-14:-7] = 0.0
            delta[-7:] = 0.0
        return float(np.mean(np.square(delta))) if delta.size else 0.0

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


def _clip_ctrl_to_model_range(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    ctrl_array = np.asarray(ctrl, dtype=float)
    if ctrl_array.shape != (model.nu,):
        raise ValueError(f"ctrl must have shape ({model.nu},), got {ctrl_array.shape}")
    if not np.isfinite(ctrl_array).all():
        raise ValueError("ctrl contains non-finite values")
    clipped = ctrl_array.copy()
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if limited.any():
        lower = np.asarray(model.actuator_ctrlrange[:, 0], dtype=float)
        upper = np.asarray(model.actuator_ctrlrange[:, 1], dtype=float)
        clipped[limited] = np.clip(clipped[limited], lower[limited], upper[limited])
    return clipped


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"missing site {name!r}")
    return int(site_id)


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
