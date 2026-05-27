from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.hand_racket_model_map import HandRacketModelMap, load_model_map
from src.grip.grip_objectives import joint_limit_margin_cost
from src.grip.paths import REPO_ROOT, reference_json_path, scene_xml_path, target_config_path
from src.grip.target_config import GripTargetConfig, load_grip_target_config

DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "right_hand_racket_grip_training.yaml"
DEFAULT_ENV_CONFIG = {
    "control_substeps": 10,
    "max_episode_steps": 500,
    "curriculum_stage": 0,
}
DEFAULT_REWARD_WEIGHTS = {
    "site_match": 4.0,
    "v_shape": 1.0,
    "anti_panhandle": 1.0,
    "anti_thumb_grip": 1.0,
    "racket_pose": 2.0,
    "racket_orient": 0.5,
    "contact": 0.5,
    "no_slip": 0.2,
    "reference_pose": 0.1,
    "effort": 0.001,
    "joint_limits": 0.1,
    "no_penetration": 0.1,
}
REWARD_TERM_NAMES = {
    "site_match": "r_site_match",
    "v_shape": "r_v_shape",
    "anti_panhandle": "r_anti_panhandle",
    "anti_thumb_grip": "r_anti_thumb_grip",
    "racket_pose": "r_racket_pose",
    "racket_orient": "r_racket_orient",
    "contact": "r_contact",
    "no_slip": "r_no_slip",
    "reference_pose": "r_reference_pose",
    "effort": "r_effort",
    "joint_limits": "r_joint_limits",
    "no_penetration": "r_no_penetration",
}


class RightHandRacketGripEnv:
    def __init__(
        self,
        xml: str | Path,
        targets: str | Path,
        reference: str | Path,
        training_config: str | Path | None = None,
    ) -> None:
        self.xml_path = Path(xml)
        self.targets_path = Path(targets)
        self.reference_path = Path(reference)
        self.training_config_path = Path(training_config) if training_config is not None else DEFAULT_TRAINING_CONFIG

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model_map = load_model_map(self.model)
        if not self.model_map.ok:
            raise ValueError(f"unresolved MuJoCo model names: {self.model_map.missing}")

        self.target_config = load_grip_target_config(self.targets_path)
        self.reference = self._load_reference(self.reference_path)
        self.reference_qpos = self._reference_vector("qpos", self.model.nq)
        self.reference_qvel = self._reference_vector("qvel", self.model.nv)

        self.config = _load_training_config(self.training_config_path)
        self.control_substeps = _positive_int(self.config["env"].get("control_substeps"), "env.control_substeps")
        self.max_episode_steps = _positive_int(self.config["env"].get("max_episode_steps"), "env.max_episode_steps")
        self.reward_weights = {
            **DEFAULT_REWARD_WEIGHTS,
            **{
                name: _finite_float(value, f"reward.{name}")
                for name, value in self.config.get("reward", {}).items()
            },
        }

        self.right_hand_actuator_ids = _actuator_ids_from_names(self.model, self.model_map.right_hand_actuator_names)
        self.racket_body_id = _body_id(self.model, self.model_map.racket_body, "racket body")
        self.grip_pose_site_id = _site_id(self.model, "grip_pose_site")
        self.palm_site_id = _site_id(self.model, self.model_map.hand_sites["palm"])
        self.right_hand_qpos_indices = _joint_qpos_indices(self.model, self.model_map.right_hand_joint_names)
        self.right_hand_limited_qpos_indices, self.right_hand_joint_ranges = _limited_joint_qpos_ranges(
            self.model,
            self.model_map.right_hand_joint_names,
        )
        self.handle_geom_ids = _geom_ids_from_names(self.model, self.model_map.handle_geoms, "handle")
        self.right_hand_contact_geom_ids = _right_hand_contact_geom_ids(self.model, self.model_map)
        if not self.handle_geom_ids:
            raise ValueError("handle contact geom map is empty")
        if not self.right_hand_contact_geom_ids:
            raise ValueError("right-hand contact geom map is empty")

        self.action_size = len(self.right_hand_actuator_ids)
        if self.action_size == 0:
            raise ValueError("right-hand actuator map is empty")

        self.reference_racket_pos, self.reference_racket_xmat = self._reference_racket_pose()
        self.reference_grip_to_palm = self._reference_grip_to_palm_vector()
        self.reference_hand_qpos = self.reference_qpos[self.right_hand_qpos_indices].copy()
        self._step_count = 0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.reference_qpos
        self.data.qvel[:] = self.reference_qvel
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must be finite")

        clipped_action = np.clip(action_array, -1.0, 1.0)
        self.data.ctrl[self.right_hand_actuator_ids] = clipped_action
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._observation()
        info = self._info()
        reward_terms = self._reward_terms(clipped_action, info)
        reward = float(sum(reward_terms.values()))
        if not math.isfinite(reward):
            raise FloatingPointError(f"non-finite reward from terms: {reward_terms}")

        info["reward_terms"] = reward_terms
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def contact_geom_id_sets(self) -> dict[str, set[int]]:
        return {
            "handle": set(self.handle_geom_ids),
            "right_hand": set(self.right_hand_contact_geom_ids),
        }

    def _load_reference(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            reference = json.load(f)
        if not isinstance(reference, dict):
            raise ValueError("reference JSON root must be an object")
        return reference

    def _reference_vector(self, key: str, expected_size: int) -> np.ndarray:
        if key not in self.reference:
            raise ValueError(f"reference missing {key!r}")
        values = np.asarray(self.reference[key], dtype=float)
        if values.shape != (expected_size,):
            raise ValueError(f"reference {key!r} must have shape ({expected_size},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reference {key!r} must be finite")
        return values.copy()

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array(self.data.qpos, dtype=float),
                np.array(self.data.qvel, dtype=float),
                np.array(self.data.ctrl[self.right_hand_actuator_ids], dtype=float),
            ],
        )

    def _info(self) -> dict[str, Any]:
        site_errors = self._site_errors()
        mean_error = float(np.mean(list(site_errors.values()))) if site_errors else 0.0
        contact_report = self._handle_contact_report()
        racket_translation_error, racket_orientation_error = self._racket_pose_errors()
        grip_slip = self._grip_slip()
        reference_pose_error = self._reference_pose_error()
        joint_limit_cost = self._joint_limit_margin_cost()
        forehand_metrics = self._forehand_v_grip_metrics()
        return {
            "site_errors_m": site_errors,
            "mean_site_error_m": mean_error,
            "v_shape_error": forehand_metrics["v_shape_error"],
            "anti_panhandle_error": forehand_metrics["anti_panhandle_error"],
            "anti_thumb_grip_error": forehand_metrics["anti_thumb_grip_error"],
            "thumb_index_y_gap_m": forehand_metrics["thumb_index_y_gap_m"],
            "v_bisector_theta_deg": forehand_metrics["v_bisector_theta_deg"],
            "palm_theta_deg": forehand_metrics["palm_theta_deg"],
            "thumb_theta_deg": forehand_metrics["thumb_theta_deg"],
            "racket_translation_error_m": racket_translation_error,
            "racket_orientation_error_deg": racket_orientation_error,
            "grip_slip_m": grip_slip,
            "reference_pose_error": reference_pose_error,
            "joint_limit_margin_cost": joint_limit_cost,
            "contact_count": contact_report["contact_count"],
            "illegal_handle_contact_count": contact_report["illegal_handle_contact_count"],
            "max_handle_penetration_m": contact_report["max_handle_penetration_m"],
            "raw_contact_count": int(self.data.ncon),
            "step_count": int(self._step_count),
        }

    def _site_errors(self) -> dict[str, float]:
        current_sites = _current_hand_site_positions(self.model, self.data, self.model_map)
        target_sites = _target_sites_world(self.model, self.data, self.target_config, self.model_map)
        return {
            name: float(np.linalg.norm(current_sites[name] - target_sites[name]))
            for name in sorted(target_sites)
        }

    def _filtered_contact_count(self) -> int:
        return int(self._handle_contact_report()["contact_count"])

    def _handle_contact_report(self) -> dict[str, int | float]:
        contacting_right_hand_geoms: set[int] = set()
        illegal_count = 0
        max_penetration = 0.0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.right_hand_contact_geom_ids and geom2 in self.handle_geom_ids:
                contacting_right_hand_geoms.add(geom1)
            elif geom2 in self.right_hand_contact_geom_ids and geom1 in self.handle_geom_ids:
                contacting_right_hand_geoms.add(geom2)
            elif geom1 in self.handle_geom_ids or geom2 in self.handle_geom_ids:
                illegal_count += 1
            if geom1 in self.handle_geom_ids or geom2 in self.handle_geom_ids:
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        return {
            "contact_count": len(contacting_right_hand_geoms),
            "illegal_handle_contact_count": illegal_count,
            "max_handle_penetration_m": max_penetration,
        }

    def _reference_racket_pose(self) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.reference_qpos
        data.qvel[:] = self.reference_qvel
        mujoco.mj_forward(self.model, data)
        return (
            np.array(data.xpos[self.racket_body_id], dtype=float),
            np.array(data.xmat[self.racket_body_id], dtype=float).reshape(3, 3),
        )

    def _reference_grip_to_palm_vector(self) -> np.ndarray:
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.reference_qpos
        data.qvel[:] = self.reference_qvel
        mujoco.mj_forward(self.model, data)
        grip_pos = np.array(data.site_xpos[self.grip_pose_site_id], dtype=float)
        palm_pos = np.array(data.site_xpos[self.palm_site_id], dtype=float)
        return grip_pos - palm_pos

    def _racket_pose_errors(self) -> tuple[float, float]:
        current_pos = np.array(self.data.xpos[self.racket_body_id], dtype=float)
        current_xmat = np.array(self.data.xmat[self.racket_body_id], dtype=float).reshape(3, 3)
        translation_error = float(np.linalg.norm(current_pos - self.reference_racket_pos))
        orientation_error = _rotation_angle_deg(current_xmat @ self.reference_racket_xmat.T)
        return translation_error, orientation_error

    def _grip_slip(self) -> float:
        grip_pos = np.array(self.data.site_xpos[self.grip_pose_site_id], dtype=float)
        palm_pos = np.array(self.data.site_xpos[self.palm_site_id], dtype=float)
        return float(np.linalg.norm((grip_pos - palm_pos) - self.reference_grip_to_palm))

    def _reference_pose_error(self) -> float:
        current = np.array(self.data.qpos[self.right_hand_qpos_indices], dtype=float)
        delta = current - self.reference_hand_qpos
        return float(np.mean(np.square(delta))) if delta.size else 0.0

    def _joint_limit_margin_cost(self) -> float:
        if self.right_hand_limited_qpos_indices.size == 0:
            return 0.0
        qpos = np.array(self.data.qpos[self.right_hand_limited_qpos_indices], dtype=float)
        return joint_limit_margin_cost(qpos, self.right_hand_joint_ranges)

    def _forehand_v_grip_metrics(self) -> dict[str, float]:
        local_sites = self._hand_sites_racket_local()
        thumb = local_sites["thumb"]
        index = local_sites["index"]
        palm = local_sites["palm"]
        thumb_theta = _theta_deg(thumb)
        index_theta = _theta_deg(index)
        palm_theta = _theta_deg(palm)
        thumb_vec = _xz_unit(thumb)
        index_vec = _xz_unit(index)
        bisector_vec = thumb_vec + index_vec
        if float(np.linalg.norm(bisector_vec)) < 1e-9:
            bisector_theta = 0.0
        else:
            bisector_theta = _theta_deg(np.array([bisector_vec[0], 0.0, bisector_vec[1]], dtype=float))

        y_gap = float(index[1] - thumb[1])
        y_gap_error = max(0.0, -y_gap) + max(0.0, y_gap - 0.015)
        bisector_error = abs(_angle_diff_deg(bisector_theta, 0.0)) / 180.0
        v_shape_error = bisector_error + y_gap_error / 0.02
        anti_panhandle_error = abs(_angle_diff_deg(palm_theta, 180.0)) / 180.0
        anti_thumb_grip_error = abs(_angle_diff_deg(thumb_theta, 45.0)) / 180.0
        return {
            "v_shape_error": float(v_shape_error),
            "anti_panhandle_error": float(anti_panhandle_error),
            "anti_thumb_grip_error": float(anti_thumb_grip_error),
            "thumb_index_y_gap_m": y_gap,
            "v_bisector_theta_deg": float(bisector_theta),
            "palm_theta_deg": float(palm_theta),
            "thumb_theta_deg": float(thumb_theta),
        }

    def _hand_sites_racket_local(self) -> dict[str, np.ndarray]:
        racket_pos = np.array(self.data.xpos[self.racket_body_id], dtype=float)
        racket_xmat = np.array(self.data.xmat[self.racket_body_id], dtype=float).reshape(3, 3)
        world_sites = _current_hand_site_positions(self.model, self.data, self.model_map)
        return {
            name: racket_xmat.T @ (position - racket_pos)
            for name, position in world_sites.items()
        }

    def _reward_terms(self, action: np.ndarray, info: dict[str, Any]) -> dict[str, float]:
        mean_error = float(info["mean_site_error_m"])
        contact_count = int(info["contact_count"])
        contact_reward = min(contact_count / 4.0, 1.0)
        effort = float(np.mean(np.square(action))) if action.size else 0.0
        return {
            "r_site_match": -self.reward_weights["site_match"] * mean_error,
            "r_v_shape": -self.reward_weights["v_shape"] * float(info["v_shape_error"]),
            "r_anti_panhandle": -self.reward_weights["anti_panhandle"] * float(info["anti_panhandle_error"]),
            "r_anti_thumb_grip": -self.reward_weights["anti_thumb_grip"] * float(info["anti_thumb_grip_error"]),
            "r_racket_pose": -self.reward_weights["racket_pose"] * float(info["racket_translation_error_m"]),
            "r_racket_orient": -self.reward_weights["racket_orient"]
            * float(info["racket_orientation_error_deg"])
            / 180.0,
            "r_contact": self.reward_weights["contact"] * contact_reward,
            "r_no_slip": -self.reward_weights["no_slip"] * float(info["grip_slip_m"]),
            "r_reference_pose": -self.reward_weights["reference_pose"] * float(info["reference_pose_error"]),
            "r_effort": -self.reward_weights["effort"] * effort,
            "r_joint_limits": -self.reward_weights["joint_limits"] * float(info["joint_limit_margin_cost"]),
            "r_no_penetration": -self.reward_weights["no_penetration"] * float(info["max_handle_penetration_m"]),
        }


def _load_training_config(path: Path) -> dict[str, Any]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"training config root must be a mapping: {path}")
    else:
        loaded = {}

    env = {**DEFAULT_ENV_CONFIG, **_mapping_value(loaded.get("env", {}), "env")}
    reward = {**DEFAULT_REWARD_WEIGHTS, **_mapping_value(loaded.get("reward", {}), "reward")}
    unknown_reward_keys = sorted(set(reward).difference(REWARD_TERM_NAMES))
    if unknown_reward_keys:
        raise ValueError(f"unsupported reward config key(s): {unknown_reward_keys}")
    return {"env": env, "reward": reward}


def _mapping_value(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"training config {name!r} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _actuator_ids_from_names(model: mujoco.MjModel, actuator_names: tuple[str, ...]) -> np.ndarray:
    ids: list[int] = []
    for actuator_name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"missing right-hand actuator {actuator_name!r}")
        ids.append(actuator_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate right-hand actuator ids: {ids}")
    return np.asarray(ids, dtype=int)


def _body_id(model: mujoco.MjModel, body_name: str | None, context: str) -> int:
    if body_name is None:
        raise ValueError(f"missing {context}")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"missing {context} {body_name!r}")
    return int(body_id)


def _site_id(model: mujoco.MjModel, site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"missing site {site_name!r}")
    return int(site_id)


def _joint_qpos_indices(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    indices: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing right-hand joint {joint_name!r}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            raise ValueError(f"right-hand joint {joint_name!r} must be 1-DoF, got MuJoCo type {joint_type}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate right-hand qpos indices: {indices}")
    return np.asarray(indices, dtype=int)


def _limited_joint_qpos_ranges(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    ranges: list[np.ndarray] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing right-hand joint {joint_name!r}")
        if int(model.jnt_limited[joint_id]) == 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            raise ValueError(f"limited right-hand joint {joint_name!r} must be 1-DoF, got MuJoCo type {joint_type}")
        indices.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.array(model.jnt_range[joint_id], dtype=float))
    if not indices:
        return np.asarray([], dtype=int), np.empty((0, 2), dtype=float)
    return np.asarray(indices, dtype=int), np.vstack(ranges)


def _geom_ids_from_names(model: mujoco.MjModel, geom_names: tuple[str, ...], context: str) -> set[int]:
    ids: set[int] = set()
    for geom_name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"missing {context} geom {geom_name!r}")
        ids.add(int(geom_id))
    return ids


def _right_hand_contact_geom_ids(model: mujoco.MjModel, model_map: HandRacketModelMap) -> set[int]:
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in set(model_map.hand_bodies.values())
    }
    missing_body_names = [
        body_name
        for body_name in sorted(set(model_map.hand_bodies.values()))
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0
    ]
    if missing_body_names:
        raise ValueError(f"missing right-hand body name(s): {missing_body_names}")

    descendant_body_ids = _descendant_body_ids(model, {int(body_id) for body_id in body_ids})
    return {
        int(geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendant_body_ids
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    }


def _descendant_body_ids(model: mujoco.MjModel, root_body_ids: set[int]) -> set[int]:
    children_by_parent: dict[int, list[int]] = {body_id: [] for body_id in range(model.nbody)}
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        children_by_parent[parent_id].append(body_id)

    descendants: set[int] = set()
    stack = list(root_body_ids)
    while stack:
        body_id = stack.pop()
        if body_id in descendants:
            continue
        descendants.add(body_id)
        stack.extend(children_by_parent[body_id])
    return descendants


def _current_hand_site_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: HandRacketModelMap,
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for logical_name, site_name in model_map.hand_sites.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"missing hand site {site_name!r}")
        positions[logical_name] = np.array(data.site_xpos[site_id], dtype=float)
    return positions


def _target_sites_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_config: GripTargetConfig,
    model_map: HandRacketModelMap,
) -> dict[str, np.ndarray]:
    if model_map.racket_body is None:
        raise ValueError("missing racket body in model map")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, model_map.racket_body)
    if body_id < 0:
        raise ValueError(f"missing racket body {model_map.racket_body!r}")

    racket_pos = np.array(data.xpos[body_id], dtype=float)
    racket_xmat = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    return {
        name: racket_pos + racket_xmat @ target_config.target_xyz(name)
        for name in sorted(target_config.target_points_racket_local)
    }


def _theta_deg(local_xyz: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(local_xyz[2]), float(local_xyz[0]))))


def _xz_unit(local_xyz: np.ndarray) -> np.ndarray:
    xz = np.array([float(local_xyz[0]), float(local_xyz[2])], dtype=float)
    norm = float(np.linalg.norm(xz))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return xz / norm


def _angle_diff_deg(angle: float, target: float) -> float:
    return float((angle - target + 180.0) % 360.0 - 180.0)


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    cosine = (float(np.trace(rotation_matrix)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the right-hand racket grip environment.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--reference", type=Path, default=reference_json_path(), help="Grip reference JSON path.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Grip training YAML path.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a reset and one zero-action step.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env = RightHandRacketGripEnv(args.xml, args.targets, args.reference, args.training_config)
    if args.smoke_test:
        obs, reset_info = env.reset()
        action = np.zeros(env.action_size, dtype=float)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        summary = {
            "reset_obs_size": int(obs.size),
            "step_obs_size": int(next_obs.size),
            "action_size": int(env.action_size),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "reset_mean_site_error_m": reset_info["mean_site_error_m"],
            "step_mean_site_error_m": step_info["mean_site_error_m"],
            "v_shape_error": step_info["v_shape_error"],
            "anti_panhandle_error": step_info["anti_panhandle_error"],
            "anti_thumb_grip_error": step_info["anti_thumb_grip_error"],
            "thumb_index_y_gap_m": step_info["thumb_index_y_gap_m"],
            "v_bisector_theta_deg": step_info["v_bisector_theta_deg"],
            "palm_theta_deg": step_info["palm_theta_deg"],
            "thumb_theta_deg": step_info["thumb_theta_deg"],
            "racket_translation_error_m": step_info["racket_translation_error_m"],
            "racket_orientation_error_deg": step_info["racket_orientation_error_deg"],
            "grip_slip_m": step_info["grip_slip_m"],
            "reference_pose_error": step_info["reference_pose_error"],
            "joint_limit_margin_cost": step_info["joint_limit_margin_cost"],
            "contact_count": step_info["contact_count"],
            "illegal_handle_contact_count": step_info["illegal_handle_contact_count"],
            "max_handle_penetration_m": step_info["max_handle_penetration_m"],
            "raw_contact_count": step_info["raw_contact_count"],
            "step_count": step_info["step_count"],
            "reward_terms": step_info["reward_terms"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"RightHandRacketGripEnv(xml={args.xml}, action_size={env.action_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
