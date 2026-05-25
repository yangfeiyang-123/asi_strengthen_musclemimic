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
        self.handle_geom_ids = _geom_ids_from_names(self.model, self.model_map.handle_geoms, "handle")
        self.right_hand_contact_geom_ids = _right_hand_contact_geom_ids(self.model, self.model_map)
        if not self.handle_geom_ids:
            raise ValueError("handle contact geom map is empty")
        if not self.right_hand_contact_geom_ids:
            raise ValueError("right-hand contact geom map is empty")

        self.action_size = len(self.right_hand_actuator_ids)
        if self.action_size == 0:
            raise ValueError("right-hand actuator map is empty")

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
        return {
            "site_errors_m": site_errors,
            "mean_site_error_m": mean_error,
            "contact_count": self._filtered_contact_count(),
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
        count = 0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in self.right_hand_contact_geom_ids and geom2 in self.handle_geom_ids) or (
                geom2 in self.right_hand_contact_geom_ids and geom1 in self.handle_geom_ids
            ):
                count += 1
        return count

    def _reward_terms(self, action: np.ndarray, info: dict[str, Any]) -> dict[str, float]:
        mean_error = float(info["mean_site_error_m"])
        contact_count = int(info["contact_count"])
        contact_reward = 1.0 if contact_count > 0 else 0.0
        effort = float(np.mean(np.square(action))) if action.size else 0.0
        return {
            "r_site_match": -self.reward_weights["site_match"] * mean_error,
            "r_racket_pose": 0.0,
            "r_racket_orient": 0.0,
            "r_contact": self.reward_weights["contact"] * contact_reward,
            "r_no_slip": 0.0,
            "r_reference_pose": 0.0,
            "r_effort": -self.reward_weights["effort"] * effort,
            "r_joint_limits": 0.0,
            "r_no_penetration": 0.0,
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
            "contact_count": step_info["contact_count"],
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
