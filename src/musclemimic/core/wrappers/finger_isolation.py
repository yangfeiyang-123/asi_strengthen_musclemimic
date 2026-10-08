"""Policy-interface isolation for full-finger MyoFullBody environments.

The physics environment keeps all 416 actuators so finger perturbations retain
their physical effect.  The body policy, however, sees the exact non-finger
observation schema and emits only the 354 non-finger actions.  Right-hand and
left-hand actions are supplied by explicit, independent providers.
"""

from __future__ import annotations

import hashlib
import json
from copy import copy, deepcopy
from typing import Any, Iterable

import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.core.utils import Box
from musclemimic.core.wrappers.mjx import BaseWrapper
from musclemimic.utils.finger_isolation import (
    FingerActuatorPartition,
    NamedObservationSchema,
    ObservationField,
    ObservationFilter,
    finger_actuator_side,
    finger_joint_side,
)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _base_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def model_action_names(env: Any) -> tuple[str, ...]:
    """Return policy action names in the environment's actual action order."""
    base = _base_env(env)
    model = getattr(base, "_model", getattr(base, "model", None))
    if model is None:
        raise ValueError("finger isolation requires an environment with a MuJoCo model")

    action_indices = getattr(base, "_action_indices", None)
    if action_indices is None:
        action_dim = int(base.info.action_space.shape[0])
        if action_dim != int(model.nu):
            raise ValueError(
                "cannot infer actuator names for a custom action space without _action_indices"
            )
        action_indices = np.arange(model.nu, dtype=int)

    names: list[str] = []
    for actuator_id in np.asarray(action_indices, dtype=int).reshape(-1):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_id))
        if not name:
            raise ValueError(f"action actuator id {int(actuator_id)} has no name")
        names.append(name)
    if len(names) != int(base.info.action_space.shape[0]):
        raise ValueError(
            "action-name count does not match environment action space: "
            f"{len(names)} vs {base.info.action_space.shape[0]}"
        )
    return tuple(names)


def _joint_width(base: Any, joint_name: str, data_type: str) -> int:
    data = getattr(base, "_data", getattr(base, "data", None))
    if data is None:
        raise ValueError("finger observation schema requires MuJoCo data")
    joint = data.joint(joint_name)
    if data_type == "qpos":
        return int(len(joint.qpos))
    if data_type == "qvel":
        return int(len(joint.qvel))
    raise ValueError(f"unsupported joint-array observation data type {data_type!r}")


def build_named_observation_schema(env: Any) -> NamedObservationSchema:
    """Build exact flattened-observation provenance from ``obs_container``."""
    base = _base_env(env)
    if not hasattr(base, "obs_container"):
        raise ValueError("finger isolation requires env.obs_container")

    fields: list[ObservationField] = []
    for entry_name, entry in base.obs_container.items():
        indices = np.asarray(entry.obs_ind, dtype=int).reshape(-1)
        if indices.size == 0:
            continue

        xml_names = tuple(getattr(entry, "_xml_names", ()) or ())
        if xml_names:
            data_type = str(entry.data_type())
            cursor = 0
            for joint_name in xml_names:
                width = _joint_width(base, str(joint_name), data_type)
                fields.append(
                    ObservationField(
                        feature_name=f"{entry_name}:{joint_name}",
                        width=width,
                        joint_name=str(joint_name),
                    )
                )
                cursor += width
            if cursor != indices.size:
                raise ValueError(
                    f"observation {entry_name!r} XML widths sum to {cursor}, "
                    f"but obs_ind has {indices.size} entries"
                )
            continue

        xml_name = getattr(entry, "xml_name", None)
        data_type = str(entry.data_type()) if hasattr(entry, "data_type") else ""
        joint_name = None
        actuator_name = None
        if xml_name is not None and data_type in {"qpos", "qvel"}:
            joint_name = str(xml_name)
        elif xml_name is not None and data_type in {
            "actuator_length",
            "actuator_velocity",
            "actuator_force",
            "ctrl",
            "act",
        }:
            actuator_name = str(xml_name)
        fields.append(
            ObservationField(
                feature_name=str(entry_name),
                width=int(indices.size),
                joint_name=joint_name,
                actuator_name=actuator_name,
            )
        )

    schema = NamedObservationSchema(tuple(fields))
    raw_dim = int(base.info.observation_space.shape[0])
    if schema.total_size != raw_dim:
        raise ValueError(
            "named observation schema does not cover the complete policy observation: "
            f"{schema.total_size} vs {raw_dim}"
        )
    return schema


def build_body_observation_filter(
    env: Any,
    *,
    sides: Iterable[str] = ("right", "left"),
    finger_feature_names: Iterable[str] = (),
) -> ObservationFilter:
    """Return the exact name-based non-finger observation projection."""
    schema = build_named_observation_schema(env)
    return schema.without_fingers(
        sides=tuple(sides),  # type: ignore[arg-type]
        finger_feature_names=tuple(finger_feature_names),
    )


class FilteredObservationContainer:
    """Observation-container view whose indices match a projection."""

    def __init__(self, source: Any, kept_indices: np.ndarray):
        kept = np.asarray(kept_indices, dtype=int)
        source_dim = int(max(kept.max(initial=-1) + 1, 0))
        if hasattr(source, "values"):
            all_indices = [
                np.asarray(entry.obs_ind, dtype=int)
                for entry in source.values()
                if np.asarray(entry.obs_ind).size
            ]
            if all_indices:
                source_dim = max(source_dim, int(np.max(np.concatenate(all_indices))) + 1)
        old_to_new = np.full(source_dim, -1, dtype=int)
        old_to_new[kept] = np.arange(kept.size, dtype=int)

        self._entries: dict[str, Any] = {}
        for name, entry in source.items():
            old = np.asarray(entry.obs_ind, dtype=int).reshape(-1)
            valid = old[(old >= 0) & (old < source_dim)]
            new = old_to_new[valid]
            new = new[new >= 0]
            if new.size == 0:
                continue
            filtered_entry = copy(entry)
            filtered_entry.obs_ind = new.astype(int)
            self._entries[str(name)] = filtered_entry

    def get_obs_ind_by_group(self, group_name: str | None = None) -> np.ndarray:
        arrays = []
        for entry in self._entries.values():
            groups = getattr(entry, "group", ()) or ()
            if group_name is None or group_name in groups:
                arrays.append(np.asarray(entry.obs_ind, dtype=int))
        return np.concatenate(arrays).astype(int) if arrays else np.array([], dtype=int)

    def get_all_group_names(self) -> list[str | None]:
        return list(
            {
                group
                for entry in self._entries.values()
                for group in (getattr(entry, "group", ()) or ())
            }
        )

    def get_randomizable_obs_indices(self) -> np.ndarray:
        arrays = [
            np.asarray(entry.obs_ind, dtype=int)
            for entry in self._entries.values()
            if bool(getattr(entry, "allow_randomization", False))
        ]
        return np.concatenate(arrays).astype(int) if arrays else np.array([], dtype=int)

    def filter_by_group(self, obs: Any, group_name: str | None = None) -> Any:
        return obs[..., self.get_obs_ind_by_group(group_name)]

    def keys(self):
        return self._entries.keys()

    def values(self):
        return self._entries.values()

    def entries(self):
        return self._entries.values()

    def items(self):
        return self._entries.items()

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> Any:
        return self._entries[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._entries.get(key, default)


class BodyFingerIsolationWrapper(BaseWrapper):
    """Expose a 354D body-policy interface over a 416D finger-enabled env."""

    def __init__(self, env: Any, config: Any = None, **kwargs: Any):
        super().__init__(env)
        # LocoEnv inherits the Mjx base even for its CPU-only variants.  The
        # generic BaseWrapper therefore selects LocoMjxWrapper by class alone;
        # retain the native 5-tuple CPU API when mjx is explicitly disabled.
        if not bool(getattr(env, "mjx_enabled", False)):
            self.env = env
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.config = cfg

        expected_sizes = tuple(int(v) for v in _cfg_get(cfg, "expected_partition", (354, 31, 31)))
        if len(expected_sizes) != 3:
            raise ValueError("finger_isolation.expected_partition must contain body/right/left sizes")
        self.partition = FingerActuatorPartition.from_actuator_names(
            model_action_names(self.env),
            expected_sizes=expected_sizes,  # type: ignore[arg-type]
        )
        self._validate_hash(
            "actuator partition",
            self.partition.schema_hash,
            _cfg_get(cfg, "expected_actuator_partition_hash", None),
        )

        sides = tuple(_cfg_get(cfg, "observation_sides", ("right", "left")))
        feature_names = tuple(_cfg_get(cfg, "finger_feature_names", ()) or ())
        self.observation_filter = build_body_observation_filter(
            self.env,
            sides=sides,
            finger_feature_names=feature_names,
        )
        removed_dim = (
            self.observation_filter.source_schema.total_size
            - self.observation_filter.target_schema.total_size
        )
        expected_removed = _cfg_get(cfg, "expected_removed_observation_dim", None)
        if expected_removed is not None and removed_dim != int(expected_removed):
            raise ValueError(
                "finger observation projection removed an unexpected number of dimensions: "
                f"expected={int(expected_removed)} actual={removed_dim}"
            )
        expected_policy_obs = _cfg_get(cfg, "expected_policy_observation_dim", None)
        if (
            expected_policy_obs is not None
            and self.observation_filter.target_schema.total_size != int(expected_policy_obs)
        ):
            raise ValueError(
                "finger-isolated policy observation dimension mismatch: "
                f"expected={int(expected_policy_obs)} "
                f"actual={self.observation_filter.target_schema.total_size}"
            )
        self._validate_hash(
            "source observation schema",
            self.observation_filter.source_schema.schema_hash,
            _cfg_get(cfg, "expected_source_observation_schema_hash", None),
        )
        self._validate_hash(
            "policy observation schema",
            self.observation_filter.target_schema.schema_hash,
            _cfg_get(cfg, "expected_policy_observation_schema_hash", None),
        )
        self._validate_hash(
            "observation filter",
            self.observation_filter.schema_hash,
            _cfg_get(cfg, "expected_observation_filter_hash", None),
        )

        provider_cfg = _cfg_get(cfg, "right_grip_provider", {}) or {}
        provider_mode = str(_cfg_get(provider_cfg, "mode", "constant")).lower()
        if provider_mode not in {"constant", "fixed", "neutral"}:
            raise ValueError(
                "Stage-1 BodyFingerIsolationWrapper supports only a constant/fixed right-grip "
                f"provider, got {provider_mode!r}"
            )
        right_value = _cfg_get(provider_cfg, "value", 0.0)
        self._right_action = self._constant_action(
            "right_grip_provider.value", right_value, self.partition.right_grip_size
        )
        self._left_action = self._constant_action(
            "left_neutral_value",
            _cfg_get(cfg, "left_neutral_value", 0.0),
            self.partition.left_neutral_size,
        )

        self.info = self._update_info(self.env.info)
        self.mdp_info = self.info
        self.action_dim = self.partition.body_size
        self.obs_container = FilteredObservationContainer(
            self.env.obs_container,
            self.observation_filter.kept_indices,
        )
        self.policy_actuator_names = self.partition.body_actuator_names
        self.policy_interface_schema_hash = self._interface_hash()
        self._validate_hash(
            "policy interface",
            self.policy_interface_schema_hash,
            _cfg_get(cfg, "expected_policy_interface_hash", None),
        )

    @staticmethod
    def _constant_action(label: str, value: Any, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(size, float(array), dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{label} must be a scalar or shape ({size},), got {array.shape}")
        if not np.isfinite(array).all() or np.any(array < -1.0) or np.any(array > 1.0):
            raise ValueError(f"{label} must contain finite normalized actions in [-1, 1]")
        return array

    @staticmethod
    def _validate_hash(label: str, actual: str, expected: Any) -> None:
        if expected is not None and str(expected) != actual:
            raise ValueError(
                f"finger-isolation {label} hash mismatch: expected={expected} actual={actual}"
            )

    def _update_info(self, info: Any) -> Any:
        new_info = deepcopy(info)
        obs_indices = self.observation_filter.kept_indices
        body_indices = self.partition.body_indices
        new_info.observation_space = Box(
            np.asarray(info.observation_space.low)[obs_indices],
            np.asarray(info.observation_space.high)[obs_indices],
        )
        new_info.action_space = Box(
            np.asarray(info.action_space.low)[body_indices],
            np.asarray(info.action_space.high)[body_indices],
        )
        return new_info

    def _interface_hash(self) -> str:
        payload = {
            "kind": "body_finger_isolation_v1",
            "actuators": self.partition.schema_hash,
            "observations": self.observation_filter.schema_hash,
            "right_provider": self._right_action.tolist(),
            "left_neutral": self._left_action.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def filter_observation(self, observation: Any) -> Any:
        if not hasattr(observation, "shape"):
            observation = np.asarray(observation)
        if int(observation.shape[-1]) != self.observation_filter.source_schema.total_size:
            raise ValueError(
                "observation last dimension must match full finger-enabled schema: "
                f"expected={self.observation_filter.source_schema.total_size} "
                f"actual={observation.shape}"
            )
        indices = self.observation_filter.kept_indices
        if isinstance(observation, np.ndarray):
            return np.take(observation, indices, axis=-1)
        return observation[..., jnp.asarray(indices, dtype=jnp.int32)]

    def expand_body_action(self, body_action: Any) -> Any:
        if not hasattr(body_action, "shape"):
            body_action = np.asarray(body_action)
        if body_action.ndim == 0 or int(body_action.shape[-1]) != self.partition.body_size:
            raise ValueError(
                "body policy action last dimension must be "
                f"{self.partition.body_size}, got {body_action.shape}"
            )

        target_shape = tuple(body_action.shape[:-1]) + (self.partition.full_size,)
        if isinstance(body_action, np.ndarray):
            full = np.empty(target_shape, dtype=body_action.dtype)
            full[..., self.partition.body_indices] = body_action
            full[..., self.partition.right_grip_indices] = self._right_action
            full[..., self.partition.left_neutral_indices] = self._left_action
            return full

        full = jnp.empty(target_shape, dtype=body_action.dtype)
        full = full.at[..., jnp.asarray(self.partition.body_indices)].set(body_action)
        full = full.at[..., jnp.asarray(self.partition.right_grip_indices)].set(
            jnp.asarray(self._right_action, dtype=body_action.dtype)
        )
        return full.at[..., jnp.asarray(self.partition.left_neutral_indices)].set(
            jnp.asarray(self._left_action, dtype=body_action.dtype)
        )

    def reset(self, *args: Any, **kwargs: Any):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return self.filter_observation(result[0]), result[1]
        return self.filter_observation(result)

    def reset_to(self, *args: Any, **kwargs: Any):
        result = self.env.reset_to(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return self.filter_observation(result[0]), result[1]
        return self.filter_observation(result)

    def step(self, *args: Any, **kwargs: Any):
        if len(args) == 1:
            result = self.env.step(self.expand_body_action(args[0]), **kwargs)
        elif len(args) == 2:
            result = self.env.step(args[0], self.expand_body_action(args[1]), **kwargs)
        else:
            raise TypeError("BodyFingerIsolationWrapper.step expects action or (state, action)")
        if not isinstance(result, tuple) or len(result) not in {5, 6}:
            raise ValueError("wrapped environment step must return a 5- or 6-tuple")
        return (self.filter_observation(result[0]), *result[1:])
