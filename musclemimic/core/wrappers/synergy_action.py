"""Policy-interface wrapper for frozen early muscle-synergy actions."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.utils import Box
from musclemimic.core.wrappers.finger_isolation import model_action_names
from musclemimic.core.wrappers.mjx import BaseWrapper
from musclemimic.synergy.action_interface import (
    EarlySynergyActionInterface,
    EarlySynergyActionOutput,
    build_early_synergy_action_interface,
)


class SynergyActionWrapper(BaseWrapper):
    """Expose a K- or (K+R)-dimensional policy over a body-action environment.

    This wrapper changes only the action representation.  Observations,
    reference trajectories, rewards, terminal conditions, and underlying body
    physics remain owned by the wrapped environment.
    """

    def __init__(self, env: Any, config: Any):
        super().__init__(env)
        # Preserve the native CPU API in the same way as the finger-isolation
        # interface.  Production Stage-1 uses the MJX state/action signature.
        if not bool(getattr(env, "mjx_enabled", False)):
            self.env = env
        self.config = config
        underlying_names = getattr(env, "policy_actuator_names", None)
        if underlying_names is None:
            underlying_names = getattr(env, "policy_action_names", None)
        if underlying_names is None:
            underlying_names = model_action_names(env)
        self.body_actuator_names = tuple(str(name) for name in underlying_names)
        underlying_dim = int(env.info.action_space.shape[0])
        if len(self.body_actuator_names) != underlying_dim:
            raise ValueError(
                "early-synergy underlying body actuator names do not match action dimension: "
                f"names={len(self.body_actuator_names)} dim={underlying_dim}"
            )
        _validate_normalized_body_action_space(env.info.action_space, underlying_dim)
        runtime_ctrlrange = _resolve_runtime_ctrlrange(env, self.body_actuator_names)
        runtime_model_hash = _resolve_runtime_model_hash(env)

        self.action_interface: EarlySynergyActionInterface = build_early_synergy_action_interface(
            config,
            expected_actuator_names=self.body_actuator_names,
            runtime_ctrlrange=runtime_ctrlrange,
            runtime_model_hash=runtime_model_hash,
        )
        self.action_manifest = dict(self.action_interface.action_manifest)
        self.info = self._update_info(env.info)
        self.mdp_info = self.info
        self.action_dim = self.action_interface.policy_action_dim
        neutral_action = jnp.zeros((self.action_dim,), dtype=jnp.float32)
        neutral_metrics = self.action_interface.metrics(self.decode_action(neutral_action))
        self._reset_metrics = {name: jnp.zeros_like(value) for name, value in neutral_metrics.items()}
        coefficient_names = tuple(
            f"synergy_coefficient_{index:03d}" for index in range(self.action_interface.synergy_dim)
        )
        residual_names = tuple(
            f"structured_residual_{index:03d}" for index in range(self.action_interface.residual_dim)
        )
        self.policy_action_names = coefficient_names + residual_names
        # Existing policy-schema consumers inspect this attribute first.  These
        # are intentionally policy-coordinate names, not physical actuators.
        self.policy_actuator_names = self.policy_action_names
        self.physical_action_interface_hash = self.action_manifest["physical_action_interface_hash"]

    @property
    def decoder_jacobian_at_zero(self) -> np.ndarray:
        return self.action_interface.decoder_jacobian_at_zero

    def _update_info(self, info: Any) -> Any:
        new_info = deepcopy(info)
        dim = self.action_interface.policy_action_dim
        new_info.action_space = Box(
            -np.ones(dim, dtype=np.float32),
            np.ones(dim, dtype=np.float32),
        )
        return new_info

    def decode_action(self, policy_action: Any) -> EarlySynergyActionOutput:
        return self.action_interface.decode(policy_action)

    def reset(self, *args: Any, **kwargs: Any):
        return _replace_reset_state_info(
            self.env.reset(*args, **kwargs),
            self._reset_metrics,
        )

    def reset_to(self, *args: Any, **kwargs: Any):
        return _replace_reset_state_info(
            self.env.reset_to(*args, **kwargs),
            self._reset_metrics,
        )

    def step(self, *args: Any, **kwargs: Any):
        if len(args) == 1:
            decoded = self.decode_action(args[0])
            result = self.env.step(decoded.body_action, **kwargs)
        elif len(args) == 2:
            decoded = self.decode_action(args[1])
            result = self.env.step(args[0], decoded.body_action, **kwargs)
        else:
            raise TypeError("SynergyActionWrapper.step expects action or (state, action)")
        if not isinstance(result, tuple) or len(result) not in {5, 6}:
            raise ValueError("wrapped early-synergy environment step must return a 5- or 6-tuple")
        items = list(result)
        info_index = 4
        info = items[info_index]
        if not isinstance(info, dict):
            raise ValueError("wrapped early-synergy environment must return an info dictionary")
        metrics = self.action_interface.metrics(decoded)
        items[info_index] = {**info, **metrics}
        # AutoResetWrapper deliberately rebuilds its public info from the
        # transition state's innermost MjxState.info.  Persist diagnostics there
        # as well as in the returned tuple, otherwise they disappear before PPO
        # and validation logging see them.
        if len(items) == 6:
            items[5] = _replace_state_info(items[5], metrics)
        return tuple(items)


def _replace_reset_state_info(
    result: Any,
    metrics: dict[str, Any],
) -> Any:
    """Seed reset state diagnostics so JAX scan carries stay structural."""

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("wrapped early-synergy environment reset must return (observation, state)")
    observation, state = result
    return observation, _replace_state_info(state, metrics)


def _replace_state_info(state: Any, metrics: dict[str, Any]) -> Any:
    """Recursively attach scalar metrics to a Flax wrapper/MJX state."""

    fields = getattr(state, "__dataclass_fields__", {})
    if "info" in fields and hasattr(state, "replace"):
        info = getattr(state, "info", None)
        if isinstance(info, dict):
            return state.replace(info={**info, **metrics})
    if "env_state" in fields and hasattr(state, "replace"):
        inner = _replace_state_info(state.env_state, metrics)
        if inner is not state.env_state:
            return state.replace(env_state=inner)
    return state


def _validate_normalized_body_action_space(action_space: Any, expected_dim: int) -> None:
    low = np.broadcast_to(np.asarray(action_space.low, dtype=np.float64), (expected_dim,))
    high = np.broadcast_to(np.asarray(action_space.high, dtype=np.float64), (expected_dim,))
    if not np.array_equal(low, -np.ones(expected_dim)) or not np.array_equal(
        high,
        np.ones(expected_dim),
    ):
        raise ValueError("early-synergy decoder requires the underlying body action ABI to be normalized [-1,1]")


def _resolve_runtime_ctrlrange(
    env: Any,
    actuator_names: tuple[str, ...],
) -> np.ndarray | None:
    current = env
    visited: set[int] = set()
    model = None
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        model = getattr(current, "_model", getattr(current, "model", None))
        if model is not None:
            break
        current = getattr(current, "env", None)
    if model is None:
        return None

    import mujoco

    rows: list[np.ndarray] = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if actuator_id < 0:
            raise ValueError(f"runtime body actuator {name!r} is missing from the MuJoCo model")
        rows.append(np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64))
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(actuator_names), 2):
        raise ValueError("runtime MuJoCo actuator ctrlrange has an invalid shape")
    return result


def _resolve_runtime_model_hash(env: Any) -> str | None:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        model = getattr(current, "_model", getattr(current, "model", None))
        if model is not None:
            payload = model.__getstate__()
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("runtime MuJoCo model has no canonical complete byte state")
            return hashlib.sha256(payload).hexdigest()
        current = getattr(current, "env", None)
    return None
