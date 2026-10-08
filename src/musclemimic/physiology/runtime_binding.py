"""Canonical runtime binding from environment action order to muscle state."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from musclemimic.physiology.effective_excitation import (
    MuscleChannelLayout,
    resolve_muscle_channel_layout,
)


def ordered_policy_actuator_names(env: Any) -> tuple[str, ...]:
    """Return names in the base environment's physical action order."""

    base = _base_env(env)
    model = getattr(base, "_model", getattr(base, "model", None))
    if model is None:
        raise ValueError("policy actuator binding requires a MuJoCo model")
    action_indices = getattr(base, "_action_indices", None)
    if action_indices is None:
        info = getattr(base, "info", getattr(base, "mdp_info", None))
        action_space = None if info is None else getattr(info, "action_space", None)
        if action_space is None or int(action_space.shape[0]) != int(model.nu):
            raise ValueError("custom policy action order requires explicit _action_indices")
        action_indices = np.arange(model.nu, dtype=np.int32)
    indices = np.asarray(action_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= int(model.nu)):
        raise ValueError("environment _action_indices are empty or out of range")
    if np.unique(indices).size != indices.size:
        raise ValueError("environment _action_indices contain duplicates")
    names = []
    for actuator_id in indices.tolist():
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise ValueError(f"policy actuator id {actuator_id} has no stable name")
        names.append(str(name))
    return tuple(names)


def resolve_ordered_policy_muscle_layout(
    env: Any,
    *,
    model: Any | None = None,
) -> MuscleChannelLayout:
    base = _base_env(env)
    runtime_model = getattr(base, "_model", getattr(base, "model", None)) if model is None else model
    if runtime_model is None:
        raise ValueError("policy muscle binding requires a MuJoCo model")
    return resolve_muscle_channel_layout(
        runtime_model,
        ordered_policy_actuator_names(base),
        require_unit_ctrlrange=True,
        require_scalar_activation=True,
    )


def resolve_all_muscle_channel_layout(model: Any) -> MuscleChannelLayout | None:
    """Resolve every model muscle in actuator order through the shared layout."""

    muscle_ids = np.flatnonzero(np.asarray(model.actuator_dyntype) == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    if muscle_ids.size == 0:
        return None
    names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_id)))
        for actuator_id in muscle_ids.tolist()
    )
    return resolve_muscle_channel_layout(
        model,
        names,
        require_unit_ctrlrange=True,
        require_scalar_activation=True,
    )


def resolve_muscle_activation_addresses(model: Any) -> np.ndarray:
    """Return native muscle ``data.act`` addresses in actuator order.

    This intentionally depends only on MuJoCo's dynamics-address arrays so it
    can also validate lightweight model fixtures.  Rich taxonomy/ABI binding
    remains the responsibility of :func:`resolve_muscle_channel_layout`.
    """

    dyntype = np.asarray(model.actuator_dyntype)
    muscle_ids = np.flatnonzero(dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    if muscle_ids.size == 0:
        return np.empty((0,), dtype=np.int32)
    actnum = np.asarray(model.actuator_actnum)[muscle_ids]
    if np.any(actnum != 1):
        invalid = muscle_ids[actnum != 1].tolist()
        raise ValueError(
            "muscle activation diagnostics require one scalar activation "
            f"state per actuator; invalid actuator ids: {invalid}"
        )
    addresses = np.asarray(model.actuator_actadr, dtype=np.int64)[muscle_ids]
    activation_count = int(model.na)
    if np.any(addresses < 0) or np.any(addresses >= activation_count):
        raise ValueError("muscle actuator activation address is out of range")
    if np.unique(addresses).size != addresses.size:
        raise ValueError("muscle actuator activation addresses must be unique")
    return addresses.astype(np.int32, copy=False)


def _base_env(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while hasattr(current, "env") and id(current) not in visited:
        visited.add(id(current))
        next_env = current.env
        if next_env is current:
            break
        current = next_env
    return current
