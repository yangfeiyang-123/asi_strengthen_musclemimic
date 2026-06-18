from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class CtrlRangeOverrideReport:
    source_count: int
    matched_count: int
    changed_count: int
    missing_actuators: tuple[str, ...]


def normalized_action_to_model_ctrl(model, action: np.ndarray) -> np.ndarray:
    """Match loco_mujoco DefaultControl direct-mode action scaling."""
    action_array = np.asarray(action, dtype=float)
    if action_array.shape != (model.nu,):
        raise ValueError(f"action must have shape ({model.nu},), got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("action contains non-finite values")

    ctrl = action_array.copy()
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if limited.any():
        lower = np.asarray(model.actuator_ctrlrange[:, 0], dtype=float)
        upper = np.asarray(model.actuator_ctrlrange[:, 1], dtype=float)
        mean = (upper + lower) / 2.0
        delta = (upper - lower) / 2.0
        ctrl[limited] = (action_array[limited] * delta[limited]) + mean[limited]
        ctrl[limited] = np.clip(ctrl[limited], lower[limited], upper[limited])
    return ctrl


def apply_checkpoint_ctrl_ranges_to_model(model, checkpoint: str | Path) -> CtrlRangeOverrideReport:
    ranges = checkpoint_actuator_ctrl_ranges(checkpoint)
    missing: list[str] = []
    matched = 0
    changed = 0
    for actuator_name, (lower, upper) in ranges.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            missing.append(actuator_name)
            continue
        matched += 1
        old_range = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=float).copy()
        new_range = np.asarray([lower, upper], dtype=float)
        if not np.allclose(old_range, new_range, rtol=0.0, atol=0.0):
            changed += 1
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id, :] = new_range
    return CtrlRangeOverrideReport(
        source_count=len(ranges),
        matched_count=matched,
        changed_count=changed,
        missing_actuators=tuple(missing),
    )


def checkpoint_actuator_ctrl_ranges(checkpoint: str | Path) -> dict[str, tuple[float, float]]:
    checkpoint_path = Path(checkpoint)
    env_params = _checkpoint_env_params(checkpoint_path)

    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(
        headless=True,
        disable_fingers=bool(env_params.get("disable_fingers", True)),
        enable_muscle_length_observations=bool(env_params.get("enable_muscle_length_observations", False)),
        enable_muscle_velocity_observations=bool(env_params.get("enable_muscle_velocity_observations", False)),
        enable_muscle_force_observations=bool(env_params.get("enable_muscle_force_observations", False)),
        enable_muscle_excitation_observations=bool(env_params.get("enable_muscle_excitation_observations", False)),
        enable_muscle_activation_observations=bool(env_params.get("enable_muscle_activation_observations", False)),
        enable_touch_sensor_observations=bool(env_params.get("enable_touch_sensor_observations", True)),
        goal_type=str(env_params.get("goal_type", "GoalTrajMimic")),
        goal_params=dict(env_params.get("goal_params", {})),
        mjx_backend=str(env_params.get("mjx_backend", "jax")),
    )
    source_model = getattr(env, "_model", None) or getattr(env, "model", None)
    if source_model is None:
        raise ValueError("could not access checkpoint source MuJoCo model")

    ranges: dict[str, tuple[float, float]] = {}
    for actuator_id in range(source_model.nu):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            continue
        lower = float(source_model.actuator_ctrlrange[actuator_id, 0])
        upper = float(source_model.actuator_ctrlrange[actuator_id, 1])
        ranges[str(name)] = (lower, upper)
    return ranges


def _checkpoint_env_params(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return dict(metadata["experiment"]["env_params"])
