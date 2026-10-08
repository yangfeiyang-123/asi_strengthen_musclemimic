from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RIGHT_HAND_FINGER_ACTUATORS = {
    "FDS2",
    "FDS3",
    "FDS4",
    "FDS5",
    "FDP2",
    "FDP3",
    "FDP4",
    "FDP5",
    "EDC2",
    "EDC3",
    "EDC4",
    "EDC5",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "RI3",
    "RI4",
    "RI5",
    "LU_RB2",
    "LU_RB3",
    "LU_RB4",
    "LU_RB5",
    "UI_UB2",
    "UI_UB3",
    "UI_UB4",
    "UI_UB5",
}
RIGHT_WRIST_ACTUATORS = {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL"}
RIGHT_FOREARM_ACTUATORS = {"SUP", "PT", "PQ", "BRD"}


@dataclass(frozen=True)
class LayeredActuatorRouter:
    all_actuator_names: list[str]
    body_actuator_names: list[str]
    grip_actuator_names: list[str]

    def __post_init__(self) -> None:
        all_names = list(self.all_actuator_names)
        body_names = list(self.body_actuator_names)
        grip_names = list(self.grip_actuator_names)

        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("body_actuator_names", body_names)
        _reject_duplicates("grip_actuator_names", grip_names)

        body_set = set(body_names)
        grip_set = set(grip_names)
        overlap = sorted(body_set & grip_set)
        if overlap:
            raise ValueError(f"actuators owned by both body and grip: {overlap}")

        all_set = set(all_names)
        missing_body = sorted(body_set - all_set)
        if missing_body:
            raise ValueError(f"body_actuator_names missing from all_actuator_names: {missing_body}")
        missing_grip = sorted(grip_set - all_set)
        if missing_grip:
            raise ValueError(f"grip_actuator_names missing from all_actuator_names: {missing_grip}")

        object.__setattr__(self, "all_actuator_names", all_names)
        object.__setattr__(self, "body_actuator_names", body_names)
        object.__setattr__(self, "grip_actuator_names", grip_names)

    @property
    def body_size(self) -> int:
        return len(self.body_actuator_names)

    @property
    def grip_size(self) -> int:
        return len(self.grip_actuator_names)

    def source_labels(self) -> list[str]:
        body_set = set(self.body_actuator_names)
        grip_set = set(self.grip_actuator_names)
        labels = []
        for name in self.all_actuator_names:
            if name in body_set:
                labels.append("body")
            elif name in grip_set:
                labels.append("grip")
            else:
                labels.append("zero")
        return labels

    def merge(self, *, body_action: np.ndarray, grip_action: np.ndarray) -> np.ndarray:
        body_action = _as_action("body_action", body_action, self.body_size)
        grip_action = _as_action("grip_action", grip_action, self.grip_size)

        body_values = dict(zip(self.body_actuator_names, body_action, strict=True))
        grip_values = dict(zip(self.grip_actuator_names, grip_action, strict=True))

        merged = np.zeros(len(self.all_actuator_names), dtype=float)
        for index, name in enumerate(self.all_actuator_names):
            if name in body_values:
                merged[index] = body_values[name]
            elif name in grip_values:
                merged[index] = grip_values[name]
        return merged


def _reject_duplicates(name: str, values: list[str]) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{name} contains duplicate actuator names: {duplicates}")


def _as_action(name: str, value: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def actuator_names_from_model(model) -> list[str]:
    if hasattr(model, "actuator_names"):
        return list(model.actuator_names)

    import mujoco

    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]


def resolve_actuator_groups(model, groups: list[str]) -> list[str]:
    all_names = actuator_names_from_model(model)
    group_sets = {
        "right_hand_fingers": RIGHT_HAND_FINGER_ACTUATORS,
        "right_wrist": RIGHT_WRIST_ACTUATORS,
        "right_forearm": RIGHT_FOREARM_ACTUATORS,
    }
    result: list[str] = []
    for group in groups:
        if group not in group_sets:
            raise ValueError(f"unknown actuator group: {group}")
        result.extend([name for name in all_names if name in group_sets[group] and name not in result])
    return result


def build_router_from_model_and_spec(model, body_manifest, residual_spec: dict) -> LayeredActuatorRouter:
    all_names = actuator_names_from_model(model)
    stage = residual_spec.get("stage")
    group_config = residual_spec.get("actuator_groups", {})
    if isinstance(group_config, dict):
        groups = group_config.get(stage, [])
    else:
        groups = group_config
    grip_names = resolve_actuator_groups(model, list(groups))
    return LayeredActuatorRouter(all_names, list(body_manifest.actuator_names), grip_names)
