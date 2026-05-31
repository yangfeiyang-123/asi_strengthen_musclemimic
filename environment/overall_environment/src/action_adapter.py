from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionAdapterReport:
    source_action_size: int
    target_action_size: int
    mapped_count: int
    missing_in_target: list[str]
    extra_in_target: list[str]


class CheckpointToFullActionAdapter:
    def __init__(self, source_actuator_names: list[str], target_actuator_names: list[str]) -> None:
        self.source_actuator_names = list(source_actuator_names)
        self.target_actuator_names = list(target_actuator_names)
        _reject_duplicates("source_actuator_names", self.source_actuator_names)
        _reject_duplicates("target_actuator_names", self.target_actuator_names)

        target_set = set(self.target_actuator_names)
        missing = [name for name in self.source_actuator_names if name not in target_set]
        if missing:
            raise ValueError(f"source actuators missing in target: {missing}")

        source_set = set(self.source_actuator_names)
        self._extra = [name for name in self.target_actuator_names if name not in source_set]
        self._source_index = {name: index for index, name in enumerate(self.source_actuator_names)}

    def adapt(self, source_action: np.ndarray) -> np.ndarray:
        action = np.asarray(source_action, dtype=float)
        if action.shape != (len(self.source_actuator_names),):
            raise ValueError(f"source_action must have shape ({len(self.source_actuator_names)},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("source_action contains non-finite values")

        full = np.zeros(len(self.target_actuator_names), dtype=float)
        for target_index, name in enumerate(self.target_actuator_names):
            source_index = self._source_index.get(name)
            if source_index is not None:
                full[target_index] = action[source_index]
        return full

    def report(self) -> ActionAdapterReport:
        return ActionAdapterReport(
            source_action_size=len(self.source_actuator_names),
            target_action_size=len(self.target_actuator_names),
            mapped_count=len(self.source_actuator_names),
            missing_in_target=[],
            extra_in_target=list(self._extra),
        )


def _reject_duplicates(name: str, values: list[str]) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{name} contains duplicate actuator names: {duplicates}")
