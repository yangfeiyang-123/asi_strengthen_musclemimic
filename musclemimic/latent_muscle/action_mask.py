"""Action partition helpers for body latent control and distal correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionMask:
    """Name-based split between body decoder actions and correction actions."""

    all_actuator_names: list[str]
    body_actuator_names: list[str]
    correction_actuator_names: list[str]
    body_indices: np.ndarray
    correction_indices: np.ndarray

    def __post_init__(self) -> None:
        all_names = list(self.all_actuator_names)
        body_names = list(self.body_actuator_names)
        correction_names = list(self.correction_actuator_names)
        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("body_actuator_names", body_names)
        _reject_duplicates("correction_actuator_names", correction_names)

        all_set = set(all_names)
        body_set = set(body_names)
        correction_set = set(correction_names)
        overlap = sorted(body_set & correction_set)
        if overlap:
            raise ValueError(f"actuators owned by both body and correction: {overlap}")
        missing = sorted((body_set | correction_set) - all_set)
        if missing:
            raise ValueError(f"actuator names missing from all_actuator_names: {missing}")
        uncovered = sorted(all_set - (body_set | correction_set))
        if uncovered:
            raise ValueError(
                f"actuators not assigned to body or correction (split/merge would silently "
                f"drop/zero them): {uncovered}"
            )

        expected_body = [all_names.index(name) for name in body_names]
        expected_correction = [all_names.index(name) for name in correction_names]
        body_indices = np.asarray(self.body_indices, dtype=np.int32)
        correction_indices = np.asarray(self.correction_indices, dtype=np.int32)
        if body_indices.shape != (len(body_names),) or not np.array_equal(body_indices, expected_body):
            raise ValueError("body_indices must match body_actuator_names in all_actuator_names order")
        if correction_indices.shape != (len(correction_names),) or not np.array_equal(
            correction_indices,
            expected_correction,
        ):
            raise ValueError("correction_indices must match correction_actuator_names in all_actuator_names order")

        object.__setattr__(self, "all_actuator_names", all_names)
        object.__setattr__(self, "body_actuator_names", body_names)
        object.__setattr__(self, "correction_actuator_names", correction_names)
        object.__setattr__(self, "body_indices", body_indices)
        object.__setattr__(self, "correction_indices", correction_indices)

    @classmethod
    def from_correction_actuators(
        cls,
        *,
        all_actuator_names: list[str],
        correction_actuator_names: list[str],
    ) -> "ActionMask":
        """Build a body/correction split, preserving the full actuator order."""
        all_names = list(all_actuator_names)
        correction_names = list(correction_actuator_names)
        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("correction_actuator_names", correction_names)
        all_set = set(all_names)
        missing = [name for name in correction_names if name not in all_set]
        if missing:
            raise ValueError(f"correction_actuator_names missing from all_actuator_names: {missing}")
        correction_set = set(correction_names)
        body_names = [name for name in all_names if name not in correction_set]
        return cls(
            all_actuator_names=all_names,
            body_actuator_names=body_names,
            correction_actuator_names=correction_names,
            body_indices=np.asarray([all_names.index(name) for name in body_names], dtype=np.int32),
            correction_indices=np.asarray([all_names.index(name) for name in correction_names], dtype=np.int32),
        )

    @classmethod
    def from_layered_router(cls, router) -> "ActionMask":
        """Build a mask from a LayeredActuatorRouter-like object."""
        return cls(
            all_actuator_names=list(router.all_actuator_names),
            body_actuator_names=list(router.body_actuator_names),
            correction_actuator_names=list(router.grip_actuator_names),
            body_indices=np.asarray(
                [list(router.all_actuator_names).index(name) for name in router.body_actuator_names],
                dtype=np.int32,
            ),
            correction_indices=np.asarray(
                [list(router.all_actuator_names).index(name) for name in router.grip_actuator_names],
                dtype=np.int32,
            ),
        )

    @property
    def action_size(self) -> int:
        return len(self.all_actuator_names)

    @property
    def body_size(self) -> int:
        return len(self.body_actuator_names)

    @property
    def correction_size(self) -> int:
        return len(self.correction_actuator_names)

    def split(self, full_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = _as_action("full_action", full_action, self.action_size)
        return action[self.body_indices], action[self.correction_indices]

    def merge(self, body_action: np.ndarray, correction_action: np.ndarray) -> np.ndarray:
        body = _as_action("body_action", body_action, self.body_size)
        correction = _as_action("correction_action", correction_action, self.correction_size)
        full = np.zeros(self.action_size, dtype=float)
        full[self.body_indices] = body
        full[self.correction_indices] = correction
        return full

    def assert_matches_partitions(
        self,
        *,
        body_actuator_names: list[str],
        correction_actuator_names: list[str],
    ) -> None:
        """Fail loudly when the latent mask disagrees with the runtime router."""
        body_names = list(body_actuator_names)
        correction_names = list(correction_actuator_names)
        if body_names != self.body_actuator_names:
            raise ValueError(
                "body actuator partition mismatch: "
                f"mask={self.body_actuator_names} expected={body_names}"
            )
        if correction_names != self.correction_actuator_names:
            raise ValueError(
                "correction actuator partition mismatch: "
                f"mask={self.correction_actuator_names} expected={correction_names}"
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


def _as_action(name: str, value: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array
