"""Action ownership for body latent control, grip correction, and neutral channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musclemimic.distill.action_schema import ordered_schema_hash


@dataclass(frozen=True)
class ActionMask:
    """Name-based three-way action ownership contract.

    ``body`` is produced by the latent decoder, ``correction`` is owned by a
    separate controller (for example the 31 right-hand grip actuators), and
    ``neutral`` is written explicitly to fixed values (for example the 31
    left-hand finger actuators).  Requiring all three partitions prevents
    unowned channels from being silently dropped or left as stale actions.
    """

    all_actuator_names: list[str]
    body_actuator_names: list[str]
    correction_actuator_names: list[str]
    body_indices: np.ndarray
    correction_indices: np.ndarray
    neutral_actuator_names: list[str] | None = None
    neutral_indices: np.ndarray | None = None
    neutral_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        all_names = list(self.all_actuator_names)
        body_names = list(self.body_actuator_names)
        correction_names = list(self.correction_actuator_names)
        neutral_names = list(self.neutral_actuator_names or [])
        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("body_actuator_names", body_names)
        _reject_duplicates("correction_actuator_names", correction_names)
        _reject_duplicates("neutral_actuator_names", neutral_names)

        all_set = set(all_names)
        body_set = set(body_names)
        correction_set = set(correction_names)
        neutral_set = set(neutral_names)
        overlap = sorted((body_set & correction_set) | (body_set & neutral_set) | (correction_set & neutral_set))
        if overlap:
            raise ValueError(f"actuators assigned to more than one action partition: {overlap}")
        missing = sorted((body_set | correction_set | neutral_set) - all_set)
        if missing:
            raise ValueError(f"actuator names missing from all_actuator_names: {missing}")
        uncovered = sorted(all_set - (body_set | correction_set | neutral_set))
        if uncovered:
            raise ValueError(
                f"actuators not assigned to body, correction, or neutral (split/merge would silently "
                f"drop/zero them): {uncovered}"
            )

        expected_body = [all_names.index(name) for name in body_names]
        expected_correction = [all_names.index(name) for name in correction_names]
        expected_neutral = [all_names.index(name) for name in neutral_names]
        body_indices = np.asarray(self.body_indices, dtype=np.int32)
        correction_indices = np.asarray(self.correction_indices, dtype=np.int32)
        neutral_indices = np.asarray(
            expected_neutral if self.neutral_indices is None else self.neutral_indices,
            dtype=np.int32,
        )
        if body_indices.shape != (len(body_names),) or not np.array_equal(body_indices, expected_body):
            raise ValueError("body_indices must match body_actuator_names in all_actuator_names order")
        if correction_indices.shape != (len(correction_names),) or not np.array_equal(
            correction_indices,
            expected_correction,
        ):
            raise ValueError("correction_indices must match correction_actuator_names in all_actuator_names order")
        if neutral_indices.shape != (len(neutral_names),) or not np.array_equal(neutral_indices, expected_neutral):
            raise ValueError("neutral_indices must match neutral_actuator_names in all_actuator_names order")
        neutral_values = np.asarray(
            np.zeros(len(neutral_names), dtype=float) if self.neutral_values is None else self.neutral_values,
            dtype=float,
        )
        if neutral_values.shape != (len(neutral_names),):
            raise ValueError(
                f"neutral_values must have shape ({len(neutral_names)},), got {neutral_values.shape}"
            )
        if not np.isfinite(neutral_values).all():
            raise ValueError("neutral_values contains non-finite values")

        object.__setattr__(self, "all_actuator_names", all_names)
        object.__setattr__(self, "body_actuator_names", body_names)
        object.__setattr__(self, "correction_actuator_names", correction_names)
        object.__setattr__(self, "neutral_actuator_names", neutral_names)
        object.__setattr__(self, "body_indices", body_indices)
        object.__setattr__(self, "correction_indices", correction_indices)
        object.__setattr__(self, "neutral_indices", neutral_indices)
        object.__setattr__(self, "neutral_values", neutral_values)

    @classmethod
    def from_correction_actuators(
        cls,
        *,
        all_actuator_names: list[str],
        correction_actuator_names: list[str],
        neutral_actuator_names: list[str] | None = None,
        neutral_values: np.ndarray | list[float] | None = None,
    ) -> "ActionMask":
        """Build a three-way split, preserving the full actuator order."""
        all_names = list(all_actuator_names)
        correction_names = list(correction_actuator_names)
        neutral_names = list(neutral_actuator_names or [])
        _reject_duplicates("all_actuator_names", all_names)
        _reject_duplicates("correction_actuator_names", correction_names)
        _reject_duplicates("neutral_actuator_names", neutral_names)
        all_set = set(all_names)
        missing = [name for name in correction_names + neutral_names if name not in all_set]
        if missing:
            raise ValueError(f"partition actuator names missing from all_actuator_names: {missing}")
        correction_set = set(correction_names)
        neutral_set = set(neutral_names)
        overlap = sorted(correction_set & neutral_set)
        if overlap:
            raise ValueError(f"actuators assigned to both correction and neutral: {overlap}")
        body_names = [name for name in all_names if name not in correction_set and name not in neutral_set]
        return cls(
            all_actuator_names=all_names,
            body_actuator_names=body_names,
            correction_actuator_names=correction_names,
            body_indices=np.asarray([all_names.index(name) for name in body_names], dtype=np.int32),
            correction_indices=np.asarray([all_names.index(name) for name in correction_names], dtype=np.int32),
            neutral_actuator_names=neutral_names,
            neutral_indices=np.asarray([all_names.index(name) for name in neutral_names], dtype=np.int32),
            neutral_values=None if neutral_values is None else np.asarray(neutral_values, dtype=float),
        )

    @classmethod
    def from_partitions(
        cls,
        *,
        all_actuator_names: list[str],
        body_actuator_names: list[str],
        correction_actuator_names: list[str],
        neutral_actuator_names: list[str],
        neutral_values: np.ndarray | list[float] | None = None,
    ) -> "ActionMask":
        """Build an explicit body/correction/neutral ownership contract."""
        all_names = list(all_actuator_names)
        return cls(
            all_actuator_names=all_names,
            body_actuator_names=list(body_actuator_names),
            correction_actuator_names=list(correction_actuator_names),
            neutral_actuator_names=list(neutral_actuator_names),
            body_indices=np.asarray([all_names.index(name) for name in body_actuator_names], dtype=np.int32),
            correction_indices=np.asarray([all_names.index(name) for name in correction_actuator_names], dtype=np.int32),
            neutral_indices=np.asarray([all_names.index(name) for name in neutral_actuator_names], dtype=np.int32),
            neutral_values=None if neutral_values is None else np.asarray(neutral_values, dtype=float),
        )

    @classmethod
    def from_layered_router(cls, router) -> "ActionMask":
        """Build a mask from a LayeredActuatorRouter-like object."""
        all_names = list(router.all_actuator_names)
        body_names = list(router.body_actuator_names)
        correction_names = list(router.grip_actuator_names)
        assigned = set(body_names) | set(correction_names)
        neutral_names = [name for name in all_names if name not in assigned]
        return cls.from_partitions(
            all_actuator_names=all_names,
            body_actuator_names=body_names,
            correction_actuator_names=correction_names,
            neutral_actuator_names=neutral_names,
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

    @property
    def neutral_size(self) -> int:
        return len(self.neutral_actuator_names)

    @property
    def schema_hash(self) -> str:
        return ordered_schema_hash(
            kind="latent_action_partitions",
            payload={
                "all_actuator_names": self.all_actuator_names,
                "body_actuator_names": self.body_actuator_names,
                "correction_actuator_names": self.correction_actuator_names,
                "neutral_actuator_names": self.neutral_actuator_names,
                "neutral_values": self.neutral_values.tolist(),
            },
        )

    def split(self, full_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = _as_action("full_action", full_action, self.action_size)
        return action[self.body_indices], action[self.correction_indices]

    def split_three(self, full_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = _as_action("full_action", full_action, self.action_size)
        return action[self.body_indices], action[self.correction_indices], action[self.neutral_indices]

    def merge(self, body_action: np.ndarray, correction_action: np.ndarray) -> np.ndarray:
        body = _as_action("body_action", body_action, self.body_size)
        correction = _as_action("correction_action", correction_action, self.correction_size)
        full = np.zeros(self.action_size, dtype=float)
        full[self.body_indices] = body
        full[self.correction_indices] = correction
        full[self.neutral_indices] = self.neutral_values
        return full

    def merge_three(
        self,
        body_action: np.ndarray,
        correction_action: np.ndarray,
        neutral_action: np.ndarray,
    ) -> np.ndarray:
        """Merge all partitions while requiring an explicit neutral vector."""
        body = _as_action("body_action", body_action, self.body_size)
        correction = _as_action("correction_action", correction_action, self.correction_size)
        neutral = _as_action("neutral_action", neutral_action, self.neutral_size)
        full = np.zeros(self.action_size, dtype=float)
        full[self.body_indices] = body
        full[self.correction_indices] = correction
        full[self.neutral_indices] = neutral
        return full

    def assert_matches_partitions(
        self,
        *,
        body_actuator_names: list[str],
        correction_actuator_names: list[str],
        neutral_actuator_names: list[str] | None = None,
    ) -> None:
        """Fail loudly when the latent mask disagrees with the runtime router."""
        body_names = list(body_actuator_names)
        correction_names = list(correction_actuator_names)
        neutral_names = self.neutral_actuator_names if neutral_actuator_names is None else list(neutral_actuator_names)
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
        if neutral_names != self.neutral_actuator_names:
            raise ValueError(
                "neutral actuator partition mismatch: "
                f"mask={self.neutral_actuator_names} expected={neutral_names}"
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
