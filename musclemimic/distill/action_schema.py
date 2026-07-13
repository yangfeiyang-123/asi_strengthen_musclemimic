"""Name-based action schemas shared by direct and latent distillation.

The order of MuJoCo actuators is part of a policy checkpoint's ABI.  Shape
checks alone cannot detect a 354-D action vector whose channels were silently
reordered, so every persisted schema carries a deterministic hash of the
ordered actuator names.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np


ACTION_SCHEMA_VERSION = "named_action_v1"


def ordered_schema_hash(*, kind: str, payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable ordered schema."""
    document = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "kind": str(kind),
        **payload,
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def actuator_schema_hash(actuator_names: Iterable[str]) -> str:
    """Hash an actuator vector including its exact channel order."""
    names = [str(name) for name in actuator_names]
    _reject_duplicate_names(names)
    return ordered_schema_hash(kind="actuator_vector", payload={"actuator_names": names})


@dataclass(frozen=True)
class ActionSelection:
    """Name-aligned selection from collected teacher actions to decoder actions."""

    source_actuator_names: tuple[str, ...]
    target_actuator_names: tuple[str, ...]
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        source = tuple(str(name) for name in self.source_actuator_names)
        target = tuple(str(name) for name in self.target_actuator_names)
        _reject_duplicate_names(source)
        _reject_duplicate_names(target)
        missing = [name for name in target if name not in source]
        if missing:
            raise ValueError(f"target actuator names missing from source action schema: {missing}")
        expected = np.asarray([source.index(name) for name in target], dtype=np.int32)
        indices = np.asarray(self.source_indices, dtype=np.int32)
        if indices.shape != expected.shape or not np.array_equal(indices, expected):
            raise ValueError("source_indices do not match target actuator name order")
        object.__setattr__(self, "source_actuator_names", source)
        object.__setattr__(self, "target_actuator_names", target)
        object.__setattr__(self, "source_indices", indices)

    @classmethod
    def from_names(
        cls,
        *,
        source_actuator_names: Iterable[str],
        target_actuator_names: Iterable[str] | None = None,
    ) -> "ActionSelection":
        source = tuple(str(name) for name in source_actuator_names)
        target = source if target_actuator_names is None else tuple(str(name) for name in target_actuator_names)
        return cls(
            source_actuator_names=source,
            target_actuator_names=target,
            source_indices=np.asarray([source.index(name) for name in target], dtype=np.int32),
        )

    @property
    def source_dim(self) -> int:
        return len(self.source_actuator_names)

    @property
    def target_dim(self) -> int:
        return len(self.target_actuator_names)

    @property
    def source_schema_hash(self) -> str:
        return actuator_schema_hash(self.source_actuator_names)

    @property
    def target_schema_hash(self) -> str:
        return actuator_schema_hash(self.target_actuator_names)

    def select(self, value: np.ndarray, *, field_name: str = "action") -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 1 or int(array.shape[-1]) != self.source_dim:
            raise ValueError(
                f"{field_name} last dimension must match source action schema "
                f"({self.source_dim}), got {array.shape}"
            )
        return np.take(array, self.source_indices, axis=-1)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SCHEMA_VERSION,
            "source_actuator_names": list(self.source_actuator_names),
            "target_actuator_names": list(self.target_actuator_names),
            "source_indices": self.source_indices.tolist(),
            "source_action_dim": self.source_dim,
            "target_action_dim": self.target_dim,
            "source_schema_hash": self.source_schema_hash,
            "target_schema_hash": self.target_schema_hash,
        }


def actuator_names_from_metadata(metadata: dict[str, Any], *, action_dim: int) -> list[str] | None:
    """Read and validate the canonical teacher action names from metadata."""
    names = metadata.get("actuator_names", metadata.get("action_actuator_names"))
    if names is None:
        return None
    result = [str(name) for name in names]
    _reject_duplicate_names(result)
    if len(result) != int(action_dim):
        raise ValueError(
            "distill metadata actuator_names length does not match teacher_action: "
            f"names={len(result)} action_dim={int(action_dim)}"
        )
    expected_hash = metadata.get("action_schema_hash")
    actual_hash = actuator_schema_hash(result)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise ValueError(
            "distill metadata action_schema_hash mismatch: "
            f"metadata={expected_hash} computed={actual_hash}"
        )
    return result


def _reject_duplicate_names(names: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        value = str(name)
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"actuator names contain duplicates: {duplicates}")
