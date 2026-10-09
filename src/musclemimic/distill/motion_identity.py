"""Stable motion and rollout identities for distillation datasets.

``traj_no`` is an environment-local array index.  It is useful for stepping a
loaded trajectory handler, but it is not a dataset identity: independent
train/validation environments both start numbering at zero.  This module
derives persistent IDs from normalized motion paths and keeps the local index
mapping explicit.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np

MOTION_IDENTITY_VERSION = "motion_identity_v1"


def normalize_motion_path(path: str) -> str:
    """Return a platform-independent, Unicode-stable relative path."""
    value = unicodedata.normalize("NFC", str(path).strip()).replace("\\", "/")
    if not value:
        raise ValueError("motion path must be non-empty")
    # PurePosixPath removes duplicate separators and ``.`` components without
    # depending on whether the path exists on the current machine.
    normalized = PurePosixPath(value).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in {"", "."}:
        raise ValueError(f"motion path normalizes to an empty identity: {path!r}")
    return normalized


def normalize_relative_motion_path(path: str) -> str:
    """Return a stable repository/data-root-relative motion identity.

    Evidence manifests are portable artifacts.  They must never capture a
    workstation-private absolute path or escape their declared data root via
    ``..`` components.
    """

    normalized = normalize_motion_path(path)
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            "motion evidence path must be relative and may not contain '..': "
            f"{path!r}"
        )
    return normalized


def _stable_int63(*, kind: str, payload: Mapping[str, Any]) -> int:
    document = {
        "schema_version": MOTION_IDENTITY_VERSION,
        "kind": str(kind),
        **dict(payload),
    }
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Keep the sign bit clear so NPZ/NumPy/JAX int64 round trips are simple.
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def stable_motion_uid(path: str) -> int:
    """Derive a deterministic signed-int64-safe ID from a motion path."""
    return _stable_int63(kind="motion", payload={"path": normalize_motion_path(path)})


def stable_collection_uid(
    motion_paths: Iterable[str],
    *,
    split: str | None,
    seed: int,
    collector: str,
    run_tag: str | int | None = None,
) -> int:
    """Identify one collection run without using an unstable filesystem path."""
    normalized = [normalize_motion_path(path) for path in motion_paths]
    return _stable_int63(
        kind="collection",
        payload={
            "motion_paths": normalized,
            "split": None if split is None else str(split),
            "seed": int(seed),
            "collector": str(collector),
            "run_tag": run_tag,
        },
    )


def stable_rollout_uid(collection_uid: int, env_index: int, episode_index: int) -> int:
    """Identify one episode in one vector-environment lane."""
    return _stable_int63(
        kind="rollout",
        payload={
            "collection_uid": int(collection_uid),
            "env_index": int(env_index),
            "episode_index": int(episode_index),
        },
    )


@dataclass(frozen=True)
class MotionIdentityMap:
    """Exact mapping from environment-local trajectory indices to motion IDs."""

    motion_paths: tuple[str, ...]
    motion_uids: np.ndarray

    def __post_init__(self) -> None:
        paths = tuple(normalize_motion_path(path) for path in self.motion_paths)
        if len(set(paths)) != len(paths):
            raise ValueError("motion identity map contains duplicate normalized paths")
        expected = np.asarray([stable_motion_uid(path) for path in paths], dtype=np.int64)
        values = np.asarray(self.motion_uids, dtype=np.int64)
        if values.shape != expected.shape or not np.array_equal(values, expected):
            raise ValueError("motion_uids do not match the ordered normalized motion paths")
        if len({int(value) for value in values.tolist()}) != len(values):
            raise ValueError("motion UID collision detected")
        object.__setattr__(self, "motion_paths", paths)
        object.__setattr__(self, "motion_uids", values)

    @classmethod
    def from_paths(cls, motion_paths: Iterable[str]) -> MotionIdentityMap:
        paths = tuple(normalize_motion_path(path) for path in motion_paths)
        return cls(paths, np.asarray([stable_motion_uid(path) for path in paths], dtype=np.int64))

    def map_traj_no(self, traj_no: Any) -> np.ndarray:
        indices = np.asarray(traj_no)
        if not np.issubdtype(indices.dtype, np.integer):
            if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
                raise ValueError("traj_no must contain integer local trajectory indices")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= len(self.motion_paths)):
            invalid = np.unique(indices[(indices < 0) | (indices >= len(self.motion_paths))]).tolist()
            raise ValueError(
                "environment traj_no is outside the stable motion identity map: "
                f"invalid={invalid} num_motions={len(self.motion_paths)}"
            )
        return self.motion_uids[indices]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": MOTION_IDENTITY_VERSION,
            "motion_paths": list(self.motion_paths),
            "motion_uids": [int(value) for value in self.motion_uids],
        }


def resolve_config_motion_paths(config: Any) -> list[str]:
    """Resolve AMASS relative paths in the exact order used by the factory."""
    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    get = dataset_conf.get if hasattr(dataset_conf, "get") else lambda key, default=None: getattr(dataset_conf, key, default)
    paths: list[str] = []
    dataset_group = get("dataset_group")
    if dataset_group is not None:
        from loco_mujoco.task_factories.dataset_confs import (
            expand_amass_dataset_group_spec,
            get_amass_dataset_groups,
        )

        groups = get_amass_dataset_groups()
        for group_name in expand_amass_dataset_group_spec(dataset_group):
            if group_name not in groups:
                raise ValueError(f"unknown AMASS dataset group: {group_name}")
            paths.extend(str(path) for path in groups[group_name])
    relative = get("rel_dataset_path")
    if relative is not None:
        if isinstance(relative, str):
            paths.append(relative)
        else:
            paths.extend(str(path) for path in relative)
    # Match ImitationFactory's ordered de-duplication.
    paths = list(dict.fromkeys(normalize_motion_path(path) for path in paths))
    max_motions = get("max_motions")
    if max_motions is not None and len(paths) > int(max_motions):
        raise ValueError(
            "stable motion identity collection does not allow random max_motions sampling; "
            "supply an explicit deterministic rel_dataset_path list"
        )
    if not paths:
        raise ValueError("could not resolve any motion paths from distillation config")
    return paths


class RolloutIdentityTracker:
    """Host-side vector-lane episode IDs and strictly increasing step numbers."""

    def __init__(self, *, num_envs: int, collection_uid: int):
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.collection_uid = int(collection_uid)
        self.episode_index = np.zeros(self.num_envs, dtype=np.int64)
        self.rollout_step = np.zeros(self.num_envs, dtype=np.int32)

    def current(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env_index = np.arange(self.num_envs, dtype=np.int32)
        rollout_uid = np.asarray(
            [
                stable_rollout_uid(self.collection_uid, int(index), int(self.episode_index[index]))
                for index in range(self.num_envs)
            ],
            dtype=np.int64,
        )
        return rollout_uid, self.rollout_step.copy(), env_index

    def advance(self, done: Any) -> None:
        terminal = np.asarray(done, dtype=bool)
        if terminal.shape != (self.num_envs,):
            raise ValueError(f"done must have shape ({self.num_envs},), got {terminal.shape}")
        self.rollout_step = np.where(terminal, 0, self.rollout_step + 1).astype(np.int32)
        self.episode_index = self.episode_index + terminal.astype(np.int64)


def validate_environment_motion_identity(env: Any, identity_map: MotionIdentityMap) -> None:
    """Fail before collection if factory expansion and path identities disagree."""
    current = env
    seen: set[int] = set()
    trajectory_handler = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        trajectory_handler = getattr(current, "th", None)
        if trajectory_handler is not None:
            break
        current = getattr(current, "env", None)
    if trajectory_handler is None or not hasattr(trajectory_handler, "n_trajectories"):
        raise ValueError("distillation environment has no trajectory handler identity to validate")
    actual = int(trajectory_handler.n_trajectories)
    expected = len(identity_map.motion_paths)
    if actual != expected:
        raise ValueError(
            "stable motion identity count does not match loaded trajectories: "
            f"motion_paths={expected} env.th.n_trajectories={actual}; "
            "use explicit one-path-per-trajectory inputs and disable random sampling"
        )


def select_transition_traj_no(
    traj_no: Any,
    done: Any,
    *,
    final_traj_no: Any | None = None,
) -> np.ndarray:
    """Select the pre-reset trajectory identity for terminal transitions."""
    current = np.asarray(traj_no, dtype=np.int32)
    terminal = np.asarray(done, dtype=bool)
    if current.shape != terminal.shape:
        raise ValueError(f"traj_no and done shapes differ: {current.shape} vs {terminal.shape}")
    if final_traj_no is None:
        return current
    final = np.asarray(final_traj_no, dtype=np.int32)
    if final.shape != current.shape:
        raise ValueError(f"final_traj_no shape differs from traj_no: {final.shape} vs {current.shape}")
    return np.where(terminal, final, current).astype(np.int32)
