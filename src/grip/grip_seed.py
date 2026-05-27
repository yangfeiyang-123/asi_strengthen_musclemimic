from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.grip.paths import REPO_ROOT, grip_seed_json_path


REQUIRED_SEED_KEYS = {
    "schema_version",
    "source_xml",
    "target_config",
    "qpos",
    "qvel",
    "right_hand_joint_names",
    "racket_freejoint_name",
    "racket_freejoint_qpos",
}


@dataclass(frozen=True)
class GripSeed:
    path: Path
    raw: dict[str, Any]
    source_xml: Path
    qpos: np.ndarray
    qvel: np.ndarray
    right_hand_joint_names: tuple[str, ...]
    racket_freejoint_name: str
    racket_freejoint_qpos: np.ndarray


def load_grip_seed(path: str | Path | None = None) -> GripSeed:
    seed_path = Path(path) if path is not None else grip_seed_json_path()
    with seed_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("grip seed JSON root must be an object")

    missing = sorted(REQUIRED_SEED_KEYS - set(raw))
    if missing:
        raise ValueError(f"grip seed missing required keys: {missing}")

    source_xml = _resolve_source_xml(seed_path, raw["source_xml"])
    model = mujoco.MjModel.from_xml_path(str(source_xml))
    qpos = _finite_vector(raw["qpos"], model.nq, "seed qpos")
    qvel = _finite_vector(raw["qvel"], model.nv, "seed qvel")
    racket_freejoint_qpos = _finite_vector(raw["racket_freejoint_qpos"], 7, "seed racket_freejoint_qpos")

    right_hand_joint_names = _string_tuple(raw["right_hand_joint_names"], "right_hand_joint_names")
    if not right_hand_joint_names:
        raise ValueError("grip seed right_hand_joint_names must be non-empty")
    for joint_name in right_hand_joint_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) < 0:
            raise ValueError(f"grip seed joint {joint_name!r} is missing from source model")

    racket_freejoint_name = str(raw["racket_freejoint_name"])
    racket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, racket_freejoint_name)
    if racket_id < 0:
        raise ValueError(f"grip seed racket freejoint {racket_freejoint_name!r} is missing from source model")
    if int(model.jnt_type[racket_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"grip seed racket joint {racket_freejoint_name!r} is not a freejoint")

    return GripSeed(
        path=seed_path,
        raw=raw,
        source_xml=source_xml,
        qpos=qpos,
        qvel=qvel,
        right_hand_joint_names=right_hand_joint_names,
        racket_freejoint_name=racket_freejoint_name,
        racket_freejoint_qpos=racket_freejoint_qpos,
    )


def apply_seed_right_hand_joints(seed: GripSeed, target_model: mujoco.MjModel, qpos: np.ndarray) -> None:
    source_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    for joint_name in seed.right_hand_joint_names:
        source_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        target_id = mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if source_id < 0:
            raise ValueError(f"seed source missing joint {joint_name!r}")
        if target_id < 0:
            raise ValueError(f"target model missing seed joint {joint_name!r}")

        width = _joint_qpos_width(source_model, source_id)
        if width != _joint_qpos_width(target_model, target_id):
            raise ValueError(f"joint {joint_name!r} qpos width differs between seed and target model")

        source_adr = int(source_model.jnt_qposadr[source_id])
        target_adr = int(target_model.jnt_qposadr[target_id])
        qpos[target_adr : target_adr + width] = seed.qpos[source_adr : source_adr + width]


def joint_shape_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or _joint_qpos_width(model, joint_id) != 1:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        value = float(qpos[adr])
        if bool(model.jnt_limited[joint_id]):
            lower = float(model.jnt_range[joint_id, 0])
            upper = float(model.jnt_range[joint_id, 1])
            metrics[joint_name] = {
                "value": value,
                "lower": lower,
                "upper": upper,
                "lower_margin": value - lower,
                "upper_margin": upper - value,
            }
        else:
            metrics[joint_name] = {"value": value}
    return metrics


def _resolve_source_xml(seed_path: Path, raw_source: object) -> Path:
    source = Path(str(raw_source))
    candidates = [source] if source.is_absolute() else [seed_path.parent / source, REPO_ROOT / source]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"grip seed source_xml does not exist: {raw_source!r}")


def _finite_vector(value: object, expected_size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"{label} must have shape ({expected_size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array.copy()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"grip seed {label} must be a list")
    return tuple(str(item) for item in value)


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1
