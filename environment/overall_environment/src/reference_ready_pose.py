"""Audited Stage-3 ready pose derived from a released reference trajectory.

The incoming-hit scene uses the standard-action joint pose at one registered
frame, while the root is rigidly aligned to the court coordinate system.  This
keeps the left-foot-forward stance in the source motion without inheriting the
capture camera's world yaw or translation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_FOOT_BODY_PAIRS = {
    "left": ("calcn_l", "toes_l"),
    "right": ("calcn_r", "toes_r"),
}


@dataclass(frozen=True)
class ReferenceReadyPoseSpec:
    """Immutable inputs and geometric acceptance gates for the ready pose."""

    path: Path
    frame_index: int
    sha256: str
    frequency_hz: float
    root_quat_wxyz: tuple[float, float, float, float]
    min_left_foot_forward_lead_m: float
    min_lateral_stance_width_m: float
    max_lateral_stance_width_m: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        resolve_path: Any,
    ) -> ReferenceReadyPoseSpec:
        allowed = {
            "path",
            "frame_index",
            "sha256",
            "frequency_hz",
            "root_quat_wxyz",
            "min_left_foot_forward_lead_m",
            "min_lateral_stance_width_m",
            "max_lateral_stance_width_m",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "scene.reference_ready_pose contains unknown keys: "
                + ", ".join(unknown)
            )
        try:
            path = Path(resolve_path(value["path"])).expanduser().resolve()
            frame_index = int(value["frame_index"])
            sha256 = str(value["sha256"]).lower()
            frequency_hz = float(value["frequency_hz"])
            root_quat = tuple(float(item) for item in value["root_quat_wxyz"])
            min_lead = float(value["min_left_foot_forward_lead_m"])
            min_width = float(value["min_lateral_stance_width_m"])
            max_width = float(value["max_lateral_stance_width_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "scene.reference_ready_pose requires path/frame/hash/frequency/root quaternion/stance gates"
            ) from exc
        if frame_index < 0:
            raise ValueError("reference ready-pose frame_index must be non-negative")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("reference ready-pose sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("reference ready-pose frequency_hz must be finite and positive")
        if len(root_quat) != 4 or not all(math.isfinite(item) for item in root_quat):
            raise ValueError("reference ready-pose root_quat_wxyz must contain four finite values")
        if not math.isclose(float(np.linalg.norm(root_quat)), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("reference ready-pose root_quat_wxyz must be unit length")
        if not (
            math.isfinite(min_lead)
            and min_lead > 0.0
            and math.isfinite(min_width)
            and min_width > 0.0
            and math.isfinite(max_width)
            and max_width > min_width
        ):
            raise ValueError("reference ready-pose stance gates are invalid")
        return cls(
            path=path,
            frame_index=frame_index,
            sha256=sha256,
            frequency_hz=frequency_hz,
            root_quat_wxyz=root_quat,
            min_left_foot_forward_lead_m=min_lead,
            min_lateral_stance_width_m=min_width,
            max_lateral_stance_width_m=max_width,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frame_index": self.frame_index,
            "sha256": self.sha256,
            "frequency_hz": self.frequency_hz,
            "root_quat_wxyz": list(self.root_quat_wxyz),
            "min_left_foot_forward_lead_m": self.min_left_foot_forward_lead_m,
            "min_lateral_stance_width_m": self.min_lateral_stance_width_m,
            "max_lateral_stance_width_m": self.max_lateral_stance_width_m,
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_ready_qpos(
    spec: ReferenceReadyPoseSpec,
    *,
    human_root_xy: tuple[float, float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load and court-align the registered frame without changing joint angles."""

    if not spec.path.is_file():
        raise FileNotFoundError(f"reference ready-pose trajectory is missing: {spec.path}")
    actual_sha = file_sha256(spec.path)
    if actual_sha != spec.sha256:
        raise ValueError(
            "reference ready-pose trajectory hash mismatch: "
            f"expected {spec.sha256}, got {actual_sha}"
        )
    with np.load(spec.path, allow_pickle=False) as payload:
        required = {"qpos", "joint_names", "frequency"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError("reference ready-pose trajectory is missing: " + ", ".join(missing))
        qpos_frames = np.asarray(payload["qpos"], dtype=float)
        joint_names = tuple(str(name) for name in np.asarray(payload["joint_names"]).tolist())
        frequency = float(np.asarray(payload["frequency"]).reshape(()))
    if qpos_frames.ndim != 2 or qpos_frames.shape[1] <= 7:
        raise ValueError("reference ready-pose qpos must be a finite [frames, nq] array")
    if not np.isfinite(qpos_frames).all():
        raise ValueError("reference ready-pose qpos contains non-finite values")
    if spec.frame_index >= qpos_frames.shape[0]:
        raise ValueError(
            f"reference ready-pose frame {spec.frame_index} is outside {qpos_frames.shape[0]} frames"
        )
    if not math.isclose(frequency, spec.frequency_hz, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"reference ready-pose frequency mismatch: expected {spec.frequency_hz}, got {frequency}"
        )
    if len(joint_names) == 0 or joint_names[0] != "root" or len(set(joint_names)) != len(joint_names):
        raise ValueError("reference ready-pose joint_names must be unique and begin with root")

    qpos = np.array(qpos_frames[spec.frame_index], dtype=float, copy=True)
    qpos[0:2] = np.asarray(human_root_xy, dtype=float)
    qpos[3:7] = np.asarray(spec.root_quat_wxyz, dtype=float)
    return qpos, joint_names


def validate_reference_ready_pose(
    model: mujoco.MjModel,
    ready_qpos: np.ndarray,
    spec: ReferenceReadyPoseSpec,
    *,
    human_root_xy: tuple[float, float],
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Rebuild the expected pose and prove its left-front/right-back geometry."""

    expected, source_joint_names = load_reference_ready_qpos(
        spec,
        human_root_xy=human_root_xy,
    )
    model_joint_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(len(source_joint_names))
    )
    joint_order_matches = model_joint_names == source_joint_names
    qpos = np.asarray(ready_qpos, dtype=float)
    qpos_shape_matches = qpos.ndim == 1 and qpos.shape[0] >= expected.shape[0]
    qpos_matches = bool(
        qpos_shape_matches
        and np.allclose(qpos[: expected.shape[0]], expected, rtol=0.0, atol=atol)
    )

    data = mujoco.MjData(model)
    if qpos.shape != (model.nq,):
        raise ValueError(f"ready qpos must have shape ({model.nq},), got {qpos.shape}")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    centers: dict[str, np.ndarray] = {}
    for side, body_names in _FOOT_BODY_PAIRS.items():
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
        if any(body_id < 0 for body_id in ids):
            raise ValueError(f"reference ready-pose validation is missing {side} foot bodies")
        centers[side] = np.mean(
            np.stack([np.asarray(data.xpos[body_id], dtype=float) for body_id in ids]),
            axis=0,
        )

    delta = centers["left"] - centers["right"]
    # The registered root quaternion aligns the court-forward axis with +X.
    forward_lead = float(delta[0])
    lateral_width = float(abs(delta[1]))
    stance_passed = bool(
        forward_lead >= spec.min_left_foot_forward_lead_m
        and spec.min_lateral_stance_width_m
        <= lateral_width
        <= spec.max_lateral_stance_width_m
    )
    passed = bool(joint_order_matches and qpos_matches and stance_passed)
    return {
        "schema_version": "stage3_reference_ready_pose_validation_v1",
        "source": spec.manifest(),
        "joint_order_matches": joint_order_matches,
        "qpos_matches_registered_frame": qpos_matches,
        "left_foot_center_xyz_m": centers["left"].tolist(),
        "right_foot_center_xyz_m": centers["right"].tolist(),
        "left_minus_right_xyz_m": delta.tolist(),
        "left_foot_forward_lead_m": forward_lead,
        "lateral_stance_width_m": lateral_width,
        "stance_gates": {
            "left_foot_forward": forward_lead >= spec.min_left_foot_forward_lead_m,
            "lateral_width_min": lateral_width >= spec.min_lateral_stance_width_m,
            "lateral_width_max": lateral_width <= spec.max_lateral_stance_width_m,
        },
        "passed": passed,
    }
