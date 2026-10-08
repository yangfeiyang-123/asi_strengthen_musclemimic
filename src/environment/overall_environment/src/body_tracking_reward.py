from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


DEFAULT_EXCLUDED_BODY_TRACKING_JOINTS = frozenset(
    {
        "cmc_flexion_r",
        "cmc_abduction_r",
        "mp_flexion_r",
        "ip_flexion_r",
        "mcp2_flexion_r",
        "mcp2_abduction_r",
        "pm2_flexion_r",
        "md2_flexion_r",
        "mcp3_flexion_r",
        "mcp3_abduction_r",
        "pm3_flexion_r",
        "md3_flexion_r",
        "mcp4_flexion_r",
        "mcp4_abduction_r",
        "pm4_flexion_r",
        "md4_flexion_r",
        "mcp5_flexion_r",
        "mcp5_abduction_r",
        "pm5_flexion_r",
        "md5_flexion_r",
        "overall_racket_free",
        "overall_shuttle_free",
    }
)


@dataclass(frozen=True)
class BodyTrackingReport:
    error: float
    qpos_mse: float
    qvel_mse: float
    tracked_joint_count: int
    tracked_qpos_count: int
    tracked_qvel_count: int
    reference: str


def trajectory_body_tracking_error(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    reference_qvel: np.ndarray,
    excluded_joint_names: set[str] | frozenset[str] = DEFAULT_EXCLUDED_BODY_TRACKING_JOINTS,
    qvel_weight: float = 0.01,
) -> BodyTrackingReport:
    qpos_pairs: list[tuple[int, int, int]] = []
    qvel_pairs: list[tuple[int, int, int]] = []
    tracked_joint_count = 0
    excluded = set(excluded_joint_names)

    for ref_joint_id in range(reference_model.njnt):
        joint_name = mujoco.mj_id2name(reference_model, mujoco.mjtObj.mjOBJ_JOINT, ref_joint_id)
        if not joint_name or joint_name in excluded:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_width = _joint_qpos_width(reference_model, ref_joint_id)
        qvel_width = _joint_dof_width(reference_model, ref_joint_id)
        if qpos_width != _joint_qpos_width(model, joint_id):
            raise ValueError(f"qpos width mismatch for body tracking joint {joint_name!r}")
        if qvel_width != _joint_dof_width(model, joint_id):
            raise ValueError(f"qvel width mismatch for body tracking joint {joint_name!r}")
        model_qpos_start = int(model.jnt_qposadr[joint_id])
        reference_qpos_start = int(reference_model.jnt_qposadr[ref_joint_id])
        if joint_name == "root" and qpos_width == 7:
            model_qpos_start += 2
            reference_qpos_start += 2
            qpos_width = 5
        qpos_pairs.append((model_qpos_start, reference_qpos_start, qpos_width))
        qvel_pairs.append(
            (
                int(model.jnt_dofadr[joint_id]),
                int(reference_model.jnt_dofadr[ref_joint_id]),
                qvel_width,
            )
        )
        tracked_joint_count += 1

    if not qpos_pairs:
        raise ValueError("no common body tracking joints were found")

    qpos_delta = _stack_deltas(np.asarray(data.qpos, dtype=float), np.asarray(reference_qpos, dtype=float), qpos_pairs)
    qvel_delta = _stack_deltas(np.asarray(data.qvel, dtype=float), np.asarray(reference_qvel, dtype=float), qvel_pairs)
    qpos_mse = float(np.mean(np.square(qpos_delta))) if qpos_delta.size else 0.0
    qvel_mse = float(np.mean(np.square(qvel_delta))) if qvel_delta.size else 0.0
    return BodyTrackingReport(
        error=float(qpos_mse + qvel_weight * qvel_mse),
        qpos_mse=qpos_mse,
        qvel_mse=qvel_mse,
        tracked_joint_count=tracked_joint_count,
        tracked_qpos_count=int(qpos_delta.size),
        tracked_qvel_count=int(qvel_delta.size),
        reference="trajectory",
    )


def reset_snapshot_body_tracking_error(qpos: np.ndarray, reference_qpos: np.ndarray) -> BodyTrackingReport:
    delta = np.asarray(qpos, dtype=float) - np.asarray(reference_qpos, dtype=float)
    qpos_mse = float(np.mean(np.square(delta))) if delta.size else 0.0
    return BodyTrackingReport(
        error=qpos_mse,
        qpos_mse=qpos_mse,
        qvel_mse=0.0,
        tracked_joint_count=0,
        tracked_qpos_count=int(delta.size),
        tracked_qvel_count=0,
        reference="reset_snapshot",
    )


def _stack_deltas(current: np.ndarray, reference: np.ndarray, spans: list[tuple[int, int, int]]) -> np.ndarray:
    return np.concatenate(
        [current[cur_start : cur_start + width] - reference[ref_start : ref_start + width] for cur_start, ref_start, width in spans]
    )


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_dof_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1
