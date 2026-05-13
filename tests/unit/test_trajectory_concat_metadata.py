import numpy as np

import loco_mujoco.smpl.retargeting  # noqa: F401
from loco_mujoco.trajectory.dataclasses import (
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
    _traj_infos_compatible_for_concat,
)


def _trajectory_info(metadata):
    model = TrajectoryModel(
        njnt=2,
        jnt_type=np.array([0, 3], dtype=np.int32),
        nbody=1,
        body_rootid=np.array([0], dtype=np.int32),
        body_weldid=np.array([0], dtype=np.int32),
        body_mocapid=np.array([-1], dtype=np.int32),
        body_pos=np.zeros((1, 3), dtype=np.float32),
        body_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        body_ipos=np.zeros((1, 3), dtype=np.float32),
        body_iquat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        nsite=1,
        site_bodyid=np.array([0], dtype=np.int32),
        site_pos=np.zeros((1, 3), dtype=np.float32),
        site_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    return TrajectoryInfo(
        joint_names=["root", "hinge"],
        model=model,
        frequency=100.0,
        body_names=["body"],
        site_names=["site"],
        metadata=metadata,
    )


def test_concat_compatibility_ignores_non_structural_metadata():
    info_a = _trajectory_info({"source_cache": "motion_a.npz"})
    info_b = _trajectory_info({"source_cache": "motion_b.npz"})

    assert _traj_infos_compatible_for_concat(info_a, info_b, backend=np)


def test_concat_succeeds_when_only_metadata_differs():
    info_a = _trajectory_info({"source_cache": "motion_a.npz"})
    info_b = _trajectory_info({"source_cache": "motion_b.npz"})
    data_a = TrajectoryData(qpos=np.zeros((1, 8)), qvel=np.zeros((1, 7)), split_points=np.array([0, 1]))
    data_b = TrajectoryData(qpos=np.zeros((1, 8)), qvel=np.zeros((1, 7)), split_points=np.array([0, 1]))

    _, info = TrajectoryData.concatenate([data_a, data_b], [info_a, info_b], backend=np)

    assert info.metadata == info_a.metadata
