"""Shared helpers for posing a MuJoCo model from a loco_mujoco Trajectory frame.

The trajectory qpos layout only matches the model qpos layout when the model was
built with the exact joint set the clip was retargeted for (bare-hand
MyoFullBody). Any extra joints — e.g. ``disable_fingers=False`` inserts 40
finger joints at qpos addresses 43..100, in the middle of the vector — shift
everything behind them, so a flat ``data.qpos[:n] = traj_qpos[:n]`` writes arm
and leg angles into finger joints and scrambles the whole pose. Always map by
joint name instead.
"""

from __future__ import annotations

import numpy as np

import mujoco


def qpos_from_trajectory_frame(model: mujoco.MjModel, traj, frame: int) -> np.ndarray:
    """Return a model-layout qpos for ``traj`` frame ``frame``, mapped by joint name.

    Joints missing from the trajectory (e.g. fingers) keep the model default
    (``qpos0``); trajectory joints missing from the model are skipped.
    """
    traj_qpos = np.asarray(traj.data.qpos)[frame]
    qpos = np.array(model.qpos0, dtype=float)
    for name, traj_ind in traj.info.joint_name2ind_qpos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        traj_ind = np.asarray(traj_ind, dtype=int)
        adr = int(model.jnt_qposadr[joint_id])
        qpos[adr : adr + len(traj_ind)] = traj_qpos[traj_ind]
    return qpos
