#!/usr/bin/env python3
"""Post-filter MyoFullBody retarget caches for smoother training trajectories."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _configure_paths(project_root: Path, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("MUSCLEMIMIC_AMASS_PATH", str(repo_root / "datasets" / "_global" / "amass_npz"))
    os.environ.setdefault("AMASS_PATH", os.environ["MUSCLEMIMIC_AMASS_PATH"])
    os.environ.setdefault(
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
        str(repo_root / "datasets" / "_global" / "muscle_trajectory" / "gmr_cache"),
    )
    os.environ.setdefault("CONVERTED_AMASS_PATH", os.environ["MUSCLEMIMIC_CONVERTED_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def _make_model(project_root: Path, repo_root: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    _configure_paths(project_root, repo_root)
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(disable_fingers=True)
    return env._model, mujoco.MjData(env._model)


def _velocity_limit_columns(values: np.ndarray, columns: np.ndarray, max_step: float) -> np.ndarray:
    """Limit per-frame scalar deltas for selected qpos columns."""
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    filtered = np.asarray(values, dtype=np.float64).copy()
    columns = np.asarray(columns, dtype=np.int64)
    if filtered.shape[0] < 2 or columns.size == 0:
        return filtered

    for frame in range(1, filtered.shape[0]):
        delta = filtered[frame, columns] - filtered[frame - 1, columns]
        filtered[frame, columns] = filtered[frame - 1, columns] + np.clip(delta, -max_step, max_step)
    return filtered


def _smooth_columns(values: np.ndarray, columns: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a zero-phase Gaussian low-pass filter to selected scalar qpos columns."""
    smoothed = np.asarray(values, dtype=np.float64).copy()
    columns = np.asarray(columns, dtype=np.int64)
    if sigma <= 0 or smoothed.shape[0] < 3 or columns.size == 0:
        return smoothed

    smoothed[:, columns] = gaussian_filter1d(smoothed[:, columns], sigma=sigma, axis=0, mode="nearest")
    return smoothed


def filter_qpos(
    qpos: np.ndarray,
    frequency: float,
    joint_sigma: float,
    max_joint_speed: float,
    root_sigma: float,
    max_root_speed: float,
) -> np.ndarray:
    """Smooth scalar joint qpos while preserving the root orientation."""
    filtered = np.asarray(qpos, dtype=np.float64).copy()
    if filtered.ndim != 2 or filtered.shape[1] < 8:
        raise ValueError(f"Expected qpos with shape (T, nq>=8), got {filtered.shape}")

    root_xyz = np.arange(0, 3)
    scalar_joints = np.arange(7, filtered.shape[1])
    filtered = _smooth_columns(filtered, scalar_joints, joint_sigma)
    filtered = _velocity_limit_columns(filtered, scalar_joints, max_joint_speed / frequency)
    filtered = _smooth_columns(filtered, scalar_joints, joint_sigma)

    filtered = _smooth_columns(filtered, root_xyz, root_sigma)
    filtered = _velocity_limit_columns(filtered, root_xyz, max_root_speed / frequency)
    filtered = _smooth_columns(filtered, root_xyz, root_sigma)

    quat_norm = np.linalg.norm(filtered[:, 3:7], axis=1, keepdims=True)
    filtered[:, 3:7] /= np.maximum(quat_norm, 1e-8)
    return filtered


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, frequency: float) -> np.ndarray:
    dt = 1.0 / float(frequency)
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    if len(qpos) < 2:
        return qvel

    forward = np.zeros((len(qpos) - 1, model.nv), dtype=np.float64)
    for frame in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, forward[frame], dt, qpos[frame], qpos[frame + 1])
    qvel[0] = forward[0]
    qvel[-1] = forward[-1]
    if len(qpos) > 2:
        qvel[1:-1] = 0.5 * (forward[:-1] + forward[1:])
    return qvel


def _site_ids(model: mujoco.MjModel, site_names: list[str]) -> np.ndarray:
    ids = []
    for name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site {name!r} is not present in MyoFullBody model")
        ids.append(site_id)
    return np.asarray(ids, dtype=np.int32)


def _forward_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
    site_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    n_frames = len(qpos)
    n_sites = len(site_ids)
    out = {
        "xpos": np.zeros((n_frames, model.nbody, 3), dtype=np.float32),
        "xquat": np.zeros((n_frames, model.nbody, 4), dtype=np.float32),
        "cvel": np.zeros((n_frames, model.nbody, 6), dtype=np.float32),
        "subtree_com": np.zeros((n_frames, model.nbody, 3), dtype=np.float32),
        "site_xpos": np.zeros((n_frames, n_sites, 3), dtype=np.float32),
        "site_xmat": np.zeros((n_frames, n_sites, 9), dtype=np.float32),
    }
    for frame in range(n_frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        out["xpos"][frame] = data.xpos
        out["xquat"][frame] = data.xquat
        out["cvel"][frame] = data.cvel
        out["subtree_com"][frame] = data.subtree_com
        out["site_xpos"][frame] = data.site_xpos[site_ids]
        out["site_xmat"][frame] = data.site_xmat[site_ids].reshape(n_sites, 9)
    return out


def _model_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> np.ndarray:
    return np.asarray([mujoco.mj_id2name(model, obj_type, i) or "" for i in range(count)])


def filter_cache_file(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    input_path: Path,
    output_path: Path,
    joint_sigma: float,
    max_joint_speed: float,
    root_sigma: float,
    max_root_speed: float,
) -> None:
    source = np.load(input_path, allow_pickle=True)
    frequency = float(np.asarray(source["frequency"]).reshape(-1)[0])
    site_names = [str(name) for name in source["site_names"]]
    site_ids = _site_ids(model, site_names)

    qpos = filter_qpos(
        np.asarray(source["qpos"], dtype=np.float64),
        frequency=frequency,
        joint_sigma=joint_sigma,
        max_joint_speed=max_joint_speed,
        root_sigma=root_sigma,
        max_root_speed=max_root_speed,
    )
    qvel = _compute_qvel(model, qpos, frequency)
    fk = _forward_kinematics(model, data, qpos, qvel, site_ids)

    payload = {key: source[key] for key in source.files}
    payload.update(fk)
    payload["qpos"] = qpos.astype(np.float32)
    payload["qvel"] = qvel.astype(np.float32)
    payload["frequency"] = np.asarray(frequency, dtype=np.float64)
    payload["body_names"] = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    payload["site_names"] = np.asarray(site_names)
    payload["metadata"] = {
        "source_cache": str(input_path),
        "filter": "gaussian_lowpass_plus_velocity_limit",
        "joint_sigma": float(joint_sigma),
        "max_joint_speed": float(max_joint_speed),
        "root_sigma": float(root_sigma),
        "max_root_speed": float(max_root_speed),
    }
    payload["njnt"] = np.asarray(model.njnt)
    payload["jnt_type"] = np.asarray(model.jnt_type)
    payload["nbody"] = np.asarray(model.nbody)
    payload["body_rootid"] = np.asarray(model.body_rootid)
    payload["body_weldid"] = np.asarray(model.body_weldid)
    payload["body_mocapid"] = np.asarray(model.body_mocapid)
    payload["body_pos"] = np.asarray(model.body_pos, dtype=np.float32)
    payload["body_quat"] = np.asarray(model.body_quat, dtype=np.float32)
    payload["body_ipos"] = np.asarray(model.body_ipos, dtype=np.float32)
    payload["body_iquat"] = np.asarray(model.body_iquat, dtype=np.float32)
    payload["nsite"] = np.asarray(len(site_ids))
    payload["site_bodyid"] = np.asarray(model.site_bodyid[site_ids])
    payload["site_pos"] = np.asarray(model.site_pos[site_ids], dtype=np.float32)
    payload["site_quat"] = np.asarray(model.site_quat[site_ids], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"[OK] {input_path} -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", action="append", required=True, help="Motion name without .npz under input root")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joint-sigma", type=float, default=2.5)
    parser.add_argument("--max-joint-speed", type=float, default=4.0, help="Scalar joint speed cap in rad/s")
    parser.add_argument("--root-sigma", type=float, default=1.0)
    parser.add_argument("--max-root-speed", type=float, default=2.0, help="Root translation speed cap in m/s")
    args = parser.parse_args()

    repo_root = _repo_root()
    project_root = _project_root()
    model, data = _make_model(project_root, repo_root)

    for motion in args.motion:
        filter_cache_file(
            model=model,
            data=data,
            input_path=args.input_root / f"{motion}.npz",
            output_path=args.output_root / f"{motion}.npz",
            joint_sigma=args.joint_sigma,
            max_joint_speed=args.max_joint_speed,
            root_sigma=args.root_sigma,
            max_root_speed=args.max_root_speed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
