#!/usr/bin/env python3
"""Append a zero-velocity stand tail to MyoFullBody retarget caches."""

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


DYNAMIC_KEYS = {
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "site_xmat",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_paths(project_root: Path, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("MUSCLEMIMIC_AMASS_PATH", str(project_root / "data" / "amass_npz"))
    os.environ.setdefault("AMASS_PATH", os.environ["MUSCLEMIMIC_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_CONVERTED_AMASS_PATH", str(repo_root / "caches" / "AMASS"))
    os.environ.setdefault("CONVERTED_AMASS_PATH", os.environ["MUSCLEMIMIC_CONVERTED_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def _make_model(project_root: Path, repo_root: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    _configure_paths(project_root, repo_root)
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(disable_fingers=True)
    return env._model, mujoco.MjData(env._model)


def _load_manifest(path: Path) -> list[str]:
    motions: list[str] = []
    for line in path.read_text().splitlines():
        motion = line.strip()
        if not motion or motion.startswith("#"):
            continue
        motions.append(motion.removesuffix(".npz"))
    if not motions:
        raise ValueError(f"empty manifest: {path}")
    return motions


def _smoothstep(alpha: np.ndarray) -> np.ndarray:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _normalize_quat(qpos: np.ndarray) -> np.ndarray:
    out = np.asarray(qpos, dtype=np.float64).copy()
    norm = np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    out[:, 3:7] /= np.maximum(norm, 1e-8)
    return out


def build_stand_tail_qpos(
    qpos: np.ndarray,
    frequency: float,
    hold_seconds: float,
    settle_seconds: float,
    anchor_window_seconds: float,
) -> tuple[np.ndarray, int, int]:
    """Return qpos with a smoothed final-pose anchor and a zero-motion tail."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] < 8:
        raise ValueError(f"expected qpos with shape (T>=2, nq>=8), got {qpos.shape}")
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if hold_seconds <= 0:
        raise ValueError(f"hold_seconds must be positive, got {hold_seconds}")
    if settle_seconds < 0:
        raise ValueError(f"settle_seconds must be non-negative, got {settle_seconds}")
    if anchor_window_seconds <= 0:
        raise ValueError(f"anchor_window_seconds must be positive, got {anchor_window_seconds}")

    hold_frames = max(1, int(round(hold_seconds * frequency)))
    settle_frames = max(0, int(round(settle_seconds * frequency)))
    window = max(1, min(qpos.shape[0], int(round(anchor_window_seconds * frequency))))

    anchor = qpos[-1].copy()
    # Use the last-window median for scalar joints to remove terminal WHAM/GMR
    # jitter, but keep root translation and orientation at the actual end frame.
    anchor[7:] = np.median(qpos[-window:, 7:], axis=0)

    tail_parts: list[np.ndarray] = []
    if settle_frames > 0:
        alpha = _smoothstep(np.linspace(0.0, 1.0, settle_frames + 1, dtype=np.float64)[1:])
        settle = (1.0 - alpha[:, None]) * qpos[-1][None, :] + alpha[:, None] * anchor[None, :]
        tail_parts.append(settle)
    tail_parts.append(np.repeat(anchor[None, :], hold_frames, axis=0))

    extended = np.concatenate([qpos, *tail_parts], axis=0)
    return _normalize_quat(extended), settle_frames, hold_frames


def compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, frequency: float) -> np.ndarray:
    dt = 1.0 / float(frequency)
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    if len(qpos) < 2:
        return qvel

    forward = np.zeros((len(qpos) - 1, model.nv), dtype=np.float64)
    for frame in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, forward[frame], dt, qpos[frame], qpos[frame + 1])
    qvel[:-1] = forward
    qvel[-1] = 0.0
    return qvel


def _site_ids(model: mujoco.MjModel, site_names: list[str]) -> np.ndarray:
    ids = []
    for name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"site {name!r} is not present in MyoFullBody")
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
    return np.asarray([mujoco.mj_id2name(model, obj_type, idx) or "" for idx in range(count)])


def _output_motion_name(source_motion: str, output_namespace: str) -> str:
    return f"{output_namespace}/{Path(source_motion).name}"


def extend_cache_file(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    input_path: Path,
    output_path: Path,
    *,
    hold_seconds: float,
    settle_seconds: float,
    anchor_window_seconds: float,
) -> dict[str, int | float | str]:
    source = np.load(input_path, allow_pickle=True)
    frequency = float(np.asarray(source["frequency"]).reshape(-1)[0])
    site_names = [str(name) for name in source["site_names"]]
    site_ids = _site_ids(model, site_names)

    original_frames = int(source["qpos"].shape[0])
    qpos, settle_frames, hold_frames = build_stand_tail_qpos(
        np.asarray(source["qpos"], dtype=np.float64),
        frequency=frequency,
        hold_seconds=hold_seconds,
        settle_seconds=settle_seconds,
        anchor_window_seconds=anchor_window_seconds,
    )
    qvel = compute_qvel(model, qpos, frequency)
    fk = _forward_kinematics(model, data, qpos, qvel, site_ids)

    payload = {key: source[key] for key in source.files if key not in DYNAMIC_KEYS}
    payload.update(fk)
    payload["qpos"] = qpos.astype(np.float32)
    payload["qvel"] = qvel.astype(np.float32)
    payload["frequency"] = np.asarray(frequency, dtype=np.float64)
    payload["split_points"] = np.asarray([0, len(qpos)], dtype=np.int32)
    payload["body_names"] = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    payload["site_names"] = np.asarray(site_names)
    payload["metadata"] = {
        "source_cache": str(input_path),
        "transform": "append_zero_velocity_stand_tail",
        "original_frames": original_frames,
        "settle_frames": int(settle_frames),
        "hold_frames": int(hold_frames),
        "frequency": float(frequency),
        "settle_seconds": float(settle_seconds),
        "hold_seconds": float(hold_seconds),
        "anchor_window_seconds": float(anchor_window_seconds),
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
    return {
        "input": str(input_path),
        "output": str(output_path),
        "original_frames": original_frames,
        "extended_frames": int(len(qpos)),
        "settle_frames": int(settle_frames),
        "hold_frames": int(hold_frames),
        "frequency": float(frequency),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("BadmintonMimic/manifests/10trajectories_smooth_27_list.txt"))
    parser.add_argument("--input-root", type=Path, default=Path("caches/AMASS/MyoFullBody/gmr"))
    parser.add_argument("--output-root", type=Path, default=Path("caches/AMASS/MyoFullBody/gmr"))
    parser.add_argument("--output-namespace", default="10trajectories_smooth_stand_tail")
    parser.add_argument("--output-manifest", type=Path, default=Path("BadmintonMimic/manifests/10trajectories_smooth_27_stand_tail_list.txt"))
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--anchor-window-seconds", type=float, default=0.25)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    project_root = _project_root()
    model, data = _make_model(project_root, repo_root)

    motions = _load_manifest(args.manifest)
    output_motions: list[str] = []
    for motion in motions:
        output_motion = _output_motion_name(motion, args.output_namespace)
        report = extend_cache_file(
            model,
            data,
            args.input_root / f"{motion}.npz",
            args.output_root / f"{output_motion}.npz",
            hold_seconds=args.hold_seconds,
            settle_seconds=args.settle_seconds,
            anchor_window_seconds=args.anchor_window_seconds,
        )
        output_motions.append(output_motion)
        print(
            "[OK] {input} -> {output} "
            "({original_frames} -> {extended_frames} frames, hold={hold_frames})".format(**report)
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("\n".join(output_motions) + "\n")
    print(f"[OK] wrote manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
