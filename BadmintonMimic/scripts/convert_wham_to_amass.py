#!/usr/bin/env python3
"""Convert common WHAM outputs to MuscleMimic/AMASS-style NPZ files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


POSE_KEYS = ("poses", "pose", "smpl_pose", "theta")
BODY_POSE_KEYS = ("body_pose", "smpl_body_pose", "pose_body")
GLOBAL_ORIENT_KEYS = ("global_orient", "root_orient", "root_pose", "orient")
TRANS_KEYS = ("trans", "transl", "translation", "root_trans", "root_translation")
BETAS_KEYS = ("betas", "shape", "smpl_betas")
FPS_KEYS = ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate")


def _load_any(path: Path) -> Any:
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        return data.item() if data.shape == () else data
    if path.suffix in {".pkl", ".pickle"}:
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as pickle_error:
            try:
                import joblib

                return joblib.load(path)
            except Exception as joblib_error:
                raise ValueError(
                    f"Could not load {path} with pickle or joblib. "
                    f"pickle_error={pickle_error!r}; joblib_error={joblib_error!r}"
                ) from joblib_error
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def _unwrap_singleton(data: Any) -> Any:
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict) and len(data) == 1:
        value = next(iter(data.values()))
        if isinstance(value, dict):
            return value
    return data


def _num_pose_frames(data: dict[str, Any]) -> int:
    for key in (*POSE_KEYS, "pose_world", *BODY_POSE_KEYS):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim > 0:
                return int(arr.shape[0])
    return -1


def _select_track(data: Any, track_id: str | None) -> Any:
    if not isinstance(data, dict) or not data:
        return data

    if all(isinstance(value, dict) for value in data.values()):
        if track_id is not None:
            for key, value in data.items():
                if str(key) == track_id:
                    return value
                try:
                    if int(key) == int(track_id):
                        return value
                except (TypeError, ValueError):
                    pass
            raise KeyError(f"Track id {track_id!r} not found. Available tracks: {list(data.keys())}")

        best_key, best_value = max(data.items(), key=lambda item: _num_pose_frames(item[1]))
        print(f"[INFO] Multiple WHAM tracks found; selected longest track {best_key!r}.")
        return best_value

    return data


def _track_sort_key(key: Any) -> tuple[int, str]:
    try:
        return int(key), str(key)
    except (TypeError, ValueError):
        return 0, str(key)


def _merge_tracks(data: Any) -> Any:
    if not isinstance(data, dict) or not data or not all(isinstance(value, dict) for value in data.values()):
        return data
    if not all("frame_ids" in value for value in data.values()):
        raise KeyError("Cannot merge WHAM tracks without frame_ids in every track.")

    ordered_tracks = sorted(data.items(), key=lambda item: _track_sort_key(item[0]))
    frame_map: dict[int, tuple[dict[str, Any], int]] = {}
    for _track_key, track in ordered_tracks:
        frame_ids = np.asarray(track["frame_ids"])
        for local_idx, frame_id in enumerate(frame_ids):
            frame_map[int(frame_id)] = (track, local_idx)

    merged: dict[str, Any] = {"frame_ids": np.asarray(sorted(frame_map), dtype=np.int32)}
    candidate_keys = set().union(*(track.keys() for _, track in ordered_tracks))
    candidate_keys.discard("frame_ids")

    for key in sorted(candidate_keys):
        pieces = []
        can_merge_key = True
        for frame_id in merged["frame_ids"]:
            track, local_idx = frame_map[int(frame_id)]
            if key not in track:
                can_merge_key = False
                break
            arr = np.asarray(track[key])
            if arr.ndim == 0 or arr.shape[0] != len(track["frame_ids"]):
                can_merge_key = False
                break
            pieces.append(arr[local_idx])
        if can_merge_key:
            merged[key] = np.stack(pieces, axis=0)

    print(
        "[INFO] Merged WHAM tracks "
        f"{[str(key) for key, _ in ordered_tracks]} into {len(merged['frame_ids'])} unique frames."
    )
    return merged


def _prefer_world_coordinates(data: Any, use_world: bool) -> Any:
    if not use_world or not isinstance(data, dict):
        return data
    out = dict(data)
    if "pose_world" in out:
        out["pose"] = out["pose_world"]
    if "trans_world" in out:
        out["trans"] = out["trans_world"]
    return out


def _find_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_axis_angle(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
        flat = arr.reshape(-1, 3, 3)
        return Rotation.from_matrix(flat).as_rotvec().reshape(*arr.shape[:-2], 3)
    if arr.ndim >= 2 and arr.shape[-1] == 4:
        flat = arr.reshape(-1, 4)
        return Rotation.from_quat(flat).as_rotvec().reshape(*arr.shape[:-1], 3)
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Cannot interpret {name} as axis-angle/quaternion/rotation matrix; shape={arr.shape}")


def _extract_poses(data: dict[str, Any]) -> np.ndarray:
    poses = _find_key(data, POSE_KEYS)
    if poses is not None:
        arr = np.asarray(poses)
        if arr.ndim == 2 and arr.shape[1] >= 66:
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr.reshape(arr.shape[0], -1).astype(np.float32)
        if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
            aa = _as_axis_angle(arr, "poses")
            return aa.reshape(aa.shape[0], -1).astype(np.float32)

    body_pose = _find_key(data, BODY_POSE_KEYS)
    global_orient = _find_key(data, GLOBAL_ORIENT_KEYS)
    if body_pose is None or global_orient is None:
        raise KeyError(
            "Could not find full poses or body_pose + global_orient in WHAM output. "
            f"Tried pose keys={POSE_KEYS}, body keys={BODY_POSE_KEYS}, root keys={GLOBAL_ORIENT_KEYS}."
        )

    body_aa = _as_axis_angle(body_pose, "body_pose")
    root_aa = _as_axis_angle(global_orient, "global_orient")
    body_flat = body_aa.reshape(body_aa.shape[0], -1)
    root_flat = root_aa.reshape(root_aa.shape[0], -1)
    return np.concatenate([root_flat, body_flat], axis=1).astype(np.float32)


def _extract_trans(data: dict[str, Any], n_frames: int) -> np.ndarray:
    trans = _find_key(data, TRANS_KEYS)
    if trans is None:
        raise KeyError(f"Could not find translation key. Tried {TRANS_KEYS}.")
    arr = np.asarray(trans, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected translation shape [T, 3], got {arr.shape}")
    if arr.shape[0] != n_frames:
        raise ValueError(f"Translation frames {arr.shape[0]} != pose frames {n_frames}")
    return arr


def _extract_betas(data: dict[str, Any]) -> np.ndarray:
    betas = _find_key(data, BETAS_KEYS)
    if betas is None:
        return np.zeros(10, dtype=np.float32)
    arr = np.asarray(betas, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr[:10].astype(np.float32)


def _extract_fps(data: dict[str, Any], fallback: float | None, force_fps: bool = False) -> float:
    if force_fps:
        if fallback is None:
            raise ValueError("--force-fps requires --fps.")
        return float(fallback)

    fps = _find_key(data, FPS_KEYS)
    if fps is None:
        if fallback is None:
            raise KeyError(f"Could not find fps key. Tried {FPS_KEYS}; pass --fps explicitly.")
        return float(fallback)
    return float(np.asarray(fps).reshape(-1)[0])


def _yup_to_zup(poses: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WHAM Y-up world frame to AMASS Z-up frame via Rx(+90 deg).

    WHAM's pose_world / trans_world keep SMPL's canonical Y-up convention
    (gravity along -Y), whereas AMASS / MuJoCo expect Z-up (gravity along -Z).
    Without this rotation the retargeted humanoid lies down.
    """
    R_align = Rotation.from_euler("x", 90.0, degrees=True)
    R_align_mat = R_align.as_matrix().astype(np.float32)

    root_aa = poses[:, :3].astype(np.float32)
    new_root_R = R_align * Rotation.from_rotvec(root_aa)
    new_root_aa = new_root_R.as_rotvec().astype(np.float32)

    new_poses = poses.copy()
    new_poses[:, :3] = new_root_aa
    new_trans = (trans.astype(np.float32) @ R_align_mat.T).astype(np.float32)
    return new_poses, new_trans


def convert(
    input_path: Path,
    output_path: Path,
    fps: float | None,
    gender: str,
    track_id: str | None,
    merge_tracks: bool,
    use_world: bool,
    align_up_axis: bool = True,
    force_fps: bool = False,
) -> None:
    raw = _unwrap_singleton(_load_any(input_path))
    if merge_tracks:
        if track_id is not None:
            raise ValueError("--merge-tracks and --track-id are mutually exclusive.")
        raw = _merge_tracks(raw)
    else:
        raw = _select_track(raw, track_id)
    raw = _prefer_world_coordinates(raw, use_world)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected WHAM output to be dict-like, got {type(raw).__name__}")

    poses = _extract_poses(raw)
    trans = _extract_trans(raw, poses.shape[0])
    betas = _extract_betas(raw)
    out_fps = _extract_fps(raw, fps, force_fps=force_fps)

    if poses.shape[1] < 66:
        raise ValueError(f"MuscleMimic expects at least 66 pose dims, got {poses.shape[1]}")

    if align_up_axis and use_world:
        poses, trans = _yup_to_zup(poses, trans)

    n_frames = poses.shape[0]
    root_orient = poses[:, :3]
    pose_body = poses[:, 3:66]
    left_hand_pose = poses[:, 66:111] if poses.shape[1] >= 111 else np.zeros((n_frames, 45), dtype=np.float32)
    right_hand_pose = poses[:, 111:156] if poses.shape[1] >= 156 else np.zeros((n_frames, 45), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        poses=poses,
        root_orient=root_orient.astype(np.float32),
        pose_body=pose_body.astype(np.float32),
        left_hand_pose=left_hand_pose.astype(np.float32),
        right_hand_pose=right_hand_pose.astype(np.float32),
        trans=trans,
        betas=betas,
        gender=np.asarray(gender),
        mocap_framerate=np.asarray(out_fps, dtype=np.float32),
        mocap_frame_rate=np.asarray(out_fps, dtype=np.float32),
    )
    print(f"[OK] {input_path} -> {output_path}")
    print(f"     frames={poses.shape[0]} pose_dim={poses.shape[1]} fps={out_fps} gender={gender}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="WHAM .npz/.npy/.pkl output")
    parser.add_argument("--output", required=True, type=Path, help="AMASS-style output .npz")
    parser.add_argument("--fps", type=float, default=None, help="Fallback FPS if WHAM output has no fps key")
    parser.add_argument("--force-fps", action="store_true", help="Use --fps even if the WHAM output has an fps field")
    parser.add_argument("--gender", default="neutral", help="AMASS gender field")
    parser.add_argument("--track-id", default=None, help="WHAM track id to convert; defaults to the longest track")
    parser.add_argument("--merge-tracks", action="store_true", help="Merge WHAM tracks by frame_ids before conversion")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local camera pose/trans instead of pose_world/trans_world when available",
    )
    parser.add_argument(
        "--no-up-align",
        action="store_true",
        help="Skip the WHAM Y-up -> AMASS Z-up Rx(+90deg) alignment (default: align).",
    )
    args = parser.parse_args()

    convert(
        args.input,
        args.output,
        args.fps,
        args.gender,
        args.track_id,
        args.merge_tracks,
        use_world=not args.local,
        align_up_axis=not args.no_up_align,
        force_fps=args.force_fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
