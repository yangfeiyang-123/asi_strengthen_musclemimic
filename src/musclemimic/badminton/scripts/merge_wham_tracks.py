#!/usr/bin/env python3
"""Merge a multi-track WHAM output pkl into a single track keyed by frame_id.

WHAM occasionally splits one subject into multiple tracks across a clip.
This script stitches the per-frame fields (pose / trans / pose_world /
trans_world / betas / verts / frame_ids) into one track ordered by frame_id.
At a seam frame that exists in more than one track, the entry from the track
with the longer remaining suffix wins (i.e., the track that "continues"
through the seam), which gives the smoothest concat.

Usage:
    python merge_wham_tracks.py --input wham_output.pkl \\
                                --output wham_output_merged.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


PER_FRAME_KEYS = (
    "pose",
    "trans",
    "pose_world",
    "trans_world",
    "betas",
    "verts",
    "frame_ids",
)


def _load(path: Path) -> Any:
    if path.suffix in {".pkl", ".pickle", ".pth"}:
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def _track_frame_set(track: dict[str, Any]) -> set[int]:
    return {int(f) for f in np.asarray(track["frame_ids"])}


def _pick_owner_per_frame(tracks: dict[Any, dict[str, Any]]) -> dict[int, Any]:
    """For each frame, choose the track that has the longest continuous suffix
    starting at that frame — the one that "continues through" the seam."""
    track_frames = {tid: sorted(_track_frame_set(t)) for tid, t in tracks.items()}
    owner: dict[int, Any] = {}
    for tid, frames in track_frames.items():
        for i, fid in enumerate(frames):
            suffix_len = len(frames) - i
            existing = owner.get(fid)
            if existing is None:
                owner[fid] = (tid, suffix_len)
            else:
                if suffix_len > existing[1]:
                    owner[fid] = (tid, suffix_len)
    return {fid: tid_len[0] for fid, tid_len in owner.items()}


def _align_world_frames(
    data: dict[Any, dict[str, Any]],
    owner: dict[int, Any],
    sorted_frames: list[int],
    per_track_index: dict[Any, dict[int, int]],
) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
    """For each track t, compute (R_align, t_align) so that t's world frame
    matches the FIRST track that owns at least one earlier frame.

    Mapping applied to track t's world-frame data:
        trans_world_new  = R_align @ trans_world_old + t_align
        root_orient_new  = R_align @ root_orient_old   (composed in rotmat)
    Pose body joints (relative to root) are unaffected.
    """
    transforms: dict[Any, tuple[np.ndarray, np.ndarray]] = {
        tid: (np.eye(3), np.zeros(3)) for tid in data
    }

    seen: list[Any] = []
    for fid in sorted_frames:
        tid = owner[fid]
        if tid in seen:
            continue

        if not seen:
            seen.append(tid)
            continue

        ref_tid = seen[0]
        # Find a frame that exists in BOTH tracks (true overlap), else use the seam frame.
        ref_frames = set(per_track_index[ref_tid])
        cur_frames = set(per_track_index[tid])
        overlap = sorted(ref_frames & cur_frames)
        anchor_frame = overlap[0] if overlap else fid

        if anchor_frame not in per_track_index[tid] or anchor_frame not in per_track_index[ref_tid]:
            # Fallback: use the previous frame from ref track and the first frame of cur track.
            anchor_frame = fid

        ref_track = data[ref_tid]
        cur_track = data[tid]
        ref_idx = per_track_index[ref_tid].get(anchor_frame)
        cur_idx = per_track_index[tid].get(anchor_frame)

        if ref_idx is None:
            # use last frame of ref track
            ref_idx = len(per_track_index[ref_tid]) - 1
        if cur_idx is None:
            cur_idx = 0

        ref_pose = np.asarray(ref_track["pose_world"])[ref_idx, :3]
        cur_pose = np.asarray(cur_track["pose_world"])[cur_idx, :3]
        ref_trans = np.asarray(ref_track["trans_world"])[ref_idx]
        cur_trans = np.asarray(cur_track["trans_world"])[cur_idx]

        R_ref = Rotation.from_rotvec(ref_pose).as_matrix()
        R_cur = Rotation.from_rotvec(cur_pose).as_matrix()
        # R_align @ R_cur = R_ref  =>  R_align = R_ref @ R_cur.T
        R_align = R_ref @ R_cur.T
        t_align = ref_trans - R_align @ cur_trans

        transforms[tid] = (R_align, t_align)
        seen.append(tid)
        print(f"[INFO] Aligning track {tid!r} to track {ref_tid!r} at frame {anchor_frame}: "
              f"|t_align|={np.linalg.norm(t_align):.4f}m, "
              f"angle={np.degrees(np.linalg.norm(Rotation.from_matrix(R_align).as_rotvec())):.2f}deg")

    return transforms


def _apply_world_transform(track: dict[str, Any], R: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    """Apply a constant SE(3) (R, t) to a track's world-frame fields."""
    out = dict(track)
    if "trans_world" in out:
        tw = np.asarray(out["trans_world"], dtype=np.float64)
        out["trans_world"] = (tw @ R.T + t).astype(np.asarray(track["trans_world"]).dtype)
    if "pose_world" in out:
        pw = np.asarray(out["pose_world"]).copy()
        root_aa = pw[:, :3].astype(np.float64)
        R_root = Rotation.from_rotvec(root_aa).as_matrix()
        R_new = np.einsum("ij,njk->nik", R, R_root)
        new_aa = Rotation.from_matrix(R_new).as_rotvec()
        pw[:, :3] = new_aa.astype(pw.dtype)
        out["pose_world"] = pw
    return out


def merge_tracks(data: dict[Any, dict[str, Any]], align_world: bool = True) -> dict[str, Any]:
    if not data:
        raise ValueError("Empty WHAM data — nothing to merge.")
    if len(data) == 1:
        only = next(iter(data.values()))
        print("[INFO] Only one track present; copying through unchanged.")
        return dict(only)

    owner = _pick_owner_per_frame(data)
    sorted_frames = sorted(owner)
    print(f"[INFO] Merging tracks {list(data.keys())} -> single track with "
          f"{len(sorted_frames)} frames (range {sorted_frames[0]}..{sorted_frames[-1]}).")

    per_track_index = {tid: {int(f): i for i, f in enumerate(np.asarray(t["frame_ids"]))}
                       for tid, t in data.items()}

    if align_world:
        transforms = _align_world_frames(data, owner, sorted_frames, per_track_index)
        data = {tid: _apply_world_transform(t, *transforms[tid]) for tid, t in data.items()}

    candidate_keys = set().union(*(t.keys() for t in data.values()))
    merged: dict[str, Any] = {}

    seam_counts = defaultdict(int)
    for fid in sorted_frames:
        seam_counts[owner[fid]] += 1
    print(f"[INFO] Frames per owning track: {dict(seam_counts)}")

    for key in sorted(candidate_keys):
        if key == "frame_ids":
            merged["frame_ids"] = np.asarray(sorted_frames, dtype=np.int32)
            continue

        sample = next(iter(data.values())).get(key)
        if sample is None:
            continue
        sample_arr = np.asarray(sample)
        if sample_arr.ndim == 0:
            merged[key] = sample_arr
            continue

        is_per_frame = key in PER_FRAME_KEYS or (
            sample_arr.ndim >= 1
            and sample_arr.shape[0] == len(np.asarray(next(iter(data.values()))["frame_ids"]))
        )

        if not is_per_frame:
            merged[key] = sample_arr
            continue

        ok = True
        pieces = []
        for fid in sorted_frames:
            tid = owner[fid]
            track = data[tid]
            if key not in track:
                ok = False
                break
            arr = np.asarray(track[key])
            local_idx = per_track_index[tid][fid]
            if arr.ndim == 0 or arr.shape[0] <= local_idx:
                ok = False
                break
            pieces.append(arr[local_idx])
        if ok:
            merged[key] = np.stack(pieces, axis=0)

    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Multi-track WHAM pkl")
    ap.add_argument("--output", required=True, type=Path, help="Output single-track pkl")
    ap.add_argument("--track-key", default=0, help="Key under which to store the merged track (default: 0)")
    ap.add_argument(
        "--no-world-align",
        action="store_true",
        help="Skip per-track world-frame SE(3) alignment at the seam (default: align).",
    )
    args = ap.parse_args()

    raw = _load(args.input)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict-of-tracks, got {type(raw).__name__}")

    merged = merge_tracks(raw, align_world=not args.no_world_align)

    try:
        track_key: Any = int(args.track_key)
    except (TypeError, ValueError):
        track_key = args.track_key
    out_obj = {track_key: merged}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    fid = merged["frame_ids"]
    print(f"[OK] {args.input} -> {args.output}")
    print(f"     1 track, frames={len(fid)} (ids {int(fid.min())}..{int(fid.max())})")
    print(f"     fields: {sorted(k for k in merged if k != 'frame_ids')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
