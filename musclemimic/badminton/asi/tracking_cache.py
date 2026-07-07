from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.asi.reference_phase import ReferencePhaseManager
from musclemimic.badminton.data.reference_bundle import load_reference_bundle


@dataclass(frozen=True)
class TrackingReferenceCacheResult:
    cache_npz: Path
    report_json: Path
    report: dict[str, Any]


def build_tracking_reference_cache(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    control_dt: float,
    allow_low_quality: bool = False,
) -> TrackingReferenceCacheResult:
    bundle = load_reference_bundle(manifest_path, allow_low_quality=allow_low_quality)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manager = ReferencePhaseManager(
        num_frames=len(bundle.poses),
        reference_fps=bundle.fps,
        control_dt=float(control_dt),
    )
    cache_npz = out_path / "tracking_reference_cache.npz"
    report_json = out_path / "retarget_report.json"

    payload: dict[str, Any] = {
        "poses_ref": bundle.poses.astype(np.float32),
        "root_orient_ref": bundle.root_orient.astype(np.float32),
        "pose_body_ref": bundle.pose_body.astype(np.float32),
        "trans_ref": bundle.trans.astype(np.float32),
        "contact_confidence": bundle.contact_confidence.astype(np.float32),
        "stance_mask": bundle.stance_mask.astype(np.bool_),
        "foot_points": bundle.foot_points.astype(np.float32),
        "foot_labels": np.asarray(bundle.foot_labels),
        "frame_ids": bundle.frame_ids.astype(np.int32),
        "reference_fps": np.asarray(bundle.fps, dtype=np.float32),
        "control_dt": np.asarray(float(control_dt), dtype=np.float32),
        "effective_ref_stride": np.asarray(manager.effective_ref_stride, dtype=np.float32),
        "coordinate_system": np.asarray(bundle.coordinate_system),
        "source_manifest": np.asarray(str(bundle.manifest_path)),
    }
    if bundle.body_keypoints is not None:
        payload["body_keypoints"] = bundle.body_keypoints.astype(np.float32)
        payload["body_keypoint_labels"] = np.asarray(bundle.body_keypoint_labels)
    if bundle.body_laplacian is not None:
        payload["body_laplacian"] = bundle.body_laplacian.astype(np.float32)

    np.savez(cache_npz, **payload)

    report = {
        "status": "ready",
        "retarget_method": "reference_bundle_direct",
        "source_manifest": str(bundle.manifest_path),
        "cache_npz": str(cache_npz),
        "num_frames": int(len(bundle.poses)),
        "fps": float(bundle.fps),
        "control_dt": float(control_dt),
        "effective_ref_stride": float(manager.effective_ref_stride),
        "coordinate_system": bundle.coordinate_system,
        "quality_tier": str(bundle.quality.get("quality_tier", "")),
        "body_keypoints_available": bundle.body_keypoints is not None,
        "qpos_ref_available": False,
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return TrackingReferenceCacheResult(cache_npz=cache_npz, report_json=report_json, report=report)
