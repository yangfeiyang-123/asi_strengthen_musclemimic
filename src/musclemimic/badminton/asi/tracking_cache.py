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
    if bundle.provenance is not None:
        payload["reference_motion_path"] = np.asarray(bundle.provenance["motion_path"])
        payload["reference_motion_uid"] = np.asarray(
            int(bundle.provenance["motion_uid"]), dtype=np.int64
        )
    if bundle.body_keypoints is not None:
        payload["body_keypoints"] = bundle.body_keypoints.astype(np.float32)
        payload["body_keypoint_labels"] = np.asarray(bundle.body_keypoint_labels)
    if bundle.body_laplacian is not None:
        payload["body_laplacian"] = bundle.body_laplacian.astype(np.float32)
    if bundle.events is not None:
        phases = bundle.events.phase_arrays()
        payload.update(
            {
                "phase_global": phases.phase_global,
                "phase_id": phases.phase_id,
                "phase_local": phases.phase_local,
                "time_to_impact_s": phases.time_to_impact_s,
                "time_from_impact_s": phases.time_from_impact_s,
                "impact_flag": phases.impact_flag,
                "event_names": np.asarray(sorted(bundle.events.events)),
                "event_frames": np.asarray(
                    [bundle.events.events[name].frame for name in sorted(bundle.events.events)],
                    dtype=np.int32,
                ),
                "event_times_s": np.asarray(
                    [bundle.events.events[name].time_s for name in sorted(bundle.events.events)],
                    dtype=np.float32,
                ),
                "event_confidence": np.asarray(
                    [bundle.events.events[name].confidence for name in sorted(bundle.events.events)],
                    dtype=np.float32,
                ),
                "event_sources": np.asarray(
                    [bundle.events.events[name].source for name in sorted(bundle.events.events)]
                ),
            }
        )
    if bundle.racket is not None:
        kinematic_confidence = float((bundle.provenance or {}).get("kinematic_confidence", 1.0))
        impact_confidence = float((bundle.provenance or {}).get("impact_confidence", 1.0))
        reference_confidence = np.minimum(
            bundle.racket.confidence,
            min(kinematic_confidence, impact_confidence),
        ).astype(np.float32)
        payload.update(
            {
                "racket_position_world": bundle.racket.position_world,
                "racket_quaternion_world": bundle.racket.quaternion_world,
                "racket_linear_velocity_world": bundle.racket.linear_velocity_world,
                "racket_angular_velocity_world": bundle.racket.angular_velocity_world,
                "stringbed_normal_world": bundle.racket.stringbed_normal_world,
                "stringbed_center_world": bundle.racket.stringbed_center_world,
                "racket_reference_confidence": bundle.racket.confidence,
                "reference_confidence": reference_confidence,
                "racket_reference_source": np.asarray(bundle.racket.source),
                "racket_quaternion_convention": np.asarray(bundle.racket.quaternion_convention),
            }
        )
    if bundle.content_fingerprint is not None:
        payload["reference_bundle_content_fingerprint"] = np.asarray(bundle.content_fingerprint)

    np.savez(cache_npz, **payload)

    report = {
        "status": "ready",
        "retarget_method": "reference_bundle_direct",
        "source_manifest": str(bundle.manifest_path),
        "cache_npz": str(cache_npz),
        "num_frames": len(bundle.poses),
        "fps": float(bundle.fps),
        "control_dt": float(control_dt),
        "effective_ref_stride": float(manager.effective_ref_stride),
        "coordinate_system": bundle.coordinate_system,
        "quality_tier": str(bundle.quality.get("quality_tier", "")),
        "body_keypoints_available": bundle.body_keypoints is not None,
        "event_reference_available": bundle.events is not None,
        "racket_reference_available": bundle.racket is not None,
        "racket_reference_source": None if bundle.racket is None else bundle.racket.source,
        "reference_bundle_content_fingerprint": bundle.content_fingerprint,
        "reference_motion_path": None
        if bundle.provenance is None
        else str(bundle.provenance["motion_path"]),
        "reference_motion_uid": None
        if bundle.provenance is None
        else int(bundle.provenance["motion_uid"]),
        "provenance": bundle.provenance,
        "qpos_ref_available": False,
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return TrackingReferenceCacheResult(cache_npz=cache_npz, report_json=report_json, report=report)
