from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.data.event_schema import EventTimeline, load_event_timeline
from musclemimic.badminton.data.racket_reference import RacketReference, load_racket_reference
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.motion_identity import (
    normalize_relative_motion_path,
    stable_motion_uid,
)

LEGACY_BUNDLE_VERSION = "contact_reference_bundle_v1"
EVENT_BUNDLE_VERSIONS = {"contact_reference_bundle_v2", "event_reference_bundle_v2"}
SUPPORTED_BUNDLE_VERSIONS = {LEGACY_BUNDLE_VERSION, *EVENT_BUNDLE_VERSIONS}
REQUIRED_V2_PROVENANCE_FIELDS = {
    "subject_id",
    "session_id",
    "trial_id",
    "motion_path",
    "motion_uid",
    "source_video_id",
    "source_kind",
    "retarget_pipeline_version",
    "cache_kind",
    "quality_tier",
    "manual_review_status",
    "legacy_fallback",
    "kinematic_confidence",
    "racket_confidence",
    "impact_confidence",
    "impact_position_uncertainty_m",
    "impact_timing_uncertainty_s",
}


@dataclass(frozen=True)
class ReferenceBundle:
    poses: np.ndarray
    root_orient: np.ndarray
    pose_body: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    fps: float
    frame_ids: np.ndarray
    contact_confidence: np.ndarray
    stance_mask: np.ndarray
    foot_points: np.ndarray
    foot_labels: list[str]
    body_graph: dict[str, Any]
    body_keypoints: np.ndarray | None
    body_laplacian: np.ndarray | None
    body_keypoint_labels: list[str]
    quality: dict[str, Any]
    coordinate_system: str
    manifest_path: Path
    manifest: dict[str, Any]
    events: EventTimeline | None = None
    racket: RacketReference | None = None
    provenance: dict[str, Any] | None = None
    content_fingerprint: str | None = None


def load_reference_bundle(
    manifest_path: str | Path,
    *,
    allow_low_quality: bool = False,
) -> ReferenceBundle:
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    base = path.parent

    version = str(manifest.get("version", ""))
    if version not in SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError(f"Unsupported reference bundle version: {version}")
    coordinate_system = str(manifest.get("coordinate_system", ""))
    if coordinate_system != "amass_zup":
        raise ValueError("ASI-PPO expects amass_zup reference bundles. Re-export from Optimized-WHAM.")
    if manifest.get("unit") != "meter":
        raise ValueError("ASI-PPO expects reference bundle unit == meter.")

    is_event_bundle = version in EVENT_BUNDLE_VERSIONS
    provenance = dict(manifest.get("provenance") or {})
    content_fingerprint = None
    if is_event_bundle:
        _validate_v2_provenance(provenance)
        content_fingerprint = validate_reference_bundle_fingerprint(path, manifest=manifest)

    motion = np.load(base / manifest["motion_npz"], allow_pickle=True)
    contact = np.load(base / manifest["contact_npz"], allow_pickle=True)
    body_graph = _read_json(base / manifest["body_graph_json"])
    quality_path = base / manifest.get("quality_report_json", "quality_report.json")
    quality = _read_json(quality_path) if quality_path.exists() else dict(manifest.get("quality", {}))
    if not quality:
        quality = dict(manifest.get("quality", {}))

    if not allow_low_quality and not bool(quality.get("usable_for_training", False)):
        raise ValueError(f"Reference bundle is not marked usable for training: {path}")

    poses = np.asarray(motion["poses"], dtype=np.float32)
    trans = np.asarray(motion["trans"], dtype=np.float32)
    frame_ids = np.asarray(motion.get("frame_ids", np.arange(len(poses))), dtype=np.int32)
    contact_confidence = np.asarray(contact["contact_confidence"], dtype=np.float32)
    stance_mask = np.asarray(contact["stance_mask"], dtype=np.bool_)
    foot_points = np.asarray(contact["foot_points"], dtype=np.float32)
    contact_frame_ids = np.asarray(contact.get("frame_ids", np.arange(len(contact_confidence))), dtype=np.int32)
    body_keypoints = _optional_body_array(motion, "body_keypoints", len(poses))
    body_laplacian = _optional_body_array(motion, "body_laplacian", len(poses))
    body_keypoint_labels = _optional_labels(motion, "body_keypoint_labels")

    _validate_frame_count(
        poses, trans, frame_ids, contact_confidence, stance_mask, foot_points, contact_frame_ids, manifest
    )
    _validate_npz_coordinate_system(motion, "motion")
    _validate_npz_coordinate_system(contact, "contact")
    _validate_motion_fps(motion, float(manifest["fps"]))
    if body_keypoints is not None:
        _validate_body_coordinate_system(motion)
        if body_laplacian is not None and body_laplacian.shape != body_keypoints.shape:
            raise ValueError("body_laplacian must match body_keypoints shape")
        if body_keypoint_labels and len(body_keypoint_labels) != body_keypoints.shape[1]:
            raise ValueError("body_keypoint_labels length must match body_keypoints width")

    events = None
    racket = None
    if is_event_bundle:
        event_key = "event_json" if "event_json" in manifest else "events_json"
        if event_key not in manifest or "racket_npz" not in manifest:
            raise ValueError("event reference bundle v2 requires event_json and racket_npz")
        events = load_event_timeline(
            base / manifest[event_key],
            num_frames=len(poses),
            fps=float(manifest["fps"]),
        )
        racket = load_racket_reference(
            base / manifest["racket_npz"],
            num_frames=len(poses),
            fps=float(manifest["fps"]),
            coordinate_system=coordinate_system,
        )
        if not np.isclose(
            float(provenance["impact_confidence"]),
            float(events.impact.confidence),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("provenance impact_confidence does not match event annotation")
        if not np.isclose(
            float(provenance["racket_confidence"]),
            float(np.mean(racket.confidence)),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("provenance racket_confidence does not match mean racket confidence")

    return ReferenceBundle(
        poses=poses,
        root_orient=poses[:, :3],
        pose_body=poses[:, 3:],
        trans=trans,
        betas=np.asarray(motion["betas"], dtype=np.float32),
        fps=float(manifest["fps"]),
        frame_ids=frame_ids,
        contact_confidence=contact_confidence,
        stance_mask=stance_mask,
        foot_points=foot_points,
        foot_labels=[str(label) for label in np.asarray(contact["foot_labels"]).tolist()],
        body_graph=body_graph,
        body_keypoints=body_keypoints,
        body_laplacian=body_laplacian,
        body_keypoint_labels=body_keypoint_labels,
        quality=quality,
        coordinate_system=coordinate_system,
        manifest_path=path,
        manifest=manifest,
        events=events,
        racket=racket,
        provenance=provenance or None,
        content_fingerprint=content_fingerprint,
    )


def reference_bundle_fingerprint(
    manifest_path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Return a content hash over bundle identity, provenance, and referenced files."""

    path = Path(manifest_path).resolve()
    payload = dict(manifest or _read_json(path))
    base = path.parent
    roles = (
        "motion_npz",
        "contact_npz",
        "body_graph_json",
        "quality_report_json",
        "event_json",
        "events_json",
        "racket_npz",
    )
    inventory: list[dict[str, Any]] = []
    for role in roles:
        relative = payload.get(role)
        if relative is None:
            continue
        file_path = (base / str(relative)).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"reference bundle file for {role} does not exist: {file_path}")
        inventory.append(
            {
                "role": role,
                "path": str(relative),
                "sha256": _file_sha256(file_path),
                "num_bytes": int(file_path.stat().st_size),
            }
        )
    identity = {
        key: payload.get(key) for key in ("version", "sequence", "num_frames", "fps", "coordinate_system", "unit")
    }
    fingerprint_payload = {
        "schema_version": "reference_bundle_content_fingerprint_v1",
        "identity": identity,
        "provenance": payload.get("provenance"),
        "files": inventory,
    }
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reference_bundle_fingerprint(
    manifest_path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    path = Path(manifest_path).resolve()
    payload = dict(manifest or _read_json(path))
    supplied = payload.get("content_fingerprint")
    if isinstance(supplied, Mapping):
        if supplied.get("algorithm", "sha256") != "sha256":
            raise ValueError("reference bundle content fingerprint algorithm must be sha256")
        supplied = supplied.get("sha256", supplied.get("value"))
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("event reference bundle v2 requires a sha256 content_fingerprint")
    actual = reference_bundle_fingerprint(path, manifest=payload)
    if supplied != actual:
        raise ValueError("reference bundle content_fingerprint mismatch")
    return actual


def _validate_v2_provenance(provenance: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_V2_PROVENANCE_FIELDS - set(provenance))
    if missing:
        raise ValueError(f"event reference bundle v2 provenance is missing {missing}")
    for key in (
        "subject_id",
        "session_id",
        "trial_id",
        "motion_path",
        "source_video_id",
        "source_kind",
        "retarget_pipeline_version",
        "cache_kind",
        "quality_tier",
        "manual_review_status",
    ):
        if not str(provenance[key]).strip():
            raise ValueError(f"reference provenance {key} must be non-empty")
    motion_path = normalize_relative_motion_path(str(provenance["motion_path"]))
    if str(provenance["motion_path"]) != motion_path:
        raise ValueError("reference provenance motion_path must already be normalized")
    motion_uid = provenance["motion_uid"]
    if isinstance(motion_uid, bool) or not isinstance(motion_uid, int | np.integer):
        raise ValueError("reference provenance motion_uid must be an integer")
    expected_uid = stable_motion_uid(motion_path)
    if int(motion_uid) != expected_uid:
        raise ValueError("reference provenance motion_uid does not match stable motion_path identity")
    if not isinstance(provenance["legacy_fallback"], bool):
        raise ValueError("reference provenance legacy_fallback must be boolean")
    for key in ("kinematic_confidence", "racket_confidence", "impact_confidence"):
        value = float(provenance[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"reference provenance {key} must be in [0,1]")
    for key in ("impact_position_uncertainty_m", "impact_timing_uncertainty_s"):
        value = float(provenance[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"reference provenance {key} must be finite and non-negative")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frame_count(
    poses: np.ndarray,
    trans: np.ndarray,
    frame_ids: np.ndarray,
    contact_confidence: np.ndarray,
    stance_mask: np.ndarray,
    foot_points: np.ndarray,
    contact_frame_ids: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    expected = int(manifest["num_frames"])
    lengths = {
        "poses": len(poses),
        "trans": len(trans),
        "frame_ids": len(frame_ids),
        "contact_confidence": len(contact_confidence),
        "stance_mask": len(stance_mask),
        "foot_points": len(foot_points),
        "contact_frame_ids": len(contact_frame_ids),
    }
    bad = {name: value for name, value in lengths.items() if value != expected}
    if bad:
        raise ValueError(f"Reference bundle frame count mismatch: expected {expected}, got {bad}")
    if not np.array_equal(frame_ids, contact_frame_ids):
        raise ValueError("Reference bundle frame count ids differ between motion and contact schedule.")


def _validate_npz_coordinate_system(npz: Any, label: str) -> None:
    if "coordinate_system" not in npz:
        raise ValueError(f"{label} npz missing coordinate_system")
    value = str(np.asarray(npz["coordinate_system"]).item())
    if value != "amass_zup":
        raise ValueError(f"{label} npz coordinate_system must be amass_zup, got {value}")


def _validate_body_coordinate_system(npz: Any) -> None:
    if "body_keypoints_coordinate_system" not in npz:
        return
    value = str(np.asarray(npz["body_keypoints_coordinate_system"]).item())
    if value != "amass_zup":
        raise ValueError(f"body_keypoints coordinate_system must be amass_zup, got {value}")


def _validate_motion_fps(npz: Any, expected_fps: float) -> None:
    values = []
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in npz:
            values.append((key, float(np.asarray(npz[key]).reshape(-1)[0])))
    for key, value in values:
        if not np.isclose(value, float(expected_fps), rtol=0.0, atol=1e-6):
            raise ValueError(f"motion fps mismatch: {key}={value:g}, manifest fps={expected_fps:g}")
    if len(values) == 2 and not np.isclose(values[0][1], values[1][1], rtol=0.0, atol=1e-6):
        raise ValueError(f"motion fps fields disagree: {values[0][1]:g} != {values[1][1]:g}")


def _optional_body_array(npz: Any, key: str, expected_frames: int) -> np.ndarray | None:
    if key not in npz:
        return None
    arr = np.asarray(npz[key], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [T, K, 3], got {arr.shape}")
    if arr.shape[0] != int(expected_frames):
        raise ValueError(f"{key} frame count mismatch: expected {expected_frames}, got {arr.shape[0]}")
    return arr


def _optional_labels(npz: Any, key: str) -> list[str]:
    if key not in npz:
        return []
    return [str(label) for label in np.asarray(npz[key]).tolist()]


def _read_json(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON contract must contain an object: {path}")
    return payload
