from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_BUNDLE_VERSIONS = {"contact_reference_bundle_v1"}


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

    _validate_frame_count(poses, trans, frame_ids, contact_confidence, stance_mask, foot_points, contact_frame_ids, manifest)
    _validate_npz_coordinate_system(motion, "motion")
    _validate_npz_coordinate_system(contact, "contact")
    _validate_motion_fps(motion, float(manifest["fps"]))
    if body_keypoints is not None:
        _validate_body_coordinate_system(motion)
        if body_laplacian is not None and body_laplacian.shape != body_keypoints.shape:
            raise ValueError("body_laplacian must match body_keypoints shape")
        if body_keypoint_labels and len(body_keypoint_labels) != body_keypoints.shape[1]:
            raise ValueError("body_keypoint_labels length must match body_keypoints width")

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
    )


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
    return json.loads(path.read_text(encoding="utf-8"))
