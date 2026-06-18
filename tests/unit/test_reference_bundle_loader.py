import json

import numpy as np
import pytest

from BadmintonMimic.data.reference_bundle import load_reference_bundle


def _write_bundle(root, *, coordinate_system="amass_zup", usable=True, contact_frames=3, body_keypoints=False):
    motion_payload = {
        "poses": np.zeros((3, 72), dtype=np.float32),
        "trans": np.zeros((3, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "mocap_framerate": np.asarray(60.0, dtype=np.float32),
        "mocap_frame_rate": np.asarray(60.0, dtype=np.float32),
        "frame_ids": np.arange(3, dtype=np.int32),
        "coordinate_system": np.asarray(coordinate_system),
    }
    if body_keypoints:
        body = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
        motion_payload.update(
            {
                "body_keypoints": body,
                "body_keypoint_labels": np.asarray(["pelvis", "left_toe"]),
                "body_laplacian": body * 0.5,
                "body_keypoints_coordinate_system": np.asarray(coordinate_system),
            }
        )
    np.savez(
        root / "motion.npz",
        **motion_payload,
    )
    np.savez(
        root / "contact_schedule.npz",
        contact_confidence=np.ones((contact_frames, 2), dtype=np.float32),
        stance_mask=np.ones((contact_frames, 2), dtype=np.bool_),
        foot_points=np.zeros((contact_frames, 2, 3), dtype=np.float32),
        foot_labels=np.asarray(["left_foot", "right_foot"]),
        frame_ids=np.arange(contact_frames, dtype=np.int32),
        coordinate_system=np.asarray(coordinate_system),
    )
    (root / "body_graph.json").write_text(json.dumps({"version": "body_graph_v1"}), encoding="utf-8")
    (root / "quality_report.json").write_text(
        json.dumps({"usable_for_training": usable, "quality_tier": "A"}),
        encoding="utf-8",
    )
    manifest = {
        "version": "contact_reference_bundle_v1",
        "sequence": "demo",
        "num_frames": 3,
        "fps": 60.0,
        "coordinate_system": coordinate_system,
        "unit": "meter",
        "motion_npz": "motion.npz",
        "contact_npz": "contact_schedule.npz",
        "body_graph_json": "body_graph.json",
        "quality_report_json": "quality_report.json",
        "quality": {"usable_for_training": usable, "quality_tier": "A"},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_reference_bundle_validates_and_returns_arrays(tmp_path):
    manifest_path = _write_bundle(tmp_path)

    bundle = load_reference_bundle(manifest_path)

    assert bundle.fps == 60.0
    assert bundle.coordinate_system == "amass_zup"
    assert bundle.poses.shape == (3, 72)
    assert bundle.contact_confidence.shape == (3, 2)
    assert bundle.foot_labels == ["left_foot", "right_foot"]


def test_load_reference_bundle_rejects_wrong_coordinate_system(tmp_path):
    manifest_path = _write_bundle(tmp_path, coordinate_system="wham_yup")

    with pytest.raises(ValueError, match="expects amass_zup"):
        load_reference_bundle(manifest_path)


def test_load_reference_bundle_rejects_frame_mismatch(tmp_path):
    manifest_path = _write_bundle(tmp_path, contact_frames=2)

    with pytest.raises(ValueError, match="frame count"):
        load_reference_bundle(manifest_path)


def test_load_reference_bundle_rejects_low_quality_by_default(tmp_path):
    manifest_path = _write_bundle(tmp_path, usable=False)

    with pytest.raises(ValueError, match="not marked usable"):
        load_reference_bundle(manifest_path)

    assert load_reference_bundle(manifest_path, allow_low_quality=True).quality["usable_for_training"] is False


def test_load_reference_bundle_returns_optional_body_graph_targets(tmp_path):
    manifest_path = _write_bundle(tmp_path, body_keypoints=True)

    bundle = load_reference_bundle(manifest_path)

    assert bundle.body_keypoints is not None
    assert bundle.body_laplacian is not None
    assert bundle.body_keypoint_labels == ["pelvis", "left_toe"]
    assert bundle.body_keypoints.shape == (3, 2, 3)


def test_load_reference_bundle_rejects_motion_fps_mismatch(tmp_path):
    manifest_path = _write_bundle(tmp_path)
    with np.load(tmp_path / "motion.npz", allow_pickle=True) as motion:
        payload = {key: motion[key] for key in motion.files}
    payload["mocap_framerate"] = np.asarray(30.0, dtype=np.float32)
    payload["mocap_frame_rate"] = np.asarray(30.0, dtype=np.float32)
    np.savez(tmp_path / "motion.npz", **payload)

    with pytest.raises(ValueError, match="fps"):
        load_reference_bundle(manifest_path)
