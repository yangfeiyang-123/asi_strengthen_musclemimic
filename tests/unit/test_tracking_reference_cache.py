import json

import numpy as np

from BadmintonMimic.asi.tracking_cache import build_tracking_reference_cache


def _write_bundle(root):
    body = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    np.savez(
        root / "motion.npz",
        poses=np.zeros((4, 72), dtype=np.float32),
        trans=np.zeros((4, 3), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        mocap_framerate=np.asarray(60.0, dtype=np.float32),
        mocap_frame_rate=np.asarray(60.0, dtype=np.float32),
        frame_ids=np.arange(4, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
        body_keypoints=body,
        body_keypoint_labels=np.asarray(["pelvis", "left_toe"]),
        body_laplacian=body * 0.1,
        body_keypoints_coordinate_system=np.asarray("amass_zup"),
    )
    np.savez(
        root / "contact_schedule.npz",
        contact_confidence=np.ones((4, 2), dtype=np.float32),
        stance_mask=np.ones((4, 2), dtype=np.bool_),
        foot_points=np.zeros((4, 2, 3), dtype=np.float32),
        foot_labels=np.asarray(["left_foot", "right_foot"]),
        frame_ids=np.arange(4, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
    )
    (root / "body_graph.json").write_text(json.dumps({"version": "body_graph_v1"}), encoding="utf-8")
    (root / "quality_report.json").write_text(
        json.dumps({"usable_for_training": True, "quality_tier": "A"}),
        encoding="utf-8",
    )
    manifest = {
        "version": "contact_reference_bundle_v1",
        "sequence": "demo",
        "num_frames": 4,
        "fps": 60.0,
        "coordinate_system": "amass_zup",
        "unit": "meter",
        "motion_npz": "motion.npz",
        "contact_npz": "contact_schedule.npz",
        "body_graph_json": "body_graph.json",
        "quality_report_json": "quality_report.json",
        "quality": {"usable_for_training": True, "quality_tier": "A"},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_build_tracking_reference_cache_writes_npz_and_report(tmp_path):
    manifest_path = _write_bundle(tmp_path)

    result = build_tracking_reference_cache(manifest_path, tmp_path / "cache", control_dt=1.0 / 30.0)

    cache = np.load(result.cache_npz, allow_pickle=True)
    report = json.loads(result.report_json.read_text(encoding="utf-8"))

    assert cache["trans_ref"].shape == (4, 3)
    assert cache["body_keypoints"].shape == (4, 2, 3)
    assert float(cache["effective_ref_stride"]) == 2.0
    assert report["status"] == "ready"
    assert report["body_keypoints_available"] is True
    assert report["effective_ref_stride"] == 2.0
