import json

from BadmintonMimic.scripts.build_contact_tracking_manifest import build_contact_tracking_manifest


def _write_manifest(root, name, *, tier="A", frames=100, fps=60.0, usable=True, penetration=1.0, sliding=2.0):
    bundle_dir = root / name
    bundle_dir.mkdir()
    manifest = {
        "version": "contact_reference_bundle_v1",
        "sequence": name,
        "num_frames": frames,
        "fps": fps,
        "coordinate_system": "amass_zup",
        "unit": "meter",
        "motion_npz": "motion.npz",
        "contact_npz": "contact_schedule.npz",
        "body_graph_json": "body_graph.json",
        "quality_report_json": "quality_report.json",
        "quality": {
            "usable_for_training": usable,
            "quality_tier": tier,
            "foot_penetration_max_cm_after": penetration,
            "stance_sliding_max_cm_after": sliding,
        },
    }
    path = bundle_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_contact_tracking_manifest_filters_by_quality_and_thresholds(tmp_path):
    _write_manifest(tmp_path, "keep_a", tier="A")
    _write_manifest(tmp_path, "drop_tier", tier="C")
    _write_manifest(tmp_path, "drop_short", tier="A", frames=20)
    _write_manifest(tmp_path, "drop_sliding", tier="B", sliding=8.0)
    out = tmp_path / "train.jsonl"

    entries = build_contact_tracking_manifest(
        reference_root=tmp_path,
        out=out,
        include_tiers={"A", "B"},
        min_frames=60,
        max_stance_sliding_cm=5.0,
    )

    assert [entry["sequence"] for entry in entries] == ["keep_a"]
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == entries
    assert lines[0]["quality_tier"] == "A"


def test_build_contact_tracking_manifest_includes_cache_metadata_when_present(tmp_path):
    manifest_path = _write_manifest(tmp_path, "keep_a", tier="A")
    cache_path = manifest_path.parent / "tracking_reference_cache.npz"
    cache_path.write_bytes(b"cache")
    (manifest_path.parent / "retarget_report.json").write_text(
        json.dumps({"effective_ref_stride": 2.0, "status": "ready"}),
        encoding="utf-8",
    )
    out = tmp_path / "train.jsonl"

    entries = build_contact_tracking_manifest(reference_root=tmp_path, out=out)

    assert entries[0]["coordinate_system"] == "amass_zup"
    assert entries[0]["cache_path"] == str(cache_path)
    assert entries[0]["effective_ref_stride"] == 2.0
