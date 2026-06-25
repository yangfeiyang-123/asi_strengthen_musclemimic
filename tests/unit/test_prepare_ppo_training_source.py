import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "prepare_ppo_training_source.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_reference_bundle(root: Path, name: str, *, usable=True, tier="B", fps=60.0, frames=4) -> Path:
    bundle_dir = root / name / "reference_bundle"
    bundle_dir.mkdir(parents=True)
    np.savez(
        bundle_dir / "motion.npz",
        poses=np.zeros((frames, 72), dtype=np.float32),
        root_orient=np.zeros((frames, 3), dtype=np.float32),
        pose_body=np.zeros((frames, 69), dtype=np.float32),
        left_hand_pose=np.zeros((frames, 45), dtype=np.float32),
        right_hand_pose=np.zeros((frames, 45), dtype=np.float32),
        trans=np.zeros((frames, 3), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(fps, dtype=np.float32),
        mocap_frame_rate=np.asarray(fps, dtype=np.float32),
        frame_ids=np.arange(frames, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
    )
    np.savez(
        bundle_dir / "contact_schedule.npz",
        contact_confidence=np.ones((frames, 2), dtype=np.float32),
        stance_mask=np.ones((frames, 2), dtype=np.bool_),
        foot_points=np.zeros((frames, 2, 3), dtype=np.float32),
        foot_labels=np.asarray(["left_foot", "right_foot"]),
        frame_ids=np.arange(frames, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
    )
    (bundle_dir / "body_graph.json").write_text(json.dumps({"version": "body_graph_v1"}), encoding="utf-8")
    (bundle_dir / "quality_report.json").write_text(
        json.dumps({"usable_for_training": usable, "quality_tier": tier}),
        encoding="utf-8",
    )
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
        "quality": {"usable_for_training": usable, "quality_tier": tier},
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_prepare_reference_bundle_source_materializes_manifests_and_config(tmp_path):
    prepare = _load_module(SCRIPT, "prepare_ppo_training_source_for_test")
    reference_root = tmp_path / "reference_root"
    _write_reference_bundle(reference_root, "clip good 1", tier="B")
    _write_reference_bundle(reference_root, "clip good 2", tier="A")
    _write_reference_bundle(reference_root, "clip bad", tier="D", usable=False)

    result = prepare.prepare_reference_bundle_source(
        reference_root=reference_root,
        namespace="forehand_clear/ref_compare",
        output_config=tmp_path / "conf_ref.yaml",
        fps=60,
        num_envs=8,
        total_timesteps=1000,
        include_tiers={"A", "B"},
        min_frames=1,
        val_count=1,
        link_mode="copy",
        amass_root=tmp_path / "amass_npz",
        manifest_dir=tmp_path / "manifests",
        metadata_out=tmp_path / "metadata.json",
    )

    assert result.source_mode == "reference_bundle"
    assert len(result.train_motions) == 1
    assert len(result.val_motions) == 1
    assert (tmp_path / "amass_npz" / f"{result.train_motions[0]}.npz").is_file()
    assert result.train_manifest.read_text(encoding="utf-8").strip() == result.train_motions[0]
    assert result.val_manifest.read_text(encoding="utf-8").strip() == result.val_motions[0]

    config_text = result.output_config.read_text(encoding="utf-8")
    assert "source_mode: reference_bundle" in config_text
    assert "source:reference_bundle" in config_text
    assert "target_fps: 60" in config_text

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["source_mode"] == "reference_bundle"
    assert metadata["selected_count"] == 2
    assert metadata["skipped_count"] == 1


def test_prepare_existing_source_keeps_existing_manifests_and_labels_config(tmp_path):
    prepare = _load_module(SCRIPT, "prepare_existing_ppo_source_for_test")
    train_manifest = tmp_path / "train.txt"
    val_manifest = tmp_path / "val.txt"
    train_manifest.write_text("badminton/train/clip1\n", encoding="utf-8")
    val_manifest.write_text("badminton/val/clip2\n", encoding="utf-8")

    result = prepare.prepare_existing_source(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_config=tmp_path / "conf_existing.yaml",
        fps=30,
        num_envs=8,
        total_timesteps=1000,
    )

    assert result.source_mode == "existing_ppo"
    assert result.train_motions == ["badminton/train/clip1"]
    assert result.val_motions == ["badminton/val/clip2"]
    config_text = result.output_config.read_text(encoding="utf-8")
    assert "source_mode: existing_ppo" in config_text
    assert "source:existing_ppo" in config_text
