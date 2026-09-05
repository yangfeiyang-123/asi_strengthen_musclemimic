from __future__ import annotations

import hashlib
import json
from pathlib import Path

import musclemimic.badminton.stage1_peasd_gate as tube_gate
import scripts.build_training_asset_manifest as asset_manifest
import scripts.server_training_preflight as preflight
from musclemimic.runner.checkpointing import _portable_source_mode


def _manifest(asset: Path, *, relative: str) -> dict:
    unsigned = {
        "schema_version": preflight.ASSET_MANIFEST_SCHEMA_VERSION,
        "action": "forehand_clear",
        "action_id": "forehandClear_standard",
        "release_binding_sha256": "a" * 64,
        "tube_gate_binding_sha256": "b" * 64,
        "includes_smplh": True,
        "file_count": 1,
        "total_bytes": asset.stat().st_size,
        "files": [
            {
                "path": relative,
                "num_bytes": asset.stat().st_size,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
        ],
    }
    return {
        **unsigned,
        "manifest_fingerprint": preflight._canonical_sha256(unsigned),
    }


def test_asset_manifest_validation_detects_content_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    asset = tmp_path / "datasets" / "motion.npz"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"motion-v1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(asset, relative="datasets/motion.npz")), encoding="utf-8")

    assert preflight.validate_asset_manifest(manifest)["passed"] is True
    asset.write_bytes(b"motion-v2")
    report = preflight.validate_asset_manifest(manifest)
    assert report["passed"] is False
    assert report["errors"] == ["asset content mismatch: datasets/motion.npz"]


def test_jax_cache_preflight_accepts_a_writable_server_path(tmp_path):
    target = tmp_path / "cache" / "task"

    writable, error = preflight._writable_directory(target)

    assert writable is True
    assert error is None
    assert target.is_dir()


def test_source_snapshot_mode_ignores_umask_but_binds_executable_bit(tmp_path):
    source = tmp_path / "launcher.sh"
    source.write_text("#!/bin/sh\n", encoding="utf-8")

    source.chmod(0o644)
    owner_write_only = _portable_source_mode(source, kind="file")
    source.chmod(0o664)
    group_writable = _portable_source_mode(source, kind="file")
    source.chmod(0o755)
    executable = _portable_source_mode(source, kind="file")

    assert owner_write_only == group_writable == 0o644
    assert executable == 0o755


def test_tube_gate_path_identity_is_portable_across_repo_roots(tmp_path, monkeypatch):
    first = tmp_path / "server-a" / "repo"
    second = tmp_path / "server-b" / "repo"
    relative = Path("artifacts/tube/emg_reference_manifest.json")
    for root in (first, second):
        manifest = root / relative
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(tube_gate, "REPO_ROOT", root)
        assert tube_gate._portable_repo_path(manifest) == relative.as_posix()


def test_private_asset_inventory_excludes_git_tracked_files(tmp_path, monkeypatch):
    tracked = tmp_path / "configs" / "recipe.json"
    private = tmp_path / "datasets" / "motion.npz"
    tracked.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    tracked.write_text("{}", encoding="utf-8")
    private.write_bytes(b"private motion")
    monkeypatch.setattr(asset_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(asset_manifest, "_git_tracked_files", lambda: {tracked.resolve()})
    monkeypatch.setattr(
        asset_manifest,
        "collect_action_assets",
        lambda _spec: (
            {tracked.resolve(), private.resolve()},
            {"release_binding_sha256": "a" * 64},
        ),
    )

    manifest = asset_manifest.build_manifest(
        action="forehand_clear",
        tube=None,
        include_smpl=False,
    )

    assert [record["path"] for record in manifest["files"]] == ["datasets/motion.npz"]


def test_release_split_manifests_collects_adjacent_text_lists_only(tmp_path):
    release = tmp_path / "release_manifest.json"
    release.write_text("{}", encoding="utf-8")
    train = tmp_path / "train_list.txt"
    validation = tmp_path / "val_list.txt"
    train.write_text("train.npz\n", encoding="utf-8")
    validation.write_text("validation.npz\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not a split manifest\n", encoding="utf-8")

    assert asset_manifest._release_split_manifests(release) == {
        train.resolve(),
        validation.resolve(),
    }


def test_asset_manifest_rejects_repository_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path / "repo")
    preflight.REPO_ROOT.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"private")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(outside, relative="../outside.bin")), encoding="utf-8")

    report = preflight.validate_asset_manifest(manifest)

    assert report["passed"] is False
    assert "asset escapes repository" in report["errors"][0]


def test_asset_manifest_rejects_tampering_and_stale_bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"licensed-model")
    payload = _manifest(asset, relative="model.bin")
    payload["action"] = "chinajump"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight.validate_asset_manifest(
        manifest,
        expected_action="forehand_clear",
        expected_release_binding="c" * 64,
        expected_tube_binding="d" * 64,
    )

    assert report["passed"] is False
    assert "asset manifest fingerprint mismatch" in report["errors"]
    assert "asset manifest belongs to another action" in report["errors"]
    assert "asset manifest release binding is stale" in report["errors"]
    assert "asset manifest verified-tube binding is stale" in report["errors"]
