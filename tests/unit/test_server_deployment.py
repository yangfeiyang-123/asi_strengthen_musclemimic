from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.server_training_preflight as preflight


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
