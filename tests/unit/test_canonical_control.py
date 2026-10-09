from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import musclemimic.synergy.canonical_control as canonical_control
from musclemimic.synergy.canonical_control import (
    _array_hash,
    _publish_content_addressed_directory,
    load_canonical_control_artifact,
)
from musclemimic.synergy.primitive_catalog import canonical_json_sha256


def _artifact(tmp_path):
    control = np.full(354, 0.25, dtype=np.float64)
    payload = {
        "schema_version": "primitive_canonical_tonic_control_v1",
        "task_id": "P01_natural_stance",
        "catalog_fingerprint": "1" * 64,
        "controller_fingerprint": "2" * 64,
        "model_hash": "3" * 64,
        "actuator_schema_hash": "4" * 64,
        "ctrlrange_schema_hash": "5" * 64,
        "action_dim": 354,
        "aggregation": "coordinate_mean_float64_train_only_v1",
        "train_trials": [
            {
                "trial_id": "train",
                "split": "train",
                "motion_uid": 7,
                "source_motion_path": "primitive/train",
                "source_sha256": "6" * 64,
                "source_frame_interval": {"start_frame": 0, "end_frame_exclusive": 3, "source_total_frames": 3},
                "rollout_manifest_sha256": "7" * 64,
                "rollout_qc_sha256": "8" * 64,
                "initial_ctrl_sha256": "9" * 64,
            }
        ],
        "control": control.tolist(),
        "control_sha256": _array_hash(control),
    }
    fingerprint = canonical_json_sha256(payload)
    payload["artifact_fingerprint"] = fingerprint
    directory = tmp_path / fingerprint
    directory.mkdir()
    path = directory / "canonical_control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_control_is_path_and_content_addressed(tmp_path):
    path = _artifact(tmp_path)
    assert load_canonical_control_artifact(path)["control"][0] == 0.25
    moved = tmp_path / "wrong"
    moved.mkdir()
    (moved / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="artifact/path fingerprint"):
        load_canonical_control_artifact(moved)


def test_canonical_control_rejects_validation_provenance_and_tamper(tmp_path):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["train_trials"][0]["split"] = "val"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_canonical_control_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("action_dim", 353),
        lambda payload: payload["train_trials"][0].pop("source_sha256"),
        lambda payload: payload["train_trials"].append(dict(payload["train_trials"][0])),
    ],
)
def test_canonical_control_rejects_width_nested_schema_and_duplicates(tmp_path, mutation):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    mutation(payload)
    unsigned = dict(payload)
    unsigned.pop("artifact_fingerprint")
    payload["artifact_fingerprint"] = canonical_json_sha256(unsigned)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_canonical_control_artifact(path, require_path_binding=False)


def test_builder_rejects_non_mjb_catalog_model_before_reading_trials(tmp_path, monkeypatch):
    catalog = SimpleNamespace(
        model_artifact_path=tmp_path / "model.xml",
        enabled_tasks=(),
    )
    catalog.model_artifact_path.write_text("/etc/hosts is not a model", encoding="utf-8")
    monkeypatch.setattr(canonical_control, "load_primitive_catalog", lambda *args, **kwargs: catalog)
    with pytest.raises(ValueError, match="exact MJB"):
        canonical_control.build_canonical_control_artifact(tmp_path / "catalog.json", tmp_path / "out")


def test_builder_rejects_etc_hosts_bytes_disguised_as_mjb(tmp_path, monkeypatch):
    fake_mjb = tmp_path / "tampered.mjb"
    fake_mjb.write_bytes(open("/etc/hosts", "rb").read())
    catalog = SimpleNamespace(model_artifact_path=fake_mjb, enabled_tasks=())
    monkeypatch.setattr(canonical_control, "load_primitive_catalog", lambda *args, **kwargs: catalog)
    with pytest.raises(ValueError, match="failed to load"):
        canonical_control.build_canonical_control_artifact(tmp_path / "catalog.json", tmp_path / "out")


def test_content_addressed_directory_concurrent_publish_has_one_valid_winner(tmp_path):
    final = tmp_path / "fingerprint"
    temporary = [tmp_path / "tmp-a", tmp_path / "tmp-b"]
    for path in temporary:
        path.mkdir()
        (path / "payload").write_text("same", encoding="utf-8")

    def publish(path):
        _publish_content_addressed_directory(
            path,
            final,
            lambda winner: (
                (winner / "payload").read_text(encoding="utf-8") == "same"
                or (_ for _ in ()).throw(ValueError("bad winner"))
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, temporary))
    assert (final / "payload").read_text(encoding="utf-8") == "same"
    assert not any(path.exists() for path in temporary)
