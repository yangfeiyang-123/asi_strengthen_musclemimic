from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fullbody.run_forehand_clear_pipeline import _require_stage3_artifact_binding
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _build_stage3_artifact_binding,
    _stage3_evaluation_content_sha256,
)


def _bound_report(tmp_path):
    checkpoint = tmp_path / "policy_latest.npz"
    np.savez(checkpoint, placeholder=np.asarray([1.0]))
    train_feed = {"schema_version": "feed_v1", "content_sha256": "a" * 64}
    eval_feed = {"schema_version": "feed_v1", "content_sha256": "b" * 64}
    prerequisite_paths = {}
    for name in ("preflight", "base_only", "feed_check"):
        path = tmp_path / f"{name}_report.json"
        path.write_text(json.dumps({"passed": True, "name": name}), encoding="utf-8")
        prerequisite_paths[name] = path
    prerequisite = {
        "schema_version": "stage3_training_prerequisite_binding_v1",
        "preflight_report_path": str(prerequisite_paths["preflight"]),
        "preflight_report_sha256": hashlib.sha256(
            prerequisite_paths["preflight"].read_bytes()
        ).hexdigest(),
        "base_only_report_path": str(prerequisite_paths["base_only"]),
        "base_only_report_sha256": hashlib.sha256(
            prerequisite_paths["base_only"].read_bytes()
        ).hexdigest(),
        "feed_check_report_path": str(prerequisite_paths["feed_check"]),
        "feed_check_report_sha256": hashlib.sha256(
            prerequisite_paths["feed_check"].read_bytes()
        ).hexdigest(),
        "latent_checkpoint_fingerprint": "c" * 64,
        "control_hash": "control-hash",
        "training_feed_manifest_sha256": hashlib.sha256(
            json.dumps(
                train_feed,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "verified": True,
    }
    prerequisite["binding_sha256"] = hashlib.sha256(
        json.dumps(
            prerequisite,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "control_hash": "control-hash",
        "training_feed_manifest": train_feed,
        "iteration": 10,
        "env_steps": 20_000_000,
        "curriculum_complete": True,
        "promotion_eligible": True,
        "training_prerequisite_binding": prerequisite,
        "curriculum_state": {
            "effective_steps": 20_000_000,
            "phase": "lambda_full",
        },
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "train_report.json").write_text(
        json.dumps(
            {
                "iterations": 10,
                "env_steps": 20_000_000,
                "curriculum_complete": True,
                "promotion_eligible": True,
                "curriculum_effective_steps": 20_000_000,
                "curriculum_phase": "lambda_full",
                "checkpoint": str(checkpoint),
                "training_prerequisite_binding": prerequisite,
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.yaml"
    scene = tmp_path / "scene.xml"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    paths = SimpleNamespace(spec_path=spec, scene_xml=scene)
    control = {
        "control_hash": "control-hash",
        "latent_checkpoint_fingerprint": "c" * 64,
    }
    report_payload = {
        "checkpoint": str(checkpoint),
        "control_manifest": control,
        "training_feed_manifest": train_feed,
        "evaluation_feed_manifest": eval_feed,
        "episodes": [{"hit": True}],
        "hit_rate": 1.0,
    }
    binding = _build_stage3_artifact_binding(
        paths=paths,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        control_manifest=control,
        training_feed_manifest=train_feed,
        evaluation_feed_manifest=eval_feed,
        evaluation_content_sha256=_stage3_evaluation_content_sha256(report_payload),
    )
    report = tmp_path / "evaluate_report.json"
    report_payload["artifact_binding"] = binding
    report.write_text(
        json.dumps(report_payload),
        encoding="utf-8",
    )
    return report, spec


def test_stage3_binding_recomputes_checkpoint_spec_scene_control_and_feed_hashes(
    tmp_path,
):
    report, _ = _bound_report(tmp_path)
    _require_stage3_artifact_binding(report)


def test_stage3_binding_rejects_mutated_bound_source(tmp_path):
    report, spec = _bound_report(tmp_path)
    spec.write_text("runner_type: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bound artifact changed"):
        _require_stage3_artifact_binding(report)


def test_stage3_binding_rejects_mutated_evaluation_metrics(tmp_path):
    report, _ = _bound_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["hit_rate"] = 0.0
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics/content changed"):
        _require_stage3_artifact_binding(report)


def test_stage3_binding_rejects_missing_training_prerequisite_evidence(tmp_path):
    report, _ = _bound_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    metadata_path = Path(payload["artifact_binding"]["checkpoint_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("training_prerequisite_binding")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="bound artifact changed"):
        _require_stage3_artifact_binding(report)


def test_stage3_evaluation_content_hash_canonicalizes_nonfinite_failure_metrics():
    raw = {
        "passed": False,
        "hit_rate": 0.0,
        "contact_racket_head_speed_m_s": float("-inf"),
        "net_clearance_m": float("inf"),
    }
    persisted = {
        "passed": False,
        "hit_rate": 0.0,
        "contact_racket_head_speed_m_s": None,
        "net_clearance_m": None,
    }

    assert _stage3_evaluation_content_sha256(raw) == (
        _stage3_evaluation_content_sha256(persisted)
    )


def test_stage3_binding_rejects_new_report_backing_old_incomplete_checkpoint(tmp_path):
    checkpoint = tmp_path / "policy_latest.npz"
    np.savez(checkpoint, placeholder=np.asarray([1.0]))
    train_feed = {"schema_version": "feed_v1"}
    metadata = {
        "control_hash": "control-hash",
        "training_feed_manifest": train_feed,
        "iteration": 5,
        "env_steps": 10_000_000,
        "curriculum_complete": False,
        "promotion_eligible": False,
        "curriculum_state": {"effective_steps": 10_000_000, "phase": "warmup"},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "train_report.json").write_text(
        json.dumps(
            {
                "iterations": 10,
                "env_steps": 20_000_000,
                "curriculum_complete": True,
                "promotion_eligible": True,
                "curriculum_effective_steps": 20_000_000,
                "curriculum_phase": "lambda_full",
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.yaml"
    scene = tmp_path / "scene.xml"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint metadata records an incomplete"):
        _build_stage3_artifact_binding(
            paths=SimpleNamespace(spec_path=spec, scene_xml=scene),
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            control_manifest={
                "control_hash": "control-hash",
                "latent_checkpoint_fingerprint": "c" * 64,
            },
            training_feed_manifest=train_feed,
            evaluation_feed_manifest={"schema_version": "eval_feed_v1"},
            evaluation_content_sha256="e" * 64,
        )
