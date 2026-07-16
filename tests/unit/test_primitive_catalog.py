from __future__ import annotations

import json
from pathlib import Path

import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError
from musclemimic.synergy.primitive_catalog import (
    load_primitive_catalog,
    load_primitive_phase_schema,
)


def _write_phase(path: Path, *, task_id: str = "P01_test") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "primitive_phase_schema_v1",
                "task_id": task_id,
                "phases": [
                    {"id": 0, "name": "prepare", "definition": "Stable preparation."},
                    {"id": 1, "name": "execute", "definition": "Execute the primitive."},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_catalog(
    root: Path,
    *,
    motion_paths: tuple[str, str, str] = (
        "primitive/train-a.npz",
        "primitive/train-b.npz",
        "primitive/val-a.npz",
    ),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_phase(root / "phase.json")
    (root / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    (root / "controller").mkdir()
    (root / "controller" / "params").write_bytes(b"controller")
    (root / "groups.json").write_text(
        json.dumps({"regions": {"all": ["a"]}}),
        encoding="utf-8",
    )
    trials = []
    for index, (split, motion_path) in enumerate(zip(("train", "train", "val"), motion_paths, strict=True)):
        raw = root / f"raw-{index}.npz"
        raw.write_bytes(b"placeholder")
        trials.append(
            {
                "trial_id": f"trial-{index}",
                "split": split,
                "motion_path": motion_path,
                "raw_npz_path": raw.name,
                "success": True,
                "quality_weight": 1.0,
            }
        )
    payload = {
        "schema_version": "primitive_synergy_catalog_v1",
        "catalog_id": "fixture",
        "target_skill_id": "ChinaJump",
        "model_xml_path": "model.xml",
        "expected_action_dim": 1,
        "regional_grouping_path": "groups.json",
        "tasks": [
            {
                "task_id": "P01_test",
                "display_name": "Test primitive",
                "enabled": True,
                "controller_artifact": "controller",
                "phase_schema_path": "phase.json",
                "trials": trials,
            }
        ],
    }
    path = root / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checked_in_p01_p12_template_is_structurally_complete():
    template = Path("fullbody/config_specific_task/stage1_body/primitive_catalog/chinajump_primitives_p01_p12_v1.json")
    catalog = load_primitive_catalog(template)

    assert len(catalog.tasks) == 12
    assert [task.task_id[:3] for task in catalog.tasks] == [f"P{index:02d}" for index in range(1, 13)]
    assert not catalog.enabled_tasks
    assert catalog.regional_grouping_path is not None
    assert catalog.regional_grouping_path.is_file()
    assert all(len(task.phase_schema.fingerprint) == 64 for task in catalog.tasks)

    with pytest.raises(ValueError, match="requires a compiled model artifact"):
        load_primitive_catalog(template, require_build_ready=True)


def test_phase_schema_identity_is_derived_from_current_semantics(tmp_path):
    path = _write_phase(tmp_path / "phase.json")
    original = load_primitive_phase_schema(path, expected_task_id="P01_test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phases"][1]["definition"] = "Changed event semantics."
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed = load_primitive_phase_schema(path, expected_task_id="P01_test")

    assert original.fingerprint != changed.fingerprint


def test_catalog_rejects_train_validation_motion_leakage(tmp_path):
    catalog_path = _write_catalog(
        tmp_path,
        motion_paths=(
            "primitive/shared.npz",
            "primitive/train-b.npz",
            "primitive/shared.npz",
        ),
    )

    with pytest.raises(ValueError, match="motion leakage"):
        load_primitive_catalog(catalog_path, require_build_ready=True)


def test_catalog_rejects_any_target_skill_namespace_from_primitive_w(tmp_path):
    catalog_path = _write_catalog(
        tmp_path,
        motion_paths=(
            "primitive/train-a.npz",
            "ChinaJump/full-target-rollout-03.npz",
            "primitive/val-a.npz",
        ),
    )

    with pytest.raises(ValueError, match="target-skill namespace"):
        load_primitive_catalog(catalog_path, require_build_ready=True)


def test_catalog_and_phase_json_reject_duplicate_or_unknown_fields(tmp_path):
    duplicate = tmp_path / "duplicate-phase.json"
    duplicate.write_text(
        '{"schema_version":"primitive_phase_schema_v1","task_id":"a","task_id":"b","phases":[]}',
        encoding="utf-8",
    )
    with pytest.raises(DuplicateJsonKeyError):
        load_primitive_phase_schema(duplicate)

    catalog_path = _write_catalog(tmp_path / "catalog-root")
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["unbound_hash"] = "user-supplied"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_primitive_catalog(catalog_path)
