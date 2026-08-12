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
    # Raw fixtures share one directory, so their producer-owned sibling path is
    # intentionally the same. Content validation belongs to primitive_ingest;
    # the catalog contract only requires that this evidence artifact exists.
    (root / "rollout_manifest.json").write_text("{}", encoding="utf-8")
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


def test_checked_in_p01_p12_catalog_is_structurally_complete():
    template = Path("fullbody/config_specific_task/stage1_body/primitive_catalog/chinajump_primitives_p01_p12_v1.json")
    catalog = load_primitive_catalog(template, require_build_ready=True)

    assert len(catalog.tasks) == 12
    assert [task.task_id[:3] for task in catalog.tasks] == [f"P{index:02d}" for index in range(1, 13)]
    assert [task.task_id for task in catalog.enabled_tasks] == ["P01_natural_stance"]
    assert catalog.regional_grouping_path is not None
    assert catalog.regional_grouping_path.is_file()
    assert all(len(task.phase_schema.fingerprint) == 64 for task in catalog.tasks)
    p01 = catalog.enabled_tasks[0]
    p12 = next(task for task in catalog.tasks if task.task_id == "P12_post_landing_recovery")
    assert [trial.split for trial in p01.trials] == ["train", "train", "val"]
    assert [trial.trial_id for trial in p01.trials] == [
        "P01-train-kit7-walking-straight-forwards07-95-125-seed3-finalimpl-a087cfg",
        "P01-train-kit7-walking-straight-forwards06-65-95-seed4-finalimpl-a087cfg",
        "P01-val-kit7-walking-straight-forwards03-78-108-seed5-finalimpl-a087cfg",
    ]
    assert len({trial.motion_uid for trial in p01.trials}) == 3
    assert all(trial.rollout_manifest_path.is_file() for trial in p01.trials)
    assert [trial.split for trial in p12.trials] == ["train", "train", "val"]
    assert [trial.trial_id for trial in p12.trials] == [
        "P12-train-amass-jumpingtwist-stand-449-487-seed51-v7-veltrack",
        "P12-train-amass-turntwist-stand-491-539-seed52-v7-veltrack",
        "P12-val-amass-punchkarate-stand-497-513-seed53-v7-veltrack",
    ]
    assert len({trial.motion_uid for trial in p12.trials}) == 3
    assert all(trial.rollout_manifest_path.is_file() for trial in p12.trials)
    assert not next(task for task in catalog.tasks if task.task_id == "P08_axial_rotation").enabled
    assert p12.enabled is False


def test_checked_in_p12_catalog_is_diagnostic_only_until_true_recovery_exists():
    catalog_path = Path("fullbody/config_specific_task/stage1_body/primitive_catalog/chinajump_primitives_p12_v1.json")
    catalog = load_primitive_catalog(catalog_path)

    assert [task.task_id for task in catalog.tasks] == ["P12_post_landing_recovery"]
    assert catalog.enabled_tasks == ()
    with pytest.raises(ValueError, match="at least one enabled task"):
        load_primitive_catalog(catalog_path, require_build_ready=True)


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


def test_catalog_rejects_reused_motion_within_training_split(tmp_path):
    catalog_path = _write_catalog(
        tmp_path,
        motion_paths=(
            "primitive/shared-train.npz",
            "primitive/shared-train.npz",
            "primitive/val-a.npz",
        ),
    )

    with pytest.raises(ValueError, match="requires independent source motions"):
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
