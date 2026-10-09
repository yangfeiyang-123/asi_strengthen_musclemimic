from __future__ import annotations

import json

import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError
from musclemimic.synergy.primitive_manifest import (
    PRIMITIVE_SOURCE_MANIFEST_SCHEMA_VERSION,
    build_primitive_source_manifest,
    canonical_rank_selection_rule,
    load_primitive_source_manifest,
    primitive_source_manifest_fingerprint,
    save_primitive_source_manifest,
    validate_primitive_source_manifest,
)


def _checkpoint_content(sha256: str, task: str) -> dict:
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": f"fixtures/{task}",
        "resolved_path": f"/fixtures/{task}",
        "sha256": sha256,
        "num_files": 1,
        "num_bytes": 1,
        "files": [{"path": "params", "sha256": "b" * 64, "num_bytes": 1}],
    }


def _kwargs() -> dict:
    return {
        "target_skill_id": "ChinaJump",
        "excluded_target_motion_paths": [
            "ChinaJump/forehandJump-1",
            "ChinaJump/forehandJump-8",
        ],
        "primitive_task_ids": ["squat", "jump", "landing"],
        "primitive_source_kinds": {
            "squat": "primitive",
            "jump": "primitive",
            "landing": "primitive",
        },
        "primitive_trial_ids": ["squat-01", "jump-01", "landing-01"],
        "train_motion_uids": [10, 11, 12],
        "validation_motion_uids": [20, 21],
        "source_checkpoint_fingerprints": {
            "squat": "1" * 64,
            "jump": "2" * 64,
            "landing": "3" * 64,
        },
        "source_checkpoint_contents": {
            "squat": _checkpoint_content("1" * 64, "squat"),
            "jump": _checkpoint_content("2" * 64, "jump"),
            "landing": _checkpoint_content("3" * 64, "landing"),
        },
        "primitive_required_phase_ids": {
            "squat": [0, 1],
            "jump": [1, 2],
            "landing": [2, 3],
        },
        "primitive_phase_schema_fingerprints": {
            "squat": "c" * 64,
            "jump": "d" * 64,
            "landing": "e" * 64,
        },
        "source_dataset_fingerprint": "9" * 64,
        "model_hash": "4" * 64,
        "actuator_schema_hash": "5" * 64,
        "control_range_hash": "6" * 64,
        "transform_ctrlrange_schema_hash": "a" * 64,
        "preprocessing_fingerprint": "7" * 64,
        "phase_weight_fingerprint": "8" * 64,
        "nmf_seeds": [17, 23],
    }


def test_save_load_round_trip_binds_all_primitive_sources(tmp_path):
    saved = save_primitive_source_manifest(tmp_path, **_kwargs())

    assert saved.path == tmp_path / "source_manifest.json"
    assert saved.manifest["schema_version"] == PRIMITIVE_SOURCE_MANIFEST_SCHEMA_VERSION
    assert saved.manifest["primitive_only"] is True
    assert saved.manifest["contains_target_skill_rollouts"] is False
    assert saved.manifest["rank_selection_rule"] == canonical_rank_selection_rule()
    assert saved.fingerprint == primitive_source_manifest_fingerprint(saved.manifest)

    loaded = load_primitive_source_manifest(
        saved.path,
        expected_fingerprint=saved.fingerprint,
    )
    assert loaded.manifest == saved.manifest
    assert loaded.fingerprint == saved.fingerprint


def test_load_rejects_duplicate_json_keys_before_validation(tmp_path):
    path = tmp_path / "source_manifest.json"
    path.write_text(
        '{"primitive_only":true,"primitive_only":false}',
        encoding="utf-8",
    )

    with pytest.raises(DuplicateJsonKeyError, match="duplicate JSON key"):
        load_primitive_source_manifest(path)


def test_manifest_tampering_and_expected_identity_mismatch_fail_closed(tmp_path):
    saved = save_primitive_source_manifest(tmp_path, **_kwargs())
    payload = json.loads(saved.path.read_text(encoding="utf-8"))
    payload["primitive_trial_ids"].append("injected-target-trial")
    saved.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_fingerprint mismatch"):
        load_primitive_source_manifest(saved.path)

    clean = build_primitive_source_manifest(**_kwargs())
    clean_path = tmp_path / "clean.json"
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from expected_fingerprint"):
        load_primitive_source_manifest(clean_path, expected_fingerprint="f" * 64)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        (
            "source_checkpoint_fingerprints",
            {"squat": "A" * 64, "jump": "2" * 64, "landing": "3" * 64},
        ),
        ("model_hash", "4" * 63),
        ("actuator_schema_hash", "not-a-sha"),
        ("control_range_hash", "6" * 65),
        ("transform_ctrlrange_schema_hash", "a" * 63),
        ("preprocessing_fingerprint", "g" * 64),
        ("phase_weight_fingerprint", 8),
    ],
)
def test_every_source_and_transform_identity_requires_sha256(field, invalid):
    kwargs = _kwargs()
    kwargs[field] = invalid

    with pytest.raises(ValueError, match="SHA-256"):
        build_primitive_source_manifest(**kwargs)


def test_train_validation_motion_identity_must_be_strictly_disjoint():
    kwargs = _kwargs()
    kwargs["validation_motion_uids"] = [20, 12]

    with pytest.raises(ValueError, match="motion_uids overlap"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["train_motion_uids"] = [10]
    kwargs["validation_motion_uids"] = ["10"]
    with pytest.raises(ValueError, match="int64 integers"):
        build_primitive_source_manifest(**kwargs)


def test_every_primitive_task_requires_an_exact_checkpoint_identity():
    kwargs = _kwargs()
    kwargs["source_checkpoint_fingerprints"].pop("landing")

    with pytest.raises(ValueError, match="keys must exactly match primitive_task_ids"):
        build_primitive_source_manifest(**kwargs)


def test_checkpoint_fingerprint_requires_grounded_content_audit():
    kwargs = _kwargs()
    kwargs["source_checkpoint_contents"].pop("landing")
    with pytest.raises(ValueError, match="keys must exactly match primitive_task_ids"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["source_checkpoint_contents"]["jump"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from source_checkpoint_fingerprints"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["source_checkpoint_contents"]["jump"]["files"] = []
    kwargs["source_checkpoint_contents"]["jump"]["num_files"] = 0
    kwargs["source_checkpoint_contents"]["jump"]["num_bytes"] = 0
    with pytest.raises(ValueError, match="files must be non-empty"):
        build_primitive_source_manifest(**kwargs)


def test_every_task_requires_explicit_phase_contract():
    kwargs = _kwargs()
    kwargs["primitive_required_phase_ids"].pop("landing")
    with pytest.raises(ValueError, match="keys must exactly match primitive_task_ids"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["primitive_required_phase_ids"]["jump"] = []
    with pytest.raises(ValueError, match="non-empty integer array"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["primitive_phase_schema_fingerprints"]["jump"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA-256"):
        build_primitive_source_manifest(**kwargs)


def test_target_skill_and_target_motion_inventory_cannot_enter_primitive_source():
    kwargs = _kwargs()
    kwargs["primitive_task_ids"] = ["squat", "ChinaJump", "landing"]
    kwargs["primitive_source_kinds"] = {
        "squat": "primitive",
        "ChinaJump": "primitive",
        "landing": "primitive",
    }
    kwargs["source_checkpoint_fingerprints"] = {
        "squat": "1" * 64,
        "ChinaJump": "2" * 64,
        "landing": "3" * 64,
    }
    kwargs["source_checkpoint_contents"] = {
        "squat": _checkpoint_content("1" * 64, "squat"),
        "ChinaJump": _checkpoint_content("2" * 64, "ChinaJump"),
        "landing": _checkpoint_content("3" * 64, "landing"),
    }
    kwargs["primitive_required_phase_ids"] = {
        "squat": [0, 1],
        "ChinaJump": [1, 2],
        "landing": [2, 3],
    }
    kwargs["primitive_phase_schema_fingerprints"] = {
        "squat": "c" * 64,
        "ChinaJump": "d" * 64,
        "landing": "e" * 64,
    }
    with pytest.raises(ValueError, match="exclude target_skill_id"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    target_uid = build_primitive_source_manifest(**_kwargs())["excluded_target_motions"][1]["motion_uid"]
    kwargs["train_motion_uids"] = [10, target_uid]
    with pytest.raises(ValueError, match="overlap excluded target motion_uids"):
        build_primitive_source_manifest(**kwargs)


def test_source_kind_inventory_is_exact_and_primitive_only():
    kwargs = _kwargs()
    kwargs["primitive_source_kinds"]["jump"] = "target_skill"
    with pytest.raises(ValueError, match="must be 'primitive'"):
        build_primitive_source_manifest(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("primitive_only", False, "primitive_only=true"),
        ("contains_target_skill_rollouts", True, "contains_target_skill_rollouts=false"),
    ],
)
def test_scope_flags_cannot_be_relabelled(field, value, message):
    payload = build_primitive_source_manifest(**_kwargs())
    payload[field] = value
    payload["manifest_fingerprint"] = primitive_source_manifest_fingerprint(payload)

    with pytest.raises(ValueError, match=message):
        validate_primitive_source_manifest(payload)


def test_rank_selection_cannot_enable_fallback_even_with_valid_self_hash():
    payload = build_primitive_source_manifest(**_kwargs())
    payload["rank_selection_rule"]["fallback_allowed"] = True
    payload["manifest_fingerprint"] = primitive_source_manifest_fingerprint(payload)

    with pytest.raises(ValueError, match="without fallback"):
        validate_primitive_source_manifest(payload)


def test_unknown_fields_and_invalid_expected_fingerprint_are_rejected():
    payload = build_primitive_source_manifest(**_kwargs())
    payload["unbound_note"] = "not part of the provenance contract"
    payload["manifest_fingerprint"] = primitive_source_manifest_fingerprint(payload)
    with pytest.raises(ValueError, match="unknown fields"):
        validate_primitive_source_manifest(payload)

    clean = build_primitive_source_manifest(**_kwargs())
    with pytest.raises(ValueError, match="expected_fingerprint.*SHA-256"):
        validate_primitive_source_manifest(clean, expected_fingerprint="not-a-sha")


def test_inventory_ids_are_nonempty_unique_and_nmf_seed_is_nonnegative():
    kwargs = _kwargs()
    kwargs["primitive_trial_ids"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="primitive_trial_ids.*duplicates"):
        build_primitive_source_manifest(**kwargs)

    kwargs = _kwargs()
    kwargs["nmf_seeds"] = [0, 0]
    with pytest.raises(ValueError, match="NMF_seeds.*distinct non-negative"):
        build_primitive_source_manifest(**kwargs)
