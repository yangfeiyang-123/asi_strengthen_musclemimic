from __future__ import annotations

import json

import pytest

from musclemimic.badminton.promotion_artifact import (
    build_promoted_artifact,
    checkpoint_identity,
    sha256_path,
    validate_promoted_artifact,
    write_promoted_artifact,
)
from musclemimic.badminton.visual_review import (
    REVIEW_SCHEMA_VERSION,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
)
from musclemimic.distill.provenance import (
    STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA,
    begin_collection,
    checkpoint_content_fingerprint,
    load_dataset_manifest,
    teacher_promotion_evidence_kind,
    validate_dataset_manifest,
    validate_stage2_teacher_promotion,
    validate_teacher_promotion_binding,
    validate_teacher_promotion_manifest,
)
from musclemimic.runner.checkpointing import build_parent_checkpoint_lineage


def _sources(tmp_path, *, policy: bytes = b"frozen-policy"):
    run = tmp_path / "stage1-run"
    checkpoint = run / "checkpoint_30"
    (checkpoint / "metadata").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("complete", encoding="utf-8")
    (checkpoint / "metadata" / "metadata").write_text(
        json.dumps(
            {
                "update_number": 30,
                "global_timestep": 3000,
                "target_global_timestep": 3200,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "train_state.bin").write_bytes(policy)
    (run / "manifest.json").write_text(
        json.dumps({"config_hash": "abc123"}), encoding="utf-8"
    )
    identity = checkpoint_identity(checkpoint)
    progress = tmp_path / "promotion_progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema_version": "forehand_clear_promotion_progress_v3",
                "stage": "stage1",
                "checkpoint_dir": str(run.resolve()),
                "config_hash": "abc123",
                "validation_count": 5,
                "consecutive_pass_streak": 3,
                "stopped_early": True,
                "history": [
                    {
                        "update_number": 30,
                        "global_timestep": 3000,
                        "passed": True,
                        "metrics": {},
                        "checkpoint_identity": identity,
                        "validation_provenance": {
                            "semantics": "evaluate_all_once_per_heldout_v1"
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    clips = []
    for index in range(5):
        clips.append(
            {
                "review_kind": STAGE1_REVIEW_KIND,
                "motion": f"heldout/{index}",
                "artifact": f"videos/{index}.mp4",
                "candidate": identity,
                "major_swing_complete": True,
                "root_tracking_spike_free": True,
                "right_hand_tracking_spike_free": True,
                "passed": True,
                "notes": "reviewed from the bound checkpoint",
            }
        )
    review = tmp_path / "visual_review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "review_kind": STAGE1_REVIEW_KIND,
                "candidate": identity,
                "passed": True,
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, progress, review


def test_promoted_artifact_binds_checkpoint_progress_and_all_visual_clips(tmp_path):
    checkpoint, progress, review = _sources(tmp_path)
    payload = build_promoted_artifact(
        stage="stage1",
        checkpoint=checkpoint,
        promotion_progress=progress,
        visual_review=review,
    )
    output = tmp_path / "promoted.json"
    write_promoted_artifact(output, payload)

    validated = validate_promoted_artifact(
        output,
        expected_stage="stage1",
        expected_checkpoint=checkpoint,
    )
    assert validated["checkpoint"]["update_number"] == 30
    assert len(validated["checkpoint"]["checkpoint_content_sha256"]) == 64


def test_stage1_body_only_teacher_binding_is_explicit_and_revalidated(tmp_path):
    checkpoint, progress, review = _sources(tmp_path)
    promoted = tmp_path / "stage1-promoted.json"
    write_promoted_artifact(
        promoted,
        build_promoted_artifact(
            stage="stage1",
            checkpoint=checkpoint,
            promotion_progress=progress,
            visual_review=review,
        ),
    )
    teacher = checkpoint_content_fingerprint(checkpoint)

    with pytest.raises(ValueError, match="explicit teacher_role='body_only'"):
        validate_teacher_promotion_manifest(
            promoted,
            teacher_checkpoint=teacher,
            expected_stage="stage1",
        )

    binding = validate_teacher_promotion_manifest(
        promoted,
        teacher_checkpoint=teacher,
        expected_stage="stage1",
        teacher_role="body_only",
    )
    assert binding["schema_version"] == STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA
    assert binding["stage"] == "stage1"
    assert binding["teacher_role"] == "body_only"
    assert teacher_promotion_evidence_kind(binding) == "verified_stage1_promotion_v1"
    assert (
        validate_teacher_promotion_binding(
            binding,
            teacher_checkpoint=teacher,
            require_promoted=True,
            expected_stage="stage1",
            expected_teacher_role="body_only",
        )
        == binding
    )
    with pytest.raises(ValueError, match="stage mismatch"):
        validate_teacher_promotion_binding(
            binding,
            teacher_checkpoint=teacher,
            require_promoted=True,
            expected_stage="stage2",
        )

    transaction = begin_collection(
        dataset_dir=tmp_path / "stage1-distill-dataset",
        teacher_checkpoint=teacher,
        teacher_promotion=binding,
        teacher_promotion_stage="stage1",
        teacher_promotion_role="body_only",
        collector="teacher_lookahead_rollout",
        split="train",
        seed=0,
        motion_paths=["ChinaJump/optimized/train_a"],
        config_payload={"action": "chinajump"},
        request_payload={"num_transitions": 1},
        resume=False,
        run_uid="stage1-body-only-run",
    )
    assert transaction.manifest["teacher_promotion"] == binding
    validate_dataset_manifest(
        tmp_path / "stage1-distill-dataset",
        require_promoted_teacher=True,
    )

    # Dataset validation must re-read every bound source, not trust the
    # embedded artifact copy or its path alone.
    review.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"changed|differs|requires|candidate"):
        validate_dataset_manifest(
            tmp_path / "stage1-distill-dataset",
            require_promoted_teacher=True,
        )


def test_promoted_artifact_fails_after_checkpoint_or_review_mutation(tmp_path):
    checkpoint, progress, review = _sources(tmp_path)
    output = tmp_path / "promoted.json"
    write_promoted_artifact(
        output,
        build_promoted_artifact(
            stage="stage1",
            checkpoint=checkpoint,
            promotion_progress=progress,
            visual_review=review,
        ),
    )
    (checkpoint / "train_state.bin").write_bytes(b"different-policy")

    with pytest.raises(ValueError, match="changed|differs|mismatch"):
        validate_promoted_artifact(
            output,
            expected_stage="stage1",
            expected_checkpoint=checkpoint,
        )


def test_stage2_promoted_artifact_binds_parent_stage1_and_baseline_content(tmp_path):
    stage1_checkpoint, stage1_progress, stage1_review = _sources(tmp_path)
    parent_path = tmp_path / "stage1-promoted.json"
    write_promoted_artifact(
        parent_path,
        build_promoted_artifact(
            stage="stage1",
            checkpoint=stage1_checkpoint,
            promotion_progress=stage1_progress,
            visual_review=stage1_review,
        ),
    )

    run = tmp_path / "stage2-run"
    checkpoint = run / "checkpoint_80"
    (checkpoint / "metadata").mkdir(parents=True)
    (checkpoint / "metadata" / "metadata").write_text(
        json.dumps({"update_number": 80, "global_timestep": 8000}),
        encoding="utf-8",
    )
    (checkpoint / "train_state.bin").write_bytes(b"stage2-policy")
    stage1_parent_lineage = build_parent_checkpoint_lineage(
        checkpoint_identity(stage1_checkpoint),
        role="stage1_promoted",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "config_hash": "stage2-config",
                "parent_checkpoint_lineage": stage1_parent_lineage,
                "experiment_config": {
                    "parent_checkpoint_lineage": {
                        "required": True,
                        "role": "stage1_promoted",
                        "identity": stage1_parent_lineage,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    identity = checkpoint_identity(checkpoint)
    progress = tmp_path / "stage2-progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema_version": "forehand_clear_promotion_progress_v3",
                "stage": "stage2",
                "checkpoint_dir": str(run.resolve()),
                "config_hash": "stage2-config",
                "baseline_metrics_path": str(stage1_progress.resolve()),
                "baseline_metrics_sha256": sha256_path(stage1_progress),
                "validation_count": 5,
                "consecutive_pass_streak": 3,
                "stopped_early": True,
                "history": [
                    {
                        "update_number": 80,
                        "global_timestep": 8000,
                        "passed": True,
                        "metrics": {},
                        "checkpoint_identity": identity,
                        "validation_provenance": {
                            "semantics": "evaluate_all_once_per_heldout_v1"
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "stage2-review.json"
    clips = [
        {
            "review_kind": STAGE2_REVIEW_KIND,
            "motion": f"heldout/{index}",
            "artifact": f"videos/stage2-{index}.mp4",
            "candidate": identity,
            "major_swing_complete": True,
            "root_tracking_spike_free": True,
            "right_hand_tracking_spike_free": True,
            "racket_head_trajectory_ok": True,
            "racket_face_orientation_ok": True,
            "passed": True,
            "notes": "reviewed",
        }
        for index in range(5)
    ]
    review.write_text(
        json.dumps(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "review_kind": STAGE2_REVIEW_KIND,
                "candidate": identity,
                "passed": True,
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )
    promoted = tmp_path / "stage2-promoted.json"
    write_promoted_artifact(
        promoted,
        build_promoted_artifact(
            stage="stage2",
            checkpoint=checkpoint,
            promotion_progress=progress,
            visual_review=review,
            parent_promoted_artifact=parent_path,
        ),
    )
    validated = validate_promoted_artifact(
        promoted,
        expected_stage="stage2",
        expected_checkpoint=checkpoint,
    )
    assert validated["parent_promoted_artifact"]["checkpoint"] == checkpoint_identity(
        stage1_checkpoint
    )
    assert (
        validated["checkpoint"]["parent_checkpoint_lineage"]
        == stage1_parent_lineage
    )
    for sha_key in (
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
    ):
        assert stage1_parent_lineage["checkpoint"][sha_key] == validated[
            "parent_promoted_artifact"
        ]["checkpoint"][sha_key]

    teacher_fingerprint = checkpoint_content_fingerprint(checkpoint)
    binding = validate_stage2_teacher_promotion(
        promoted,
        teacher_checkpoint=teacher_fingerprint,
    )
    # The generic entrypoint defaults to Stage-2 and must preserve the exact
    # historical binding bytes/semantics.
    assert validate_teacher_promotion_manifest(
        promoted,
        teacher_checkpoint=teacher_fingerprint,
    ) == binding
    assert binding["artifact"]["checkpoint"]["update_number"] == 80
    assert binding["artifact"]["checkpoint"]["config_hash"] == "stage2-config"
    assert binding["artifact"]["visual_review"]["review_kind"] == STAGE2_REVIEW_KIND

    other_stage1_checkpoint, other_progress, other_review = _sources(
        tmp_path / "other-parent",
        policy=b"different-stage1-policy",
    )
    other_parent_path = tmp_path / "other-stage1-promoted.json"
    write_promoted_artifact(
        other_parent_path,
        build_promoted_artifact(
            stage="stage1",
            checkpoint=other_stage1_checkpoint,
            promotion_progress=other_progress,
            visual_review=other_review,
        ),
    )
    with pytest.raises(ValueError, match="lineage does not contain"):
        build_promoted_artifact(
            stage="stage2",
            checkpoint=checkpoint,
            promotion_progress=progress,
            visual_review=review,
            parent_promoted_artifact=other_parent_path,
        )

    transaction = begin_collection(
        dataset_dir=tmp_path / "distill-dataset",
        teacher_checkpoint=teacher_fingerprint,
        teacher_promotion=binding,
        collector="teacher_lookahead_rollout",
        split="train",
        seed=0,
        motion_paths=["ForehandClear/raw/train_a"],
        config_payload={"test": True},
        request_payload={"num_transitions": 1},
        resume=False,
        run_uid="promotion-bound-run",
    )
    assert transaction.manifest["teacher_promotion"] == binding
    assert load_dataset_manifest(tmp_path / "distill-dataset")[
        "teacher_promotion"
    ] == binding

    other_checkpoint = tmp_path / "checkpoint_81"
    other_checkpoint.mkdir()
    (other_checkpoint / "weights.bin").write_bytes(b"other-teacher")
    with pytest.raises(ValueError, match="different checkpoint"):
        validate_stage2_teacher_promotion(
            promoted,
            teacher_checkpoint=checkpoint_content_fingerprint(other_checkpoint),
        )

    stage1_progress.write_text("{}", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="changed|differs|requires|different stage|incompatible",
    ):
        validate_promoted_artifact(promoted, expected_stage="stage2")
