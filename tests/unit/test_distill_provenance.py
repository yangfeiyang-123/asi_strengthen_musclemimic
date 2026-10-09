from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.dataset import write_split_shard
from musclemimic.distill.provenance import (
    begin_collection,
    checkpoint_content_fingerprint,
    file_sha256,
    load_dataset_manifest,
    validate_dataset_manifest,
    validate_stage1_peasd_reference_promotion,
)


def _checkpoint(path, value: bytes):
    path.mkdir(parents=True)
    (path / "weights.bin").write_bytes(value)
    return checkpoint_content_fingerprint(path, canonicalize=False)


def _arrays(value: float = 0.0, n: int = 2):
    return {
        "student_obs": np.full((n, 3), value, dtype=np.float32),
        "teacher_action": np.full((n, 2), value, dtype=np.float32),
        "teacher_mu": np.full((n, 2), value, dtype=np.float32),
        "reference_features": np.full((n, 2), value, dtype=np.float32),
        "traj_no": np.zeros(n, dtype=np.int32),
        "subtraj_step_no": np.arange(n, dtype=np.int32),
        "motion_uid": np.full(n, 1, dtype=np.int64),
        "rollout_uid": np.full(n, 2, dtype=np.int64),
        "rollout_step": np.arange(n, dtype=np.int32),
        "env_index": np.zeros(n, dtype=np.int32),
    }


def _begin(
    dataset,
    teacher,
    *,
    resume: bool,
    collector: str = "teacher_lookahead_rollout",
    iteration: int | None = None,
    student=None,
    seed: int = 0,
):
    return begin_collection(
        dataset_dir=dataset,
        teacher_checkpoint=teacher,
        student_checkpoint=student,
        collector=collector,
        split="train",
        seed=seed,
        motion_paths=["ForehandClear/raw/motion_a"],
        config_payload={"model": "test", "iteration": iteration},
        request_payload={"num_transitions": 2, "shard_size": 2},
        resume=resume,
        run_uid="run-fixed",
        dagger_iteration=iteration,
        allow_test_only_unpromoted_teacher=True,
    )


def test_fresh_collection_rejects_nonempty_or_old_dagger_pollution(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(dataset / "train_999999.npz", **_arrays(9.0))

    with pytest.raises(ValueError, match="non-empty"):
        _begin(dataset, teacher, resume=False)


def test_production_collection_rejects_teacher_without_promotion_manifest(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    with pytest.raises(ValueError, match="teacher_promotion_manifest"):
        begin_collection(
            dataset_dir=tmp_path / "dataset",
            teacher_checkpoint=teacher,
            collector="teacher_lookahead_rollout",
            split="train",
            seed=0,
            motion_paths=["ForehandClear/raw/train_a"],
            config_payload={"test": True},
            request_payload={"num_transitions": 2},
            resume=False,
        )


def test_stage1_peasd_collection_binding_rebuilds_promotion_and_rejects_tamper(
    tmp_path,
    monkeypatch,
):
    promotion_path = tmp_path / "stage1_peasd_promotion.json"
    promotion_path.write_text('{"sealed":true}\n', encoding="utf-8")
    promotion = {
        "binding_sha256": "a" * 64,
        "emg_reference_binding": {
            "reference_fingerprint": "b" * 64,
            "array_bundle_sha256": "c" * 64,
            "mapping_sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(
        "musclemimic.badminton.stage1_peasd_gate.validate_stage1_peasd_teacher_promotion",
        lambda _path, *, expected_tube=None: promotion,
    )
    binding = {
        "path": str(promotion_path.resolve()),
        "content_sha256": file_sha256(promotion_path),
        "binding_sha256": promotion["binding_sha256"],
        "emg_reference_binding": promotion["emg_reference_binding"],
    }

    assert validate_stage1_peasd_reference_promotion(
        binding,
        expected_promotion=promotion_path,
        expected_tube=tmp_path / "tube.json",
    ) == binding

    promotion_path.write_text('{"sealed":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        validate_stage1_peasd_reference_promotion(binding)


def test_manifest_binds_exact_shard_set_hash_and_sample_count(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    dataset = tmp_path / "dataset"
    transaction = _begin(dataset, teacher, resume=False)
    staged = write_split_shard(
        transaction.output_dir,
        _arrays(1.0),
        split="train",
        metadata={"collector": "teacher_lookahead_rollout"},
    )
    committed = transaction.commit([staged])

    manifest = validate_dataset_manifest(dataset)
    assert manifest["totals"] == {"num_samples": 2, "num_shards": 1}
    assert manifest["shards"][0]["filename"] == committed[0].name
    assert len(manifest["shards"][0]["sha256"]) == 64
    contract = manifest["collections"][0]["contract"]
    assert contract["motion_paths"] == ["ForehandClear/raw/motion_a"]
    assert len(contract["config_fingerprint"]) == 64
    assert len(contract["motion_split_fingerprint"]) == 64
    with pytest.raises(ValueError, match="test-only unpromoted teacher"):
        validate_dataset_manifest(dataset, require_promoted_teacher=True)

    with np.load(committed[0]) as shard:
        changed = {name: np.asarray(shard[name]) for name in shard.files}
    changed["teacher_action"] = changed["teacher_action"] + 1.0
    np.savez_compressed(committed[0], **changed)
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_dataset_manifest(dataset)


def test_resume_rejects_teacher_mismatch_and_unmanifested_old_shard(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    other = _checkpoint(tmp_path / "other_teacher", b"different")
    dataset = tmp_path / "dataset"
    transaction = _begin(dataset, teacher, resume=False)
    staged = write_split_shard(transaction.output_dir, _arrays(), split="train")
    transaction.commit([staged])

    with pytest.raises(ValueError, match="teacher checkpoint"):
        _begin(dataset, other, resume=True)

    np.savez_compressed(dataset / "train_123456.npz", **_arrays(3.0))
    with pytest.raises(ValueError, match="shard set"):
        _begin(dataset, teacher, resume=True)


def test_dagger_iteration_is_idempotent_and_student_bound(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    student = _checkpoint(tmp_path / "student", b"student-v1")
    other_student = _checkpoint(tmp_path / "student-other", b"student-v2")
    dataset = tmp_path / "dataset"

    base = _begin(dataset, teacher, resume=False)
    base_shard = write_split_shard(base.output_dir, _arrays(0.0), split="train")
    base.commit([base_shard])

    dagger = _begin(
        dataset,
        teacher,
        resume=True,
        collector="dagger_student_rollout_teacher_relabel",
        iteration=0,
        student=student,
        seed=10,
    )
    dagger_shard = write_split_shard(dagger.output_dir, _arrays(1.0), split="train")
    first = dagger.commit([dagger_shard])
    manifest_before = load_dataset_manifest(dataset)

    repeated = _begin(
        dataset,
        teacher,
        resume=True,
        collector="dagger_student_rollout_teacher_relabel",
        iteration=0,
        student=student,
        seed=10,
    )
    assert repeated.already_complete is True
    assert repeated.existing_paths == first
    assert load_dataset_manifest(dataset)["manifest_fingerprint"] == manifest_before["manifest_fingerprint"]
    assert len(list(dataset.glob("train_*.npz"))) == 2

    with pytest.raises(ValueError, match="different immutable contract"):
        _begin(
            dataset,
            teacher,
            resume=True,
            collector="dagger_student_rollout_teacher_relabel",
            iteration=0,
            student=other_student,
            seed=10,
        )


def test_manifest_json_tampering_fails_closed(tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    dataset = tmp_path / "dataset"
    transaction = _begin(dataset, teacher, resume=False)
    staged = write_split_shard(transaction.output_dir, _arrays(), split="train")
    transaction.commit([staged])
    path = dataset / "dataset_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_uid"] = "forged"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_fingerprint"):
        load_dataset_manifest(dataset)


def test_latent_train_rejects_dataset_from_different_teacher(monkeypatch, tmp_path):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    other = _checkpoint(tmp_path / "other_teacher", b"other")
    dataset = tmp_path / "dataset"
    transaction = _begin(dataset, teacher, resume=False)
    staged = write_split_shard(transaction.output_dir, _arrays(), split="train")
    transaction.commit([staged])

    from musclemimic.latent_muscle import train_latent as module
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig

    monkeypatch.setattr(module, "checkpoint_content_fingerprint", lambda _path: other)
    with pytest.raises(ValueError, match="teacher checkpoint fingerprint mismatch"):
        module.train_latent(
            LatentTrainConfig(
                dataset_dir=str(dataset),
                output_dir=str(tmp_path / "output"),
                teacher_ckpt="/unused",
                require_dataset_provenance=True,
                test_only_allow_unpromoted_teacher=True,
                num_steps=1,
            )
        )


def test_latent_production_rejects_missing_teacher_promotion_manifest(
    monkeypatch, tmp_path
):
    teacher = _checkpoint(tmp_path / "teacher", b"teacher")
    dataset = tmp_path / "dataset"
    transaction = _begin(dataset, teacher, resume=False)
    staged = write_split_shard(transaction.output_dir, _arrays(), split="train")
    transaction.commit([staged])

    from musclemimic.latent_muscle import train_latent as module
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig

    monkeypatch.setattr(module, "checkpoint_content_fingerprint", lambda _path: teacher)
    with pytest.raises(ValueError, match="teacher_promotion_manifest"):
        module.train_latent(
            LatentTrainConfig(
                dataset_dir=str(dataset),
                output_dir=str(tmp_path / "output"),
                teacher_ckpt="/unused",
                require_dataset_provenance=True,
                num_steps=1,
            )
        )
