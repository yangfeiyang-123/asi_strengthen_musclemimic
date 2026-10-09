"""Tests for distillation dataset shard IO."""

import json

import pytest

import numpy as np

from musclemimic.distill.dataset import (
    DistillDataset,
    LatentDistillDataset,
    SequenceDistillDataset,
    load_metadata,
    write_distill_shard,
    write_split_shard,
)
from musclemimic.distill.inspect_dataset import inspect_distill_dataset


def _sample_data(offset: float, n: int = 3) -> dict[str, np.ndarray]:
    return {
        "student_obs": np.arange(offset, offset + n * 4, dtype=np.float32).reshape(n, 4),
        "teacher_action": np.full((n, 2), offset, dtype=np.float32),
        "teacher_value": np.arange(n, dtype=np.float32) + offset,
        "reward": np.ones(n, dtype=np.float32),
        "done": np.zeros(n, dtype=bool),
        "absorbing": np.zeros(n, dtype=bool),
        "traj_no": np.arange(n, dtype=np.int32),
        "subtraj_step_no": np.arange(n, dtype=np.int32) + 10,
        "phase": np.linspace(0.0, 1.0, n, dtype=np.float32),
    }


def test_write_distill_shard_creates_metadata_and_npz(tmp_path):
    data = _sample_data(0.0, n=4)
    shard_path = write_distill_shard(
        tmp_path / "shard_000000.npz",
        data,
        metadata={"teacher_ckpt": "/tmp/teacher", "split": "train"},
    )

    assert shard_path.is_file()
    metadata = load_metadata(tmp_path)
    assert metadata["teacher_ckpt"] == "/tmp/teacher"
    assert metadata["num_samples"] == 4
    assert metadata["schema_version"] == "distill_v1"
    assert metadata["student_obs_dim"] == 4
    assert metadata["action_dim"] == 2
    assert "student_obs" in metadata["fields"]

    loaded = np.load(shard_path)
    np.testing.assert_array_equal(loaded["student_obs"], data["student_obs"])
    np.testing.assert_array_equal(loaded["teacher_action"], data["teacher_action"])


def test_distill_dataset_iterates_across_multiple_shards(tmp_path):
    write_distill_shard(tmp_path / "shard_000000.npz", _sample_data(0.0, n=3), metadata={})
    write_distill_shard(tmp_path / "shard_000001.npz", _sample_data(100.0, n=2), metadata={})

    dataset = DistillDataset(tmp_path, split="train", seed=0)
    assert dataset.num_samples == 5
    assert dataset.student_obs_dim == 4
    assert dataset.action_dim == 2

    batches = list(dataset.iter_batches(batch_size=2, shuffle=False, repeat=False))
    assert [batch["student_obs"].shape[0] for batch in batches] == [2, 2, 1]
    np.testing.assert_array_equal(batches[0]["student_obs"], _sample_data(0.0, n=3)["student_obs"][:2])
    np.testing.assert_array_equal(batches[-1]["teacher_action"], _sample_data(100.0, n=2)["teacher_action"][1:])


def test_distill_dataset_loads_mixed_teacher_and_dagger_schema(tmp_path):
    teacher_data = _sample_data(0.0, n=2)
    dagger_data = {
        **_sample_data(100.0, n=3),
        "student_action": np.full((3, 2), 1.0, dtype=np.float32),
        "rollout_action": np.full((3, 2), 2.0, dtype=np.float32),
        "used_teacher_action": np.array([True, False, False], dtype=bool),
        "teacher_log_prob_teacher_mu": np.full(3, -0.1, dtype=np.float32),
        "teacher_log_prob_student_action": np.full(3, -1.0, dtype=np.float32),
        "teacher_log_prob_rollout_action": np.full(3, -0.5, dtype=np.float32),
    }
    write_split_shard(
        tmp_path,
        teacher_data,
        split="train",
        shard_idx=0,
        metadata={"collector": "teacher_lookahead_rollout"},
    )
    write_split_shard(
        tmp_path,
        dagger_data,
        split="train",
        shard_idx=1,
        metadata={"collector": "dagger_student_rollout_teacher_relabel"},
    )

    dataset = DistillDataset(tmp_path, split="train")

    assert dataset.num_samples == 5
    np.testing.assert_allclose(dataset.arrays["student_action"][:2], teacher_data["teacher_action"])
    np.testing.assert_allclose(dataset.arrays["rollout_action"][:2], teacher_data["teacher_action"])
    np.testing.assert_array_equal(dataset.arrays["used_teacher_action"][:2], np.ones(2, dtype=bool))
    assert dataset.arrays["teacher_log_prob_student_action"].shape == (5,)


def test_distill_dataset_backfills_legacy_mixed_schema_on_load(tmp_path):
    np.savez_compressed(
        tmp_path / "train_000000.npz",
        **_sample_data(0.0, n=2),
    )
    np.savez_compressed(
        tmp_path / "train_000001.npz",
        **{
            **_sample_data(100.0, n=3),
            "student_action": np.full((3, 2), 1.0, dtype=np.float32),
            "rollout_action": np.full((3, 2), 2.0, dtype=np.float32),
            "used_teacher_action": np.array([True, False, False], dtype=bool),
        },
    )

    dataset = DistillDataset(tmp_path, split="train")

    assert dataset.num_samples == 5
    np.testing.assert_allclose(dataset.arrays["student_action"][:2], dataset.arrays["teacher_action"][:2])
    np.testing.assert_array_equal(dataset.arrays["used_teacher_action"][:2], np.ones(2, dtype=bool))


def test_write_split_shard_creates_train_and_val_prefixes(tmp_path):
    train_path = write_split_shard(tmp_path, _sample_data(0.0, n=2), split="train", shard_idx=0)
    val_path = write_split_shard(tmp_path, _sample_data(10.0, n=1), split="val", shard_idx=0)

    assert train_path.name == "train_000000.npz"
    assert val_path.name == "val_000000.npz"
    assert DistillDataset(tmp_path, split="train").num_samples == 2
    assert DistillDataset(tmp_path, split="val").num_samples == 1


def test_inspect_distill_dataset_reports_split_stats(tmp_path):
    write_split_shard(tmp_path, _sample_data(0.0, n=2), split="train", shard_idx=0)

    report = inspect_distill_dataset(tmp_path)

    assert report["metadata"]["schema_version"] == "distill_v1"
    assert report["splits"]["train"]["num_samples"] == 2
    assert report["splits"]["train"]["field_stats"]["phase"]["min"] == 0.0


def test_inspect_distill_dataset_reports_shard_level_schema_without_aggregate_load(tmp_path):
    np.savez_compressed(
        tmp_path / "train_000000.npz",
        student_obs=np.zeros((2, 4), dtype=np.float32),
        teacher_action=np.zeros((2, 2), dtype=np.float32),
    )
    np.savez_compressed(
        tmp_path / "train_000001.npz",
        student_obs=np.zeros((3, 4), dtype=np.float32),
        teacher_action=np.zeros((3, 2), dtype=np.float32),
        student_action=np.ones((3, 2), dtype=np.float32),
    )

    report = inspect_distill_dataset(tmp_path, shard_level=True)

    assert report["shard_level"] is True
    assert [item["filename"] for item in report["shards"]] == ["train_000000.npz", "train_000001.npz"]
    assert report["shards"][0]["num_samples"] == 2
    assert report["shards"][0]["field_info"]["student_obs"]["shape"] == [2, 4]
    assert "student_action" in report["shards"][0]["missing_optional_fields"]
    assert report["shards"][1]["field_info"]["student_action"]["dtype"] == "float32"


def test_load_metadata_accepts_existing_metadata_json(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"num_samples": 7, "student_obs_dim": 9, "action_dim": 3}),
        encoding="utf-8",
    )

    assert load_metadata(tmp_path)["num_samples"] == 7


def test_distill_metadata_records_reference_feature_dimension_when_present(tmp_path):
    data = {
        **_sample_data(0.0, n=4),
        "reference_features": np.ones((4, 6), dtype=np.float32),
    }

    write_split_shard(tmp_path, data, split="train", shard_idx=0)

    metadata = load_metadata(tmp_path)
    dataset = DistillDataset(tmp_path, split="train")

    assert metadata["reference_features_dim"] == 6
    assert "reference_features" in metadata["fields"]
    assert dataset.arrays["reference_features"].shape == (4, 6)


def test_distill_dataset_loads_when_optional_field_present_in_only_some_shards(tmp_path):
    """Regression: mixing shards with/without an optional field (e.g. reference_features)
    must not crash on concatenation/validation; the partial field is dropped."""
    with_rf = {
        **_sample_data(0.0, n=3),
        "reference_features": np.ones((3, 6), dtype=np.float32),
    }
    without_rf = _sample_data(100.0, n=2)  # no reference_features

    write_split_shard(tmp_path, with_rf, split="train", shard_idx=0)
    write_split_shard(tmp_path, without_rf, split="train", shard_idx=1)

    dataset = DistillDataset(tmp_path, split="train")

    # All 5 rows load; required fields keep full length, partial optional field is dropped.
    assert dataset.num_samples == 5
    assert dataset.arrays["student_obs"].shape == (5, 4)
    assert "reference_features" not in dataset.arrays


def test_latent_distill_dataset_requires_reference_features_in_every_shard(tmp_path):
    with_rf = {
        **_sample_data(0.0, n=3),
        "reference_features": np.ones((3, 6), dtype=np.float32),
    }
    without_rf = _sample_data(100.0, n=2)

    write_split_shard(tmp_path, with_rf, split="train", shard_idx=0)
    write_split_shard(tmp_path, without_rf, split="train", shard_idx=1)

    with pytest.raises(ValueError, match="reference_features.*train_000001.npz"):
        LatentDistillDataset(tmp_path, split="train")


def test_distill_dataset_strict_schema_reports_partial_optional_fields(tmp_path):
    with_rf = {
        **_sample_data(0.0, n=3),
        "reference_features": np.ones((3, 6), dtype=np.float32),
    }
    without_rf = _sample_data(100.0, n=2)

    write_split_shard(tmp_path, with_rf, split="train", shard_idx=0)
    write_split_shard(tmp_path, without_rf, split="train", shard_idx=1)

    with pytest.raises(ValueError, match="strict_schema.*reference_features.*train_000001.npz"):
        DistillDataset(
            tmp_path,
            split="train",
            strict_schema=True,
            required_optional_fields=("reference_features",),
        )


def _emg_sample_data(offset: float, n: int = 3, channels: int = 3, synergies: int = 2) -> dict[str, np.ndarray]:
    """A shard carrying a full EMG reference row per transition."""
    return {
        **_sample_data(offset, n=n),
        "emg_anchor_mean": np.full((n, channels), 0.4, dtype=np.float32),
        "emg_anchor_scale": np.full((n, channels), 0.1, dtype=np.float32),
        "emg_channel_confidence": np.ones((n, channels), dtype=np.float32),
        "emg_synergy_mean": np.full((n, synergies), 0.5, dtype=np.float32),
        "emg_synergy_scale": np.full((n, synergies), 0.2, dtype=np.float32),
        "emg_synergy_valid": np.ones((n, synergies), dtype=np.float32),
    }


def test_emg_reference_dims_recorded_in_metadata(tmp_path):
    write_split_shard(tmp_path, _emg_sample_data(0.0, n=3, channels=5, synergies=3), split="train", shard_idx=0)
    write_split_shard(tmp_path, _emg_sample_data(50.0, n=2, channels=5, synergies=3), split="train", shard_idx=1)

    metadata = load_metadata(tmp_path)

    assert metadata["emg_anchor_dim"] == 5
    assert metadata["emg_synergy_dim"] == 3


def test_emg_reference_dim_mismatch_across_shards_is_rejected(tmp_path):
    write_split_shard(tmp_path, _emg_sample_data(0.0, channels=5), split="train", shard_idx=0)

    # The second write folds the new shard into the directory contract, so the
    # electrode-count disagreement surfaces here rather than at load time.
    with pytest.raises(ValueError, match="emg_anchor"):
        write_split_shard(tmp_path, _emg_sample_data(50.0, channels=4), split="train", shard_idx=1)


def test_emg_reference_fields_are_not_actuator_selected(tmp_path):
    # teacher_action is actuator space and must follow the reordering; the EMG
    # rows are electrode space and must survive it byte for byte.
    names = ["m0", "m1", "m2", "m3"]
    teacher_action = np.arange(12, dtype=np.float32).reshape(3, 4)
    anchor = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    data = {
        **_emg_sample_data(0.0, n=3, channels=3),
        "teacher_action": teacher_action,
        "emg_anchor_mean": anchor,
    }
    write_split_shard(tmp_path, data, split="train", metadata={"actuator_names": names})

    dataset = DistillDataset(tmp_path, split="train", target_actuator_names=["m3", "m1"])

    np.testing.assert_array_equal(dataset.arrays["teacher_action"], teacher_action[:, [3, 1]])
    np.testing.assert_array_equal(dataset.arrays["emg_anchor_mean"], anchor)
    assert dataset.arrays["emg_anchor_mean"].shape == (3, 3)
    assert dataset.arrays["emg_synergy_mean"].shape == (3, 2)


def test_emg_reference_rejects_negative_scale(tmp_path):
    data = _emg_sample_data(0.0, n=3)
    data["emg_anchor_scale"] = np.full((3, 3), -0.01, dtype=np.float32)

    with pytest.raises(ValueError, match=r"emg_anchor_scale.*non-negative"):
        write_split_shard(tmp_path, data, split="train")


def test_emg_reference_rejects_confidence_outside_unit_interval(tmp_path):
    data = _emg_sample_data(0.0, n=3)
    data["emg_channel_confidence"] = np.full((3, 3), 1.5, dtype=np.float32)

    with pytest.raises(ValueError, match=r"emg_channel_confidence.*\[0,1\]"):
        write_split_shard(tmp_path, data, split="train")


def test_emg_reference_rejects_non_finite_anchor_mean(tmp_path):
    data = _emg_sample_data(0.0, n=3)
    data["emg_anchor_mean"] = np.full((3, 3), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match=r"emg_anchor_mean.*finite"):
        write_split_shard(tmp_path, data, split="train")


def test_latent_dataset_require_emg_reference_names_missing_field_and_shard(tmp_path):
    write_split_shard(
        tmp_path,
        {**_emg_sample_data(0.0, n=3), "reference_features": np.ones((3, 6), dtype=np.float32)},
        split="train",
        shard_idx=0,
    )
    write_split_shard(
        tmp_path,
        {**_sample_data(100.0, n=2), "reference_features": np.ones((2, 6), dtype=np.float32)},
        split="train",
        shard_idx=1,
    )

    with pytest.raises(ValueError, match=r"emg_anchor_mean.*train_000001\.npz"):
        LatentDistillDataset(tmp_path, split="train", require_emg_reference=True)


def test_latent_dataset_without_require_emg_reference_tolerates_absent_rows(tmp_path):
    write_split_shard(
        tmp_path,
        {**_sample_data(0.0, n=3), "reference_features": np.ones((3, 6), dtype=np.float32)},
        split="train",
    )

    dataset = LatentDistillDataset(tmp_path, split="train")

    assert dataset.emg_anchor_dim is None
    assert "emg_anchor_mean" not in dataset.arrays


def test_shards_with_divergent_emg_semantics_cannot_share_a_directory(tmp_path):
    base = {
        "schema_version": 1,
        "emg_reference_semantics": {
            "schema_version": "emg_reference_capture_v1",
            "reference_id": "forehand_clear_v1",
            "channel_count": 3,
        },
    }
    other = {
        "schema_version": 1,
        "emg_reference_semantics": {
            "schema_version": "emg_reference_capture_v1",
            "reference_id": "smash_v1",
            "channel_count": 3,
        },
    }
    write_split_shard(tmp_path, _emg_sample_data(0.0), split="train", shard_idx=0, metadata=base)

    with pytest.raises(ValueError, match="emg_reference_semantics"):
        write_split_shard(tmp_path, _emg_sample_data(50.0), split="train", shard_idx=1, metadata=other)


def test_sequence_distill_dataset_batches_by_traj_and_step_order(tmp_path):
    data = {
        "student_obs": np.array(
            [
                [11, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [10, 0, 0, 0],
                [1, 0, 0, 0],
                [12, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "teacher_action": np.zeros((6, 2), dtype=np.float32),
        "reference_features": np.ones((6, 3), dtype=np.float32),
        "traj_no": np.array([1, 0, 0, 1, 0, 1], dtype=np.int32),
        "subtraj_step_no": np.array([1, 2, 0, 0, 1, 2], dtype=np.int32),
    }
    write_split_shard(tmp_path, data, split="train", shard_idx=0)

    dataset = SequenceDistillDataset(tmp_path, split="train")
    batches = list(dataset.iter_sequence_batches(batch_size=2, horizon=3, shuffle=False))

    assert len(batches) == 1
    batch = batches[0]
    assert batch["student_obs"].shape == (2, 3, 4)
    np.testing.assert_array_equal(batch["traj_no"], np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32))
    np.testing.assert_array_equal(batch["subtraj_step_no"], np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(batch["student_obs"][:, :, 0], np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32))
