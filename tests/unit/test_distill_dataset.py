"""Tests for distillation dataset shard IO."""

import json

import numpy as np

from musclemimic.distill.dataset import DistillDataset, load_metadata, write_distill_shard


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


def test_load_metadata_accepts_existing_metadata_json(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"num_samples": 7, "student_obs_dim": 9, "action_dim": 3}),
        encoding="utf-8",
    )

    assert load_metadata(tmp_path)["num_samples"] == 7
