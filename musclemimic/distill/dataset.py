"""NPZ shard dataset utilities for policy distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REQUIRED_FIELDS = ("student_obs", "teacher_action")
SCHEMA_VERSION = "distill_v1"
ACTION_FIELDS = (
    "teacher_mu",
    "teacher_log_std",
    "student_action",
    "rollout_action",
)
SCALAR_FLOAT_FIELDS = (
    "teacher_value",
    "teacher_log_prob",
    "teacher_log_prob_teacher_mu",
    "teacher_log_prob_student_action",
    "teacher_log_prob_rollout_action",
    "reward",
    "phase",
)
SCALAR_BOOL_FIELDS = ("done", "absorbing", "used_teacher_action")
SCALAR_INT_FIELDS = ("traj_no", "subtraj_step_no")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _validate_data(data: dict[str, np.ndarray]) -> int:
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"distill shard missing required field: {field}")
    num_samples = int(np.asarray(data["student_obs"]).shape[0])
    for name, array in data.items():
        if np.asarray(array).shape[0] != num_samples:
            raise ValueError(
                f"field {name!r} has first dimension {np.asarray(array).shape[0]}, expected {num_samples}"
            )
    return num_samples


def complete_distill_schema(
    data: dict[str, np.ndarray],
    *,
    used_teacher_action_default: bool = False,
) -> dict[str, np.ndarray]:
    """Fill optional distillation diagnostics so mixed shards concatenate cleanly."""
    arrays = {name: np.asarray(value) for name, value in data.items()}
    for field in REQUIRED_FIELDS:
        if field not in arrays:
            raise ValueError(f"distill shard missing required field: {field}")

    student_obs = np.asarray(arrays["student_obs"])
    teacher_action = np.asarray(arrays["teacher_action"], dtype=np.float32)
    n = int(student_obs.shape[0])
    arrays["teacher_action"] = teacher_action

    arrays.setdefault("teacher_mu", teacher_action)
    arrays.setdefault("teacher_log_std", np.zeros_like(teacher_action, dtype=np.float32))
    arrays.setdefault("student_action", teacher_action)
    arrays.setdefault("rollout_action", arrays["student_action"])

    arrays.setdefault("teacher_value", np.zeros((n,), dtype=np.float32))
    arrays.setdefault("teacher_log_prob", np.zeros((n,), dtype=np.float32))
    arrays.setdefault("teacher_log_prob_teacher_mu", arrays["teacher_log_prob"])
    arrays.setdefault("teacher_log_prob_student_action", arrays["teacher_log_prob"])
    arrays.setdefault("teacher_log_prob_rollout_action", arrays["teacher_log_prob"])
    arrays.setdefault("reward", np.zeros((n,), dtype=np.float32))
    arrays.setdefault("phase", np.zeros((n,), dtype=np.float32))

    arrays.setdefault("done", np.zeros((n,), dtype=bool))
    arrays.setdefault("absorbing", np.zeros((n,), dtype=bool))
    arrays.setdefault("used_teacher_action", np.full((n,), bool(used_teacher_action_default), dtype=bool))

    arrays.setdefault("traj_no", np.full((n,), -1, dtype=np.int32))
    arrays.setdefault("subtraj_step_no", np.full((n,), -1, dtype=np.int32))

    for field in ACTION_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=np.float32)
    for field in SCALAR_FLOAT_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=np.float32)
    for field in SCALAR_BOOL_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=bool)
    for field in SCALAR_INT_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=np.int32)
    return arrays


def _infer_metadata(dataset_dir: Path, user_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    shards = sorted(
        {
            *dataset_dir.glob("shard_*.npz"),
            *dataset_dir.glob("train_*.npz"),
            *dataset_dir.glob("val_*.npz"),
            *dataset_dir.glob("test_*.npz"),
        }
    )
    if not shards:
        return dict(user_metadata or {})

    total = 0
    fields: list[str] | None = None
    student_obs_dim = None
    action_dim = None
    for shard in shards:
        with np.load(shard) as data:
            total += int(data["student_obs"].shape[0])
            fields = list(data.files) if fields is None else fields
            student_obs_dim = int(data["student_obs"].shape[-1])
            action_dim = int(data["teacher_action"].shape[-1])

    metadata = {"schema_version": SCHEMA_VERSION}
    metadata.update(user_metadata or {})
    metadata.update(
        {
            "num_samples": total,
            "student_obs_dim": student_obs_dim,
            "action_dim": action_dim,
            "fields": fields or [],
            "shards": [p.name for p in shards],
        }
    )
    return metadata


def write_distill_shard(path: str | Path, data: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> Path:
    """Write one distillation shard and update dataset metadata.json."""
    shard_path = Path(path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = complete_distill_schema(
        data,
        used_teacher_action_default=(metadata or {}).get("collector") == "teacher_lookahead_rollout",
    )
    _validate_data(arrays)
    np.savez_compressed(shard_path, **arrays)

    metadata_path = shard_path.parent / "metadata.json"
    merged_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        merged_metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata = _infer_metadata(shard_path.parent, merged_metadata)
    metadata_path.write_text(
        json.dumps(_jsonable(merged_metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return shard_path


def write_split_shard(
    dataset_dir: str | Path,
    data: dict[str, np.ndarray],
    *,
    split: str,
    shard_idx: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a split-prefixed distillation shard such as train_000000.npz."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported distill split {split!r}; expected train, val, or test")
    return write_distill_shard(
        Path(dataset_dir) / f"{split}_{int(shard_idx):06d}.npz",
        data,
        metadata={"split": split, **(metadata or {})},
    )


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return _infer_metadata(dataset_path)


class DistillDataset:
    """In-memory shard dataset for offline BC/KD training."""

    def __init__(self, dataset_dir: str | Path, split: str = "train", seed: int = 0):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.seed = int(seed)
        self.metadata = load_metadata(self.dataset_dir)
        self.shard_paths = sorted(self.dataset_dir.glob(f"{split}_*.npz"))
        if not self.shard_paths:
            self.shard_paths = sorted(self.dataset_dir.glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"no distill shards found in {self.dataset_dir}")

        loaded: dict[str, list[np.ndarray]] = {}
        for shard_path in self.shard_paths:
            with np.load(shard_path) as shard:
                shard_data = {field: np.asarray(shard[field]) for field in shard.files}
                legacy_teacher_like = "student_action" not in shard_data and "rollout_action" not in shard_data
                shard_data = complete_distill_schema(
                    shard_data,
                    used_teacher_action_default=legacy_teacher_like,
                )
                for field, array in shard_data.items():
                    loaded.setdefault(field, []).append(array)
        self.arrays = {field: np.concatenate(parts, axis=0) for field, parts in loaded.items()}
        self.num_samples = _validate_data(self.arrays)
        self.student_obs_dim = int(self.arrays["student_obs"].shape[-1])
        self.action_dim = int(self.arrays["teacher_action"].shape[-1])

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
    ) -> Iterator[dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.num_samples)
        while True:
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, self.num_samples, int(batch_size)):
                batch_idx = indices[start:start + int(batch_size)]
                yield {field: array[batch_idx] for field, array in self.arrays.items()}
            if not repeat:
                break
