"""NPZ shard dataset utilities for policy distillation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from musclemimic.distill.action_schema import (
    ActionSelection,
    actuator_names_from_metadata,
    actuator_schema_hash,
    ordered_schema_hash,
)

logger = logging.getLogger(__name__)

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
SCALAR_INT_FIELDS = ("traj_no", "subtraj_step_no", "rollout_step", "env_index")
SCALAR_INT64_FIELDS = ("motion_uid", "rollout_uid")
OPTIONAL_FLOAT_ARRAY_FIELDS = ("reference_features",)


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
    for field in OPTIONAL_FLOAT_ARRAY_FIELDS:
        if field in arrays:
            arrays[field] = np.asarray(arrays[field], dtype=np.float32)
    for field in SCALAR_FLOAT_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=np.float32)
    for field in SCALAR_BOOL_FIELDS:
        arrays[field] = np.asarray(arrays[field], dtype=bool)
    for field in SCALAR_INT_FIELDS:
        if field in arrays:
            arrays[field] = np.asarray(arrays[field], dtype=np.int32)
    for field in SCALAR_INT64_FIELDS:
        if field in arrays:
            arrays[field] = np.asarray(arrays[field], dtype=np.int64)
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
    field_names: set[str] = set()
    student_obs_dim = None
    action_dim = None
    reference_features_dim = None
    for shard in shards:
        with np.load(shard) as data:
            total += int(data["student_obs"].shape[0])
            fields = list(data.files) if fields is None else fields
            field_names.update(str(name) for name in data.files)
            shard_obs_dim = int(data["student_obs"].shape[-1])
            shard_action_dim = int(data["teacher_action"].shape[-1])
            if student_obs_dim is not None and student_obs_dim != shard_obs_dim:
                raise ValueError(
                    f"student_obs dimension mismatch across distill shards: {student_obs_dim} vs "
                    f"{shard_obs_dim} in {shard}"
                )
            if action_dim is not None and action_dim != shard_action_dim:
                raise ValueError(
                    f"teacher_action dimension mismatch across distill shards: {action_dim} vs "
                    f"{shard_action_dim} in {shard}"
                )
            student_obs_dim = shard_obs_dim
            action_dim = shard_action_dim
            if "reference_features" in data:
                shard_ref_dim = int(data["reference_features"].shape[-1])
                if reference_features_dim is not None and reference_features_dim != shard_ref_dim:
                    raise ValueError(
                        "reference_features dimension mismatch across distill shards: "
                        f"{reference_features_dim} vs {shard_ref_dim} in {shard}"
                    )
                reference_features_dim = shard_ref_dim

    metadata = {"schema_version": SCHEMA_VERSION}
    metadata.update(user_metadata or {})
    inferred = {
        "num_samples": total,
        "student_obs_dim": student_obs_dim,
        "action_dim": action_dim,
        "fields": sorted(field_names) if field_names else (fields or []),
        "shards": [p.name for p in shards],
    }
    if reference_features_dim is not None:
        inferred["reference_features_dim"] = reference_features_dim
    metadata.update(inferred)
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
    metadata = _validated_action_metadata(metadata or {}, action_dim=int(arrays["teacher_action"].shape[-1]))
    np.savez_compressed(shard_path, **arrays)

    metadata_path = shard_path.parent / "metadata.json"
    merged_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        merged_metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    if metadata:
        merged_metadata = _merge_dataset_metadata(merged_metadata, metadata)
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

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        seed: int = 0,
        *,
        strict_schema: bool = False,
        required_optional_fields: tuple[str, ...] = (),
        target_actuator_names: tuple[str, ...] | list[str] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.seed = int(seed)
        self.strict_schema = bool(strict_schema)
        self.required_optional_fields = tuple(str(field) for field in required_optional_fields)
        self.metadata = load_metadata(self.dataset_dir)
        self.shard_paths = sorted(self.dataset_dir.glob(f"{split}_*.npz"))
        if not self.shard_paths:
            self.shard_paths = sorted(self.dataset_dir.glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"no distill shards found in {self.dataset_dir}")

        loaded: dict[str, list[np.ndarray]] = {}
        shard_field_sets: list[set[str]] = []
        raw_shard_field_sets: list[tuple[Path, set[str]]] = []
        for shard_path in self.shard_paths:
            with np.load(shard_path) as shard:
                raw_fields = set(str(field) for field in shard.files)
                raw_shard_field_sets.append((shard_path, raw_fields))
                missing_required = sorted(set(self.required_optional_fields) - raw_fields)
                if missing_required:
                    raise ValueError(
                        "strict_schema distill dataset requires fields "
                        f"{missing_required} missing from shard {shard_path.name}"
                    )
                shard_data = {field: np.asarray(shard[field]) for field in shard.files}
                legacy_teacher_like = "student_action" not in shard_data and "rollout_action" not in shard_data
                shard_data = complete_distill_schema(
                    shard_data,
                    used_teacher_action_default=legacy_teacher_like,
                )
                shard_field_sets.append(set(shard_data.keys()))
                for field, array in shard_data.items():
                    loaded.setdefault(field, []).append(array)
        # Optional fields (e.g. reference_features, full_obs) may be absent in some
        # shards when flags were toggled across runs/DAgger iterations. Concatenating a
        # field present in only a subset of shards yields fewer rows than num_samples and
        # crashes _validate_data. Drop any field not present in EVERY shard so mixed
        # datasets still load (the union-based metadata already records the inconsistency).
        common_fields = set.intersection(*shard_field_sets) if shard_field_sets else set()
        dropped = sorted(field for field in loaded if field not in common_fields)
        if dropped:
            if self.strict_schema:
                missing_by_field = {
                    field: [path.name for path, fields in raw_shard_field_sets if field not in fields]
                    for field in dropped
                }
                raise ValueError(
                    "strict_schema distill dataset has fields present in only some shards: "
                    f"{missing_by_field}"
                )
            logger.warning(
                "Distill dataset %s has fields present in only some shards; dropping to "
                "avoid row-count mismatch: %s",
                self.dataset_dir,
                dropped,
            )
            for field in dropped:
                loaded.pop(field, None)
        self.arrays = {field: np.concatenate(parts, axis=0) for field, parts in loaded.items()}
        self.num_samples = _validate_data(self.arrays)
        self.student_obs_dim = int(self.arrays["student_obs"].shape[-1])
        source_action_dim = int(self.arrays["teacher_action"].shape[-1])
        source_names = actuator_names_from_metadata(self.metadata, action_dim=source_action_dim)
        if source_names is None:
            if target_actuator_names is not None and len(target_actuator_names) == source_action_dim:
                # Legacy shards can only be accepted when the caller supplies the
                # complete ordered source schema.  A 416 -> 354 slice without names
                # is rejected because a dimensional guess could reorder muscles.
                source_names = [str(name) for name in target_actuator_names]
            else:
                source_names = [f"action_{index}" for index in range(source_action_dim)]
        self.action_selection = ActionSelection.from_names(
            source_actuator_names=source_names,
            target_actuator_names=target_actuator_names,
        )
        self._select_action_fields()
        self.source_actuator_names = list(self.action_selection.source_actuator_names)
        self.actuator_names = list(self.action_selection.target_actuator_names)
        self.source_action_dim = int(self.action_selection.source_dim)
        self.action_dim = int(self.action_selection.target_dim)
        self.action_schema_hash = self.action_selection.target_schema_hash
        self.source_actuator_ctrlrange = self._load_source_ctrlrange()
        self.actuator_ctrlrange = (
            None
            if self.source_actuator_ctrlrange is None
            else np.take(self.source_actuator_ctrlrange, self.action_selection.source_indices, axis=0)
        )

    def _load_source_ctrlrange(self) -> np.ndarray | None:
        payload = self.metadata.get("actuator_ctrlrange")
        if payload is None:
            return None
        ctrlrange = np.asarray(payload, dtype=np.float64)
        if ctrlrange.shape != (self.source_action_dim, 2) or not np.all(np.isfinite(ctrlrange)):
            raise ValueError(
                "distill actuator_ctrlrange must match source action schema: "
                f"expected=({self.source_action_dim}, 2) got={ctrlrange.shape}"
            )
        expected = ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={
                "actuator_names": self.source_actuator_names,
                "ctrlrange": ctrlrange.tolist(),
            },
        )
        supplied = self.metadata.get("ctrlrange_schema_hash")
        if supplied is not None and str(supplied) != expected:
            raise ValueError("distill ctrlrange_schema_hash mismatch")
        return ctrlrange

    def _select_action_fields(self) -> None:
        selection = self.action_selection
        for field in ("teacher_action",) + ACTION_FIELDS:
            if field not in self.arrays:
                continue
            self.arrays[field] = selection.select(self.arrays[field], field_name=field)

    def subset_by_motion_ids(
        self,
        motion_ids: set[int] | list[int] | tuple[int, ...],
        *,
        motion_field: str = "traj_no",
        split: str | None = None,
    ) -> "DistillDataset":
        """Return a row subset containing complete motions, never random frames."""
        if motion_field not in self.arrays:
            raise ValueError(f"motion split field {motion_field!r} is missing from distill dataset")
        selected_ids = np.asarray(sorted({int(value) for value in motion_ids}), dtype=np.int64)
        if selected_ids.size == 0:
            raise ValueError("motion subset cannot be empty")
        values = np.asarray(self.arrays[motion_field], dtype=np.int64)
        row_mask = np.isin(values, selected_ids)
        if not np.any(row_mask):
            raise ValueError(f"motion subset {selected_ids.tolist()} selected no rows")
        clone = object.__new__(type(self))
        clone.__dict__ = dict(self.__dict__)
        clone.arrays = {field: array[row_mask] for field, array in self.arrays.items()}
        clone.num_samples = int(np.sum(row_mask))
        clone.split = self.split if split is None else str(split)
        clone.selected_motion_ids = selected_ids.tolist()
        return clone

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


class LatentDistillDataset(DistillDataset):
    """Strict distillation dataset for latent posterior/prior/decoder training."""

    REQUIRED_LATENT_FIELDS = ("reference_features", "teacher_mu", "traj_no", "subtraj_step_no")
    STABLE_ID_FIELDS = ("motion_uid", "rollout_uid", "rollout_step", "env_index")

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        seed: int = 0,
        *,
        target_actuator_names: tuple[str, ...] | list[str] | None = None,
        require_stable_ids: bool = False,
    ):
        required_fields = self.REQUIRED_LATENT_FIELDS + (self.STABLE_ID_FIELDS if require_stable_ids else ())
        super().__init__(
            dataset_dir,
            split=split,
            seed=seed,
            strict_schema=True,
            required_optional_fields=required_fields,
            target_actuator_names=target_actuator_names,
        )
        if "reference_features" not in self.arrays:
            raise ValueError("latent distill dataset requires reference_features")
        if np.any(np.asarray(self.arrays["traj_no"]) < 0) or np.any(np.asarray(self.arrays["subtraj_step_no"]) < 0):
            raise ValueError("latent distill dataset requires non-negative traj_no and subtraj_step_no")
        self.require_stable_ids = bool(require_stable_ids)
        if self.require_stable_ids:
            for field in ("motion_uid", "rollout_uid", "rollout_step", "env_index"):
                if np.any(np.asarray(self.arrays[field]) < 0):
                    raise ValueError(f"latent distill dataset requires non-negative {field}")
        self.reference_features_dim = int(self.arrays["reference_features"].shape[-1])


class SequenceDistillDataset(LatentDistillDataset):
    """Latent dataset view that returns time-ordered fixed-horizon batches."""

    def _sequence_windows(self, horizon: int) -> list[np.ndarray]:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        stable_rollouts = "rollout_uid" in self.arrays and "rollout_step" in self.arrays
        identity = np.asarray(self.arrays["rollout_uid" if stable_rollouts else "traj_no"])
        step_no = np.asarray(self.arrays["rollout_step" if stable_rollouts else "subtraj_step_no"])
        windows: list[np.ndarray] = []
        for rollout in sorted(np.unique(identity).tolist()):
            rollout_indices = np.flatnonzero(identity == rollout)
            ordered = rollout_indices[np.argsort(step_no[rollout_indices], kind="stable")]
            ordered_steps = step_no[ordered]
            if stable_rollouts and len(np.unique(ordered_steps)) != len(ordered_steps):
                raise ValueError(f"rollout_uid={int(rollout)} contains duplicate rollout_step values")
            # Split at every discontinuity.  A fixed-horizon sample may never
            # bridge a dropped row, episode reset, or parallel environment.
            segment_start = 0
            boundaries = np.flatnonzero(np.diff(ordered_steps) != 1) + 1
            for segment_end in [*boundaries.tolist(), len(ordered)]:
                segment = ordered[segment_start:segment_end]
                for start in range(0, len(segment) - horizon + 1, horizon):
                    window = np.asarray(segment[start:start + horizon], dtype=np.int64)
                    if not np.all(np.diff(step_no[window]) == 1):
                        raise RuntimeError("sequence window continuity invariant violated")
                    windows.append(window)
                segment_start = segment_end
        return windows

    def iter_sequence_batches(
        self,
        batch_size: int,
        horizon: int,
        shuffle: bool = True,
        repeat: bool = False,
        drop_remainder: bool = False,
    ) -> Iterator[dict[str, np.ndarray]]:
        windows = self._sequence_windows(horizon)
        if not windows:
            raise ValueError(f"no sequence windows available for horizon={int(horizon)}")
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(windows))
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        while True:
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                if drop_remainder and len(selected) < batch_size:
                    continue
                batch_windows = np.stack([windows[int(index)] for index in selected], axis=0)
                yield {field: array[batch_windows] for field, array in self.arrays.items()}
            if not repeat:
                break


def motion_split_datasets(
    dataset_dir: str | Path,
    *,
    dataset_cls=DistillDataset,
    seed: int = 0,
    val_fraction: float = 0.1,
    motion_field: str = "motion_uid",
    target_actuator_names: tuple[str, ...] | list[str] | None = None,
    require_stable_ids: bool = False,
) -> tuple[DistillDataset, DistillDataset | None, dict[str, Any]]:
    """Load or create leakage-free train/validation datasets by whole motion.

    Explicit ``train_*.npz``/``val_*.npz`` shards are honored but still
    checked for motion overlap.  If no validation shards exist, whole motion
    IDs are deterministically held out from the training shards.
    """
    path = Path(dataset_dir)
    fraction = float(val_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")

    kwargs = {
        "seed": int(seed),
        "target_actuator_names": target_actuator_names,
    }
    if issubclass(dataset_cls, LatentDistillDataset):
        kwargs["require_stable_ids"] = bool(require_stable_ids)
    train = dataset_cls(path, split="train", **kwargs)
    explicit_val = bool(sorted(path.glob("val_*.npz")))
    if explicit_val:
        val = dataset_cls(path, split="val", **kwargs)
        if motion_field == "motion_uid" and (
            motion_field not in train.arrays or motion_field not in val.arrays
        ):
            raise ValueError(
                "explicit train/val distill shards require stable motion_uid; "
                "traj_no is local to each independently constructed environment"
            )
        train_ids = _motion_id_set(train, motion_field)
        val_ids = _motion_id_set(val, motion_field)
        overlap = sorted(train_ids & val_ids)
        if overlap:
            raise ValueError(
                "train/val motion leakage detected in explicit distill shards: "
                f"motion_field={motion_field!r} overlap={overlap}"
            )
        mode = "explicit_motion_shards"
    elif fraction > 0.0:
        all_ids = sorted(_motion_id_set(train, motion_field))
        if len(all_ids) < 2:
            raise ValueError(
                "motion-level validation requires at least two unique motions; "
                f"found {len(all_ids)} in field {motion_field!r}"
            )
        rng = np.random.default_rng(int(seed))
        shuffled = np.asarray(all_ids, dtype=np.int64)
        rng.shuffle(shuffled)
        val_count = min(len(all_ids) - 1, max(1, int(round(len(all_ids) * fraction))))
        val_ids = {int(value) for value in shuffled[:val_count]}
        train_ids = set(all_ids) - val_ids
        source = train
        train = source.subset_by_motion_ids(train_ids, motion_field=motion_field, split="train")
        val = source.subset_by_motion_ids(val_ids, motion_field=motion_field, split="val")
        mode = "deterministic_motion_holdout"
    else:
        train_ids = _motion_id_set(train, motion_field)
        val_ids = set()
        val = None
        mode = "train_only"

    manifest = {
        "schema_version": "motion_split_v1",
        "mode": mode,
        "motion_field": str(motion_field),
        "seed": int(seed),
        "val_fraction": fraction,
        "train_motion_ids": sorted(int(value) for value in train_ids),
        "val_motion_ids": sorted(int(value) for value in val_ids),
        "train_num_samples": int(train.num_samples),
        "val_num_samples": 0 if val is None else int(val.num_samples),
    }
    return train, val, manifest


def _motion_id_set(dataset: DistillDataset, motion_field: str) -> set[int]:
    if motion_field not in dataset.arrays:
        raise ValueError(f"motion split field {motion_field!r} is missing from distill dataset")
    values = np.asarray(dataset.arrays[motion_field])
    if values.ndim != 1:
        raise ValueError(f"motion split field {motion_field!r} must be rank-1, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"motion split field {motion_field!r} must contain integer IDs")
    ids = {int(value) for value in values.tolist()}
    if any(value < 0 for value in ids):
        raise ValueError(f"motion split field {motion_field!r} contains negative/unknown IDs")
    return ids


def _validated_action_metadata(metadata: dict[str, Any], *, action_dim: int) -> dict[str, Any]:
    result = dict(metadata)
    names = result.get("actuator_names", result.get("action_actuator_names"))
    if names is None:
        return result
    names = [str(name) for name in names]
    if len(names) != int(action_dim):
        raise ValueError(
            "actuator_names length must match teacher_action dimension when writing distill shard: "
            f"names={len(names)} action_dim={int(action_dim)}"
        )
    schema_hash = actuator_schema_hash(names)
    supplied_hash = result.get("action_schema_hash")
    if supplied_hash is not None and str(supplied_hash) != schema_hash:
        raise ValueError(
            f"action_schema_hash mismatch while writing distill shard: supplied={supplied_hash} computed={schema_hash}"
        )
    result["actuator_names"] = names
    result["action_schema_hash"] = schema_hash
    return result


_DATASET_ABI_METADATA_KEYS = (
    "actuator_names",
    "action_schema_hash",
    "actuator_ctrlrange",
    "ctrlrange_schema_hash",
    "student_state_schema_hash",
    "body_obs_schema_hash",
    "student_obs_filter",
    "student_obs_dim",
    "teacher_action_semantics",
    "teacher_mu_semantics",
    "normalized_action_bounds",
)


def _merge_dataset_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge shard metadata without allowing train/val ABI overwrite.

    Collection-specific values (motion paths, checkpoints, split) are retained
    under ``split_metadata``.  Ordered action/state contracts remain global and
    must be identical across every shard in the directory.
    """
    result = dict(existing)
    for key in _DATASET_ABI_METADATA_KEYS:
        if key in result and key in incoming and _jsonable(result[key]) != _jsonable(incoming[key]):
            raise ValueError(
                "distill dataset ABI metadata mismatch across shards: "
                f"key={key!r}; keep train/val state and ordered actuator schemas identical"
            )
    result.update(incoming)
    split = incoming.get("split")
    if split is not None:
        split_metadata = dict(existing.get("split_metadata") or {})
        split_metadata[str(split)] = _jsonable(incoming)
        result["split_metadata"] = split_metadata
    return result
