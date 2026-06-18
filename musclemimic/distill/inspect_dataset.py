"""Inspect distillation dataset shards before BC/KD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.dataset import (
    ACTION_FIELDS,
    REQUIRED_FIELDS,
    SCALAR_BOOL_FIELDS,
    SCALAR_FLOAT_FIELDS,
    SCALAR_INT_FIELDS,
    DistillDataset,
    load_metadata,
)


OPTIONAL_FIELDS = tuple(
    field
    for field in (
        *ACTION_FIELDS,
        *SCALAR_FLOAT_FIELDS,
        *SCALAR_BOOL_FIELDS,
        *SCALAR_INT_FIELDS,
    )
    if field not in REQUIRED_FIELDS
)


def _stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    if arr.size == 0:
        return {"shape": list(arr.shape)}
    if arr.dtype == bool:
        return {"shape": list(arr.shape), "mean": float(arr.mean())}
    if np.issubdtype(arr.dtype, np.number):
        return {
            "shape": list(arr.shape),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _infer_split(path: Path) -> str | None:
    stem = path.stem
    for split in ("train", "val", "test"):
        if stem.startswith(f"{split}_"):
            return split
    return None


def inspect_distill_shards(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Return per-shard diagnostics without constructing DistillDataset."""
    dataset_path = Path(dataset_dir)
    shards = sorted(
        {
            *dataset_path.glob("shard_*.npz"),
            *dataset_path.glob("train_*.npz"),
            *dataset_path.glob("val_*.npz"),
            *dataset_path.glob("test_*.npz"),
        }
    )
    payload: list[dict[str, Any]] = []
    for shard_path in shards:
        with np.load(shard_path) as shard:
            fields = sorted(shard.files)
            field_info = {
                field: {"shape": list(shard[field].shape), "dtype": str(shard[field].dtype)}
                for field in fields
            }
            num_samples = int(shard["student_obs"].shape[0]) if "student_obs" in shard else None
            payload.append(
                {
                    "filename": shard_path.name,
                    "split": _infer_split(shard_path),
                    "num_samples": num_samples,
                    "fields": fields,
                    "field_info": field_info,
                    "missing_required_fields": [field for field in REQUIRED_FIELDS if field not in shard],
                    "missing_optional_fields": [field for field in OPTIONAL_FIELDS if field not in shard],
                }
            )
    return payload


def inspect_distill_dataset(dataset_dir: str | Path, *, shard_level: bool = False) -> dict[str, Any]:
    """Return split and field diagnostics for a distillation dataset directory."""
    dataset_path = Path(dataset_dir)
    metadata = load_metadata(dataset_path)
    payload: dict[str, Any] = {
        "dataset_dir": str(dataset_path),
        "metadata": metadata,
        "shard_level": bool(shard_level),
        "splits": {},
    }
    if shard_level:
        payload["shards"] = inspect_distill_shards(dataset_path)
        return payload

    for split in ("train", "val", "test"):
        try:
            dataset = DistillDataset(dataset_path, split=split)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            payload["splits"][split] = {
                "error": str(exc),
                "recommendation": "rerun with --shard_level to inspect individual NPZ shard schemas",
            }
            continue
        split_payload: dict[str, Any] = {
            "num_samples": int(dataset.num_samples),
            "student_obs_dim": int(dataset.student_obs_dim),
            "action_dim": int(dataset.action_dim),
            "fields": sorted(dataset.arrays),
            "field_stats": {},
        }
        for field in (
            "phase",
            "teacher_action",
            "student_action",
            "rollout_action",
            "used_teacher_action",
            "reward",
            "traj_no",
            "subtraj_step_no",
        ):
            if field in dataset.arrays:
                split_payload["field_stats"][field] = _stats(dataset.arrays[field])
        payload["splits"][split] = split_payload
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect distillation NPZ shard dataset.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--shard_level", action="store_true", default=False)
    args = parser.parse_args()

    payload = inspect_distill_dataset(args.dataset_dir, shard_level=bool(args.shard_level))
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
