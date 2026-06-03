"""Inspect distillation dataset shards before BC/KD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.dataset import DistillDataset, load_metadata


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


def inspect_distill_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Return split and field diagnostics for a distillation dataset directory."""
    dataset_path = Path(dataset_dir)
    metadata = load_metadata(dataset_path)
    payload: dict[str, Any] = {
        "dataset_dir": str(dataset_path),
        "metadata": metadata,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        try:
            dataset = DistillDataset(dataset_path, split=split)
        except FileNotFoundError:
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
    args = parser.parse_args()

    payload = inspect_distill_dataset(args.dataset_dir)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
