"""Checkpoint IO for latent posterior/prior/decoder training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from flax import serialization

from musclemimic.latent_muscle.action_mask import ActionMask


def save_latent_checkpoint(
    checkpoint_dir: str | Path,
    *,
    encoder_variables: Any,
    prior_variables: Any,
    decoder_variables: Any,
    optimizer_state: Any,
    action_mask: ActionMask,
    config: dict[str, Any],
    train_metrics: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
    obs_norm: dict[str, Any] | None = None,
    action_norm: dict[str, Any] | None = None,
) -> Path:
    """Persist a complete latent distillation checkpoint directory."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write_msgpack(path / "encoder.msgpack", encoder_variables)
    _write_msgpack(path / "prior.msgpack", prior_variables)
    _write_msgpack(path / "decoder.msgpack", decoder_variables)
    _write_msgpack(path / "optimizer_state.msgpack", optimizer_state)
    (path / "action_mask.json").write_text(
        json.dumps(_action_mask_manifest(action_mask), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "latent_config.yaml").write_text(
        json.dumps(_jsonable(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if obs_norm is not None:
        (path / "obs_norm.json").write_text(json.dumps(_jsonable(obs_norm), indent=2, sort_keys=True), encoding="utf-8")
    if action_norm is not None:
        (path / "action_norm.json").write_text(
            json.dumps(_jsonable(action_norm), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _write_metrics_csv(path / "train_metrics.csv", train_metrics)
    (path / "eval_metrics.json").write_text(
        json.dumps(_jsonable(eval_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_latent_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load a latent checkpoint saved by :func:`save_latent_checkpoint`."""
    path = Path(checkpoint_dir)
    return {
        "encoder_variables": _read_msgpack(path / "encoder.msgpack"),
        "prior_variables": _read_msgpack(path / "prior.msgpack"),
        "decoder_variables": _read_msgpack(path / "decoder.msgpack"),
        "optimizer_state": _read_msgpack(path / "optimizer_state.msgpack"),
        "action_mask": json.loads((path / "action_mask.json").read_text(encoding="utf-8")),
        "config": json.loads((path / "latent_config.yaml").read_text(encoding="utf-8")),
        "eval_metrics": json.loads((path / "eval_metrics.json").read_text(encoding="utf-8")),
    }


def _write_msgpack(path: Path, value: Any) -> None:
    path.write_bytes(serialization.msgpack_serialize(serialization.to_state_dict(value)))


def _read_msgpack(path: Path) -> Any:
    return serialization.msgpack_restore(path.read_bytes())


def _action_mask_manifest(mask: ActionMask) -> dict[str, Any]:
    return {
        "all_actuator_names": list(mask.all_actuator_names),
        "body_actuator_names": list(mask.body_actuator_names),
        "correction_actuator_names": list(mask.correction_actuator_names),
        "body_indices": mask.body_indices.tolist(),
        "correction_indices": mask.correction_indices.tolist(),
        "decoder_action_dim": int(mask.body_size),
        "full_action_dim": int(mask.action_size),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
