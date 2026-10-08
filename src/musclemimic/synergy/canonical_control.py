"""Train-only, content-addressed canonical tonic muscle control artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.synergy.primitive_catalog import canonical_json_sha256, load_primitive_catalog

SCHEMA_VERSION = "primitive_canonical_tonic_control_v1"


def _publish_content_addressed_directory(temporary: Path, final: Path, validate: Any) -> None:
    """Publish one immutable directory; concurrent losers validate the winner."""

    try:
        os.rename(temporary, final)
    except OSError:
        if not final.exists():
            raise
        validate(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(4, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_canonical_control_artifact(
    path: str | Path, *, expected_width: int = 354, require_path_binding: bool = True
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "canonical_control.json"
    payload = load_json_strict(source)
    if not isinstance(payload, Mapping):
        raise ValueError("canonical control artifact must contain an object")
    result = dict(payload)
    fingerprint = result.pop("artifact_fingerprint", None)
    expected_fields = {
        "schema_version",
        "task_id",
        "catalog_fingerprint",
        "controller_fingerprint",
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
        "action_dim",
        "aggregation",
        "train_trials",
        "control",
        "control_sha256",
    }
    if set(result) != expected_fields:
        raise ValueError("canonical control artifact fields differ from schema")
    if canonical_json_sha256(result) != fingerprint or (require_path_binding and source.parent.name != fingerprint):
        raise ValueError("canonical control artifact/path fingerprint mismatch")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("task_id") != "P01_natural_stance":
        raise ValueError("canonical control artifact schema/task is unsupported")
    if result.get("aggregation") != "coordinate_mean_float64_train_only_v1":
        raise ValueError("canonical control artifact aggregation contract is unsupported")
    trials = result.get("train_trials")
    if (
        not isinstance(trials, list)
        or not trials
        or any(not isinstance(item, Mapping) or item.get("split") != "train" for item in trials)
    ):
        raise ValueError("canonical control artifact requires non-empty train-only provenance")
    trial_fields = {
        "trial_id",
        "split",
        "motion_uid",
        "source_motion_path",
        "source_sha256",
        "source_frame_interval",
        "rollout_manifest_sha256",
        "rollout_qc_sha256",
        "initial_ctrl_sha256",
    }
    trial_ids: set[str] = set()
    motion_uids: set[int] = set()
    for item in trials:
        if set(item) != trial_fields:
            raise ValueError("canonical control train-trial fields differ from schema")
        if not isinstance(item["trial_id"], str) or not item["trial_id"]:
            raise ValueError("canonical control trial_id must be non-empty")
        if type(item["motion_uid"]) is not int:
            raise ValueError("canonical control motion_uid must be an integer")
        if not isinstance(item["source_motion_path"], str) or not item["source_motion_path"]:
            raise ValueError("canonical control source_motion_path must be non-empty")
        interval = item["source_frame_interval"]
        if not isinstance(interval, Mapping) or set(interval) != {
            "start_frame",
            "end_frame_exclusive",
            "source_total_frames",
        }:
            raise ValueError("canonical control source frame interval is malformed")
        start, end, total = (interval[key] for key in ("start_frame", "end_frame_exclusive", "source_total_frames"))
        if any(type(value) is not int for value in (start, end, total)) or start < 0 or end <= start or total < end:
            raise ValueError("canonical control source frame interval is invalid")
        if item["trial_id"] in trial_ids or item["motion_uid"] in motion_uids:
            raise ValueError("canonical control train provenance contains duplicates")
        trial_ids.add(item["trial_id"])
        motion_uids.add(item["motion_uid"])
        for field in ("source_sha256", "rollout_manifest_sha256", "rollout_qc_sha256", "initial_ctrl_sha256"):
            value = item[field]
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"canonical control train-trial {field} is not SHA-256")
    for field in (
        "catalog_fingerprint",
        "controller_fingerprint",
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
    ):
        value = result.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"canonical control {field} is not SHA-256")
    control = np.asarray(result.get("control"), dtype=np.float64)
    if (
        result.get("action_dim") != expected_width
        or control.shape != (expected_width,)
        or not np.all(np.isfinite(control))
        or np.any(control < 0)
        or np.any(control > 1)
    ):
        raise ValueError("canonical control vector is malformed or outside [0,1]")
    if result.get("control_sha256") != _array_hash(control):
        raise ValueError("canonical control vector hash mismatch")
    result["artifact_fingerprint"] = fingerprint
    result["path"] = str(source)
    return result


def build_canonical_control_artifact(catalog_path: str | Path, output_store: str | Path) -> Path:
    from musclemimic.synergy.primitive_ingest import (
        _load_controller_contract,
        _load_model_contract,
        _load_raw_trial,
    )

    catalog = load_primitive_catalog(catalog_path, require_build_ready=True)
    if catalog.model_artifact_path is None or catalog.model_artifact_path.suffix.casefold() != ".mjb":
        raise ValueError("canonical control catalog model must be an exact MJB artifact")
    try:
        model_contract = _load_model_contract(catalog.model_artifact_path)
    except Exception as exc:
        # MuJoCo's invalid-model exception type depends on the process-wide
        # warning callback installed by other imported modules.  Keep this
        # public builder contract deterministic and fail closed either way.
        raise ValueError(f"canonical control model failed to load: {catalog.model_artifact_path}") from exc
    tasks = [task for task in catalog.enabled_tasks if task.task_id == "P01_natural_stance"]
    if len(tasks) != 1:
        raise ValueError("catalog must contain exactly one enabled P01_natural_stance task")
    task = tasks[0]
    controller_contract = _load_controller_contract(task, model_contract=model_contract)
    train = [trial for trial in task.trials if trial.split == "train"]
    if not train:
        raise ValueError("canonical control requires at least one P01 train trial")
    rows: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    controller: str | None = None
    model_hash: str | None = None
    actuator_hash: str | None = None
    ctrlrange_hash: str | None = None
    for trial in train:
        loaded = _load_raw_trial(
            catalog=catalog,
            model_contract=model_contract,
            controller_contract=controller_contract,
            task=task,
            trial=trial,
        )
        manifest_path = loaded.rollout_manifest_path
        manifest = load_json_strict(manifest_path)
        artifact = manifest.get("artifacts", {}).get("rollout_qc", {})
        qc_path = manifest_path.with_name(str(artifact.get("filename", "")))
        if not qc_path.is_file() or file_sha256(qc_path) != artifact.get("sha256"):
            raise ValueError("rollout QC artifact hash mismatch")
        identities = (
            loaded.optimizer_fingerprint,
            controller_contract.manifest.get("model_hash"),
            controller_contract.manifest.get("actuator_schema_hash"),
            controller_contract.manifest.get("ctrlrange_schema_hash"),
        )
        if controller is None:
            controller, model_hash, actuator_hash, ctrlrange_hash = identities
        elif identities != (controller, model_hash, actuator_hash, ctrlrange_hash):
            raise ValueError("P01 train trials do not share one controller/model/action ABI")
        with np.load(qc_path, allow_pickle=False) as arrays:
            control = np.asarray(arrays["initial_ctrl"])
        if control.shape != (catalog.expected_action_dim,) or np.any(control < 0) or np.any(control > 1):
            raise ValueError("rollout initial_ctrl is malformed")
        expected_initial_hash = manifest.get("initialization_contract", {}).get("initial_ctrl_sha256")
        if _array_hash(control) != expected_initial_hash:
            raise ValueError("rollout initial_ctrl differs from producer manifest hash")
        rows.append(control)
        interval = manifest.get("source_frame_interval")
        evidence.append(
            {
                "trial_id": trial.trial_id,
                "split": "train",
                "motion_uid": trial.motion_uid,
                "source_motion_path": trial.motion_path,
                "source_sha256": manifest.get("source_artifact_sha256"),
                "source_frame_interval": interval,
                "rollout_manifest_sha256": file_sha256(manifest_path),
                "rollout_qc_sha256": file_sha256(qc_path),
                "initial_ctrl_sha256": expected_initial_hash,
            }
        )
    control = np.mean(np.stack(rows).astype(np.float64), axis=0, dtype=np.float64)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "catalog_fingerprint": catalog.fingerprint,
        "controller_fingerprint": controller,
        "model_hash": model_hash,
        "actuator_schema_hash": actuator_hash,
        "ctrlrange_schema_hash": ctrlrange_hash,
        "action_dim": catalog.expected_action_dim,
        "aggregation": "coordinate_mean_float64_train_only_v1",
        "train_trials": evidence,
        "control": control.tolist(),
        "control_sha256": _array_hash(control),
    }
    fingerprint = canonical_json_sha256(payload)
    payload["artifact_fingerprint"] = fingerprint
    store = Path(output_store).expanduser().resolve()
    final = store / fingerprint
    if final.exists():
        load_canonical_control_artifact(final, expected_width=catalog.expected_action_dim)
        return final
    store.mkdir(parents=True, exist_ok=True)
    temporary = store / f".tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    (temporary / "canonical_control.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _publish_content_addressed_directory(
        temporary,
        final,
        lambda path: load_canonical_control_artifact(path, expected_width=catalog.expected_action_dim),
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_canonical_control_artifact(args.catalog, args.output_store))
    return 0
