"""Build strict primitive-synergy train/validation shards from raw rollouts.

Each raw NPZ represents one complete trial.  It must contain the physical
MuJoCo control that was actually applied and an integer, task-specific event
phase.  Signed normalized policy actions and kinematics-only motion files are
not muscle-excitation evidence and are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import (
    physical_ctrl_to_unit_excitation,
    physical_signal_metadata,
    validate_ordered_ctrlrange,
    validate_unit_muscle_activation,
)
from musclemimic.synergy.grouping import load_grouping_json
from musclemimic.synergy.primitive_catalog import (
    PrimitiveCatalog,
    PrimitiveTaskSpec,
    PrimitiveTrialSpec,
    canonical_json_sha256,
    load_primitive_catalog,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash

PRIMITIVE_INGEST_SCHEMA_VERSION = "primitive_synergy_ingest_v1"
PRIMITIVE_DATASET_METADATA_SCHEMA_VERSION = "primitive_synergy_rollout_dataset_v1"
PRIMITIVE_DATASET_QC_SCHEMA_VERSION = "primitive_synergy_dataset_qc_v1"

_CTRL_FIELDS = ("teacher_ctrl_physical", "applied_ctrl")
_NORMALIZED_ACTION_FIELDS = {
    "normalized_action",
    "policy_action_normalized",
    "teacher_action",
}
_OPTIONAL_SAMPLE_FIELDS = (
    "muscle_activation",
    "muscle_force",
    "muscle_tendon_length",
    "muscle_tendon_velocity",
    "phase_local",
)


@dataclass(frozen=True)
class PrimitiveIngestResult:
    """Paths and identity of one completed or idempotently reused dataset."""

    output_dir: Path
    metadata_path: Path
    source_checkpoints_path: Path
    dataset_qc_path: Path
    regional_grouping_path: Path
    build_fingerprint: str
    idempotent: bool


@dataclass(frozen=True)
class _ModelContract:
    model: mujoco.MjModel
    model_hash: str
    model_artifact_sha256: str


@dataclass(frozen=True)
class _RawTrial:
    task: PrimitiveTaskSpec
    trial: PrimitiveTrialSpec
    arrays: dict[str, np.ndarray]
    actuator_names: tuple[str, ...]
    actuator_ctrlrange: np.ndarray
    raw_sha256: str


def ingest_primitive_catalog(
    catalog_path: str | Path,
    output_dir: str | Path,
) -> PrimitiveIngestResult:
    """Validate raw trials and atomically materialize fit-ready NPZ shards.

    Existing output is reused only if its build fingerprint matches every
    current catalog/model/controller/phase/raw input and every recorded output
    file still matches its SHA-256.  Otherwise the function refuses to mutate
    the directory.
    """

    catalog = load_primitive_catalog(catalog_path, require_build_ready=True)
    assert catalog.model_artifact_path is not None
    assert catalog.regional_grouping_path is not None
    model_contract = _load_model_contract(catalog.model_artifact_path)
    controllers = {
        task.task_id: fingerprint_controller_artifact(task.controller_artifact) for task in catalog.enabled_tasks
    }
    checkpoint_fingerprints = {task_id: str(record["sha256"]) for task_id, record in controllers.items()}
    phase_fingerprints = {task.task_id: task.phase_schema.fingerprint for task in catalog.enabled_tasks}
    raw_inventory = [
        {
            "task_id": task.task_id,
            "trial_id": trial.trial_id,
            "split": trial.split,
            "motion_path": trial.motion_path,
            "motion_uid": trial.motion_uid,
            "path": str(trial.raw_npz_path),
            "sha256": file_sha256(trial.raw_npz_path),
        }
        for task in catalog.enabled_tasks
        for trial in task.trials
    ]
    build_contract = {
        "schema_version": PRIMITIVE_INGEST_SCHEMA_VERSION,
        "catalog_fingerprint": catalog.fingerprint,
        "model_hash": model_contract.model_hash,
        "model_artifact_sha256": model_contract.model_artifact_sha256,
        "regional_grouping_source_sha256": file_sha256(catalog.regional_grouping_path),
        "controller_fingerprints": checkpoint_fingerprints,
        "phase_schema_fingerprints": phase_fingerprints,
        "raw_trials": raw_inventory,
    }
    build_fingerprint = canonical_json_sha256(build_contract)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return _reuse_idempotent_output(output, expected_build_fingerprint=build_fingerprint)

    loaded_trials: list[_RawTrial] = []
    reference_names: tuple[str, ...] | None = None
    reference_ctrlrange: np.ndarray | None = None
    optional_fields: set[str] | None = None
    for task in catalog.enabled_tasks:
        for trial in task.trials:
            loaded = _load_raw_trial(
                catalog=catalog,
                model_contract=model_contract,
                task=task,
                trial=trial,
            )
            if reference_names is None:
                reference_names = loaded.actuator_names
                reference_ctrlrange = loaded.actuator_ctrlrange
                optional_fields = set(_OPTIONAL_SAMPLE_FIELDS) & set(loaded.arrays)
            elif loaded.actuator_names != reference_names or not np.array_equal(
                loaded.actuator_ctrlrange,
                reference_ctrlrange,
            ):
                raise ValueError("primitive raw trials do not share one exact actuator/ctrlrange contract")
            elif (set(_OPTIONAL_SAMPLE_FIELDS) & set(loaded.arrays)) != optional_fields:
                raise ValueError("optional primitive sample fields must be present in every trial or none")
            loaded_trials.append(loaded)
    if reference_names is None or reference_ctrlrange is None or optional_fields is None:
        raise ValueError("primitive ingest resolved no raw trials")

    actuator_hash = actuator_schema_hash(reference_names)
    # Validate the checked-in exact indexed grouping against the final raw
    # actuator ABI before copying it into the immutable dataset.
    regional_groups = load_grouping_json(
        catalog.regional_grouping_path,
        muscle_names=reference_names,
        require_complete=True,
    )
    source_ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": list(reference_names),
            "ctrlrange": reference_ctrlrange.tolist(),
        },
    )
    transform_ctrlrange_hash = ctrlrange_schema_hash(
        reference_names,
        reference_ctrlrange,
    )
    metadata = _build_metadata(
        catalog=catalog,
        model_contract=model_contract,
        loaded_trials=loaded_trials,
        actuator_names=reference_names,
        actuator_ctrlrange=reference_ctrlrange,
        actuator_hash=actuator_hash,
        source_ctrlrange_hash=source_ctrlrange_hash,
        transform_ctrlrange_hash=transform_ctrlrange_hash,
        controllers=controllers,
        build_contract=build_contract,
        build_fingerprint=build_fingerprint,
        optional_fields=optional_fields,
        regional_groups=regional_groups,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        shard_paths = _write_shards(temporary, loaded_trials)
        metadata_path = temporary / "metadata.json"
        source_checkpoints_path = temporary / "source_checkpoints.json"
        regional_grouping_path = temporary / "regional_grouping.json"
        _write_json(metadata_path, metadata)
        _write_json(source_checkpoints_path, checkpoint_fingerprints)
        shutil.copyfile(catalog.regional_grouping_path, regional_grouping_path)
        output_inventory = {
            path.relative_to(temporary).as_posix(): file_sha256(path)
            for path in sorted(
                [
                    *shard_paths,
                    metadata_path,
                    source_checkpoints_path,
                    regional_grouping_path,
                ],
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
        }
        qc = _build_dataset_qc(
            catalog=catalog,
            loaded_trials=loaded_trials,
            build_contract=build_contract,
            build_fingerprint=build_fingerprint,
            model_contract=model_contract,
            actuator_hash=actuator_hash,
            source_ctrlrange_hash=source_ctrlrange_hash,
            checkpoint_fingerprints=checkpoint_fingerprints,
            phase_fingerprints=phase_fingerprints,
            output_inventory=output_inventory,
        )
        _write_json(temporary / "dataset_qc.json", qc)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PrimitiveIngestResult(
        output_dir=output,
        metadata_path=output / "metadata.json",
        source_checkpoints_path=output / "source_checkpoints.json",
        dataset_qc_path=output / "dataset_qc.json",
        regional_grouping_path=output / "regional_grouping.json",
        build_fingerprint=build_fingerprint,
        idempotent=False,
    )


def fingerprint_controller_artifact(path: str | Path | None) -> dict[str, Any]:
    """Hash the current bytes of one controller file or directory."""

    if path is None:
        raise ValueError("controller artifact path is required")
    supplied = str(path)
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        root = resolved
        files = sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    elif resolved.is_file():
        root = resolved.parent
        files = [resolved]
    else:
        raise FileNotFoundError(f"controller artifact does not exist: {resolved}")
    if not files:
        raise ValueError(f"controller artifact contains no files: {resolved}")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        payload = item.read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        total_bytes += len(payload)
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "num_bytes": len(payload),
            }
        )
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": supplied,
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "num_files": len(records),
        "num_bytes": int(total_bytes),
        "files": records,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model_contract(model_artifact_path: Path) -> _ModelContract:
    suffix = model_artifact_path.suffix.casefold()
    if suffix == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(model_artifact_path))
    elif suffix in {".xml", ".mjcf"}:
        model = mujoco.MjModel.from_xml_path(str(model_artifact_path))
    else:
        raise ValueError(f"primitive compiled model artifact must use .mjb, .xml, or .mjcf: {model_artifact_path}")
    state = model.__getstate__()
    if not isinstance(state, bytes) or not state:
        raise ValueError("MuJoCo model has no canonical complete byte state")
    return _ModelContract(
        model=model,
        model_hash=hashlib.sha256(state).hexdigest(),
        model_artifact_sha256=file_sha256(model_artifact_path),
    )


def save_compiled_model_artifact(
    model: mujoco.MjModel,
    output_mjb: str | Path,
) -> Path:
    """Atomically save and verify the exact runtime ``MjModel`` as MJB.

    Existing output is reused only when loading it reproduces the complete
    ``MjModel.__getstate__`` hash.  A different existing model is never
    overwritten.
    """

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("save_compiled_model_artifact requires mujoco.MjModel")
    output = Path(output_mjb).expanduser().resolve()
    if output.suffix.casefold() != ".mjb":
        raise ValueError("compiled model artifact output must use the .mjb suffix")
    state = model.__getstate__()
    if not isinstance(state, bytes) or not state:
        raise ValueError("MuJoCo model has no canonical complete byte state")
    expected_hash = hashlib.sha256(state).hexdigest()
    if output.exists():
        if not output.is_file():
            raise FileExistsError(f"compiled model artifact exists and is not a file: {output}")
        existing = mujoco.MjModel.from_binary_path(str(output))
        existing_state = existing.__getstate__()
        if not isinstance(existing_state, bytes) or hashlib.sha256(existing_state).hexdigest() != expected_hash:
            raise FileExistsError("compiled model artifact already exists with different model content")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{uuid.uuid4().hex}.mjb")
    try:
        mujoco.mj_saveModel(model, str(temporary))
        restored = mujoco.MjModel.from_binary_path(str(temporary))
        restored_state = restored.__getstate__()
        if not isinstance(restored_state, bytes) or hashlib.sha256(restored_state).hexdigest() != expected_hash:
            raise ValueError("saved MJB does not reproduce the complete runtime model hash")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _load_raw_trial(
    *,
    catalog: PrimitiveCatalog,
    model_contract: _ModelContract,
    task: PrimitiveTaskSpec,
    trial: PrimitiveTrialSpec,
) -> _RawTrial:
    with np.load(trial.raw_npz_path, allow_pickle=False) as source:
        fields = set(source.files)
        present_ctrl = [field for field in _CTRL_FIELDS if field in source]
        if not present_ctrl:
            normalized = sorted(fields & _NORMALIZED_ACTION_FIELDS)
            if normalized:
                raise ValueError(
                    f"raw trial {trial.trial_id!r} contains only normalized action "
                    f"fields {normalized}; actual teacher_ctrl_physical/applied_ctrl is required"
                )
            raise ValueError(
                f"raw trial {trial.trial_id!r} is kinematics-only or lacks physical control; "
                "qpos/qvel cannot be relabeled as muscle excitation"
            )
        if "phase_id" not in source:
            raise ValueError(f"raw trial {trial.trial_id!r} lacks integer phase_id")
        if "actuator_names" not in source:
            raise ValueError(f"raw trial {trial.trial_id!r} lacks ordered actuator_names")
        names_array = np.asarray(source["actuator_names"])
        if names_array.ndim != 1 or names_array.dtype.kind not in {"U", "S"}:
            raise ValueError("raw actuator_names must be a rank-1 non-object string array")
        names = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in names_array.tolist()
        )
        if len(names) != catalog.expected_action_dim or len(set(names)) != len(names):
            raise ValueError(
                f"raw actuator_names must be unique and match expected_action_dim ({catalog.expected_action_dim})"
            )
        ctrl = np.asarray(source[present_ctrl[0]], dtype=np.float64)
        if ctrl.ndim != 2 or ctrl.shape[1] != len(names) or ctrl.shape[0] <= 0:
            raise ValueError(f"raw physical ctrl must have shape [T,{len(names)}], got {ctrl.shape}")
        if not np.all(np.isfinite(ctrl)):
            raise ValueError("raw physical ctrl contains non-finite values")
        for alias in present_ctrl[1:]:
            other = np.asarray(source[alias], dtype=np.float64)
            if other.shape != ctrl.shape or not np.allclose(
                other,
                ctrl,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError("teacher_ctrl_physical and applied_ctrl differ in the same raw trial")
        phase_id = np.asarray(source["phase_id"])
        if (
            phase_id.shape != (ctrl.shape[0],)
            or np.issubdtype(phase_id.dtype, np.bool_)
            or not np.issubdtype(phase_id.dtype, np.integer)
        ):
            raise ValueError("raw phase_id must be an integer vector with shape [T]")
        if np.any(phase_id < 0) or np.any(phase_id > np.iinfo(np.int32).max):
            raise ValueError("raw phase_id must contain non-negative int32 values")
        phase_id = phase_id.astype(np.int32)
        observed_phases = {int(value) for value in np.unique(phase_id).tolist()}
        required_phases = set(task.phase_schema.required_phase_ids)
        unknown = sorted(observed_phases - required_phases)
        missing = sorted(required_phases - observed_phases)
        if unknown:
            raise ValueError(f"raw trial {trial.trial_id!r} contains phase ids absent from its schema: {unknown}")
        if missing:
            raise ValueError(f"raw trial {trial.trial_id!r} misses required phase ids: {missing}")
        underpopulated = {
            phase: int(np.count_nonzero(phase_id == phase))
            for phase in sorted(required_phases)
            if int(np.count_nonzero(phase_id == phase)) < 2
        }
        if underpopulated:
            raise ValueError(
                "each primitive trial phase requires at least two samples for held-out "
                f"cell metrics; trial={trial.trial_id!r} counts={underpopulated}"
            )

        actuator_ids: list[int] = []
        for name in names:
            actuator_id = mujoco.mj_name2id(
                model_contract.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            if actuator_id < 0:
                raise ValueError(f"raw actuator {name!r} is absent from the catalog MuJoCo model")
            actuator_ids.append(int(actuator_id))
        ctrlrange = validate_ordered_ctrlrange(
            names,
            model_contract.model.actuator_ctrlrange[np.asarray(actuator_ids, dtype=np.int32)],
        )
        excitation = physical_ctrl_to_unit_excitation(ctrl, ctrlrange)
        if "muscle_excitation" in source:
            stored_excitation = np.asarray(source["muscle_excitation"], dtype=np.float64)
            if stored_excitation.shape != excitation.shape or not np.allclose(
                stored_excitation,
                excitation,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("raw muscle_excitation differs from the model-derived ctrlrange transform")
        _validate_optional_embedded_contracts(
            source,
            names=names,
            ctrlrange=ctrlrange,
            model_hash=model_contract.model_hash,
        )
        _validate_raw_success(source, trial=trial, sample_count=ctrl.shape[0])
        arrays: dict[str, np.ndarray] = {
            "teacher_ctrl_physical": ctrl.astype(np.float32),
            "muscle_excitation": excitation,
            "phase_id": phase_id,
        }
        for field in _OPTIONAL_SAMPLE_FIELDS:
            if field not in source:
                continue
            value = np.asarray(source[field])
            if field == "phase_local":
                value = np.asarray(value, dtype=np.float64)
                if value.shape != (ctrl.shape[0],) or not np.all(np.isfinite(value)):
                    raise ValueError("raw phase_local must be a finite vector with shape [T]")
                arrays[field] = value.astype(np.float32)
                continue
            if value.shape != ctrl.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"raw {field} must be finite and have shape {ctrl.shape}")
            arrays[field] = (
                validate_unit_muscle_activation(value)
                if field == "muscle_activation"
                else np.asarray(value, dtype=np.float32)
            )
    return _RawTrial(
        task=task,
        trial=trial,
        arrays=arrays,
        actuator_names=names,
        actuator_ctrlrange=ctrlrange,
        raw_sha256=file_sha256(trial.raw_npz_path),
    )


def _validate_optional_embedded_contracts(
    source: Any,
    *,
    names: tuple[str, ...],
    ctrlrange: np.ndarray,
    model_hash: str,
) -> None:
    expected_actuator_hash = actuator_schema_hash(names)
    expected_ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={"actuator_names": list(names), "ctrlrange": ctrlrange.tolist()},
    )
    if "actuator_ctrlrange" in source and not np.array_equal(
        np.asarray(source["actuator_ctrlrange"], dtype=np.float64),
        ctrlrange,
    ):
        raise ValueError("raw embedded actuator_ctrlrange differs from the live MuJoCo model")
    for field, expected in (
        ("model_hash", model_hash),
        ("actuator_schema_hash", expected_actuator_hash),
        ("ctrlrange_schema_hash", expected_ctrlrange_hash),
    ):
        if field in source and _scalar_string(source[field], field) != expected:
            raise ValueError(f"raw embedded {field} differs from the live derived contract")


def _validate_raw_success(
    source: Any,
    *,
    trial: PrimitiveTrialSpec,
    sample_count: int,
) -> None:
    if not trial.success:
        raise ValueError(f"raw primitive trial {trial.trial_id!r} is not declared successful")
    if "success" not in source:
        return
    success = np.asarray(source["success"])
    if success.ndim == 0:
        success = np.full(sample_count, success.item())
    if success.shape != (sample_count,) or not np.all(np.asarray(success, dtype=np.float64) == 1.0):
        raise ValueError(f"raw primitive trial {trial.trial_id!r} contains unsuccessful samples")


def _scalar_string(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"raw embedded {field} must be a scalar string")
    return str(array.tolist())


def _build_metadata(
    *,
    catalog: PrimitiveCatalog,
    model_contract: _ModelContract,
    loaded_trials: Sequence[_RawTrial],
    actuator_names: tuple[str, ...],
    actuator_ctrlrange: np.ndarray,
    actuator_hash: str,
    source_ctrlrange_hash: str,
    transform_ctrlrange_hash: str,
    controllers: Mapping[str, Mapping[str, Any]],
    build_contract: Mapping[str, Any],
    build_fingerprint: str,
    optional_fields: set[str],
    regional_groups: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    tasks = catalog.enabled_tasks
    metadata: dict[str, Any] = {
        "schema_version": PRIMITIVE_DATASET_METADATA_SCHEMA_VERSION,
        "collector": "primitive_catalog_raw_physical_ingest",
        "catalog_id": catalog.catalog_id,
        "catalog_path": str(catalog.path),
        "catalog_fingerprint": catalog.fingerprint,
        "ingest_contract": dict(build_contract),
        "ingest_build_fingerprint": build_fingerprint,
        "actuator_names": list(actuator_names),
        "actuator_ctrlrange": actuator_ctrlrange.tolist(),
        "action_schema_hash": actuator_hash,
        "ctrlrange_schema_hash": source_ctrlrange_hash,
        "transform_ctrlrange_schema_hash": transform_ctrlrange_hash,
        "physical_signal_semantics": physical_signal_metadata(),
        "model_artifact_path": str(catalog.model_artifact_path),
        "model_artifact_format": catalog.model_artifact_path.suffix.casefold(),
        "model_artifact_sha256": model_contract.model_artifact_sha256,
        "model_hash": model_contract.model_hash,
        "regional_grouping_source": str(catalog.regional_grouping_path),
        "regional_grouping_source_sha256": file_sha256(catalog.regional_grouping_path),
        "regional_grouping_output": "regional_grouping.json",
        "regional_group_names": list(regional_groups),
        "source_checkpoint_fingerprints": {task.task_id: str(controllers[task.task_id]["sha256"]) for task in tasks},
        "source_checkpoint_contents": {task.task_id: dict(controllers[task.task_id]) for task in tasks},
        "primitive_required_phase_ids": {task.task_id: list(task.phase_schema.required_phase_ids) for task in tasks},
        "primitive_phase_schema_fingerprints": {task.task_id: task.phase_schema.fingerprint for task in tasks},
        "primitive_phase_schemas": {task.task_id: task.phase_schema.payload for task in tasks},
        "raw_trial_sources": [
            {
                "task_id": loaded.task.task_id,
                "trial_id": loaded.trial.trial_id,
                "split": loaded.trial.split,
                "motion_path": loaded.trial.motion_path,
                "motion_uid": loaded.trial.motion_uid,
                "path": str(loaded.trial.raw_npz_path),
                "sha256": loaded.raw_sha256,
            }
            for loaded in loaded_trials
        ],
        "fields": sorted(
            {
                "teacher_ctrl_physical",
                "muscle_excitation",
                "phase_id",
                "motion_uid",
                "task_id",
                "trial_id",
                "source_kind",
                "success",
                "quality_weight",
                *optional_fields,
            }
        ),
        "num_samples": int(sum(item.arrays["phase_id"].shape[0] for item in loaded_trials)),
    }
    if "muscle_activation" in optional_fields:
        actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model_contract.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                for name in actuator_names
            ],
            dtype=np.int32,
        )
        valid = (model_contract.model.actuator_actadr[actuator_ids] >= 0) & (
            model_contract.model.actuator_actnum[actuator_ids] == 1
        )
        if not np.all(valid):
            invalid = [actuator_names[int(index)] for index in np.flatnonzero(~valid).tolist()]
            raise ValueError(
                f"raw muscle_activation is present for actuators without one scalar MuJoCo activation state: {invalid}"
            )
        metadata["physical_capture"] = {
            "schema_version": "physical_capture_spec_v1",
            "actuator_names": list(actuator_names),
            "model_nu": int(model_contract.model.nu),
            "model_nv": int(model_contract.model.nv),
            "model_na": int(model_contract.model.na),
            "activation_valid_mask": valid.tolist(),
            "racket_site_name": None,
        }
    return metadata


def _write_shards(root: Path, loaded_trials: Sequence[_RawTrial]) -> list[Path]:
    result: list[Path] = []
    counters = {"train": 0, "val": 0}
    ordered = sorted(
        loaded_trials,
        key=lambda item: (
            0 if item.trial.split == "train" else 1,
            item.task.task_id,
            item.trial.trial_id,
        ),
    )
    for loaded in ordered:
        sample_count = int(loaded.arrays["phase_id"].shape[0])
        arrays = {
            **loaded.arrays,
            "motion_uid": np.full(
                sample_count,
                loaded.trial.motion_uid,
                dtype=np.int64,
            ),
            "task_id": np.full(sample_count, loaded.task.task_id),
            "trial_id": np.full(sample_count, loaded.trial.trial_id),
            "source_kind": np.full(sample_count, "primitive"),
            "success": np.ones(sample_count, dtype=np.int8),
            "quality_weight": np.full(
                sample_count,
                loaded.trial.quality_weight,
                dtype=np.float32,
            ),
            "actuator_names": np.asarray(loaded.actuator_names),
            "actuator_ctrlrange": loaded.actuator_ctrlrange.astype(np.float64),
        }
        split = loaded.trial.split
        path = root / f"{split}_{counters[split]:06d}.npz"
        np.savez_compressed(path, **arrays)
        result.append(path)
        counters[split] += 1
    return result


def _build_dataset_qc(
    *,
    catalog: PrimitiveCatalog,
    loaded_trials: Sequence[_RawTrial],
    build_contract: Mapping[str, Any],
    build_fingerprint: str,
    model_contract: _ModelContract,
    actuator_hash: str,
    source_ctrlrange_hash: str,
    checkpoint_fingerprints: Mapping[str, str],
    phase_fingerprints: Mapping[str, str],
    output_inventory: Mapping[str, str],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split in ("train", "val"):
        selected = [item for item in loaded_trials if item.trial.split == split]
        task_phase_samples: dict[str, dict[str, int]] = {}
        task_trial_counts: dict[str, int] = {}
        for item in selected:
            task_trial_counts[item.task.task_id] = task_trial_counts.get(item.task.task_id, 0) + 1
            phase_counts = task_phase_samples.setdefault(item.task.task_id, {})
            phases = item.arrays["phase_id"]
            for phase in np.unique(phases):
                key = str(int(phase))
                phase_counts[key] = phase_counts.get(key, 0) + int(np.count_nonzero(phases == phase))
        split_summary[split] = {
            "num_trials": len(selected),
            "num_samples": int(sum(item.arrays["phase_id"].shape[0] for item in selected)),
            "task_trial_counts": task_trial_counts,
            "task_phase_sample_counts": task_phase_samples,
            "motion_uids": sorted({item.trial.motion_uid for item in selected}),
            "trial_ids": sorted(item.trial.trial_id for item in selected),
        }
    return {
        "schema_version": PRIMITIVE_DATASET_QC_SCHEMA_VERSION,
        "passed": True,
        "failed": [],
        "build_fingerprint": build_fingerprint,
        "build_contract": dict(build_contract),
        "catalog_fingerprint": catalog.fingerprint,
        "model_hash": model_contract.model_hash,
        "model_artifact_sha256": model_contract.model_artifact_sha256,
        "actuator_schema_hash": actuator_hash,
        "ctrlrange_schema_hash": source_ctrlrange_hash,
        "source_checkpoint_fingerprints": dict(checkpoint_fingerprints),
        "primitive_phase_schema_fingerprints": dict(phase_fingerprints),
        "split_summary": split_summary,
        "output_inventory": dict(output_inventory),
    }


def _reuse_idempotent_output(
    output: Path,
    *,
    expected_build_fingerprint: str,
) -> PrimitiveIngestResult:
    if not output.is_dir():
        raise FileExistsError(f"primitive ingest output exists and is not a directory: {output}")
    qc_path = output / "dataset_qc.json"
    if not qc_path.is_file():
        raise FileExistsError("primitive ingest output already exists without dataset_qc.json; refusing overwrite")
    qc = load_json_strict(qc_path)
    if not isinstance(qc, Mapping):
        raise ValueError("existing primitive dataset_qc.json must contain an object")
    if (
        qc.get("schema_version") != PRIMITIVE_DATASET_QC_SCHEMA_VERSION
        or qc.get("passed") is not True
        or qc.get("failed") != []
    ):
        raise ValueError("existing primitive dataset QC is not a passed v1 contract")
    if qc.get("build_fingerprint") != expected_build_fingerprint:
        raise FileExistsError("primitive ingest output was built from different current inputs; refusing overwrite")
    inventory = qc.get("output_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("existing primitive dataset QC lacks output_inventory")
    for relative, expected_hash in inventory.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("existing primitive output_inventory is malformed")
        path = output / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"existing primitive dataset output changed after QC: {relative}")
    return PrimitiveIngestResult(
        output_dir=output,
        metadata_path=output / "metadata.json",
        source_checkpoints_path=output / "source_checkpoints.json",
        dataset_qc_path=qc_path,
        regional_grouping_path=output / "regional_grouping.json",
        build_fingerprint=expected_build_fingerprint,
        idempotent=True,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = ingest_primitive_catalog(args.catalog, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "metadata": str(result.metadata_path),
                "source_checkpoints": str(result.source_checkpoints_path),
                "dataset_qc": str(result.dataset_qc_path),
                "regional_grouping": str(result.regional_grouping_path),
                "build_fingerprint": result.build_fingerprint,
                "idempotent": result.idempotent,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
