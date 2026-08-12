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
from musclemimic.distill.motion_identity import normalize_relative_motion_path, stable_motion_uid
from musclemimic.distill.physical import (
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    physical_ctrl_to_effective_muscle_excitation,
    physical_signal_metadata,
    resolve_muscle_channel_contract,
    validate_muscle_channel_contract,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.synergy.canonical_control import load_canonical_control_artifact
from musclemimic.synergy.grouping import load_grouping_json
from musclemimic.synergy.primitive_catalog import (
    PrimitiveCatalog,
    PrimitiveTaskSpec,
    PrimitiveTrialSpec,
    canonical_json_sha256,
    load_primitive_catalog,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash
from musclemimic.synergy.semantic_contracts import (
    P12_MAX_POST_IMPACT_COM_VERTICAL_SPEED,
    P12_MAX_READY_HOLD_COM_VERTICAL_SPEED,
    P12_MIN_COM_VERTICAL_EXCURSION,
    P12_MIN_READY_HOLD_FRAMES,
    P12_REQUIRED_SEMANTIC_GATES,
    PRIMITIVE_SEMANTIC_ATTESTATION_SCHEMA_VERSION,
    primitive_semantic_contracts,
    validate_primitive_semantic_contracts,
)

PRIMITIVE_INGEST_SCHEMA_VERSION = "primitive_synergy_ingest_v3"
PRIMITIVE_DATASET_METADATA_SCHEMA_VERSION = "primitive_synergy_rollout_dataset_v3"
PRIMITIVE_DATASET_QC_SCHEMA_VERSION = "primitive_synergy_dataset_qc_v3"
ROLLOUT_MANIFEST_SCHEMA_VERSION = "primitive_physical_rollout_manifest_v7"
OPTIMIZER_MANIFEST_SCHEMA_VERSION = "primitive_physical_optimizer_manifest_v1"
AXIAL_ROTATION_CONTRACT_SCHEMA_VERSION = "p08_named_axial_rotation_signal_contract_v1"

_P12_COM_VERTICAL_SIGNAL = "root_subtree_com_z/delta_over_transition_duration"
_P12_RECOVERY_METRIC_NAMES = (
    "landing_stabilization_terminal_com_height",
    "posture_restore_terminal_com_height",
    "posture_restore_com_rise",
    "ready_hold_min_com_height",
    "ready_hold_min_height_above_landing_baseline",
)

_ROLLOUT_MANIFEST_FIELDS = {
    "schema_version",
    "trial_id",
    "task_id",
    "target_skill_id",
    "contains_target_skill_rollout",
    "source_motion_path",
    "source_motion_uid",
    "source_artifact_sha256",
    "source_frequency_hz",
    "source_frame_interval",
    "phase_schema_fingerprint",
    "contact_contract",
    "axial_rotation_signal_contract",
    "target_contact_semantics",
    "seed",
    "optimizer_fingerprint",
    "controller_source_kind",
    "runtime_model_binding",
    "runtime_model_provenance",
    "policy_rollout_binding",
    "optimizer_artifact",
    "initialization_contract",
    "transition_contract",
    "requested_transition_count",
    "recorded_transition_count",
    "qc_config",
    "qc",
    "artifacts",
    "status",
    "success",
    "production_eligible",
    "rollout_fingerprint",
}
_OPTIMIZER_MANIFEST_FIELDS = {
    "schema_version",
    "source_kind",
    "algorithm",
    "initial_state_contract",
    "control_coordinates",
    "normalized_action_accepted",
    "emg_used",
    "success_decision",
    "shooting_proposal_residual_can_mark_success",
    "optimizer_config",
    "policy_controller_binding",
    "production_eligible",
    "runtime_model_binding",
    "implementation_sha256",
    "model_hash",
    "model_artifact_filename",
    "model_artifact_sha256",
    "model_nq",
    "model_nv",
    "model_nu",
    "model_na",
    "physics_timestep",
    "actuator_names",
    "actuator_schema_hash",
    "ctrlrange",
    "ctrlrange_schema_hash",
    "transform_ctrlrange_schema_hash",
    "optimizer_fingerprint",
}
_ROLLOUT_QC_FIELDS = {
    "passed",
    "gates",
    "metrics",
    "recorded_transition_count",
    "expected_transition_count",
    "contact_contract",
    "axial_rotation_signal_contract",
    "target_contact_semantics",
    "actual_contact_semantics",
    "axial_rotation_direction_evidence",
    "initialization_evidence",
    "replay_contact_evidence",
    "replay_axial_rotation_evidence",
    "success_does_not_depend_on_shooting_proposal_residual",
}
_AXIAL_ROTATION_JOINT_NAMES = (
    "axial_rotation",
    "Abs_r3",
    "L4_L5_AR",
    "L3_L4_AR",
    "L2_L3_AR",
    "L1_L2_AR",
)

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
class _ControllerContract:
    path: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    optimizer_fingerprint: str
    model_artifact_path: Path
    model_artifact_sha256: str


@dataclass(frozen=True)
class _RawTrial:
    task: PrimitiveTaskSpec
    trial: PrimitiveTrialSpec
    arrays: dict[str, np.ndarray]
    actuator_names: tuple[str, ...]
    actuator_ctrlrange: np.ndarray
    raw_sha256: str
    rollout_manifest_path: Path
    rollout_manifest_sha256: str
    rollout_fingerprint: str
    source_artifact_sha256: str
    optimizer_fingerprint: str
    axial_rotation_direction: int | None


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
    controller_contracts = {
        task.task_id: _load_controller_contract(task, model_contract=model_contract) for task in catalog.enabled_tasks
    }
    controllers = {
        task.task_id: fingerprint_controller_artifact(task.controller_artifact) for task in catalog.enabled_tasks
    }
    checkpoint_fingerprints = {task_id: str(record["sha256"]) for task_id, record in controllers.items()}
    phase_fingerprints = {task.task_id: task.phase_schema.fingerprint for task in catalog.enabled_tasks}
    semantic_contracts = primitive_semantic_contracts([task.task_id for task in catalog.enabled_tasks])
    raw_inventory = [
        _raw_inventory_record(task=task, trial=trial) for task in catalog.enabled_tasks for trial in task.trials
    ]
    build_contract = {
        "schema_version": PRIMITIVE_INGEST_SCHEMA_VERSION,
        "ingest_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "catalog_fingerprint": catalog.fingerprint,
        "model_hash": model_contract.model_hash,
        "model_artifact_sha256": model_contract.model_artifact_sha256,
        "regional_grouping_source_sha256": file_sha256(catalog.regional_grouping_path),
        "controller_fingerprints": checkpoint_fingerprints,
        "phase_schema_fingerprints": phase_fingerprints,
        "primitive_semantic_contracts": semantic_contracts,
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
                controller_contract=controller_contracts[task.task_id],
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
    _validate_task_semantic_coverage(loaded_trials)

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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ from schema: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _load_rollout_manifest_envelope(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"primitive rollout manifest must contain an object: {path}")
    manifest = dict(payload)
    _require_exact_fields(manifest, _ROLLOUT_MANIFEST_FIELDS, "primitive rollout manifest")
    if manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"primitive rollout manifest must use {ROLLOUT_MANIFEST_SCHEMA_VERSION}; "
            f"got {manifest.get('schema_version')!r}"
        )
    declared = manifest.pop("rollout_fingerprint", None)
    if not _is_sha256(declared) or canonical_json_sha256(manifest) != declared:
        raise ValueError("primitive rollout manifest self-fingerprint mismatch")
    manifest["rollout_fingerprint"] = declared
    return manifest


def _raw_inventory_record(*, task: PrimitiveTaskSpec, trial: PrimitiveTrialSpec) -> dict[str, Any]:
    manifest = _load_rollout_manifest_envelope(trial.rollout_manifest_path)
    return {
        "task_id": task.task_id,
        "trial_id": trial.trial_id,
        "split": trial.split,
        "motion_path": trial.motion_path,
        "motion_uid": trial.motion_uid,
        "path": str(trial.raw_npz_path),
        "sha256": file_sha256(trial.raw_npz_path),
        "rollout_manifest_path": str(trial.rollout_manifest_path),
        "rollout_manifest_sha256": file_sha256(trial.rollout_manifest_path),
        "rollout_fingerprint": manifest["rollout_fingerprint"],
    }


def _load_controller_contract(
    task: PrimitiveTaskSpec,
    *,
    model_contract: _ModelContract,
) -> _ControllerContract:
    if task.controller_artifact is None or not task.controller_artifact.is_dir():
        raise ValueError(f"enabled primitive task {task.task_id!r} requires a controller artifact directory")
    controller_path = task.controller_artifact.resolve()
    manifest_path = controller_path / "optimizer_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"primitive controller lacks optimizer_manifest.json: {controller_path}")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("primitive optimizer manifest must contain an object")
    manifest = dict(payload)
    _require_exact_fields(manifest, _OPTIMIZER_MANIFEST_FIELDS, "primitive optimizer manifest")
    if manifest.get("schema_version") != OPTIMIZER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("primitive controller uses an unsupported optimizer manifest schema")
    declared = manifest.pop("optimizer_fingerprint", None)
    if not _is_sha256(declared) or canonical_json_sha256(manifest) != declared or controller_path.name != declared:
        raise ValueError("primitive optimizer manifest/path fingerprint mismatch")
    manifest["optimizer_fingerprint"] = declared
    if manifest.get("production_eligible") is not True:
        raise ValueError("primitive controller is not production eligible")
    model_filename = manifest.get("model_artifact_filename")
    if not isinstance(model_filename, str) or Path(model_filename).name != model_filename:
        raise ValueError("primitive optimizer model artifact filename is unsafe")
    model_path = controller_path / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(f"primitive controller model artifact does not exist: {model_path}")
    model_sha256 = file_sha256(model_path)
    if not _is_sha256(manifest.get("model_artifact_sha256")) or manifest["model_artifact_sha256"] != model_sha256:
        raise ValueError("primitive controller MJB byte hash mismatch")
    restored = mujoco.MjModel.from_binary_path(str(model_path))
    restored_state = restored.__getstate__()
    if not isinstance(restored_state, bytes):
        raise ValueError("primitive controller MJB has no canonical model state")
    restored_hash = hashlib.sha256(restored_state).hexdigest()
    if restored_hash != model_contract.model_hash or manifest.get("model_hash") != model_contract.model_hash:
        raise ValueError("primitive controller model differs from the catalog runtime model")
    for field, expected in (
        ("model_nq", int(model_contract.model.nq)),
        ("model_nv", int(model_contract.model.nv)),
        ("model_nu", int(model_contract.model.nu)),
        ("model_na", int(model_contract.model.na)),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"primitive controller {field} differs from the catalog runtime model")
    runtime_binding = manifest.get("runtime_model_binding")
    if not isinstance(runtime_binding, Mapping) or runtime_binding.get("production_eligible") is not True:
        raise ValueError("primitive controller lacks a production runtime binding")
    if runtime_binding.get("num_env_model_hash_invariant") is not True or any(
        runtime_binding.get(field) != model_contract.model_hash
        for field in ("construction_model_hash", "declared_num_env_model_hash")
    ):
        raise ValueError("primitive controller runtime binding differs from the catalog model")
    live_names = tuple(
        str(mujoco.mj_id2name(model_contract.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index))
        for index in range(int(model_contract.model.nu))
    )
    if tuple(manifest.get("actuator_names", ())) != live_names:
        raise ValueError("primitive controller actuator ABI differs from the catalog model")
    live_ctrlrange = np.asarray(model_contract.model.actuator_ctrlrange, dtype=np.float64)
    declared_ctrlrange = np.asarray(manifest.get("ctrlrange"), dtype=np.float64)
    if declared_ctrlrange.shape != live_ctrlrange.shape or not np.array_equal(declared_ctrlrange, live_ctrlrange):
        raise ValueError("primitive controller ctrlrange differs from the catalog runtime model")
    if manifest.get("physics_timestep") != float(model_contract.model.opt.timestep):
        raise ValueError("primitive controller physics timestep differs from the catalog runtime model")
    expected_actuator_hash = actuator_schema_hash(live_names)
    expected_ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={"actuator_names": list(live_names), "ctrlrange": live_ctrlrange.tolist()},
    )
    expected_transform_hash = ctrlrange_schema_hash(live_names, live_ctrlrange)
    for field, expected in (
        ("actuator_schema_hash", expected_actuator_hash),
        ("ctrlrange_schema_hash", expected_ctrlrange_hash),
        ("transform_ctrlrange_schema_hash", expected_transform_hash),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"primitive controller {field} differs from the catalog runtime model")
    if (
        manifest.get("normalized_action_accepted") is not False
        or manifest.get("emg_used") is not False
        or manifest.get("shooting_proposal_residual_can_mark_success") is not False
    ):
        raise ValueError("primitive controller permits a forbidden non-physical success path")
    source_kind = manifest.get("source_kind")
    if source_kind not in {"trajectory_optimizer", "full_action_teacher", "canonical_tonic_control"}:
        raise ValueError("primitive controller source_kind is not explicitly supported")
    if source_kind == "trajectory_optimizer":
        if (
            manifest.get("algorithm") != "contact_forward_transition_shooting_bounded_cem_v1"
            or not isinstance(manifest.get("optimizer_config"), Mapping)
            or manifest.get("policy_controller_binding") is not None
        ):
            raise ValueError("trajectory optimizer contract is malformed")
    elif source_kind == "full_action_teacher":
        if (
            manifest.get("algorithm") != "full_354_policy_actual_ctrl_cpu_replay_v1"
            or manifest.get("optimizer_config") is not None
            or not isinstance(manifest.get("policy_controller_binding"), Mapping)
        ):
            raise ValueError("full-action teacher contract is malformed")
    else:
        config = manifest.get("optimizer_config")
        if (
            manifest.get("algorithm") != "train_only_canonical_tonic_hold_v1"
            or manifest.get("initial_state_contract") != "canonical_tonic_activation_control_v1"
            or manifest.get("policy_controller_binding") is not None
            or not isinstance(config, Mapping)
        ):
            raise ValueError("canonical tonic optimizer contract is malformed")
        if (
            set(config) != {"canonical_control_binding", "canonical_control_filename"}
            or config.get("canonical_control_filename") != "canonical_control.json"
        ):
            raise ValueError("canonical tonic optimizer config is malformed")
        canonical = load_canonical_control_artifact(
            controller_path / "canonical_control.json",
            expected_width=int(model_contract.model.nu),
            require_path_binding=False,
        )
        portable = {key: value for key, value in canonical.items() if key != "path"}
        if (
            portable != config.get("canonical_control_binding")
            or canonical.get("model_hash") != model_contract.model_hash
            or canonical.get("actuator_schema_hash") != expected_actuator_hash
            or canonical.get("ctrlrange_schema_hash") != expected_ctrlrange_hash
        ):
            raise ValueError("canonical tonic artifact binding differs from controller/model ABI")
    return _ControllerContract(
        path=controller_path,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        manifest=manifest,
        optimizer_fingerprint=declared,
        model_artifact_path=model_path,
        model_artifact_sha256=model_sha256,
    )


def _phase_runs_from_array(phase_id: np.ndarray) -> list[dict[str, int]]:
    if phase_id.size == 0:
        return []
    runs: list[dict[str, int]] = []
    start = 0
    for index in range(1, int(phase_id.size) + 1):
        if index == int(phase_id.size) or int(phase_id[index]) != int(phase_id[start]):
            runs.append(
                {
                    "phase_id": int(phase_id[start]),
                    "start": start,
                    "end": index,
                    "length": index - start,
                }
            )
            start = index
    return runs


def _require_passed_semantic_report(report: Any, *, task_id: str, phase_runs: list[dict[str, int]], label: str) -> None:
    if not isinstance(report, Mapping):
        raise ValueError(f"{label} must contain an object")
    if report.get("passed") is not True or report.get("production_eligible") is not True:
        raise ValueError(f"{label} is not passed and production eligible")
    if report.get("task_id") != task_id or report.get("phase_runs") != phase_runs:
        raise ValueError(f"{label} task/phase evidence differs from the raw trial")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(value is not True for value in gates.values()):
        raise ValueError(f"{label} contains a failed or malformed semantic gate")


def _validate_p12_recovery_semantics(
    path: Path,
    *,
    phase_id: np.ndarray,
    qc_config: Mapping[str, Any],
    target_report: Mapping[str, Any],
    actual_report: Mapping[str, Any],
) -> None:
    """Recompute P12 recovery-height gates from the hash-bound QC arrays."""

    threshold = qc_config.get("min_com_vertical_excursion")
    min_ready_hold_frames = qc_config.get("min_ready_hold_frames")
    max_post_impact_speed = qc_config.get("max_post_impact_com_vertical_speed")
    max_ready_hold_speed = qc_config.get("max_ready_hold_com_vertical_speed")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not np.isfinite(float(threshold))
        or float(threshold) < P12_MIN_COM_VERTICAL_EXCURSION
        or isinstance(min_ready_hold_frames, bool)
        or not isinstance(min_ready_hold_frames, int)
        or int(min_ready_hold_frames) < P12_MIN_READY_HOLD_FRAMES
        or isinstance(max_post_impact_speed, bool)
        or not isinstance(max_post_impact_speed, int | float)
        or not np.isfinite(float(max_post_impact_speed))
        or float(max_post_impact_speed) > P12_MAX_POST_IMPACT_COM_VERTICAL_SPEED
        or float(max_post_impact_speed) < 0.0
        or isinstance(max_ready_hold_speed, bool)
        or not isinstance(max_ready_hold_speed, int | float)
        or not np.isfinite(float(max_ready_hold_speed))
        or float(max_ready_hold_speed) > P12_MAX_READY_HOLD_COM_VERTICAL_SPEED
        or float(max_ready_hold_speed) < 0.0
    ):
        raise ValueError("P12 rollout QC config weakens or lacks the canonical recovery contract")
    threshold = float(threshold)
    min_ready_hold_frames = int(min_ready_hold_frames)
    max_post_impact_speed = float(max_post_impact_speed)
    max_ready_hold_speed = float(max_ready_hold_speed)
    runs = _phase_runs_from_array(phase_id)
    if tuple(run["phase_id"] for run in runs) != (0, 1, 2) or any(run["length"] <= 0 for run in runs):
        raise ValueError("P12 recovery evidence requires consecutive landing, restore, and ready-hold phases")

    with np.load(path, allow_pickle=False) as source:
        required = {
            "phase_id",
            "target_com_vertical_position",
            "target_com_vertical_velocity",
            "target_left_foot_floor_contact",
            "target_right_foot_floor_contact",
            "target_site_proxy_left_foot_contact",
            "target_site_proxy_right_foot_contact",
            "actual_com_vertical_position",
            "actual_com_vertical_velocity",
            "actual_left_foot_floor_contact",
            "actual_right_foot_floor_contact",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"P12 rollout QC NPZ lacks required recovery evidence: {sorted(missing)}")
        if not np.array_equal(np.asarray(source["phase_id"]), phase_id):
            raise ValueError("P12 rollout QC phase evidence differs from the raw trial")
        stored_positions = {
            "target": np.asarray(source["target_com_vertical_position"], dtype=np.float64),
            "actual": np.asarray(source["actual_com_vertical_position"], dtype=np.float64),
        }
        stored_velocities = {
            "target": np.asarray(source["target_com_vertical_velocity"], dtype=np.float64),
            "actual": np.asarray(source["actual_com_vertical_velocity"], dtype=np.float64),
        }
        target_proxy = target_report.get("evidence_kind") == "target_site_xpos_hysteresis_proxy"
        target_left_key = "target_site_proxy_left_foot_contact" if target_proxy else "target_left_foot_floor_contact"
        target_right_key = "target_site_proxy_right_foot_contact" if target_proxy else "target_right_foot_floor_contact"
        stored_contacts = {
            "target": (
                np.asarray(source[target_left_key], dtype=np.bool_),
                np.asarray(source[target_right_key], dtype=np.bool_),
            ),
            "actual": (
                np.asarray(source["actual_left_foot_floor_contact"], dtype=np.bool_),
                np.asarray(source["actual_right_foot_floor_contact"], dtype=np.bool_),
            ),
        }

    sample_count = int(phase_id.size)
    masks = {phase: phase_id == phase for phase in (0, 1, 2)}
    for label, report in (("target", target_report), ("actual", actual_report)):
        gates = report.get("gates")
        evidence = report.get("evidence")
        metrics = report.get("metrics")
        thresholds = report.get("thresholds")
        if (
            report.get("vertical_signal") != _P12_COM_VERTICAL_SIGNAL
            or not isinstance(gates, Mapping)
            or not set(P12_REQUIRED_SEMANTIC_GATES).issubset(gates)
            or any(gates.get(name) is not True for name in P12_REQUIRED_SEMANTIC_GATES)
            or not isinstance(evidence, Mapping)
            or not isinstance(metrics, Mapping)
            or not isinstance(thresholds, Mapping)
        ):
            raise ValueError(f"P12 {label} semantic report lacks the required recovery-height contract")
        expected_thresholds = {
            "min_com_vertical_excursion": threshold,
            "min_ready_hold_frames": min_ready_hold_frames,
            "max_post_impact_com_vertical_speed": max_post_impact_speed,
            "max_ready_hold_com_vertical_speed": max_ready_hold_speed,
        }
        malformed_threshold = False
        for name, expected in expected_thresholds.items():
            declared = thresholds.get(name)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int | float)
                or float(declared) != float(expected)
            ):
                malformed_threshold = True
                break
        if malformed_threshold:
            raise ValueError(f"P12 {label} semantic threshold differs from the rollout QC config")

        position = stored_positions[label]
        velocity = stored_velocities[label]
        left_contact, right_contact = stored_contacts[label]
        report_position = np.asarray(evidence.get("vertical_position"), dtype=np.float64)
        report_velocity = np.asarray(evidence.get("vertical_velocity"), dtype=np.float64)
        report_left = np.asarray(evidence.get("left_foot_floor_contact"), dtype=np.bool_)
        report_right = np.asarray(evidence.get("right_foot_floor_contact"), dtype=np.bool_)
        if (
            position.shape != (sample_count,)
            or velocity.shape != (sample_count,)
            or left_contact.shape != (sample_count,)
            or right_contact.shape != (sample_count,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))
            or not np.array_equal(report_position, position)
            or not np.array_equal(report_velocity, velocity)
            or not np.array_equal(report_left, left_contact)
            or not np.array_equal(report_right, right_contact)
        ):
            raise ValueError(f"P12 {label} semantic evidence is malformed or differs from the QC NPZ")
        landing = position[masks[0]]
        restore = position[masks[1]]
        ready_hold = position[masks[2]]
        if not landing.size or not restore.size or not ready_hold.size:
            raise ValueError(f"P12 {label} recovery-height evidence has an empty required phase")

        landing_terminal = float(landing[-1])
        restore_terminal = float(restore[-1])
        ready_min = float(np.min(ready_hold))
        restore_rise = restore_terminal - landing_terminal
        ready_margin = ready_min - landing_terminal
        recomputed = {
            "landing_stabilization_terminal_com_height": landing_terminal,
            "posture_restore_terminal_com_height": restore_terminal,
            "posture_restore_com_rise": restore_rise,
            "ready_hold_min_com_height": ready_min,
            "ready_hold_min_height_above_landing_baseline": ready_margin,
        }
        for name in _P12_RECOVERY_METRIC_NAMES:
            declared = metrics.get(name)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int | float)
                or not np.isfinite(float(declared))
                or not np.isclose(float(declared), recomputed[name], rtol=0.0, atol=1.0e-12)
            ):
                raise ValueError(f"P12 {label} semantic metric {name!r} differs from QC evidence")
        bilateral = left_contact & right_contact
        if (
            not np.all(bilateral)
            or abs(float(velocity[0])) > max_post_impact_speed
            or int(np.count_nonzero(masks[2])) < min_ready_hold_frames
            or not np.all(bilateral[masks[2]])
            or not np.all(np.abs(velocity[masks[2]]) <= max_ready_hold_speed)
            or restore_rise < threshold
            or ready_margin < threshold
        ):
            raise ValueError(f"P12 {label} evidence does not satisfy the recovery-height contract")


def _artifact_record(payload: Any, *, label: str) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} artifact record must contain an object")
    _require_exact_fields(payload, {"filename", "sha256"}, f"{label} artifact record")
    filename = payload.get("filename")
    sha256 = payload.get("sha256")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or not _is_sha256(sha256):
        raise ValueError(f"{label} artifact record is malformed or unsafe")
    return filename, sha256


def _expected_axial_rotation_contract(model: mujoco.MjModel) -> dict[str, Any]:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in _AXIAL_ROTATION_JOINT_NAMES:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"P08 runtime model lacks named axial hinge {name!r}")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    if any(len(set(values)) != len(values) for values in (joint_ids, qpos_addresses, dof_addresses)):
        raise ValueError("P08 named axial hinges do not resolve uniquely")
    if any(address < 0 or address >= int(model.nq) for address in qpos_addresses):
        raise ValueError("P08 axial qpos address lies outside the runtime model")
    if any(address < 0 or address >= int(model.nv) for address in dof_addresses):
        raise ValueError("P08 axial dof address lies outside the runtime model")
    root_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root"))
    if root_joint_id < 0 or int(model.jnt_type[root_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("P08 runtime model requires a free joint named 'root'")
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    if root_qpos_address < 0 or root_qpos_address + 6 >= int(model.nq):
        raise ValueError("P08 root free-joint qpos address lies outside the runtime model")
    return {
        "schema_version": AXIAL_ROTATION_CONTRACT_SCHEMA_VERSION,
        "available": True,
        "joint_names": list(_AXIAL_ROTATION_JOINT_NAMES),
        "joint_ids": joint_ids,
        "qpos_addresses": qpos_addresses,
        "dof_addresses": dof_addresses,
        "position_signal": "sum_named_hinge_qpos_post_transition",
        "velocity_signal": "delta_position_over_transition_duration",
        "sample_time": "post_transition",
        "initial_sample": "pre_first_transition_qpos",
        "root_yaw_signal": "unwrapped_yaw_from_root_freejoint_qpos_quaternion",
        "root_xy_signal": "root_freejoint_qpos_xy",
        "root_joint_id": root_joint_id,
        "root_qpos_address": root_qpos_address,
        "proxy_fallback_allowed": False,
        "unavailable_reason": None,
    }


def _axial_direction_from_report(
    report: Mapping[str, Any],
    *,
    sample_count: int,
    label: str,
) -> int:
    evidence = report.get("evidence")
    metrics = report.get("metrics")
    if not isinstance(evidence, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError(f"{label} lacks P08 axial evidence/metrics")
    position = np.asarray(evidence.get("axial_position"), dtype=np.float64)
    initial = evidence.get("axial_initial_position")
    if (
        position.shape != (sample_count,)
        or not np.all(np.isfinite(position))
        or isinstance(initial, bool)
        or not isinstance(initial, int | float)
        or not np.isfinite(float(initial))
    ):
        raise ValueError(f"{label} P08 axial evidence is malformed")
    deviation = position - float(initial)
    direction = int(np.sign(deviation[int(np.argmax(np.abs(deviation)))]))
    if direction not in {-1, 1} or metrics.get("axial_rotation_direction") != float(direction):
        raise ValueError(f"{label} P08 axial direction is zero or differs from its metric")
    return direction


def _axial_direction_from_replay(
    evidence: Any,
    *,
    expected_contract: Mapping[str, Any],
    sample_count: int,
) -> int:
    if not isinstance(evidence, Mapping):
        raise ValueError("primitive rollout QC lacks P08 replay axial evidence")
    if evidence.get("signal_contract") != expected_contract:
        raise ValueError("P08 replay axial signal contract differs from the runtime model")
    position = np.asarray(evidence.get("position"), dtype=np.float64)
    initial = evidence.get("initial_position")
    if (
        position.shape != (sample_count,)
        or not np.all(np.isfinite(position))
        or isinstance(initial, bool)
        or not isinstance(initial, int | float)
        or not np.isfinite(float(initial))
    ):
        raise ValueError("P08 replay axial evidence is malformed")
    deviation = position - float(initial)
    direction = int(np.sign(deviation[int(np.argmax(np.abs(deviation)))]))
    if direction not in {-1, 1}:
        raise ValueError("P08 replay axial direction is zero")
    return direction


def _load_p08_qc_evidence(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
    phase_id: np.ndarray,
    physics_substeps: np.ndarray,
    target_report: Mapping[str, Any],
    actual_report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
) -> tuple[int, int, int]:
    sample_count = int(phase_id.size)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "phase_id",
            "transition_substeps",
            "target_axial_rotation_position",
            "target_axial_rotation_initial_position",
            "target_axial_rotation_signal_contract_json",
            "actual_axial_rotation_position",
            "actual_axial_rotation_initial_position",
            "actual_axial_rotation_signal_contract_json",
            "replay_axial_rotation_position",
            "replay_axial_rotation_initial_position",
            "replay_axial_rotation_signal_contract_json",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"P08 rollout QC NPZ lacks required direction evidence: {sorted(missing)}")
        stored_phase = np.asarray(source["phase_id"])
        stored_substeps = np.asarray(source["transition_substeps"])
        if not np.array_equal(stored_phase, phase_id) or not np.array_equal(stored_substeps, physics_substeps):
            raise ValueError("P08 rollout QC phase/substep evidence differs from the raw trial or manifest")

        directions: list[int] = []
        reports = (target_report, actual_report, replay_report)
        evidence_keys = ("axial_position", "axial_position", "position")
        initial_keys = ("axial_initial_position", "axial_initial_position", "initial_position")
        for prefix, report, evidence_key, initial_key in zip(
            ("target", "actual", "replay"),
            reports,
            evidence_keys,
            initial_keys,
            strict=True,
        ):
            contract_json = _scalar_string(
                source[f"{prefix}_axial_rotation_signal_contract_json"],
                f"{prefix}_axial_rotation_signal_contract_json",
            )
            try:
                stored_contract = json.loads(contract_json)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P08 {prefix} QC axial contract JSON is malformed") from exc
            if stored_contract != expected_contract:
                raise ValueError(f"P08 {prefix} QC axial contract differs from the runtime model")
            position = np.asarray(source[f"{prefix}_axial_rotation_position"], dtype=np.float64)
            initial_array = np.asarray(source[f"{prefix}_axial_rotation_initial_position"], dtype=np.float64)
            if (
                position.shape != (sample_count,)
                or initial_array.shape != ()
                or not np.all(np.isfinite(position))
                or not np.isfinite(float(initial_array))
            ):
                raise ValueError(f"P08 {prefix} QC axial direction evidence is malformed")
            report_evidence = report.get("evidence") if prefix != "replay" else report
            if not isinstance(report_evidence, Mapping):
                raise ValueError(f"P08 {prefix} manifest axial direction evidence is malformed")
            report_position = np.asarray(report_evidence.get(evidence_key), dtype=np.float64)
            report_initial = report_evidence.get(initial_key)
            if (
                not np.array_equal(report_position, position)
                or isinstance(report_initial, bool)
                or not isinstance(report_initial, int | float)
                or float(report_initial) != float(initial_array)
            ):
                raise ValueError(f"P08 {prefix} QC axial evidence differs from the rollout manifest")
            deviation = position - float(initial_array)
            direction = int(np.sign(deviation[int(np.argmax(np.abs(deviation)))]))
            if direction not in {-1, 1}:
                raise ValueError(f"P08 {prefix} QC axial direction is zero")
            directions.append(direction)
    return directions[0], directions[1], directions[2]


def _validate_qc_npz_core(
    path: Path,
    *,
    physical_ctrl: np.ndarray,
    phase_id: np.ndarray,
    physics_substeps: np.ndarray,
) -> None:
    with np.load(path, allow_pickle=False) as source:
        required = {"applied_ctrl", "phase_id", "transition_substeps"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"primitive rollout QC NPZ lacks core evidence arrays: {sorted(missing)}")
        qc_ctrl = np.asarray(source["applied_ctrl"], dtype=np.float64)
        qc_phase = np.asarray(source["phase_id"])
        qc_substeps = np.asarray(source["transition_substeps"])
        if qc_ctrl.shape != physical_ctrl.shape or not np.array_equal(
            qc_ctrl.astype(np.float32), physical_ctrl.astype(np.float32)
        ):
            raise ValueError("primitive rollout QC applied_ctrl differs from the raw physical control")
        if not np.array_equal(qc_phase, phase_id) or not np.array_equal(qc_substeps, physics_substeps):
            raise ValueError("primitive rollout QC phase/substep arrays differ from the raw trial or manifest")


def _load_and_validate_rollout_manifest(
    *,
    catalog: PrimitiveCatalog,
    model_contract: _ModelContract,
    controller_contract: _ControllerContract,
    task: PrimitiveTaskSpec,
    trial: PrimitiveTrialSpec,
    raw_sha256: str,
    physical_ctrl: np.ndarray,
    phase_id: np.ndarray,
    sample_count: int,
) -> dict[str, Any]:
    manifest_path = trial.rollout_manifest_path
    manifest = _load_rollout_manifest_envelope(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    if manifest.get("trial_id") != trial.trial_id or manifest.get("task_id") != task.task_id:
        raise ValueError("primitive rollout manifest trial/task identity differs from the catalog")
    if (
        manifest.get("target_skill_id") != catalog.target_skill_id
        or manifest.get("contains_target_skill_rollout") is not False
    ):
        raise ValueError("primitive rollout manifest contains or misidentifies the target skill")
    source_motion_path = manifest.get("source_motion_path")
    if not isinstance(source_motion_path, str):
        raise ValueError("primitive rollout manifest source_motion_path must be a string")
    try:
        normalized_source = normalize_relative_motion_path(source_motion_path)
    except ValueError as exc:
        raise ValueError("primitive rollout manifest source_motion_path is invalid") from exc
    if (
        normalized_source != trial.motion_path
        or manifest.get("source_motion_uid") != trial.motion_uid
        or stable_motion_uid(normalized_source) != trial.motion_uid
    ):
        raise ValueError("primitive rollout manifest source-motion identity differs from the catalog")
    source_sha256 = manifest.get("source_artifact_sha256")
    if not _is_sha256(source_sha256):
        raise ValueError("primitive rollout manifest source artifact SHA-256 is malformed")
    source_frequency = manifest.get("source_frequency_hz")
    if (
        isinstance(source_frequency, bool)
        or not isinstance(source_frequency, int | float)
        or not np.isfinite(float(source_frequency))
        or float(source_frequency) <= 0.0
    ):
        raise ValueError("primitive rollout manifest source frequency must be finite and positive")
    interval = manifest.get("source_frame_interval")
    if not isinstance(interval, Mapping):
        raise ValueError("primitive rollout manifest source frame interval must contain an object")
    _require_exact_fields(
        interval,
        {"start_frame", "end_frame_exclusive", "source_total_frames"},
        "primitive rollout source frame interval",
    )
    start = interval.get("start_frame")
    end = interval.get("end_frame_exclusive")
    total = interval.get("source_total_frames")
    if (
        any(type(value) is not int for value in (start, end, total))
        or start < 0
        or end <= start
        or total < end
        or end - start - 1 != sample_count
    ):
        raise ValueError("primitive rollout source frame interval differs from its transition count")
    if manifest.get("phase_schema_fingerprint") != task.phase_schema.fingerprint:
        raise ValueError("primitive rollout phase-schema fingerprint differs from the catalog")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0:
        raise ValueError("primitive rollout optimizer seed must be a non-negative integer")
    if any(
        manifest.get(field) != sample_count for field in ("requested_transition_count", "recorded_transition_count")
    ):
        raise ValueError("primitive rollout requested/recorded transition count differs from the raw trial")

    transition = manifest.get("transition_contract")
    if not isinstance(transition, Mapping):
        raise ValueError("primitive rollout transition contract must contain an object")
    _require_exact_fields(
        transition,
        {"control", "activation", "foot_floor_contact", "physics_substeps", "mujoco_state_spec"},
        "primitive rollout transition contract",
    )
    raw_substeps = transition.get("physics_substeps")
    if (
        not isinstance(raw_substeps, list)
        or len(raw_substeps) != sample_count
        or any(type(value) is not int or value <= 0 for value in raw_substeps)
    ):
        raise ValueError("primitive rollout physics_substeps must be positive integers aligned to samples")
    physics_substeps = np.asarray(raw_substeps, dtype=np.int32)
    expected_duration = 1.0 / float(source_frequency)
    actual_durations = physics_substeps.astype(np.float64) * float(model_contract.model.opt.timestep)
    if not np.allclose(actual_durations, expected_duration, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("primitive rollout source frequency differs from its MuJoCo transition durations")
    initialization = manifest.get("initialization_contract")
    if not isinstance(initialization, Mapping) or initialization.get("contract") != controller_contract.manifest.get(
        "initial_state_contract"
    ):
        raise ValueError("primitive rollout initialization differs from the optimizer artifact")

    optimizer = manifest.get("optimizer_artifact")
    if not isinstance(optimizer, Mapping):
        raise ValueError("primitive rollout optimizer artifact record must contain an object")
    _require_exact_fields(
        optimizer,
        {"path", "manifest_sha256", "model_artifact_sha256"},
        "primitive rollout optimizer artifact",
    )
    optimizer_path = optimizer.get("path")
    if not isinstance(optimizer_path, str) or Path(optimizer_path).expanduser().resolve() != controller_contract.path:
        raise ValueError("primitive rollout optimizer artifact path differs from the catalog controller")
    if (
        manifest.get("optimizer_fingerprint") != controller_contract.optimizer_fingerprint
        or optimizer.get("manifest_sha256") != controller_contract.manifest_sha256
        or optimizer.get("model_artifact_sha256") != controller_contract.model_artifact_sha256
        or manifest.get("controller_source_kind") != controller_contract.manifest.get("source_kind")
        or manifest.get("runtime_model_binding") != controller_contract.manifest.get("runtime_model_binding")
    ):
        raise ValueError("primitive rollout optimizer/runtime binding differs from the catalog controller")
    source_kind = controller_contract.manifest.get("source_kind")
    if (
        source_kind in {"trajectory_optimizer", "canonical_tonic_control"}
        and manifest.get("policy_rollout_binding") is not None
    ):
        raise ValueError("non-policy primitive rollout unexpectedly declares a policy binding")
    provenance = manifest.get("runtime_model_provenance")
    runtime_binding = controller_contract.manifest["runtime_model_binding"]
    if provenance is not None:
        if not isinstance(provenance, Mapping) or any(
            provenance.get(field) != expected
            for field, expected in (
                ("model_hash", model_contract.model_hash),
                ("config_name", runtime_binding.get("config_name")),
                ("hydra_overrides", runtime_binding.get("hydra_overrides")),
            )
        ):
            raise ValueError("primitive rollout runtime provenance differs from the controller binding")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("primitive rollout artifacts must contain an object")
    _require_exact_fields(artifacts, {"primitive_trial", "rollout_qc"}, "primitive rollout artifacts")
    trial_filename, trial_hash = _artifact_record(artifacts.get("primitive_trial"), label="primitive trial")
    if trial_filename != trial.raw_npz_path.name or trial_hash != raw_sha256:
        raise ValueError("primitive rollout manifest raw artifact differs from the catalog NPZ")
    qc_filename, qc_hash = _artifact_record(artifacts.get("rollout_qc"), label="primitive rollout QC")
    qc_path = manifest_path.parent / qc_filename
    if not qc_path.is_file() or file_sha256(qc_path) != qc_hash:
        raise ValueError("primitive rollout QC artifact is missing or its byte hash changed")
    _validate_qc_npz_core(
        qc_path,
        physical_ctrl=physical_ctrl,
        phase_id=phase_id,
        physics_substeps=physics_substeps,
    )

    qc = manifest.get("qc")
    if not isinstance(qc, Mapping):
        raise ValueError("primitive rollout QC report must contain an object")
    _require_exact_fields(qc, _ROLLOUT_QC_FIELDS, "primitive rollout QC report")
    qc_gates = qc.get("gates")
    if (
        manifest.get("status") != "success"
        or manifest.get("success") is not True
        or manifest.get("production_eligible") is not True
        or qc.get("passed") is not True
        or not isinstance(qc_gates, Mapping)
        or not qc_gates
        or any(value is not True for value in qc_gates.values())
        or qc.get("recorded_transition_count") != sample_count
        or qc.get("expected_transition_count") != sample_count
        or qc.get("success_does_not_depend_on_shooting_proposal_residual") is not True
    ):
        raise ValueError("primitive rollout manifest is not a complete production-passed QC result")
    qc_config = manifest.get("qc_config")
    if not isinstance(qc_config, Mapping):
        raise ValueError("primitive rollout manifest lacks its QC threshold configuration")
    phase_runs = _phase_runs_from_array(phase_id)
    target_semantics = manifest.get("target_contact_semantics")
    if not isinstance(target_semantics, Mapping) or target_semantics != qc.get("target_contact_semantics"):
        raise ValueError("primitive rollout target semantic report is missing or differs inside QC")
    if target_semantics.get("passed") is not True or target_semantics.get("task_id") != task.task_id:
        raise ValueError("primitive rollout target semantic wrapper is not passed for the catalog task")
    gate_basis = target_semantics.get("gate_basis")
    if gate_basis == "exact_mj_forward_contact":
        target_report = target_semantics.get("exact")
    elif gate_basis == "site_xpos_hysteresis_proxy" and target_semantics.get("proxy_fallback_allowed") is True:
        target_report = target_semantics.get("site_proxy")
    else:
        raise ValueError("primitive rollout target semantic gate basis is unsupported or failed closed")
    _require_passed_semantic_report(
        target_report,
        task_id=task.task_id,
        phase_runs=phase_runs,
        label="primitive rollout selected target semantic report",
    )
    actual_report = qc.get("actual_contact_semantics")
    _require_passed_semantic_report(
        actual_report,
        task_id=task.task_id,
        phase_runs=phase_runs,
        label="primitive rollout actual semantic report",
    )
    if actual_report.get("evidence_kind") != "actual_rollout_exact_contact":
        raise ValueError("primitive rollout actual semantics are not based on exact rollout contact")
    task_family = task.task_id.split("_", 1)[0]
    if task_family == "P12":
        _validate_p12_recovery_semantics(
            qc_path,
            phase_id=phase_id,
            qc_config=qc_config,
            target_report=target_report,
            actual_report=actual_report,
        )
    initialization_evidence = qc.get("initialization_evidence")
    if not isinstance(initialization_evidence, Mapping) or initialization_evidence.get(
        "contract"
    ) != controller_contract.manifest.get("initial_state_contract"):
        raise ValueError("primitive rollout QC initialization evidence differs from the optimizer artifact")
    contact_contract = manifest.get("contact_contract")
    if (
        not isinstance(contact_contract, Mapping)
        or contact_contract.get("available") is not True
        or qc.get("contact_contract") != contact_contract
        or target_semantics.get("contact_contract") != contact_contract
    ):
        raise ValueError("primitive rollout exact contact contract is unavailable or internally inconsistent")
    axial_contract = manifest.get("axial_rotation_signal_contract")
    if not isinstance(axial_contract, Mapping) or qc.get("axial_rotation_signal_contract") != axial_contract:
        raise ValueError("primitive rollout axial signal contract is missing or internally inconsistent")

    axial_direction: int | None = None
    if gate_basis == "site_xpos_hysteresis_proxy" and task_family not in {"P05", "P06", "P07", "P11", "P12"}:
        raise ValueError(f"primitive task family {task_family!r} may not use target site-proxy semantics")
    if task_family == "P08":
        expected_axial = _expected_axial_rotation_contract(model_contract.model)
        replay_report = qc.get("replay_axial_rotation_evidence")
        if (
            gate_basis != "exact_mj_forward_contact"
            or target_semantics.get("proxy_fallback_allowed") is not False
            or axial_contract != expected_axial
            or target_report.get("axial_signal_contract") != expected_axial
            or actual_report.get("axial_signal_contract") != expected_axial
            or not isinstance(replay_report, Mapping)
            or replay_report.get("signal_contract") != expected_axial
        ):
            raise ValueError("P08 rollout does not use one exact named axial signal contract without fallback")
        target_direction = _axial_direction_from_report(
            target_report,
            sample_count=sample_count,
            label="P08 target report",
        )
        actual_direction = _axial_direction_from_report(
            actual_report,
            sample_count=sample_count,
            label="P08 actual report",
        )
        replay_direction = _axial_direction_from_replay(
            replay_report,
            expected_contract=expected_axial,
            sample_count=sample_count,
        )
        stored_directions = _load_p08_qc_evidence(
            qc_path,
            expected_contract=expected_axial,
            phase_id=phase_id,
            physics_substeps=physics_substeps,
            target_report=target_report,
            actual_report=actual_report,
            replay_report=replay_report,
        )
        direction_evidence = qc.get("axial_rotation_direction_evidence")
        if (
            target_direction != actual_direction
            or target_direction != replay_direction
            or stored_directions != (target_direction, actual_direction, replay_direction)
            or not isinstance(direction_evidence, Mapping)
            or direction_evidence.get("required") is not True
            or direction_evidence.get("target") != float(target_direction)
            or direction_evidence.get("actual") != float(actual_direction)
        ):
            raise ValueError("P08 target/actual/replay axial rotation directions are inconsistent")
        axial_direction = target_direction

    return {
        "manifest_sha256": manifest_sha256,
        "rollout_fingerprint": manifest["rollout_fingerprint"],
        "source_artifact_sha256": source_sha256,
        "optimizer_fingerprint": controller_contract.optimizer_fingerprint,
        "axial_rotation_direction": axial_direction,
    }


def _validate_task_semantic_coverage(loaded_trials: Sequence[_RawTrial]) -> None:
    source_hashes_by_split = {
        split: {loaded.source_artifact_sha256 for loaded in loaded_trials if loaded.trial.split == split}
        for split in ("train", "val")
    }
    source_overlap = source_hashes_by_split["train"] & source_hashes_by_split["val"]
    if source_overlap:
        raise ValueError(
            f"primitive train/validation source-artifact content leakage detected: {sorted(source_overlap)}"
        )
    task_ids = sorted({loaded.task.task_id for loaded in loaded_trials})
    for task_id in task_ids:
        if task_id.split("_", 1)[0] != "P08":
            continue
        selected = [loaded for loaded in loaded_trials if loaded.task.task_id == task_id]
        if any(loaded.axial_rotation_direction not in {-1, 1} for loaded in selected):
            raise ValueError(f"P08 task {task_id!r} contains a trial without an exact axial direction")
        train_directions = {
            int(loaded.axial_rotation_direction)
            for loaded in selected
            if loaded.trial.split == "train" and loaded.axial_rotation_direction is not None
        }
        if train_directions != {-1, 1}:
            raise ValueError(
                f"P08 task {task_id!r} requires both axial directions in train; got {sorted(train_directions)}"
            )


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
    controller_contract: _ControllerContract,
    task: PrimitiveTaskSpec,
    trial: PrimitiveTrialSpec,
) -> _RawTrial:
    raw_sha256 = file_sha256(trial.raw_npz_path)
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

        channel_contract = resolve_muscle_channel_contract(
            model_contract.model,
            names,
        )
        ctrlrange = validate_unit_muscle_ctrlrange(
            names,
            model_contract.model.actuator_ctrlrange[np.asarray(channel_contract.actuator_ids, dtype=np.int32)],
        )
        excitation = physical_ctrl_to_effective_muscle_excitation(
            ctrl,
            channel_contract=channel_contract,
        )
        if "muscle_excitation" in source:
            stored_excitation = np.asarray(source["muscle_excitation"], dtype=np.float64)
            if stored_excitation.shape != excitation.shape or not np.allclose(
                stored_excitation,
                excitation,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("raw muscle_excitation differs from clip(raw data.ctrl,0,1)")
        _validate_optional_embedded_contracts(
            source,
            names=names,
            ctrlrange=ctrlrange,
            model_hash=model_contract.model_hash,
            channel_contract=channel_contract.to_metadata(),
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
    rollout = _load_and_validate_rollout_manifest(
        catalog=catalog,
        model_contract=model_contract,
        controller_contract=controller_contract,
        task=task,
        trial=trial,
        raw_sha256=raw_sha256,
        physical_ctrl=ctrl,
        phase_id=phase_id,
        sample_count=int(ctrl.shape[0]),
    )
    return _RawTrial(
        task=task,
        trial=trial,
        arrays=arrays,
        actuator_names=names,
        actuator_ctrlrange=ctrlrange,
        raw_sha256=raw_sha256,
        rollout_manifest_path=trial.rollout_manifest_path,
        rollout_manifest_sha256=rollout["manifest_sha256"],
        rollout_fingerprint=rollout["rollout_fingerprint"],
        source_artifact_sha256=rollout["source_artifact_sha256"],
        optimizer_fingerprint=rollout["optimizer_fingerprint"],
        axial_rotation_direction=rollout["axial_rotation_direction"],
    )


def _validate_optional_embedded_contracts(
    source: Any,
    *,
    names: tuple[str, ...],
    ctrlrange: np.ndarray,
    model_hash: str,
    channel_contract: Mapping[str, Any],
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
    contract_fields = {
        "physical_signal_schema_version",
        "muscle_excitation_transform",
        "muscle_channel_contract_schema_version",
        "actuator_ids",
        "actuator_dyntype",
        "actuator_actnum",
        "actuator_actadr",
        "model_na",
    }
    present = contract_fields & set(source.files)
    if present and present != contract_fields:
        raise ValueError(
            f"raw embedded v2 muscle channel contract is partial; missing={sorted(contract_fields - present)}"
        )
    if present:
        if (
            _scalar_string(
                source["physical_signal_schema_version"],
                "physical_signal_schema_version",
            )
            != PHYSICAL_SIGNAL_SCHEMA_VERSION
            or _scalar_string(
                source["muscle_excitation_transform"],
                "muscle_excitation_transform",
            )
            != UNIT_EXCITATION_TRANSFORM
        ):
            raise ValueError("raw embedded physical signal schema/transform is legacy or unsupported")
        embedded = {
            "schema_version": _scalar_string(
                source["muscle_channel_contract_schema_version"],
                "muscle_channel_contract_schema_version",
            ),
            "actuator_names": list(names),
            "actuator_ids": np.asarray(source["actuator_ids"]).tolist(),
            "actuator_dyntype": [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(source["actuator_dyntype"]).tolist()
            ],
            "actuator_actnum": np.asarray(source["actuator_actnum"]).tolist(),
            "actuator_actadr": np.asarray(source["actuator_actadr"]).tolist(),
            "model_na": int(np.asarray(source["model_na"]).item()),
        }
        if (
            validate_muscle_channel_contract(
                embedded,
                expected_names=names,
            ).to_metadata()
            != validate_muscle_channel_contract(
                channel_contract,
                expected_names=names,
            ).to_metadata()
        ):
            raise ValueError("raw embedded muscle channel contract differs from the live MuJoCo model")


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
        "primitive_semantic_contracts": dict(build_contract["primitive_semantic_contracts"]),
        "raw_trial_sources": [
            {
                "task_id": loaded.task.task_id,
                "trial_id": loaded.trial.trial_id,
                "split": loaded.trial.split,
                "motion_path": loaded.trial.motion_path,
                "motion_uid": loaded.trial.motion_uid,
                "path": str(loaded.trial.raw_npz_path),
                "sha256": loaded.raw_sha256,
                "rollout_manifest_path": str(loaded.rollout_manifest_path),
                "rollout_manifest_sha256": loaded.rollout_manifest_sha256,
                "rollout_fingerprint": loaded.rollout_fingerprint,
                "source_artifact_sha256": loaded.source_artifact_sha256,
                "optimizer_fingerprint": loaded.optimizer_fingerprint,
                "axial_rotation_direction": loaded.axial_rotation_direction,
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
    channel_contract = resolve_muscle_channel_contract(
        model_contract.model,
        actuator_names,
    )
    metadata["physical_capture"] = {
        "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "actuator_names": list(actuator_names),
        "model_nu": int(model_contract.model.nu),
        "model_nv": int(model_contract.model.nv),
        "model_na": int(model_contract.model.na),
        "activation_valid_mask": [True] * len(actuator_names),
        "muscle_channel_contract": channel_contract.to_metadata(),
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
            "task_axial_rotation_directions": {
                task_id: sorted(
                    {
                        int(item.axial_rotation_direction)
                        for item in selected
                        if item.task.task_id == task_id and item.axial_rotation_direction is not None
                    }
                )
                for task_id in sorted({item.task.task_id for item in selected})
                if task_id.split("_", 1)[0] == "P08"
            },
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
        "primitive_semantic_contracts": dict(build_contract["primitive_semantic_contracts"]),
        "rollout_evidence": [
            {
                "task_id": item.task.task_id,
                "trial_id": item.trial.trial_id,
                "split": item.trial.split,
                "rollout_manifest_sha256": item.rollout_manifest_sha256,
                "rollout_fingerprint": item.rollout_fingerprint,
                "source_artifact_sha256": item.source_artifact_sha256,
                "optimizer_fingerprint": item.optimizer_fingerprint,
                "axial_rotation_direction": item.axial_rotation_direction,
            }
            for item in loaded_trials
        ],
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
        raise ValueError("existing primitive dataset QC is not a passed v3 contract")
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


def validate_ingested_primitive_dataset(path: str | Path) -> dict[str, Any]:
    """Validate a sealed ingest directory and return its semantic attestation identity."""

    supplied = Path(path).expanduser().resolve()
    output = supplied if supplied.is_dir() else supplied.parent
    metadata_path = output / "metadata.json"
    qc_path = output / "dataset_qc.json"
    if not metadata_path.is_file() or not qc_path.is_file():
        raise ValueError("versioned primitive semantics require metadata.json and dataset_qc.json")
    metadata = load_json_strict(metadata_path)
    qc = load_json_strict(qc_path)
    if not isinstance(metadata, Mapping) or not isinstance(qc, Mapping):
        raise ValueError("primitive dataset metadata/QC attestation must contain objects")
    required_phases = metadata.get("primitive_required_phase_ids")
    if not isinstance(required_phases, Mapping) or not required_phases:
        raise ValueError("primitive dataset metadata lacks its task/phase inventory")
    task_ids = tuple(str(task_id) for task_id in required_phases)
    declared_contracts = validate_primitive_semantic_contracts(
        task_ids,
        metadata.get("primitive_semantic_contracts"),
        label="primitive dataset metadata",
    )
    build_contract = metadata.get("ingest_contract")
    build_fingerprint = metadata.get("ingest_build_fingerprint")
    if (
        not isinstance(build_contract, Mapping)
        or not isinstance(build_fingerprint, str)
        or canonical_json_sha256(build_contract) != build_fingerprint
        or build_contract.get("primitive_semantic_contracts") != declared_contracts
        or qc.get("build_contract") != build_contract
        or qc.get("build_fingerprint") != build_fingerprint
        or qc.get("primitive_semantic_contracts") != declared_contracts
    ):
        raise ValueError("primitive dataset semantic attestation differs across metadata, build contract, and QC")
    _reuse_idempotent_output(output, expected_build_fingerprint=build_fingerprint)
    attestation = {
        "schema_version": PRIMITIVE_SEMANTIC_ATTESTATION_SCHEMA_VERSION,
        "primitive_semantic_contracts": declared_contracts,
        "ingest_build_fingerprint": build_fingerprint,
        "dataset_qc_sha256": file_sha256(qc_path),
    }
    attestation["attestation_fingerprint"] = canonical_json_sha256(attestation)
    return attestation


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
