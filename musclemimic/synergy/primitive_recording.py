"""Atomic writer for one primitive physical-control rollout trial.

This module is deliberately collection-side only: callers provide the exact
compiled runtime ``MjModel`` and the control applied by that model.  It never
converts a normalized policy action or selects a body-actuator subset.  For a
354-D primitive, the caller must first construct the 354-actuator
``disable_fingers=True`` runtime model.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import (
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    physical_ctrl_to_effective_muscle_excitation,
    resolve_muscle_channel_contract,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash


def write_primitive_trial_npz(
    path: str | Path,
    *,
    model: mujoco.MjModel,
    actuator_names: Sequence[str],
    teacher_ctrl_physical: Any | None = None,
    applied_ctrl: Any | None = None,
    phase_id: Any,
    success: Any,
    muscle_activation: Any | None = None,
    muscle_force: Any | None = None,
    muscle_tendon_length: Any | None = None,
    muscle_tendon_velocity: Any | None = None,
    phase_local: Any | None = None,
    overwrite: bool = False,
) -> str:
    """Write one ingest-ready trial and return its file SHA-256.

    Exactly one physical-control ABI is accepted: ``actuator_names`` must equal
    the complete ordered actuator list of ``model`` and the control width must
    be ``model.nu``.  ``teacher_ctrl_physical`` and ``applied_ctrl`` are aliases;
    when both are supplied their values must be identical.
    """

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("primitive trial recording requires mujoco.MjModel")
    names = tuple(str(name) for name in actuator_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("actuator_names must be non-empty, unique strings")
    model_names = _complete_model_actuator_names(model)
    if int(model.nu) != len(names) or names != model_names:
        raise ValueError(
            "actuator_names must equal the compiled model complete actuator order; "
            f"model.nu={int(model.nu)} supplied={len(names)}"
        )
    channel_contract = resolve_muscle_channel_contract(model, names)
    ctrlrange = validate_unit_muscle_ctrlrange(names, model.actuator_ctrlrange)
    ctrl = _resolve_physical_ctrl(
        teacher_ctrl_physical=teacher_ctrl_physical,
        applied_ctrl=applied_ctrl,
        action_dim=len(names),
    )
    if np.any(ctrl < ctrlrange[:, 0]) or np.any(ctrl > ctrlrange[:, 1]):
        raise ValueError("physical ctrl lies outside the exact compiled model ctrlrange")
    excitation = physical_ctrl_to_effective_muscle_excitation(
        ctrl,
        channel_contract=channel_contract,
    )
    sample_count = int(ctrl.shape[0])
    phases = _validate_phase_id(phase_id, sample_count=sample_count)
    recorded_success = _validate_success(success, sample_count=sample_count)

    arrays: dict[str, np.ndarray] = {
        "teacher_ctrl_physical": ctrl.astype(np.float32),
        "muscle_excitation": excitation,
        "phase_id": phases,
        "success": recorded_success,
        "actuator_names": np.asarray(names),
        "actuator_ctrlrange": ctrlrange.astype(np.float64),
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_excitation_transform": np.asarray(UNIT_EXCITATION_TRANSFORM),
        "muscle_channel_contract_schema_version": np.asarray(channel_contract.to_metadata()["schema_version"]),
        "actuator_ids": np.asarray(channel_contract.actuator_ids, dtype=np.int32),
        "actuator_dyntype": np.asarray(channel_contract.actuator_dyntype),
        "actuator_actnum": np.asarray(channel_contract.actuator_actnum, dtype=np.int32),
        "actuator_actadr": np.asarray(channel_contract.actuator_actadr, dtype=np.int32),
        "model_na": np.asarray(channel_contract.model_na, dtype=np.int32),
    }
    for field, value in (
        ("muscle_force", muscle_force),
        ("muscle_tendon_length", muscle_tendon_length),
        ("muscle_tendon_velocity", muscle_tendon_velocity),
    ):
        if value is not None:
            arrays[field] = _validate_optional_matrix(
                value,
                field=field,
                expected_shape=ctrl.shape,
            )
    if muscle_activation is not None:
        activation = np.asarray(muscle_activation)
        if activation.shape != ctrl.shape:
            raise ValueError(f"muscle_activation must have shape {ctrl.shape}, got {activation.shape}")
        arrays["muscle_activation"] = validate_unit_muscle_activation(activation)
    if phase_local is not None:
        local = np.asarray(phase_local, dtype=np.float64)
        if local.shape != (sample_count,) or not np.all(np.isfinite(local)):
            raise ValueError("phase_local must be a finite vector with shape [T]")
        arrays["phase_local"] = local.astype(np.float32)

    model_state = model.__getstate__()
    if not isinstance(model_state, bytes) or not model_state:
        raise ValueError("MuJoCo model has no canonical complete byte state")
    arrays.update(
        {
            "model_hash": np.asarray(hashlib.sha256(model_state).hexdigest()),
            "actuator_schema_hash": np.asarray(actuator_schema_hash(names)),
            "ctrlrange_schema_hash": np.asarray(
                ordered_schema_hash(
                    kind="actuator_ctrlrange",
                    payload={
                        "actuator_names": list(names),
                        "ctrlrange": ctrlrange.tolist(),
                    },
                )
            ),
            "transform_ctrlrange_schema_hash": np.asarray(ctrlrange_schema_hash(names, ctrlrange)),
        }
    )

    output = Path(path).expanduser().resolve()
    if output.suffix.casefold() != ".npz":
        raise ValueError("primitive trial recording path must use the .npz suffix")
    if output.exists() and not overwrite:
        raise FileExistsError(f"primitive trial recording already exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        fingerprint = _file_sha256(temporary)
        if output.exists() and not overwrite:
            raise FileExistsError(f"primitive trial recording appeared concurrently; refusing overwrite: {output}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return fingerprint


def _complete_model_actuator_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        if not name:
            raise ValueError(f"compiled model actuator {actuator_id} has no stable name")
        names.append(str(name))
    if len(names) != len(set(names)):
        raise ValueError("compiled model actuator names are not unique")
    return tuple(names)


def _resolve_physical_ctrl(
    *,
    teacher_ctrl_physical: Any | None,
    applied_ctrl: Any | None,
    action_dim: int,
) -> np.ndarray:
    if teacher_ctrl_physical is None and applied_ctrl is None:
        raise ValueError("teacher_ctrl_physical or applied_ctrl is required; normalized action is not accepted")
    primary = teacher_ctrl_physical if teacher_ctrl_physical is not None else applied_ctrl
    ctrl = np.asarray(primary, dtype=np.float64)
    if ctrl.ndim != 2 or ctrl.shape[0] <= 0 or ctrl.shape[1] != int(action_dim):
        raise ValueError(f"physical ctrl must have shape [T,{int(action_dim)}], got {ctrl.shape}")
    if not np.all(np.isfinite(ctrl)):
        raise ValueError("physical ctrl contains non-finite values")
    if teacher_ctrl_physical is not None and applied_ctrl is not None:
        alias = np.asarray(applied_ctrl, dtype=np.float64)
        if alias.shape != ctrl.shape or not np.array_equal(alias, ctrl):
            raise ValueError("teacher_ctrl_physical and applied_ctrl must be exactly identical")
    return ctrl


def _validate_phase_id(value: Any, *, sample_count: int) -> np.ndarray:
    phases = np.asarray(value)
    if (
        phases.shape != (int(sample_count),)
        or np.issubdtype(phases.dtype, np.bool_)
        or not np.issubdtype(phases.dtype, np.integer)
    ):
        raise ValueError("phase_id must be an integer vector with shape [T]")
    if np.any(phases < 0) or np.any(phases > np.iinfo(np.int32).max):
        raise ValueError("phase_id must contain non-negative int32 values")
    return phases.astype(np.int32)


def _validate_success(value: Any, *, sample_count: int) -> np.ndarray:
    success = np.asarray(value)
    if success.dtype.kind != "b" or success.shape not in {(), (int(sample_count),)}:
        raise ValueError("success must be a boolean scalar or boolean vector with shape [T]")
    return success.astype(bool, copy=False)


def _validate_optional_matrix(
    value: Any,
    *,
    field: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite and have shape {expected_shape}")
    return array.astype(np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
