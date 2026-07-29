"""MuJoCo muscle-channel and effective-excitation contracts.

MuJoCo's muscle activation dynamics consume a control signal clamped to
``[0, 1]``.  The actuator ``ctrlrange`` is a separate outer control limit and
must not be used as a generic affine definition of muscle excitation.

This module is intentionally independent from rewards and policy decoders.  It
provides one canonical signal helper plus fail-closed runtime-model validation
that can be shared by data collection, diagnostics, and later training code.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_EXCITATION_SEMANTICS,
    MUSCLE_EXCITATION_SOURCE,
    MuscleChannelContract,
    physical_ctrl_to_effective_muscle_excitation,
    resolve_muscle_channel_contract,
    validate_ordered_ctrlrange,
    validate_unit_muscle_ctrlrange,
)

# Compatibility names for the physiology namespace.  These are aliases to the
# canonical distillation/collection contract, never an independent contract.
EFFECTIVE_EXCITATION_SEMANTICS = MUSCLE_EXCITATION_SEMANTICS
EFFECTIVE_EXCITATION_SOURCE = MUSCLE_EXCITATION_SOURCE
UNIT_MUSCLE_CTRLRANGE = (0.0, 1.0)

# Production conversion is exactly the canonical, contract-validating helper.
effective_mujoco_muscle_excitation = physical_ctrl_to_effective_muscle_excitation


@dataclass(frozen=True)
class MuscleChannelLayout:
    """Exact ordered mapping from policy muscle channels to MuJoCo state."""

    actuator_names: tuple[str, ...]
    actuator_ids: np.ndarray
    activation_addresses: np.ndarray
    activation_counts: np.ndarray
    ctrlrange: np.ndarray
    dyntype_ids: np.ndarray
    actuator_schema_hash: str
    runtime_model_hash: str

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.actuator_names)
        width = len(names)
        if width == 0 or len(set(names)) != width:
            raise ValueError("muscle channel layout requires non-empty unique actuator names")
        arrays = {
            "actuator_ids": (self.actuator_ids, (width,), np.int32),
            "activation_addresses": (self.activation_addresses, (width,), np.int32),
            "activation_counts": (self.activation_counts, (width,), np.int32),
            "ctrlrange": (self.ctrlrange, (width, 2), np.float64),
            "dyntype_ids": (self.dyntype_ids, (width,), np.int32),
        }
        normalized: dict[str, np.ndarray] = {}
        for field_name, (value, shape, dtype) in arrays.items():
            array = np.asarray(value, dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{field_name} must have shape {shape}, got {array.shape}")
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                raise ValueError(f"{field_name} must be finite")
            array = array.copy()
            array.setflags(write=False)
            normalized[field_name] = array
        if self.actuator_schema_hash != actuator_schema_hash(names):
            raise ValueError("muscle channel layout actuator_schema_hash is stale")
        if not _is_sha256(self.runtime_model_hash):
            raise ValueError("muscle channel layout runtime_model_hash must be a lowercase SHA-256")
        object.__setattr__(self, "actuator_names", names)
        for field_name, array in normalized.items():
            object.__setattr__(self, field_name, array)

    @property
    def width(self) -> int:
        return len(self.actuator_names)


def jax_effective_muscle_excitation(raw_ctrl: Any, *, backend: Any) -> Any:
    """Numerically mirror the canonical excitation clamp inside JAX code.

    Model/channel validation must happen before entering JIT through
    :func:`resolve_muscle_channel_layout`.  Persisted or production-facing
    conversion must use :func:`effective_mujoco_muscle_excitation`, which is
    an alias of the canonical contract-validating helper in
    :mod:`musclemimic.distill.physical`.
    """

    values = backend.asarray(raw_ctrl)
    return backend.clip(values, UNIT_MUSCLE_CTRLRANGE[0], UNIT_MUSCLE_CTRLRANGE[1])


def normalized_policy_action_to_unit_muscle_ctrl(action: Any, *, backend: Any = np) -> Any:
    """Map the stable policy ABI ``[-1, 1]`` to physical muscle ctrl ``[0, 1]``."""

    values = backend.asarray(action)
    return backend.clip(0.5 * (values + 1.0), 0.0, 1.0)


def effective_excitation_clip_diagnostics(raw_ctrl: Any, *, backend: Any = np) -> dict[str, Any]:
    """Compute command-level clipping diagnostics without changing reward."""

    raw = backend.asarray(raw_ctrl)
    effective = jax_effective_muscle_excitation(raw, backend=backend)
    correction = effective - raw
    return {
        "effective_excitation": effective,
        "preclip_out_of_range_fraction": backend.mean((raw < 0.0) | (raw > 1.0)),
        "clip_correction_rms": backend.sqrt(backend.mean(backend.square(correction))),
    }


def resolve_muscle_channel_layout(
    model: Any,
    actuator_names: Sequence[str],
    *,
    require_unit_ctrlrange: bool = True,
    require_scalar_activation: bool = True,
) -> MuscleChannelLayout:
    """Resolve and validate exact ordered muscle channels from an ``MjModel``.

    Validation is fail closed:

    * every name must exist exactly once;
    * every selected actuator must use ``mjDYN_MUSCLE``;
    * one scalar activation state is required unconditionally;
    * activation addresses must be unique;
    * physical muscle ctrlrange is exactly ``[0, 1]`` when requested.
    """

    if not require_scalar_activation:
        raise ValueError(
            "physiology muscle channels always require one scalar activation "
            "state; require_scalar_activation cannot be disabled"
        )
    contract: MuscleChannelContract = resolve_muscle_channel_contract(
        model,
        actuator_names,
    )
    names = contract.actuator_names
    actuator_ids = np.asarray(contract.actuator_ids, dtype=np.int32)
    ctrlranges = np.asarray(
        model.actuator_ctrlrange[actuator_ids],
        dtype=np.float64,
    )
    validate_ordered_ctrlrange(names, ctrlranges)
    if require_unit_ctrlrange:
        validate_unit_muscle_ctrlrange(names, ctrlranges)
    model_state = model.__getstate__()
    if not isinstance(model_state, bytes) or not model_state:
        raise ValueError("runtime MuJoCo model has no canonical byte state")
    return MuscleChannelLayout(
        actuator_names=names,
        actuator_ids=actuator_ids,
        activation_addresses=np.asarray(contract.actuator_actadr, dtype=np.int32),
        activation_counts=np.asarray(contract.actuator_actnum, dtype=np.int32),
        ctrlrange=ctrlranges,
        dyntype_ids=np.asarray(
            model.actuator_dyntype[actuator_ids],
            dtype=np.int32,
        ),
        actuator_schema_hash=actuator_schema_hash(names),
        runtime_model_hash=hashlib.sha256(model_state).hexdigest(),
    )


def ordered_body_activation(data: Any, layout: MuscleChannelLayout, *, backend: Any = np) -> Any:
    """Read ordered scalar activation using validated ``actuator_actadr`` values."""

    addresses = backend.asarray(layout.activation_addresses)
    return backend.take(data.act, addresses, axis=-1)


def actuator_transmission_target(model: Any, actuator_id: int) -> dict[str, Any]:
    """Return a stable name-resolved actuator transmission target."""

    import mujoco

    index = int(actuator_id)
    if index < 0 or index >= int(model.nu):
        raise ValueError(f"actuator_id is out of range: {index}")
    transmission_id = int(model.actuator_trntype[index])
    transmission = mujoco.mjtTrn(transmission_id).name
    object_id = int(model.actuator_trnid[index, 0])
    object_type = {
        int(mujoco.mjtTrn.mjTRN_JOINT): ("joint", mujoco.mjtObj.mjOBJ_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): (
            "joint",
            mujoco.mjtObj.mjOBJ_JOINT,
        ),
        int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): ("site", mujoco.mjtObj.mjOBJ_SITE),
        int(mujoco.mjtTrn.mjTRN_TENDON): ("tendon", mujoco.mjtObj.mjOBJ_TENDON),
        int(mujoco.mjtTrn.mjTRN_SITE): ("site", mujoco.mjtObj.mjOBJ_SITE),
        int(mujoco.mjtTrn.mjTRN_BODY): ("body", mujoco.mjtObj.mjOBJ_BODY),
    }.get(transmission_id)
    if object_type is None:
        return {
            "transmission": transmission,
            "transmission_id": transmission_id,
            "object_type": "unresolved",
            "object_id": object_id,
            "name": None,
        }
    object_label, object_enum = object_type
    object_name = None if object_id < 0 else mujoco.mj_id2name(model, object_enum, object_id)
    return {
        "transmission": transmission,
        "transmission_id": transmission_id,
        "object_type": object_label,
        "object_id": object_id,
        "name": None if object_name is None else str(object_name),
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
