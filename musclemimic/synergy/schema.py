"""Strict schemas for signals admitted to non-negative synergy fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    UNIT_INTERVAL_TOLERANCE,
    validate_muscle_channel_contract,
    validate_unit_muscle_ctrlrange,
)

EXCITATION_SIGNAL_KIND = "physical_excitation_unit"
ACTIVATION_SIGNAL_KIND = "muscle_activation"
SIGNED_CONTROL_KINDS = frozenset(
    {
        "raw_ctrl",
        "applied_ctrl",
        "teacher_ctrl_physical",
        "signed_ctrl",
        "teacher_action_normalized",
    }
)


@dataclass(frozen=True)
class SignalTransform:
    kind: str
    raw_signal_kind: str
    formula: str
    ctrlrange: np.ndarray | None = None
    actuator_names: tuple[str, ...] = ()
    ctrlrange_schema_hash: str | None = None
    roundoff_policy: str = "none"
    physical_signal_schema_version: str | None = None
    muscle_channel_contract: Mapping[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "raw_signal_kind": self.raw_signal_kind,
            "formula": self.formula,
            "ctrlrange": None if self.ctrlrange is None else np.asarray(self.ctrlrange).tolist(),
            "actuator_names": list(self.actuator_names),
            "ctrlrange_schema_hash": self.ctrlrange_schema_hash,
            "roundoff_policy": self.roundoff_policy,
            "physical_signal_schema_version": self.physical_signal_schema_version,
            "muscle_channel_contract": (
                None if self.muscle_channel_contract is None else dict(self.muscle_channel_contract)
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> SignalTransform | None:
        if payload is None:
            return None
        ctrlrange = payload.get("ctrlrange")
        return cls(
            kind=str(payload.get("kind", "")),
            raw_signal_kind=str(payload.get("raw_signal_kind", "")),
            formula=str(payload.get("formula", "")),
            ctrlrange=None if ctrlrange is None else np.asarray(ctrlrange, dtype=np.float64),
            actuator_names=tuple(str(name) for name in payload.get("actuator_names", ())),
            ctrlrange_schema_hash=(
                None if payload.get("ctrlrange_schema_hash") is None else str(payload["ctrlrange_schema_hash"])
            ),
            roundoff_policy=str(payload.get("roundoff_policy", "none")),
            physical_signal_schema_version=(
                None
                if payload.get("physical_signal_schema_version") is None
                else str(payload["physical_signal_schema_version"])
            ),
            muscle_channel_contract=(
                None if payload.get("muscle_channel_contract") is None else dict(payload["muscle_channel_contract"])
            ),
        )


@dataclass(frozen=True)
class SynergySignal:
    values: np.ndarray
    muscle_names: tuple[str, ...]
    signal_kind: str
    transform: SignalTransform | None = None

    def validated(self) -> SynergySignal:
        values = validate_nmf_signal(
            self.values,
            signal_kind=self.signal_kind,
            muscle_names=self.muscle_names,
            transform=self.transform,
        )
        return SynergySignal(
            values=values,
            muscle_names=tuple(self.muscle_names),
            signal_kind=self.signal_kind,
            transform=self.transform,
        )


def validate_nmf_signal(
    values: np.ndarray,
    *,
    signal_kind: str,
    muscle_names: Sequence[str],
    transform: SignalTransform | Mapping[str, Any] | None = None,
    tolerance: float = UNIT_INTERVAL_TOLERANCE,
) -> np.ndarray:
    """Validate signal semantics; signed controls are never silently clipped."""

    kind = str(signal_kind)
    if kind in SIGNED_CONTROL_KINDS:
        raise ValueError(
            f"signed/raw control signal {kind!r} cannot be used for NMF; "
            "first verify scalar MuJoCo muscle channels and apply clip(raw_ctrl,0,1)"
        )
    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("muscle_names must be non-empty and unique")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(f"synergy signal must be [samples,{len(names)}], got {array.shape}")
    if array.shape[0] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("synergy signal must contain at least two finite samples")
    if np.min(array) < -float(tolerance):
        raise ValueError("NMF signal contains negative values; clipping is forbidden")
    if kind == EXCITATION_SIGNAL_KIND:
        parsed = transform if isinstance(transform, SignalTransform) else SignalTransform.from_mapping(transform)
        _validate_excitation_transform(parsed, names)
        if np.max(array) > 1.0 + float(tolerance):
            raise ValueError("physical_excitation_unit must lie in [0,1]")
    elif kind == ACTIVATION_SIGNAL_KIND:
        parsed = transform if isinstance(transform, SignalTransform) else SignalTransform.from_mapping(transform)
        _validate_activation_transform(parsed, names)
        if np.max(array) > 1.0 + float(tolerance):
            raise ValueError("muscle_activation must lie in [0,1]")
    else:
        raise ValueError(
            f"unsupported synergy signal_kind {kind!r}; expected "
            f"{EXCITATION_SIGNAL_KIND!r} or {ACTIVATION_SIGNAL_KIND!r}"
        )
    # Remove only unit-interval floating roundoff after semantic validation.
    return np.clip(array, 0.0, 1.0)


def ctrlrange_schema_hash(actuator_names: Sequence[str], ctrlrange: np.ndarray) -> str:
    payload = {
        "kind": "actuator_ctrlrange",
        "actuator_names": [str(name) for name in actuator_names],
        "ctrlrange": np.asarray(ctrlrange, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_excitation_transform(
    transform: SignalTransform | None,
    names: tuple[str, ...],
) -> None:
    if transform is None or transform.kind != UNIT_EXCITATION_TRANSFORM:
        raise ValueError("physical_excitation_unit requires the v2 verified-muscle raw-ctrl clipping transform")
    if transform.raw_signal_kind not in {"applied_ctrl", "teacher_ctrl_physical", "raw_ctrl"}:
        raise ValueError("excitation transform must identify the raw applied control signal")
    if transform.formula != MUSCLE_EXCITATION_FORMULA:
        raise ValueError("unsupported excitation transform formula")
    if transform.roundoff_policy != MUSCLE_EXCITATION_ROUNDOFF_POLICY:
        raise ValueError("excitation transform must declare exact MuJoCo muscle clipping semantics")
    if transform.physical_signal_schema_version != PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("excitation transform must bind the physical muscle signal v2 schema")
    if tuple(transform.actuator_names) != names:
        raise ValueError("excitation transform actuator_names/order mismatch")
    ctrlrange = validate_unit_muscle_ctrlrange(names, transform.ctrlrange)
    expected = ctrlrange_schema_hash(names, ctrlrange)
    if transform.ctrlrange_schema_hash != expected:
        raise ValueError("excitation transform ctrlrange_schema_hash mismatch")
    validate_muscle_channel_contract(
        transform.muscle_channel_contract,
        expected_names=names,
    )


def _validate_activation_transform(
    transform: SignalTransform | None,
    names: tuple[str, ...],
) -> None:
    if transform is None or transform.kind != "identity_nonnegative_activation":
        raise ValueError("muscle_activation requires transform.kind='identity_nonnegative_activation'")
    if transform.raw_signal_kind != MUSCLE_ACTIVATION_SOURCE:
        raise ValueError(
            "muscle_activation transform must identify transition_state.data.act via model.actuator_actadr"
        )
    if transform.formula != "activation":
        raise ValueError("muscle_activation transform formula must be 'activation'")
    if tuple(transform.actuator_names) != names:
        raise ValueError("muscle_activation transform actuator_names/order mismatch")
    if transform.roundoff_policy != UNIT_INTERVAL_ROUNDOFF_POLICY:
        raise ValueError("muscle_activation transform roundoff policy is unsupported")
    if transform.physical_signal_schema_version != PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("muscle_activation transform must bind the physical muscle signal v2 schema")
    validate_muscle_channel_contract(
        transform.muscle_channel_contract,
        expected_names=names,
    )
