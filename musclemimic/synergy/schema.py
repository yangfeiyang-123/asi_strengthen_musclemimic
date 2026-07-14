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
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    UNIT_INTERVAL_TOLERANCE,
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

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "raw_signal_kind": self.raw_signal_kind,
            "formula": self.formula,
            "ctrlrange": None if self.ctrlrange is None else np.asarray(self.ctrlrange).tolist(),
            "actuator_names": list(self.actuator_names),
            "ctrlrange_schema_hash": self.ctrlrange_schema_hash,
            "roundoff_policy": self.roundoff_policy,
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
            "first apply an explicit name-aligned ctrlrange affine transform"
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
    if transform is None or transform.kind != "ctrlrange_affine_to_unit":
        raise ValueError("physical_excitation_unit requires transform.kind='ctrlrange_affine_to_unit'")
    if transform.raw_signal_kind not in {"applied_ctrl", "teacher_ctrl_physical", "raw_ctrl"}:
        raise ValueError("excitation transform must identify the raw applied control signal")
    if transform.formula != "(ctrl-low)/(high-low)":
        raise ValueError("unsupported excitation transform formula")
    if transform.roundoff_policy != "fail_outside_ctrlrange_then_clamp_within_tolerance_only":
        raise ValueError("excitation transform must declare fail-closed roundoff handling")
    if tuple(transform.actuator_names) != names:
        raise ValueError("excitation transform actuator_names/order mismatch")
    ctrlrange = np.asarray(transform.ctrlrange, dtype=np.float64)
    if ctrlrange.shape != (len(names), 2) or not np.all(np.isfinite(ctrlrange)):
        raise ValueError("excitation transform ctrlrange shape/content is invalid")
    if np.any(ctrlrange[:, 1] <= ctrlrange[:, 0]):
        raise ValueError("excitation transform ctrlrange must have positive width")
    expected = ctrlrange_schema_hash(names, ctrlrange)
    if transform.ctrlrange_schema_hash != expected:
        raise ValueError("excitation transform ctrlrange_schema_hash mismatch")


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
