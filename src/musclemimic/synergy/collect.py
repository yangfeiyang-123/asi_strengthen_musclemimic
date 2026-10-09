"""Pure helpers for constructing scientifically explicit synergy signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    physical_ctrl_to_effective_muscle_excitation,
    validate_muscle_channel_contract,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.synergy.schema import (
    EXCITATION_SIGNAL_KIND,
    SignalTransform,
    SynergySignal,
    ctrlrange_schema_hash,
)


def ctrl_to_unit_excitation(
    applied_ctrl: np.ndarray,
    *,
    ctrlrange: np.ndarray,
    actuator_names: Sequence[str],
    muscle_channel_contract: Mapping[str, Any],
    tolerance: float = 1e-6,
) -> SynergySignal:
    """Derive effective excitation from verified raw MuJoCo muscle control."""

    names = tuple(str(name) for name in actuator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("actuator_names must be non-empty and unique")
    raw = np.asarray(applied_ctrl, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError(f"applied_ctrl must have shape [samples,{len(names)}], got {raw.shape}")
    if not np.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    limits = validate_unit_muscle_ctrlrange(names, ctrlrange)
    contract = validate_muscle_channel_contract(
        muscle_channel_contract,
        expected_names=names,
    )
    unit = physical_ctrl_to_effective_muscle_excitation(
        raw,
        channel_contract=contract,
    )
    transform = SignalTransform(
        kind=UNIT_EXCITATION_TRANSFORM,
        raw_signal_kind="applied_ctrl",
        formula=MUSCLE_EXCITATION_FORMULA,
        ctrlrange=limits,
        actuator_names=names,
        ctrlrange_schema_hash=ctrlrange_schema_hash(names, limits),
        roundoff_policy=MUSCLE_EXCITATION_ROUNDOFF_POLICY,
        physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
        muscle_channel_contract=contract.to_metadata(),
    )
    return SynergySignal(
        values=unit.astype(np.float32),
        muscle_names=names,
        signal_kind=EXCITATION_SIGNAL_KIND,
        transform=transform,
    ).validated()


def select_named_channels(
    values: np.ndarray,
    *,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    source = tuple(str(name) for name in source_names)
    target = tuple(str(name) for name in target_names)
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("source and target names must be unique")
    missing = [name for name in target if name not in source]
    if missing:
        raise ValueError(f"target muscle names are absent from source schema: {missing}")
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(source):
        raise ValueError("values/source_names dimension mismatch")
    return array[:, [source.index(name) for name in target]]
