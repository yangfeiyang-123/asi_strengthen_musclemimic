"""Pure helpers for constructing scientifically explicit synergy signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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
    tolerance: float = 1e-6,
) -> SynergySignal:
    """Map applied control to [0,1] with an auditable per-actuator affine map."""

    names = tuple(str(name) for name in actuator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("actuator_names must be non-empty and unique")
    raw = np.asarray(applied_ctrl, dtype=np.float64)
    limits = np.asarray(ctrlrange, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError(f"applied_ctrl must have shape [samples,{len(names)}], got {raw.shape}")
    if limits.shape != (len(names), 2) or not np.all(np.isfinite(limits)):
        raise ValueError("ctrlrange must have finite shape [actuators,2]")
    if not np.all(np.isfinite(raw)) or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("applied control or ctrlrange is invalid")
    low, high = limits[:, 0], limits[:, 1]
    if np.any(raw < low - tolerance) or np.any(raw > high + tolerance):
        raise ValueError("applied_ctrl lies outside name-aligned ctrlrange")
    unit = (raw - low) / (high - low)
    # Only correct floating roundoff that already passed the explicit raw-range
    # check above.  Scientific evidence outside ctrlrange is never clipped.
    unit = np.where(unit < 0.0, 0.0, np.where(unit > 1.0, 1.0, unit))
    transform = SignalTransform(
        kind="ctrlrange_affine_to_unit",
        raw_signal_kind="applied_ctrl",
        formula="(ctrl-low)/(high-low)",
        ctrlrange=limits,
        actuator_names=names,
        ctrlrange_schema_hash=ctrlrange_schema_hash(names, limits),
        roundoff_policy="fail_outside_ctrlrange_then_clamp_within_tolerance_only",
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
