"""Train-only normalization and phase balancing for synergy extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from musclemimic.synergy.schema import SignalTransform, validate_nmf_signal

DEFAULT_PHASE_WEIGHTS = {0: 1.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 2.0, 5: 1.0}


@dataclass(frozen=True)
class PreprocessState:
    muscle_names: tuple[str, ...]
    kept_indices: np.ndarray
    scales: np.ndarray
    normalization: str
    near_zero_threshold: float

    @property
    def kept_muscle_names(self) -> tuple[str, ...]:
        return tuple(self.muscle_names[int(index)] for index in self.kept_indices)

    def to_manifest(self) -> dict:
        return {
            "normalization": self.normalization,
            "near_zero_threshold": float(self.near_zero_threshold),
            "source_muscle_names": list(self.muscle_names),
            "kept_indices": self.kept_indices.astype(int).tolist(),
            "kept_muscle_names": list(self.kept_muscle_names),
            "scales": self.scales.astype(float).tolist(),
            "fit_scope": "train_only",
        }


def fit_preprocessor(
    train_values: np.ndarray,
    *,
    muscle_names: Sequence[str],
    signal_kind: str,
    transform: SignalTransform | Mapping | None = None,
    normalization: str = "channel_max",
    near_zero_threshold: float = 1e-8,
) -> tuple[np.ndarray, PreprocessState]:
    values = validate_nmf_signal(
        train_values,
        signal_kind=signal_kind,
        muscle_names=muscle_names,
        transform=transform,
    )
    peak = np.max(values, axis=0)
    kept = np.flatnonzero(peak > float(near_zero_threshold)).astype(np.int32)
    if kept.size == 0:
        raise ValueError("all muscle channels are near-zero")
    selected = values[:, kept]
    if normalization == "channel_max":
        scales = np.max(selected, axis=0)
    elif normalization == "channel_l2":
        scales = np.linalg.norm(selected, axis=0)
    elif normalization == "none":
        scales = np.ones(selected.shape[1], dtype=np.float64)
    else:
        raise ValueError("normalization must be channel_max, channel_l2, or none")
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("invalid train-only normalization scales")
    state = PreprocessState(
        muscle_names=tuple(str(name) for name in muscle_names),
        kept_indices=kept,
        scales=np.asarray(scales, dtype=np.float64),
        normalization=normalization,
        near_zero_threshold=float(near_zero_threshold),
    )
    return (selected / scales).astype(np.float64), state


def apply_preprocessor(
    values: np.ndarray,
    state: PreprocessState,
    *,
    signal_kind: str,
    transform: SignalTransform | Mapping | None = None,
) -> np.ndarray:
    validated = validate_nmf_signal(
        values,
        signal_kind=signal_kind,
        muscle_names=state.muscle_names,
        transform=transform,
    )
    return validated[:, state.kept_indices] / state.scales


def phase_balanced_weights(
    phase_id: np.ndarray,
    *,
    weights: Mapping[int, float] | None = None,
) -> np.ndarray:
    phases = np.asarray(phase_id)
    if phases.ndim != 1:
        raise ValueError("phase_id must be rank-1")
    mapping = {int(key): float(value) for key, value in (weights or DEFAULT_PHASE_WEIGHTS).items()}
    if not mapping or any(not np.isfinite(value) or value <= 0.0 for value in mapping.values()):
        raise ValueError("phase weights must be finite and positive")
    missing = sorted({int(value) for value in np.unique(phases)} - set(mapping))
    if missing:
        raise ValueError(f"phase weights missing phase ids {missing}")
    sample = np.asarray([mapping[int(value)] for value in phases], dtype=np.float64)
    return sample / np.mean(sample)


def apply_sample_weights(values: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (array.shape[0],) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample_weights must be finite positive [samples]")
    return array * np.sqrt(weights)[:, None]
