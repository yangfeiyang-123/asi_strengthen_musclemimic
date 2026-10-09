from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, sosfilt, sosfiltfilt

from .models import ProcessingConfig


def _validate_emg(emg_mV: np.ndarray) -> np.ndarray:
    emg = np.asarray(emg_mV, dtype=np.float64)
    if emg.ndim != 2 or emg.shape[0] < 2 or emg.shape[1] < 1:
        raise ValueError("EMG must have shape [samples, channels]")
    return emg


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1), constant_values=False)
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def repair_missing_values(
    emg_mV: np.ndarray,
    config: ProcessingConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Repair non-finite samples independently in each channel.

    Linear interpolation retains the original sample grid. Long gaps are still
    filled so downstream filters remain numerically defined, but are recorded as
    quality failures rather than silently accepted.
    """

    emg = _validate_emg(emg_mV).copy()
    sample_axis = np.arange(emg.shape[0], dtype=np.float64)
    max_gap_samples = int(round(config.max_interpolation_gap_s * config.sample_rate_hz))
    reports: list[dict[str, Any]] = []
    for channel in range(emg.shape[1]):
        values = emg[:, channel]
        missing = ~np.isfinite(values)
        runs = _true_runs(missing)
        count = int(np.sum(missing))
        if count:
            if config.missing_value_policy == "raise":
                raise ValueError(f"Channel {channel + 1} contains {count} non-finite samples")
            good = ~missing
            if not np.any(good):
                raise ValueError(f"Channel {channel + 1} contains no finite samples")
            values[missing] = np.interp(sample_axis[missing], sample_axis[good], values[good])
        longest = max((stop - start for start, stop in runs), default=0)
        reports.append(
            {
                "channel_index": channel,
                "missing_samples": count,
                "missing_fraction": count / len(values),
                "missing_runs": len(runs),
                "longest_missing_run_samples": longest,
                "longest_missing_run_s": longest / config.sample_rate_hz,
                "long_gap_detected": bool(longest > max_gap_samples),
                "exceeds_missing_fraction": bool(count / len(values) > config.max_missing_fraction),
            }
        )
    return emg, reports


def repair_isolated_outliers(
    emg_mV: np.ndarray,
    config: ProcessingConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Detect robust-amplitude outliers and interpolate only short runs.

    Longer runs are retained and reported because replacing sustained high EMG
    activation would risk deleting physiology. This function is intentionally
    conservative and channel independent.
    """

    emg = _validate_emg(emg_mV).copy()
    sample_axis = np.arange(emg.shape[0], dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for channel in range(emg.shape[1]):
        values = emg[:, channel]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= np.finfo(float).eps:
            robust_sigma = float(np.std(values))
        candidates = (
            np.abs(values - median) > config.outlier_mad_threshold * robust_sigma
            if robust_sigma > np.finfo(float).eps
            else np.zeros(len(values), dtype=bool)
        )
        candidate_runs = _true_runs(candidates)
        replace_mask = np.zeros(len(values), dtype=bool)
        for start, stop in candidate_runs:
            if stop - start <= config.max_outlier_run_samples:
                replace_mask[start:stop] = True
        candidate_count = int(np.sum(candidates))
        if candidate_count and config.outlier_policy == "raise":
            raise ValueError(f"Channel {channel + 1} contains {candidate_count} robust outliers")
        if config.outlier_policy == "interpolate" and np.any(replace_mask):
            good = ~replace_mask
            values[replace_mask] = np.interp(sample_axis[replace_mask], sample_axis[good], values[good])
        replaced = int(np.sum(replace_mask)) if config.outlier_policy == "interpolate" else 0
        reports.append(
            {
                "channel_index": channel,
                "robust_sigma_mV": robust_sigma,
                "outlier_candidates": candidate_count,
                "outlier_fraction": candidate_count / len(values),
                "outlier_runs": len(candidate_runs),
                "outliers_interpolated": replaced,
                "outliers_retained": candidate_count - replaced,
            }
        )
    return emg, reports


def remove_dc_offset(emg_mV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    emg = _validate_emg(emg_mV)
    means = np.mean(emg, axis=0, dtype=np.float64)
    return emg - means[None, :], means


def _ensure_filterable(emg: np.ndarray, config: ProcessingConfig) -> None:
    duration_s = len(emg) / config.sample_rate_hz
    if duration_s < config.minimum_duration_s:
        raise ValueError(
            f"EMG duration {duration_s:.4f}s is shorter than configured minimum "
            f"{config.minimum_duration_s:.4f}s"
        )
    if not np.all(np.isfinite(emg)):
        raise ValueError("Filtering input must be finite after missing-value repair")


def _apply_sos(sos: np.ndarray, values: np.ndarray, zero_phase: bool, label: str) -> np.ndarray:
    try:
        return sosfiltfilt(sos, values, axis=0, padtype="odd") if zero_phase else sosfilt(sos, values, axis=0)
    except ValueError as exc:
        raise ValueError(f"{label} failed; signal is too short for filter edge padding: {exc}") from exc


def bandpass_filter(emg_mV: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    emg = _validate_emg(emg_mV)
    _ensure_filterable(emg, config)
    sos = butter(
        config.filter_order,
        [config.bandpass_low_hz, config.bandpass_high_hz],
        btype="bandpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    return _apply_sos(sos, emg, config.zero_phase, "Band-pass filtering")


def notch_filter(emg_mV: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    emg = _validate_emg(emg_mV)
    if config.notch_hz is None:
        return emg.copy()
    quality_factor = config.effective_notch_quality_factor
    if quality_factor is None:
        return emg.copy()
    b, a = iirnotch(config.notch_hz, quality_factor, fs=config.sample_rate_hz)
    try:
        return filtfilt(b, a, emg, axis=0, padtype="odd") if config.zero_phase else lfilter(b, a, emg, axis=0)
    except ValueError as exc:
        raise ValueError(f"Notch filtering failed; signal is too short for filter edge padding: {exc}") from exc


def full_wave_rectify(emg_mV: np.ndarray) -> np.ndarray:
    return np.abs(_validate_emg(emg_mV))


def lowpass_envelope(rectified_mV: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    rectified = _validate_emg(rectified_mV)
    sos = butter(
        config.filter_order,
        config.envelope_lowpass_hz,
        btype="lowpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    envelope = _apply_sos(sos, rectified, config.zero_phase, "Envelope low-pass filtering")
    return np.maximum(envelope, 0.0)


def preprocess_signal_stages(emg_mV: np.ndarray, config: ProcessingConfig) -> dict[str, Any]:
    raw = _validate_emg(emg_mV)
    repaired, missing_report = repair_missing_values(raw, config)
    cleaned, outlier_report = repair_isolated_outliers(repaired, config)
    demeaned, removed_means = remove_dc_offset(cleaned)
    bandpassed = bandpass_filter(demeaned, config)
    notched = notch_filter(bandpassed, config)
    rectified = full_wave_rectify(notched)
    envelope = lowpass_envelope(rectified, config)
    return {
        "raw_emg_mV": raw.astype(np.float32, copy=False),
        "cleaned_emg_mV": cleaned.astype(np.float32),
        "demeaned_mV": demeaned.astype(np.float32),
        "bandpassed_mV": bandpassed.astype(np.float32),
        "notched_mV": notched.astype(np.float32),
        "filtered_mV": notched.astype(np.float32),
        "rectified_mV": rectified.astype(np.float32),
        "envelope_mV": envelope.astype(np.float32),
        "removed_channel_means_mV": removed_means.tolist(),
        "missing_value_report": missing_report,
        "outlier_report": outlier_report,
        "edge_guard_samples": int(round(config.edge_guard_s * config.sample_rate_hz)),
    }


def emg_envelope(
    emg_mV: np.ndarray,
    config: ProcessingConfig,
    envelope_lowpass_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    effective = replace(config, envelope_lowpass_hz=envelope_lowpass_hz) if envelope_lowpass_hz is not None else config
    stages = preprocess_signal_stages(emg_mV, effective)
    return stages["filtered_mV"], stages["envelope_mV"]


def stable_window_max(envelope: np.ndarray, fs_hz: float, window_s: float = 0.5) -> np.ndarray:
    values = np.asarray(envelope, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    window = max(1, int(round(window_s * fs_hz)))
    if values.shape[0] < window:
        raise ValueError("MVC recording is shorter than the stable averaging window")
    kernel = np.ones(window, dtype=np.float64) / window
    result = [np.max(np.convolve(values[:, channel], kernel, mode="valid")) for channel in range(values.shape[1])]
    return np.asarray(result, dtype=np.float64)


def normalize_envelope(envelope_mV: np.ndarray, mvc_values_mV: np.ndarray) -> np.ndarray:
    envelope = _validate_emg(envelope_mV)
    mvc = np.asarray(mvc_values_mV, dtype=np.float64).reshape(1, -1)
    if mvc.shape[1] != envelope.shape[1] or np.any(~np.isfinite(mvc)) or np.any(mvc <= 0):
        raise ValueError("MVC values must be finite, positive, and match channel count")
    return np.maximum(envelope / mvc, 0.0)


def preprocess_emg(
    emg_mV: np.ndarray,
    config: ProcessingConfig,
    mvc_values: np.ndarray | None = None,
    explicit_fallback: str | None = None,
    dynamic_reference_values: np.ndarray | None = None,
) -> dict[str, Any]:
    stages = preprocess_signal_stages(emg_mV, config)
    envelope = np.asarray(stages["envelope_mV"], dtype=np.float64)
    method = config.normalization
    fallback_used: str | None = None
    if method == "mvc" and mvc_values is None:
        if explicit_fallback not in {"dynamic_p95", "none"}:
            raise FileNotFoundError(
                "MVC normalization requested but no compatible MVC values were found; "
                "pass an explicit fallback method to continue"
            )
        method = explicit_fallback
        fallback_used = explicit_fallback
    if method == "mvc":
        normalized = normalize_envelope(envelope, np.asarray(mvc_values))
        reference_values = np.asarray(mvc_values, dtype=np.float64).reshape(-1)
    elif method == "dynamic_p95":
        denominator = (
            np.asarray(dynamic_reference_values, dtype=np.float64).reshape(1, -1)
            if dynamic_reference_values is not None
            else np.percentile(envelope, 95, axis=0, keepdims=True)
        )
        if denominator.shape[1] != envelope.shape[1] or np.any(~np.isfinite(denominator)):
            raise ValueError("Dynamic p95 reference must be finite and match the EMG channel count")
        denominator = np.maximum(denominator, np.finfo(float).eps)
        normalized = np.maximum(envelope / denominator, 0.0)
        reference_values = denominator.reshape(-1)
    elif method == "none":
        normalized = envelope.copy()
        reference_values = None
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    stages.update(
        {
            "normalized_envelope": normalized.astype(np.float32),
            "normalization_method": method,
            "fallback_method": fallback_used,
            "normalization_reference_values_mV": (
                None if reference_values is None else reference_values.tolist()
            ),
        }
    )
    return stages


def with_normalization(config: ProcessingConfig, normalization: str) -> ProcessingConfig:
    return replace(config, normalization=normalization)  # type: ignore[arg-type]
