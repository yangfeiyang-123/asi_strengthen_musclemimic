from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from .models import ChannelProfile, ProcessingConfig


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _power_ratio(values: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    if len(centered) < 64 or np.allclose(centered, 0.0):
        return 0.0
    frequencies, power = welch(centered, fs=fs_hz, nperseg=min(4096, len(centered)))
    total_mask = (frequencies >= 1.0) & (frequencies <= min(500.0, fs_hz / 2.0))
    selected_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    total = float(np.sum(power[total_mask]))
    return float(np.sum(power[selected_mask]) / total) if total > 0 else 0.0


def assess_preprocessing_quality(
    raw_emg_mV: np.ndarray,  # noqa: N803 - public API preserves physical-unit spelling
    bandpassed_mV: np.ndarray,  # noqa: N803
    filtered_mV: np.ndarray,  # noqa: N803
    envelope_mV: np.ndarray,  # noqa: N803
    normalized_envelope: np.ndarray,
    missing_report: list[dict[str, Any]],
    outlier_report: list[dict[str, Any]],
    config: ProcessingConfig,
    profile: ChannelProfile,
    normalization_method: str = "mvc",
) -> dict[str, Any]:
    arrays = [np.asarray(item, dtype=np.float64) for item in (
        raw_emg_mV, bandpassed_mV, filtered_mV, envelope_mV, normalized_envelope
    )]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays) or shape[1] != len(profile.channels):
        raise ValueError("Preprocessing QC arrays and channel profile must have identical shape")
    guard = round(config.edge_guard_s * config.sample_rate_hz)
    interior = slice(guard, -guard) if guard > 0 and len(arrays[0]) > 2 * guard + 16 else slice(None)
    channels: list[dict[str, Any]] = []
    critical_channel_count = 0
    for index, channel in enumerate(profile.channels):
        raw = arrays[0][interior, index]
        bandpassed = arrays[1][interior, index]
        filtered = arrays[2][interior, index]
        envelope = arrays[3][interior, index]
        normalized = arrays[4][interior, index]
        bandpassed_variance = float(np.var(bandpassed))
        filtered_variance = float(np.var(filtered))
        removed_fraction = (
            float(np.clip(1.0 - filtered_variance / bandpassed_variance, 0.0, 1.0))
            if bandpassed_variance > np.finfo(float).eps else 0.0
        )
        powerline_half_band = config.notch_bandwidth_hz / 2.0
        raw_powerline = _power_ratio(
            raw, config.sample_rate_hz,
            (config.notch_hz or 50.0) - powerline_half_band,
            (config.notch_hz or 50.0) + powerline_half_band,
        )
        filtered_powerline = _power_ratio(
            filtered, config.sample_rate_hz,
            (config.notch_hz or 50.0) - powerline_half_band,
            (config.notch_hz or 50.0) + powerline_half_band,
        )
        warnings: list[str] = []
        critical: list[str] = []
        missing = missing_report[index]
        outliers = outlier_report[index]
        if missing["missing_samples"]:
            warnings.append("missing_values_interpolated")
        if missing["long_gap_detected"]:
            critical.append("missing_gap_too_long")
        if missing["exceeds_missing_fraction"]:
            critical.append("missing_fraction_too_high")
        if outliers["outliers_interpolated"]:
            warnings.append("isolated_outliers_interpolated")
        if outliers["outliers_retained"]:
            warnings.append("sustained_outliers_retained")
        filtered_rms = _rms(filtered)
        if np.std(raw) <= 1e-5:
            critical.append("raw_flatline")
        if raw_powerline > config.powerline_ratio_warn:
            warnings.append("raw_powerline_contamination")
        if removed_fraction > config.max_notch_removed_variance_fraction:
            critical.append("signal_dominated_by_notched_powerline")
        if filtered_rms < config.minimum_filtered_rms_mV:
            critical.append("filtered_signal_near_flatline")
        if normalization_method == "mvc" and normalized.size and float(np.max(normalized)) > 2.0:
            warnings.append("normalized_envelope_exceeds_200pct_mvc")
            warnings.append("mvc_reference_may_be_underestimated")
        if critical:
            critical_channel_count += 1
        channels.append(
            {
                "channel_index": index,
                "sensor_id": channel.sensor_id,
                "side": channel.side,
                "muscle_slug": channel.muscle_slug,
                "channel_name": f"S{channel.sensor_id} {channel.side}:{channel.muscle_slug}",
                "raw_mean_mV": float(np.mean(raw)),
                "raw_rms_mV": _rms(raw),
                "filtered_rms_mV": filtered_rms,
                "envelope_mean_mV": float(np.mean(envelope)),
                "envelope_peak_mV": float(np.max(envelope)),
                "normalized_peak_mvc": float(np.max(normalized)),
                "normalized_p95_mvc": float(np.percentile(normalized, 95.0)),
                "normalized_p99_mvc": float(np.percentile(normalized, 99.0)),
                "mvc_exceedance_is_critical": False,
                "raw_powerline_ratio": raw_powerline,
                "filtered_powerline_ratio": filtered_powerline,
                "notch_removed_variance_fraction": removed_fraction,
                "missing_values": missing,
                "outliers": outliers,
                "warnings": sorted(set(warnings)),
                "critical_failures": sorted(set(critical)),
                "analysis_ready": not critical,
            }
        )
    return {
        "qc_pass": critical_channel_count == 0,
        "analysis_ready": critical_channel_count == 0,
        "mvc_exceedance_policy": {
            "schema_version": "mvc_exceedance_dual_track_qc_v1",
            "signal_quality_separate_from_mvc_reference_quality": True,
            "percent_mvc_unclipped": True,
            "exceedance_alone_is_critical": False,
            "reported_statistics": ["p95", "p99", "max"],
        },
        "critical_channel_count": critical_channel_count,
        "channel_count": len(channels),
        "edge_guard_samples": guard,
        "edge_guard_s": config.edge_guard_s,
        "interpretation": (
            "Preprocessing completed and no critical channel-level failure was detected."
            if critical_channel_count == 0
            else "Preprocessing completed, but one or more channels are not safe for analysis."
        ),
        "channels": channels,
    }
