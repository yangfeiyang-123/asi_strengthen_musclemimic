from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.signal import welch

import matplotlib.pyplot as plt

from .models import ChannelProfile


@dataclass(frozen=True)
class QCThresholds:
    flatline_std_mV: float = 1e-5
    baseline_rms_max_mV: float = 0.08
    action_to_baseline_ratio_min: float = 1.5
    clipping_fraction_warn: float = 0.01
    spike_z_warn: float = 12.0
    powerline_ratio_warn: float = 0.2
    low_frequency_ratio_warn: float = 0.35
    high_correlation_warn: float = 0.98
    sample_tolerance: int = 0


def _band_power_ratio(x: np.ndarray, fs_hz: float, low: float, high: float) -> float:
    if len(x) < 64 or np.allclose(x, x[0]):
        return 0.0
    freqs, power = welch(x, fs=fs_hz, nperseg=min(1024, len(x)))
    total = float(np.sum(power[(freqs >= 1) & (freqs <= min(500, fs_hz / 2))]))
    selected = float(np.sum(power[(freqs >= low) & (freqs <= high)]))
    return selected / total if total > 0 else 0.0


def assess_signal_quality(
    emg_mV: np.ndarray,
    fs_hz: float,
    expected_samples: int,
    baseline_samples: int,
    action_start_sample: int,
    action_end_sample: int,
    thresholds: QCThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or QCThresholds()
    emg = np.asarray(emg_mV, dtype=np.float64)
    if emg.ndim != 2:
        raise ValueError("EMG QC expects [samples, channels]")
    baseline_end = min(max(baseline_samples, 1), len(emg))
    action_start = min(max(action_start_sample, 0), len(emg))
    action_end = min(max(action_end_sample, action_start + 1), len(emg))
    baseline = emg[:baseline_end]
    action = emg[action_start:action_end]
    channel_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for channel in range(emg.shape[1]):
        x = emg[:, channel]
        finite = np.isfinite(x)
        finite_x = x[finite]
        std = float(np.std(finite_x)) if finite_x.size else float("nan")
        zero = bool(finite_x.size == 0 or np.all(finite_x == 0))
        flatline = bool(not np.isfinite(std) or std <= limits.flatline_std_mV)
        base_rms = float(np.sqrt(np.nanmean(baseline[:, channel] ** 2))) if baseline.size else float("nan")
        action_rms = float(np.sqrt(np.nanmean(action[:, channel] ** 2))) if action.size else float("nan")
        ratio = action_rms / max(base_rms, np.finfo(float).eps)
        abs_x = np.abs(finite_x)
        maximum = float(np.max(abs_x)) if abs_x.size else 0.0
        clip_fraction = float(np.mean(np.isclose(abs_x, maximum, rtol=1e-6, atol=1e-8))) if maximum else 0.0
        # Compare to the trial's active-signal scale. A global MAD is dominated by
        # quiet baseline and falsely labels ordinary movement activation as spikes.
        active_scale = float(np.percentile(abs_x, 95)) if abs_x.size else 0.0
        spike_fraction = (
            float(np.mean(abs_x > limits.spike_z_warn * max(active_scale, 1e-9)))
            if abs_x.size else 0.0
        )
        powerline = _band_power_ratio(np.nan_to_num(x), fs_hz, 49.0, 51.0)
        low_frequency = _band_power_ratio(np.nan_to_num(x), fs_hz, 1.0, 15.0)
        channel_warnings: list[str] = []
        if zero:
            channel_warnings.append("all_zero")
        if flatline:
            channel_warnings.append("flatline")
        if not np.all(finite):
            channel_warnings.append("nan_or_inf")
        if base_rms > limits.baseline_rms_max_mV:
            channel_warnings.append("high_baseline_rms")
        if ratio < limits.action_to_baseline_ratio_min:
            channel_warnings.append("low_action_to_baseline_ratio")
        if clip_fraction > limits.clipping_fraction_warn:
            channel_warnings.append("possible_clipping")
        if spike_fraction > 0:
            channel_warnings.append("extreme_spikes")
        if powerline > limits.powerline_ratio_warn:
            channel_warnings.append("50hz_contamination")
        if low_frequency > limits.low_frequency_ratio_warn:
            channel_warnings.append("low_frequency_motion_artifact")
        warnings.extend(f"channel_{channel + 1}:{warning}" for warning in channel_warnings)
        channel_results.append(
            {
                "column_index": channel,
                "all_zero": zero,
                "flatline": flatline,
                "finite": bool(np.all(finite)),
                "baseline_rms_mV": base_rms,
                "action_rms_mV": action_rms,
                "action_to_baseline_ratio": ratio,
                "clipping_fraction": clip_fraction,
                "spike_fraction": spike_fraction,
                "power_50hz_ratio": powerline,
                "low_frequency_power_ratio": low_frequency,
                "warnings": channel_warnings,
            }
        )
    high_corr_pairs: list[dict[str, Any]] = []
    if len(emg) > 2 and emg.shape[1] > 1:
        correlation = np.corrcoef(np.nan_to_num(emg), rowvar=False)
        for i in range(emg.shape[1]):
            for j in range(i + 1, emg.shape[1]):
                if abs(correlation[i, j]) >= limits.high_correlation_warn:
                    pair = {"column_a": i, "column_b": j, "correlation": float(correlation[i, j])}
                    high_corr_pairs.append(pair)
                    warnings.append(f"high_interchannel_correlation:{i + 1}-{j + 1}")
    sample_mismatch = abs(len(emg) - expected_samples) > limits.sample_tolerance
    if sample_mismatch:
        warnings.append("sample_count_mismatch_or_short_stream")
    return {
        "qc_pass": len(warnings) == 0,
        "manual_review_required": len(warnings) > 0,
        "received_samples": int(len(emg)),
        "expected_samples": int(expected_samples),
        "sample_count_ok": not sample_mismatch,
        "short_stream_or_dropout_samples": max(expected_samples - len(emg), 0),
        "channels": channel_results,
        "high_correlation_pairs": high_corr_pairs,
        "warnings": sorted(set(warnings)),
        "thresholds": asdict(limits),
    }


def create_preview_figure(
    emg_mV: np.ndarray,
    fs_hz: float,
    profile: ChannelProfile,
    title: str,
) -> plt.Figure:
    emg = np.asarray(emg_mV)
    if emg.ndim != 2 or emg.shape[1] != len(profile.channels):
        raise ValueError("Preview channel count does not match profile")
    channel_count = len(profile.channels)
    if channel_count == 16:
        rows, columns = 4, 4
    else:
        columns = int(np.ceil(np.sqrt(channel_count)))
        rows = int(np.ceil(channel_count / columns))
    time_s = np.arange(len(emg)) / fs_hz
    fig, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(16, 11 if channel_count == 16 else max(5, rows * 2.8)),
        sharex=True,
        squeeze=False,
    )
    axes = list(axes_grid.ravel())
    for index, (axis, channel) in enumerate(zip(axes, profile.channels)):
        axis.plot(time_s, emg[:, index], linewidth=0.55, color="tab:blue")
        axis.set_title(
            f"S{channel.sensor_id} | {channel.side} | {channel.abbreviation}\n{channel.muscle_slug}",
            fontsize=9,
        )
        axis.set_ylabel("mV", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.2)
    for axis in axes[channel_count:]:
        axis.set_visible(False)
    for axis in axes[max(0, (rows - 1) * columns):channel_count]:
        axis.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save_preview(
    path: Path,
    emg_mV: np.ndarray,
    fs_hz: float,
    profile: ChannelProfile,
    title: str,
    show: bool = False,
) -> Path:
    fig = create_preview_figure(emg_mV, fs_hz, profile, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temp_name, dpi=160, format="png")
        if show:
            plt.show(block=True)
        plt.close(fig)
        os.replace(temp_name, path)
    except BaseException:
        plt.close(fig)
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path
