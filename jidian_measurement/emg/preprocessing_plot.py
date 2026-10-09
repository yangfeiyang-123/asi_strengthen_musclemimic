from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

from .models import ChannelProfile


PALETTE = {
    "raw": "#616161",
    "filtered": "#2196F3",
    "rectified": "#FF9800",
    "envelope": "#F44336",
    "normalized": "#4CAF50",
}


def _grid_shape(channel_count: int) -> tuple[int, int]:
    columns = 4 if channel_count > 9 else 3 if channel_count > 4 else 2
    rows = int(np.ceil(channel_count / columns))
    return rows, columns


def _shade_edges(axis: plt.Axes, time_s: np.ndarray, edge_guard_samples: int) -> None:
    if edge_guard_samples <= 0 or len(time_s) <= 2 * edge_guard_samples:
        return
    axis.axvspan(time_s[0], time_s[edge_guard_samples], color="#BDBDBD", alpha=0.18, linewidth=0)
    axis.axvspan(time_s[-edge_guard_samples - 1], time_s[-1], color="#BDBDBD", alpha=0.18, linewidth=0)


def create_processing_comparison_figure(
    time_s: np.ndarray,
    raw_emg_mV: np.ndarray,
    filtered_mV: np.ndarray,
    rectified_mV: np.ndarray,
    envelope_mV: np.ndarray,
    profile: ChannelProfile,
    title: str,
    edge_guard_samples: int = 0,
) -> plt.Figure:
    time = np.asarray(time_s, dtype=np.float64)
    arrays = [np.asarray(item) for item in (raw_emg_mV, filtered_mV, rectified_mV, envelope_mV)]
    if any(array.shape != arrays[0].shape for array in arrays) or arrays[0].shape != (len(time), len(profile.channels)):
        raise ValueError("Comparison figure inputs must have shape [time, profile channels]")
    rows, columns = _grid_shape(len(profile.channels))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.4 * rows), sharex=True, squeeze=False)
    for index, (axis, channel) in enumerate(zip(axes.flat, profile.channels)):
        axis.plot(time, arrays[0][:, index], color=PALETTE["raw"], linewidth=0.45, alpha=0.55, label="raw")
        axis.plot(time, arrays[1][:, index], color=PALETTE["filtered"], linewidth=0.55, alpha=0.9, label="filtered")
        axis.plot(time, arrays[2][:, index], color=PALETTE["rectified"], linewidth=0.35, alpha=0.32, label="rectified")
        axis.plot(time, arrays[3][:, index], color=PALETTE["envelope"], linewidth=1.2, label="4 Hz envelope")
        _shade_edges(axis, time, edge_guard_samples)
        axis.set_title(f"S{channel.sensor_id} {channel.side} {channel.abbreviation}", fontsize=10)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("sEMG (mV)")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(profile.channels):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{title}\nRaw, 30-300 Hz + 50 Hz notch, rectified, and 4 Hz envelope", fontsize=13, y=0.998)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def create_normalized_envelope_figure(
    time_s: np.ndarray,
    normalized_envelope: np.ndarray,
    profile: ChannelProfile,
    title: str,
    edge_guard_samples: int = 0,
    normalization_method: str = "mvc",
) -> plt.Figure:
    time = np.asarray(time_s, dtype=np.float64)
    normalized = np.asarray(normalized_envelope)
    if normalized.shape != (len(time), len(profile.channels)):
        raise ValueError("Normalized figure input must have shape [time, profile channels]")
    rows, columns = _grid_shape(len(profile.channels))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.0 * rows), sharex=True, squeeze=False)
    if normalization_method == "mvc":
        display = 100.0 * normalized
        ylabel = "Activation (%MVC)"
        subtitle = "MVC-normalized sEMG envelopes"
    elif normalization_method == "dynamic_p95":
        display = normalized
        ylabel = "Activation (x session P95)"
        subtitle = "Session-P95-normalized sEMG envelopes"
    else:
        display = normalized
        ylabel = "Envelope (mV)"
        subtitle = "Unnormalized sEMG envelopes"
    for index, (axis, channel) in enumerate(zip(axes.flat, profile.channels)):
        axis.plot(time, display[:, index], color=PALETTE["normalized"], linewidth=1.2, label=normalization_method)
        _shade_edges(axis, time, edge_guard_samples)
        axis.set_title(f"S{channel.sensor_id} {channel.side} {channel.abbreviation}", fontsize=10)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(profile.channels):]:
        axis.set_visible(False)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def save_processing_figures(
    output_dir: Path,
    time_s: np.ndarray,
    stages: dict,
    profile: ChannelProfile,
    title: str,
    normalization_method: str = "mvc",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = create_processing_comparison_figure(
        time_s,
        stages["raw_emg_mV"],
        stages["filtered_mV"],
        stages["rectified_mV"],
        stages["envelope_mV"],
        profile,
        title,
        stages.get("edge_guard_samples", 0),
    )
    normalized = create_normalized_envelope_figure(
        time_s,
        stages["normalized_envelope"],
        profile,
        title,
        stages.get("edge_guard_samples", 0),
        normalization_method,
    )
    paths = [output_dir / "preprocessing_comparison.png", output_dir / "normalized_envelope.png"]
    try:
        comparison.savefig(paths[0], dpi=300, bbox_inches="tight")
        normalized.savefig(paths[1], dpi=300, bbox_inches="tight")
    finally:
        plt.close(comparison)
        plt.close(normalized)
    return paths
