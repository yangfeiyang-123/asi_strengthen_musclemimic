from __future__ import annotations

from dataclasses import asdict
import csv
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .models import ActionSpec, ChannelProfile, ProcessingConfig
from .processing import preprocess_signal_stages
from .profiles import require_analysis_profile
from .storage import atomic_save_npz, atomic_write_json, read_json


class InsufficientTrialsError(ValueError):
    """Raised when a sample variance cannot be estimated for an action."""


def pointwise_mean_variance(envelopes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pointwise trial mean, sample variance, and standard deviation.

    ``envelopes`` must have shape ``[trials, samples, channels]``.  Sample
    variance (``ddof=1``) is deliberate: the recorded repetitions are treated
    as a sample of the participant's repeatable performance, not the complete
    population of all possible repetitions.
    """

    values = np.asarray(envelopes, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Action statistics expect [trials, samples, channels]")
    if values.shape[0] < 2:
        raise InsufficientTrialsError("At least two included trials are required to estimate sample variance")
    if values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("Action statistics require at least two samples and one channel")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action statistics input contains NaN or Inf after preprocessing")
    mean = np.mean(values, axis=0)
    variance = np.var(values, axis=0, ddof=1)
    return mean, variance, np.sqrt(variance)


def _profile_signature(snapshot: Mapping[str, Any]) -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (int(channel["sensor_id"]), str(channel["side"]), str(channel["muscle_slug"]))
        for channel in snapshot["channels"]
    )


def _event_sample(path: Path, event_name: str) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_name") == event_name and row.get("sample_index") not in {None, ""}:
                return int(row["sample_index"])
    return None


def _action_value(action: ActionSpec | Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    if action is None:
        return default
    if isinstance(action, Mapping):
        return action.get(name, default)
    return getattr(action, name, default)


def _action_from_session_snapshot(action_dir: Path) -> Mapping[str, Any] | None:
    protocol_path = action_dir.parent.parent / "protocol.json"
    if not protocol_path.exists():
        return None
    protocol = read_json(protocol_path)
    for action in protocol.get("actions", []):
        if action.get("action_id") == action_dir.name:
            return action
    return None


def _common_time_grid(times: list[np.ndarray], fs_hz: float) -> np.ndarray:
    if not times:
        raise ValueError("No trial time axes were supplied")
    for index, time_s in enumerate(times, start=1):
        if time_s.ndim != 1 or len(time_s) < 2 or not np.all(np.isfinite(time_s)):
            raise ValueError(f"Trial {index} has an invalid time axis")
        if np.any(np.diff(time_s) <= 0):
            raise ValueError(f"Trial {index} time axis is not strictly increasing")
    start_s = max(float(time_s[0]) for time_s in times)
    stop_s = min(float(time_s[-1]) for time_s in times)
    sample_count = int(np.floor((stop_s - start_s) * fs_hz + 1e-9)) + 1
    if sample_count < 2:
        raise ValueError("Included trials do not share a usable time interval")
    return start_s + np.arange(sample_count, dtype=np.float64) / fs_hz


def _resample_to_grid(values: np.ndarray, source_time_s: np.ndarray, target_time_s: np.ndarray) -> np.ndarray:
    if len(values) == len(target_time_s) and np.allclose(source_time_s, target_time_s, rtol=0, atol=1e-9):
        return np.asarray(values, dtype=np.float64)
    return np.column_stack(
        [np.interp(target_time_s, source_time_s, values[:, channel]) for channel in range(values.shape[1])]
    )


def load_action_envelopes(
    action_dir: Path,
    profile: ChannelProfile,
    *,
    only_valid: bool = True,
    processing_config: ProcessingConfig | None = None,
) -> dict[str, Any]:
    """Load, validate, preprocess, and time-align repetitions of one action."""

    action_dir = Path(action_dir).resolve()
    trial_dirs = sorted(path for path in action_dir.glob("trial_*") if path.is_dir())
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found under {action_dir}")

    expected_ids = np.asarray(profile.channel_ids, dtype=np.int16)
    expected_signature = _profile_signature(profile.to_dict())
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    raw_trials: list[dict[str, Any]] = []
    sample_rates: list[float] = []
    requested_durations: list[float] = []
    cue_times: list[float] = []

    for trial_dir in trial_dirs:
        metadata_path = trial_dir / "metadata.json"
        raw_path = trial_dir / "raw_emg.npz"
        if not metadata_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"Trial bundle is missing metadata.json or raw_emg.npz: {trial_dir}")
        metadata = read_json(metadata_path)
        trial_id = str(metadata.get("trial_id", trial_dir.name))
        if str(metadata.get("action_id")) != action_dir.name:
            raise ValueError(f"Action metadata does not match directory at {trial_dir}")
        if metadata.get("channel_profile_id") != profile.profile_id:
            raise ValueError(f"Channel profile ID differs at {trial_dir}")
        snapshot = metadata.get("channel_profile_snapshot")
        if not isinstance(snapshot, dict) or _profile_signature(snapshot) != expected_signature:
            raise ValueError(f"Channel profile order or identity differs at {trial_dir}")
        if only_valid and not bool(metadata.get("valid_for_analysis", False)):
            labels = ",".join(str(value) for value in metadata.get("error_labels", [])) or "not_valid"
            excluded.append({"trial_id": trial_id, "reason": labels})
            continue

        with np.load(raw_path, allow_pickle=False) as payload:
            emg = np.asarray(payload["emg_mV"], dtype=np.float64)
            fs_hz = float(payload["fs_hz"])
            time_s = np.asarray(payload["time_s"], dtype=np.float64)
            stream_ids = np.asarray(payload["stream_channel_ids"], dtype=np.int16)
        if emg.ndim != 2 or emg.shape[1] != len(profile.channels):
            raise ValueError(f"Raw EMG channel count mismatch at {raw_path}")
        if len(time_s) != len(emg):
            raise ValueError(f"Raw EMG/time length mismatch at {raw_path}")
        if not np.array_equal(stream_ids, expected_ids):
            raise ValueError(f"Raw sensor order differs from the channel profile at {raw_path}")
        if not np.all(np.isfinite(emg)):
            raise ValueError(f"Raw EMG contains NaN or Inf at {raw_path}")
        included.append(trial_id)
        sample_rates.append(fs_hz)
        requested_durations.append(float(metadata.get("requested_duration_s", len(emg) / fs_hz)))
        cue_sample = _event_sample(trial_dir / "events.csv", "movement_cue")
        if cue_sample is not None:
            cue_times.append(cue_sample / fs_hz)
        raw_trials.append({"trial_dir": trial_dir, "emg_mV": emg, "time_s": time_s, "fs_hz": fs_hz})

    if len(raw_trials) < 2:
        raise InsufficientTrialsError(
            f"Action {action_dir.name} has {len(raw_trials)} included trial(s); at least two are required"
        )
    reference_fs = sample_rates[0]
    if any(not np.isclose(value, reference_fs) for value in sample_rates[1:]):
        raise ValueError(f"Mixed sample rates are not allowed in action statistics: {action_dir}")
    config = processing_config or ProcessingConfig(sample_rate_hz=reference_fs, normalization="none")
    if not np.isclose(config.sample_rate_hz, reference_fs):
        raise ValueError("Action-statistics processing sample rate does not match the raw trials")

    times: list[np.ndarray] = []
    envelopes: list[np.ndarray] = []
    for trial in raw_trials:
        stages = preprocess_signal_stages(trial["emg_mV"], config)
        times.append(trial["time_s"])
        envelopes.append(np.asarray(stages["envelope_mV"], dtype=np.float64))
    common_time_s = _common_time_grid(times, reference_fs)
    aligned = np.stack(
        [_resample_to_grid(envelope, time_s, common_time_s) for envelope, time_s in zip(envelopes, times)],
        axis=0,
    )
    return {
        "envelopes_mV": aligned,
        "time_s": common_time_s,
        "fs_hz": reference_fs,
        "processing_config": config,
        "included_trial_ids": included,
        "excluded_trials": excluded,
        "source_sample_counts": [len(trial["emg_mV"]) for trial in raw_trials],
        "requested_durations_s": requested_durations,
        "movement_cue_s": float(np.median(cue_times)) if cue_times else None,
    }


def create_action_statistics_figure(
    time_s: np.ndarray,
    mean_envelope_mV: np.ndarray,
    variance_envelope_mV2: np.ndarray,
    std_envelope_mV: np.ndarray,
    profile: ChannelProfile,
    action_id: str,
    n_trials: int,
    *,
    movement_cue_s: float | None = None,
    movement_end_s: float | None = None,
) -> plt.Figure:
    """Create the requested 4x4 mean/variance overview for 16 sensors."""

    mean = np.asarray(mean_envelope_mV, dtype=np.float64)
    variance = np.asarray(variance_envelope_mV2, dtype=np.float64)
    std = np.asarray(std_envelope_mV, dtype=np.float64)
    if mean.shape != variance.shape or mean.shape != std.shape:
        raise ValueError("Mean, variance, and standard-deviation arrays must have identical shapes")
    if mean.shape != (len(time_s), len(profile.channels)):
        raise ValueError("Action-statistics arrays do not match time/profile dimensions")

    channel_count = len(profile.channels)
    columns = 4 if channel_count == 16 else int(np.ceil(np.sqrt(channel_count)))
    rows = 4 if channel_count == 16 else int(np.ceil(channel_count / columns))
    fig, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(18, 12 if channel_count == 16 else max(6, rows * 3.0)),
        sharex=True,
        squeeze=False,
    )
    axes = list(axes_grid.ravel())
    stride = max(1, len(time_s) // 1800)
    plot_time = np.asarray(time_s)[::stride]
    blue = "#2F5D8A"
    blue_fill = "#B9D0E5"
    marker_color = "#5B6470"
    movement_mask = np.ones(len(time_s), dtype=bool)
    if movement_cue_s is not None:
        movement_mask &= time_s >= movement_cue_s
    if movement_end_s is not None:
        movement_mask &= time_s <= movement_end_s
    if not np.any(movement_mask):
        movement_mask[:] = True

    for index, (axis, channel) in enumerate(zip(axes, profile.channels)):
        channel_mean = mean[:, index]
        channel_std = std[:, index]
        lower = np.maximum(channel_mean - channel_std, 0.0)
        upper = channel_mean + channel_std
        axis.fill_between(
            plot_time,
            lower[::stride],
            upper[::stride],
            color=blue_fill,
            alpha=0.62,
            linewidth=0,
        )
        axis.plot(plot_time, channel_mean[::stride], color=blue, linewidth=1.15)
        if movement_cue_s is not None:
            axis.axvline(movement_cue_s, color=marker_color, linestyle="--", linewidth=0.8)
        if movement_end_s is not None:
            axis.axvline(movement_end_s, color=marker_color, linestyle=":", linewidth=0.8)
        average_variance = float(np.mean(variance[movement_mask, index]))
        side = "R" if channel.side == "right" else "L" if channel.side == "left" else channel.side
        axis.set_title(
            f"S{channel.sensor_id} | {side} | {channel.abbreviation}\n{channel.muscle_slug}",
            fontsize=9,
        )
        axis.text(
            0.98,
            0.96,
            f"mean $\\sigma^2$={average_variance:.3g} mV$^2$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#343A40",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        axis.set_ylabel("Envelope (mV)", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(color="#D9DEE5", alpha=0.65, linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(bottom=0)
    for axis in axes[channel_count:]:
        axis.set_visible(False)
    for axis in axes[max(0, (rows - 1) * columns):channel_count]:
        axis.set_xlabel("Recording time (s)", fontsize=8)

    fig.suptitle(f"Action statistics: {action_id}", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.956,
        f"N={n_trials} valid trials | line: pointwise mean | band: mean ± 1 SD | sample variance uses ddof=1",
        ha="center",
        va="top",
        fontsize=9,
        color="#4B5563",
    )
    legend_items = [
        Line2D([0], [0], color=blue, linewidth=1.5, label="Mean envelope"),
        Patch(facecolor=blue_fill, edgecolor="none", alpha=0.62, label="Mean ± 1 SD"),
    ]
    if movement_cue_s is not None:
        legend_items.append(Line2D([0], [0], color=marker_color, linestyle="--", label="Movement cue"))
    if movement_end_s is not None:
        legend_items.append(Line2D([0], [0], color=marker_color, linestyle=":", label="Movement end"))
    fig.legend(handles=legend_items, loc="lower center", ncol=len(legend_items), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.94))
    return fig


def _save_figure_atomic(fig: plt.Figure, path: Path, *, show: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temp_name, dpi=180, format="png", facecolor="white")
        if show:
            plt.show(block=True)
        plt.close(fig)
        os.replace(temp_name, path)
    except BaseException:
        plt.close(fig)
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def save_action_statistics(
    action_dir: Path,
    profile: ChannelProfile,
    action: ActionSpec | Mapping[str, Any] | None = None,
    *,
    only_valid: bool = True,
    show: bool = False,
    processing_config: ProcessingConfig | None = None,
) -> dict[str, Path]:
    """Save numerical and graphical repeated-trial statistics for one action."""

    require_analysis_profile(profile.profile_id)
    action_dir = Path(action_dir).resolve()
    action = action or _action_from_session_snapshot(action_dir)
    loaded = load_action_envelopes(
        action_dir,
        profile,
        only_valid=only_valid,
        processing_config=processing_config,
    )
    envelopes = loaded["envelopes_mV"]
    mean, variance, std = pointwise_mean_variance(envelopes)
    time_s = loaded["time_s"]
    cue_s = loaded["movement_cue_s"]
    post_s = float(_action_value(action, "post_s", 0.0))
    requested_duration_s = float(np.median(loaded["requested_durations_s"]))
    movement_end_s = min(float(time_s[-1]), requested_duration_s - post_s) if cue_s is not None else None
    n_trials = int(envelopes.shape[0])

    npz_path = action_dir / "action_statistics.npz"
    json_path = action_dir / "action_statistics.json"
    figure_path = action_dir / "action_mean_variance.png"
    atomic_save_npz(
        npz_path,
        time_s=time_s.astype(np.float64),
        mean_envelope_mV=mean.astype(np.float32),
        variance_envelope_mV2=variance.astype(np.float32),
        std_envelope_mV=std.astype(np.float32),
        n_trials=np.asarray(n_trials, dtype=np.int64),
        trial_ids=np.asarray(loaded["included_trial_ids"]),
        stream_channel_ids=np.asarray(profile.channel_ids, dtype=np.int16),
        fs_hz=np.asarray(loaded["fs_hz"], dtype=np.float64),
    )

    movement_mask = np.ones(len(time_s), dtype=bool)
    if cue_s is not None:
        movement_mask &= time_s >= cue_s
    if movement_end_s is not None:
        movement_mask &= time_s <= movement_end_s
    if not np.any(movement_mask):
        movement_mask[:] = True
    channel_summary = []
    for index, channel in enumerate(profile.channels):
        channel_summary.append(
            {
                "sensor_id": channel.sensor_id,
                "side": channel.side,
                "muscle_slug": channel.muscle_slug,
                "abbreviation": channel.abbreviation,
                "movement_mean_envelope_mV": float(np.mean(mean[movement_mask, index])),
                "movement_peak_mean_envelope_mV": float(np.max(mean[movement_mask, index])),
                "movement_mean_pointwise_variance_mV2": float(np.mean(variance[movement_mask, index])),
                "full_recording_mean_pointwise_variance_mV2": float(np.mean(variance[:, index])),
            }
        )
    metadata = {
        "schema_version": 1,
        "action_id": action_dir.name,
        "channel_profile_id": profile.profile_id,
        "channel_profile_version": profile.version,
        "channel_profile_snapshot": profile.to_dict(),
        "statistical_grain": "pointwise_across_repeated_trials",
        "source_signal": "processed_envelope_mV_using_the_recorded_processing_config_below",
        "variance_definition": "sample_variance_ddof_1",
        "visual_uncertainty": "mean_plus_or_minus_one_standard_deviation",
        "only_valid_trials": only_valid,
        "n_trials": n_trials,
        "included_trial_ids": loaded["included_trial_ids"],
        "excluded_trials": loaded["excluded_trials"],
        "source_sample_counts": loaded["source_sample_counts"],
        "common_sample_count": int(len(time_s)),
        "common_duration_s": float(time_s[-1] - time_s[0] + 1.0 / loaded["fs_hz"]),
        "sample_rate_hz": float(loaded["fs_hz"]),
        "movement_cue_s": cue_s,
        "movement_end_s": movement_end_s,
        "processing_config": asdict(loaded["processing_config"]),
        "array_shapes": {
            "mean_envelope_mV": list(mean.shape),
            "variance_envelope_mV2": list(variance.shape),
            "std_envelope_mV": list(std.shape),
        },
        "channel_summary": channel_summary,
        "outputs": {
            "statistics_npz": str(npz_path.resolve()),
            "figure_png": str(figure_path.resolve()),
        },
    }
    atomic_write_json(json_path, metadata)
    fig = create_action_statistics_figure(
        time_s,
        mean,
        variance,
        std,
        profile,
        action_dir.name,
        n_trials,
        movement_cue_s=cue_s,
        movement_end_s=movement_end_s,
    )
    _save_figure_atomic(fig, figure_path, show=show)
    return {"npz": npz_path, "json": json_path, "figure": figure_path}


def summarize_session_actions(
    session_dir: Path,
    profile_id: str,
    *,
    action_ids: list[str] | None = None,
    only_valid: bool = True,
    show: bool = False,
    processing_config: ProcessingConfig | None = None,
) -> list[dict[str, Path]]:
    """Generate action statistics for selected action directories in a session."""

    session_dir = Path(session_dir).resolve()
    profile = require_analysis_profile(profile_id)
    snapshot = read_json(session_dir / "channel_profile.json")
    if _profile_signature(snapshot) != _profile_signature(profile.to_dict()):
        raise ValueError("Session channel profile snapshot does not match the requested profile")
    action_filter = set(action_ids or [])
    action_dirs = sorted(path for path in (session_dir / "trials").iterdir() if path.is_dir())
    if action_filter:
        missing = action_filter - {path.name for path in action_dirs}
        if missing:
            raise FileNotFoundError(f"Action directories were not found: {sorted(missing)}")
        action_dirs = [path for path in action_dirs if path.name in action_filter]
    return [
        save_action_statistics(
            action_dir,
            profile,
            only_valid=only_valid,
            show=show,
            processing_config=processing_config,
        )
        for action_dir in action_dirs
    ]
