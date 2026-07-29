"""Deprecated six-channel high-clear wrapper using ``legacy_high_clear_6ch``.

The wrapper keeps the small CSV/path helpers used by historical scripts, but
all new acquisition is delegated to the shared ``emg`` package. It never
mutates globals in ``delysis_measure``.
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from emg.acquisition import collect_action
from emg.models import TrignoConfig
from emg.profiles import LEGACY_HIGH_CLEAR_6CH


BASE_DIR = Path(__file__).resolve().parent / "dataset_root"
DEFAULT_ACTION_LABEL = "forehand_high_clear"
DEFAULT_DURATION = 3.0
DEFAULT_TRIALS = 10
DEFAULT_REST_SECONDS = 45.0
sensor_channels = list(LEGACY_HIGH_CLEAR_6CH.channel_ids)
channel_names = {channel.sensor_id: channel.name_en for channel in LEGACY_HIGH_CLEAR_6CH.channels}


def sanitize_identifier(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    if not text or ".." in text or any(ch in text for ch in '<>:"/\\|?*'):
        raise ValueError(f"Identifier contains unsafe path characters: {value!r}")
    return text


def normalize_error_label(value: str) -> str:
    return sanitize_identifier(str(value).strip() or "correct")


def movement_quality(error_label: str) -> str:
    return "correct" if normalize_error_label(error_label) == "correct" else "error"


def build_trial_path(
    base_dir: Path,
    participant_id: str,
    action_label: str,
    trial_index: int,
    timestamp: str,
    error_label: str = "correct",
) -> Path:
    participant_id = sanitize_identifier(participant_id)
    action_label = sanitize_identifier(action_label)
    error_label = normalize_error_label(error_label)
    filename = f"{action_label}_{timestamp}_trial_{trial_index:02d}_{error_label}_raw_emg.csv"
    return Path(base_dir) / participant_id / action_label / error_label / filename


def selected_channel_headers() -> list[str]:
    return ["time_s"] + [
        f"sensor_{channel.sensor_id}_{channel.side}_{channel.muscle_slug}_mV"
        for channel in LEGACY_HIGH_CLEAR_6CH.channels
    ]


def save_trial_csv(
    path: Path,
    emg_arr: np.ndarray,
    participant_id: str,
    action_label: str,
    trial_index: int,
    timestamp: str,
    error_label: str = "correct",
) -> None:
    if emg_arr.ndim != 2 or emg_arr.shape[1] != len(sensor_channels):
        raise ValueError("Legacy CSV data must contain exactly six configured columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# participant_id", participant_id])
        writer.writerow(["# action_label", action_label])
        writer.writerow(["# movement_quality", movement_quality(error_label)])
        writer.writerow(["# error_label", normalize_error_label(error_label)])
        writer.writerow(["# trial_index", trial_index])
        writer.writerow(["# timestamp", timestamp])
        writer.writerow(["# sample_rate_hz", 2000])
        writer.writerow(["# channel_profile_id", LEGACY_HIGH_CLEAR_6CH.profile_id])
        writer.writerow(["# sensor_channels", " ".join(map(str, sensor_channels))])
        writer.writerow(selected_channel_headers())
        for index, values in enumerate(emg_arr):
            writer.writerow([f"{index / 2000:.6f}", *[f"{float(value):.8f}" for value in values]])


def plot_trial_emg(
    emg_arr: np.ndarray,
    participant_id: str,
    action_label: str,
    trial_index: int,
    error_label: str = "correct",
    show: bool = True,
):
    if emg_arr.ndim != 2 or emg_arr.shape[1] == 0:
        raise ValueError("emg_arr must be [samples, channels]")
    time_s = np.arange(len(emg_arr)) / 2000.0
    fig, axes = plt.subplots(emg_arr.shape[1], 1, figsize=(12, max(3, 2.2 * emg_arr.shape[1])), sharex=True)
    if emg_arr.shape[1] == 1:
        axes = [axes]
    for index, axis in enumerate(axes):
        channel = LEGACY_HIGH_CLEAR_6CH.channels[index]
        axis.plot(time_s, emg_arr[:, index], linewidth=0.8)
        axis.set_title(f"Sensor {channel.sensor_id}: {channel.name_en}")
        axis.set_ylabel("EMG (mV)")
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.3)
    fig.suptitle(f"{participant_id} | {action_label} | trial {trial_index:02d} | {movement_quality(error_label)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    return fig


def run_collection(
    participant_id: str,
    action_label: str = DEFAULT_ACTION_LABEL,
    trial_count: int = DEFAULT_TRIALS,
    duration: float = DEFAULT_DURATION,
    rest_seconds: float = DEFAULT_REST_SECONDS,
    base_dir: Path = BASE_DIR,
    session_id: str | None = None,
    dry_run: bool = False,
):
    warnings.warn("Use `python -m emg.cli collect --profile legacy_high_clear_6ch`", DeprecationWarning, stacklevel=2)
    return collect_action(
        Path(base_dir),
        sanitize_identifier(participant_id),
        sanitize_identifier(session_id or datetime.now().strftime("LEGACY_%Y%m%d")),
        LEGACY_HIGH_CLEAR_6CH.profile_id,
        "badminton_primitive_protocol_v1",
        action_label,
        handedness="right",
        dry_run=dry_run,
        interactive=not dry_run,
        target_valid_trials=trial_count,
        action_duration_s=duration,
        rest_seconds=rest_seconds,
        trigno_config=TrignoConfig(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated 6-channel wrapper; data uses legacy_high_clear_6ch")
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("--session")
    parser.add_argument("-a", "--action", default=DEFAULT_ACTION_LABEL)
    parser.add_argument("-n", "--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("-d", "--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("-r", "--rest", type=float, default=DEFAULT_REST_SECONDS)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_collection(
        args.participant,
        args.action,
        args.trials,
        args.duration,
        args.rest,
        args.base_dir,
        args.session,
        args.dry_run,
    )
    print(f"Saved {len(paths)} trial bundle(s).")


if __name__ == "__main__":
    main()
