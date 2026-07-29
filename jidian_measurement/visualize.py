import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "visualize"


@dataclass
class TrialData:
    path: Path
    metadata: dict
    headers: list
    time_s: np.ndarray
    emg: np.ndarray


def read_trial_csv(csv_path):
    csv_path = Path(csv_path)
    metadata = {}
    headers = None
    rows = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            first = row[0].strip()
            if first.startswith("#"):
                key = first.lstrip("#").strip()
                metadata[key] = row[1].strip() if len(row) > 1 else ""
                continue

            if headers is None:
                headers = [item.strip() for item in row]
                continue

            rows.append([float(item) for item in row])

    if headers is None:
        raise ValueError(f"No data header found in {csv_path}")
    if len(headers) < 2:
        raise ValueError(f"CSV must include time_s and at least one EMG channel: {csv_path}")
    if not rows:
        raise ValueError(f"No EMG samples found in {csv_path}")

    data = np.asarray(rows, dtype=np.float32)
    return TrialData(
        path=csv_path,
        metadata=metadata,
        headers=headers,
        time_s=data[:, 0],
        emg=data[:, 1:],
    )


def build_output_path(csv_path, input_dir, output_dir):
    csv_path = Path(csv_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    folder_parts = input_dir.parts[-3:] if len(input_dir.parts) >= 3 else input_dir.parts
    group_name = "_".join(folder_parts) if folder_parts else "collection"

    return output_dir / group_name / f"{csv_path.stem}.png"


def plot_trial(trial, output_path):
    channel_count = trial.emg.shape[1]
    if channel_count == 0:
        raise ValueError(f"No EMG channels found in {trial.path}")

    fig, axes = plt.subplots(
        channel_count,
        1,
        figsize=(12, max(2.2 * channel_count, 3.0)),
        sharex=True,
    )
    if channel_count == 1:
        axes = [axes]

    channel_headers = trial.headers[1:]
    for index, axis in enumerate(axes):
        label = channel_headers[index] if index < len(channel_headers) else f"channel_{index + 1}"
        axis.plot(trial.time_s, trial.emg[:, index], linewidth=0.8, color="tab:blue")
        axis.set_title(label, fontsize=10)
        axis.set_ylabel("EMG (mV)")
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)

    participant = trial.metadata.get("participant_id", "unknown")
    action = trial.metadata.get("action_label", "unknown_action")
    trial_index = trial.metadata.get("trial_index", "?")
    error_label = trial.metadata.get("error_label", "unknown")
    quality = trial.metadata.get("movement_quality", "unknown")

    fig.suptitle(
        f"{participant} | {action} | trial {trial_index} | {quality}: {error_label}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def visualize_folder(input_dir, output_dir=DEFAULT_OUTPUT_DIR):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_dir}")

    saved_paths = []
    for csv_path in csv_paths:
        trial = read_trial_csv(csv_path)
        output_path = build_output_path(csv_path, input_dir, output_dir)
        saved_paths.append(plot_trial(trial, output_path))

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize all EMG CSV files in one collection folder."
    )
    parser.add_argument("folder", type=Path, help="Folder containing raw EMG CSV files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for visualization PNG files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saved_paths = visualize_folder(args.folder, args.output)

    print("Saved visualizations:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
