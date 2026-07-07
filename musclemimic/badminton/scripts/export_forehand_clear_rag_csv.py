from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ACTION_TYPE = "forehand_clear"
DEFAULT_OUTCOME_LABEL = "low_speed"
SUPPORTED_OUTCOME_LABELS = {"ball_high_not_far", "low_speed", "uncoordinated_power"}

JOINT_SIGNAL_TO_MYO = {
    "trunk_rotation": "axial_rotation",
    "forearm_pronation": "pro_sup_r",
    "shoulder_internal_rotation": "shoulder_rot_r",
    "elbow_flexion": "elbow_flex_r",
    "wrist_extension": "flexion_r",
}

MUSCLE_SIGNAL_TO_MYO_GROUPS = {
    # MyoFullBody does not expose an external-oblique label in the default asset.
    # rect_abd_* is the closest stable abdominal activation proxy available in rollout exports.
    "external_oblique": ("rect_abd_r", "rect_abd_l"),
    "anterior_deltoid": ("DELT1",),
    "triceps_brachii": ("TRIlong", "TRIlat", "TRImed"),
    "forearm_pronator_group": ("PT", "PQ"),
}

DISABLE_FINGERS_ACTUATOR_INDEX = {
    # Fallback for legacy MyoFullBody rollout files written before muscle_names was exported.
    # These indices are the MuJoCo actuator order for MyoFullBody(disable_fingers=True), nu=354.
    "rect_abd_r": 22,
    "rect_abd_l": 23,
    "DELT1": 210,
    "TRIlong": 225,
    "TRIlat": 226,
    "TRImed": 227,
    "PT": 240,
    "PQ": 241,
}

CSV_COLUMNS = [
    "sample_id",
    "split",
    "action_type",
    "outcome_label",
    "time",
    "event_impact",
    "event_acceleration_start",
    "event_follow_through_end",
    "joint_trunk_rotation",
    "joint_forearm_pronation",
    "joint_shoulder_internal_rotation",
    "joint_elbow_flexion",
    "joint_wrist_extension",
    "muscle_external_oblique",
    "muscle_anterior_deltoid",
    "muscle_triceps_brachii",
    "muscle_forearm_pronator_group",
]


@dataclass(frozen=True)
class ExportedSample:
    sample_id: str
    split: str
    outcome_label: str
    rows: list[dict[str, object]]


def _safe_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


def _load_names(data: np.lib.npyio.NpzFile, key: str) -> list[str] | None:
    if key not in data.files:
        return None
    return [str(item) for item in np.asarray(data[key]).tolist()]


def _signal_safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def _individual_muscle_column_names(muscle_names: list[str] | None, activation_dim: int) -> list[str]:
    if muscle_names is not None:
        if len(muscle_names) != activation_dim:
            raise ValueError(f"muscle_names length {len(muscle_names)} does not match activation dimension {activation_dim}")
        raw_names = [_signal_safe_name(name) for name in muscle_names]
    else:
        raw_names = [f"actuator_{idx:03d}" for idx in range(activation_dim)]

    seen: dict[str, int] = {}
    columns: list[str] = []
    for raw_name in raw_names:
        count = seen.get(raw_name, 0)
        seen[raw_name] = count + 1
        suffix = "" if count == 0 else f"_{count + 1}"
        prefix = "muscle_myo" if muscle_names is not None else "muscle"
        columns.append(f"{prefix}_{raw_name}{suffix}")
    return columns


def _episode_ids(data: np.lib.npyio.NpzFile) -> list[int]:
    if "n_episodes" in data.files:
        count = int(np.asarray(data["n_episodes"]).item())
        return [idx for idx in range(count) if f"episode_{idx}_joint_positions" in data.files]

    ids: list[int] = []
    for key in data.files:
        match = re.fullmatch(r"episode_(\d+)_joint_positions", key)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _parse_episode_selector(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    selected: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        selected.add(int(raw))
    return selected


def _split_for_episode(
    episode_id: int,
    all_episode_ids: list[int],
    split_hint: str | None,
    correct_episodes: set[int] | None,
    eval_episodes: set[int] | None,
) -> str:
    if split_hint is not None:
        return split_hint
    if correct_episodes is not None or eval_episodes is not None:
        if correct_episodes is not None and episode_id in correct_episodes:
            return "correct"
        if eval_episodes is not None and episode_id in eval_episodes:
            return "eval"
        raise ValueError(f"episode {episode_id} is not assigned to correct or eval")
    return "correct" if episode_id == all_episode_ids[0] else "eval"


def _joint_qpos_indices(joint_names: list[str], qpos_dim: int) -> dict[str, int]:
    if qpos_dim == len(joint_names):
        return {name: idx for idx, name in enumerate(joint_names)}
    if joint_names and joint_names[0] == "root" and qpos_dim == len(joint_names) + 6:
        return {name: idx + 6 for idx, name in enumerate(joint_names) if name != "root"}
    raise ValueError(f"cannot map {len(joint_names)} joint names to qpos dimension {qpos_dim}")


def _muscle_indices(
    muscle_names: list[str] | None,
    activation_dim: int,
    signal_to_groups: dict[str, tuple[str, ...]] = MUSCLE_SIGNAL_TO_MYO_GROUPS,
) -> dict[str, list[int]]:
    if muscle_names is not None:
        if len(muscle_names) != activation_dim:
            raise ValueError(f"muscle_names length {len(muscle_names)} does not match activation dimension {activation_dim}")
        name_to_index = {name: idx for idx, name in enumerate(muscle_names)}
    elif activation_dim == 354:
        name_to_index = DISABLE_FINGERS_ACTUATOR_INDEX
    else:
        raise ValueError(
            "rollout is missing muscle_names and activation dimension is not the legacy MyoFullBody 354-actuator layout"
        )

    mapped: dict[str, list[int]] = {}
    for rag_signal, myo_names in signal_to_groups.items():
        missing = [name for name in myo_names if name not in name_to_index]
        if missing:
            raise ValueError(f"cannot map muscle_{rag_signal}; missing actuator names: {missing}")
        mapped[rag_signal] = [name_to_index[name] for name in myo_names]
    return mapped


def _infer_impact_time(
    time: np.ndarray,
    joint_degrees: dict[str, np.ndarray],
    explicit_time: float | None,
    explicit_frame: int | None,
) -> float:
    if explicit_time is not None:
        if explicit_time < float(time[0]) or explicit_time > float(time[-1]):
            raise ValueError(f"impact time {explicit_time} is outside sample time range [{time[0]}, {time[-1]}]")
        return explicit_time
    if explicit_frame is not None:
        if explicit_frame < 0 or explicit_frame >= len(time):
            raise ValueError(f"impact frame {explicit_frame} is outside sample length {len(time)}")
        return float(time[explicit_frame])

    release_signals = [
        joint_degrees["shoulder_internal_rotation"],
        joint_degrees["elbow_flexion"],
        joint_degrees["forearm_pronation"],
        joint_degrees["wrist_extension"],
    ]
    score = np.zeros(len(time), dtype=np.float64)
    for values in release_signals:
        score += np.abs(np.gradient(values, time, edge_order=1))
    return float(time[int(np.argmax(score))])


def _event_times(time: np.ndarray, impact_time: float) -> tuple[float, float, float]:
    acceleration_start = max(float(time[0]), impact_time - 0.10)
    follow_through_end = min(float(time[-1]), impact_time + 0.10)
    return impact_time, acceleration_start, follow_through_end


def export_npz_to_samples(
    path: Path,
    split_hint: str | None = None,
    correct_episodes: set[int] | None = None,
    eval_episodes: set[int] | None = None,
    outcome_label: str = DEFAULT_OUTCOME_LABEL,
    impact_time: float | None = None,
    impact_frame: int | None = None,
    include_all_muscles: bool = True,
) -> list[ExportedSample]:
    if outcome_label not in SUPPORTED_OUTCOME_LABELS:
        raise ValueError(f"unsupported outcome label: {outcome_label}")

    data = np.load(path, allow_pickle=True)
    joint_names = _load_names(data, "joint_names")
    if not joint_names:
        raise ValueError(f"{path} is missing joint_names")
    muscle_names = _load_names(data, "muscle_names")
    episode_ids = _episode_ids(data)
    if not episode_ids:
        raise ValueError(f"{path} does not contain episode_*_joint_positions arrays")

    samples: list[ExportedSample] = []
    path_id = _safe_id(path.stem)
    for episode_id in episode_ids:
        prefix = f"episode_{episode_id}"
        qpos = np.asarray(data[f"{prefix}_joint_positions"], dtype=np.float64)
        activations = np.asarray(data[f"{prefix}_muscle_activations"], dtype=np.float64)
        time = np.asarray(data[f"{prefix}_timesteps"], dtype=np.float64)
        if qpos.ndim != 2 or activations.ndim != 2 or time.ndim != 1:
            raise ValueError(f"{path}:{prefix} has invalid qpos/activation/time dimensions")
        if len(qpos) != len(activations) or len(qpos) != len(time):
            raise ValueError(f"{path}:{prefix} qpos, activation, and time lengths differ")
        if len(time) < 2:
            raise ValueError(f"{path}:{prefix} must contain at least two time points")

        joint_indices = _joint_qpos_indices(joint_names, qpos.shape[1])
        muscle_indices = _muscle_indices(muscle_names, activations.shape[1])

        joint_degrees = {
            rag_signal: np.rad2deg(qpos[:, joint_indices[myo_joint]])
            for rag_signal, myo_joint in JOINT_SIGNAL_TO_MYO.items()
        }
        muscle_values = {
            rag_signal: np.clip(np.mean(activations[:, indices], axis=1), 0.0, 1.0)
            for rag_signal, indices in muscle_indices.items()
        }
        individual_muscle_columns = (
            _individual_muscle_column_names(muscle_names, activations.shape[1]) if include_all_muscles else []
        )

        inferred_impact = _infer_impact_time(time, joint_degrees, impact_time, impact_frame)
        event_impact, event_acceleration_start, event_follow_through_end = _event_times(time, inferred_impact)
        split = _split_for_episode(episode_id, episode_ids, split_hint, correct_episodes, eval_episodes)
        sample_id = f"{path_id}_episode_{episode_id:03d}"

        rows: list[dict[str, object]] = []
        for frame in range(len(time)):
            row: dict[str, object] = {
                "sample_id": sample_id,
                "split": split,
                "action_type": ACTION_TYPE,
                "outcome_label": outcome_label,
                "time": float(time[frame]),
                "event_impact": event_impact,
                "event_acceleration_start": event_acceleration_start,
                "event_follow_through_end": event_follow_through_end,
            }
            for signal, values in joint_degrees.items():
                row[f"joint_{signal}"] = float(values[frame])
            for signal, values in muscle_values.items():
                row[f"muscle_{signal}"] = float(values[frame])
            for index, column in enumerate(individual_muscle_columns):
                row[column] = float(np.clip(activations[frame, index], 0.0, 1.0))
            rows.append(row)

        samples.append(ExportedSample(sample_id=sample_id, split=split, outcome_label=outcome_label, rows=rows))

    return samples


def write_samples_csv(samples: Iterable[ExportedSample], output: Path) -> None:
    samples = list(samples)
    extra_columns = sorted({column for sample in samples for row in sample.rows for column in row if column not in CSV_COLUMNS})
    fieldnames = [*CSV_COLUMNS, *extra_columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerows(sample.rows)


def _collect_samples(args: argparse.Namespace) -> list[ExportedSample]:
    samples: list[ExportedSample] = []
    correct_episodes = _parse_episode_selector(args.correct_episodes)
    eval_episodes = _parse_episode_selector(args.eval_episodes)

    for path in args.input:
        samples.extend(
            export_npz_to_samples(
                path,
                correct_episodes=correct_episodes,
                eval_episodes=eval_episodes,
                outcome_label=args.outcome_label,
                impact_time=args.impact_time,
                impact_frame=args.impact_frame,
                include_all_muscles=not args.semantic_muscles_only,
            )
        )
    for path in args.correct_input:
        samples.extend(
            export_npz_to_samples(
                path,
                split_hint="correct",
                outcome_label=args.outcome_label,
                impact_time=args.impact_time,
                impact_frame=args.impact_frame,
                include_all_muscles=not args.semantic_muscles_only,
            )
        )
    for path in args.eval_input:
        samples.extend(
            export_npz_to_samples(
                path,
                split_hint="eval",
                outcome_label=args.outcome_label,
                impact_time=args.impact_time,
                impact_frame=args.impact_frame,
                include_all_muscles=not args.semantic_muscles_only,
            )
        )

    if not samples:
        raise ValueError("provide at least one --input, --correct-input, or --eval-input")
    if not any(sample.split == "correct" for sample in samples):
        raise ValueError("export requires at least one correct sample for BadmintonRAG")
    if not any(sample.split == "eval" for sample in samples):
        raise ValueError("export requires at least one eval sample for BadmintonRAG")
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MyoFullBody forehand-clear rollout npz files to the BadmintonRAG simulation CSV contract."
    )
    parser.add_argument("--input", action="append", type=Path, default=[], help="Rollout npz; default split is episode 0 correct, remaining episodes eval.")
    parser.add_argument("--correct-input", action="append", type=Path, default=[], help="Rollout npz whose episodes are all reference-template samples.")
    parser.add_argument("--eval-input", action="append", type=Path, default=[], help="Rollout npz whose episodes are all diagnosis samples.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--correct-episodes", help="Comma-separated episode ids to mark correct for --input files.")
    parser.add_argument("--eval-episodes", help="Comma-separated episode ids to mark eval for --input files.")
    parser.add_argument("--outcome-label", default=DEFAULT_OUTCOME_LABEL, choices=sorted(SUPPORTED_OUTCOME_LABELS))
    parser.add_argument("--impact-time", type=float, help="Impact time in seconds, applied to every exported sample.")
    parser.add_argument("--impact-frame", type=int, help="Impact frame index, applied to every exported sample.")
    parser.add_argument(
        "--semantic-muscles-only",
        action="store_true",
        help="Only write the compact BadmintonRAG semantic muscle groups, not every individual muscle activation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.impact_time is not None and args.impact_frame is not None:
        parser.error("--impact-time and --impact-frame are mutually exclusive")
    samples = _collect_samples(args)
    write_samples_csv(samples, args.output)
    print(f"Wrote {sum(len(sample.rows) for sample in samples)} rows from {len(samples)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
