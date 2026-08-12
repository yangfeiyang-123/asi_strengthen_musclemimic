#!/usr/bin/env python
# ruff: noqa: RUF001
"""Prepare and run the human EMG review required before PEASD training.

The tool deliberately separates machine diagnostics from human decisions:

* ``prepare`` fingerprints every source file, computes unclipped per-trial
  diagnostics, and writes plots plus an unanswered questionnaire;
* ``wizard`` walks the reviewer through every mapping, trial, comparable
  channel, and known S9 risk while saving after every answer;
* ``finalize`` emits a reviewed mapping and action-specific review JSON files
  only when no answer remains pending and the reviewer types the attestation;
* ``validate`` re-runs the exact fail-closed validators used by the tube builder.

Values above MVC are always shown, never clipped, and never become an automatic
exclusion recommendation.  The script cannot decide anatomy or signal quality
for the reviewer and cannot launch training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.physiology.emg_reference import (
    EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION,
    EMG_TUBE_MIN_TRIALS,
)

try:
    from scripts.build_emg_reference_tube import (
        DEFAULT_MAPPING,
        DEFAULT_SESSION,
        _load_verified_trial_qc_review,
        _normalized_trials,
        _require_verified_mapping,
        _sha256,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts.build_emg_reference_tube":
        raise
    # Direct ``python scripts/review_emg_for_training.py`` puts ``scripts/``
    # rather than the repository root on sys.path.
    from build_emg_reference_tube import (  # type: ignore[no-redef]
        DEFAULT_MAPPING,
        DEFAULT_SESSION,
        _load_verified_trial_qc_review,
        _normalized_trials,
        _require_verified_mapping,
        _sha256,
    )

REVIEW_PACKET_SCHEMA_VERSION = "emg_human_review_packet_v2"
REVIEW_ANSWERS_SCHEMA_VERSION = "emg_human_review_answers_v2"
REVIEW_VALIDATION_SCHEMA_VERSION = "emg_human_review_validation_v2"
DEFAULT_ACTIONS = (
    "forehand_high_clear",
    "forehand_lift_footwork",
    "china_jump_high_clear",
)
DEFAULT_REVIEW_ROOT = Path("artifacts/emg_human_review_v2")
ATTESTATION = "我已逐项审查并对这些决定负责"
S9_SENSOR_ID = 9


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    os.replace(temporary, path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _nonempty(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _channel_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _quality_grade(ratio: float) -> tuple[str, float]:
    if ratio <= 1.20:
        return "good", 1.0
    if ratio <= 1.50:
        return "questionable", 0.7
    if ratio <= 2.00:
        return "unreliable", 0.4
    return "invalid_for_absolute_amplitude", 0.2


def _qc_channel_map(qc: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for entry in qc.get("channels", ()):
        if isinstance(entry, Mapping):
            result[int(entry.get("sensor_id", -1))] = entry
    return result


def _trial_diagnostics(
    trial_dir: Path,
    *,
    expected_channel_names: Sequence[str],
) -> dict[str, Any]:
    normalized_path = trial_dir / "mvc_normalized_emg.npz"
    qc_path = trial_dir / "preprocessing_qc.json"
    metadata_path = trial_dir / "metadata.json"
    if not qc_path.is_file():
        raise FileNotFoundError(f"missing preprocessing_qc.json: {trial_dir}")
    qc = _read_json(qc_path)
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    with np.load(normalized_path, allow_pickle=False) as payload:
        values = np.asarray(payload["normalized_envelope"], dtype=np.float64)
        channel_names = tuple(_channel_name(item) for item in payload["channel_names"])
        fs_hz = float(payload["fs_hz"])
    if values.ndim != 2 or values.shape[1] != len(expected_channel_names):
        raise ValueError(
            f"{normalized_path}: expected [sample, {len(expected_channel_names)}], "
            f"found {values.shape}"
        )
    if channel_names != tuple(expected_channel_names):
        raise ValueError(f"channel order differs from the mapping/profile: {trial_dir}")

    finite = bool(np.all(np.isfinite(values)))
    nonnegative = bool(np.all(values >= 0.0)) if finite else False
    safe_values = values if finite else np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    p95 = np.percentile(safe_values, 95.0, axis=0)
    p99 = np.percentile(safe_values, 99.0, axis=0)
    maximum = np.max(safe_values, axis=0)
    mean = np.mean(safe_values, axis=0)
    std = np.std(safe_values, axis=0)
    qc_by_sensor = _qc_channel_map(qc)
    channels: list[dict[str, Any]] = []
    critical_failures: list[str] = []
    warnings: list[str] = []
    for index, name in enumerate(expected_channel_names):
        sensor_id = index + 1
        qc_entry = qc_by_sensor.get(sensor_id, {})
        channel_critical = sorted(
            str(item) for item in qc_entry.get("critical_failures", ())
        )
        channel_warnings = sorted(str(item) for item in qc_entry.get("warnings", ()))
        critical_failures.extend(f"S{sensor_id}:{item}" for item in channel_critical)
        warnings.extend(f"S{sensor_id}:{item}" for item in channel_warnings)
        quality, confidence = _quality_grade(float(p99[index]))
        channels.append(
            {
                "sensor_id": sensor_id,
                "channel_name": name,
                "p95_percent_mvc": float(p95[index]),
                "p99_over_mvc": float(p99[index]),
                "max_over_mvc": float(maximum[index]),
                "mean_percent_mvc": float(mean[index]),
                "std_percent_mvc": float(std[index]),
                "mvc_quality": quality,
                "amplitude_confidence": confidence,
                "mvc_exceedance_is_exclusion_reason": False,
                "warnings": channel_warnings,
                "critical_failures": channel_critical,
            }
        )

    s9 = safe_values[:, S9_SENSOR_ID - 1]
    window = max(round(0.2 * len(s9)), 1)
    s9_start = float(np.median(s9[:window]))
    s9_end = float(np.median(s9[-window:]))
    s9_end_over_start = s9_end / max(s9_start, 1e-12)
    s9_manual_attention = bool(s9_start > 1e-8 and s9_end_over_start < 0.25)
    signal_hard_failure = bool(
        (not finite)
        or (not nonnegative)
        or critical_failures
        or not bool(qc.get("analysis_ready", qc.get("qc_pass", False)))
        or metadata.get("valid_for_analysis") is False
    )
    recommendation = (
        "manual_qc_resolution_required"
        if signal_hard_failure
        else "manual_review_s9_progressive_decay"
        if s9_manual_attention
        else "include_after_visual_confirmation"
    )
    return {
        "trial_id": trial_dir.name,
        "trial_dir": str(trial_dir.resolve()),
        "mvc_normalized_emg_path": str(normalized_path.resolve()),
        "mvc_normalized_emg_sha256": _sha256(normalized_path),
        "preprocessing_qc_path": str(qc_path.resolve()),
        "preprocessing_qc_sha256": _sha256(qc_path),
        "metadata_path": str(metadata_path.resolve()) if metadata_path.is_file() else None,
        "metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "metadata_valid_for_analysis": metadata.get("valid_for_analysis"),
        "sample_count": int(values.shape[0]),
        "sample_rate_hz": fs_hz,
        "duration_s": float(values.shape[0] / fs_hz),
        "all_finite": finite,
        "all_nonnegative": nonnegative,
        "preprocessing_analysis_ready": bool(
            qc.get("analysis_ready", qc.get("qc_pass", False))
        ),
        "critical_failures": sorted(set(critical_failures)),
        "warnings": sorted(set(warnings)),
        "super_mvc_channels": [
            item["channel_name"] for item in channels if item["max_over_mvc"] > 1.0
        ],
        "channels": channels,
        "s9_review": {
            "sensor_id": S9_SENSOR_ID,
            "start_window_median_percent_mvc": s9_start,
            "end_window_median_percent_mvc": s9_end,
            "end_over_start_ratio": s9_end_over_start,
            "manual_attention": s9_manual_attention,
            "automatic_exclusion": False,
        },
        "machine_recommendation": recommendation,
        "machine_recommendation_is_not_a_human_decision": True,
    }


def _action_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channel_names = [item["channel_name"] for item in trials[0]["channels"]]
    p99 = np.asarray(
        [[channel["p99_over_mvc"] for channel in trial["channels"]] for trial in trials],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [[channel["max_over_mvc"] for channel in trial["channels"]] for trial in trials],
        dtype=np.float64,
    )
    s9_p99 = np.asarray(
        [trial["channels"][S9_SENSOR_ID - 1]["p99_over_mvc"] for trial in trials],
        dtype=np.float64,
    )
    edge_count = min(3, len(s9_p99))
    first_s9 = float(np.median(s9_p99[:edge_count]))
    last_s9 = float(np.median(s9_p99[-edge_count:]))
    return {
        "trial_count": len(trials),
        "machine_recommendation_counts": {
            label: sum(trial["machine_recommendation"] == label for trial in trials)
            for label in sorted({str(trial["machine_recommendation"]) for trial in trials})
        },
        "channel_summary": [
            {
                "sensor_id": index + 1,
                "channel_name": name,
                "trial_p99_min": float(np.min(p99[:, index])),
                "trial_p99_median": float(np.median(p99[:, index])),
                "trial_p99_max": float(np.max(p99[:, index])),
                "trial_absolute_max": float(np.max(maximum[:, index])),
                "trials_with_max_over_mvc": int(np.sum(maximum[:, index] > 1.0)),
                "trials_with_preprocessing_critical": int(
                    sum(bool(trial["channels"][index]["critical_failures"]) for trial in trials)
                ),
            }
            for index, name in enumerate(channel_names)
        ],
        "s9_end_over_start_by_trial": [
            {
                "trial_id": trial["trial_id"],
                "end_over_start_ratio": trial["s9_review"]["end_over_start_ratio"],
                "manual_attention": trial["s9_review"]["manual_attention"],
            }
            for trial in trials
        ],
        "s9_across_trials": {
            "first_trials_median_p99": first_s9,
            "last_trials_median_p99": last_s9,
            "last_over_first_ratio": last_s9 / max(first_s9, 1e-12),
            "p99_by_trial": [
                {
                    "trial_id": trial["trial_id"],
                    "p99_over_mvc": float(s9_p99[index]),
                }
                for index, trial in enumerate(trials)
            ],
            "automatic_exclusion": False,
        },
    }


def _session_s9_context(session_root: Path) -> dict[str, Any]:
    """Summarize S9 across the whole recording session in chronological order."""

    records: list[dict[str, Any]] = []
    for trial_dir in sorted((session_root / "trials").glob("*/trial_*")):
        normalized_path = trial_dir / "mvc_normalized_emg.npz"
        qc_path = trial_dir / "preprocessing_qc.json"
        metadata_path = trial_dir / "metadata.json"
        if not normalized_path.is_file() or not qc_path.is_file():
            continue
        qc = _read_json(qc_path)
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        with np.load(normalized_path, allow_pickle=False) as payload:
            values = np.asarray(
                payload["normalized_envelope"][:, S9_SENSOR_ID - 1],
                dtype=np.float64,
            )
        qc_s9 = _qc_channel_map(qc).get(S9_SENSOR_ID, {})
        records.append(
            {
                "action": trial_dir.parent.name,
                "trial_id": trial_dir.name,
                "start_time": str(metadata.get("start_time", "")),
                "metadata_valid_for_analysis": metadata.get("valid_for_analysis"),
                "p99_over_mvc": float(np.percentile(values, 99.0)),
                "mean_percent_mvc": float(np.mean(values)),
                "filtered_rms_mV": float(qc_s9.get("filtered_rms_mV", float("nan"))),
                "critical_failures": sorted(
                    str(item) for item in qc_s9.get("critical_failures", ())
                ),
                "warnings": sorted(str(item) for item in qc_s9.get("warnings", ())),
                "mvc_normalized_emg_path": str(normalized_path.resolve()),
                "mvc_normalized_emg_sha256": _sha256(normalized_path),
                "preprocessing_qc_path": str(qc_path.resolve()),
                "preprocessing_qc_sha256": _sha256(qc_path),
                "metadata_path": str(metadata_path.resolve())
                if metadata_path.is_file()
                else None,
                "metadata_sha256": _sha256(metadata_path)
                if metadata_path.is_file()
                else None,
            }
        )
    records.sort(key=lambda item: (item["start_time"], item["action"], item["trial_id"]))
    for index, record in enumerate(records):
        record["sequence_index"] = index
    if not records:
        raise ValueError("session has no S9 trial evidence")
    filtered_rms = np.asarray([item["filtered_rms_mV"] for item in records], dtype=np.float64)
    if not np.all(np.isfinite(filtered_rms)) or np.any(filtered_rms < 0.0):
        raise ValueError("session S9 filtered RMS diagnostics are absent or invalid")
    edge_count = min(5, len(records))
    early = float(np.median(filtered_rms[:edge_count]))
    late = float(np.median(filtered_rms[-edge_count:]))
    mvc_reference_path = session_root / "preprocessing_mvc_reference.json"
    mvc_reference = _read_json(mvc_reference_path)
    mvc_channels = mvc_reference.get("channels")
    if not isinstance(mvc_channels, list) or len(mvc_channels) < S9_SENSOR_ID:
        raise ValueError("MVC reference does not contain S9")
    s9_mvc = mvc_channels[S9_SENSOR_ID - 1]
    return {
        "scope": "all_processed_trials_in_session_chronological",
        "record_count": len(records),
        "early_record_count": edge_count,
        "early_filtered_rms_mV_median": early,
        "late_filtered_rms_mV_median": late,
        "late_over_early_filtered_rms_ratio": late / max(early, 1e-12),
        "critical_record_count": sum(bool(item["critical_failures"]) for item in records),
        "first_critical_sequence_index": next(
            (item["sequence_index"] for item in records if item["critical_failures"]),
            None,
        ),
        "records": records,
        "mvc_reference": {
            "path": str(mvc_reference_path.resolve()),
            "sha256": _sha256(mvc_reference_path),
            "selected_peak_mV": float(s9_mvc["selected_peak_mV"]),
            "valid_repetitions": list(s9_mvc.get("valid_repetitions", ())),
        },
        "automatic_exclusion": False,
        "human_review_required": True,
    }


def _plot_session_s9(context: Mapping[str, Any], destination: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "musclemimic-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = context["records"]
    x = np.arange(len(records))
    p99 = np.asarray([item["p99_over_mvc"] for item in records], dtype=np.float64)
    rms = np.asarray([item["filtered_rms_mV"] for item in records], dtype=np.float64)
    figure, axis_p99 = plt.subplots(figsize=(18, 6))
    axis_p99.plot(x, p99, marker="o", markersize=2.5, label="S9 P99/MVC")
    axis_p99.axhline(1.0, color="#ff7f0e", linestyle="--", linewidth=0.8)
    axis_p99.set_ylabel("S9 P99 / current MVC (unclipped)")
    axis_p99.set_xlabel("chronological processed trial index")
    axis_rms = axis_p99.twinx()
    axis_rms.plot(x, rms, color="#2ca02c", alpha=0.7, label="S9 filtered RMS mV")
    axis_rms.set_ylabel("S9 filtered RMS (mV)")
    critical = [index for index, item in enumerate(records) if item["critical_failures"]]
    if critical:
        axis_p99.scatter(
            critical,
            p99[critical],
            color="#d62728",
            marker="x",
            label="old QC critical",
        )
    axis_p99.grid(alpha=0.2)
    handles_a, labels_a = axis_p99.get_legend_handles_labels()
    handles_b, labels_b = axis_rms.get_legend_handles_labels()
    axis_p99.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _plot_trial(trial: Mapping[str, Any], destination: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "musclemimic-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(str(trial["mvc_normalized_emg_path"]))
    with np.load(source, allow_pickle=False) as payload:
        values = np.asarray(payload["normalized_envelope"], dtype=np.float64)
    progress = np.linspace(0.0, 100.0, len(values), dtype=np.float64)
    figure, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for index, axis in enumerate(axes.flat):
        channel = trial["channels"][index]
        axis.plot(progress, values[:, index], color="#1f77b4", linewidth=0.7)
        axis.axhline(1.0, color="#ff7f0e", linewidth=0.8, linestyle="--")
        axis.axhline(2.0, color="#d62728", linewidth=0.8, linestyle=":")
        axis.set_title(
            f"S{index + 1} P99={channel['p99_over_mvc']:.2f} "
            f"max={channel['max_over_mvc']:.2f}",
            fontsize=8,
        )
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"{trial['trial_id']} — unclipped percent-MVC (1.0=100% MVC)",
        fontsize=13,
    )
    figure.supxlabel("trial progress (%)")
    figure.supylabel("current MVC reference ratio (unclipped)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _write_report(packet: Mapping[str, Any], output_dir: Path) -> Path:
    lines = [
        "# EMG 人工审查包",
        "",
        "> 红线：超过 100%/200% MVC 不是排除理由。橙/红横线只帮助识别当前 MVC "
        "reference 的比例；请检查波形、采集记录和 QC，不要裁剪。",
        "",
        "## S9 session 级证据",
        "",
        f"- 全 session processed trials: {packet['session_s9_context']['record_count']}",
        f"- 旧 QC critical records: {packet['session_s9_context']['critical_record_count']}",
        "- late/early filtered RMS: "
        f"{packet['session_s9_context']['late_over_early_filtered_rms_ratio']:.4f}",
        f"- [时序图]({Path(str(packet['session_s9_context']['plot_path'])).resolve().relative_to(output_dir.resolve()).as_posix()})"
        if packet["session_s9_context"].get("plot_path")
        else "- 未生成 S9 时序图",
        "",
        "## Mapping",
        "",
    ]
    for entry in packet["mapping_questions"]:
        if entry["mapping_status"] == "mapped":
            target = ", ".join(entry["simulation_actuators"])
            lines.append(
                f"- **{entry['emg_channel']}** → `{target}`；待核："
                f"{entry['mapping_uncertainty']}"
            )
        else:
            lines.append(
                f"- **{entry['emg_channel']}**：排除候选；理由：{entry['exclusion_reason']}"
            )
    for action in packet["actions"]:
        lines.extend(["", f"## {action['action']}", ""])
        summary = action["summary"]
        lines.append(f"- trials：{summary['trial_count']}")
        lines.append(
            "- 机器提示（不是决定）："
            + json.dumps(summary["machine_recommendation_counts"], ensure_ascii=False)
        )
        lines.append("- 每个 trial 必须逐图查看：")
        lines.append("")
        for trial in action["trials"]:
            plot = trial.get("plot_path")
            relative_plot = (
                Path(str(plot)).resolve().relative_to(output_dir.resolve()).as_posix()
                if plot
                else "未生成"
            )
            lines.append(
                f"  - `{trial['trial_id']}`：建议 `{trial['machine_recommendation']}`；"
                f"S9 end/start={trial['s9_review']['end_over_start_ratio']:.3f}；"
                f"超 MVC 通道数={len(trial['super_mvc_channels'])}；"
                f"[波形]({relative_plot})"
            )
    lines.extend(
        [
            "",
            "## 开始填写",
            "",
            "```bash",
            ".venv/bin/python scripts/review_emg_for_training.py wizard \\",
            f"  --packet {output_dir / 'review_packet.json'}",
            "```",
            "",
            "向导会逐题保存，可随时输入 `q` 退出，之后运行同一命令继续。",
        ]
    )
    path = output_dir / "review_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mapping_questions(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for channel in mapping.get("channels", ()):
        status = str(channel.get("mapping_status", ""))
        base = {
            "sensor_id": int(channel["sensor_id"]),
            "emg_channel": str(channel["emg_channel"]),
            "side": str(channel["side"]),
            "muscle_slug": str(channel["muscle_slug"]),
            "mapping_status": status,
        }
        if status == "mapped":
            base.update(
                {
                    "simulation_actuators": list(channel["simulation_actuators"]),
                    "weights": [float(item) for item in channel["weights"]],
                    "mapping_uncertainty": str(channel["mapping_uncertainty"]),
                    "question": (
                        "根据电极位置记录和模型 actuator 清单，此 observation mapping "
                        "是否可用于 measured-subspace 对照？"
                    ),
                    "allowed_decisions": ["high", "medium", "low", "defer", "reject"],
                }
            )
        else:
            base.update(
                {
                    "exclusion_reason": str(channel.get("exclusion_reason", "")),
                    "question": "是否确认当前模型没有可核验同源 actuator，因此保留原始通道但排除比较？",
                    "allowed_decisions": ["accept_exclusion", "defer", "reject"],
                }
            )
        questions.append(base)
    return questions


def _empty_answers(packet: Mapping[str, Any], *, packet_path: Path) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_ANSWERS_SCHEMA_VERSION,
        "packet_path": str(packet_path.resolve()),
        "packet_sha256": str(packet["packet_sha256"]),
        "reviewer": {
            "reviewer_id": "",
            "role": "",
            "evidence_statement": "",
        },
        "mapping_decisions": [
            {
                "sensor_id": question["sensor_id"],
                "decision": "pending",
                "reason": "",
            }
            for question in packet["mapping_questions"]
        ],
        "actions": [
            {
                "action": action["action"],
                "trial_decisions": [
                    {
                        "trial_id": trial["trial_id"],
                        "decision": "pending",
                        "reason": "",
                        "qc_resolution_evidence": [],
                    }
                    for trial in action["trials"]
                ],
                "channel_decisions": [
                    {
                        "emg_channel": question["emg_channel"],
                        "decision": "pending",
                        "reason": "",
                    }
                    for question in packet["mapping_questions"]
                    if question["mapping_status"] == "mapped"
                ],
                "risk_decisions": [
                    {
                        "risk_id": "s9_progressive_near_flatline",
                        "decision": "pending",
                        "reason": "",
                        "evidence": [],
                    }
                ],
            }
            for action in packet["actions"]
        ],
        "attestation": "",
        "completed_at": None,
    }


def prepare_review_packet(
    *,
    session_root: Path,
    mapping_path: Path,
    output_dir: Path,
    actions: Sequence[str] = DEFAULT_ACTIONS,
    make_plots: bool = True,
) -> tuple[Path, Path]:
    session = session_root.expanduser().resolve(strict=True)
    mapping_source = mapping_path.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve()
    existing_packet_path = output / "review_packet.json"
    existing_answers_path = output / "review_answers.json"
    if existing_packet_path.exists():
        existing = _read_json(existing_packet_path)
        _validate_packet_identity(existing, existing_packet_path)
        if Path(str(existing["session_root"])).resolve() != session:
            raise ValueError("existing review packet belongs to a different session")
        if Path(str(existing["mapping_source"]["path"])).resolve() != mapping_source:
            raise ValueError("existing review packet belongs to a different mapping")
        if tuple(item["action"] for item in existing["actions"]) != tuple(actions):
            raise ValueError("existing review packet belongs to a different action set")
        if not existing_answers_path.exists():
            _atomic_write_json(
                existing_answers_path,
                _empty_answers(existing, packet_path=existing_packet_path),
            )
        _write_report(existing, output)
        return existing_packet_path, existing_answers_path
    mapping = _read_json(mapping_source)
    profile_path = session / "channel_profile.json"
    profile = _read_json(profile_path)
    profile_channels = profile.get("channels")
    if not isinstance(profile_channels, list) or len(profile_channels) != 16:
        raise ValueError("review tool requires the exact 16-channel acquisition profile")
    channel_names = tuple(
        f"S{int(item['sensor_id'])} {item['side']}:{item['muscle_slug']}"
        for item in profile_channels
    )
    expected_mapping_names = tuple(
        str(item["emg_channel"]) for item in mapping.get("channels", ())
    )
    if expected_mapping_names != channel_names:
        raise ValueError("mapping and acquisition profile channel order differ")
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("actions must be non-empty and unique")

    action_entries: list[dict[str, Any]] = []
    for action in actions:
        trial_dirs = sorted((session / "trials" / action).glob("trial_*"))
        trial_dirs = [
            path for path in trial_dirs if (path / "mvc_normalized_emg.npz").is_file()
        ]
        if not trial_dirs:
            raise FileNotFoundError(f"no processed trials for action {action!r}")
        trials = [
            _trial_diagnostics(path, expected_channel_names=channel_names)
            for path in trial_dirs
        ]
        if make_plots:
            for trial in trials:
                plot_path = output / "plots" / action / f"{trial['trial_id']}.png"
                _plot_trial(trial, plot_path)
                trial["plot_path"] = str(plot_path.resolve())
                trial["plot_sha256"] = _sha256(plot_path)
        else:
            for trial in trials:
                trial["plot_path"] = None
                trial["plot_sha256"] = None
        action_entries.append(
            {
                "action": action,
                "trials": trials,
                "summary": _action_summary(trials),
            }
        )

    session_s9_context = _session_s9_context(session)
    if make_plots:
        s9_plot = output / "plots" / "session_s9_chronology.png"
        _plot_session_s9(session_s9_context, s9_plot)
        session_s9_context["plot_path"] = str(s9_plot.resolve())
        session_s9_context["plot_sha256"] = _sha256(s9_plot)
    else:
        session_s9_context["plot_path"] = None
        session_s9_context["plot_sha256"] = None

    packet: dict[str, Any] = {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "session_root": str(session),
        "mapping_source": {
            "path": str(mapping_source),
            "sha256": _sha256(mapping_source),
            "mapping_id": str(mapping.get("mapping_id", "")),
        },
        "channel_profile_source": {
            "path": str(profile_path.resolve()),
            "sha256": _sha256(profile_path),
            "profile_id": str(profile.get("profile_id", "")),
        },
        "policy": {
            "super_mvc": "warning_only_never_clip_never_auto_exclude",
            "signal_quality_and_mvc_quality_are_separate": True,
            "mapping_review_is_anatomical_not_signal_qc": True,
            "machine_recommendations_are_not_human_decisions": True,
        },
        "mapping_questions": _mapping_questions(mapping),
        "session_s9_context": session_s9_context,
        "actions": action_entries,
    }
    packet["packet_sha256"] = _json_sha256(packet)
    packet_path = _atomic_write_json(output / "review_packet.json", packet)
    answers_path = output / "review_answers.json"
    if answers_path.exists():
        existing = _read_json(answers_path)
        if existing.get("packet_sha256") != packet["packet_sha256"]:
            raise ValueError(
                "existing review_answers.json belongs to different source bytes; "
                "choose a fresh output directory"
            )
    else:
        _atomic_write_json(
            answers_path,
            _empty_answers(packet, packet_path=packet_path),
        )
    _write_report(packet, output)
    return packet_path, answers_path


def _validate_packet_identity(packet: Mapping[str, Any], packet_path: Path) -> None:
    if packet.get("schema_version") != REVIEW_PACKET_SCHEMA_VERSION:
        raise ValueError("unsupported review packet schema")
    declared = str(packet.get("packet_sha256", ""))
    material = dict(packet)
    material.pop("packet_sha256", None)
    if declared != _json_sha256(material):
        raise ValueError(f"review packet content hash mismatch: {packet_path}")
    for source_field in ("mapping_source", "channel_profile_source"):
        source = packet[source_field]
        path = Path(str(source["path"])).resolve(strict=True)
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"{source_field} bytes changed after packet preparation")
    session_s9 = packet["session_s9_context"]
    mvc_source = session_s9["mvc_reference"]
    mvc_path = Path(str(mvc_source["path"])).resolve(strict=True)
    if _sha256(mvc_path) != mvc_source["sha256"]:
        raise ValueError("session S9 MVC reference changed after packet preparation")
    for record in session_s9["records"]:
        for path_key, hash_key in (
            ("mvc_normalized_emg_path", "mvc_normalized_emg_sha256"),
            ("preprocessing_qc_path", "preprocessing_qc_sha256"),
        ):
            source_path = Path(str(record[path_key])).resolve(strict=True)
            if _sha256(source_path) != record[hash_key]:
                raise ValueError(
                    f"session S9 source changed after packet preparation: {source_path}"
                )
        if record.get("metadata_path"):
            metadata_path = Path(str(record["metadata_path"])).resolve(strict=True)
            if _sha256(metadata_path) != record["metadata_sha256"]:
                raise ValueError(
                    f"session S9 metadata changed after packet preparation: {metadata_path}"
                )
    if session_s9.get("plot_path"):
        s9_plot = Path(str(session_s9["plot_path"])).resolve(strict=True)
        if _sha256(s9_plot) != session_s9["plot_sha256"]:
            raise ValueError("session S9 plot changed after packet preparation")
    for action in packet["actions"]:
        for trial in action["trials"]:
            for path_key, hash_key in (
                ("mvc_normalized_emg_path", "mvc_normalized_emg_sha256"),
                ("preprocessing_qc_path", "preprocessing_qc_sha256"),
            ):
                path = Path(str(trial[path_key])).resolve(strict=True)
                if _sha256(path) != trial[hash_key]:
                    raise ValueError(
                        f"source bytes changed after packet preparation: {path}"
                    )
            if trial.get("metadata_path"):
                metadata_path = Path(str(trial["metadata_path"])).resolve(strict=True)
                if _sha256(metadata_path) != trial["metadata_sha256"]:
                    raise ValueError(
                        f"metadata changed after packet preparation: {metadata_path}"
                    )
            plot = trial.get("plot_path")
            if plot:
                plot_path = Path(str(plot)).resolve(strict=True)
                if _sha256(plot_path) != trial["plot_sha256"]:
                    raise ValueError(f"review plot changed after packet preparation: {plot_path}")


def _answer_lookup(
    entries: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[Any, Mapping[str, Any]]:
    result: dict[Any, Mapping[str, Any]] = {}
    for entry in entries:
        identity = entry.get(key)
        if identity in result:
            raise ValueError(f"duplicate review answer for {key}={identity!r}")
        result[identity] = entry
    return result


def _require_complete_answers(
    packet: Mapping[str, Any],
    answers: Mapping[str, Any],
) -> None:
    if answers.get("schema_version") != REVIEW_ANSWERS_SCHEMA_VERSION:
        raise ValueError("unsupported review answers schema")
    if answers.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("review answers belong to a different packet")
    reviewer = answers.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise ValueError("reviewer identity is missing")
    for field in ("reviewer_id", "role", "evidence_statement"):
        _nonempty(reviewer.get(field), field=f"reviewer.{field}")
    if answers.get("attestation") != ATTESTATION:
        raise ValueError(f"final attestation must exactly equal: {ATTESTATION}")

    mapping_answers = _answer_lookup(answers.get("mapping_decisions", ()), key="sensor_id")
    if set(mapping_answers) != {
        question["sensor_id"] for question in packet["mapping_questions"]
    }:
        raise ValueError("mapping decisions do not match the packet")
    for question in packet["mapping_questions"]:
        answer = mapping_answers[question["sensor_id"]]
        if answer.get("decision") not in question["allowed_decisions"]:
            raise ValueError(f"invalid mapping answer for {question['emg_channel']}")
        if answer.get("decision") in {"pending", "defer", "reject"}:
            raise ValueError(f"mapping review unresolved for {question['emg_channel']}")
        _nonempty(answer.get("reason"), field=f"mapping reason {question['emg_channel']}")

    action_answers = _answer_lookup(answers.get("actions", ()), key="action")
    if set(action_answers) != {action["action"] for action in packet["actions"]}:
        raise ValueError("action review answers do not match the packet")
    comparable_names = {
        question["emg_channel"]
        for question in packet["mapping_questions"]
        if question["mapping_status"] == "mapped"
    }
    for action in packet["actions"]:
        answer = action_answers[action["action"]]
        trial_answers = _answer_lookup(answer.get("trial_decisions", ()), key="trial_id")
        if set(trial_answers) != {trial["trial_id"] for trial in action["trials"]}:
            raise ValueError(f"trial answers differ from {action['action']} packet")
        trial_sources = {trial["trial_id"]: trial for trial in action["trials"]}
        included_count = 0
        for trial_id, entry in trial_answers.items():
            if entry.get("decision") not in {"include", "exclude"}:
                raise ValueError(f"trial review unresolved: {action['action']}/{trial_id}")
            _nonempty(entry.get("reason"), field=f"trial reason {trial_id}")
            if entry.get("decision") == "include":
                included_count += 1
                trial = trial_sources[trial_id]
                if not trial["all_finite"] or not trial["all_nonnegative"]:
                    raise ValueError(
                        f"non-finite/negative signal cannot be human-overridden: "
                        f"{action['action']}/{trial_id}"
                    )
                needs_resolution = bool(
                    not trial["preprocessing_analysis_ready"]
                    or trial["critical_failures"]
                    or trial["metadata_valid_for_analysis"] is False
                )
                resolution = entry.get("qc_resolution_evidence")
                if needs_resolution and (
                    not isinstance(resolution, list) or not resolution
                ):
                    raise ValueError(
                        f"included flagged trial requires explicit QC-resolution evidence: "
                        f"{action['action']}/{trial_id}"
                    )
                if isinstance(resolution, list):
                    for index, item in enumerate(resolution):
                        _nonempty(item, field=f"trial QC evidence[{index}]")
        if included_count < EMG_TUBE_MIN_TRIALS:
            raise ValueError(
                f"{action['action']} retains {included_count} trials; at least "
                f"{EMG_TUBE_MIN_TRIALS} are required"
            )
        channel_answers = _answer_lookup(
            answer.get("channel_decisions", ()), key="emg_channel"
        )
        if set(channel_answers) != comparable_names:
            raise ValueError(f"channel answers differ from {action['action']} packet")
        for name, entry in channel_answers.items():
            if entry.get("decision") != "include_after_review":
                raise ValueError(
                    f"channel {name!r} unresolved; use a new mapping/profile to exclude it"
                )
            _nonempty(entry.get("reason"), field=f"channel reason {name}")
        risk_answers = _answer_lookup(answer.get("risk_decisions", ()), key="risk_id")
        risk = risk_answers.get("s9_progressive_near_flatline")
        if not isinstance(risk, Mapping) or risk.get("decision") not in {
            "accepted_after_review",
            "mitigated",
        }:
            raise ValueError(f"S9 risk remains unresolved for {action['action']}")
        _nonempty(risk.get("reason"), field="S9 risk reason")
        evidence = risk.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"S9 risk evidence is empty for {action['action']}")
        for index, item in enumerate(evidence):
            _nonempty(item, field=f"S9 risk evidence[{index}]")


def finalize_reviews(
    *,
    packet_path: Path,
    answers_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    packet_source = packet_path.expanduser().resolve(strict=True)
    answer_source = answers_path.expanduser().resolve(strict=True)
    packet = _read_json(packet_source)
    answers = _read_json(answer_source)
    _validate_packet_identity(packet, packet_source)
    _require_complete_answers(packet, answers)
    output = output_dir.expanduser().resolve()
    existing_mapping = output / "reviewed_mapping.json"
    expected_review_paths = [
        output / str(action["action"]) / "emg_trial_qc_review.json"
        for action in packet["actions"]
    ]
    existing_outputs = [existing_mapping, *expected_review_paths]
    if all(path.is_file() for path in existing_outputs):
        validation = validate_review_outputs(
            review_dir=output,
            packet_path=packet_source,
        )
        result = {"mapping": existing_mapping, "validation": validation}
        result.update(
            {
                str(action["action"]): path
                for action, path in zip(
                    packet["actions"], expected_review_paths, strict=True
                )
            }
        )
        return result
    if any(path.exists() for path in existing_outputs):
        raise ValueError(
            "final review output is partial; preserve it for audit and choose a fresh output directory"
        )
    reviewer = answers["reviewer"]
    completed_at = _utc_now()
    evidence = [
        f"review_packet:{packet_source}",
        f"review_packet_sha256:{packet['packet_sha256']}",
        f"reviewer_id:{reviewer['reviewer_id']}",
        f"reviewer_role:{reviewer['role']}",
        f"reviewed_at:{completed_at}",
        f"reviewer_statement:{reviewer['evidence_statement']}",
    ]

    mapping = _read_json(Path(str(packet["mapping_source"]["path"])))
    mapping_answers = _answer_lookup(answers["mapping_decisions"], key="sensor_id")
    for channel in mapping["channels"]:
        answer = mapping_answers[int(channel["sensor_id"])]
        if channel["mapping_status"] == "mapped":
            channel["mapping_confidence"] = str(answer["decision"])
            channel["mapping_review_reason"] = str(answer["reason"]).strip()
        else:
            channel["exclusion_review_status"] = "verified"
            channel["exclusion_review_reason"] = str(answer["reason"]).strip()
    mapping["review_status"] = "verified"
    mapping["training_enabled"] = True
    mapping["evaluation_mode"] = "verified_measured_subspace_mapping"
    mapping["review_evidence"] = evidence
    mapping["review_record"] = {
        "schema_version": REVIEW_ANSWERS_SCHEMA_VERSION,
        "reviewer_id": str(reviewer["reviewer_id"]),
        "reviewer_role": str(reviewer["role"]),
        "reviewed_at": completed_at,
        "packet_sha256": str(packet["packet_sha256"]),
        "attestation": ATTESTATION,
    }
    mapping_path = _atomic_write_json(output / "reviewed_mapping.json", mapping)
    _require_verified_mapping(_read_json(mapping_path))
    mapping_sha256 = _sha256(mapping_path)

    action_answers = _answer_lookup(answers["actions"], key="action")
    outputs: dict[str, Path] = {"mapping": mapping_path}
    for action in packet["actions"]:
        action_id = str(action["action"])
        answer = action_answers[action_id]
        trial_answers = _answer_lookup(answer["trial_decisions"], key="trial_id")
        trial_sources = {trial["trial_id"]: trial for trial in action["trials"]}
        review = {
            "schema_version": EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION,
            "action": action_id,
            "review_status": "verified",
            "training_enabled": True,
            "reviewer_id": str(reviewer["reviewer_id"]),
            "reviewed_at": completed_at,
            "review_evidence": evidence,
            "mapping_sha256": mapping_sha256,
            "trial_decisions": [
                {
                    "trial_id": trial_id,
                    "decision": str(trial_answers[trial_id]["decision"]),
                    "reason": str(trial_answers[trial_id]["reason"]).strip(),
                    "mvc_normalized_emg_sha256": trial_sources[trial_id][
                        "mvc_normalized_emg_sha256"
                    ],
                    "preprocessing_qc_sha256": trial_sources[trial_id][
                        "preprocessing_qc_sha256"
                    ],
                    "qc_resolution_evidence": list(
                        trial_answers[trial_id].get("qc_resolution_evidence", ())
                    ),
                }
                for trial_id in sorted(trial_sources)
            ],
            "channel_decisions": [
                {
                    "emg_channel": str(entry["emg_channel"]),
                    "decision": "include_after_review",
                    "reason": str(entry["reason"]).strip(),
                }
                for entry in answer["channel_decisions"]
            ],
            "risk_decisions": [copy.deepcopy(item) for item in answer["risk_decisions"]],
        }
        review_path = _atomic_write_json(
            output / action_id / "emg_trial_qc_review.json",
            review,
        )
        trials = _normalized_trials(Path(str(packet["session_root"])), action_id)
        comparable_names = [
            str(item["emg_channel"])
            for item in mapping["channels"]
            if item["mapping_status"] == "mapped"
        ]
        _load_verified_trial_qc_review(
            review_path,
            action=action_id,
            mapping_sha256=mapping_sha256,
            trials=trials,
            channel_names=comparable_names,
        )
        outputs[action_id] = review_path

    validation_path = validate_review_outputs(
        review_dir=output,
        packet_path=packet_source,
    )
    outputs["validation"] = validation_path
    answers["completed_at"] = completed_at
    _atomic_write_json(answer_source, answers)
    return outputs


def validate_review_outputs(*, review_dir: Path, packet_path: Path) -> Path:
    output = review_dir.expanduser().resolve(strict=True)
    packet_source = packet_path.expanduser().resolve(strict=True)
    packet = _read_json(packet_source)
    _validate_packet_identity(packet, packet_source)
    mapping_path = output / "reviewed_mapping.json"
    mapping = _read_json(mapping_path)
    _require_verified_mapping(mapping)
    mapping_sha256 = _sha256(mapping_path)
    action_results: list[dict[str, Any]] = []
    comparable_names = [
        str(item["emg_channel"])
        for item in mapping["channels"]
        if item["mapping_status"] == "mapped"
    ]
    for action in packet["actions"]:
        action_id = str(action["action"])
        review_path = output / action_id / "emg_trial_qc_review.json"
        trials = _normalized_trials(Path(str(packet["session_root"])), action_id)
        included, binding = _load_verified_trial_qc_review(
            review_path,
            action=action_id,
            mapping_sha256=mapping_sha256,
            trials=trials,
            channel_names=comparable_names,
        )
        action_results.append(
            {
                "action": action_id,
                "review_path": str(review_path),
                "review_sha256": _sha256(review_path),
                "included_trial_count": len(included),
                "excluded_trial_count": len(trials) - len(included),
                "binding_review_sha256": binding["review_sha256"],
                "passed": True,
            }
        )
    report = {
        "schema_version": REVIEW_VALIDATION_SCHEMA_VERSION,
        "validated_at": _utc_now(),
        "passed": True,
        "packet_path": str(packet_source),
        "packet_sha256": str(packet["packet_sha256"]),
        "mapping_path": str(mapping_path),
        "mapping_sha256": mapping_sha256,
        "actions": action_results,
        "next_step": (
            "Build each v2 tube with --verified, this reviewed mapping, and the "
            "matching action review; then run the Stage1 tube gate."
        ),
    }
    return _atomic_write_json(output / "review_validation.json", report)


def _prompt(
    message: str,
    *,
    allowed: Mapping[str, str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> str:
    while True:
        suffix = f" [{'/'.join(allowed)}]" if allowed else ""
        answer = input_fn(f"{message}{suffix}: ").strip()
        if answer.lower() == "q":
            raise KeyboardInterrupt
        if allowed is None:
            if answer:
                return answer
        else:
            normalized = answer.lower()
            if normalized in allowed:
                return allowed[normalized]
        print("输入无效；输入 q 可保存并退出。")


def _wizard_mapping(
    packet: Mapping[str, Any],
    answers: dict[str, Any],
    *,
    save: Callable[[], None],
    input_fn: Callable[[str], str],
) -> None:
    by_sensor = _answer_lookup(answers["mapping_decisions"], key="sensor_id")
    print("\n=== A. 解剖 observation mapping 审查 ===")
    print("这一步判断电极通道与模型 actuator 是否可比，不判断 MVC 大小。")
    print("high=电极位置和一对一同源关系已确认。")
    print("medium=总体同源，但存在 compartment 聚合或可解释串扰。")
    print("low=仍可作为 measured-subspace observation，但不确定性明显，必须写理由。")
    print("如果你无法依据贴片记录/模型清单判断，请选 defer，不要猜。")
    for question in packet["mapping_questions"]:
        answer = by_sensor[question["sensor_id"]]
        if answer["decision"] != "pending":
            continue
        print(f"\n{question['emg_channel']}")
        if question["mapping_status"] == "mapped":
            print(
                "模型目标："
                + ", ".join(question["simulation_actuators"])
                + f"；权重={question['weights']}"
            )
            print(f"已知不确定性：{question['mapping_uncertainty']}")
            decision = _prompt(
                "可信度：h=高，m=中，l=低，d=暂缓，r=拒绝/需新 mapping",
                allowed={"h": "high", "m": "medium", "l": "low", "d": "defer", "r": "reject"},
                input_fn=input_fn,
            )
        else:
            print(f"当前排除理由：{question['exclusion_reason']}")
            decision = _prompt(
                "a=确认排除，d=暂缓，r=拒绝/需寻找同源 actuator",
                allowed={"a": "accept_exclusion", "d": "defer", "r": "reject"},
                input_fn=input_fn,
            )
        reason = _prompt("请写下依据/理由（不能只写 OK）", input_fn=input_fn)
        answer["decision"] = decision
        answer["reason"] = reason
        save()


def _wizard_actions(
    packet: Mapping[str, Any],
    answers: dict[str, Any],
    *,
    save: Callable[[], None],
    input_fn: Callable[[str], str],
) -> None:
    action_answers = _answer_lookup(answers["actions"], key="action")
    for action in packet["actions"]:
        action_id = action["action"]
        answer = action_answers[action_id]
        print(f"\n=== B. {action_id} trial 审查 ===")
        trial_answers = _answer_lookup(answer["trial_decisions"], key="trial_id")
        for trial in action["trials"]:
            target = trial_answers[trial["trial_id"]]
            if target["decision"] != "pending":
                continue
            print(f"\n{trial['trial_id']}  波形：{trial.get('plot_path')}")
            print(
                f"机器提示={trial['machine_recommendation']}；"
                f"hard failures={trial['critical_failures']}；"
                f"S9 end/start={trial['s9_review']['end_over_start_ratio']:.3f}"
            )
            if trial["super_mvc_channels"]:
                print(
                    "超 MVC（只报告，不得据此排除）："
                    + ", ".join(trial["super_mvc_channels"])
                )
            decision = _prompt(
                "查看波形/记录后：i=纳入，e=排除，d=暂缓",
                allowed={"i": "include", "e": "exclude", "d": "defer"},
                input_fn=input_fn,
            )
            reason = _prompt("请写下纳入/排除依据", input_fn=input_fn)
            target["decision"] = decision
            target["reason"] = reason
            needs_resolution = bool(
                not trial["preprocessing_analysis_ready"]
                or trial["critical_failures"]
                or trial["metadata_valid_for_analysis"] is False
            )
            if decision == "include" and needs_resolution:
                evidence = _prompt(
                    "该 trial 有旧 QC/metadata 标记；填写你复核后仍纳入的证据（分号分隔）",
                    input_fn=input_fn,
                )
                target["qc_resolution_evidence"] = [
                    item.strip() for item in evidence.split(";") if item.strip()
                ]
            save()

        print(f"\n=== C. {action_id} comparable channel 审查 ===")
        channel_answers = _answer_lookup(answer["channel_decisions"], key="emg_channel")
        summary_by_name = {
            item["channel_name"]: item for item in action["summary"]["channel_summary"]
        }
        for name, target in channel_answers.items():
            if target["decision"] != "pending":
                continue
            summary = summary_by_name[name]
            print(
                f"\n{name}: trial P99 median={summary['trial_p99_median']:.3f}, "
                f"max={summary['trial_absolute_max']:.3f}, "
                f"QC critical trials={summary['trials_with_preprocessing_critical']}"
            )
            if summary["trials_with_max_over_mvc"]:
                print("注意：超 MVC 只降低幅值可信度，不是排除理由。")
            decision = _prompt(
                "i=审查后保留，d=暂缓，n=需要新 mapping/profile 才能排除",
                allowed={"i": "include_after_review", "d": "defer", "n": "needs_new_mapping"},
                input_fn=input_fn,
            )
            reason = _prompt("请写下通道判断依据", input_fn=input_fn)
            target["decision"] = decision
            target["reason"] = reason
            save()

        print(f"\n=== D. {action_id} S9 专项 ===")
        risk = answer["risk_decisions"][0]
        if risk["decision"] == "pending":
            session_s9 = packet["session_s9_context"]
            print(
                "全 session S9: "
                f"records={session_s9['record_count']}, "
                f"old-QC-critical={session_s9['critical_record_count']}, "
                f"late/early RMS={session_s9['late_over_early_filtered_rms_ratio']:.3f}, "
                f"图={session_s9.get('plot_path')}"
            )
            print("逐 trial S9 end/start：")
            for item in action["summary"]["s9_end_over_start_by_trial"]:
                print(
                    f"  {item['trial_id']}: {item['end_over_start_ratio']:.3f}"
                    + ("  <-- 需重点查看" if item["manual_attention"] else "")
                )
            across = action["summary"]["s9_across_trials"]
            print(
                "跨 trial S9 P99: "
                f"前段中位数={across['first_trials_median_p99']:.4f}, "
                f"后段中位数={across['last_trials_median_p99']:.4f}, "
                f"后/前={across['last_over_first_ratio']:.3f}"
            )
            decision = _prompt(
                "a=看过证据后接受，m=已采取明确缓解措施，d=暂缓",
                allowed={"a": "accepted_after_review", "m": "mitigated", "d": "defer"},
                input_fn=input_fn,
            )
            reason = _prompt("说明 S9 判断及处理", input_fn=input_fn)
            evidence = _prompt(
                "填写证据（采集记录/图路径/复核说明，可用分号分隔）",
                input_fn=input_fn,
            )
            risk["decision"] = decision
            risk["reason"] = reason
            risk["evidence"] = [item.strip() for item in evidence.split(";") if item.strip()]
            save()


def run_wizard(
    *,
    packet_path: Path,
    answers_path: Path | None = None,
    final_output_dir: Path | None = None,
    input_fn: Callable[[str], str] = input,
) -> dict[str, Path] | None:
    packet_source = packet_path.expanduser().resolve(strict=True)
    packet = _read_json(packet_source)
    _validate_packet_identity(packet, packet_source)
    answer_source = (
        answers_path.expanduser().resolve()
        if answers_path is not None
        else packet_source.with_name("review_answers.json")
    )
    answers = _read_json(answer_source)

    def save() -> None:
        _atomic_write_json(answer_source, answers)

    try:
        reviewer = answers["reviewer"]
        if not reviewer["reviewer_id"]:
            reviewer["reviewer_id"] = _prompt("你的 reviewer ID/姓名缩写", input_fn=input_fn)
            reviewer["role"] = _prompt("你的角色（PI/实验员/解剖顾问等）", input_fn=input_fn)
            reviewer["evidence_statement"] = _prompt(
                "你依据了哪些记录/知识（简述）", input_fn=input_fn
            )
            save()
        _wizard_mapping(packet, answers, save=save, input_fn=input_fn)
        _wizard_actions(packet, answers, save=save, input_fn=input_fn)
        answers["attestation"] = _prompt(
            f"全部完成后请输入：{ATTESTATION}", input_fn=input_fn
        )
        save()
        target = (
            final_output_dir.expanduser().resolve()
            if final_output_dir is not None
            else packet_source.parent / "final"
        )
        return finalize_reviews(
            packet_path=packet_source,
            answers_path=answer_source,
            output_dir=target,
        )
    except KeyboardInterrupt:
        save()
        print(f"\n已保存进度：{answer_source}")
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build diagnostics and unanswered review files")
    prepare.add_argument(
        "--session-root",
        type=Path,
        default=Path("jidian_measurement/data") / DEFAULT_SESSION,
    )
    prepare.add_argument("--mapping", type=Path, default=Path(DEFAULT_MAPPING))
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_ROOT)
    prepare.add_argument("--action", action="append", dest="actions")
    prepare.add_argument("--no-plots", action="store_true")

    wizard = subparsers.add_parser("wizard", help="answer every review question interactively")
    wizard.add_argument("--packet", type=Path, required=True)
    wizard.add_argument("--answers", type=Path, default=None)
    wizard.add_argument("--output-dir", type=Path, default=None)

    run = subparsers.add_parser("run", help="prepare then immediately enter the wizard")
    run.add_argument(
        "--session-root",
        type=Path,
        default=Path("jidian_measurement/data") / DEFAULT_SESSION,
    )
    run.add_argument("--mapping", type=Path, default=Path(DEFAULT_MAPPING))
    run.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_ROOT)
    run.add_argument("--action", action="append", dest="actions")
    run.add_argument("--no-plots", action="store_true")

    finalize = subparsers.add_parser("finalize", help="finalize a completed answers JSON")
    finalize.add_argument("--packet", type=Path, required=True)
    finalize.add_argument("--answers", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="revalidate finalized review outputs")
    validate.add_argument("--review-dir", type=Path, required=True)
    validate.add_argument("--packet", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command in {"prepare", "run"}:
        actions = tuple(args.actions) if args.actions else DEFAULT_ACTIONS
        packet_path, answers_path = prepare_review_packet(
            session_root=args.session_root,
            mapping_path=args.mapping,
            output_dir=args.output_dir,
            actions=actions,
            make_plots=not args.no_plots,
        )
        print(f"review_packet: {packet_path}")
        print(f"review_answers: {answers_path}")
        print(f"review_report: {packet_path.with_name('review_report.md')}")
        if args.command == "prepare":
            return 0
        outputs = run_wizard(
            packet_path=packet_path,
            answers_path=answers_path,
            final_output_dir=args.output_dir / "final",
        )
        if outputs is not None:
            for key, path in outputs.items():
                print(f"{key}: {path}")
        return 0
    if args.command == "wizard":
        outputs = run_wizard(
            packet_path=args.packet,
            answers_path=args.answers,
            final_output_dir=args.output_dir,
        )
        if outputs is not None:
            for key, path in outputs.items():
                print(f"{key}: {path}")
        return 0
    if args.command == "finalize":
        outputs = finalize_reviews(
            packet_path=args.packet,
            answers_path=args.answers,
            output_dir=args.output_dir,
        )
        for key, path in outputs.items():
            print(f"{key}: {path}")
        return 0
    if args.command == "validate":
        report = validate_review_outputs(
            review_dir=args.review_dir,
            packet_path=args.packet,
        )
        print(f"validation_report: {report}")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
