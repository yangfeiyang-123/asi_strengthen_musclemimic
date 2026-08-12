from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import ProcessingConfig
from .mvc_reference import participant_mvc_envelope_peaks
from .preprocessing_plot import save_processing_figures
from .preprocessing_qc import assess_preprocessing_quality
from .processing import emg_envelope, preprocess_emg
from .profiles import require_analysis_profile
from .protocols import get_protocol
from .qc import assess_signal_quality, save_preview
from .storage import append_jsonl, atomic_save_npz, atomic_write_json, read_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_processing_config(path: Path | None = None, **overrides: Any) -> ProcessingConfig:
    payload: dict[str, Any] = {}
    if path is not None:
        payload = read_json(path)
        if "processing" in payload:
            payload = payload["processing"]
        if not isinstance(payload, dict):
            raise ValueError("Processing configuration must be a JSON object")
    valid_fields = {item.name for item in fields(ProcessingConfig)}
    unknown = sorted(set(payload) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown processing configuration field(s): {', '.join(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return ProcessingConfig(**payload)


def _raw_trial_paths(session_dir: Path) -> list[Path]:
    paths = sorted((session_dir / "trials").glob("*/trial_*/raw_emg.npz"))
    if not paths:
        raise FileNotFoundError(f"No raw_emg.npz trials found under {session_dir}")
    return paths


def _load_raw_trial(raw_path: Path, expected_channel_ids: tuple[int, ...]) -> dict[str, Any]:
    with np.load(raw_path, allow_pickle=False) as raw:
        emg = np.asarray(raw["emg_mV"], dtype=np.float64)
        fs_hz = float(raw["fs_hz"])
        sample_index = np.asarray(raw["sample_index"]) if "sample_index" in raw.files else np.arange(len(emg))
        time_s = np.asarray(raw["time_s"]) if "time_s" in raw.files else np.arange(len(emg)) / fs_hz
        stream_ids = (
            np.asarray(raw["stream_channel_ids"], dtype=np.int16)
            if "stream_channel_ids" in raw.files else np.asarray(expected_channel_ids, dtype=np.int16)
        )
    if emg.ndim != 2 or emg.shape[1] != len(expected_channel_ids):
        raise ValueError(f"Raw channel count mismatch at {raw_path}")
    if len(sample_index) != len(emg) or len(time_s) != len(emg):
        raise ValueError(f"Raw time/index length mismatch at {raw_path}")
    if not np.array_equal(stream_ids, np.asarray(expected_channel_ids)):
        raise ValueError(f"Raw sensor order differs from channel profile at {raw_path}")
    if len(sample_index) > 1 and (np.any(np.diff(sample_index) <= 0) or np.any(np.diff(time_s) <= 0)):
        raise ValueError(f"Raw sample index/time is not strictly increasing at {raw_path}")
    return {
        "emg_mV": emg,
        "fs_hz": fs_hz,
        "sample_index": sample_index,
        "time_s": time_s,
        "stream_channel_ids": stream_ids,
    }


def _trial_metadata(raw_path: Path) -> dict[str, Any]:
    metadata_path = raw_path.parent / "metadata.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    return {
        "participant_id": raw_path.parents[4].name,
        "session_id": raw_path.parents[3].name,
        "action_id": raw_path.parents[1].name,
        "trial_id": raw_path.parent.name,
        "trial_index": int(raw_path.parent.name.removeprefix("trial_")),
        "valid_for_analysis": True,
    }


def _common_stage_arrays(
    raw: dict[str, Any],
    metadata: dict[str, Any],
    channel_names: list[str],
    muscle_slugs: list[str],
    sides: list[str],
) -> dict[str, Any]:
    return {
        "time_s": raw["time_s"],
        "sample_index": raw["sample_index"],
        "fs_hz": np.asarray(raw["fs_hz"], dtype=np.float64),
        "stream_channel_ids": raw["stream_channel_ids"],
        "channel_names": np.asarray(channel_names),
        "muscle_slugs": np.asarray(muscle_slugs),
        "sides": np.asarray(sides),
        "participant_id": np.asarray(str(metadata.get("participant_id", ""))),
        "session_id": np.asarray(str(metadata.get("session_id", ""))),
        "trial_id": np.asarray(str(metadata.get("trial_id", "unknown_trial"))),
        "trial_index": np.asarray(int(metadata.get("trial_index", 0)), dtype=np.int64),
        "action_id": np.asarray(str(metadata.get("action_id", "unknown"))),
        "action_label": np.asarray(str(metadata.get("action_id", "unknown"))),
    }


def _save_stage_files(
    trial_dir: Path,
    raw: dict[str, Any],
    metadata: dict[str, Any],
    processed: dict[str, Any],
    profile: Any,
) -> dict[str, str]:
    channel_names = [f"S{item.sensor_id} {item.side}:{item.muscle_slug}" for item in profile.channels]
    muscle_slugs = [item.muscle_slug for item in profile.channels]
    sides = [item.side for item in profile.channels]
    common = _common_stage_arrays(raw, metadata, channel_names, muscle_slugs, sides)
    filtered_path = atomic_save_npz(
        trial_dir / "filtered_emg.npz",
        cleaned_emg_mV=processed["cleaned_emg_mV"],
        demeaned_mV=processed["demeaned_mV"],
        bandpassed_mV=processed["bandpassed_mV"],
        notched_mV=processed["notched_mV"],
        filtered_mV=processed["filtered_mV"],
        **common,
    )
    rectified_path = atomic_save_npz(
        trial_dir / "rectified_emg.npz",
        rectified_mV=processed["rectified_mV"],
        **common,
    )
    envelope_path = atomic_save_npz(
        trial_dir / "envelope_emg.npz",
        envelope_mV=processed["envelope_mV"],
        **common,
    )
    normalized_filename = (
        "mvc_normalized_emg.npz"
        if processed["normalization_method"] == "mvc"
        else "normalized_emg.npz"
    )
    normalized_path = atomic_save_npz(
        trial_dir / normalized_filename,
        normalized_envelope=processed["normalized_envelope"],
        **common,
    )
    aggregate_path = atomic_save_npz(
        trial_dir / "processed_emg.npz",
        raw_emg_mV=processed["raw_emg_mV"],
        cleaned_emg_mV=processed["cleaned_emg_mV"],
        demeaned_mV=processed["demeaned_mV"],
        bandpassed_mV=processed["bandpassed_mV"],
        notched_mV=processed["notched_mV"],
        filtered_mV=processed["filtered_mV"],
        rectified_mV=processed["rectified_mV"],
        envelope_mV=processed["envelope_mV"],
        normalized_envelope=processed["normalized_envelope"],
        **common,
    )
    return {
        "raw": str((trial_dir / "raw_emg.npz").resolve()),
        "filtered": str(filtered_path.resolve()),
        "rectified": str(rectified_path.resolve()),
        "envelope": str(envelope_path.resolve()),
        "normalized": str(normalized_path.resolve()),
        "aggregate_compatibility": str(aggregate_path.resolve()),
    }


def preprocess_session(
    session_dir: Path,
    profile_id: str,
    normalization: str | None = None,
    fallback: str | None = None,
    notch_50hz: bool | None = None,
    *,
    config: ProcessingConfig | None = None,
    mvc_scope: str = "participant",
    save_figures: bool = True,
    continue_on_error: bool = False,
) -> list[Path]:
    session_dir = Path(session_dir).resolve()
    profile = require_analysis_profile(profile_id)
    snapshot = read_json(session_dir / "channel_profile.json")
    if snapshot != profile.to_dict():
        raise ValueError("Session channel profile snapshot does not exactly match requested profile")
    raw_paths = _raw_trial_paths(session_dir)
    first_raw = _load_raw_trial(raw_paths[0], profile.channel_ids)
    base_config = config or ProcessingConfig(sample_rate_hz=first_raw["fs_hz"])
    if normalization is not None:
        base_config = replace(base_config, normalization=normalization)  # type: ignore[arg-type]
    if notch_50hz is True:
        base_config = replace(base_config, notch_hz=50.0)
    elif notch_50hz is False:
        base_config = replace(base_config, notch_hz=None)
    if not np.isclose(base_config.sample_rate_hz, first_raw["fs_hz"]):
        raise ValueError(
            f"Configured sample rate {base_config.sample_rate_hz} Hz does not match raw "
            f"sample rate {first_raw['fs_hz']} Hz"
        )
    if fallback not in {None, "dynamic_p95", "none"}:
        raise ValueError("Fallback normalization must be dynamic_p95, none, or omitted")
    mvc_values: np.ndarray | None = None
    mvc_provenance: dict[str, Any] | None = None
    effective_normalization = base_config.normalization
    if base_config.normalization == "mvc":
        mvc_values, mvc_provenance = participant_mvc_envelope_peaks(
            session_dir, profile, base_config, scope=mvc_scope
        )
        atomic_write_json(session_dir / "preprocessing_mvc_reference.json", mvc_provenance)
        if mvc_values is None:
            if fallback not in {"dynamic_p95", "none"}:
                missing = mvc_provenance["missing_sensor_ids"]
                raise FileNotFoundError(
                    f"No complete compatible raw MVC set for sensors {missing}; "
                    "collect MVC or pass an explicit fallback normalization"
                )
            effective_normalization = fallback
    dynamic_reference: np.ndarray | None = None
    if effective_normalization == "dynamic_p95":
        session_envelopes: list[np.ndarray] = []
        reference_config = replace(base_config, normalization="none")
        for raw_path in raw_paths:
            raw = _load_raw_trial(raw_path, profile.channel_ids)
            _, envelope = emg_envelope(raw["emg_mV"], reference_config)
            session_envelopes.append(envelope)
        dynamic_reference = np.percentile(np.vstack(session_envelopes), 95, axis=0)
    outputs: list[Path] = []
    failures: list[dict[str, Any]] = []
    log_path = session_dir / "preprocessing.log.jsonl"
    for raw_path in raw_paths:
        started_at = _utc_now()
        append_jsonl(log_path, {"timestamp": started_at, "status": "started", "raw_path": str(raw_path.resolve())})
        try:
            raw = _load_raw_trial(raw_path, profile.channel_ids)
            if not np.isclose(raw["fs_hz"], base_config.sample_rate_hz):
                raise ValueError(f"Mixed sample rates are not allowed: {raw_path}")
            metadata = _trial_metadata(raw_path)
            metadata_snapshot = metadata.get("channel_profile_snapshot")
            if metadata_snapshot is not None and metadata_snapshot != profile.to_dict():
                raise ValueError(f"Profile mismatch at {raw_path.parent}")
            processed = preprocess_emg(
                raw["emg_mV"],
                base_config,
                mvc_values=mvc_values,
                explicit_fallback=fallback,
                dynamic_reference_values=dynamic_reference,
            )
            quality = assess_preprocessing_quality(
                processed["raw_emg_mV"],
                processed["bandpassed_mV"],
                processed["filtered_mV"],
                processed["envelope_mV"],
                processed["normalized_envelope"],
                processed["missing_value_report"],
                processed["outlier_report"],
                base_config,
                profile,
                processed["normalization_method"],
            )
            output_files = _save_stage_files(raw_path.parent, raw, metadata, processed, profile)
            qc_path = atomic_write_json(raw_path.parent / "preprocessing_qc.json", quality)
            figure_paths = (
                save_processing_figures(
                    raw_path.parent,
                    raw["time_s"],
                    processed,
                    profile,
                    str(metadata.get("trial_id", raw_path.parent.name)),
                    processed["normalization_method"],
                )
                if save_figures else []
            )
            reference_scope = (
                "participant_raw_mvc_across_sessions"
                if processed["normalization_method"] == "mvc"
                else "session_all_trials"
                if processed["normalization_method"] == "dynamic_p95"
                else "none"
            )
            processing_record = {
                "processing_format_version": 2,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "status": "completed",
                "pipeline_order": [
                    "repair_missing_values",
                    "repair_isolated_outliers",
                    "remove_channel_mean",
                    "butterworth_bandpass",
                    "iir_notch",
                    "full_wave_rectification",
                    "butterworth_lowpass_envelope",
                    "normalization",
                ],
                "processing_config": base_config.to_dict(),
                "normalization_method": processed["normalization_method"],
                "fallback_method": processed["fallback_method"],
                "normalization_reference_scope": reference_scope,
                "normalization_reference_values_mV": processed["normalization_reference_values_mV"],
                "mvc_reference_manifest": (
                    str((session_dir / "preprocessing_mvc_reference.json").resolve())
                    if mvc_provenance is not None else None
                ),
                "channel_profile_id": profile.profile_id,
                "channel_profile_snapshot": profile.to_dict(),
                "raw_source": "raw_emg.npz",
                "raw_source_path": str(raw_path.resolve()),
                "metadata_preserved": [
                    "time_s", "sample_index", "participant_id", "session_id",
                    "action_id", "trial_id", "trial_index", "channel_names",
                    "muscle_slugs", "sides", "stream_channel_ids",
                ],
                "edge_handling": {
                    "method": "odd-reflection padding with zero-phase filtering" if base_config.zero_phase else "causal filtering",
                    "guard_samples_for_qc_and_figures": processed["edge_guard_samples"],
                    "samples_trimmed": 0,
                },
                "quality_summary": {
                    "qc_pass": quality["qc_pass"],
                    "analysis_ready": quality["analysis_ready"],
                    "critical_channel_count": quality["critical_channel_count"],
                },
                "output_files": output_files,
                "preprocessing_qc": str(qc_path.resolve()),
                "figures": [str(path.resolve()) for path in figure_paths],
            }
            atomic_write_json(raw_path.parent / "processing.json", processing_record)
            aggregate = raw_path.parent / "processed_emg.npz"
            outputs.append(aggregate)
            append_jsonl(
                log_path,
                {
                    "timestamp": processing_record["completed_at"],
                    "status": "completed",
                    "raw_path": str(raw_path.resolve()),
                    "output_path": str(aggregate.resolve()),
                    "analysis_ready": quality["analysis_ready"],
                    "critical_channel_count": quality["critical_channel_count"],
                },
            )
            if processed["fallback_method"]:
                print(f"WARNING: {raw_path.parent.name} used explicit normalization fallback {processed['fallback_method']}")
        except Exception as exc:
            failure = {
                "raw_path": str(raw_path.resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            append_jsonl(
                log_path,
                {
                    "timestamp": _utc_now(),
                    "status": "failed",
                    **failure,
                },
            )
            if not continue_on_error:
                raise
    atomic_write_json(
        session_dir / "preprocessing_session_summary.json",
        {
            "created_at": _utc_now(),
            "session_path": str(session_dir),
            "profile_id": profile_id,
            "processing_config": base_config.to_dict(),
            "discovered_trials": len(raw_paths),
            "completed_trials": len(outputs),
            "failed_trials": len(failures),
            "failures": failures,
        },
    )
    return outputs


def preprocess_dataset(
    dataset_root: Path,
    profile_id: str,
    config: ProcessingConfig,
    fallback: str | None = None,
    participant_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    mvc_scope: str = "participant",
    save_figures: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    require_analysis_profile(profile_id)
    dataset_root = Path(dataset_root).resolve()
    sessions = sorted(path.parent for path in dataset_root.glob("*/*/session.json"))
    selected: list[Path] = []
    for session in sessions:
        metadata = read_json(session / "session.json")
        if metadata.get("channel_profile_id") != profile_id:
            continue
        if participant_ids and metadata.get("participant_id") not in participant_ids:
            continue
        if session_ids and metadata.get("session_id") not in session_ids:
            continue
        selected.append(session)
    if not selected:
        raise FileNotFoundError("No sessions matched the requested dataset/profile filters")
    results: list[dict[str, Any]] = []
    for session in selected:
        try:
            outputs = preprocess_session(
                session,
                profile_id,
                fallback=fallback,
                config=config,
                mvc_scope=mvc_scope,
                save_figures=save_figures,
                continue_on_error=continue_on_error,
            )
            session_summary = read_json(session / "preprocessing_session_summary.json")
            failed_trials = int(session_summary["failed_trials"])
            results.append(
                {
                    "session": str(session),
                    "status": "completed" if failed_trials == 0 else "partial",
                    "trial_count": len(outputs),
                    "failed_trial_count": failed_trials,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "session": str(session),
                    "status": "failed",
                    "trial_count": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                raise
    summary = {
        "created_at": _utc_now(),
        "dataset_root": str(dataset_root),
        "profile_id": profile_id,
        "processing_config": config.to_dict(),
        "fallback_normalization": fallback,
        "session_count": len(results),
        "completed_sessions": sum(item["status"] == "completed" for item in results),
        "partial_sessions": sum(item["status"] == "partial" for item in results),
        "failed_sessions": sum(item["status"] in {"partial", "failed"} for item in results),
        "sessions": results,
    }
    atomic_write_json(dataset_root / "preprocessing_batch_summary.json", summary)
    return summary


def qc_session(session_dir: Path, profile_id: str, show_preview: bool = False) -> list[Path]:
    profile = require_analysis_profile(profile_id)
    protocol_payload = read_json(session_dir / "protocol.json")
    protocol = get_protocol(protocol_payload["protocol_id"])
    outputs: list[Path] = []
    for raw_path in sorted((session_dir / "trials").glob("*/trial_*/raw_emg.npz")):
        metadata = read_json(raw_path.parent / "metadata.json")
        if metadata["channel_profile_snapshot"] != profile.to_dict():
            raise ValueError(f"Profile mismatch at {raw_path.parent}")
        action = protocol.action(metadata["action_id"])
        with np.load(raw_path, allow_pickle=False) as raw:
            emg = raw["emg_mV"]
            fs_hz = float(raw["fs_hz"])
        qc = assess_signal_quality(
            emg,
            fs_hz,
            int(metadata["expected_samples"]),
            int(round(action.baseline_s * fs_hz)),
            int(round((action.baseline_s + action.cue_s) * fs_hz)),
            int(round((action.baseline_s + action.cue_s + action.duration_s) * fs_hz)),
        )
        qc_path = atomic_write_json(raw_path.parent / "qc.json", qc)
        save_preview(raw_path.parent / "preview.png", emg, fs_hz, profile, metadata["trial_id"], show_preview)
        outputs.append(qc_path)
    return outputs
