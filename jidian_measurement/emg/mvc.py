from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .models import ProcessingConfig, RecordResult, TrignoConfig
from .processing import emg_envelope, stable_window_max
from .profiles import require_mvc_profile
from .storage import atomic_save_npz, atomic_write_json, git_commit_hash, initialize_session, read_json
from .synthetic import generate_synthetic_emg
from .trigno import TrignoClient


MVC_FLATLINE_STD_MV = 1e-5


def _assess_mvc_hard_qc(result: RecordResult) -> dict[str, Any]:
    """Assess only non-negotiable stream and raw-signal integrity gates."""

    raw = np.asarray(result.emg_mV)
    shape_ok = (
        raw.ndim == 2
        and raw.shape[1] == 1
        and raw.shape[0] == result.received_samples
        and result.received_samples > 0
    )
    finite = bool(shape_ok and np.all(np.isfinite(raw)))
    raw_std_mV = float(np.std(raw[:, 0])) if finite else None
    all_zero = bool(finite and np.all(raw[:, 0] == 0))
    flatline = bool(finite and raw_std_mV is not None and raw_std_mV <= MVC_FLATLINE_STD_MV)
    stream_complete = bool(
        result.received_samples == result.expected_samples
        and result.dropped_samples == 0
        and not result.interrupted
        and result.receive_error is None
    )
    failures: list[str] = []
    if not stream_complete:
        failures.append("incomplete_stream")
    if not shape_ok:
        failures.append("invalid_or_empty_target_sensor_shape")
    if shape_ok and not finite:
        failures.append("nonfinite_raw")
    if all_zero:
        failures.append("all_zero_raw")
    if flatline:
        failures.append("flatline_raw")
    return {
        "schema_version": "emg_mvc_hard_qc_v1",
        "hard_qc_pass": not failures,
        "hard_failures": failures,
        "stream_complete": stream_complete,
        "expected_samples": int(result.expected_samples),
        "received_samples": int(result.received_samples),
        "dropped_samples": int(result.dropped_samples),
        "interrupted": bool(result.interrupted),
        "receive_error": result.receive_error,
        "target_sensor_shape_ok": shape_ok,
        "finite_raw": finite,
        "all_zero_raw": all_zero,
        "flatline_raw": flatline,
        "flatline_std_threshold_mV": MVC_FLATLINE_STD_MV,
        "raw_std_mV": raw_std_mV,
    }


def _finalize_mvc_review(
    result: RecordResult,
    *,
    operator_accepted: bool,
    notes: str,
    qc_summary: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Combine operator review with non-overridable stream/signal integrity."""

    notes = str(notes).strip()
    if not operator_accepted and not notes:
        raise ValueError("A non-empty reason is required when an MVC repetition is rejected")
    qc = qc_summary or _assess_mvc_hard_qc(result)
    if bool(qc.get("hard_qc_pass")):
        return bool(operator_accepted), notes
    automatic_reason = (
        "Automatic invalidation: MVC hard QC failed "
        f"({','.join(qc.get('hard_failures', [])) or 'unspecified'}); "
        f"received {result.received_samples}/{result.expected_samples} samples; "
        f"interrupted={result.interrupted}; receive_error={result.receive_error or 'none'}"
    )
    combined = f"{notes}; {automatic_reason}" if notes else automatic_reason
    return False, combined


def _load_existing_valid_repetitions(
    muscle_dir: Path,
    *,
    sensor_id: int,
    processing: ProcessingConfig,
    window_s: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Revalidate saved raw repetitions instead of trusting a stale summary."""

    values: list[float] = []
    records: list[dict[str, Any]] = []
    for rep_dir in sorted(path for path in muscle_dir.glob("rep_*") if path.is_dir()):
        metadata_path = rep_dir / "metadata.json"
        raw_path = rep_dir / "mvc_timeseries.npz"
        if not metadata_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"Incomplete saved MVC repetition bundle: {rep_dir}")
        metadata = read_json(metadata_path)
        records.append(metadata)
        if not bool(metadata.get("valid", False)):
            continue
        try:
            expected_samples = int(metadata["expected_samples"])
            received_samples = int(metadata["received_samples"])
        except (KeyError, TypeError, ValueError):
            print(f"Ignoring legacy MVC repetition without stream counts: {rep_dir}")
            continue
        with np.load(raw_path, allow_pickle=False) as payload:
            raw = np.asarray(payload["raw_emg_mV"], dtype=np.float64)
            fs_hz = float(payload["fs_hz"])
            stream_ids = np.asarray(payload["stream_channel_ids"]).reshape(-1)
        if stream_ids.size != 1 or int(stream_ids[0]) != sensor_id:
            raise ValueError(f"MVC sensor ID mismatch at {raw_path}")
        if not np.isclose(fs_hz, processing.sample_rate_hz, rtol=0.0, atol=1e-9):
            raise ValueError(f"MVC sample rate mismatch at {raw_path}")
        result = RecordResult(
            emg_mV=raw,
            fs_hz=fs_hz,
            stream_channel_ids=stream_ids,
            expected_samples=expected_samples,
            received_samples=received_samples,
            dropped_samples=int(
                metadata.get(
                    "dropped_samples",
                    max(expected_samples - received_samples, 0),
                )
            ),
            start_time="",
            stop_time="",
            interrupted=bool(metadata.get("interrupted", False)),
            receive_error=metadata.get("receive_error"),
        )
        hard_qc = _assess_mvc_hard_qc(result)
        stored_qc = metadata.get("qc")
        if isinstance(stored_qc, dict) and not bool(stored_qc.get("hard_qc_pass", False)):
            hard_qc["hard_qc_pass"] = False
            hard_qc["hard_failures"].append("stored_hard_qc_failed")
        if not hard_qc["hard_qc_pass"]:
            print(f"Ignoring MVC repetition that fails hard QC: {rep_dir} {hard_qc['hard_failures']}")
            continue
        try:
            _, envelope = emg_envelope(raw, processing, processing.mvc_envelope_lowpass_hz)
            value = float(stable_window_max(envelope, fs_hz, window_s)[0])
        except (ValueError, FloatingPointError) as exc:
            print(f"Ignoring MVC repetition that cannot be processed: {rep_dir} ({exc})")
            continue
        if np.isfinite(value) and value > 0:
            values.append(value)
    return values, records


def collect_mvc(
    dataset_root: Path,
    participant_id: str,
    session_id: str,
    profile_id: str,
    protocol_id: str,
    session_metadata: dict[str, Any],
    repetitions: int = 3,
    contraction_s: float = 4.0,
    rest_s: float = 60.0,
    window_s: float = 0.5,
    muscle_slugs: list[str] | None = None,
    dry_run: bool = False,
    interactive: bool = True,
    seed: int = 20260720,
    trigno_config: TrignoConfig | None = None,
    allow_retrospective_profile: bool = False,
) -> Path:
    from .protocols import get_protocol

    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if (
        not np.isfinite(contraction_s)
        or contraction_s <= 0
        or not np.isfinite(rest_s)
        or rest_s < 0
        or not np.isfinite(window_s)
        or window_s <= 0
    ):
        raise ValueError("MVC contraction/window must be positive and rest must be non-negative")
    profile = require_mvc_profile(
        profile_id,
        explicit_retrospective=allow_retrospective_profile,
    )
    protocol = get_protocol(protocol_id)
    profile.validate_handedness(session_metadata.get("handedness", ""))
    session_dir = initialize_session(dataset_root, participant_id, session_id, session_metadata, profile, protocol.to_dict())
    requested = set(muscle_slugs or [])
    selected = [
        channel
        for channel in profile.channels
        if not requested
        or channel.muscle_slug in requested
        or f"{channel.side}_{channel.muscle_slug}" in requested
    ]
    if not selected:
        raise ValueError("No profile channels match the requested --muscle values")
    processing = ProcessingConfig(sample_rate_hz=(trigno_config or TrignoConfig()).sample_rate_hz)
    results_path = session_dir / "mvc" / "mvc_results.json"
    summary: dict[str, Any] = read_json(results_path) if results_path.exists() else {
        "participant_id": participant_id,
        "session_id": session_id,
        "channel_profile_id": profile.profile_id,
        "channel_profile_version": profile.version,
        "channel_profile_snapshot": profile.to_dict(),
        "algorithm": "maximum sliding mean of rectified bandpass envelope",
        "window_s": window_s,
        "processing_config": processing.to_dict(),
        "final_strategy": "maximum_of_valid_repetitions",
        "muscles": {},
        "software_version": __version__,
        "git_commit_hash": git_commit_hash(Path.cwd()),
    }
    trigno = trigno_config or TrignoConfig()
    for channel in selected:
        identity = f"{channel.side}_{channel.muscle_slug}"
        muscle_dir = session_dir / "mvc" / identity
        muscle_dir.mkdir(parents=True, exist_ok=True)
        existing_reps = sorted(muscle_dir.glob("rep_*"))
        attempt = len(existing_reps)
        valid_values, repetition_records = _load_existing_valid_repetitions(
            muscle_dir,
            sensor_id=channel.sensor_id,
            processing=processing,
            window_s=window_s,
        )
        print(f"\nMVC Sensor {channel.sensor_id}: {channel.name_zh} ({channel.abbreviation})")
        print(channel.mvc_instruction_zh)
        while len(valid_values) < repetitions and attempt < repetitions * 3:
            attempt += 1
            if interactive:
                input(f"准备第 {attempt} 次，按 Enter 开始...")
            if dry_run:
                result = generate_synthetic_emg(
                    [channel.sensor_id], contraction_s + 1.0, trigno.sample_rate_hz,
                    baseline_s=0.4, cue_s=0.1, post_s=0.5, seed=seed + channel.sensor_id * 100 + attempt,
                )
            else:
                with TrignoClient(trigno) as client:
                    result = client.record(contraction_s + 1.0, [channel.sensor_id])
            qc_summary = _assess_mvc_hard_qc(result)
            envelope = np.full(np.asarray(result.emg_mV).shape, np.nan, dtype=np.float32)
            value: float | None = None
            signal_processable = (
                qc_summary["target_sensor_shape_ok"]
                and qc_summary["finite_raw"]
                and not qc_summary["all_zero_raw"]
                and not qc_summary["flatline_raw"]
            )
            if signal_processable:
                try:
                    _, envelope = emg_envelope(
                        result.emg_mV,
                        processing,
                        processing.mvc_envelope_lowpass_hz,
                    )
                    value = float(stable_window_max(envelope, result.fs_hz, window_s)[0])
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("stable-window MVC is not finite and positive")
                except (ValueError, FloatingPointError) as exc:
                    qc_summary["hard_qc_pass"] = False
                    qc_summary["hard_failures"].append("mvc_processing_failed")
                    qc_summary["processing_error"] = str(exc)
            operator_accepted = True
            notes = ""
            if interactive:
                value_display = "unavailable" if value is None else f"{value:.5f} mV"
                operator_accepted = input(
                    f"稳定窗口MVC={value_display}，本次有效？[Y/n]: "
                ).strip().lower() not in {"n", "no", "0"}
                notes = input("备注（有效重复可空）: ").strip()
                while not operator_accepted and not notes:
                    notes = input("无效 MVC 重复必须填写原因，请输入备注: ").strip()
            valid, notes = _finalize_mvc_review(
                result,
                operator_accepted=operator_accepted,
                notes=notes,
                qc_summary=qc_summary,
            )
            rep_dir = muscle_dir / f"rep_{attempt:03d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_npz(
                rep_dir / "mvc_timeseries.npz",
                raw_emg_mV=result.emg_mV,
                envelope_mV=envelope,
                time_s=result.time_s,
                sample_index=result.sample_index,
                fs_hz=np.asarray(result.fs_hz),
                stream_channel_ids=result.stream_channel_ids,
            )
            record = {
                "repetition_index": attempt,
                "valid": valid,
                "stable_window_value_mV": value,
                "expected_samples": result.expected_samples,
                "received_samples": result.received_samples,
                "dropped_samples": result.dropped_samples,
                "interrupted": result.interrupted,
                "receive_error": result.receive_error,
                "qc": qc_summary,
                "raw_path": str((rep_dir / "mvc_timeseries.npz").relative_to(session_dir)),
                "notes": notes,
                "post_repetition_rest": None,
            }
            atomic_write_json(rep_dir / "metadata.json", record)
            repetition_records.append(record)
            if valid:
                assert value is not None
                valid_values.append(value)
            from .acquisition import _rest_not_required, _timed_rest_record

            if len(valid_values) >= repetitions:
                record["post_repetition_rest"] = _rest_not_required(
                    reason="target_reached",
                    rest_kind="mvc_repetition",
                )
            elif attempt >= repetitions * 3:
                record["post_repetition_rest"] = _rest_not_required(
                    reason="attempt_limit_reached",
                    rest_kind="mvc_repetition",
                )
            elif dry_run:
                record["post_repetition_rest"] = _rest_not_required(
                    reason="dry_run",
                    rest_kind="mvc_repetition",
                    planned_seconds=rest_s,
                )
            elif rest_s <= 0:
                record["post_repetition_rest"] = _rest_not_required(
                    reason="protocol_zero_rest",
                    rest_kind="mvc_repetition",
                )
            else:
                record["post_repetition_rest"] = _timed_rest_record(
                    rest_s,
                    f"{channel.name_zh} 下一次MVC",
                    rest_kind="mvc_repetition",
                )
            atomic_write_json(rep_dir / "metadata.json", record)
        if len(valid_values) < repetitions:
            raise RuntimeError(f"Could not obtain {repetitions} valid MVC repetitions for {identity}")
        chosen = valid_values[-repetitions:]
        summary["muscles"][identity] = {
            "sensor_id": channel.sensor_id,
            "side": channel.side,
            "muscle_slug": channel.muscle_slug,
            "valid_repetition_values_mV": chosen,
            "maximum_mV": float(np.max(chosen)),
            "median_mV": float(np.median(chosen)),
            "final_normalization_value_mV": float(np.max(chosen)),
            "repetitions": repetition_records,
        }
        atomic_write_json(results_path, summary)
    values = []
    for channel in profile.channels:
        item = summary["muscles"].get(f"{channel.side}_{channel.muscle_slug}")
        values.append(None if item is None else item["final_normalization_value_mV"])
    summary["normalization_values_channel_order_mV"] = values
    atomic_write_json(results_path, summary)
    print(f"MVC results saved: {results_path}")
    return results_path


def compatible_mvc_values(session_dir: Path, profile_id: str, channel_count: int) -> np.ndarray | None:
    path = session_dir / "mvc" / "mvc_results.json"
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("channel_profile_id") != profile_id:
        raise ValueError("MVC channel profile does not match trial channel profile")
    values = payload.get("normalization_values_channel_order_mV")
    if not isinstance(values, list) or len(values) != channel_count or any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)
