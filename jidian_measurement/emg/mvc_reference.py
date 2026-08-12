from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .models import ChannelProfile, ProcessingConfig, RecordResult
from .mvc import _assess_mvc_hard_qc
from .processing import emg_envelope
from .storage import read_json


def _candidate_sessions(session_dir: Path, scope: str) -> list[Path]:
    session_dir = Path(session_dir).resolve()
    if scope == "session":
        return [session_dir]
    if scope != "participant":
        raise ValueError("MVC search scope must be 'session' or 'participant'")
    participant_dir = session_dir.parent
    candidates = sorted(path for path in participant_dir.iterdir() if path.is_dir() and (path / "mvc").exists())
    return candidates or [session_dir]


def participant_mvc_envelope_peaks(
    session_dir: Path,
    profile: ChannelProfile,
    config: ProcessingConfig,
    scope: str = "participant",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Recompute per-muscle MVC peaks from raw MVC repetitions.

    The maximum smoothed-envelope peak across valid repetitions and selected
    sessions is used for each side+muscle identity. Stored legacy summaries are
    deliberately not mixed with a different processing configuration.
    """

    sessions = _candidate_sessions(session_dir, scope)
    values: list[float | None] = []
    channel_sources: list[dict[str, Any]] = []
    compatible_sessions: list[Path] = []
    skipped_sessions: list[dict[str, str]] = []
    for candidate_session in sessions:
        snapshot_path = candidate_session / "channel_profile.json"
        if snapshot_path.exists() and read_json(snapshot_path) != profile.to_dict():
            skipped_sessions.append(
                {"session": str(candidate_session.resolve()), "reason": "channel_profile_mismatch"}
            )
            continue
        compatible_sessions.append(candidate_session)
    for channel in profile.channels:
        identity = f"{channel.side}_{channel.muscle_slug}"
        repetitions: list[dict[str, Any]] = []
        rejected_repetitions: list[dict[str, Any]] = []
        for candidate_session in compatible_sessions:
            for raw_path in sorted((candidate_session / "mvc" / identity).glob("rep_*/mvc_timeseries.npz")):
                metadata_path = raw_path.parent / "metadata.json"
                if not metadata_path.exists():
                    rejected_repetitions.append(
                        {"path": str(raw_path.resolve()), "reason": "missing_metadata"}
                    )
                    continue
                metadata = read_json(metadata_path)
                if not metadata.get("valid", False):
                    continue
                with np.load(raw_path, allow_pickle=False) as payload:
                    raw_key = "raw_emg_mV" if "raw_emg_mV" in payload.files else "emg_mV"
                    raw = np.asarray(payload[raw_key], dtype=np.float64)
                    fs_hz = float(payload["fs_hz"])
                    if not np.isclose(fs_hz, config.sample_rate_hz):
                        raise ValueError(
                            f"MVC sample rate {fs_hz} Hz does not match configured "
                            f"{config.sample_rate_hz} Hz at {raw_path}"
                        )
                    if "stream_channel_ids" in payload.files:
                        stream_ids = np.asarray(payload["stream_channel_ids"]).reshape(-1)
                        if stream_ids.size != 1 or int(stream_ids[0]) != channel.sensor_id:
                            raise ValueError(f"MVC sensor ID mismatch at {raw_path}")
                    else:
                        stream_ids = np.asarray([channel.sensor_id])
                if raw.ndim == 1:
                    raw = raw[:, None]
                if raw.ndim != 2 or raw.shape[1] != 1:
                    raise ValueError(f"MVC repetition must contain exactly one channel: {raw_path}")
                try:
                    expected_samples = int(metadata["expected_samples"])
                    received_samples = int(metadata["received_samples"])
                except (KeyError, TypeError, ValueError):
                    rejected_repetitions.append(
                        {"path": str(raw_path.resolve()), "reason": "missing_stream_counts"}
                    )
                    continue
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
                    rejected_repetitions.append(
                        {
                            "path": str(raw_path.resolve()),
                            "reason": "mvc_hard_qc_failed",
                            "hard_failures": hard_qc["hard_failures"],
                        }
                    )
                    continue
                effective = replace(config, normalization="none")
                _, envelope = emg_envelope(raw, effective, effective.mvc_envelope_lowpass_hz)
                guard = int(round(effective.edge_guard_s * fs_hz))
                usable = envelope[guard:-guard] if guard > 0 and len(envelope) > 2 * guard else envelope
                peak = float(np.max(usable[:, 0]))
                if not np.isfinite(peak) or peak <= 0:
                    continue
                repetitions.append(
                    {
                        "path": str(raw_path.resolve()),
                        "session_id": candidate_session.name,
                        "envelope_peak_mV": peak,
                        "hard_qc": hard_qc,
                    }
                )
        selected = max((item["envelope_peak_mV"] for item in repetitions), default=None)
        values.append(selected)
        channel_sources.append(
            {
                "sensor_id": channel.sensor_id,
                "side": channel.side,
                "muscle_slug": channel.muscle_slug,
                "selected_peak_mV": selected,
                "valid_repetitions": repetitions,
                "rejected_repetitions": rejected_repetitions,
            }
        )
    missing = [item["sensor_id"] for item in channel_sources if item["selected_peak_mV"] is None]
    provenance = {
        "scope": scope,
        "participant_id": session_dir.parent.name,
        "searched_sessions": [str(path.resolve()) for path in compatible_sessions],
        "skipped_sessions": skipped_sessions,
        "algorithm": (
            "maximum 4 Hz envelope peak across operator-valid, "
            "stream-complete, finite, nonzero, non-flatline raw MVC repetitions"
        ),
        "processing_config": config.to_dict(),
        "channels": channel_sources,
        "missing_sensor_ids": missing,
    }
    if missing:
        return None, provenance
    return np.asarray(values, dtype=np.float64), provenance
