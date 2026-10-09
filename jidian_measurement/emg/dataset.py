from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample

from .event_annotation import ALLOWED_ANNOTATION_SOURCES, verify_committed_annotation
from .profiles import require_analysis_profile
from .storage import atomic_save_npz, atomic_write_json, dataset_sha256, read_json


def _event_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"Event-aware cropping requires events.csv: {path.parent}")
    events: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("event_name", "")).strip()
            events.setdefault(name, []).append({key: str(value or "") for key, value in row.items()})
    return events


def _required_event(
    rows: dict[str, list[dict[str, str]]],
    name: str,
    trial_dir: Path,
) -> dict[str, str]:
    matches = rows.get(name, [])
    if len(matches) != 1:
        raise ValueError(
            f"Strict event cropping requires exactly one {name!r} row, found {len(matches)}: {trial_dir}"
        )
    row = matches[0]
    if not row.get("sample_index", "").strip():
        raise ValueError(f"Strict event cropping requires an annotated {name!r}: {trial_dir}")
    return row


def _event_sample(row: dict[str, str], name: str, trial_dir: Path) -> int:
    try:
        value = int(row["sample_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sample_index for {name!r}: {trial_dir}") from exc
    if value < 0:
        raise ValueError(f"Negative sample_index for {name!r}: {trial_dir}")
    return value


def _profile_signature(snapshot: dict[str, Any]) -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (int(channel["sensor_id"]), str(channel["side"]), str(channel["muscle_slug"]))
        for channel in snapshot["channels"]
    )


def build_synergy_dataset(
    dataset_root: Path,
    output_path: Path,
    profile_id: str,
    protocol_id: str,
    only_valid: bool = True,
    scope: str = "all",
    action_ids: list[str] | None = None,
    participant_ids: list[str] | None = None,
    crop_mode: str = "annotated_movement_events",
    time_normalize_samples: int | None = None,
) -> Path:
    profile = require_analysis_profile(profile_id)
    expected_signature = _profile_signature(profile.to_dict())
    candidates = sorted(dataset_root.glob("*/*/trials/*/trial_*/processed_emg.npz"))
    if not candidates:
        raise FileNotFoundError("No processed_emg.npz files found; run preprocess first")
    segments: list[np.ndarray] = []
    trials: list[dict[str, Any]] = []
    reference_processing: dict[str, Any] | None = None
    normalization_references: dict[str, dict[str, Any]] = {}
    boundary = 0
    boundaries = [0]
    for processed_path in candidates:
        trial_dir = processed_path.parent
        metadata = read_json(trial_dir / "metadata.json")
        participant = metadata["participant_id"]
        if participant_ids and participant not in participant_ids:
            continue
        if metadata["protocol_id"] != protocol_id:
            continue
        if only_valid and not metadata["valid_for_analysis"]:
            continue
        action = metadata["action_id"]
        category = "primitive" if action in {
            "quiet_stance", "split_step", "forehand_lunge_and_return",
            "single_leg_countermovement_jump_land", "trunk_rotation",
            "shadow_forehand_high_clear", "shadow_forehand_lift", "china_jump_shadow",
            "forehand_lift_footwork_shadow",
        } else "complete"
        if scope in {"primitive", "complete"} and category != scope:
            continue
        if action_ids and action not in action_ids:
            continue
        snapshot = metadata["channel_profile_snapshot"]
        if metadata["channel_profile_id"] != profile_id or _profile_signature(snapshot) != expected_signature:
            raise ValueError(f"Refusing to merge a different or reordered channel profile: {trial_dir}")
        processing = read_json(trial_dir / "processing.json")
        preprocessing_qc_path = trial_dir / "preprocessing_qc.json"
        preprocessing_analysis_ready: bool | None = None
        if only_valid and not preprocessing_qc_path.exists():
            raise ValueError(
                f"--only-valid requires preprocessing_qc.json and cannot infer analysis readiness: {trial_dir}"
            )
        if preprocessing_qc_path.exists():
            preprocessing_analysis_ready = bool(read_json(preprocessing_qc_path).get("analysis_ready", False))
            if only_valid and not preprocessing_analysis_ready:
                continue

        normalization_method = str(processing["normalization_method"])
        normalization_scope = processing.get("normalization_reference_scope")
        normalization_values = processing.get("normalization_reference_values_mV")
        if normalization_method == "mvc":
            reference_key = f"participant:{participant}"
            expected_scope = "participant_raw_mvc_across_sessions"
        elif normalization_method == "dynamic_p95":
            reference_key = f"session:{participant}/{metadata['session_id']}"
            expected_scope = "session_all_trials"
        elif normalization_method == "none":
            reference_key = "none"
            expected_scope = "none"
        else:
            raise ValueError(f"Unknown normalization method at {trial_dir}: {normalization_method!r}")
        if normalization_scope != expected_scope:
            raise ValueError(
                f"Normalization scope {normalization_scope!r} is incompatible with "
                f"method {normalization_method!r} at {trial_dir}"
            )
        if normalization_method in {"mvc", "dynamic_p95"}:
            values = np.asarray(normalization_values, dtype=np.float64)
            if (
                values.shape != (len(profile.channels),)
                or not np.all(np.isfinite(values))
                or np.any(values <= 0)
            ):
                raise ValueError(
                    f"{normalization_method} reference must contain one finite positive value per channel: {trial_dir}"
                )
            canonical_values: list[float] | None = values.tolist()
        else:
            if normalization_values is not None and normalization_values != []:
                raise ValueError(f"Unnormalized trial unexpectedly records denominator values: {trial_dir}")
            canonical_values = None

        reference_entry = normalization_references.get(reference_key)
        if reference_entry is None:
            reference_entry = {
                "normalization_method": normalization_method,
                "normalization_reference_scope": expected_scope,
                "normalization_reference_values_mV": canonical_values,
                "participants": [],
                "sessions": [],
                "mvc_reference_manifests": [],
            }
            normalization_references[reference_key] = reference_entry
        elif reference_entry["normalization_reference_values_mV"] != canonical_values:
            raise ValueError(
                f"Normalization denominator changed within {reference_key}; reprocess that scope consistently: {trial_dir}"
            )
        if participant not in reference_entry["participants"]:
            reference_entry["participants"].append(participant)
        session_identity = f"{participant}/{metadata['session_id']}"
        if session_identity not in reference_entry["sessions"]:
            reference_entry["sessions"].append(session_identity)
        manifest = processing.get("mvc_reference_manifest")
        if manifest and manifest not in reference_entry["mvc_reference_manifests"]:
            reference_entry["mvc_reference_manifests"].append(manifest)

        processing_signature = {
            "processing_config": processing["processing_config"],
            "normalization_method": normalization_method,
            "fallback_method": processing.get("fallback_method"),
            # Denominators are scoped provenance, not globally shared
            # processing parameters. Different participants must have their
            # own MVC values while using the same algorithm and configuration.
            "normalization_reference_scope": expected_scope,
            "channel_profile_snapshot": processing["channel_profile_snapshot"],
        }
        if reference_processing is None:
            reference_processing = processing_signature
        elif processing_signature != reference_processing:
            raise ValueError(f"Preprocessing parameters differ at {trial_dir}; explicitly reprocess before merging")
        with np.load(processed_path, allow_pickle=False) as payload:
            segment = np.asarray(payload["normalized_envelope"], dtype=np.float32)
            fs_hz = float(payload["fs_hz"])
        if segment.ndim != 2 or segment.shape[1] != len(profile.channels):
            raise ValueError(f"Processed channel count mismatch at {processed_path}")
        if not np.all(np.isfinite(segment)) or np.any(segment < 0):
            raise ValueError(f"Processed synergy input must be finite and non-negative: {processed_path}")
        configured_fs_hz = float(processing["processing_config"]["sample_rate_hz"])
        if (
            not np.isfinite(fs_hz)
            or fs_hz <= 0
            or not np.isclose(fs_hz, configured_fs_hz, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"Processed sample rate/provenance mismatch at {processed_path}")
        crop_note = "full_trial"
        start, stop = 0, len(segment)
        crop_is_exploratory = False
        annotation_provenance: dict[str, Any] | None = None
        if crop_mode == "annotated_movement_events":
            event_rows = _event_rows(trial_dir / "events.csv")
            start_row = _required_event(event_rows, "movement_start_manual", trial_dir)
            stop_row = _required_event(event_rows, "recording_stop", trial_dir)
            start_source = start_row.get("source", "").strip()
            if start_source not in ALLOWED_ANNOTATION_SOURCES:
                raise ValueError(
                    "annotated_movement_events requires evidence-backed movement_start_manual "
                    f"(not software/unannotated); got source={start_source!r}: {trial_dir}"
                )
            start = _event_sample(start_row, "movement_start_manual", trial_dir)
            stop = _event_sample(stop_row, "recording_stop", trial_dir)
            annotation_provenance = verify_committed_annotation(
                trial_dir,
                "movement_start_manual",
            )
            crop_note = "annotated_movement_start_to_recording_stop"
        elif crop_mode == "software_cue_exploratory":
            event_rows = _event_rows(trial_dir / "events.csv")
            start_row = _required_event(event_rows, "movement_cue", trial_dir)
            stop_row = _required_event(event_rows, "recording_stop", trial_dir)
            start = _event_sample(start_row, "movement_cue", trial_dir)
            stop = _event_sample(stop_row, "recording_stop", trial_dir)
            crop_note = "software_cue_to_recording_stop_exploratory"
            crop_is_exploratory = True
        elif crop_mode == "full_trial":
            pass
        else:
            raise ValueError(
                "crop_mode must be annotated_movement_events, software_cue_exploratory, or full_trial"
            )
        if not 0 <= start < stop <= len(segment):
            raise ValueError(
                f"Invalid crop bounds start={start}, stop={stop}, samples={len(segment)}: {trial_dir}"
            )
        segment = segment[start:stop]
        if not len(segment):
            raise ValueError(f"Empty action segment after cropping: {trial_dir}")
        if time_normalize_samples:
            if time_normalize_samples < 2:
                raise ValueError("time_normalize_samples must be >= 2")
            segment = np.maximum(resample(segment, time_normalize_samples, axis=0), 0).astype(np.float32)
            crop_note += f"_resampled_{time_normalize_samples}"
        segments.append(segment)
        boundary += len(segment)
        boundaries.append(boundary)
        trials.append(
            {
                "participant_id": participant,
                "session_id": metadata["session_id"],
                "trial_id": metadata["trial_id"],
                "action_id": action,
                "category": category,
                "valid_for_analysis": metadata["valid_for_analysis"],
                "preprocessing_analysis_ready": preprocessing_analysis_ready,
                "normalization_reference_key": reference_key,
                "normalization_reference_values_mV": canonical_values,
                "source": str(processed_path),
                "crop_method": crop_note,
                "crop_is_exploratory": crop_is_exploratory,
                "movement_start_annotation": annotation_provenance,
                "source_start_sample": start,
                "source_stop_sample": stop,
                "output_samples": len(segment),
            }
        )
    if not segments:
        raise ValueError("No trials matched the requested dataset filters")
    concatenated = np.vstack(segments)
    V = concatenated.T
    metadata_out = {
        "dataset_format_version": 1,
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "channel_profile_snapshot": profile.to_dict(),
        "protocol_id": protocol_id,
        "processing": reference_processing,
        "normalization_references": normalization_references,
        "only_valid": only_valid,
        "only_valid_includes_preprocessing_qc": True,
        "scope": scope,
        "crop_mode": crop_mode,
        "exploratory": crop_mode == "software_cue_exploratory",
        "event_semantics": (
            "evidence_backed_manual_movement_start"
            if crop_mode == "annotated_movement_events"
            else "software_cue_is_not_physical_movement_start"
            if crop_mode == "software_cue_exploratory"
            else "no_event_crop"
        ),
        "time_normalize_samples": time_normalize_samples,
        "participants": sorted({item["participant_id"] for item in trials}),
        "sessions": sorted({f"{item['participant_id']}/{item['session_id']}" for item in trials}),
        "actions": sorted({item["action_id"] for item in trials}),
        "trials": trials,
        "matrix_shape": list(V.shape),
        "trial_boundaries": boundaries,
        "fs_hz": fs_hz,
    }
    metadata_out["dataset_sha256"] = dataset_sha256(V, metadata_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        output_path,
        V=V.astype(np.float32),
        channel_ids=np.asarray(profile.channel_ids, dtype=np.int16),
        muscle_slugs=np.asarray([channel.muscle_slug for channel in profile.channels]),
        sides=np.asarray([channel.side for channel in profile.channels]),
        trial_boundaries=np.asarray(boundaries, dtype=np.int64),
        fs_hz=np.asarray(fs_hz),
    )
    atomic_write_json(output_path.with_suffix(".json"), metadata_out)
    return output_path
