"""Fail-closed import of preprocessed Jidian sEMG for MuscleMimic evaluation.

The importer deliberately consumes one explicit selection manifest.  It never
walks a participant directory, guesses a profile, infers an event from a cue,
or re-filters the signal.  The only accepted signal is Jidian's already
processed ``normalized_envelope`` from the exact right-handed 16-channel v2
profile.

Selection manifest (``jidian_emg_selection_v1``)::

    {
      "schema_version": "jidian_emg_selection_v1",
      "session_path": "../jidian_measurement/data/P003/S20260801_A",
      "subject_uid": "participant-registry-uuid",
      "session_uid": "emg-session-registry-uuid",
      "action_id": "forehand_smash",
      "trial_ids": ["forehand_smash_trial_001"],
      "dataset_split": "heldout",
      "comparison_design": "unpaired_action_cohort_v1",
      "comparison_set": {
        "comparison_set_uid": "forehand-smash-heldout-v1"
      },
      "training_session_uids": ["training-session-registry-uuid"],
      "alignment": {
        "mode": "impact",
        "event_name": "racket_contact",
        "minimum_confidence": 0.8,
        "pre_event_s": 0.5,
        "post_event_s": 0.8
      }
    }

For ``paired_same_reference_v1``, the same ``comparison_set`` object must also
contain ``reference_trial_fingerprints`` with exactly one lowercase SHA-256 per
selected trial, *and* an ``external_reference_provenance`` attestation
(``attestation="externally_verified_shared_reference_trial"`` plus a non-empty
``evidence_reference``).  Independently collected Jidian sEMG has no shared
reference-trial fingerprint with the Stage-3 policy feed, so the paired design is
rejected at parse time without that external attestation; use
``unpaired_action_cohort_v1``.  ``unpaired_action_cohort_v1`` forbids both the
``reference_trial_fingerprints`` and ``external_reference_provenance`` fields.

Every selected trial must pass.  A rejection writes a JSON audit but never an
NPZ.  This all-or-nothing rule prevents a post-hoc eligible subset from silently
redefining a pre-registered comparison set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict, loads_json_strict

SELECTION_SCHEMA_VERSION = "jidian_emg_selection_v1"
IMPORT_SCHEMA_VERSION = "jidian_emg_import_v1"
AUDIT_SCHEMA_VERSION = "jidian_emg_import_audit_v1"
PROCESSING_MANIFEST_SCHEMA_VERSION = "jidian_semg_processing_manifest_v2"
EMG_SIGNAL_KIND = "preprocessed_normalized_envelope_v1"
PAIRED_COMPARISON_DESIGN = "paired_same_reference_v1"
UNPAIRED_COMPARISON_DESIGN = "unpaired_action_cohort_v1"
# Backward-compatible public name for callers which imported the original
# paired-only constant.  New code should use the two explicit names above.
COMPARISON_DESIGN = PAIRED_COMPARISON_DESIGN

EXPECTED_PROFILE_ID = "badminton_synergy_16_v2"
EXPECTED_PROFILE_VERSION = 2
EXPECTED_HANDEDNESS = "right"
ACQUIRED_CHANNEL_COUNT = 16
COMPARABLE_CHANNEL_COUNT = 15
EXCLUDED_SENSOR_IDS = (1,)
EXCLUDED_SENSOR_REASON = "no_verified_354_actuator_model_homolog"

EXPECTED_CHANNELS: tuple[tuple[int, str, str], ...] = (
    (1, "right", "upper_trapezius"),
    (2, "right", "anterior_deltoid"),
    (3, "right", "posterior_deltoid"),
    (4, "right", "pectoralis_major_clavicular"),
    (5, "right", "latissimus_dorsi"),
    (6, "right", "triceps_lateral"),
    (7, "right", "pronator_teres"),
    (8, "right", "extensor_carpi_radialis"),
    (9, "right", "external_oblique"),
    (10, "left", "external_oblique"),
    (11, "right", "vastus_lateralis"),
    (12, "left", "vastus_lateralis"),
    (13, "right", "biceps_femoris_long_head"),
    (14, "left", "biceps_femoris_long_head"),
    (15, "right", "gastrocnemius_medialis"),
    (16, "left", "gastrocnemius_medialis"),
)
EXPECTED_STREAM_CHANNEL_IDS = tuple(channel[0] for channel in EXPECTED_CHANNELS)
EXPECTED_SIDES = tuple(channel[1] for channel in EXPECTED_CHANNELS)
EXPECTED_MUSCLE_SLUGS = tuple(channel[2] for channel in EXPECTED_CHANNELS)
EXPECTED_CHANNEL_NAMES = tuple(
    f"S{sensor_id} {side}:{muscle_slug}" for sensor_id, side, muscle_slug in EXPECTED_CHANNELS
)

EVENT_COLUMNS = (
    "event_name",
    "sample_index",
    "emg_time_s",
    "monotonic_time_ns",
    "wall_clock_iso",
    "source",
    "confidence",
    "notes",
)
EVENT_ANNOTATION_AUDIT_SCHEMA_VERSION = "emg_event_annotation_audit_v1"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_PHYSICAL_EVENT_SOURCES = frozenset(
    {
        "manual_video",
        "manual_audio",
        "manual_multimodal",
        "manual_operator_record",
        "hardware_ttl",
        "instrumented_racket",
        "force_plate",
    }
)


class JidianEmgImportError(RuntimeError):
    """Raised after a fail-closed audit has been written."""

    def __init__(self, message: str, *, audit_path: Path, report: Mapping[str, Any]):
        super().__init__(f"{message}; audit: {audit_path}")
        self.audit_path = audit_path
        self.report = dict(report)


class _TrialRejectionError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class _Selection:
    source_path: Path
    source_sha256: str
    session_path: Path
    subject_uid: str
    session_uid: str
    action_id: str
    trial_ids: tuple[str, ...]
    dataset_split: str
    comparison_design: str
    comparison_set_uid: str
    reference_trial_fingerprints: Mapping[str, str]
    training_session_uids: tuple[str, ...]
    event_name: str
    minimum_confidence: float
    pre_event_s: float
    post_event_s: float


@dataclass(frozen=True)
class _SessionEvidence:
    participant_id: str
    source_session_id: str
    profile: Mapping[str, Any]
    profile_sha256: str
    session_sha256: str
    mvc_manifest_sha256: str
    mvc_source_hashes: Mapping[str, str]
    mvc_source_set_sha256: str
    mvc_values: np.ndarray
    processing_config: Mapping[str, Any]


@dataclass(frozen=True)
class _ImportedTrial:
    source_trial_id: str
    trial_uid: str
    reference_trial_fingerprint: str | None
    emg: np.ndarray
    sampling_rate_hz: float
    impact_frame: int
    source_impact_frame: int
    crop_start: int
    crop_stop: int
    processing_sha256: str
    source_file_set_sha256: str
    file_hashes: Mapping[str, str]
    warning_codes: tuple[str, ...]
    event_evidence: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_identifier(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if _SAFE_IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{name} contains unsupported path characters")
    return text


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return text


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {name} is missing: {path}")
    value = load_json_strict(path)
    return _require_mapping(value, name)


def _descendant(root: Path, relative: Path, name: str) -> Path:
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} resolves outside the selected session") from exc
    return result


def _string_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise _TrialRejectionError("processed_identity_mismatch", f"processed {name} must be scalar")
    result = str(array.reshape(-1)[0]).strip()
    if not result:
        raise _TrialRejectionError("processed_identity_mismatch", f"processed {name} is empty")
    return result


def _integer_scalar(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise _TrialRejectionError("processed_identity_mismatch", f"processed {name} must be an integer scalar")
    return int(array.reshape(-1)[0])


def _float_scalar(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise _TrialRejectionError("processed_signal_invalid", f"processed {name} must be scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise _TrialRejectionError("processed_signal_invalid", f"processed {name} must be finite")
    return result


def stable_jidian_trial_uid(
    *,
    subject_uid: str,
    session_uid: str,
    action_id: str,
    source_trial_id: str,
) -> str:
    """Return a stable acquisition identity independent of row order and paths."""

    payload = {
        "schema_version": "jidian_acquisition_trial_identity_v1",
        "subject_uid": _require_text(subject_uid, "subject_uid"),
        "session_uid": _require_text(session_uid, "session_uid"),
        "action_id": _require_identifier(action_id, "action_id"),
        "source_trial_id": _require_identifier(source_trial_id, "source_trial_id"),
    }
    return f"jidian-emg-{_canonical_sha256(payload)}"


def _parse_selection(path: Path) -> _Selection:
    payload = _json_object(path, "selection manifest")
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"selection schema_version must be {SELECTION_SCHEMA_VERSION!r}")
    session_text = _require_text(payload.get("session_path"), "session_path")
    supplied_session = Path(session_text).expanduser()
    session_path = (
        supplied_session.resolve() if supplied_session.is_absolute() else (path.parent / supplied_session).resolve()
    )
    if not session_path.is_dir():
        raise ValueError(f"selected session_path is not a directory: {session_path}")

    subject_uid = _require_text(payload.get("subject_uid"), "subject_uid")
    session_uid = _require_text(payload.get("session_uid"), "session_uid")
    action_id = _require_identifier(payload.get("action_id"), "action_id")
    raw_trial_ids = payload.get("trial_ids")
    if not isinstance(raw_trial_ids, list) or not raw_trial_ids:
        raise ValueError("trial_ids must be a non-empty JSON array")
    trial_ids = tuple(_require_identifier(value, "trial_ids[]") for value in raw_trial_ids)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial_ids must be unique")
    expected_prefix = f"{action_id}_trial_"
    for trial_id in trial_ids:
        suffix = trial_id.removeprefix(expected_prefix)
        if not trial_id.startswith(expected_prefix) or re.fullmatch(r"[0-9]{3}", suffix) is None:
            raise ValueError(f"trial_id {trial_id!r} must be exactly {action_id}_trial_NNN")

    dataset_split = _require_text(payload.get("dataset_split"), "dataset_split").lower()
    if dataset_split not in {"heldout", "validation", "test"}:
        raise ValueError("dataset_split must be heldout, validation, or test")
    comparison_design = _require_text(
        payload.get("comparison_design"),
        "comparison_design",
    )
    if comparison_design not in {PAIRED_COMPARISON_DESIGN, UNPAIRED_COMPARISON_DESIGN}:
        raise ValueError("comparison_design must be paired_same_reference_v1 or unpaired_action_cohort_v1")
    comparison = _require_mapping(payload.get("comparison_set"), "comparison_set")
    uid_value = comparison.get("comparison_set_uid")
    legacy_id = comparison.get("comparison_set_id")
    if uid_value is not None and legacy_id is not None and uid_value != legacy_id:
        raise ValueError("comparison_set_uid and legacy comparison_set_id disagree")
    comparison_set_uid = _require_text(
        uid_value if uid_value is not None else legacy_id,
        "comparison_set.comparison_set_uid",
    )
    if comparison_design == PAIRED_COMPARISON_DESIGN:
        # Jidian sEMG is collected independently of the Stage-3 policy feed, so it
        # has no genuine shared reference-trial fingerprint.  Hand-authored
        # ``reference_trial_fingerprints`` must not be silently accepted as paired
        # evidence: require an explicit external cross-fingerprint provenance
        # attestation before the paired design is allowed at parse time.  Without
        # it the boundary fails closed here rather than only downstream.
        provenance = comparison.get("external_reference_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "paired_same_reference_v1 requires comparison_set.external_reference_provenance; "
                "independently collected Jidian sEMG has no shared reference-trial fingerprint, "
                "so use unpaired_action_cohort_v1 unless externally verified paired provenance is supplied"
            )
        attested = provenance.get("attestation")
        if not isinstance(attested, str) or attested != "externally_verified_shared_reference_trial":
            raise ValueError(
                "external_reference_provenance.attestation must be "
                "'externally_verified_shared_reference_trial' to justify paired_same_reference_v1"
            )
        evidence_reference = provenance.get("evidence_reference")
        if not isinstance(evidence_reference, str) or not evidence_reference.strip():
            raise ValueError(
                "external_reference_provenance.evidence_reference is required and must be a non-empty string"
            )
        reference_raw = _require_mapping(
            comparison.get("reference_trial_fingerprints"),
            "comparison_set.reference_trial_fingerprints",
        )
        if set(reference_raw) != set(trial_ids):
            missing = sorted(set(trial_ids) - set(reference_raw))
            extra = sorted(set(reference_raw) - set(trial_ids))
            raise ValueError(
                "reference_trial_fingerprints must have exactly the selected trial_ids "
                f"(missing={missing}, extra={extra})"
            )
        references = {
            trial_id: _require_sha256(
                reference_raw[trial_id],
                f"reference_trial_fingerprints[{trial_id!r}]",
            )
            for trial_id in trial_ids
        }
    else:
        if "reference_trial_fingerprints" in comparison:
            raise ValueError("unpaired_action_cohort_v1 forbids reference_trial_fingerprints")
        if "external_reference_provenance" in comparison:
            raise ValueError("unpaired_action_cohort_v1 forbids external_reference_provenance")
        references = {}

    raw_training = payload.get("training_session_uids")
    if not isinstance(raw_training, list) or not raw_training:
        raise ValueError("training_session_uids must be a non-empty JSON array")
    training = tuple(_require_text(value, "training_session_uids[]") for value in raw_training)
    if len(set(training)) != len(training):
        raise ValueError("training_session_uids must be unique")
    if session_uid in set(training):
        raise ValueError("held-out session_uid appears in training_session_uids")

    alignment = _require_mapping(payload.get("alignment"), "alignment")
    if alignment.get("mode") != "impact":
        raise ValueError("alignment.mode must be 'impact'; cue/full-trial fallback is forbidden")
    event_name = _require_identifier(alignment.get("event_name"), "alignment.event_name")
    minimum_confidence = _require_finite_float(
        alignment.get("minimum_confidence"),
        "alignment.minimum_confidence",
    )
    if not 0.0 < minimum_confidence <= 1.0:
        raise ValueError("alignment.minimum_confidence must lie in (0,1]")
    pre_event_s = _require_finite_float(alignment.get("pre_event_s"), "alignment.pre_event_s")
    post_event_s = _require_finite_float(alignment.get("post_event_s"), "alignment.post_event_s")
    if pre_event_s <= 0.0 or post_event_s <= 0.0:
        raise ValueError("impact windows must have positive pre_event_s and post_event_s")

    return _Selection(
        source_path=path,
        source_sha256=_file_sha256(path),
        session_path=session_path,
        subject_uid=subject_uid,
        session_uid=session_uid,
        action_id=action_id,
        trial_ids=trial_ids,
        dataset_split=dataset_split,
        comparison_design=comparison_design,
        comparison_set_uid=comparison_set_uid,
        reference_trial_fingerprints=references,
        training_session_uids=training,
        event_name=event_name,
        minimum_confidence=minimum_confidence,
        pre_event_s=pre_event_s,
        post_event_s=post_event_s,
    )


def _profile_channels(profile: Mapping[str, Any]) -> tuple[tuple[int, str, str], ...]:
    raw_channels = profile.get("channels")
    if not isinstance(raw_channels, list):
        raise ValueError("channel_profile.channels must be an array")
    result: list[tuple[int, str, str]] = []
    for index, raw in enumerate(raw_channels):
        channel = _require_mapping(raw, f"channel_profile.channels[{index}]")
        sensor_id = channel.get("sensor_id")
        if type(sensor_id) is not int:
            raise ValueError(f"channel_profile.channels[{index}].sensor_id must be integer")
        result.append(
            (
                sensor_id,
                _require_text(channel.get("side"), f"channel {sensor_id} side"),
                _require_text(channel.get("muscle_slug"), f"channel {sensor_id} muscle_slug"),
            )
        )
    return tuple(result)


def _resolve_mvc_source(session_path: Path, supplied: Any, *, channel_directory: str) -> Path:
    text = _require_text(supplied, "MVC repetition path")
    path = Path(text).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        candidates.append((session_path / path).resolve())
    parts = path.parts
    if "mvc" in parts:
        mvc_index = len(parts) - 1 - tuple(reversed(parts)).index("mvc")
        candidates.append((session_path / Path(*parts[mvc_index:])).resolve())
    expected_root = (session_path / "mvc" / channel_directory).resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            continue
        if candidate.name != "mvc_timeseries.npz":
            continue
        return candidate
    raise ValueError(
        f"MVC repetition does not resolve to an existing explicit local source under mvc/{channel_directory}: {text}"
    )


def _load_session_evidence(selection: _Selection) -> _SessionEvidence:
    session_path = selection.session_path
    profile_path = _descendant(session_path, Path("channel_profile.json"), "channel profile")
    session_json_path = _descendant(session_path, Path("session.json"), "session metadata")
    mvc_path = _descendant(
        session_path,
        Path("preprocessing_mvc_reference.json"),
        "MVC reference manifest",
    )
    profile = _json_object(profile_path, "channel_profile.json")
    session = _json_object(session_json_path, "session.json")
    mvc = _json_object(mvc_path, "preprocessing_mvc_reference.json")

    if profile.get("profile_id") != EXPECTED_PROFILE_ID:
        raise ValueError(f"channel profile must be exactly {EXPECTED_PROFILE_ID!r}")
    if profile.get("version") != EXPECTED_PROFILE_VERSION:
        raise ValueError(f"channel profile version must be exactly {EXPECTED_PROFILE_VERSION}")
    if profile.get("intended_handedness") != EXPECTED_HANDEDNESS:
        raise ValueError("channel profile intended_handedness must be right")
    if _profile_channels(profile) != EXPECTED_CHANNELS:
        raise ValueError("channel profile must have the exact ordered v2 sensor/side/muscle identities")
    # The mapping-v2 contract uses this exact canonical JSON definition.
    profile_sha256 = _canonical_sha256(profile)

    participant_id = _require_text(session.get("participant_id"), "session participant_id")
    source_session_id = _require_text(session.get("session_id"), "session session_id")
    if session.get("channel_profile_id") != EXPECTED_PROFILE_ID:
        raise ValueError("session channel_profile_id differs from the exact v2 profile")
    if session.get("handedness") != EXPECTED_HANDEDNESS:
        raise ValueError("session handedness must be right")

    if mvc.get("scope") != "participant":
        raise ValueError("MVC reference scope must be participant")
    if mvc.get("participant_id") != participant_id:
        raise ValueError("MVC participant_id differs from the selected session")
    if mvc.get("missing_sensor_ids") != []:
        raise ValueError("MVC reference has missing_sensor_ids")
    processing_config = _require_mapping(mvc.get("processing_config"), "MVC processing_config")
    if processing_config.get("normalization") != "mvc":
        raise ValueError("MVC processing_config.normalization must be mvc")
    mvc_channels = mvc.get("channels")
    if not isinstance(mvc_channels, list) or len(mvc_channels) != ACQUIRED_CHANNEL_COUNT:
        raise ValueError("MVC reference must contain exactly 16 channels")

    values: list[float] = []
    source_hashes: dict[str, str] = {}
    for index, (raw, expected) in enumerate(zip(mvc_channels, EXPECTED_CHANNELS, strict=True)):
        channel = _require_mapping(raw, f"MVC channels[{index}]")
        identity = (
            channel.get("sensor_id"),
            channel.get("side"),
            channel.get("muscle_slug"),
        )
        if identity != expected:
            raise ValueError("MVC channel order/identity differs from the exact v2 profile")
        peak = _require_finite_float(channel.get("selected_peak_mV"), f"MVC sensor {expected[0]} peak")
        if peak <= 0.0:
            raise ValueError(f"MVC sensor {expected[0]} selected_peak_mV must be positive")
        repetitions = channel.get("valid_repetitions")
        if not isinstance(repetitions, list) or not repetitions:
            raise ValueError(f"MVC sensor {expected[0]} has no valid raw repetition")
        repetition_peaks: list[float] = []
        channel_directory = f"{expected[1]}_{expected[2]}"
        for repetition in repetitions:
            record = _require_mapping(repetition, f"MVC sensor {expected[0]} repetition")
            if record.get("session_id") != source_session_id:
                raise ValueError(f"MVC sensor {expected[0]} repetition has a foreign session_id")
            repetition_peak = _require_finite_float(
                record.get("envelope_peak_mV"),
                f"MVC sensor {expected[0]} repetition peak",
            )
            if repetition_peak <= 0.0:
                raise ValueError(f"MVC sensor {expected[0]} repetition peak must be positive")
            repetition_peaks.append(repetition_peak)
            source = _resolve_mvc_source(
                session_path,
                record.get("path"),
                channel_directory=channel_directory,
            )
            relative = source.relative_to(session_path).as_posix()
            source_hashes[relative] = _file_sha256(source)
        if not np.isclose(peak, max(repetition_peaks), rtol=1e-9, atol=1e-12):
            raise ValueError(f"MVC sensor {expected[0]} selected peak is not the valid repetition maximum")
        values.append(peak)

    return _SessionEvidence(
        participant_id=participant_id,
        source_session_id=source_session_id,
        profile=dict(profile),
        profile_sha256=profile_sha256,
        session_sha256=_file_sha256(session_json_path),
        mvc_manifest_sha256=_file_sha256(mvc_path),
        mvc_source_hashes=dict(sorted(source_hashes.items())),
        mvc_source_set_sha256=_canonical_sha256(dict(sorted(source_hashes.items()))),
        mvc_values=np.asarray(values, dtype=np.float64),
        processing_config=dict(processing_config),
    )


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _reject(condition: bool, code: str, detail: str) -> None:
    if condition:
        raise _TrialRejectionError(code, detail)


def _validate_trial_metadata(
    metadata: Mapping[str, Any],
    *,
    selection: _Selection,
    session: _SessionEvidence,
    trial_id: str,
    trial_index: int,
) -> None:
    expected = {
        "participant_id": session.participant_id,
        "session_id": session.source_session_id,
        "trial_id": trial_id,
        "trial_index": trial_index,
        "action_id": selection.action_id,
        "channel_profile_id": EXPECTED_PROFILE_ID,
        "channel_profile_version": EXPECTED_PROFILE_VERSION,
        "handedness": EXPECTED_HANDEDNESS,
    }
    for name, value in expected.items():
        _reject(
            metadata.get(name) != value,
            "metadata_identity_mismatch",
            f"metadata {name} differs from the explicit selection/profile",
        )
    _reject(
        metadata.get("channel_profile_snapshot") != session.profile,
        "profile_snapshot_mismatch",
        "metadata channel_profile_snapshot differs from channel_profile.json",
    )
    _reject(
        metadata.get("valid_for_analysis") is not True,
        "manual_validity_rejected",
        "metadata valid_for_analysis is not true",
    )
    _reject(
        type(metadata.get("interrupted")) is not bool or metadata["interrupted"] is not False,
        "acquisition_interrupted",
        "metadata interrupted must be the explicit boolean false",
    )
    receive_error = metadata.get("receive_error")
    _reject(
        "receive_error" not in metadata
        or not (receive_error is None or isinstance(receive_error, str))
        or (isinstance(receive_error, str) and receive_error.strip() != ""),
        "acquisition_receive_error",
        "metadata receive_error must be explicitly null or an empty string",
    )
    expected_samples = metadata.get("expected_samples")
    received_samples = metadata.get("received_samples")
    _reject(
        type(expected_samples) is not int or type(received_samples) is not int,
        "acquisition_completeness_invalid",
        "metadata expected_samples/received_samples must be integers",
    )
    _reject(
        expected_samples != received_samples
        or type(metadata.get("dropped_samples")) is not int
        or metadata["dropped_samples"] != 0,
        "acquisition_incomplete",
        "trial has missing/dropped acquisition samples",
    )


def _validate_processing(
    processing: Mapping[str, Any],
    *,
    session: _SessionEvidence,
) -> np.ndarray:
    required = {
        "processing_format_version",
        "status",
        "channel_profile_id",
        "channel_profile_snapshot",
        "processing_config",
        "normalization_method",
        "fallback_method",
        "normalization_reference_scope",
        "mvc_reference_manifest",
        "normalization_reference_values_mV",
        "raw_source",
        "quality_summary",
    }
    missing = sorted(required - set(processing))
    _reject(
        bool(missing),
        "processing_fields_missing",
        f"processing manifest is missing required fields: {missing}",
    )
    _reject(
        processing.get("processing_format_version") != 2 or processing.get("status") != "completed",
        "processing_incomplete",
        "processing must be completed format version 2",
    )
    _reject(
        processing.get("channel_profile_id") != EXPECTED_PROFILE_ID
        or processing.get("channel_profile_snapshot") != session.profile,
        "profile_snapshot_mismatch",
        "processing profile/snapshot differs from channel_profile.json",
    )
    _reject(
        processing.get("processing_config") != session.processing_config,
        "processing_config_mismatch",
        "trial processing_config differs from the MVC processing_config",
    )
    _reject(
        processing.get("normalization_method") != "mvc",
        "normalization_not_mvc",
        "trial normalization_method must be mvc",
    )
    _reject(
        processing.get("fallback_method") is not None,
        "normalization_fallback_used",
        "trial processing used a normalization fallback",
    )
    _reject(
        processing.get("normalization_reference_scope") != "participant_raw_mvc_across_sessions",
        "mvc_reference_scope_mismatch",
        "trial normalization_reference_scope is not participant raw MVC",
    )
    mvc_manifest_reference = processing.get("mvc_reference_manifest")
    _reject(
        not isinstance(mvc_manifest_reference, str)
        or Path(mvc_manifest_reference).name != "preprocessing_mvc_reference.json",
        "mvc_reference_manifest_mismatch",
        "trial processing does not explicitly name preprocessing_mvc_reference.json",
    )
    _reject(
        processing.get("raw_source") != "raw_emg.npz",
        "raw_source_mismatch",
        "processing raw_source must be the same trial's raw_emg.npz",
    )
    quality = processing.get("quality_summary")
    _reject(not isinstance(quality, Mapping), "processing_qc_rejected", "processing quality_summary is missing")
    assert isinstance(quality, Mapping)
    _reject(
        quality.get("analysis_ready") is not True
        or quality.get("qc_pass") is not True
        or quality.get("critical_channel_count") != 0,
        "processing_qc_rejected",
        "processing quality_summary is not analysis-ready with zero critical channels",
    )
    raw_values = processing.get("normalization_reference_values_mV")
    try:
        values = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _TrialRejectionError("mvc_reference_mismatch", "normalization reference is not numeric") from exc
    _reject(
        values.shape != (ACQUIRED_CHANNEL_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.allclose(values, session.mvc_values, rtol=1e-12, atol=1e-12),
        "mvc_reference_mismatch",
        "trial normalization values differ from the explicit MVC manifest",
    )
    return values


def _validate_preprocessing_qc(qc: Mapping[str, Any]) -> tuple[str, ...]:
    _reject(
        qc.get("analysis_ready") is not True
        or qc.get("qc_pass") is not True
        or qc.get("critical_channel_count") != 0
        or qc.get("channel_count") != ACQUIRED_CHANNEL_COUNT,
        "preprocessing_qc_rejected",
        "preprocessing QC is not analysis-ready for all 16 channels",
    )
    raw_channels = qc.get("channels")
    _reject(
        not isinstance(raw_channels, list) or len(raw_channels) != ACQUIRED_CHANNEL_COUNT,
        "preprocessing_qc_rejected",
        "preprocessing QC must contain exactly 16 channel records",
    )
    assert isinstance(raw_channels, list)
    warnings: set[str] = set()
    for index, (raw, expected) in enumerate(zip(raw_channels, EXPECTED_CHANNELS, strict=True)):
        _reject(not isinstance(raw, Mapping), "preprocessing_qc_rejected", f"QC channel {index} is invalid")
        assert isinstance(raw, Mapping)
        identity = (raw.get("sensor_id"), raw.get("side"), raw.get("muscle_slug"))
        _reject(
            raw.get("channel_index") != index or identity != expected,
            "preprocessing_qc_rejected",
            "preprocessing QC channel order/identity differs from the exact v2 profile",
        )
        critical = raw.get("critical_failures")
        _reject(
            raw.get("analysis_ready") is not True or critical != [],
            "preprocessing_qc_rejected",
            f"preprocessing QC rejected sensor {expected[0]}",
        )
        channel_warnings = raw.get("warnings", [])
        _reject(
            not isinstance(channel_warnings, list)
            or any(not isinstance(value, str) or not value for value in channel_warnings),
            "preprocessing_qc_rejected",
            f"preprocessing QC warnings are malformed for sensor {expected[0]}",
        )
        warnings.update(str(value) for value in channel_warnings)
    return tuple(sorted(warnings))


def _load_raw_signal(
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required = {"emg_mV", "time_s", "sample_index", "stream_channel_ids", "fs_hz"}
    try:
        with np.load(path, allow_pickle=False) as source:
            missing = sorted(required - set(source.files))
            _reject(bool(missing), "raw_fields_missing", f"raw_emg.npz missing fields: {missing}")
            emg = np.asarray(source["emg_mV"], dtype=np.float64).copy()
            time_s = np.asarray(source["time_s"], dtype=np.float64).copy()
            sample_index = np.asarray(source["sample_index"]).copy()
            stream_ids = np.asarray(source["stream_channel_ids"]).copy()
            fs = _float_scalar(source["fs_hz"], "raw fs_hz")
    except _TrialRejectionError:
        raise
    except (OSError, ValueError) as exc:
        raise _TrialRejectionError("raw_npz_unreadable", f"cannot read raw_emg.npz: {exc}") from exc
    _reject(
        emg.ndim != 2 or emg.shape[1] != ACQUIRED_CHANNEL_COUNT or emg.shape[0] < 2,
        "raw_signal_invalid",
        "raw emg_mV must have shape [time,16]",
    )
    _reject(
        not np.all(np.isfinite(emg)),
        "raw_signal_nonfinite",
        "raw emg_mV contains NaN/Inf",
    )
    length = emg.shape[0]
    _reject(
        time_s.shape != (length,) or not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0),
        "raw_time_invalid",
        "raw time_s must be finite and strictly increasing",
    )
    _reject(
        sample_index.shape != (length,)
        or not np.issubdtype(sample_index.dtype, np.integer)
        or not np.array_equal(sample_index, np.arange(length)),
        "raw_time_invalid",
        "raw sample_index must be exact ordered integers 0..T-1",
    )
    _reject(
        stream_ids.shape != (ACQUIRED_CHANNEL_COUNT,)
        or not np.issubdtype(stream_ids.dtype, np.integer)
        or stream_ids.astype(int).tolist() != list(EXPECTED_STREAM_CHANNEL_IDS),
        "raw_profile_mismatch",
        "raw stream_channel_ids must be exact ordered integers 1..16",
    )
    _reject(fs <= 0.0, "raw_signal_invalid", "raw fs_hz must be positive")
    _reject(
        not np.allclose(time_s, np.arange(length, dtype=np.float64) / fs, rtol=0.0, atol=1e-9),
        "raw_time_invalid",
        "raw time_s is inconsistent with sample_index/fs_hz",
    )
    _reject(
        metadata.get("received_samples") != length,
        "raw_sample_mismatch",
        "raw signal length differs from metadata received_samples",
    )
    metadata_fs = _require_finite_float(metadata.get("sample_rate_hz"), "metadata sample_rate_hz")
    _reject(
        not np.isclose(metadata_fs, fs, rtol=0.0, atol=1e-9),
        "sample_rate_mismatch",
        "raw fs_hz differs from metadata sample_rate_hz",
    )
    return emg, time_s, sample_index.astype(np.int64), fs


def _validate_raw_qc(
    qc: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    raw_emg: np.ndarray,
) -> tuple[str, ...]:
    """Reject acquisition-fatal QC while retaining ordinary review warnings."""

    expected_samples = metadata.get("expected_samples")
    received_samples = metadata.get("received_samples")
    _reject(
        type(qc.get("qc_pass")) is not bool,
        "raw_qc_invalid",
        "raw QC qc_pass must be an explicit boolean",
    )
    _reject(
        qc.get("expected_samples") != expected_samples
        or qc.get("received_samples") != received_samples
        or qc.get("sample_count_ok") is not True
        or qc.get("short_stream_or_dropout_samples") != 0,
        "raw_qc_sample_mismatch",
        "raw QC reports a sample-count mismatch or dropout",
    )
    raw_channels = qc.get("channels")
    _reject(
        not isinstance(raw_channels, list) or len(raw_channels) != ACQUIRED_CHANNEL_COUNT,
        "raw_qc_invalid",
        "raw QC must contain exactly 16 ordered channel records",
    )
    assert isinstance(raw_channels, list)
    warning_codes: set[str] = set()
    fatal_labels = {"all_zero", "flatline", "nan_or_inf"}
    thresholds = qc.get("thresholds", {})
    _reject(not isinstance(thresholds, Mapping), "raw_qc_invalid", "raw QC thresholds must be an object")
    assert isinstance(thresholds, Mapping)
    _reject(
        "flatline_std_mV" not in thresholds,
        "raw_qc_invalid",
        "raw QC thresholds lack flatline_std_mV",
    )
    flatline_threshold = _require_finite_float(thresholds.get("flatline_std_mV"), "raw QC flatline_std_mV")
    _reject(flatline_threshold <= 0.0, "raw_qc_invalid", "raw QC flatline threshold is not positive")
    for index, raw in enumerate(raw_channels):
        _reject(
            not isinstance(raw, Mapping) or raw.get("column_index") != index,
            "raw_qc_invalid",
            "raw QC channel records are missing or out of order",
        )
        assert isinstance(raw, Mapping)
        _reject(
            type(raw.get("all_zero")) is not bool
            or type(raw.get("flatline")) is not bool
            or type(raw.get("finite")) is not bool,
            "raw_qc_invalid",
            f"raw QC fatal flags are not explicit booleans for channel {index + 1}",
        )
        channel_warnings = raw.get("warnings", [])
        _reject(
            not isinstance(channel_warnings, list)
            or any(not isinstance(value, str) or not value for value in channel_warnings),
            "raw_qc_invalid",
            f"raw QC warnings are malformed for channel {index + 1}",
        )
        present_fatal = fatal_labels.intersection(channel_warnings)
        if raw.get("all_zero") is True:
            present_fatal.add("all_zero")
        if raw.get("flatline") is True:
            present_fatal.add("flatline")
        if raw.get("finite") is not True:
            present_fatal.add("nan_or_inf")
        signal = raw_emg[:, index]
        if np.all(signal == 0.0):
            present_fatal.add("all_zero")
        if float(np.std(signal)) <= flatline_threshold:
            present_fatal.add("flatline")
        _reject(
            bool(present_fatal),
            "raw_qc_fatal_channel",
            f"raw QC channel {index + 1} has fatal labels: {sorted(present_fatal)}",
        )
        warning_codes.update(f"raw_channel_{index + 1}:{value}" for value in channel_warnings)
    top_warnings = qc.get("warnings", [])
    _reject(
        not isinstance(top_warnings, list) or any(not isinstance(value, str) or not value for value in top_warnings),
        "raw_qc_invalid",
        "raw QC top-level warnings are malformed",
    )
    _reject(
        "sample_count_mismatch_or_short_stream" in top_warnings
        or any(value.rsplit(":", maxsplit=1)[-1] in fatal_labels for value in top_warnings),
        "raw_qc_fatal_warning",
        "raw QC contains a fatal channel/sample warning",
    )
    warning_codes.update(f"raw:{value}" for value in top_warnings)
    if qc.get("qc_pass") is not True:
        warning_codes.add("raw:qc_pass_false_manual_review")
    return tuple(sorted(warning_codes))


def _load_processed_signal(
    path: Path,
    *,
    selection: _Selection,
    session: _SessionEvidence,
    metadata: Mapping[str, Any],
    trial_id: str,
    trial_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    required = {
        "normalized_envelope",
        "time_s",
        "sample_index",
        "fs_hz",
        "stream_channel_ids",
        "channel_names",
        "muscle_slugs",
        "sides",
        "participant_id",
        "session_id",
        "trial_id",
        "trial_index",
        "action_id",
    }
    try:
        with np.load(path, allow_pickle=False) as source:
            missing = sorted(required - set(source.files))
            _reject(bool(missing), "processed_fields_missing", f"processed_emg.npz missing fields: {missing}")
            envelope = np.asarray(source["normalized_envelope"], dtype=np.float64).copy()
            time_s = np.asarray(source["time_s"], dtype=np.float64).copy()
            sample_index = np.asarray(source["sample_index"]).copy()
            fs = _float_scalar(source["fs_hz"], "fs_hz")
            stream_ids = np.asarray(source["stream_channel_ids"]).copy()
            channel_names = np.asarray(source["channel_names"]).astype(str).tolist()
            muscle_slugs = np.asarray(source["muscle_slugs"]).astype(str).tolist()
            sides = np.asarray(source["sides"]).astype(str).tolist()
            scalar_identity = {
                "participant_id": _string_scalar(source["participant_id"], "participant_id"),
                "session_id": _string_scalar(source["session_id"], "session_id"),
                "trial_id": _string_scalar(source["trial_id"], "trial_id"),
                "trial_index": _integer_scalar(source["trial_index"], "trial_index"),
                "action_id": _string_scalar(source["action_id"], "action_id"),
            }
    except _TrialRejectionError:
        raise
    except (OSError, ValueError) as exc:
        raise _TrialRejectionError("processed_npz_unreadable", f"cannot read processed_emg.npz: {exc}") from exc

    _reject(
        envelope.ndim != 2 or envelope.shape[1] != ACQUIRED_CHANNEL_COUNT or envelope.shape[0] < 2,
        "processed_signal_invalid",
        "normalized_envelope must have shape [time,16]",
    )
    _reject(
        not np.all(np.isfinite(envelope)) or np.any(envelope < 0.0),
        "processed_signal_invalid",
        "normalized_envelope contains NaN/Inf or negative values",
    )
    length = envelope.shape[0]
    _reject(
        time_s.shape != (length,) or not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0),
        "processed_time_invalid",
        "processed time_s must be finite and strictly increasing",
    )
    _reject(
        sample_index.shape != (length,)
        or not np.issubdtype(sample_index.dtype, np.integer)
        or not np.array_equal(sample_index, np.arange(length)),
        "processed_time_invalid",
        "processed sample_index must be exact ordered integers 0..T-1",
    )
    _reject(fs <= 0.0, "processed_signal_invalid", "processed fs_hz must be positive")
    _reject(
        not np.allclose(time_s, np.arange(length, dtype=np.float64) / fs, rtol=0.0, atol=1e-9),
        "processed_time_invalid",
        "processed time_s is inconsistent with sample_index/fs_hz",
    )
    _reject(
        stream_ids.shape != (ACQUIRED_CHANNEL_COUNT,)
        or not np.issubdtype(stream_ids.dtype, np.integer)
        or stream_ids.astype(int).tolist() != list(EXPECTED_STREAM_CHANNEL_IDS),
        "processed_profile_mismatch",
        "processed stream_channel_ids must be exact ordered integers 1..16",
    )
    _reject(
        channel_names != list(EXPECTED_CHANNEL_NAMES)
        or muscle_slugs != list(EXPECTED_MUSCLE_SLUGS)
        or sides != list(EXPECTED_SIDES),
        "processed_profile_mismatch",
        "processed channel names/sides/muscle slugs differ from the exact v2 profile",
    )
    expected_identity = {
        "participant_id": session.participant_id,
        "session_id": session.source_session_id,
        "trial_id": trial_id,
        "trial_index": trial_index,
        "action_id": selection.action_id,
    }
    _reject(
        scalar_identity != expected_identity,
        "processed_identity_mismatch",
        "processed scalar identity differs from metadata/selection",
    )
    _reject(
        metadata.get("received_samples") != length,
        "acquisition_completeness_invalid",
        "processed signal length differs from metadata received_samples",
    )
    metadata_fs = _require_finite_float(metadata.get("sample_rate_hz"), "metadata sample_rate_hz")
    _reject(
        not np.isclose(metadata_fs, fs, rtol=0.0, atol=1e-9),
        "sample_rate_mismatch",
        "processed fs_hz differs from metadata sample_rate_hz",
    )
    config_fs = _require_finite_float(
        session.processing_config.get("sample_rate_hz"),
        "processing_config.sample_rate_hz",
    )
    _reject(
        not np.isclose(config_fs, fs, rtol=0.0, atol=1e-9),
        "sample_rate_mismatch",
        "processed fs_hz differs from MVC processing_config",
    )
    return envelope, time_s, fs


def _events_semantically_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if set(first) != set(EVENT_COLUMNS) or set(second) != set(EVENT_COLUMNS):
        return False
    text_fields = {"event_name", "wall_clock_iso", "source", "notes"}
    if any(str(first[name]).strip() != str(second[name]).strip() for name in text_fields):
        return False
    try:
        if int(str(first["sample_index"]).strip()) != int(str(second["sample_index"]).strip()):
            return False
        first_monotonic = str(first["monotonic_time_ns"]).strip()
        second_monotonic = str(second["monotonic_time_ns"]).strip()
        if not first_monotonic or not second_monotonic:
            if first_monotonic != second_monotonic:
                return False
        elif int(first_monotonic) != int(second_monotonic):
            return False
        for name in ("emg_time_s", "confidence"):
            left = float(str(first[name]).strip())
            right = float(str(second[name]).strip())
            if not np.isfinite(left) or not np.isfinite(right) or left != right:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _validate_committed_annotation_record(
    records: Sequence[Any],
    committed_index: int,
) -> Mapping[str, Any]:
    committed = records[committed_index]
    _reject(not isinstance(committed, Mapping), "event_annotation_audit_invalid", "committed audit row is invalid")
    assert isinstance(committed, Mapping)
    _reject(
        committed.get("audit_schema_version") != EVENT_ANNOTATION_AUDIT_SCHEMA_VERSION,
        "event_annotation_audit_invalid",
        "committed annotation uses an unsupported audit schema",
    )
    annotation_id = _require_text(committed.get("annotation_id"), "annotation_id")
    annotated_at = _require_text(committed.get("annotated_at"), "annotated_at")
    committed_event_name = _require_text(committed.get("event_name"), "event_name")
    _require_text(committed.get("annotator"), "annotator")
    _require_text(committed.get("evidence_reference"), "evidence_reference")
    evidence_sha256 = _require_sha256(committed.get("evidence_sha256"), "evidence_sha256")
    before_sha256 = _require_sha256(committed.get("before_sha256"), "before_sha256")
    after_sha256 = _require_sha256(committed.get("after_sha256"), "after_sha256")
    supplied_manifest_sha256 = _require_sha256(
        committed.get("annotation_manifest_sha256"),
        "annotation_manifest_sha256",
    )
    _reject(
        committed.get("events_path") != "events.csv" or type(committed.get("overwrite")) is not bool,
        "event_annotation_audit_mismatch",
        "committed annotation has invalid events_path/overwrite evidence",
    )
    after_event = committed.get("after_event")
    _reject(
        not isinstance(after_event, Mapping) or str(after_event.get("event_name", "")).strip() != committed_event_name,
        "event_annotation_event_mismatch",
        "committed annotation after_event does not bind its event_name",
    )
    unsigned = dict(committed)
    unsigned.pop("transaction_state", None)
    unsigned.pop("annotation_manifest_sha256", None)
    _reject(
        _canonical_sha256(unsigned) != supplied_manifest_sha256,
        "event_annotation_manifest_mismatch",
        "committed annotation_manifest_sha256 is invalid",
    )
    prepared_matches = []
    for record in records[:committed_index]:
        if not isinstance(record, Mapping):
            continue
        if (
            record.get("transaction_state") == "prepared"
            and record.get("annotation_id") == annotation_id
            and record.get("annotation_manifest_sha256") == supplied_manifest_sha256
        ):
            prepared_unsigned = dict(record)
            prepared_unsigned.pop("transaction_state", None)
            prepared_unsigned.pop("annotation_manifest_sha256", None)
            if prepared_unsigned == unsigned:
                prepared_matches.append(record)
    _reject(
        len(prepared_matches) != 1,
        "event_annotation_transaction_incomplete",
        "latest committed annotation lacks one exact preceding prepared record",
    )
    return {
        "audit_schema_version": EVENT_ANNOTATION_AUDIT_SCHEMA_VERSION,
        "annotation_id": annotation_id,
        "annotated_at": annotated_at,
        "event_name": committed_event_name,
        "annotation_manifest_sha256": supplied_manifest_sha256,
        "evidence_reference": str(committed["evidence_reference"]),
        "evidence_sha256": evidence_sha256,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "after_event": dict(after_event),
        "transaction_state": "committed",
    }


def _validate_event_annotation_audit(
    path: Path,
    *,
    events_path: Path,
    event_name: str,
    event_row: Mapping[str, str],
) -> Mapping[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [loads_json_strict(line) for line in lines]
    except (OSError, ValueError) as exc:
        raise _TrialRejectionError(
            "event_annotation_audit_invalid",
            f"cannot parse events.annotation.audit.jsonl: {exc}",
        ) from exc
    _reject(
        not records or any(not isinstance(record, Mapping) for record in records),
        "event_annotation_audit_invalid",
        "events.annotation.audit.jsonl has no valid records",
    )
    committed_indices = [
        index
        for index, record in enumerate(records)
        if isinstance(record, Mapping) and record.get("transaction_state") == "committed"
    ]
    _reject(
        not committed_indices,
        "event_annotation_not_committed",
        "events annotation audit has no committed transaction",
    )
    global_evidence = _validate_committed_annotation_record(records, committed_indices[-1])
    _reject(
        global_evidence["after_sha256"] != _file_sha256(events_path),
        "event_annotation_after_hash_mismatch",
        "events.csv bytes differ from the latest global committed after_sha256",
    )
    target_indices = [
        index
        for index in committed_indices
        if isinstance(records[index], Mapping) and records[index].get("event_name") == event_name
    ]
    _reject(
        not target_indices,
        "event_annotation_target_not_committed",
        f"events annotation audit has no committed transaction for {event_name!r}",
    )
    target_evidence = _validate_committed_annotation_record(records, target_indices[-1])
    _reject(
        not _events_semantically_equal(target_evidence["after_event"], event_row),
        "event_annotation_event_mismatch",
        "events.csv impact row differs from the latest target committed after_event",
    )
    target_result = dict(target_evidence)
    target_result.pop("after_event", None)
    target_result["latest_global_annotation_id"] = global_evidence["annotation_id"]
    target_result["latest_global_event_name"] = global_evidence["event_name"]
    target_result["latest_global_after_sha256"] = global_evidence["after_sha256"]
    return target_result


def _read_impact_event(
    path: Path,
    *,
    annotation_audit_path: Path,
    event_name: str,
    minimum_confidence: float,
    time_s: np.ndarray,
    sampling_rate_hz: float,
    pre_event_s: float,
    post_event_s: float,
) -> tuple[int, int, int, Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if tuple(header) != EVENT_COLUMNS or len(set(header)) != len(header):
                raise _TrialRejectionError(
                    "event_schema_invalid",
                    "events.csv must have the exact Jidian event columns",
                )
            rows = [dict(zip(header, row, strict=True)) for row in reader]
    except _TrialRejectionError:
        raise
    except (OSError, StopIteration, ValueError) as exc:
        raise _TrialRejectionError("event_schema_invalid", f"cannot parse events.csv: {exc}") from exc

    matches = [row for row in rows if row["event_name"].strip() == event_name]
    _reject(
        len(matches) != 1,
        "impact_event_not_unique",
        f"expected exactly one {event_name!r} event, found {len(matches)}",
    )
    event = matches[0]
    sample_text = event["sample_index"].strip()
    event_time_text = event["emg_time_s"].strip()
    source = event["source"].strip().lower()
    confidence_text = event["confidence"].strip()
    _reject(
        not sample_text or not event_time_text or not source or not confidence_text,
        "impact_event_incomplete",
        "impact event lacks sample_index, emg_time_s, source, or confidence",
    )
    _reject(
        re.fullmatch(r"[0-9]+", sample_text) is None,
        "impact_event_invalid",
        "impact event sample_index must be a non-negative integer",
    )
    sample = int(sample_text)
    try:
        event_time = float(event_time_text)
        confidence = float(confidence_text)
    except ValueError as exc:
        raise _TrialRejectionError("impact_event_invalid", "impact event time/confidence is not numeric") from exc
    _reject(
        not np.isfinite(event_time) or not np.isfinite(confidence),
        "impact_event_invalid",
        "impact event time/confidence must be finite",
    )
    _reject(
        source not in ALLOWED_PHYSICAL_EVENT_SOURCES,
        "impact_event_unannotated",
        "impact event source is not an allowed physical annotation source",
    )
    _reject(
        confidence < minimum_confidence or confidence > 1.0,
        "impact_event_low_confidence",
        f"impact event confidence {confidence} is below {minimum_confidence} or above 1",
    )
    _reject(
        sample < 0 or sample >= len(time_s),
        "impact_event_out_of_range",
        "impact event sample_index lies outside the processed signal",
    )
    tolerance = max(1e-9, 0.51 / sampling_rate_hz)
    _reject(
        abs(float(time_s[sample]) - event_time) > tolerance or abs(sample / sampling_rate_hz - event_time) > tolerance,
        "impact_event_time_mismatch",
        "impact event sample_index and emg_time_s are inconsistent",
    )
    # Ceil preserves the complete requested physical-time interval even when
    # duration * sampling-rate is fractional; the evaluator may interpolate at
    # that fractional endpoint later.
    pre_samples = int(np.ceil(pre_event_s * sampling_rate_hz))
    post_samples = int(np.ceil(post_event_s * sampling_rate_hz))
    _reject(
        pre_samples <= 0 or post_samples <= 0,
        "impact_window_invalid",
        "impact window rounds to zero samples",
    )
    start = sample - pre_samples
    stop = sample + post_samples + 1
    _reject(
        start < 0 or stop > len(time_s),
        "impact_window_incomplete",
        "impact event does not have a complete requested pre/post window",
    )
    evidence = {
        "event_name": event_name,
        "sample_index": sample,
        "emg_time_s": event_time,
        "source": source,
        "confidence": confidence,
        "minimum_confidence": minimum_confidence,
        "time_tolerance_s": tolerance,
        "cue_fallback_used": False,
        "annotation_audit": _validate_event_annotation_audit(
            annotation_audit_path,
            events_path=path,
            event_name=event_name,
            event_row=event,
        ),
    }
    return sample, start, stop, evidence


def _import_trial(
    *,
    selection: _Selection,
    session: _SessionEvidence,
    trial_id: str,
) -> _ImportedTrial:
    trial_index = int(trial_id.rsplit("_", maxsplit=1)[1])
    trial_directory = _descendant(
        selection.session_path,
        Path("trials") / selection.action_id / f"trial_{trial_index:03d}",
        f"trial {trial_id}",
    )
    required_paths = {
        "metadata": trial_directory / "metadata.json",
        "raw_emg": trial_directory / "raw_emg.npz",
        "processed_emg": trial_directory / "processed_emg.npz",
        "processing": trial_directory / "processing.json",
        "raw_qc": trial_directory / "qc.json",
        "preprocessing_qc": trial_directory / "preprocessing_qc.json",
        "events": trial_directory / "events.csv",
        "events_annotation_audit": trial_directory / "events.annotation.audit.jsonl",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    _reject(bool(missing), "required_file_missing", f"trial is missing required files: {missing}")
    metadata = _json_object(required_paths["metadata"], "trial metadata")
    processing = _json_object(required_paths["processing"], "processing manifest")
    raw_qc = _json_object(required_paths["raw_qc"], "raw acquisition QC")
    qc = _json_object(required_paths["preprocessing_qc"], "preprocessing QC")
    _validate_trial_metadata(
        metadata,
        selection=selection,
        session=session,
        trial_id=trial_id,
        trial_index=trial_index,
    )
    _validate_processing(processing, session=session)
    raw_emg, raw_time_s, raw_sample_index, raw_fs = _load_raw_signal(
        required_paths["raw_emg"],
        metadata=metadata,
    )
    raw_warning_codes = _validate_raw_qc(raw_qc, metadata=metadata, raw_emg=raw_emg)
    preprocessing_warning_codes = _validate_preprocessing_qc(qc)
    warning_codes = tuple(
        sorted(set(raw_warning_codes) | {f"preprocessing:{value}" for value in preprocessing_warning_codes})
    )
    envelope, time_s, fs = _load_processed_signal(
        required_paths["processed_emg"],
        selection=selection,
        session=session,
        metadata=metadata,
        trial_id=trial_id,
        trial_index=trial_index,
    )
    _reject(
        envelope.shape[0] != raw_emg.shape[0]
        or not np.array_equal(raw_sample_index, np.arange(envelope.shape[0]))
        or not np.allclose(raw_time_s, time_s, rtol=0.0, atol=1e-12)
        or not np.isclose(raw_fs, fs, rtol=0.0, atol=1e-12),
        "raw_processed_mismatch",
        "raw and processed signal length/time/sample-rate evidence differs",
    )
    impact, start, stop, event_evidence = _read_impact_event(
        required_paths["events"],
        annotation_audit_path=required_paths["events_annotation_audit"],
        event_name=selection.event_name,
        minimum_confidence=selection.minimum_confidence,
        time_s=time_s,
        sampling_rate_hz=fs,
        pre_event_s=selection.pre_event_s,
        post_event_s=selection.post_event_s,
    )
    hashes = {name: _file_sha256(path) for name, path in required_paths.items()}
    source_file_set_sha256 = _canonical_sha256(dict(sorted(hashes.items())))
    return _ImportedTrial(
        source_trial_id=trial_id,
        trial_uid=stable_jidian_trial_uid(
            subject_uid=selection.subject_uid,
            session_uid=selection.session_uid,
            action_id=selection.action_id,
            source_trial_id=trial_id,
        ),
        reference_trial_fingerprint=selection.reference_trial_fingerprints.get(trial_id),
        # This is a direct crop of normalized_envelope.  No filtering, rectifying,
        # smoothing, clipping, or renormalization occurs in this importer.
        emg=envelope[start:stop].copy(),
        sampling_rate_hz=fs,
        impact_frame=impact - start,
        source_impact_frame=impact,
        crop_start=start,
        crop_stop=stop,
        processing_sha256=hashes["processing"],
        source_file_set_sha256=source_file_set_sha256,
        file_hashes=dict(sorted(hashes.items())),
        warning_codes=warning_codes,
        event_evidence=event_evidence,
    )


def _trial_audit(selection: _Selection, trial_id: str) -> dict[str, Any]:
    audit = {
        "source_trial_id": trial_id,
        "trial_uid": stable_jidian_trial_uid(
            subject_uid=selection.subject_uid,
            session_uid=selection.session_uid,
            action_id=selection.action_id,
            source_trial_id=trial_id,
        ),
        "eligible": False,
        "included": False,
        "reason_codes": [],
        "reasons": [],
        "hashes": {},
    }
    if selection.comparison_design == PAIRED_COMPARISON_DESIGN:
        audit["reference_trial_fingerprint"] = selection.reference_trial_fingerprints[trial_id]
    return audit


def _build_arrays(
    selection: _Selection,
    session: _SessionEvidence,
    trials: Sequence[_ImportedTrial],
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    fs_values = np.asarray([trial.sampling_rate_hz for trial in trials], dtype=np.float64)
    if not np.allclose(fs_values, fs_values[0], rtol=0.0, atol=1e-9):
        raise ValueError("selected trials have inconsistent sampling rates")
    lengths = {trial.emg.shape[0] for trial in trials}
    if len(lengths) != 1:
        raise ValueError("selected event-window crops have inconsistent lengths")
    processing_binding = {
        "schema_version": PROCESSING_MANIFEST_SCHEMA_VERSION,
        "channel_profile_sha256": session.profile_sha256,
        "normalization_method": "mvc",
        "processing_fallback_method": "none",
        "processing_config": session.processing_config,
        "mvc_manifest_sha256": session.mvc_manifest_sha256,
        "mvc_source_set_sha256": session.mvc_source_set_sha256,
        "trials": {
            trial.source_trial_id: {
                "processing_json_sha256": trial.file_hashes["processing"],
                "raw_emg_npz_sha256": trial.file_hashes["raw_emg"],
                "processed_emg_npz_sha256": trial.file_hashes["processed_emg"],
                "preprocessing_qc_json_sha256": trial.file_hashes["preprocessing_qc"],
                "raw_qc_json_sha256": trial.file_hashes["raw_qc"],
                "events_csv_sha256": trial.file_hashes["events"],
                "events_annotation_audit_jsonl_sha256": trial.file_hashes["events_annotation_audit"],
                "source_file_set_sha256": trial.source_file_set_sha256,
            }
            for trial in trials
        },
    }
    processing_manifest_sha256 = _canonical_sha256(processing_binding)
    source_binding = {
        "schema_version": "jidian_emg_source_provenance_v1",
        "selection_manifest_sha256": selection.source_sha256,
        "session_json_sha256": session.session_sha256,
        "channel_profile_sha256": session.profile_sha256,
        "mvc_manifest_sha256": session.mvc_manifest_sha256,
        "mvc_source_set_sha256": session.mvc_source_set_sha256,
        "trial_source_file_set_sha256": {trial.source_trial_id: trial.source_file_set_sha256 for trial in trials},
    }
    source_provenance_sha256 = _canonical_sha256(source_binding)
    arrays = {
        "import_schema_version": np.asarray(IMPORT_SCHEMA_VERSION),
        "emg": np.stack([trial.emg for trial in trials], axis=0).astype(np.float64),
        "emg_signal_kind": np.asarray(EMG_SIGNAL_KIND),
        "channel_names": np.asarray(EXPECTED_CHANNEL_NAMES),
        "stream_channel_ids": np.asarray(EXPECTED_STREAM_CHANNEL_IDS, dtype=np.int16),
        "sides": np.asarray(EXPECTED_SIDES),
        "muscle_slugs": np.asarray(EXPECTED_MUSCLE_SLUGS),
        "sampling_rate_hz": np.asarray(float(fs_values[0]), dtype=np.float64),
        "impact_frame": np.asarray([trial.impact_frame for trial in trials], dtype=np.int64),
        "trial_uid": np.asarray([trial.trial_uid for trial in trials]),
        "subject_uid": np.asarray([selection.subject_uid] * len(trials)),
        "session_uid": np.asarray([selection.session_uid] * len(trials)),
        "dataset_split": np.asarray(selection.dataset_split),
        "training_session_uid": np.asarray(selection.training_session_uids),
        "comparison_design": np.asarray(selection.comparison_design),
        "comparison_set_uid": np.asarray(selection.comparison_set_uid),
        "processing_manifest_schema_version": np.asarray(PROCESSING_MANIFEST_SCHEMA_VERSION),
        "processing_manifest_sha256": np.asarray(processing_manifest_sha256),
        "source_provenance_sha256": np.asarray(source_provenance_sha256),
        "selection_manifest_sha256": np.asarray(selection.source_sha256),
        "channel_profile_id": np.asarray(EXPECTED_PROFILE_ID),
        "channel_profile_version": np.asarray(EXPECTED_PROFILE_VERSION, dtype=np.int64),
        "channel_profile_sha256": np.asarray(session.profile_sha256),
        "handedness": np.asarray(EXPECTED_HANDEDNESS),
        "normalization_method": np.asarray("mvc"),
        "processing_fallback_method": np.asarray("none"),
        "mvc_values": session.mvc_values.astype(np.float64),
        "acquired_channel_count": np.asarray(ACQUIRED_CHANNEL_COUNT, dtype=np.int64),
        "comparable_channel_count": np.asarray(COMPARABLE_CHANNEL_COUNT, dtype=np.int64),
        "excluded_sensor_ids": np.asarray(EXCLUDED_SENSOR_IDS, dtype=np.int16),
        "excluded_sensor_reason": np.asarray(EXCLUDED_SENSOR_REASON),
        "action_id": np.asarray(selection.action_id),
        "source_trial_id": np.asarray([trial.source_trial_id for trial in trials]),
        "source_impact_frame": np.asarray([trial.source_impact_frame for trial in trials], dtype=np.int64),
        "source_crop_start": np.asarray([trial.crop_start for trial in trials], dtype=np.int64),
        "source_crop_stop": np.asarray([trial.crop_stop for trial in trials], dtype=np.int64),
        "trial_source_file_set_sha256": np.asarray([trial.source_file_set_sha256 for trial in trials]),
        "qc_warning_count": np.asarray([len(trial.warning_codes) for trial in trials], dtype=np.int64),
    }
    if selection.comparison_design == PAIRED_COMPARISON_DESIGN:
        arrays["reference_trial_fingerprint"] = np.asarray([trial.reference_trial_fingerprint for trial in trials])
    provenance = {
        "processing_binding": processing_binding,
        "processing_manifest_sha256": processing_manifest_sha256,
        "source_binding": source_binding,
        "source_provenance_sha256": source_provenance_sha256,
    }
    return arrays, provenance


def _validate_export_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    emg = np.asarray(arrays["emg"])
    if emg.ndim != 3 or emg.shape[0] < 1 or emg.shape[2] != ACQUIRED_CHANNEL_COUNT:
        raise ValueError("export emg must have shape [trial,time,16]")
    if not np.all(np.isfinite(emg)) or np.any(emg < 0.0):
        raise ValueError("export emg contains NaN/Inf or negative values")
    count = emg.shape[0]
    if np.asarray(arrays["channel_names"]).astype(str).tolist() != list(EXPECTED_CHANNEL_NAMES):
        raise ValueError("export channel_names changed after validation")
    if np.asarray(arrays["stream_channel_ids"]).astype(int).tolist() != list(EXPECTED_STREAM_CHANNEL_IDS):
        raise ValueError("export stream_channel_ids changed after validation")
    if len(set(np.asarray(arrays["trial_uid"]).astype(str).tolist())) != count:
        raise ValueError("export trial_uid values must be unique")
    for name in (
        "trial_uid",
        "subject_uid",
        "session_uid",
        "source_trial_id",
        "source_impact_frame",
        "source_crop_start",
        "source_crop_stop",
        "trial_source_file_set_sha256",
        "qc_warning_count",
        "impact_frame",
    ):
        if np.asarray(arrays[name]).shape != (count,):
            raise ValueError(f"export {name} must have one value per trial")
    for name in (
        "processing_manifest_sha256",
        "source_provenance_sha256",
        "selection_manifest_sha256",
        "channel_profile_sha256",
    ):
        _require_sha256(str(np.asarray(arrays[name]).reshape(-1)[0]), name)
    design = str(np.asarray(arrays["comparison_design"]).reshape(-1)[0])
    if design == PAIRED_COMPARISON_DESIGN:
        references = np.asarray(arrays.get("reference_trial_fingerprint"))
        if references.shape != (count,):
            raise ValueError("paired export requires one reference_trial_fingerprint per trial")
        for value in references.astype(str).tolist():
            _require_sha256(value, "reference_trial_fingerprint")
    elif design == UNPAIRED_COMPARISON_DESIGN:
        if "reference_trial_fingerprint" in arrays:
            raise ValueError("unpaired export must not contain reference_trial_fingerprint")
    else:
        raise ValueError("export comparison_design is unsupported")
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise ValueError("export arrays must not use pickle/object dtypes")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finalize_audit(report: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(report)
    unsigned.pop("audit_fingerprint", None)
    report["audit_fingerprint"] = _canonical_sha256(unsigned)
    return report


def _write_rejection(
    report: dict[str, Any],
    *,
    audit_path: Path,
    message: str,
) -> JidianEmgImportError:
    report["status"] = "rejected"
    report["output_npz_written"] = False
    _finalize_audit(report)
    _atomic_write_json(audit_path, report)
    return JidianEmgImportError(message, audit_path=audit_path, report=report)


def import_jidian_emg(
    selection_manifest: str | Path,
    output_npz: str | Path,
    *,
    audit_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically export one explicit Jidian comparison set.

    On every contract rejection, :class:`JidianEmgImportError` points to the
    written audit.  Existing NPZ outputs are never overwritten.
    """

    selection_path = Path(selection_manifest).expanduser().resolve()
    output_path = Path(output_npz).expanduser().resolve()
    audit_path = (
        Path(audit_json).expanduser().resolve() if audit_json is not None else output_path.with_suffix(".audit.json")
    )
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "validating",
        "selection_manifest_path": str(selection_path),
        "output_npz_path": str(output_path),
        "audit_path": str(audit_path),
        "output_npz_written": False,
        "fail_closed": True,
        "selection_all_or_nothing": True,
        "global_reason_codes": [],
        "global_reasons": [],
        "trials": [],
    }
    if audit_path == output_path:
        audit_path = output_path.with_suffix(output_path.suffix + ".rejected.audit.json")
        report["audit_path"] = str(audit_path)
        report["global_reason_codes"].append("output_path_conflict")
        report["global_reasons"].append(
            _reason("output_path_conflict", "audit_json and output_npz must be different paths")
        )
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected")
    if output_path.exists():
        report["global_reason_codes"].append("output_exists")
        report["global_reasons"].append(_reason("output_exists", "existing NPZ outputs are never overwritten"))
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected")

    try:
        selection = _parse_selection(selection_path)
        report["selection"] = {
            "selection_manifest_sha256": selection.source_sha256,
            "session_path": str(selection.session_path),
            "subject_uid": selection.subject_uid,
            "session_uid": selection.session_uid,
            "action_id": selection.action_id,
            "trial_ids": list(selection.trial_ids),
            "dataset_split": selection.dataset_split,
            "comparison_design": selection.comparison_design,
            "comparison_set_uid": selection.comparison_set_uid,
            "training_session_uids": list(selection.training_session_uids),
            "alignment": {
                "mode": "impact",
                "event_name": selection.event_name,
                "minimum_confidence": selection.minimum_confidence,
                "pre_event_s": selection.pre_event_s,
                "post_event_s": selection.post_event_s,
                "cue_fallback_allowed": False,
            },
        }
        session = _load_session_evidence(selection)
        report["session_evidence"] = {
            "participant_id": session.participant_id,
            "source_session_id": session.source_session_id,
            "session_json_sha256": session.session_sha256,
            "channel_profile_id": EXPECTED_PROFILE_ID,
            "channel_profile_version": EXPECTED_PROFILE_VERSION,
            "channel_profile_sha256": session.profile_sha256,
            "handedness": EXPECTED_HANDEDNESS,
            "mvc_manifest_sha256": session.mvc_manifest_sha256,
            "mvc_source_set_sha256": session.mvc_source_set_sha256,
            "mvc_source_hashes": session.mvc_source_hashes,
            "normalization_method": "mvc",
            "processing_fallback_method": "none",
            "acquired_channel_count": ACQUIRED_CHANNEL_COUNT,
            "comparable_channel_count": COMPARABLE_CHANNEL_COUNT,
            "excluded_sensor_ids": list(EXCLUDED_SENSOR_IDS),
            "excluded_sensor_reason": EXCLUDED_SENSOR_REASON,
        }
    except (OSError, ValueError, KeyError) as exc:
        report["global_reason_codes"].append("selection_or_session_contract_invalid")
        report["global_reasons"].append(_reason("selection_or_session_contract_invalid", str(exc)))
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected") from exc

    imported: list[_ImportedTrial] = []
    audit_by_id: dict[str, dict[str, Any]] = {}
    for trial_id in selection.trial_ids:
        audit = _trial_audit(selection, trial_id)
        audit_by_id[trial_id] = audit
        report["trials"].append(audit)
        try:
            trial = _import_trial(selection=selection, session=session, trial_id=trial_id)
        except _TrialRejectionError as exc:
            audit["reason_codes"].append(exc.code)
            audit["reasons"].append(_reason(exc.code, exc.detail))
        except (OSError, ValueError, KeyError) as exc:
            audit["reason_codes"].append("trial_contract_invalid")
            audit["reasons"].append(_reason("trial_contract_invalid", str(exc)))
        else:
            imported.append(trial)
            audit.update(
                {
                    "eligible": True,
                    "hashes": dict(trial.file_hashes),
                    "source_file_set_sha256": trial.source_file_set_sha256,
                    "event": dict(trial.event_evidence),
                    "source_impact_frame": trial.source_impact_frame,
                    "crop_start": trial.crop_start,
                    "crop_stop": trial.crop_stop,
                    "output_impact_frame": trial.impact_frame,
                    "output_samples": int(trial.emg.shape[0]),
                    "qc_warning_codes": list(trial.warning_codes),
                    "qc_warning_count": len(trial.warning_codes),
                }
            )

    report["summary"] = {
        "selected_trial_count": len(selection.trial_ids),
        "eligible_trial_count": len(imported),
        "included_trial_count": 0,
        "rejected_trial_count": len(selection.trial_ids) - len(imported),
    }
    if len(imported) != len(selection.trial_ids):
        for trial in imported:
            audit = audit_by_id[trial.source_trial_id]
            audit["reason_codes"].append("selection_incomplete")
            audit["reasons"].append(
                _reason(
                    "selection_incomplete",
                    "another explicitly selected trial failed; partial comparison-set export is forbidden",
                )
            )
        report["global_reason_codes"].append("selection_incomplete")
        report["global_reasons"].append(
            _reason(
                "selection_incomplete",
                "not every explicitly selected trial passed the import contract",
            )
        )
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected")

    try:
        arrays, provenance = _build_arrays(selection, session, imported)
        _validate_export_arrays(arrays)
    except (ValueError, TypeError) as exc:
        for audit in report["trials"]:
            audit["reason_codes"].append("cross_trial_contract_invalid")
            audit["reasons"].append(_reason("cross_trial_contract_invalid", str(exc)))
        report["global_reason_codes"].append("cross_trial_contract_invalid")
        report["global_reasons"].append(_reason("cross_trial_contract_invalid", str(exc)))
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.npz")
    temporary_audit = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    wrote_output = False
    try:
        np.savez_compressed(temporary_npz, **arrays)
        with np.load(temporary_npz, allow_pickle=False) as stored:
            reloaded = {name: np.asarray(stored[name]) for name in stored.files}
        _validate_export_arrays(reloaded)
        for name, expected in arrays.items():
            if not np.array_equal(reloaded[name], expected, equal_nan=False):
                raise RuntimeError(f"persisted temporary NPZ field changed: {name}")
        npz_sha256 = _file_sha256(temporary_npz)
        array_hashes = {name: _array_sha256(value) for name, value in sorted(arrays.items())}
        for audit in report["trials"]:
            audit["included"] = True
        report["summary"]["included_trial_count"] = len(imported)
        report["summary"]["rejected_trial_count"] = 0
        report["status"] = "exported"
        report["output_npz_written"] = True
        report["npz_sha256"] = npz_sha256
        report["array_sha256"] = array_hashes
        report["provenance"] = provenance
        _finalize_audit(report)
        temporary_audit.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_npz, output_path)
        wrote_output = True
        os.replace(temporary_audit, audit_path)
    except Exception:
        if wrote_output and output_path.exists():
            output_path.unlink()
        raise
    finally:
        if temporary_npz.exists():
            temporary_npz.unlink()
        if temporary_audit.exists():
            temporary_audit.unlink()

    if _file_sha256(output_path) != report["npz_sha256"]:
        output_path.unlink()
        report["status"] = "rejected"
        report["output_npz_written"] = False
        report["global_reason_codes"].append("persisted_npz_verification_failed")
        report["global_reasons"].append(_reason("persisted_npz_verification_failed", "final NPZ content hash changed"))
        raise _write_rejection(report, audit_path=audit_path, message="Jidian EMG import rejected")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly import one explicit Jidian v2 sEMG comparison set",
    )
    parser.add_argument("selection_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--audit-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = import_jidian_emg(
            args.selection_manifest,
            args.output_npz,
            audit_json=args.audit_json,
        )
    except JidianEmgImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"exported {report['summary']['included_trial_count']} trial(s) to {report['output_npz_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
