from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .storage import append_jsonl


EVENT_FIELDS = (
    "event_name",
    "sample_index",
    "emg_time_s",
    "monotonic_time_ns",
    "wall_clock_iso",
    "source",
    "confidence",
    "notes",
)

# Software cues are acquisition facts and must remain immutable.  Only slots
# deliberately created for later evidence-backed annotation can be edited.
ANNOTATABLE_EVENTS = frozenset(
    {
        "movement_start_manual",
        "racket_contact",
        "foot_contact",
        "takeoff",
        "landing",
    }
)

# These names describe the evidence used to place an event.  In particular,
# no software-cue source is accepted for a physical contact or movement event.
ALLOWED_ANNOTATION_SOURCES = frozenset(
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

AUDIT_SCHEMA_VERSION = "emg_event_annotation_audit_v1"
DEFAULT_AUDIT_FILENAME = "events.annotation.audit.jsonl"
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class EventAnnotationError(ValueError):
    """Raised when an annotation would weaken the recorded event contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _annotation_manifest_sha256(base_audit: dict[str, Any]) -> str:
    payload = json.dumps(
        base_audit,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _read_events(path: Path) -> tuple[list[dict[str, str]], bytes]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise EventAnnotationError(f"Missing event table: {path}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EventAnnotationError(f"Event table is not UTF-8: {path}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != EVENT_FIELDS:
        raise EventAnnotationError(
            f"Event table header must be exactly {','.join(EVENT_FIELDS)}"
        )
    rows = [{field: str(row.get(field, "")) for field in EVENT_FIELDS} for row in reader]
    names = [row["event_name"].strip() for row in rows]
    duplicates = sorted(name for name, count in Counter(names).items() if not name or count != 1)
    if duplicates:
        raise EventAnnotationError(
            f"Event names must be non-empty and unique; invalid names: {duplicates}"
        )
    return rows, raw


def _serialize_events(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EVENT_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in EVENT_FIELDS})
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_time_contract(trial_dir: Path) -> tuple[int, float, np.ndarray]:
    raw_path = trial_dir / "raw_emg.npz"
    if not raw_path.exists():
        raise EventAnnotationError(f"Missing raw EMG bundle: {raw_path}")
    try:
        with np.load(raw_path, allow_pickle=False) as payload:
            emg = np.asarray(payload["emg_mV"])
            time_s = np.asarray(payload["time_s"], dtype=np.float64)
            sample_index = np.asarray(payload["sample_index"])
            fs_hz = float(payload["fs_hz"])
    except (KeyError, OSError, ValueError) as exc:
        raise EventAnnotationError(f"Cannot read raw EMG timing contract at {raw_path}: {exc}") from exc
    if emg.ndim != 2 or len(emg) < 1:
        raise EventAnnotationError("Raw EMG must be a non-empty [samples, channels] array")
    if time_s.ndim != 1 or len(time_s) != len(emg) or not np.all(np.isfinite(time_s)):
        raise EventAnnotationError("Raw EMG time_s must be finite and match the sample count")
    if (
        sample_index.ndim != 1
        or len(sample_index) != len(emg)
        or not np.issubdtype(sample_index.dtype, np.integer)
        or not np.array_equal(sample_index, np.arange(len(emg)))
    ):
        raise EventAnnotationError("Raw EMG sample_index must be contiguous zero-based integers")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise EventAnnotationError("Raw EMG time_s must be strictly increasing")
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise EventAnnotationError("Raw EMG fs_hz must be finite and positive")
    expected_time_s = np.arange(len(emg), dtype=np.float64) / fs_hz
    if not np.allclose(time_s, expected_time_s, rtol=0.0, atol=0.5 / fs_hz + 1e-12):
        raise EventAnnotationError("Raw EMG time_s is inconsistent with sample_index/fs_hz")
    return len(emg), fs_hz, time_s


def _canonical_sample_and_time(
    *,
    sample_count: int,
    fs_hz: float,
    time_axis_s: np.ndarray,
    sample_index: int | None,
    emg_time_s: float | None,
) -> tuple[int, float]:
    if sample_index is None and emg_time_s is None:
        raise EventAnnotationError("Provide --sample-index, --emg-time-s, or both")
    if emg_time_s is not None and (not math.isfinite(emg_time_s) or emg_time_s < 0):
        raise EventAnnotationError("emg_time_s must be finite and non-negative")
    if sample_index is not None and (
        isinstance(sample_index, bool) or not isinstance(sample_index, (int, np.integer))
    ):
        raise EventAnnotationError("sample_index must be an integer")
    if sample_index is None:
        assert emg_time_s is not None
        sample_index = int(round(emg_time_s * fs_hz))
    if sample_index < 0 or sample_index >= sample_count:
        raise EventAnnotationError(
            f"sample_index must be in [0, {sample_count - 1}], got {sample_index}"
        )
    canonical_time = float(time_axis_s[sample_index])
    if emg_time_s is not None:
        tolerance_s = 0.5 / fs_hz + 1e-9
        if abs(float(emg_time_s) - canonical_time) > tolerance_s:
            raise EventAnnotationError(
                f"sample/time mismatch: sample {sample_index} is {canonical_time:.9f}s, "
                f"not {float(emg_time_s):.9f}s"
            )
    return int(sample_index), canonical_time


def verify_committed_annotation(
    trial_dir: Path | str,
    event_name: str,
) -> dict[str, Any]:
    """Verify a target event and the latest whole-table audit commit.

    A later annotation of another event legitimately changes the whole CSV
    hash.  Therefore the latest *global* committed record must bind the current
    table, while the latest target-event record must still match the current
    target row exactly.  This closes the manual-CSV-edit bypass without
    preventing multiple independently annotated slots.
    """

    trial_dir = Path(trial_dir).resolve()
    rows, current_bytes = _read_events(trial_dir / "events.csv")
    current_sha256 = _sha256_bytes(current_bytes)
    current_matches = [row for row in rows if row["event_name"] == event_name]
    if len(current_matches) != 1:
        raise EventAnnotationError(
            f"Expected exactly one current {event_name!r} row, found {len(current_matches)}"
        )
    current_event = current_matches[0]
    audit_path = trial_dir / DEFAULT_AUDIT_FILENAME
    if not audit_path.exists():
        raise EventAnnotationError(f"Missing committed annotation audit: {audit_path}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise EventAnnotationError(f"Audit line {line_number} is not a JSON object")
            records.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventAnnotationError(f"Cannot parse annotation audit {audit_path}: {exc}") from exc

    committed_with_index = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("transaction_state") == "committed"
    ]
    if not committed_with_index:
        raise EventAnnotationError(f"No committed annotation exists in {audit_path}")
    latest_global_index, latest_global = committed_with_index[-1]
    target_commits = [
        (index, record)
        for index, record in committed_with_index
        if record.get("event_name") == event_name
    ]
    if not target_commits:
        raise EventAnnotationError(f"No committed annotation exists for {event_name!r}")
    target_index, target_record = target_commits[-1]

    def validate_record(index: int, record: dict[str, Any]) -> None:
        if record.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            raise EventAnnotationError("Unsupported annotation audit schema")
        for field in ("before_sha256", "after_sha256", "evidence_sha256"):
            if not SHA256_PATTERN.fullmatch(str(record.get(field, ""))):
                raise EventAnnotationError(f"Committed annotation has invalid {field}")
        base_audit = {
            key: value
            for key, value in record.items()
            if key not in {"transaction_state", "annotation_manifest_sha256"}
        }
        expected_manifest = _annotation_manifest_sha256(base_audit)
        if record.get("annotation_manifest_sha256") != expected_manifest:
            raise EventAnnotationError("Committed annotation manifest hash mismatch")
        annotation_id = record.get("annotation_id")
        prepared = [
            candidate
            for candidate in records[:index]
            if candidate.get("transaction_state") == "prepared"
            and candidate.get("annotation_id") == annotation_id
        ]
        committed_common = {
            key: value for key, value in record.items() if key != "transaction_state"
        }
        if not prepared or {
            key: value for key, value in prepared[-1].items() if key != "transaction_state"
        } != committed_common:
            raise EventAnnotationError("Committed annotation has no matching prepared audit record")

    validate_record(latest_global_index, latest_global)
    if target_index != latest_global_index:
        validate_record(target_index, target_record)
    if latest_global.get("after_sha256") != current_sha256:
        raise EventAnnotationError(
            "Current events.csv is not bound by the latest committed annotation hash"
        )
    if target_record.get("after_event") != current_event:
        raise EventAnnotationError(
            f"Current {event_name!r} row differs from its latest committed annotation"
        )
    source = current_event.get("source", "").strip()
    if source not in ALLOWED_ANNOTATION_SOURCES:
        raise EventAnnotationError(
            f"Current {event_name!r} source is not evidence-backed: {source!r}"
        )
    try:
        confidence = float(current_event["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EventAnnotationError(f"Current {event_name!r} confidence is invalid") from exc
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise EventAnnotationError(f"Current {event_name!r} confidence must be in (0, 1]")
    return {
        "annotation_id": target_record["annotation_id"],
        "annotation_manifest_sha256": target_record["annotation_manifest_sha256"],
        "evidence_reference": target_record["evidence_reference"],
        "evidence_sha256": target_record["evidence_sha256"],
        "events_sha256": current_sha256,
        "source": source,
        "confidence": confidence,
    }


def annotate_event(
    trial_dir: Path | str,
    event_name: str,
    *,
    source: str,
    confidence: float,
    annotator: str,
    evidence_reference: str,
    evidence_sha256: str,
    sample_index: int | None = None,
    emg_time_s: float | None = None,
    notes: str = "",
    overwrite: bool = False,
    expected_before_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically annotate one predeclared event slot and append a two-phase audit.

    The ``prepared`` audit record is flushed before the atomic replacement and
    the ``committed`` record afterwards.  If a process is interrupted, the two
    recorded hashes make it possible to determine whether the old or new event
    table is present without guessing.
    """

    trial_dir = Path(trial_dir).resolve()
    event_name = str(event_name).strip()
    source = str(source).strip()
    annotator = str(annotator).strip()
    evidence_reference = str(evidence_reference).strip()
    evidence_sha256 = str(evidence_sha256).strip().lower()
    notes = str(notes).strip()
    if event_name not in ANNOTATABLE_EVENTS:
        raise EventAnnotationError(
            f"Event {event_name!r} is immutable or unsupported; annotatable events are "
            f"{sorted(ANNOTATABLE_EVENTS)}. movement_cue can never be relabeled as contact."
        )
    if source not in ALLOWED_ANNOTATION_SOURCES:
        raise EventAnnotationError(
            f"Unsupported annotation source {source!r}; choose from {sorted(ALLOWED_ANNOTATION_SOURCES)}"
        )
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise EventAnnotationError("confidence must be finite and in (0, 1]")
    if not annotator:
        raise EventAnnotationError("annotator is required (use a pseudonymous operator ID)")
    if not evidence_reference:
        raise EventAnnotationError(
            "evidence_reference is required; a software cue is not evidence of physical contact"
        )
    if not SHA256_PATTERN.fullmatch(evidence_sha256):
        raise EventAnnotationError(
            "evidence_sha256 is required and must be exactly 64 hexadecimal characters"
        )

    events_path = trial_dir / "events.csv"
    rows, before_bytes = _read_events(events_path)
    before_sha256 = _sha256_bytes(before_bytes)
    if expected_before_sha256 is not None and expected_before_sha256 != before_sha256:
        raise EventAnnotationError(
            f"events.csv changed since review: expected {expected_before_sha256}, got {before_sha256}"
        )
    matching = [index for index, row in enumerate(rows) if row["event_name"] == event_name]
    if len(matching) != 1:
        raise EventAnnotationError(f"Expected exactly one {event_name!r} event slot, found {len(matching)}")
    row_index = matching[0]
    before_event = dict(rows[row_index])
    already_annotated = (
        before_event["sample_index"].strip() != ""
        or before_event["source"].strip() not in {"", "unannotated"}
    )
    if already_annotated and not overwrite:
        raise EventAnnotationError(
            f"Event {event_name!r} is already annotated; pass --overwrite after reviewing its audit history"
        )

    sample_count, fs_hz, time_axis_s = _load_time_contract(trial_dir)
    canonical_sample, canonical_time = _canonical_sample_and_time(
        sample_count=sample_count,
        fs_hz=fs_hz,
        time_axis_s=time_axis_s,
        sample_index=sample_index,
        emg_time_s=emg_time_s,
    )
    row_notes = f"evidence_reference={evidence_reference}"
    if notes:
        row_notes += f"; {notes}"
    after_event = {
        "event_name": event_name,
        "sample_index": str(canonical_sample),
        "emg_time_s": f"{canonical_time:.9f}",
        # A post-hoc annotation has no acquisition-time monotonic/wall clock.
        # The annotation timestamp itself is retained in the audit sidecar.
        "monotonic_time_ns": "",
        "wall_clock_iso": "",
        "source": source,
        "confidence": f"{float(confidence):.6g}",
        "notes": row_notes,
    }
    rows[row_index] = after_event
    after_bytes = _serialize_events(rows)
    after_sha256 = _sha256_bytes(after_bytes)
    if after_sha256 == before_sha256:
        raise EventAnnotationError("Annotation produced no change to events.csv")

    annotation_id = uuid.uuid4().hex
    annotated_at = datetime.now(timezone.utc).isoformat()
    audit_path = trial_dir / DEFAULT_AUDIT_FILENAME
    base_audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "annotation_id": annotation_id,
        "annotated_at": annotated_at,
        "event_name": event_name,
        "annotator": annotator,
        "evidence_reference": evidence_reference,
        "evidence_sha256": evidence_sha256,
        "overwrite": bool(overwrite),
        "events_path": "events.csv",
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "before_event": before_event,
        "after_event": after_event,
    }
    annotation_manifest_sha256 = _annotation_manifest_sha256(base_audit)
    audit_common = {
        **base_audit,
        "annotation_manifest_sha256": annotation_manifest_sha256,
    }
    append_jsonl(audit_path, {**audit_common, "transaction_state": "prepared"})
    _atomic_write_bytes(events_path, after_bytes)
    actual_after_sha256 = _sha256_bytes(events_path.read_bytes())
    if actual_after_sha256 != after_sha256:
        raise RuntimeError(
            f"Atomic annotation hash mismatch: expected {after_sha256}, got {actual_after_sha256}"
        )
    append_jsonl(audit_path, {**audit_common, "transaction_state": "committed"})
    return {
        "annotation_id": annotation_id,
        "event_name": event_name,
        "sample_index": canonical_sample,
        "emg_time_s": canonical_time,
        "source": source,
        "confidence": float(confidence),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "evidence_sha256": evidence_sha256,
        "annotation_manifest_sha256": annotation_manifest_sha256,
        "audit_path": str(audit_path),
    }
