from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import __version__
from .models import (
    SESSION_METADATA_SCHEMA_VERSION,
    SESSION_METADATA_V2_OPTIONAL_PROVENANCE_FIELDS,
    ChannelProfile,
    RecordResult,
    SessionMetadata,
    TrialMetadata,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def safe_identifier(value: str, label: str = "identifier") -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or not SAFE_ID.fullmatch(value) or ".." in value:
        raise ValueError(f"Unsafe {label}: {value!r}; use ASCII letters, digits, dot, dash, or underscore")
    return value


def safe_child(root: Path | str, *parts: str) -> Path:
    root = Path(root).expanduser().resolve()
    safe_parts = [safe_identifier(part) for part in parts]
    target = root.joinpath(*safe_parts).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes dataset root: {target}")
    return target


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _atomic_target(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    fd, temp_path = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return path


def atomic_save_npz(path: Path | str, **arrays: Any) -> Path:
    path = Path(path)
    fd, temp_path = _atomic_target(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return path


def atomic_write_csv(path: Path | str, rows: Iterable[Iterable[Any]]) -> Path:
    path = Path(path)
    fd, temp_path = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return path


def append_jsonl(path: Path | str, payload: Any) -> Path:
    """Append one UTF-8 JSON record and flush it to disk for audit logging."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=_json_default, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def session_path(dataset_root: Path | str, participant_id: str, session_id: str) -> Path:
    return safe_child(dataset_root, participant_id, session_id)


def trial_path(
    dataset_root: Path | str,
    participant_id: str,
    session_id: str,
    action_id: str,
    trial_index: int,
) -> Path:
    if trial_index < 1:
        raise ValueError("trial_index must be >= 1")
    return safe_child(
        dataset_root,
        participant_id,
        session_id,
        "trials",
        action_id,
        f"trial_{trial_index:03d}",
    )


def initialize_session(
    root: Path,
    participant_id: str,
    session_id: str,
    session_metadata: dict[str, Any],
    profile: ChannelProfile,
    protocol: dict[str, Any],
) -> Path:
    path = session_path(root, participant_id, session_id)
    path.mkdir(parents=True, exist_ok=True)
    session_file = path / "session.json"
    if session_file.exists():
        existing = read_json(session_file)
        # collection_date is generated on each invocation but the original
        # value remains authoritative.  Every other supplied session field is
        # immutable so a resumed trial cannot silently disagree with its
        # session-level anthropometrics, equipment, operator, or laterality.
        legacy_schema = "schema_version" not in existing
        if legacy_schema:
            warnings.warn(
                f"Session {participant_id}/{session_id} has no schema_version; "
                "accepting it as legacy metadata without modifying the recorded file",
                RuntimeWarning,
                stacklevel=2,
            )
        differences = []
        session_fields = {item.name for item in fields(SessionMetadata)} - {"collection_date"}
        for key in sorted(session_fields):
            stored_missing = key not in existing
            incoming_missing = key not in session_metadata
            if legacy_schema and key == "schema_version" and stored_missing:
                if incoming_missing or session_metadata[key] == SESSION_METADATA_SCHEMA_VERSION:
                    continue
            if (
                legacy_schema
                and key in SESSION_METADATA_V2_OPTIONAL_PROVENANCE_FIELDS
                and stored_missing
                and (incoming_missing or session_metadata[key] is None)
            ):
                continue
            if stored_missing or incoming_missing or existing[key] != session_metadata[key]:
                differences.append(
                    f"{key}: stored={existing.get(key)!r}, "
                    f"incoming={session_metadata.get(key)!r}"
                )
        if differences:
            raise ValueError(
                "Cannot resume session with different immutable metadata; create a new session: "
                + "; ".join(differences)
            )
    else:
        atomic_write_json(session_file, session_metadata)
    profile_path = path / "channel_profile.json"
    if profile_path.exists() and read_json(profile_path) != profile.to_dict():
        raise ValueError(
            "Cannot resume session: the saved channel profile snapshot differs from the current "
            "profile definition; create a new version instead of changing v1 in place"
        )
    if not profile_path.exists():
        atomic_write_json(profile_path, profile.to_dict())
    protocol_path = path / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError(
            "Cannot resume session: the saved action protocol snapshot differs; create a new protocol version"
        )
    if not protocol_path.exists():
        atomic_write_json(protocol_path, protocol)
    (path / "mvc").mkdir(exist_ok=True)
    (path / "trials").mkdir(exist_ok=True)
    return path


def save_events(path: Path, events: list[dict[str, Any]]) -> Path:
    header = [
        "event_name",
        "sample_index",
        "emg_time_s",
        "monotonic_time_ns",
        "wall_clock_iso",
        "source",
        "confidence",
        "notes",
    ]
    rows: list[list[Any]] = [header]
    rows.extend([[event.get(key, "") for key in header] for event in events])
    return atomic_write_csv(path, rows)


def save_legacy_csv(path: Path, result: RecordResult, profile: ChannelProfile) -> Path:
    headers = ["sample_index", "time_s"] + [
        f"sensor_{channel.sensor_id}_{channel.side}_{channel.muscle_slug}_mV"
        for channel in profile.channels
    ]
    rows: list[list[Any]] = [headers]
    for idx in range(result.received_samples):
        rows.append(
            [idx, f"{idx / result.fs_hz:.9f}"]
            + [f"{float(value):.9g}" for value in result.emg_mV[idx]]
        )
    return atomic_write_csv(path, rows)


def save_trial_bundle(
    path: Path,
    result: RecordResult,
    metadata: TrialMetadata,
    events: list[dict[str, Any]],
    qc: dict[str, Any],
    profile: ChannelProfile,
    preview_writer: Any | None = None,
    legacy_csv: bool = True,
) -> Path:
    metadata.validate_shape(result.emg_mV)
    path.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        path / "raw_emg.npz",
        emg_mV=result.emg_mV.astype(np.float32, copy=False),
        time_s=result.time_s,
        sample_index=result.sample_index,
        stream_channel_ids=result.stream_channel_ids.astype(np.int16, copy=False),
        fs_hz=np.asarray(result.fs_hz, dtype=np.float64),
    )
    atomic_write_json(path / "metadata.json", metadata.to_dict())
    save_events(path / "events.csv", events)
    atomic_write_json(path / "qc.json", qc)
    if legacy_csv:
        save_legacy_csv(path / "legacy_raw_emg.csv", result, profile)
    if preview_writer is not None:
        preview_writer(path / "preview.png")
    return path


def dataset_sha256(array: np.ndarray, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=_json_default).encode("utf-8"))
    return digest.hexdigest()


def software_info() -> dict[str, Any]:
    return {"software_version": __version__, "git_commit_hash": git_commit_hash(Path.cwd())}
