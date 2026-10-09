from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .profiles import (
    BADMINTON_SYNERGY_16_V1,
    get_profile_registration,
    require_analysis_profile,
)
from .qc import save_preview
from .storage import atomic_write_json, read_json


FORMAL_V2_ID = "badminton_synergy_16_v2"
MIGRATION_SCHEMA_VERSION = 1


class UnsafeProfileMigrationError(RuntimeError):
    """Raised when the session does not satisfy migration invariants."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _profile_signature(snapshot: dict[str, Any] | None) -> tuple[tuple[int, str, str], ...]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("channels"), list):
        return ()
    return tuple(
        (int(channel["sensor_id"]), str(channel["side"]), str(channel["muscle_slug"]))
        for channel in snapshot["channels"]
    )


def _expected_csv_header(profile_id: str) -> list[str]:
    profile = require_analysis_profile(profile_id)
    return ["sample_index", "time_s"] + [
        f"sensor_{channel.sensor_id}_{channel.side}_{channel.muscle_slug}_mV"
        for channel in profile.channels
    ]


def _csv_parts(path: Path) -> tuple[list[str], bytes, bytes, bytes]:
    """Return parsed header, BOM, line ending, and byte-identical numeric body."""

    payload = path.read_bytes()
    bom = b"\xef\xbb\xbf" if payload.startswith(b"\xef\xbb\xbf") else b""
    content = payload[len(bom):]
    newline_index = content.find(b"\n")
    if newline_index < 0:
        raise ValueError(f"CSV has no complete header line: {path}")
    header_line = content[: newline_index + 1]
    line_ending = b"\r\n" if header_line.endswith(b"\r\n") else b"\n"
    header_text = header_line[: -len(line_ending)].decode("utf-8")
    header = next(csv.reader(io.StringIO(header_text)))
    body = content[newline_index + 1 :]
    return header, bom, line_ending, body


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


def _rewrite_csv_header(path: Path, expected_header: list[str]) -> dict[str, Any]:
    old_header, bom, line_ending, body = _csv_parts(path)
    old_body_hash = _bytes_sha256(body)
    new_header = ",".join(expected_header).encode("ascii") + line_ending
    _atomic_write_bytes(path, bom + new_header + body)
    written_header, _, _, written_body = _csv_parts(path)
    new_body_hash = _bytes_sha256(written_body)
    if written_header != expected_header or old_body_hash != new_body_hash:
        raise RuntimeError(f"CSV migration invariant failed at {path}")
    return {
        "old_header": old_header,
        "new_header": written_header,
        "numeric_body_sha256_before": old_body_hash,
        "numeric_body_sha256_after": new_body_hash,
        "numeric_body_unchanged": True,
    }


def _channel_differences(
    declared: tuple[tuple[int, str, str], ...],
    actual: tuple[tuple[int, str, str], ...],
) -> list[dict[str, Any]]:
    declared_by_id = {sensor_id: (side, muscle) for sensor_id, side, muscle in declared}
    differences: list[dict[str, Any]] = []
    for sensor_id, actual_side, actual_muscle in actual:
        old_side, old_muscle = declared_by_id.get(sensor_id, (None, None))
        if (old_side, old_muscle) != (actual_side, actual_muscle):
            differences.append(
                {
                    "sensor_id": sensor_id,
                    "declared_side": old_side,
                    "declared_muscle_slug": old_muscle,
                    "actual_side": actual_side,
                    "actual_muscle_slug": actual_muscle,
                }
            )
    return differences


def _audit_raw_trial(raw_path: Path, expected_channel_ids: tuple[int, ...]) -> dict[str, Any]:
    with np.load(raw_path, allow_pickle=False) as payload:
        required = {"emg_mV", "time_s", "sample_index", "stream_channel_ids", "fs_hz"}
        missing = sorted(required - set(payload.files))
        if missing:
            return {
                "path": str(raw_path),
                "sha256": _sha256(raw_path),
                "missing_arrays": missing,
                "channel_count": None,
                "sample_count": None,
                "stream_channel_ids": [],
                "sample_index_sequential": False,
                "time_length_matches": False,
            }
        emg = payload["emg_mV"]
        time_s = payload["time_s"]
        sample_index = payload["sample_index"]
        stream_ids = tuple(int(item) for item in payload["stream_channel_ids"].tolist())
        sample_count = int(emg.shape[0]) if emg.ndim == 2 else None
        sequential = (
            sample_count is not None
            and sample_index.shape == (sample_count,)
            and np.array_equal(sample_index, np.arange(sample_count, dtype=sample_index.dtype))
        )
        return {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "missing_arrays": [],
            "array_shape": list(emg.shape),
            "channel_count": int(emg.shape[1]) if emg.ndim == 2 else None,
            "sample_count": sample_count,
            "stream_channel_ids": list(stream_ids),
            "stream_channel_ids_match": stream_ids == expected_channel_ids,
            "sample_index_sequential": bool(sequential),
            "time_length_matches": bool(sample_count is not None and time_s.shape == (sample_count,)),
            "fs_hz": float(payload["fs_hz"]),
        }


def audit_session_profile(session_path: Path | str, actual_profile_id: str) -> dict[str, Any]:
    """Read-only structural and semantic audit for one acquisition session."""

    session = Path(session_path).resolve()
    profile = require_analysis_profile(actual_profile_id)
    actual_signature = _profile_signature(profile.to_dict())
    declared_v1_signature = _profile_signature(BADMINTON_SYNERGY_16_V1.to_dict())
    severe: list[str] = []
    if not session.is_dir():
        raise FileNotFoundError(f"Session path does not exist: {session}")

    trial_dirs = sorted(path for path in (session / "trials").glob("*/trial_*") if path.is_dir())
    if not trial_dirs:
        severe.append("no_trial_directories")
    action_counts = Counter(path.parent.name for path in trial_dirs)
    raw_records: list[dict[str, Any]] = []
    profile_id_counts: Counter[str] = Counter()
    snapshot_id_counts: Counter[str] = Counter()
    channel_count_counts: Counter[str] = Counter()
    stream_id_counts: Counter[str] = Counter()
    metadata_difference_counts: Counter[tuple[int, str | None, str | None, str, str]] = Counter()
    csv_difference_counts: Counter[tuple[int, str, str]] = Counter()
    csv_body_hashes: dict[str, str] = {}
    missing_files: list[str] = []
    v2_claiming_trials: list[str] = []
    target_claiming_trials: list[str] = []

    expected_header = _expected_csv_header(actual_profile_id)
    allowed_source_signatures = {declared_v1_signature, actual_signature}
    allowed_profile_ids = {BADMINTON_SYNERGY_16_V1.profile_id, actual_profile_id}
    for trial_dir in trial_dirs:
        raw_path = trial_dir / "raw_emg.npz"
        metadata_path = trial_dir / "metadata.json"
        csv_path = trial_dir / "legacy_raw_emg.csv"
        for required_path in (raw_path, metadata_path, csv_path):
            if not required_path.is_file():
                missing_files.append(str(required_path.relative_to(session)))
        if raw_path.is_file():
            record = _audit_raw_trial(raw_path, profile.channel_ids)
            record["relative_path"] = str(raw_path.relative_to(session))
            raw_records.append(record)
            channel_count_counts[str(record.get("channel_count"))] += 1
            stream_key = ",".join(str(value) for value in record.get("stream_channel_ids", []))
            stream_id_counts[stream_key] += 1
            if (
                record.get("missing_arrays")
                or record.get("channel_count") != len(profile.channels)
                or not record.get("stream_channel_ids_match")
                or not record.get("sample_index_sequential")
                or not record.get("time_length_matches")
            ):
                severe.append(f"raw_invariant_failed:{raw_path.relative_to(session)}")
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            profile_id = str(metadata.get("channel_profile_id"))
            snapshot = metadata.get("channel_profile_snapshot")
            snapshot_id = str(snapshot.get("profile_id")) if isinstance(snapshot, dict) else "None"
            profile_id_counts[profile_id] += 1
            snapshot_id_counts[snapshot_id] += 1
            if profile_id == FORMAL_V2_ID or snapshot_id == FORMAL_V2_ID:
                v2_claiming_trials.append(str(trial_dir.relative_to(session)))
            if profile_id == actual_profile_id and snapshot_id == actual_profile_id:
                target_claiming_trials.append(str(trial_dir.relative_to(session)))
            snapshot_signature = _profile_signature(snapshot)
            if profile_id not in allowed_profile_ids or snapshot_signature not in allowed_source_signatures:
                severe.append(f"unexpected_profile_declaration:{trial_dir.relative_to(session)}")
            for difference in _channel_differences(snapshot_signature, actual_signature):
                key = (
                    difference["sensor_id"],
                    difference["declared_side"],
                    difference["declared_muscle_slug"],
                    difference["actual_side"],
                    difference["actual_muscle_slug"],
                )
                metadata_difference_counts[key] += 1
        if csv_path.is_file():
            try:
                header, _, _, body = _csv_parts(csv_path)
                csv_body_hashes[str(csv_path.relative_to(session))] = _bytes_sha256(body)
                for index in range(max(len(header), len(expected_header))):
                    declared = header[index] if index < len(header) else "<missing>"
                    actual = expected_header[index] if index < len(expected_header) else "<extra>"
                    if declared != actual:
                        csv_difference_counts[(index, declared, actual)] += 1
            except (OSError, UnicodeError, ValueError) as exc:
                severe.append(f"unreadable_csv:{csv_path.relative_to(session)}:{exc}")

    if missing_files:
        severe.append("required_trial_files_missing")
    if v2_claiming_trials:
        severe.append("existing_trial_claims_badminton_synergy_16_v2")
    processed_files = sorted(session.glob("trials/*/trial_*/processed_emg.npz"))
    if processed_files:
        severe.append("processed_data_exists_and_requires_separate_semantic_revalidation")
    mvc_dir = session / "mvc"
    mvc_files = sorted(path for path in mvc_dir.rglob("*") if path.is_file()) if mvc_dir.exists() else []
    if mvc_files:
        severe.append("mvc_data_exists_and_requires_separate_profile_review")

    orphan_path = session / "channel_profile_v2.json"
    orphan: dict[str, Any] = {"exists": orphan_path.is_file(), "status": "not_present"}
    if orphan_path.is_file():
        draft = read_json(orphan_path)
        orphan.update(
            {
                "profile_id": draft.get("profile_id"),
                "version": draft.get("version"),
                "channel_signature": [list(item) for item in _profile_signature(draft)],
                "status": "referenced_existing_definition" if v2_claiming_trials else "unregistered_draft",
            }
        )

    return {
        "audit_schema_version": MIGRATION_SCHEMA_VERSION,
        "audited_at": _utc_now(),
        "session_path": str(session),
        "actual_profile_id": actual_profile_id,
        "actual_profile_version": profile.version,
        "trial_count": len(trial_dirs),
        "action_count": len(action_counts),
        "actions": dict(sorted(action_counts.items())),
        "channel_count_distribution": dict(sorted(channel_count_counts.items())),
        "stream_channel_ids_distribution": dict(sorted(stream_id_counts.items())),
        "raw_files": raw_records,
        "metadata_profile_id_distribution": dict(sorted(profile_id_counts.items())),
        "snapshot_profile_id_distribution": dict(sorted(snapshot_id_counts.items())),
        "metadata_channel_differences": [
            {
                "sensor_id": key[0],
                "declared_side": key[1],
                "declared_muscle_slug": key[2],
                "actual_side": key[3],
                "actual_muscle_slug": key[4],
                "trial_count": count,
            }
            for key, count in sorted(metadata_difference_counts.items())
        ],
        "csv_count": sum((trial_dir / "legacy_raw_emg.csv").is_file() for trial_dir in trial_dirs),
        "csv_header_differences": [
            {
                "column_index": key[0],
                "declared": key[1],
                "actual": key[2],
                "trial_count": count,
            }
            for key, count in sorted(csv_difference_counts.items())
        ],
        "csv_numeric_body_sha256": csv_body_hashes,
        "mvc_exists": bool(mvc_files),
        "mvc_file_count": len(mvc_files),
        "processed_exists": bool(processed_files),
        "processed_file_count": len(processed_files),
        "v2_claiming_trial_count": len(v2_claiming_trials),
        "v2_claiming_trials": v2_claiming_trials,
        "target_claiming_trial_count": len(target_claiming_trials),
        "missing_files": missing_files,
        "orphan_v2": orphan,
        "severe_discrepancies": sorted(set(severe)),
        "migration_safe": not severe,
    }


def format_audit_report(audit: dict[str, Any]) -> str:
    """Human-readable summary followed by no hidden interpretation."""

    lines = [
        f"Session: {audit['session_path']}",
        f"Trials: {audit['trial_count']}",
        f"Actions: {audit['action_count']} | {json.dumps(audit['actions'], ensure_ascii=False)}",
        f"Channel counts: {json.dumps(audit['channel_count_distribution'], ensure_ascii=False)}",
        f"stream_channel_ids: {json.dumps(audit['stream_channel_ids_distribution'], ensure_ascii=False)}",
        f"MVC exists: {audit['mvc_exists']} | files={audit['mvc_file_count']}",
        f"Processed exists: {audit['processed_exists']} | files={audit['processed_file_count']}",
        f"Trials claiming {FORMAL_V2_ID}: {audit['v2_claiming_trial_count']}",
        f"Metadata channel differences: {len(audit['metadata_channel_differences'])}",
        f"CSV header differences: {len(audit['csv_header_differences'])}",
        f"Orphan V2 status: {audit['orphan_v2']['status']}",
        f"Migration safe: {audit['migration_safe']}",
    ]
    if audit["severe_discrepancies"]:
        lines.append(f"Severe discrepancies: {json.dumps(audit['severe_discrepancies'], ensure_ascii=False)}")
    if audit["metadata_channel_differences"]:
        lines.append("Metadata semantic differences:")
        lines.extend(
            "  S{sensor_id}: {declared_side}:{declared_muscle_slug} -> "
            "{actual_side}:{actual_muscle_slug} ({trial_count} trial(s))".format(**item)
            for item in audit["metadata_channel_differences"]
        )
    if audit["csv_header_differences"]:
        lines.append("CSV header differences:")
        lines.extend(
            "  column {column_index}: {declared} -> {actual} ({trial_count} trial(s))".format(**item)
            for item in audit["csv_header_differences"]
        )
    return "\n".join(lines)


def _copy_backup(session: Path, backup: Path, source: Path) -> None:
    if not source.is_file():
        return
    relative = source.relative_to(session)
    destination = backup / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _backup_sources(session: Path, trial_dirs: list[Path]) -> list[Path]:
    sources = [
        session / "session.json",
        session / "channel_profile.json",
        session / "channel_profile_v2.json",
    ]
    for trial_dir in trial_dirs:
        sources.extend(
            [
                trial_dir / "metadata.json",
                trial_dir / "legacy_raw_emg.csv",
                trial_dir / "preview.png",
            ]
        )
    for action_dir in sorted(path for path in (session / "trials").glob("*") if path.is_dir()):
        sources.extend(
            [
                action_dir / "action_statistics.json",
                action_dir / "action_statistics.npz",
                action_dir / "action_mean_variance.png",
            ]
        )
    return [path for path in sources if path.is_file()]


def migrate_session_profile(
    session_path: Path | str,
    actual_profile_id: str,
    *,
    apply: bool = False,
    regenerate_previews: bool = True,
    regenerate_action_statistics: bool = True,
) -> dict[str, Any]:
    """Correct profile semantics without writing raw EMG or reordering any column."""

    session = Path(session_path).resolve()
    profile = require_analysis_profile(actual_profile_id)
    registration = get_profile_registration(actual_profile_id)
    before = audit_session_profile(session, actual_profile_id)
    if not before["migration_safe"]:
        raise UnsafeProfileMigrationError(
            "Profile migration stopped because severe discrepancies were found: "
            + ", ".join(before["severe_discrepancies"])
        )
    if (
        before["trial_count"] > 0
        and before["target_claiming_trial_count"] == before["trial_count"]
        and not before["metadata_channel_differences"]
        and not before["csv_header_differences"]
    ):
        existing = read_json(session / "profile_migration.json") if (session / "profile_migration.json").is_file() else {}
        return {"status": "already_migrated", "audit": before, "manifest": existing}
    if not apply:
        return {
            "status": "dry_run",
            "audit": before,
            "planned_target_profile": profile.to_dict(),
            "raw_npz_write_planned": False,
            "csv_numeric_body_write_planned": False,
        }

    trial_dirs = sorted(path for path in (session / "trials").glob("*/trial_*") if path.is_dir())
    migrated_at = _utc_now()
    migration_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = session / "profile_migration_backups" / migration_id
    backup.mkdir(parents=True, exist_ok=False)
    backup_sources = _backup_sources(session, trial_dirs)
    for source in backup_sources:
        _copy_backup(session, backup, source)
    atomic_write_json(backup / "audit_before.json", before)

    raw_hashes_before = {
        record["relative_path"]: record["sha256"]
        for record in before["raw_files"]
    }
    csv_body_hashes_before = dict(before["csv_numeric_body_sha256"])
    migration_note = {
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_id": migration_id,
        "migrated_at": migrated_at,
        "source_profile_id": BADMINTON_SYNERGY_16_V1.profile_id,
        "actual_profile_id": profile.profile_id,
        "actual_profile_version": profile.version,
        "method": "semantic_metadata_and_csv_header_correction_only",
        "raw_emg_values_modified": False,
        "sample_order_modified": False,
        "sensor_columns_reordered": False,
    }

    session_metadata_path = session / "session.json"
    session_metadata = read_json(session_metadata_path)
    source_session_profile = session_metadata.get("channel_profile_id")
    session_metadata["channel_profile_id"] = profile.profile_id
    session_metadata["channel_profile_version"] = profile.version
    session_metadata["profile_migration"] = {**migration_note, "source_profile_id": source_session_profile}
    atomic_write_json(session_metadata_path, session_metadata)
    atomic_write_json(session / "channel_profile.json", profile.to_dict())

    csv_records: dict[str, dict[str, Any]] = {}
    for trial_dir in trial_dirs:
        metadata_path = trial_dir / "metadata.json"
        metadata = read_json(metadata_path)
        source_profile_id = metadata.get("channel_profile_id")
        source_profile_version = metadata.get("channel_profile_version")
        metadata["channel_profile_id"] = profile.profile_id
        metadata["channel_profile_version"] = profile.version
        metadata["channel_profile_snapshot"] = profile.to_dict()
        metadata["profile_migration"] = {
            **migration_note,
            "source_profile_id": source_profile_id,
            "source_profile_version": source_profile_version,
            "raw_emg_sha256": raw_hashes_before[str((trial_dir / "raw_emg.npz").relative_to(session))],
        }
        atomic_write_json(metadata_path, metadata)
        csv_path = trial_dir / "legacy_raw_emg.csv"
        csv_records[str(csv_path.relative_to(session))] = _rewrite_csv_header(
            csv_path,
            _expected_csv_header(profile.profile_id),
        )

    draft_path = session / "channel_profile_v2.json"
    draft_status: dict[str, Any] | None = None
    if draft_path.is_file():
        draft_status = {
            "status": "unregistered_draft",
            "registered": False,
            "source_file": draft_path.name,
            "recorded_at": migrated_at,
            "reason": (
                "No trial referenced this file or badminton_synergy_16_v2. "
                "The formal registry definition is authoritative and this draft must not be used for analysis."
            ),
        }
        atomic_write_json(session / "channel_profile_v2.registration.json", draft_status)

    derived_results: dict[str, Any] = {
        "previews_regenerated": 0,
        "action_statistics_regenerated": 0,
        "warnings": [],
    }
    if regenerate_previews:
        for trial_dir in trial_dirs:
            metadata = read_json(trial_dir / "metadata.json")
            with np.load(trial_dir / "raw_emg.npz", allow_pickle=False) as payload:
                emg = np.asarray(payload["emg_mV"])
                fs_hz = float(payload["fs_hz"])
            save_preview(
                trial_dir / "preview.png",
                emg,
                fs_hz,
                profile,
                f"{metadata['trial_id']} | corrected historical channel semantics",
                show=False,
            )
            derived_results["previews_regenerated"] += 1
    if regenerate_action_statistics:
        try:
            from .action_statistics import summarize_session_actions

            outputs = summarize_session_actions(session, profile.profile_id, show=False)
            derived_results["action_statistics_regenerated"] = len(outputs)
        except Exception as exc:  # migration remains valid; derived artifact failure is explicit
            derived_results["warnings"].append(
                f"Action statistics were not fully regenerated: {type(exc).__name__}: {exc}"
            )

    raw_hashes_after = {
        str(path.relative_to(session)): _sha256(path)
        for path in sorted(session.glob("trials/*/trial_*/raw_emg.npz"))
    }
    raw_hashes_unchanged = raw_hashes_before == raw_hashes_after
    csv_body_hashes_after = {
        str(path.relative_to(session)): _bytes_sha256(_csv_parts(path)[3])
        for path in sorted(session.glob("trials/*/trial_*/legacy_raw_emg.csv"))
    }
    csv_numeric_bodies_unchanged = csv_body_hashes_before == csv_body_hashes_after
    if not raw_hashes_unchanged or not csv_numeric_bodies_unchanged:
        raise RuntimeError(
            "Critical migration invariant failed: raw NPZ or CSV numeric data changed unexpectedly"
        )

    after = audit_session_profile(session, actual_profile_id)
    if after["metadata_channel_differences"] or after["csv_header_differences"]:
        raise RuntimeError("Semantic migration verification failed: metadata or CSV headers still differ")
    manifest = {
        **migration_note,
        "status": "completed",
        "profile_registration": registration.to_dict(),
        "session_path": str(session),
        "backup_path": str(backup),
        "backup_file_count": len(backup_sources),
        "trial_count": before["trial_count"],
        "action_count": before["action_count"],
        "orphan_v2": draft_status,
        "raw_npz_sha256_before": raw_hashes_before,
        "raw_npz_sha256_after": raw_hashes_after,
        "raw_npz_files_unchanged": raw_hashes_unchanged,
        "csv_numeric_body_sha256_before": csv_body_hashes_before,
        "csv_numeric_body_sha256_after": csv_body_hashes_after,
        "csv_numeric_bodies_unchanged": csv_numeric_bodies_unchanged,
        "csv_migration_records": csv_records,
        "derived_artifacts": derived_results,
        "audit_before_summary": {
            "metadata_channel_differences": before["metadata_channel_differences"],
            "csv_header_differences": before["csv_header_differences"],
        },
        "audit_after_summary": {
            "metadata_channel_differences": after["metadata_channel_differences"],
            "csv_header_differences": after["csv_header_differences"],
            "migration_safe": after["migration_safe"],
        },
        "software_version": __version__,
    }
    atomic_write_json(session / "profile_migration.json", manifest)
    atomic_write_json(backup / "migration_manifest.json", manifest)
    return {"status": "completed", "audit_before": before, "audit_after": after, "manifest": manifest}
