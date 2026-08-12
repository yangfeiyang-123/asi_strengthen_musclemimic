import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from emg.cli import build_parser
from emg.profile_migration import (
    UnsafeProfileMigrationError,
    audit_session_profile,
    migrate_session_profile,
)
from emg.profiles import (
    BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1,
    BADMINTON_SYNERGY_16_V1,
    BADMINTON_SYNERGY_16_V2,
    PROFILE_REGISTRY,
    ProfilePermissionError,
    require_mvc_profile,
)
from emg.storage import atomic_save_npz, atomic_write_csv, atomic_write_json, read_json


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_body(path: Path) -> bytes:
    return path.read_bytes().split(b"\n", 1)[1]


def _make_misdeclared_session(root: Path, *, trials: int = 2) -> Path:
    session = root / "P001" / "PILOT01"
    session.mkdir(parents=True)
    atomic_write_json(
        session / "session.json",
        {
            "participant_id": "P001",
            "session_id": "PILOT01",
            "channel_profile_id": BADMINTON_SYNERGY_16_V1.profile_id,
            "protocol_id": "badminton_primitive_protocol_v1",
            "handedness": "right",
        },
    )
    atomic_write_json(session / "channel_profile.json", BADMINTON_SYNERGY_16_V1.to_dict())
    atomic_write_json(
        session / "channel_profile_v2.json",
        {
            "profile_id": "badminton_synergy_16_v2",
            "version": 2,
            "channels": [
                {
                    "sensor_id": channel.sensor_id,
                    "side": channel.side,
                    "muscle_slug": channel.muscle_slug,
                }
                for channel in BADMINTON_SYNERGY_16_V1.channels
            ],
        },
    )
    fs_hz = 2000.0
    sample_count = 100
    for index in range(1, trials + 1):
        trial = session / "trials" / "quiet_stance" / f"trial_{index:03d}"
        trial.mkdir(parents=True)
        emg = np.arange(sample_count * 16, dtype=np.float32).reshape(sample_count, 16) + index
        atomic_save_npz(
            trial / "raw_emg.npz",
            emg_mV=emg,
            time_s=np.arange(sample_count) / fs_hz,
            sample_index=np.arange(sample_count, dtype=np.int64),
            stream_channel_ids=np.arange(1, 17, dtype=np.int16),
            fs_hz=np.asarray(fs_hz),
        )
        atomic_write_json(
            trial / "metadata.json",
            {
                "participant_id": "P001",
                "session_id": "PILOT01",
                "trial_id": f"quiet_stance_trial_{index:03d}",
                "trial_index": index,
                "action_id": "quiet_stance",
                "protocol_id": "badminton_primitive_protocol_v1",
                "channel_profile_id": BADMINTON_SYNERGY_16_V1.profile_id,
                "channel_profile_version": 1,
                "channel_profile_snapshot": BADMINTON_SYNERGY_16_V1.to_dict(),
                "valid_for_analysis": True,
            },
        )
        header = ["sample_index", "time_s"] + [
            f"sensor_{channel.sensor_id}_{channel.side}_{channel.muscle_slug}_mV"
            for channel in BADMINTON_SYNERGY_16_V1.channels
        ]
        rows = [header]
        rows.extend(
            [sample, f"{sample / fs_hz:.9f}"] + [f"{value:.9g}" for value in emg[sample]]
            for sample in range(sample_count)
        )
        atomic_write_csv(trial / "legacy_raw_emg.csv", rows)
    return session


def test_profile_registry_contains_exact_v2_and_historical_policies():
    v1 = PROFILE_REGISTRY[BADMINTON_SYNERGY_16_V1.profile_id]
    historical = PROFILE_REGISTRY[BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id]
    v2 = PROFILE_REGISTRY[BADMINTON_SYNERGY_16_V2.profile_id]
    assert (v1.status, v1.collection_allowed, v1.analysis_allowed) == (
        "deprecated_misdeclared", False, False
    )
    assert v1.reason == "The declared channel map does not match the physical placement used in PILOT01."
    assert (historical.status, historical.collection_allowed, historical.analysis_allowed) == (
        "legacy_actual", False, True
    )
    assert historical.retrospective_mvc_allowed == "explicit_only"
    assert (v2.status, v2.collection_allowed, v2.analysis_allowed) == ("active", True, True)
    assert "gluteus_maximus" not in [channel.muscle_slug for channel in v2.profile.channels]
    assert [(channel.sensor_id, channel.side, channel.muscle_slug) for channel in v2.profile.channels][6:14] == [
        (7, "right", "pronator_teres"),
        (8, "right", "extensor_carpi_radialis"),
        (9, "right", "external_oblique"),
        (10, "left", "external_oblique"),
        (11, "right", "vastus_lateralis"),
        (12, "left", "vastus_lateralis"),
        (13, "right", "biceps_femoris_long_head"),
        (14, "left", "biceps_femoris_long_head"),
    ]


def test_cli_default_is_formal_v2():
    args = build_parser().parse_args(
        ["collect", "--participant", "P001", "--session", "S001", "--action", "quiet_stance"]
    )
    assert args.profile == BADMINTON_SYNERGY_16_V2.profile_id


def test_historical_mvc_requires_explicit_confirmation():
    with pytest.raises(ProfilePermissionError, match="explicit retrospective MVC confirmation"):
        require_mvc_profile(BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id)
    assert require_mvc_profile(
        BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id,
        explicit_retrospective=True,
    ) is BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1


def test_audit_and_migration_preserve_raw_and_csv_numeric_payload(tmp_path):
    session = _make_misdeclared_session(tmp_path)
    raw_paths = sorted(session.glob("trials/*/trial_*/raw_emg.npz"))
    csv_paths = sorted(session.glob("trials/*/trial_*/legacy_raw_emg.csv"))
    raw_before = {path: _digest(path) for path in raw_paths}
    csv_body_before = {path: _csv_body(path) for path in csv_paths}
    arrays_before = {}
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as payload:
            arrays_before[path] = {
                key: np.array(payload[key], copy=True)
                for key in payload.files
            }

    audit = audit_session_profile(session, BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id)
    assert audit["trial_count"] == 2
    assert audit["channel_count_distribution"] == {"16": 2}
    assert audit["stream_channel_ids_distribution"] == {",".join(map(str, range(1, 17))): 2}
    assert audit["mvc_exists"] is False
    assert audit["processed_exists"] is False
    assert audit["v2_claiming_trial_count"] == 0
    assert audit["orphan_v2"]["status"] == "unregistered_draft"
    assert [item["sensor_id"] for item in audit["metadata_channel_differences"]] == list(range(7, 15))
    assert audit["migration_safe"] is True

    dry_run = migrate_session_profile(
        session,
        BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id,
        apply=False,
        regenerate_previews=False,
        regenerate_action_statistics=False,
    )
    assert dry_run["status"] == "dry_run"
    assert not (session / "profile_migration.json").exists()

    result = migrate_session_profile(
        session,
        BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id,
        apply=True,
        regenerate_previews=False,
        regenerate_action_statistics=False,
    )
    assert result["status"] == "completed"
    assert result["manifest"]["raw_npz_files_unchanged"] is True
    assert result["manifest"]["csv_numeric_bodies_unchanged"] is True
    assert read_json(session / "session.json")["channel_profile_id"] == BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id
    assert read_json(session / "channel_profile.json") == BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.to_dict()
    assert read_json(session / "channel_profile_v2.registration.json")["status"] == "unregistered_draft"
    assert Path(result["manifest"]["backup_path"]).is_dir()

    expected_header = ["sample_index", "time_s"] + [
        f"sensor_{channel.sensor_id}_{channel.side}_{channel.muscle_slug}_mV"
        for channel in BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.channels
    ]
    for path in raw_paths:
        assert _digest(path) == raw_before[path]
        with np.load(path, allow_pickle=False) as payload:
            for key, expected in arrays_before[path].items():
                np.testing.assert_array_equal(payload[key], expected)
    for path in csv_paths:
        assert _csv_body(path) == csv_body_before[path]
        with path.open("r", encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle)) == expected_header
    for metadata_path in session.glob("trials/*/trial_*/metadata.json"):
        metadata = read_json(metadata_path)
        assert metadata["channel_profile_id"] == BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id
        assert metadata["profile_migration"]["sensor_columns_reordered"] is False

    second = migrate_session_profile(
        session,
        BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id,
        apply=True,
        regenerate_previews=False,
        regenerate_action_statistics=False,
    )
    assert second["status"] == "already_migrated"


def test_processed_data_stops_migration_before_writes(tmp_path):
    session = _make_misdeclared_session(tmp_path, trials=1)
    trial = next(session.glob("trials/*/trial_*"))
    atomic_save_npz(trial / "processed_emg.npz", normalized_envelope=np.ones((10, 16)))
    session_before = (session / "session.json").read_bytes()
    with pytest.raises(UnsafeProfileMigrationError, match="processed_data_exists"):
        migrate_session_profile(
            session,
            BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id,
            apply=True,
            regenerate_previews=False,
            regenerate_action_statistics=False,
        )
    assert (session / "session.json").read_bytes() == session_before
    assert not (session / "profile_migration_backups").exists()
