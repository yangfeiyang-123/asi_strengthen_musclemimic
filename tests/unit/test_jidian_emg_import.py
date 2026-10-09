from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jidian_measurement.emg.event_annotation import annotate_event
from musclemimic.evaluation.jidian_emg_import import (
    AUDIT_SCHEMA_VERSION,
    EMG_SIGNAL_KIND,
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_CHANNELS,
    EXPECTED_MUSCLE_SLUGS,
    EXPECTED_PROFILE_ID,
    EXPECTED_SIDES,
    IMPORT_SCHEMA_VERSION,
    PAIRED_COMPARISON_DESIGN,
    SELECTION_SCHEMA_VERSION,
    UNPAIRED_COMPARISON_DESIGN,
    JidianEmgImportError,
    import_jidian_emg,
)

FS_HZ = 100.0
ACTION_ID = "forehand_smash"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _profile() -> dict[str, Any]:
    return {
        "profile_id": EXPECTED_PROFILE_ID,
        "version": 2,
        "description": "synthetic exact profile",
        "intended_handedness": "right",
        "channels": [
            {
                "sensor_id": sensor_id,
                "side": side,
                "muscle_slug": slug,
                "name_en": slug,
            }
            for sensor_id, side, slug in EXPECTED_CHANNELS
        ],
    }


def _processing_config() -> dict[str, Any]:
    return {
        "sample_rate_hz": FS_HZ,
        "bandpass_low_hz": 20.0,
        "bandpass_high_hz": 40.0,
        "filter_order": 4,
        "envelope_lowpass_hz": 4.0,
        "normalization": "mvc",
    }


def _write_events(
    path: Path,
    *,
    sample: int | str,
    event_time: float | str,
    source: str = "manual_video",
    confidence: float | str = 0.95,
    duplicate: bool = False,
) -> dict[str, str]:
    header = "event_name,sample_index,emg_time_s,monotonic_time_ns,wall_clock_iso,source,confidence,notes\n"
    row = f"racket_contact,{sample},{event_time},1,2026-01-01T00:00:00Z,{source},{confidence},annotated\n"
    path.write_text(header + row + (row if duplicate else ""), encoding="utf-8")
    return {
        "event_name": "racket_contact",
        "sample_index": str(sample),
        "emg_time_s": str(event_time),
        "monotonic_time_ns": "1",
        "wall_clock_iso": "2026-01-01T00:00:00Z",
        "source": str(source),
        "confidence": str(confidence),
        "notes": "annotated",
    }


def _write_annotation_audit(path: Path, events_path: Path, after_event: dict[str, str]) -> None:
    after_sha256 = hashlib.sha256(events_path.read_bytes()).hexdigest()
    base = {
        "audit_schema_version": "emg_event_annotation_audit_v1",
        "annotation_id": "synthetic-annotation-1",
        "annotated_at": "2026-01-01T00:00:00Z",
        "event_name": "racket_contact",
        "annotator": "unit-test-annotator",
        "evidence_reference": "synthetic://high-speed-video/frame-10",
        "evidence_sha256": hashlib.sha256(b"synthetic video evidence").hexdigest(),
        "overwrite": False,
        "events_path": "events.csv",
        "before_sha256": "0" * 64,
        "after_sha256": after_sha256,
        "before_event": None,
        "after_event": after_event,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    records = [
        {**base, "annotation_manifest_sha256": manifest_sha256, "transaction_state": state}
        for state in ("prepared", "committed")
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _raw_qc(length: int) -> dict[str, Any]:
    channels = []
    for index in range(16):
        warnings = ["low_action_to_baseline_ratio"] if index == 0 else []
        channels.append(
            {
                "column_index": index,
                "all_zero": False,
                "flatline": False,
                "finite": True,
                "warnings": warnings,
            }
        )
    return {
        # The real P002 files are commonly false because non-fatal manual-review
        # warnings exist; the importer must not turn that into a fatal gate.
        "qc_pass": False,
        "manual_review_required": True,
        "received_samples": length,
        "expected_samples": length,
        "sample_count_ok": True,
        "short_stream_or_dropout_samples": 0,
        "channels": channels,
        "warnings": ["channel_1:low_action_to_baseline_ratio"],
        "thresholds": {"flatline_std_mV": 1e-5},
    }


def _preprocessing_qc() -> dict[str, Any]:
    channels = []
    for index, (sensor_id, side, slug) in enumerate(EXPECTED_CHANNELS):
        channels.append(
            {
                "channel_index": index,
                "sensor_id": sensor_id,
                "side": side,
                "muscle_slug": slug,
                "analysis_ready": True,
                "critical_failures": [],
                "warnings": ["normalized_envelope_exceeds_200pct_mvc"] if index == 1 else [],
            }
        )
    return {
        "qc_pass": True,
        "analysis_ready": True,
        "critical_channel_count": 0,
        "channel_count": 16,
        "channels": channels,
    }


def _make_fixture(
    root: Path,
    *,
    trial_specs: tuple[tuple[int, int, int], ...] = ((1, 300, 150), (2, 350, 200)),
    comparison_design: str = UNPAIRED_COMPARISON_DESIGN,
) -> dict[str, Any]:
    session = root / "data" / "P900" / "SYNTH_A"
    session.mkdir(parents=True)
    profile = _profile()
    config = _processing_config()
    mvc_values = [0.1 + index * 0.01 for index in range(16)]
    _write_json(session / "channel_profile.json", profile)
    _write_json(
        session / "session.json",
        {
            "participant_id": "P900",
            "session_id": "SYNTH_A",
            "channel_profile_id": EXPECTED_PROFILE_ID,
            "handedness": "right",
        },
    )
    mvc_channels = []
    for (sensor_id, side, slug), peak in zip(EXPECTED_CHANNELS, mvc_values, strict=True):
        source = session / "mvc" / f"{side}_{slug}" / "rep_001" / "mvc_timeseries.npz"
        source.parent.mkdir(parents=True)
        np.savez(source, envelope=np.asarray([peak], dtype=np.float32))
        mvc_channels.append(
            {
                "sensor_id": sensor_id,
                "side": side,
                "muscle_slug": slug,
                "selected_peak_mV": peak,
                "valid_repetitions": [
                    {
                        "path": source.relative_to(session).as_posix(),
                        "session_id": "SYNTH_A",
                        "envelope_peak_mV": peak,
                    }
                ],
            }
        )
    _write_json(
        session / "preprocessing_mvc_reference.json",
        {
            "scope": "participant",
            "participant_id": "P900",
            "processing_config": config,
            "channels": mvc_channels,
            "missing_sensor_ids": [],
        },
    )

    source_envelopes: dict[str, np.ndarray] = {}
    trial_ids: list[str] = []
    for trial_index, length, impact in trial_specs:
        trial_id = f"{ACTION_ID}_trial_{trial_index:03d}"
        trial_ids.append(trial_id)
        trial = session / "trials" / ACTION_ID / f"trial_{trial_index:03d}"
        trial.mkdir(parents=True)
        metadata = {
            "participant_id": "P900",
            "session_id": "SYNTH_A",
            "trial_id": trial_id,
            "trial_index": trial_index,
            "action_id": ACTION_ID,
            "channel_profile_id": EXPECTED_PROFILE_ID,
            "channel_profile_version": 2,
            "channel_profile_snapshot": profile,
            "handedness": "right",
            "sample_rate_hz": FS_HZ,
            "expected_samples": length,
            "received_samples": length,
            "dropped_samples": 0,
            "interrupted": False,
            "receive_error": None,
            "valid_for_analysis": True,
        }
        _write_json(trial / "metadata.json", metadata)
        _write_json(trial / "qc.json", _raw_qc(length))
        _write_json(trial / "preprocessing_qc.json", _preprocessing_qc())
        _write_json(
            trial / "processing.json",
            {
                "processing_format_version": 2,
                "status": "completed",
                "processing_config": config,
                "normalization_method": "mvc",
                "fallback_method": None,
                "normalization_reference_scope": "participant_raw_mvc_across_sessions",
                "mvc_reference_manifest": str(session / "preprocessing_mvc_reference.json"),
                "normalization_reference_values_mV": mvc_values,
                "channel_profile_id": EXPECTED_PROFILE_ID,
                "channel_profile_snapshot": profile,
                "raw_source": "raw_emg.npz",
                "quality_summary": {
                    "qc_pass": True,
                    "analysis_ready": True,
                    "critical_channel_count": 0,
                },
            },
        )
        time_s = np.arange(length, dtype=np.float64) / FS_HZ
        raw_emg = (
            np.sin(np.arange(length, dtype=np.float64)[:, None] / 10.0)
            + np.arange(16, dtype=np.float64)[None, :] / 10.0
        )
        np.savez(
            trial / "raw_emg.npz",
            emg_mV=raw_emg.astype(np.float32),
            time_s=time_s,
            sample_index=np.arange(length, dtype=np.int64),
            stream_channel_ids=np.arange(1, 17, dtype=np.int16),
            fs_hz=np.asarray(FS_HZ),
        )
        envelope = (
            trial_index
            + np.arange(length, dtype=np.float64)[:, None] / 1000.0
            + np.arange(16, dtype=np.float64)[None, :] / 100.0
        )
        source_envelopes[trial_id] = envelope
        np.savez(
            trial / "processed_emg.npz",
            normalized_envelope=envelope.astype(np.float32),
            time_s=time_s,
            sample_index=np.arange(length, dtype=np.int64),
            fs_hz=np.asarray(FS_HZ),
            stream_channel_ids=np.arange(1, 17, dtype=np.int16),
            channel_names=np.asarray(EXPECTED_CHANNEL_NAMES),
            muscle_slugs=np.asarray(EXPECTED_MUSCLE_SLUGS),
            sides=np.asarray(EXPECTED_SIDES),
            participant_id=np.asarray("P900"),
            session_id=np.asarray("SYNTH_A"),
            trial_id=np.asarray(trial_id),
            trial_index=np.asarray(trial_index, dtype=np.int64),
            action_id=np.asarray(ACTION_ID),
        )
        after_event = _write_events(
            trial / "events.csv",
            sample=impact,
            event_time=impact / FS_HZ,
        )
        _write_annotation_audit(
            trial / "events.annotation.audit.jsonl",
            trial / "events.csv",
            after_event,
        )

    comparison_set: dict[str, Any] = {"comparison_set_uid": "synthetic-cohort-v1"}
    if comparison_design == PAIRED_COMPARISON_DESIGN:
        comparison_set["reference_trial_fingerprints"] = {
            trial_id: hashlib.sha256(f"reference:{trial_id}".encode()).hexdigest() for trial_id in trial_ids
        }
    selection = root / "selection.json"
    _write_json(
        selection,
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "session_path": str(session),
            "subject_uid": "subject-registry-900",
            "session_uid": "session-registry-synth-a",
            "action_id": ACTION_ID,
            "trial_ids": trial_ids,
            "dataset_split": "heldout",
            "comparison_design": comparison_design,
            "comparison_set": comparison_set,
            "training_session_uids": ["session-registry-training-a"],
            "alignment": {
                "mode": "impact",
                "event_name": "racket_contact",
                "minimum_confidence": 0.8,
                "pre_event_s": 0.2,
                "post_event_s": 0.3,
            },
        },
    )
    return {
        "session": session,
        "selection": selection,
        "trial_ids": trial_ids,
        "source_envelopes": source_envelopes,
        "impacts": {trial_id: spec[2] for trial_id, spec in zip(trial_ids, trial_specs, strict=True)},
    }


def _trial_path(fixture: dict[str, Any], trial_id: str) -> Path:
    index = int(trial_id.rsplit("_", maxsplit=1)[1])
    return fixture["session"] / "trials" / ACTION_ID / f"trial_{index:03d}"


def _reason_codes(error: JidianEmgImportError) -> set[str]:
    return {code for trial in error.report.get("trials", []) for code in trial.get("reason_codes", [])} | set(
        error.report.get("global_reason_codes", [])
    )


def test_unpaired_import_keeps_all_16_preprocessed_channels_and_provenance(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    # This unselected invalid directory proves the importer does not discover or
    # merge arbitrary trial folders.
    extra = fixture["session"] / "trials" / ACTION_ID / "trial_999"
    extra.mkdir(parents=True)
    (extra / "processed_emg.npz").write_bytes(b"not an npz")
    output = tmp_path / "imported.npz"

    report = import_jidian_emg(fixture["selection"], output)

    assert report["status"] == "exported"
    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["summary"] == {
        "selected_trial_count": 2,
        "eligible_trial_count": 2,
        "included_trial_count": 2,
        "rejected_trial_count": 0,
    }
    with np.load(output, allow_pickle=False) as data:
        assert str(data["import_schema_version"]) == IMPORT_SCHEMA_VERSION
        assert str(data["emg_signal_kind"]) == EMG_SIGNAL_KIND
        assert str(data["comparison_design"]) == UNPAIRED_COMPARISON_DESIGN
        assert str(data["comparison_set_uid"]) == "synthetic-cohort-v1"
        assert "reference_trial_fingerprint" not in data.files
        assert data["emg"].shape == (2, 51, 16)
        assert data["channel_names"].astype(str).tolist() == list(EXPECTED_CHANNEL_NAMES)
        assert data["stream_channel_ids"].astype(int).tolist() == list(range(1, 17))
        assert int(data["acquired_channel_count"]) == 16
        assert int(data["comparable_channel_count"]) == 15
        assert data["excluded_sensor_ids"].astype(int).tolist() == [1]
        for row, trial_id in enumerate(fixture["trial_ids"]):
            impact = fixture["impacts"][trial_id]
            expected = fixture["source_envelopes"][trial_id][impact - 20 : impact + 31]
            np.testing.assert_allclose(data["emg"][row], expected.astype(np.float32), rtol=0, atol=0)
        first_uid = str(data["trial_uid"][0])
    assert first_uid.startswith("jidian-emg-")
    assert len(first_uid) == len("jidian-emg-") + 64
    assert (tmp_path / "imported.audit.json").is_file()
    assert report["trials"][0]["qc_warning_count"] >= 2
    processing = report["provenance"]["processing_binding"]
    assert set(processing["trials"][fixture["trial_ids"][0]]) >= {
        "processing_json_sha256",
        "raw_emg_npz_sha256",
        "processed_emg_npz_sha256",
        "raw_qc_json_sha256",
        "preprocessing_qc_json_sha256",
        "events_csv_sha256",
        "events_annotation_audit_jsonl_sha256",
    }


def test_paired_mode_requires_real_explicit_fingerprints_and_uid_is_design_stable(tmp_path: Path) -> None:
    unpaired = _make_fixture(tmp_path / "unpaired", trial_specs=((1, 300, 150),))
    first_output = tmp_path / "unpaired.npz"
    import_jidian_emg(unpaired["selection"], first_output)
    with np.load(first_output, allow_pickle=False) as data:
        unpaired_uid = str(data["trial_uid"][0])

    selection = json.loads(unpaired["selection"].read_text(encoding="utf-8"))
    trial_id = selection["trial_ids"][0]
    selection["comparison_design"] = PAIRED_COMPARISON_DESIGN
    expected_reference = hashlib.sha256(b"real-explicit-reference-manifest").hexdigest()
    selection["comparison_set"]["reference_trial_fingerprints"] = {trial_id: expected_reference}
    _write_json(unpaired["selection"], selection)

    # Independently collected Jidian EMG has no shared reference-trial fingerprint,
    # so paired design must be rejected at parse time unless explicit external
    # cross-fingerprint provenance is attested.
    provenance_missing_output = tmp_path / "paired_without_provenance.npz"
    with pytest.raises(JidianEmgImportError) as missing_provenance:
        import_jidian_emg(unpaired["selection"], provenance_missing_output)
    assert "selection_or_session_contract_invalid" in _reason_codes(missing_provenance.value)
    assert not provenance_missing_output.exists()

    selection["comparison_set"]["external_reference_provenance"] = {
        "attestation": "externally_verified_shared_reference_trial",
        "evidence_reference": "external_paired_reference_manifest_2026-07-29",
    }
    _write_json(unpaired["selection"], selection)
    paired_output = tmp_path / "paired.npz"
    import_jidian_emg(unpaired["selection"], paired_output)
    with np.load(paired_output, allow_pickle=False) as data:
        assert str(data["comparison_design"]) == PAIRED_COMPARISON_DESIGN
        assert data["reference_trial_fingerprint"].astype(str).tolist() == [expected_reference]
        assert str(data["trial_uid"][0]) == unpaired_uid

    selection["comparison_design"] = UNPAIRED_COMPARISON_DESIGN
    _write_json(unpaired["selection"], selection)
    rejected_output = tmp_path / "must_not_exist.npz"
    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(unpaired["selection"], rejected_output)
    assert "selection_or_session_contract_invalid" in _reason_codes(caught.value)
    assert not rejected_output.exists()


@pytest.mark.parametrize(
    ("event_change", "expected_code"),
    [
        ({"sample": "", "event_time": "", "source": "unannotated", "confidence": 0.0}, "impact_event_incomplete"),
        ({"sample": 150, "event_time": 1.5, "duplicate": True}, "impact_event_not_unique"),
        ({"sample": 150, "event_time": 1.5, "confidence": 0.2}, "impact_event_low_confidence"),
        ({"sample": 150, "event_time": 2.0}, "impact_event_time_mismatch"),
        ({"sample": 5, "event_time": 0.05}, "impact_window_incomplete"),
        ({"sample": 150, "event_time": 1.5, "source": "software_audio_visual"}, "impact_event_unannotated"),
    ],
)
def test_impact_event_contract_is_fail_closed(
    tmp_path: Path,
    event_change: dict[str, Any],
    expected_code: str,
) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    trial_id = fixture["trial_ids"][0]
    defaults: dict[str, Any] = {
        "sample": 150,
        "event_time": 1.5,
        "source": "manual_video",
        "confidence": 0.95,
        "duplicate": False,
    }
    defaults.update(event_change)
    _write_events(_trial_path(fixture, trial_id) / "events.csv", **defaults)
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert expected_code in _reason_codes(caught.value)
    assert caught.value.report["summary"]["included_trial_count"] == 0
    assert caught.value.audit_path.is_file()
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_code"), [("all_zero", "raw_qc_fatal_channel"), ("samples", "raw_qc_sample_mismatch")]
)
def test_raw_qc_fatal_conditions_are_rejected(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    qc_path = _trial_path(fixture, fixture["trial_ids"][0]) / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if mutation == "all_zero":
        qc["channels"][3]["all_zero"] = True
        qc["channels"][3]["warnings"].append("all_zero")
    else:
        qc["sample_count_ok"] = False
        qc["short_stream_or_dropout_samples"] = 1
        qc["warnings"].append("sample_count_mismatch_or_short_stream")
    _write_json(qc_path, qc)
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert expected_code in _reason_codes(caught.value)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_audit", "required_file_missing"),
        ("events_changed_after_commit", "event_annotation_after_hash_mismatch"),
    ],
)
def test_event_annotation_transaction_is_required_and_content_bound(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    trial = _trial_path(fixture, fixture["trial_ids"][0])
    if mutation == "missing_audit":
        (trial / "events.annotation.audit.jsonl").unlink()
    else:
        _write_events(
            trial / "events.csv",
            sample=150,
            event_time=1.5,
            confidence=0.9,
        )
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert expected_code in _reason_codes(caught.value)
    assert not output.exists()


def test_later_annotation_of_another_event_preserves_target_impact_evidence(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    trial = _trial_path(fixture, fixture["trial_ids"][0])
    events_path = trial / "events.csv"
    audit_path = trial / "events.annotation.audit.jsonl"
    before_sha256 = hashlib.sha256(events_path.read_bytes()).hexdigest()
    landing_event = {
        "event_name": "landing",
        "sample_index": "250",
        "emg_time_s": "2.5",
        "monotonic_time_ns": "2",
        "wall_clock_iso": "2026-01-01T00:00:01Z",
        "source": "manual_video",
        "confidence": "0.9",
        "notes": "later annotation",
    }
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(
            ",".join(
                landing_event[name]
                for name in (
                    "event_name",
                    "sample_index",
                    "emg_time_s",
                    "monotonic_time_ns",
                    "wall_clock_iso",
                    "source",
                    "confidence",
                    "notes",
                )
            )
            + "\n"
        )
    after_sha256 = hashlib.sha256(events_path.read_bytes()).hexdigest()
    base = {
        "audit_schema_version": "emg_event_annotation_audit_v1",
        "annotation_id": "synthetic-annotation-landing",
        "annotated_at": "2026-01-01T00:00:01Z",
        "event_name": "landing",
        "annotator": "unit-test-annotator",
        "evidence_reference": "synthetic://high-speed-video/frame-20",
        "evidence_sha256": hashlib.sha256(b"synthetic landing evidence").hexdigest(),
        "overwrite": False,
        "events_path": "events.csv",
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "before_event": None,
        "after_event": landing_event,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with audit_path.open("a", encoding="utf-8") as stream:
        for state in ("prepared", "committed"):
            stream.write(
                json.dumps(
                    {
                        **base,
                        "annotation_manifest_sha256": manifest_sha256,
                        "transaction_state": state,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    output = tmp_path / "accepted.npz"
    report = import_jidian_emg(fixture["selection"], output)

    assert report["status"] == "exported"
    annotation = report["trials"][0]["event"]["annotation_audit"]
    assert annotation["event_name"] == "racket_contact"
    assert annotation["latest_global_event_name"] == "landing"
    assert annotation["latest_global_after_sha256"] == after_sha256


def test_jidian_annotation_writer_output_imports_without_schema_translation(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    trial = _trial_path(fixture, fixture["trial_ids"][0])
    _write_events(
        trial / "events.csv",
        sample="",
        event_time="",
        source="unannotated",
        confidence=0.0,
    )
    (trial / "events.annotation.audit.jsonl").unlink()
    evidence_sha256 = hashlib.sha256(b"actual Jidian writer evidence").hexdigest()
    result = annotate_event(
        trial,
        "racket_contact",
        sample_index=150,
        emg_time_s=1.5,
        source="manual_video",
        confidence=0.95,
        annotator="operator_unit_test",
        evidence_reference="synthetic://video/frame-150",
        evidence_sha256=evidence_sha256,
    )

    output = tmp_path / "accepted.npz"
    report = import_jidian_emg(fixture["selection"], output)

    annotation = report["trials"][0]["event"]["annotation_audit"]
    assert report["status"] == "exported"
    assert annotation["annotation_manifest_sha256"] == result["annotation_manifest_sha256"]
    assert annotation["evidence_sha256"] == evidence_sha256


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("interrupted", True, "acquisition_interrupted"),
        ("interrupted", "false", "acquisition_interrupted"),
        ("interrupted", "__delete__", "acquisition_interrupted"),
        ("receive_error", "socket closed", "acquisition_receive_error"),
        ("receive_error", "__delete__", "acquisition_receive_error"),
        ("dropped_samples", False, "acquisition_incomplete"),
    ],
)
def test_metadata_transport_failures_are_rejected(
    tmp_path: Path,
    field: str,
    value: Any,
    expected_code: str,
) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    metadata_path = _trial_path(fixture, fixture["trial_ids"][0]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if value == "__delete__":
        metadata.pop(field)
    else:
        metadata[field] = value
    _write_json(metadata_path, metadata)
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert expected_code in _reason_codes(caught.value)
    assert not output.exists()


def test_raw_npz_is_validated_instead_of_only_hashed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    raw_path = _trial_path(fixture, fixture["trial_ids"][0]) / "raw_emg.npz"
    with np.load(raw_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["stream_channel_ids"] = np.arange(2, 18, dtype=np.int16)
    np.savez(raw_path, **arrays)
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert "raw_profile_mismatch" in _reason_codes(caught.value)
    assert not output.exists()


def test_manual_preprocessing_mvc_and_cross_trial_contracts_are_all_or_nothing(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    second = _trial_path(fixture, fixture["trial_ids"][1])
    processing_path = second / "processing.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    processing["processing_config"]["envelope_lowpass_hz"] = 6.0
    processing["fallback_method"] = "dynamic_p95"
    _write_json(processing_path, processing)
    output = tmp_path / "partial_is_forbidden.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    codes = _reason_codes(caught.value)
    assert "processing_config_mismatch" in codes
    assert "selection_incomplete" in codes
    assert caught.value.report["summary"]["eligible_trial_count"] == 1
    assert caught.value.report["summary"]["included_trial_count"] == 0
    assert not output.exists()


def test_missing_fallback_field_cannot_masquerade_as_null(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    processing_path = _trial_path(fixture, fixture["trial_ids"][0]) / "processing.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    processing.pop("fallback_method")
    _write_json(processing_path, processing)
    output = tmp_path / "rejected.npz"

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)

    assert "processing_fields_missing" in _reason_codes(caught.value)
    assert not output.exists()


def test_profile_hash_is_exact_canonical_json_and_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, trial_specs=((1, 300, 150),))
    output = tmp_path / "imported.npz"
    first = import_jidian_emg(fixture["selection"], output)
    profile = json.loads((fixture["session"] / "channel_profile.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256(
        json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert first["session_evidence"]["channel_profile_sha256"] == expected

    with pytest.raises(JidianEmgImportError) as caught:
        import_jidian_emg(fixture["selection"], output)
    assert "output_exists" in _reason_codes(caught.value)
    assert output.is_file()
