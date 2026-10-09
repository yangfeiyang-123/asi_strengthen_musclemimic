import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from emg import event_annotation
from emg.cli import build_parser
from emg.event_annotation import EventAnnotationError, annotate_event, verify_committed_annotation
from emg.storage import atomic_save_npz, atomic_write_csv


EVENT_HEADER = [
    "event_name",
    "sample_index",
    "emg_time_s",
    "monotonic_time_ns",
    "wall_clock_iso",
    "source",
    "confidence",
    "notes",
]
EVIDENCE_SHA256 = "a" * 64


def _trial(tmp_path: Path, *, duplicate_contact: bool = False) -> Path:
    trial = tmp_path / "trial_001"
    trial.mkdir(parents=True)
    fs_hz = 1000.0
    time_s = np.arange(10, dtype=np.float64) / fs_hz
    atomic_save_npz(
        trial / "raw_emg.npz",
        emg_mV=np.zeros((10, 2), dtype=np.float32),
        time_s=time_s,
        sample_index=np.arange(10),
        stream_channel_ids=np.asarray([1, 2]),
        fs_hz=np.asarray(fs_hz),
    )
    rows = [
        EVENT_HEADER,
        ["recording_start", 0, 0.0, 1, "x", "software", 0.5, ""],
        ["movement_cue", 2, 0.002, 2, "x", "software_audio_visual", 0.5, ""],
        ["movement_start_manual", "", "", "", "", "unannotated", 0.0, "pending"],
        ["racket_contact", "", "", 3, "x", "unannotated", 0.0, "pending"],
        ["recording_stop", 10, 0.010, 4, "x", "software", 0.5, ""],
    ]
    if duplicate_contact:
        rows.insert(-1, ["racket_contact", "", "", 5, "x", "unannotated", 0.0, "duplicate"])
    atomic_write_csv(trial / "events.csv", rows)
    return trial


def _event_row(path: Path, event_name: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if row["event_name"] == event_name)


def test_annotation_is_atomic_and_records_before_after_hashes(tmp_path):
    trial = _trial(tmp_path)
    events_path = trial / "events.csv"
    before_hash = hashlib.sha256(events_path.read_bytes()).hexdigest()
    result = annotate_event(
        trial,
        "racket_contact",
        sample_index=5,
        emg_time_s=0.005,
        source="manual_video",
        confidence=0.9,
        annotator="operator_01",
        evidence_reference="video_cam1.mp4#frame=150",
        evidence_sha256=EVIDENCE_SHA256,
        notes="first visible racket-shuttle contact",
        expected_before_sha256=before_hash,
    )
    row = _event_row(events_path, "racket_contact")
    assert row["sample_index"] == "5"
    assert float(row["emg_time_s"]) == pytest.approx(0.005)
    assert row["source"] == "manual_video"
    assert float(row["confidence"]) == pytest.approx(0.9)
    assert row["monotonic_time_ns"] == row["wall_clock_iso"] == ""
    assert "video_cam1.mp4#frame=150" in row["notes"]
    after_hash = hashlib.sha256(events_path.read_bytes()).hexdigest()
    assert result["before_sha256"] == before_hash
    assert result["after_sha256"] == after_hash
    assert before_hash != after_hash

    records = [json.loads(line) for line in (trial / "events.annotation.audit.jsonl").read_text().splitlines()]
    assert [record["transaction_state"] for record in records] == ["prepared", "committed"]
    assert records[0]["annotation_id"] == records[1]["annotation_id"] == result["annotation_id"]
    assert records[0]["before_sha256"] == before_hash
    assert records[1]["after_sha256"] == after_hash
    assert records[0]["before_event"]["source"] == "unannotated"
    assert records[1]["after_event"]["source"] == "manual_video"
    assert records[1]["evidence_sha256"] == EVIDENCE_SHA256
    manifest = {
        key: value
        for key, value in records[1].items()
        if key not in {"transaction_state", "annotation_manifest_sha256"}
    }
    expected_manifest_hash = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert records[1]["annotation_manifest_sha256"] == expected_manifest_hash
    assert result["annotation_manifest_sha256"] == expected_manifest_hash
    verified = verify_committed_annotation(trial, "racket_contact")
    assert verified["annotation_id"] == result["annotation_id"]
    assert verified["evidence_sha256"] == EVIDENCE_SHA256

    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(EventAnnotationError, match="latest committed annotation hash"):
        verify_committed_annotation(trial, "racket_contact")


@pytest.mark.parametrize("event_name", ["movement_cue", "recording_start", "recording_stop"])
def test_acquisition_events_are_immutable_and_cannot_be_relabelled_as_contact(tmp_path, event_name):
    trial = _trial(tmp_path)
    before = (trial / "events.csv").read_bytes()
    with pytest.raises(EventAnnotationError, match="immutable|movement_cue"):
        annotate_event(
            trial,
            event_name,
            sample_index=5,
            source="manual_video",
            confidence=0.9,
            annotator="operator_01",
            evidence_reference="video#frame=5",
            evidence_sha256=EVIDENCE_SHA256,
        )
    assert (trial / "events.csv").read_bytes() == before


def test_annotation_rejects_software_source_time_mismatch_and_duplicate_events(tmp_path):
    trial = _trial(tmp_path)
    common = {
        "sample_index": 5,
        "confidence": 0.9,
        "annotator": "operator_01",
        "evidence_reference": "video#frame=5",
        "evidence_sha256": EVIDENCE_SHA256,
    }
    with pytest.raises(EventAnnotationError, match="Unsupported annotation source"):
        annotate_event(trial, "racket_contact", source="software_audio_visual", **common)
    with pytest.raises(EventAnnotationError, match="sample/time mismatch"):
        annotate_event(
            trial,
            "racket_contact",
            source="manual_video",
            emg_time_s=0.008,
            **common,
        )
    duplicate = _trial(tmp_path / "duplicate", duplicate_contact=True)
    with pytest.raises(EventAnnotationError, match="unique"):
        annotate_event(duplicate, "racket_contact", source="manual_video", **common)

    with pytest.raises(EventAnnotationError, match="64 hexadecimal"):
        annotate_event(
            trial,
            "racket_contact",
            source="manual_video",
            **{**common, "evidence_sha256": "not-a-sha256"},
        )


def test_failed_atomic_replace_leaves_original_and_prepared_audit(tmp_path, monkeypatch):
    trial = _trial(tmp_path)
    events_path = trial / "events.csv"
    before = events_path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(event_annotation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        annotate_event(
            trial,
            "racket_contact",
            sample_index=5,
            source="manual_video",
            confidence=0.9,
            annotator="operator_01",
            evidence_reference="video#frame=5",
            evidence_sha256=EVIDENCE_SHA256,
        )
    assert events_path.read_bytes() == before
    records = [json.loads(line) for line in (trial / "events.annotation.audit.jsonl").read_text().splitlines()]
    assert [record["transaction_state"] for record in records] == ["prepared"]
    assert not list(trial.glob(".events.csv.*.tmp"))


def test_verifier_allows_later_commits_for_other_event_slots(tmp_path):
    trial = _trial(tmp_path)
    racket = annotate_event(
        trial,
        "racket_contact",
        sample_index=5,
        source="manual_video",
        confidence=0.9,
        annotator="operator_01",
        evidence_reference="video#racket",
        evidence_sha256="a" * 64,
    )
    movement = annotate_event(
        trial,
        "movement_start_manual",
        sample_index=3,
        source="manual_video",
        confidence=0.8,
        annotator="operator_01",
        evidence_reference="video#movement",
        evidence_sha256="b" * 64,
    )
    verified = verify_committed_annotation(trial, "racket_contact")
    assert verified["annotation_id"] == racket["annotation_id"]
    assert verified["events_sha256"] == movement["after_sha256"]


def test_annotation_cli_contract_accepts_sample_or_time():
    args = build_parser().parse_args(
        [
            "annotate-event",
            "--trial-path", "trial_001",
            "--event-name", "racket_contact",
            "--emg-time-s", "1.25",
            "--source", "manual_video",
            "--confidence", "0.9",
            "--annotator", "operator_01",
            "--evidence-reference", "video#frame=250",
            "--evidence-sha256", EVIDENCE_SHA256,
        ]
    )
    assert args.command == "annotate-event"
    assert args.sample_index is None
    assert args.emg_time_s == pytest.approx(1.25)
    assert args.evidence_sha256 == EVIDENCE_SHA256
