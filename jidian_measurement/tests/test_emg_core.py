import json
import subprocess
import struct
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emg import acquisition, cli
from emg.models import (
    ChannelProfile,
    ChannelSpec,
    ProcessingConfig,
    RecordResult,
    SessionMetadata,
    TrialMetadata,
    TrignoConfig,
)
from emg.mvc import _assess_mvc_hard_qc, _finalize_mvc_review
from emg.processing import emg_envelope, preprocess_emg, stable_window_max
from emg.profiles import (
    BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1,
    BADMINTON_SYNERGY_16_V1,
    BADMINTON_SYNERGY_16_V2,
    LEGACY_HIGH_CLEAR_6CH,
    PROFILE_REGISTRY,
    ProfilePermissionError,
    require_analysis_profile,
    require_collection_profile,
)
from emg.qc import create_preview_figure
from emg.protocols import BADMINTON_PRIMITIVE_PROTOCOL_V1
from emg.storage import (
    atomic_save_npz,
    atomic_write_json,
    initialize_session,
    read_json,
    safe_child,
    save_trial_bundle,
)
from emg.trigno import TrignoClient


class FakeSocket:
    def __init__(self, payload=b"", interrupt=False):
        self.payload = bytearray(payload)
        self.interrupt = interrupt
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, value):
        self.sent.append(value)

    def recv(self, count):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if self.sent and not self.payload:
            return b"OK"
        if not self.payload:
            return b"READY"
        take = min(count, 17, len(self.payload))
        value = bytes(self.payload[:take])
        del self.payload[:take]
        return value

    def close(self):
        self.closed = True


class DisconnectingSocket(FakeSocket):
    def recv(self, count):
        if not self.payload:
            return b""
        take = min(count, 17, len(self.payload))
        value = bytes(self.payload[:take])
        del self.payload[:take]
        return value


def test_trigno_packet_decode_preserves_sensor_order():
    frames = (np.arange(32, dtype=np.float32).reshape(2, 16) + 0.25) / 1000.0
    packet = struct.pack("<" + "f" * frames.size, *frames.ravel())
    decoded = TrignoClient.decode_packet(packet, scale_to_mV=1000.0)
    np.testing.assert_allclose(decoded, frames * 1000.0)
    assert decoded.shape == (2, 16)


def test_record_receives_exact_requested_sample_count_and_sends_stop():
    frames = np.arange(64, dtype=np.float32).reshape(4, 16) / 1000.0
    data = FakeSocket(struct.pack("<" + "f" * frames.size, *frames.ravel()))
    command = FakeSocket()
    sockets = iter([command, data])
    config = TrignoConfig(sample_rate_hz=2000, samples_per_read=3)
    with TrignoClient(config, socket_factory=lambda *_args, **_kwargs: next(sockets)) as client:
        result = client.record(4 / 2000, [1, 16])
    assert result.received_samples == result.expected_samples == 4
    np.testing.assert_allclose(result.emg_mV, frames[:, [0, 15]] * 1000.0)
    assert any(b"START" in item for item in command.sent)
    assert any(b"STOP" in item for item in command.sent)
    assert command.closed and data.closed


def test_keyboard_interrupt_still_stops_and_closes():
    command = FakeSocket()
    data = FakeSocket(interrupt=True)
    sockets = iter([command, data])
    with TrignoClient(TrignoConfig(), socket_factory=lambda *_args, **_kwargs: next(sockets)) as client:
        result = client.record(0.1, [1])
    assert result.interrupted
    assert any(b"STOP" in item for item in command.sent)
    assert command.closed and data.closed


def test_partial_receive_keeps_complete_frames_and_reports_missing_samples():
    frame = np.arange(16, dtype=np.float32).reshape(1, 16) / 1000.0
    data = DisconnectingSocket(struct.pack("<" + "f" * frame.size, *frame.ravel()))
    command = FakeSocket()
    sockets = iter([command, data])
    config = TrignoConfig(sample_rate_hz=2000, samples_per_read=2)
    with TrignoClient(config, socket_factory=lambda *_args, **_kwargs: next(sockets)) as client:
        result = client.record(2 / 2000, [1, 16])
    assert result.received_samples == 1
    assert result.dropped_samples == 1
    assert result.incomplete
    np.testing.assert_allclose(result.emg_mV, frame[:, [0, 15]] * 1000.0)


def test_profile_order_and_bilateral_duplicate_slug_rules():
    assert BADMINTON_SYNERGY_16_V1.channel_ids == tuple(range(1, 17))
    assert len(BADMINTON_SYNERGY_16_V1.channels) == 16
    assert [c.muscle_slug for c in BADMINTON_SYNERGY_16_V1.channels].count("external_oblique") == 2
    assert BADMINTON_SYNERGY_16_V2.channel_ids == tuple(range(1, 17))
    assert [c.muscle_slug for c in BADMINTON_SYNERGY_16_V2.channels][10:14] == [
        "vastus_lateralis", "vastus_lateralis",
        "biceps_femoris_long_head", "biceps_femoris_long_head",
    ]
    assert [c.muscle_slug for c in BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.channels][6:14] == [
        "external_oblique", "external_oblique",
        "gluteus_maximus", "gluteus_maximus",
        "vastus_lateralis", "vastus_lateralis",
        "biceps_femoris_long_head", "biceps_femoris_long_head",
    ]
    with pytest.raises(ValueError, match="unique"):
        ChannelProfile(
            "bad",
            1,
            "duplicate sensor",
            "right",
            (
                ChannelSpec(1, "right", "a", "a", "a", "A"),
                ChannelSpec(1, "left", "b", "b", "b", "B"),
            ),
        )


def test_left_handed_participant_is_not_silently_mirrored():
    with pytest.raises(ValueError, match="refusing to silently mirror"):
        BADMINTON_SYNERGY_16_V2.validate_handedness("left")


def test_misdeclared_v1_is_preserved_but_blocked_for_collection_and_analysis():
    registration = PROFILE_REGISTRY[BADMINTON_SYNERGY_16_V1.profile_id]
    assert registration.status == "deprecated_misdeclared"
    assert registration.collection_allowed is False
    assert registration.analysis_allowed is False
    assert BADMINTON_SYNERGY_16_V1.channels[6].muscle_slug == "pronator_teres"
    with pytest.raises(ProfilePermissionError, match="not allowed for collection"):
        require_collection_profile(BADMINTON_SYNERGY_16_V1.profile_id)
    with pytest.raises(ProfilePermissionError, match="not allowed for analysis"):
        require_analysis_profile(BADMINTON_SYNERGY_16_V1.profile_id)


def test_safe_paths_reject_traversal(tmp_path):
    assert safe_child(tmp_path, "P001", "S001").is_relative_to(tmp_path.resolve())
    with pytest.raises(ValueError):
        safe_child(tmp_path, "../outside")


def test_atomic_json_and_npz_leave_complete_readable_files(tmp_path):
    json_path = atomic_write_json(tmp_path / "a.json", {"中文": "值", "x": 1})
    npz_path = atomic_save_npz(tmp_path / "a.npz", data=np.arange(5))
    assert read_json(json_path)["中文"] == "值"
    with np.load(npz_path) as payload:
        np.testing.assert_array_equal(payload["data"], np.arange(5))
    assert not list(tmp_path.glob(".*.tmp"))


def test_mvc_stable_window_uses_mean_not_single_peak():
    envelope = np.zeros((2000, 1))
    envelope[500:1500] = 2.0
    envelope[100] = 100.0
    value = stable_window_max(envelope, fs_hz=2000, window_s=0.5)
    assert value[0] == pytest.approx(2.0)


def test_filter_dimensions_and_nonnegative_envelope():
    rng = np.random.default_rng(7)
    emg = rng.normal(size=(4000, 16)).astype(np.float32)
    bandpassed, envelope = emg_envelope(emg, ProcessingConfig(normalization="none"))
    assert bandpassed.shape == envelope.shape == emg.shape
    assert np.all(envelope >= 0)


def test_missing_mvc_never_silently_changes_normalization():
    emg = np.random.default_rng(1).normal(size=(4000, 2))
    config = ProcessingConfig(normalization="mvc")
    with pytest.raises(FileNotFoundError, match="explicit fallback"):
        preprocess_emg(emg, config)
    processed = preprocess_emg(emg, config, explicit_fallback="dynamic_p95")
    assert processed["normalization_method"] == "dynamic_p95"
    assert processed["fallback_method"] == "dynamic_p95"


def test_sixteen_channel_preview_uses_four_by_four_grid():
    emg = np.zeros((200, 16), dtype=np.float32)
    fig = create_preview_figure(
        emg,
        2000.0,
        BADMINTON_SYNERGY_16_V2,
        "trial preview",
    )
    try:
        assert len(fig.axes) == 16
        positions = {
            (
                axis.get_subplotspec().rowspan.start,
                axis.get_subplotspec().colspan.start,
            )
            for axis in fig.axes
        }
        assert positions == {(row, column) for row in range(4) for column in range(4)}
    finally:
        plt.close(fig)


def test_rest_countdown_enter_skips_immediately(monkeypatch):
    monkeypatch.setattr(acquisition, "_enter_pressed", lambda: True)
    assert acquisition._rest_countdown(45.0, "next trial") is True


def test_timed_rest_record_retains_planned_actual_and_skip(monkeypatch):
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(acquisition.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(acquisition, "_rest_countdown", lambda _seconds, _label: True)
    record = acquisition._timed_rest_record(60.0, "next", rest_kind="trial")
    assert record["planned_seconds"] == pytest.approx(60.0)
    assert record["actual_seconds"] == pytest.approx(0.25)
    assert record["skipped"] is True
    assert record["status"] == "skipped"


def test_collect_cli_forwards_block_rest_override(monkeypatch):
    captured = {}

    def fake_collect_action(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "collect_action", fake_collect_action)
    result = cli.main(
        [
            "collect",
            "--participant", "P001",
            "--session", "S001",
            "--action", "forehand_high_clear",
            "--block-rest", "123",
            "--dry-run",
        ]
    )
    assert result == 0
    assert captured["block_rest_seconds"] == pytest.approx(123.0)


def test_invalid_noninteractive_trial_requires_a_reason():
    with pytest.raises(ValueError, match="non-empty.*reason"):
        acquisition._review_trial(False, "other", "")
    valid, labels, notes = acquisition._review_trial(False, "wrong_footwork", "late recovery step")
    assert valid is False
    assert labels == ["wrong_footwork"]
    assert notes == "late recovery step"


def test_session_metadata_file_cannot_silently_override_protected_fields():
    profile = BADMINTON_SYNERGY_16_V2
    protocol = BADMINTON_PRIMITIVE_PROTOCOL_V1
    accepted = acquisition._session_metadata(
        "P001",
        "S001",
        profile,
        protocol,
        "right",
        "right",
        {
            "participant_id": "P001",
            "handedness": "right",
            "schema_version": "emg_session_metadata_v2",
            "device_model": "device_01",
        },
    )
    assert accepted.device_model == "device_01"
    with pytest.raises(ValueError, match="protected field 'handedness'.*conflicts"):
        acquisition._session_metadata(
            "P001",
            "S001",
            profile,
            protocol,
            "right",
            "right",
            {"handedness": "left"},
        )
    with pytest.raises(ValueError, match="cannot provide collection_date"):
        acquisition._session_metadata(
            "P001",
            "S001",
            profile,
            protocol,
            "right",
            "right",
            {"collection_date": "2026-01-01T00:00:00+00:00"},
        )


def test_incomplete_mvc_cannot_be_approved_and_manual_rejection_needs_reason():
    incomplete = RecordResult(
        np.zeros((9, 1), dtype=np.float32),
        1000.0,
        np.asarray([1]),
        10,
        9,
        1,
        "start",
        "stop",
        receive_error="short stream",
    )
    valid, notes = _finalize_mvc_review(incomplete, operator_accepted=True, notes="")
    assert valid is False
    assert "Automatic invalidation" in notes
    assert "9/10" in notes

    complete = RecordResult(
        np.zeros((10, 1), dtype=np.float32),
        1000.0,
        np.asarray([1]),
        10,
        10,
        0,
        "start",
        "stop",
    )
    with pytest.raises(ValueError, match="non-empty reason"):
        _finalize_mvc_review(complete, operator_accepted=False, notes="")


@pytest.mark.parametrize("raw", [np.zeros((10, 1)), np.ones((10, 1))])
def test_mvc_all_zero_or_flatline_is_a_non_overridable_hard_failure(raw):
    result = RecordResult(
        raw.astype(np.float32),
        1000.0,
        np.asarray([1]),
        10,
        10,
        0,
        "start",
        "stop",
    )
    qc = _assess_mvc_hard_qc(result)
    valid, notes = _finalize_mvc_review(
        result,
        operator_accepted=True,
        notes="",
        qc_summary=qc,
    )
    assert qc["hard_qc_pass"] is False
    assert valid is False
    assert "Automatic invalidation" in notes


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("dominant_leg", "left"),
        ("mass_kg", 75.0),
        ("operator", "operator_02"),
        ("device_model", "different_device"),
        ("placement_protocol_id", "different_placement"),
        ("video_reference", "different_video"),
    ],
)
def test_resume_rejects_any_changed_immutable_session_metadata(tmp_path, field, different_value):
    profile = BADMINTON_SYNERGY_16_V2
    protocol = BADMINTON_PRIMITIVE_PROTOCOL_V1
    original = SessionMetadata(
        participant_id="P001",
        session_id="S001",
        channel_profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        handedness="right",
        dominant_leg="right",
        mass_kg=70.0,
        operator="operator_01",
        device_model="device_01",
        placement_protocol_id="placement_v1",
        video_reference="video/session_a.mp4",
        collection_date="2026-01-01T00:00:00+00:00",
    ).to_dict()
    initialize_session(tmp_path, "P001", "S001", original, profile, protocol.to_dict())

    same_session_new_invocation = {**original, "collection_date": "2026-01-02T00:00:00+00:00"}
    initialize_session(
        tmp_path,
        "P001",
        "S001",
        same_session_new_invocation,
        profile,
        protocol.to_dict(),
    )
    changed = {**same_session_new_invocation, field: different_value}
    with pytest.raises(ValueError, match=f"immutable metadata.*{field}"):
        initialize_session(tmp_path, "P001", "S001", changed, profile, protocol.to_dict())


def test_legacy_session_resume_accepts_only_missing_null_v2_provenance(tmp_path):
    profile = BADMINTON_SYNERGY_16_V2
    protocol = BADMINTON_PRIMITIVE_PROTOCOL_V1
    current = SessionMetadata(
        participant_id="P001",
        session_id="S001",
        channel_profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        handedness="right",
        dominant_leg="right",
        mass_kg=70.0,
    ).to_dict()
    new_fields = {
        "schema_version",
        "biological_subject_uid",
        "consent_protocol_id",
        "site_code",
        "device_model",
        "control_utility_version",
        "device_firmware",
        "sensor_serials",
        "placement_protocol_id",
        "electrode_coordinates",
        "electrode_orientations",
        "interelectrode_distance_mm",
        "video_reference",
    }
    legacy = {key: value for key, value in current.items() if key not in new_fields}
    initialize_session(tmp_path, "P001", "S001", legacy, profile, protocol.to_dict())
    with pytest.warns(RuntimeWarning, match="legacy metadata"):
        initialize_session(tmp_path, "P001", "S001", current, profile, protocol.to_dict())
    with pytest.warns(RuntimeWarning, match="legacy metadata"):
        with pytest.raises(ValueError, match="biological_subject_uid"):
            initialize_session(
                tmp_path,
                "P001",
                "S001",
                {**current, "biological_subject_uid": "NEW_SUBJECT_BINDING"},
                profile,
                protocol.to_dict(),
            )


def test_acquisition_cli_import_does_not_require_sklearn():
    script = """
import importlib.abc
import sys

class RejectSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise AssertionError('acquisition CLI imported sklearn')
        return None

sys.meta_path.insert(0, RejectSklearn())
from emg import cli
args = cli.build_parser().parse_args(['sensor-check', '--dry-run'])
assert args.command == 'sensor-check'
assert not any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_trial_bundle_is_still_saved(tmp_path):
    profile = LEGACY_HIGH_CLEAR_6CH
    emg = np.zeros((20, 6), dtype=np.float32)
    result = RecordResult(
        emg, 2000, np.arange(1, 7), 20, 20, 0, "start", "stop"
    )
    metadata = TrialMetadata(
        "P001", "S001", "trial_001", 1, "forehand_high_clear",
        "badminton_primitive_protocol_v1", 1, profile.profile_id, profile.version,
        profile.to_dict(), "right", "right", 2000, 0.01, 0.01, 20, 20, 0,
        "start", False, ["wrong_footwork"], "kept", {}, True,
        "software_cue_only", "1.0.0", None, False,
    )
    out = tmp_path / "trial_001"
    save_trial_bundle(out, result, metadata, [], {"qc_pass": False}, profile, legacy_csv=False)
    assert (out / "raw_emg.npz").exists()
    assert read_json(out / "metadata.json")["valid_for_analysis"] is False
    assert read_json(out / "metadata.json")["error_labels"] == ["wrong_footwork"]
