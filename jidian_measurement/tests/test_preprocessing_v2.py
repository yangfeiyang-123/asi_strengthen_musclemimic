import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.signal import welch

matplotlib.use("Agg")

from emg.models import ProcessingConfig
from emg.mvc_reference import participant_mvc_envelope_peaks
from emg.offline import load_processing_config, preprocess_dataset, preprocess_session
from emg.preprocessing_plot import create_processing_comparison_figure
from emg.preprocessing_qc import assess_preprocessing_quality
from emg.processing import preprocess_emg, preprocess_signal_stages
from emg.profiles import LEGACY_HIGH_CLEAR_6CH
from emg.storage import atomic_save_npz, atomic_write_json, read_json


def _spectral_amplitude(values, fs_hz, frequency_hz):
    frequencies, power = welch(values, fs=fs_hz, nperseg=min(4096, len(values)))
    return float(power[np.argmin(np.abs(frequencies - frequency_hz))])


def _write_session(root: Path, participant: str, session_id: str, gain: float = 1.0) -> Path:
    profile = LEGACY_HIGH_CLEAR_6CH
    session = root / participant / session_id
    trial = session / "trials" / "quiet_stance" / "trial_001"
    trial.mkdir(parents=True)
    atomic_write_json(
        session / "session.json",
        {
            "participant_id": participant,
            "session_id": session_id,
            "channel_profile_id": profile.profile_id,
        },
    )
    atomic_write_json(session / "channel_profile.json", profile.to_dict())
    fs_hz = 2000.0
    time_s = np.arange(4000) / fs_hz
    carriers = np.column_stack(
        [gain * (0.04 + 0.01 * channel) * np.sin(2 * np.pi * (80 + 10 * channel) * time_s) for channel in range(6)]
    ).astype(np.float32)
    atomic_save_npz(
        trial / "raw_emg.npz",
        emg_mV=carriers,
        time_s=time_s,
        sample_index=np.arange(len(time_s)),
        stream_channel_ids=np.asarray(profile.channel_ids, dtype=np.int16),
        fs_hz=np.asarray(fs_hz),
    )
    atomic_write_json(
        trial / "metadata.json",
        {
            "participant_id": participant,
            "session_id": session_id,
            "trial_id": f"{session_id}_quiet_stance_trial_001",
            "trial_index": 1,
            "action_id": "quiet_stance",
            "channel_profile_id": profile.profile_id,
            "channel_profile_snapshot": profile.to_dict(),
            "valid_for_analysis": True,
        },
    )
    return session


def test_default_config_matches_requested_semg_pipeline(tmp_path):
    config = ProcessingConfig()
    assert (config.bandpass_low_hz, config.bandpass_high_hz) == (30.0, 300.0)
    assert config.filter_order == 4
    assert config.notch_hz == 50.0
    assert config.notch_bandwidth_hz == 5.0
    assert config.effective_notch_quality_factor == pytest.approx(10.0)
    assert config.envelope_lowpass_hz == 4.0
    assert config.zero_phase is True
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"processing": config.to_dict()}), encoding="utf-8")
    assert load_processing_config(path) == config


def test_demean_bandpass_and_notch_suppress_dc_and_50hz():
    fs_hz = 2000.0
    time_s = np.arange(int(6 * fs_hz)) / fs_hz
    signal = 1.2 + np.sin(2 * np.pi * 50 * time_s) + 0.3 * np.sin(2 * np.pi * 100 * time_s)
    emg = np.column_stack([signal, 0.5 * signal])
    stages = preprocess_signal_stages(emg, ProcessingConfig(normalization="none"))
    assert np.max(np.abs(np.mean(stages["demeaned_mV"], axis=0))) < 1e-6
    guard = stages["edge_guard_samples"]
    filtered = stages["filtered_mV"][guard:-guard, 0]
    ratio = _spectral_amplitude(filtered, fs_hz, 50.0) / _spectral_amplitude(filtered, fs_hz, 100.0)
    assert ratio < 0.01
    assert np.all(stages["rectified_mV"] >= 0)
    assert np.all(stages["envelope_mV"] >= 0)


def test_zero_phase_envelope_preserves_burst_peak_time():
    fs_hz = 2000.0
    time_s = np.arange(int(6 * fs_hz)) / fs_hz
    expected_peak_s = 3.0
    burst = np.exp(-0.5 * ((time_s - expected_peak_s) / 0.25) ** 2)
    emg = (burst * np.sin(2 * np.pi * 100 * time_s))[:, None]
    result = preprocess_emg(emg, ProcessingConfig(normalization="none"))
    observed_peak_s = time_s[int(np.argmax(result["envelope_mV"][:, 0]))]
    assert abs(observed_peak_s - expected_peak_s) < 0.02


def test_super_mvc_is_warning_only_and_keeps_trial_analysis_ready():
    profile = LEGACY_HIGH_CLEAR_6CH
    config = ProcessingConfig(normalization="mvc", edge_guard_s=0.0)
    time_s = np.arange(4000) / config.sample_rate_hz
    signal = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * (80 + index * 10) * time_s)
            for index in range(len(profile.channels))
        ]
    )
    normalized = np.full_like(signal, 2.5)
    missing = [
        {
            "missing_samples": 0,
            "long_gap_detected": False,
            "exceeds_missing_fraction": False,
        }
        for _channel in profile.channels
    ]
    outliers = [
        {"outliers_interpolated": 0, "outliers_retained": 0}
        for _channel in profile.channels
    ]

    report = assess_preprocessing_quality(
        signal,
        signal,
        signal,
        np.abs(signal),
        normalized,
        missing,
        outliers,
        config,
        profile,
        normalization_method="mvc",
    )

    assert report["analysis_ready"] is True
    assert report["critical_channel_count"] == 0
    assert report["mvc_exceedance_policy"] == {
        "schema_version": "mvc_exceedance_dual_track_qc_v1",
        "signal_quality_separate_from_mvc_reference_quality": True,
        "percent_mvc_unclipped": True,
        "exceedance_alone_is_critical": False,
        "reported_statistics": ["p95", "p99", "max"],
    }
    for channel in report["channels"]:
        assert channel["mvc_exceedance_is_critical"] is False
        assert channel["normalized_p99_mvc"] == pytest.approx(2.5)
        assert "mvc_reference_may_be_underestimated" in channel["warnings"]


def test_missing_values_and_isolated_spikes_are_interpolated():
    fs_hz = 2000.0
    time_s = np.arange(4000) / fs_hz
    emg = np.sin(2 * np.pi * 100 * time_s)[:, None]
    emg[500:503] = np.nan
    emg[1500] = 100.0
    result = preprocess_signal_stages(emg, ProcessingConfig(normalization="none", max_interpolation_gap_s=0.01))
    assert result["missing_value_report"][0]["missing_samples"] == 3
    assert result["outlier_report"][0]["outliers_interpolated"] >= 1
    assert np.isfinite(result["filtered_mV"]).all()


def test_short_signal_is_rejected_with_clear_error():
    emg = np.ones((100, 2))
    with pytest.raises(ValueError, match="shorter than configured minimum"):
        preprocess_signal_stages(emg, ProcessingConfig())


def test_participant_mvc_is_recomputed_from_raw_repetitions(tmp_path):
    profile = LEGACY_HIGH_CLEAR_6CH
    current_session = tmp_path / "P1" / "ACTION"
    current_session.mkdir(parents=True)
    mvc_session = tmp_path / "P1" / "MVC"
    fs_hz = 2000.0
    time_s = np.arange(4000) / fs_hz
    for channel in profile.channels:
        rep = mvc_session / "mvc" / f"{channel.side}_{channel.muscle_slug}" / "rep_001"
        rep.mkdir(parents=True)
        amplitude = 0.1 + channel.sensor_id * 0.01
        raw = (amplitude * np.sin(2 * np.pi * 100 * time_s))[:, None]
        atomic_save_npz(
            rep / "mvc_timeseries.npz",
            raw_emg_mV=raw,
            time_s=time_s,
            sample_index=np.arange(len(time_s)),
            stream_channel_ids=np.asarray([channel.sensor_id]),
            fs_hz=np.asarray(fs_hz),
        )
        atomic_write_json(
            rep / "metadata.json",
            {
                "valid": True,
                "expected_samples": len(time_s),
                "received_samples": len(time_s),
                "dropped_samples": 0,
                "interrupted": False,
                "receive_error": None,
            },
        )
    values, provenance = participant_mvc_envelope_peaks(
        current_session,
        profile,
        ProcessingConfig(normalization="mvc"),
        scope="participant",
    )
    assert values is not None and values.shape == (6,)
    assert np.all(values > 0)
    assert np.all(np.diff(values) > 0)
    assert provenance["missing_sensor_ids"] == []


def test_incomplete_mvc_marked_valid_is_still_rejected_from_normalization(tmp_path):
    profile = LEGACY_HIGH_CLEAR_6CH
    current_session = tmp_path / "P1" / "ACTION"
    current_session.mkdir(parents=True)
    mvc_session = tmp_path / "P1" / "MVC"
    fs_hz = 2000.0
    time_s = np.arange(3999) / fs_hz
    channel = profile.channels[0]
    rep = mvc_session / "mvc" / f"{channel.side}_{channel.muscle_slug}" / "rep_001"
    rep.mkdir(parents=True)
    raw = (0.1 * np.sin(2 * np.pi * 100 * time_s))[:, None]
    atomic_save_npz(
        rep / "mvc_timeseries.npz",
        raw_emg_mV=raw,
        time_s=time_s,
        sample_index=np.arange(len(time_s)),
        stream_channel_ids=np.asarray([channel.sensor_id]),
        fs_hz=np.asarray(fs_hz),
    )
    atomic_write_json(
        rep / "metadata.json",
        {
            "valid": True,
            "expected_samples": 4000,
            "received_samples": 3999,
            "dropped_samples": 1,
        },
    )
    values, provenance = participant_mvc_envelope_peaks(
        current_session,
        profile,
        ProcessingConfig(normalization="mvc"),
        scope="participant",
    )
    assert values is None
    first_channel = provenance["channels"][0]
    assert first_channel["valid_repetitions"] == []
    assert first_channel["rejected_repetitions"][0]["reason"] == "mvc_hard_qc_failed"
    assert "incomplete_stream" in first_channel["rejected_repetitions"][0]["hard_failures"]


def test_session_preprocessing_saves_all_stages_and_metadata(tmp_path):
    session = _write_session(tmp_path, "P1", "S1")
    outputs = preprocess_session(
        session,
        LEGACY_HIGH_CLEAR_6CH.profile_id,
        config=ProcessingConfig(normalization="none"),
        save_figures=False,
    )
    assert len(outputs) == 1
    trial = outputs[0].parent
    required = {
        "raw_emg.npz",
        "filtered_emg.npz",
        "rectified_emg.npz",
        "envelope_emg.npz",
        "normalized_emg.npz",
        "processed_emg.npz",
        "preprocessing_qc.json",
        "processing.json",
    }
    assert required <= {path.name for path in trial.iterdir()}
    with np.load(outputs[0], allow_pickle=False) as processed:
        for key in ("raw_emg_mV", "filtered_mV", "rectified_mV", "envelope_mV", "normalized_envelope"):
            assert processed[key].shape == (4000, 6)
        assert processed["action_id"].item() == "quiet_stance"
        assert processed["trial_id"].item() == "S1_quiet_stance_trial_001"
        assert processed["channel_names"].shape == (6,)
    record = read_json(trial / "processing.json")
    assert record["processing_config"]["bandpass_low_hz"] == 30.0
    assert record["normalization_method"] == "none"
    assert (session / "preprocessing.log.jsonl").exists()


def test_dataset_batch_processes_multiple_participants(tmp_path):
    _write_session(tmp_path, "P1", "S1", gain=1.0)
    _write_session(tmp_path, "P2", "S1", gain=2.0)
    summary = preprocess_dataset(
        tmp_path,
        LEGACY_HIGH_CLEAR_6CH.profile_id,
        ProcessingConfig(normalization="none"),
        save_figures=False,
    )
    assert summary["session_count"] == 2
    assert summary["completed_sessions"] == 2
    assert summary["failed_sessions"] == 0
    assert (tmp_path / "preprocessing_batch_summary.json").exists()


def test_comparison_figure_has_one_labeled_axis_per_channel():
    profile = LEGACY_HIGH_CLEAR_6CH
    time_s = np.arange(200) / 2000.0
    values = np.zeros((len(time_s), len(profile.channels)))
    fig = create_processing_comparison_figure(
        time_s, values, values, values, values, profile, "test", edge_guard_samples=10
    )
    try:
        visible = [axis for axis in fig.axes if axis.get_visible()]
        assert len(visible) == len(profile.channels)
        assert all(axis.get_xlabel() == "Time (s)" for axis in visible)
        assert all(axis.get_ylabel() == "sEMG (mV)" for axis in visible)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
