import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emg.action_statistics import (
    create_action_statistics_figure,
    pointwise_mean_variance,
    save_action_statistics,
)
from emg.profiles import BADMINTON_SYNERGY_16_V2
from emg.storage import atomic_save_npz, atomic_write_csv, atomic_write_json


def test_pointwise_mean_variance_uses_sample_variance():
    envelopes = np.asarray(
        [
            [[[1.0], [3.0]], [[2.0], [4.0]]],
            [[[3.0], [5.0]], [[4.0], [6.0]]],
        ]
    ).reshape(2, 4, 1)
    mean, variance, std = pointwise_mean_variance(envelopes)
    np.testing.assert_allclose(mean[:, 0], [2.0, 4.0, 3.0, 5.0])
    np.testing.assert_allclose(variance[:, 0], 2.0)
    np.testing.assert_allclose(std[:, 0], np.sqrt(2.0))


def test_action_statistics_figure_is_four_by_four():
    time_s = np.arange(200) / 100.0
    mean = np.full((200, 16), 0.2)
    variance = np.full((200, 16), 0.01)
    std = np.sqrt(variance)
    fig = create_action_statistics_figure(
        time_s,
        mean,
        variance,
        std,
        BADMINTON_SYNERGY_16_V2,
        "test_action",
        5,
        movement_cue_s=0.5,
        movement_end_s=1.5,
    )
    try:
        assert len(fig.axes) == 16
        positions = {
            (axis.get_subplotspec().rowspan.start, axis.get_subplotspec().colspan.start)
            for axis in fig.axes
        }
        assert positions == {(row, column) for row in range(4) for column in range(4)}
    finally:
        plt.close(fig)


def test_save_action_statistics_excludes_invalid_trials(tmp_path):
    action_dir = tmp_path / "P001" / "S001" / "trials" / "test_action"
    fs_hz = 2000.0
    sample_count = 2000
    time_s = np.arange(sample_count) / fs_hz
    profile = BADMINTON_SYNERGY_16_V2
    for trial_index, (amplitude, valid) in enumerate(((0.2, True), (0.4, True), (2.0, False)), start=1):
        trial_dir = action_dir / f"trial_{trial_index:03d}"
        trial_dir.mkdir(parents=True)
        carrier = np.sin(2 * np.pi * 80.0 * time_s)[:, None]
        channel_scale = np.linspace(1.0, 1.5, 16)[None, :]
        emg = (amplitude * carrier * channel_scale).astype(np.float32)
        atomic_save_npz(
            trial_dir / "raw_emg.npz",
            emg_mV=emg,
            time_s=time_s,
            sample_index=np.arange(sample_count),
            stream_channel_ids=np.arange(1, 17),
            fs_hz=np.asarray(fs_hz),
        )
        atomic_write_json(
            trial_dir / "metadata.json",
            {
                "trial_id": f"test_action_trial_{trial_index:03d}",
                "trial_index": trial_index,
                "action_id": "test_action",
                "channel_profile_id": profile.profile_id,
                "channel_profile_snapshot": profile.to_dict(),
                "valid_for_analysis": valid,
                "error_labels": ["correct"] if valid else ["other"],
                "requested_duration_s": sample_count / fs_hz,
            },
        )
        atomic_write_csv(
            trial_dir / "events.csv",
            [
                ["event_name", "sample_index", "emg_time_s"],
                ["movement_cue", 400, 0.2],
                ["recording_stop", sample_count, sample_count / fs_hz],
            ],
        )

    outputs = save_action_statistics(
        action_dir,
        profile,
        {"action_id": "test_action", "post_s": 0.1},
        only_valid=True,
    )
    assert all(path.exists() for path in outputs.values())
    with np.load(outputs["npz"], allow_pickle=False) as payload:
        assert int(payload["n_trials"]) == 2
        assert payload["mean_envelope_mV"].shape == (sample_count, 16)
        assert payload["variance_envelope_mV2"].shape == (sample_count, 16)
        assert np.all(payload["variance_envelope_mV2"] >= 0)
    metadata = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert metadata["n_trials"] == 2
    assert metadata["channel_profile_id"] == profile.profile_id
    assert metadata["channel_profile_snapshot"] == profile.to_dict()
    assert metadata["excluded_trials"] == [
        {"trial_id": "test_action_trial_003", "reason": "other"}
    ]
