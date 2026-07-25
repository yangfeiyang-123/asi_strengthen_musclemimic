from __future__ import annotations

import numpy as np

from musclemimic.badminton.scripts.prepare_forehand_lift_root_smooth_v2 import (
    smooth_vertical_root,
    vertical_rms_acceleration,
)


def test_root_smoothing_uses_shortest_safe_window_and_reduces_acceleration():
    frames = np.arange(121, dtype=np.float64)
    clean = 0.9 + 0.04 * np.sin(frames / 15.0)
    noisy = clean + 0.008 * np.sin(frames * 2.2)
    before = vertical_rms_acceleration(noisy, 60.0)

    repaired, report = smooth_vertical_root(
        noisy,
        fps=60.0,
        target_rms_acceleration=before * 0.55,
        margin=1.0,
    )

    assert report["filter"] == "savgol"
    assert report["window_length"] >= 5
    assert vertical_rms_acceleration(repaired, 60.0) <= before * 0.55
    assert np.max(np.abs(repaired - noisy)) <= 0.06


def test_root_smoothing_is_identity_when_trajectory_already_passes():
    vertical = np.linspace(0.9, 1.0, 40, dtype=np.float64)

    repaired, report = smooth_vertical_root(
        vertical,
        fps=60.0,
        target_rms_acceleration=1.0,
    )

    np.testing.assert_array_equal(repaired, vertical)
    assert report["filter"] == "identity"
    assert report["window_length"] == 1
