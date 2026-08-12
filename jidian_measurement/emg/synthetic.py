from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .models import RecordResult


def generate_synthetic_emg(
    channel_ids: tuple[int, ...] | list[int],
    duration_s: float,
    fs_hz: float = 2000.0,
    baseline_s: float = 1.0,
    cue_s: float = 0.5,
    post_s: float = 1.0,
    seed: int = 20260720,
    powerline_50hz: float = 0.0,
    clipping_mV: float | None = None,
    dropped_samples: int = 0,
) -> RecordResult:
    if duration_s <= 0 or not channel_ids:
        raise ValueError("Synthetic duration and channel_ids must be provided")
    expected = int(round(duration_s * fs_hz))
    rng = np.random.default_rng(seed)
    t = np.arange(expected, dtype=np.float64) / fs_hz
    channels = len(channel_ids)
    noise = rng.normal(size=(expected, channels))
    sos = butter(4, [25.0 / (fs_hz / 2), 350.0 / (fs_hz / 2)], btype="bandpass", output="sos")
    carrier = sosfiltfilt(sos, noise, axis=0)
    start = min(baseline_s + cue_s, duration_s * 0.6)
    end = max(start + 0.05, duration_s - post_s)
    phase = np.clip((t - start) / max(end - start, 1 / fs_hz), 0.0, 1.0)
    active = np.sin(np.pi * phase) ** 2
    active[(t < start) | (t > end)] = 0.0
    channel_gain = np.linspace(0.15, 0.65, channels)
    modulation = 0.8 + 0.2 * np.sin(2 * np.pi * (1.0 + np.arange(channels) / channels)[None, :] * t[:, None])
    envelope = 0.008 + active[:, None] * channel_gain[None, :] * modulation
    emg = carrier * envelope
    emg += powerline_50hz * np.sin(2 * np.pi * 50.0 * t)[:, None]
    if clipping_mV is not None:
        emg = np.clip(emg, -abs(clipping_mV), abs(clipping_mV))
    drop = min(max(int(dropped_samples), 0), max(expected - 1, 0))
    if drop:
        emg = emg[:-drop]
    received = emg.shape[0]
    now = datetime.now(timezone.utc).isoformat()
    return RecordResult(
        emg_mV=emg.astype(np.float32),
        fs_hz=fs_hz,
        stream_channel_ids=np.asarray(channel_ids, dtype=np.int16),
        expected_samples=expected,
        received_samples=received,
        dropped_samples=expected - received,
        start_time=now,
        stop_time=datetime.now(timezone.utc).isoformat(),
        receive_error="synthetic_dropped_samples" if drop else None,
    )
