from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    FeedConfig,
    HitWindow,
    build_feed_bank,
    integrate_shuttle_flight,
    launch_quat_from_velocity,
    load_feed_bank,
    sample_feed,
    save_feed_bank,
)


def test_integrate_reaches_terminal_velocity() -> None:
    trajectory = integrate_shuttle_flight(
        np.array([0.0, 0.0, 30.0]), np.zeros(3), dt=0.002, t_max=4.0
    )
    final_speed = float(np.linalg.norm(trajectory[-1, 4:7]))
    assert 6.5 <= final_speed <= 7.2  # nominal vt = 6.86 m/s


def test_sample_feed_intercepts_window() -> None:
    rng = np.random.default_rng(11)
    cfg = FeedConfig()
    window = HitWindow()
    for _ in range(20):
        sample = sample_feed(rng, cfg, window)
        assert window.contains(sample.intercept_point[None])[0]
        assert sample.intercept_velocity[0] < 0.0
        assert sample.launch_pos[0] >= cfg.launch_x_range[0]
        assert sample.intercept_time_s > 0.0


def test_feed_bank_deterministic_and_roundtrip(tmp_path: Path) -> None:
    bank_a = build_feed_bank(4, seed=3)
    bank_b = build_feed_bank(4, seed=3)
    for a, b in zip(bank_a, bank_b):
        np.testing.assert_allclose(a.launch_pos, b.launch_pos)
        np.testing.assert_allclose(a.launch_vel, b.launch_vel)

    path = save_feed_bank(tmp_path / "bank.npz", bank_a)
    loaded = load_feed_bank(path)
    assert len(loaded) == len(bank_a)
    for a, b in zip(bank_a, loaded):
        np.testing.assert_allclose(a.trajectory, b.trajectory)
        assert a.intercept_index == b.intercept_index
        np.testing.assert_allclose(a.intercept_point, b.intercept_point)


def test_rejection_raises_on_impossible_window() -> None:
    rng = np.random.default_rng(0)
    impossible = HitWindow(x_range=(-3.0, -2.5), y_range=(-0.5, 0.5), z_range=(-2.0, -1.0))
    cfg = FeedConfig(max_attempts=20)
    with pytest.raises(RuntimeError):
        sample_feed(rng, cfg, impossible)


def test_launch_quat_aligns_nose_with_velocity() -> None:
    vel = np.array([-8.0, 1.5, 3.0])
    quat = launch_quat_from_velocity(vel)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
    w, x, y, z = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    nose_world = rot @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(nose_world, vel / np.linalg.norm(vel), atol=1e-9)
