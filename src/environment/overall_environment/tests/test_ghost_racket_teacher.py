from __future__ import annotations

import numpy as np

from environment.overall_environment.src.ghost_racket_teacher import (
    build_ghost_racket_trajectory,
    compute_ghost_reward,
    sample_ghost_at_phase,
)


def test_build_ghost_racket_trajectory_from_mock_right_hand_positions():
    right_hand_pos = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.1],
            [0.3, 0.0, 1.3],
            [0.6, 0.0, 1.45],
            [0.8, 0.0, 1.5],
        ],
        dtype=float,
    )

    trajectory = build_ghost_racket_trajectory(right_hand_pos, fps=50.0)

    assert len(trajectory.frames) == 5
    assert trajectory.impact_phase > 0.0
    assert trajectory.impact_phase < 1.0
    assert trajectory.impact_index == 3
    assert np.isfinite(trajectory.head_speed).all()


def test_sample_ghost_at_phase_returns_finite_frame():
    right_hand_pos = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
        dtype=float,
    )
    trajectory = build_ghost_racket_trajectory(right_hand_pos, fps=30.0)

    frame = sample_ghost_at_phase(trajectory, 0.5)

    assert frame.pos.shape == (3,)
    assert frame.xmat.shape == (3, 3)
    assert frame.velocity.shape == (3,)
    assert np.isfinite(frame.pos).all()


def test_compute_ghost_reward_prefers_close_pose_and_velocity():
    right_hand_pos = np.array(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 1.1], [0.5, 0.0, 1.3]],
        dtype=float,
    )
    trajectory = build_ghost_racket_trajectory(right_hand_pos, fps=30.0)
    target = sample_ghost_at_phase(trajectory, 0.5)

    good = compute_ghost_reward(
        racket_pos=target.pos,
        racket_xmat=target.xmat,
        racket_velocity=target.velocity,
        ghost_frame=target,
    )
    bad = compute_ghost_reward(
        racket_pos=target.pos + np.array([1.0, 0.0, 0.0]),
        racket_xmat=-target.xmat,
        racket_velocity=-target.velocity,
        ghost_frame=target,
    )

    assert good["total"] > bad["total"]
    assert good["pos"] > bad["pos"]
