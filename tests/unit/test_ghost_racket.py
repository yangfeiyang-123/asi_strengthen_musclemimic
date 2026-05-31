from __future__ import annotations

import numpy as np

from environment.overall_environment.src.ghost_racket import (
    GhostRacketFrame,
    GhostRacketTrajectory,
    interpolate_ghost,
)


def test_interpolate_ghost_midpoint_position():
    trajectory = GhostRacketTrajectory(
        phase=np.array([0.0, 1.0]),
        frames=[
            GhostRacketFrame(pos=np.array([0.0, 0.0, 1.0]), xmat=np.eye(3), velocity=np.array([1.0, 0.0, 0.0])),
            GhostRacketFrame(pos=np.array([2.0, 0.0, 1.0]), xmat=np.eye(3), velocity=np.array([1.0, 0.0, 0.0])),
        ],
    )

    frame = interpolate_ghost(trajectory, 0.5)

    np.testing.assert_allclose(frame.pos, np.array([1.0, 0.0, 1.0]))
