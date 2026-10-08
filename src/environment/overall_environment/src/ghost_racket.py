from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GhostRacketFrame:
    pos: np.ndarray
    xmat: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class GhostRacketTrajectory:
    phase: np.ndarray
    frames: list[GhostRacketFrame]

    def __post_init__(self) -> None:
        phases = np.asarray(self.phase, dtype=float)
        if len(self.frames) != len(phases):
            raise ValueError("phase and frames length mismatch")
        if len(self.frames) < 2:
            raise ValueError("ghost trajectory requires at least two frames")
        if np.any(np.diff(phases) <= 0.0):
            raise ValueError("ghost trajectory phase must be strictly increasing")
        object.__setattr__(self, "phase", phases)


def interpolate_ghost(trajectory: GhostRacketTrajectory, phase: float) -> GhostRacketFrame:
    phases = np.asarray(trajectory.phase, dtype=float)
    value = float(np.clip(phase, phases[0], phases[-1]))
    hi = int(np.searchsorted(phases, value, side="right"))
    hi = min(max(hi, 1), len(phases) - 1)
    lo = hi - 1
    alpha = (value - phases[lo]) / max(phases[hi] - phases[lo], 1e-12)
    a = trajectory.frames[lo]
    b = trajectory.frames[hi]
    return GhostRacketFrame(
        pos=(1.0 - alpha) * np.asarray(a.pos, dtype=float) + alpha * np.asarray(b.pos, dtype=float),
        xmat=(1.0 - alpha) * np.asarray(a.xmat, dtype=float) + alpha * np.asarray(b.xmat, dtype=float),
        velocity=(1.0 - alpha) * np.asarray(a.velocity, dtype=float) + alpha * np.asarray(b.velocity, dtype=float),
    )
