from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferencePhaseManager:
    num_frames: int
    reference_fps: float
    control_dt: float
    start_frame: int = 0

    def __post_init__(self) -> None:
        if int(self.num_frames) <= 0:
            raise ValueError("num_frames must be positive")
        if float(self.reference_fps) <= 0:
            raise ValueError("reference_fps must be positive")
        if float(self.control_dt) <= 0:
            raise ValueError("control_dt must be positive")
        if int(self.start_frame) < 0:
            raise ValueError("start_frame must be non-negative")

    @property
    def effective_ref_stride(self) -> float:
        return float(self.reference_fps) * float(self.control_dt)

    def frame_at_control_step(self, control_step: int) -> int:
        frame = int(round(int(self.start_frame) + int(control_step) * self.effective_ref_stride))
        return self._clamp_frame(frame)

    def sample_indices(self, *, start_frame: int, offsets: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        values = np.asarray(offsets, dtype=np.int32) + int(start_frame)
        return np.asarray([self._clamp_frame(int(value)) for value in values], dtype=np.int32)

    def _clamp_frame(self, frame: int) -> int:
        return max(0, min(int(self.num_frames) - 1, int(frame)))

