from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContactTrackingData:
    stance_mask: np.ndarray
    foot_points: np.ndarray
    body_laplacian: np.ndarray | None
    foot_labels: list[str]
    num_frames: int
    reference_fps: float
    control_dt: float
    effective_ref_stride: float

    def frame_at_traj_step(self, traj_step: int) -> int:
        frame = int(round(traj_step * self.effective_ref_stride))
        return min(max(frame, 0), self.num_frames - 1)


def load_contact_tracking_data(
    source: str | Path,
    control_dt: float,
) -> ContactTrackingData:
    path = Path(source).resolve()
    cache_npz = _find_cache_npz(path)
    data = np.load(cache_npz, allow_pickle=True)
    stance_mask = np.asarray(data["stance_mask"], dtype=np.bool_)
    foot_points = np.asarray(data["foot_points"], dtype=np.float32)
    foot_labels = [str(l) for l in np.asarray(data["foot_labels"]).tolist()]
    ref_fps = float(np.asarray(data["reference_fps"]).item())
    eff_stride = float(np.asarray(data.get("effective_ref_stride", ref_fps * control_dt)).item())
    body_lap = np.asarray(data["body_laplacian"], dtype=np.float32) if "body_laplacian" in data else None
    return ContactTrackingData(
        stance_mask=stance_mask,
        foot_points=foot_points,
        body_laplacian=body_lap,
        foot_labels=foot_labels,
        num_frames=int(stance_mask.shape[0]),
        reference_fps=ref_fps,
        control_dt=float(control_dt),
        effective_ref_stride=eff_stride,
    )


def _find_cache_npz(path: Path) -> Path:
    if path.is_file() and path.suffix == ".npz":
        return path
    candidate = path / "tracking_reference_cache.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No tracking cache found at {path}")
