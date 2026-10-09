"""Validated per-frame racket references for event-aware motion bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RACKET_REFERENCE_SCHEMA_VERSION = "forehand_clear_racket_reference_v1"
RACKET_REFERENCE_SOURCES = frozenset({"measured", "fused", "derived_rigid"})


@dataclass(frozen=True)
class RacketReference:
    position_world: np.ndarray
    quaternion_world: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    stringbed_normal_world: np.ndarray
    stringbed_center_world: np.ndarray
    confidence: np.ndarray
    source: str
    fps: float
    coordinate_system: str
    quaternion_convention: str = "wxyz"
    schema_version: str = RACKET_REFERENCE_SCHEMA_VERSION

    @property
    def num_frames(self) -> int:
        return int(self.position_world.shape[0])


def load_racket_reference(
    path: str | Path,
    *,
    num_frames: int,
    fps: float,
    coordinate_system: str = "amass_zup",
) -> RacketReference:
    with np.load(Path(path), allow_pickle=False) as npz:
        version = _scalar_string(npz, "schema_version")
        if version != RACKET_REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported racket reference schema: {version!r}")
        source = _scalar_string(npz, "racket_reference_source")
        if source not in RACKET_REFERENCE_SOURCES:
            raise ValueError(
                f"racket_reference_source must be one of {sorted(RACKET_REFERENCE_SOURCES)}, got {source!r}"
            )
        quaternion_convention = _scalar_string(npz, "quaternion_convention")
        if quaternion_convention != "wxyz":
            raise ValueError(f"racket reference quaternion_convention must be 'wxyz'; got {quaternion_convention!r}")
        stored_coordinate_system = _scalar_string(npz, "coordinate_system")
        if stored_coordinate_system != coordinate_system:
            raise ValueError(
                f"racket reference coordinate system {stored_coordinate_system!r} != {coordinate_system!r}"
            )
        stored_fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
        if not np.isclose(stored_fps, float(fps), atol=1e-6, rtol=0.0):
            raise ValueError(f"racket reference fps={stored_fps:g}; expected {fps:g}")
        arrays = {
            "position_world": _array(npz, "racket_position_world", (num_frames, 3)),
            "quaternion_world": _array(npz, "racket_quaternion_world", (num_frames, 4)),
            "linear_velocity_world": _array(npz, "racket_linear_velocity_world", (num_frames, 3)),
            "angular_velocity_world": _array(npz, "racket_angular_velocity_world", (num_frames, 3)),
            "stringbed_normal_world": _array(npz, "stringbed_normal_world", (num_frames, 3)),
            "stringbed_center_world": _array(npz, "stringbed_center_world", (num_frames, 3)),
            "confidence": _array(npz, "racket_reference_confidence", (num_frames,)),
        }
    quaternion_norm = np.linalg.norm(arrays["quaternion_world"], axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError("racket quaternions must be unit normalized")
    normal_norm = np.linalg.norm(arrays["stringbed_normal_world"], axis=-1)
    if not np.allclose(normal_norm, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError("stringbed normals must be unit normalized")
    confidence = arrays["confidence"]
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("racket_reference_confidence must lie in [0,1]")
    return RacketReference(
        **arrays,
        source=source,
        fps=float(fps),
        coordinate_system=stored_coordinate_system,
        quaternion_convention=quaternion_convention,
    )


def racket_reference_metrics(reference: RacketReference) -> dict[str, float]:
    return {
        "max_linear_speed_m_s": float(np.max(np.linalg.norm(reference.linear_velocity_world, axis=-1))),
        "max_angular_speed_rad_s": float(np.max(np.linalg.norm(reference.angular_velocity_world, axis=-1))),
        "min_confidence": float(np.min(reference.confidence)),
        "mean_confidence": float(np.mean(reference.confidence)),
    }


def _array(npz: Any, key: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if key not in npz:
        raise ValueError(f"racket reference is missing {key}")
    value = np.asarray(npz[key], dtype=np.float32)
    if value.shape != expected_shape:
        raise ValueError(f"{key} must have shape {expected_shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains NaN/Inf")
    return value


def _scalar_string(npz: Any, key: str) -> str:
    if key not in npz:
        raise ValueError(f"racket reference is missing {key}")
    value = np.asarray(npz[key])
    if value.size != 1:
        raise ValueError(f"racket reference {key} must be scalar")
    return str(value.reshape(-1)[0])
