from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musclemimic.badminton.data.event_lookup import EventReferenceLookup
from musclemimic.distill.motion_identity import MotionIdentityMap


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
    phase_global: np.ndarray | None = None
    phase_id: np.ndarray | None = None
    phase_local: np.ndarray | None = None
    time_to_impact_s: np.ndarray | None = None
    time_from_impact_s: np.ndarray | None = None
    impact_flag: np.ndarray | None = None
    racket_position_world: np.ndarray | None = None
    racket_quaternion_world: np.ndarray | None = None
    racket_linear_velocity_world: np.ndarray | None = None
    racket_angular_velocity_world: np.ndarray | None = None
    stringbed_normal_world: np.ndarray | None = None
    stringbed_center_world: np.ndarray | None = None
    racket_reference_confidence: np.ndarray | None = None
    racket_reference_source: str | None = None
    reference_bundle_content_fingerprint: str | None = None

    def frame_at_traj_step(self, traj_step: int) -> int:
        frame = round(traj_step * self.effective_ref_stride)
        return min(max(frame, 0), self.num_frames - 1)


@dataclass(frozen=True)
class ContactTrackingBank:
    """Padded, trajectory-indexed contact/event/racket caches for JIT lookup."""

    stance_mask: np.ndarray
    foot_points: np.ndarray
    body_laplacian: None
    foot_labels: list[str]
    num_frames: np.ndarray
    reference_fps: np.ndarray
    control_dt: float
    effective_ref_stride: np.ndarray
    phase_global: np.ndarray
    phase_id: np.ndarray
    phase_local: np.ndarray
    time_to_impact_s: np.ndarray
    time_from_impact_s: np.ndarray
    impact_flag: np.ndarray
    racket_position_world: np.ndarray
    racket_quaternion_world: np.ndarray
    racket_linear_velocity_world: np.ndarray
    racket_angular_velocity_world: np.ndarray
    stringbed_normal_world: np.ndarray
    stringbed_center_world: np.ndarray
    racket_reference_confidence: np.ndarray
    racket_reference_source: tuple[str, ...]
    reference_bundle_content_fingerprint: tuple[str, ...]
    motion_uids: np.ndarray
    motion_paths: tuple[str, ...]
    event_reference_bank_fingerprint: str

    @property
    def num_trajectories(self) -> int:
        return int(self.stance_mask.shape[0])

    def frame_at_traj_step(self, traj_no: int, traj_step: int) -> int:
        trajectory = int(traj_no)
        if trajectory < 0 or trajectory >= self.num_trajectories:
            raise ValueError(f"trajectory index outside contact bank: {trajectory}")
        frame = round(traj_step * float(self.effective_ref_stride[trajectory]))
        return min(max(frame, 0), int(self.num_frames[trajectory]) - 1)


def load_contact_tracking_data(
    source: str | Path,
    control_dt: float,
    *,
    strict_contract: bool = False,
) -> ContactTrackingData:
    path = Path(source).resolve()
    cache_npz = _find_cache_npz(path)
    data = np.load(cache_npz, allow_pickle=True)
    stance_mask = np.asarray(data["stance_mask"], dtype=np.bool_)
    foot_points = np.asarray(data["foot_points"], dtype=np.float32)
    foot_labels = [str(label) for label in np.asarray(data["foot_labels"]).tolist()]
    ref_fps = float(np.asarray(data["reference_fps"]).item())
    eff_stride = float(np.asarray(data.get("effective_ref_stride", ref_fps * control_dt)).item())
    runtime_dt = float(control_dt)
    if strict_contract:
        if not np.isfinite(runtime_dt) or runtime_dt <= 0.0:
            raise ValueError("contact tracking runtime control_dt must be finite and positive")
        if not np.isfinite(ref_fps) or ref_fps <= 0.0:
            raise ValueError("contact tracking reference_fps must be finite and positive")
        if "control_dt" in data:
            stored_dt = float(np.asarray(data["control_dt"]).item())
            if not np.isclose(stored_dt, runtime_dt, atol=1e-8, rtol=0.0):
                raise ValueError(f"contact tracking cache control_dt={stored_dt:g} differs from runtime {runtime_dt:g}")
        expected_stride = ref_fps * runtime_dt
        if not np.isfinite(eff_stride) or not np.isclose(eff_stride, expected_stride, atol=1e-6, rtol=0.0):
            raise ValueError(
                "contact tracking effective_ref_stride is inconsistent with reference_fps/control_dt: "
                f"stored={eff_stride:g} expected={expected_stride:g}"
            )
        if (
            stance_mask.ndim != 2
            or foot_points.shape != (*stance_mask.shape, 3)
            or len(foot_labels) != stance_mask.shape[1]
            or not np.all(np.isfinite(foot_points))
        ):
            raise ValueError("contact tracking stance_mask/foot_points/foot_labels have inconsistent shapes")
    body_lap = np.asarray(data["body_laplacian"], dtype=np.float32) if "body_laplacian" in data else None
    event_keys = (
        "phase_global",
        "phase_id",
        "phase_local",
        "time_to_impact_s",
        "time_from_impact_s",
        "impact_flag",
    )
    racket_keys = (
        "racket_position_world",
        "racket_quaternion_world",
        "racket_linear_velocity_world",
        "racket_angular_velocity_world",
        "stringbed_normal_world",
        "stringbed_center_world",
        "racket_reference_confidence",
        "racket_reference_source",
    )
    if strict_contract:
        _require_complete_group(data, event_keys, label="event reference")
        _require_complete_group(data, racket_keys, label="racket reference")
    num_frames = int(stance_mask.shape[0])
    optional: dict[str, object] = {}
    if all(key in data for key in event_keys):
        optional.update(
            phase_global=_frame_array(data, "phase_global", num_frames, np.float32),
            phase_id=_frame_array(data, "phase_id", num_frames, np.int16),
            phase_local=_frame_array(data, "phase_local", num_frames, np.float32),
            time_to_impact_s=_frame_array(data, "time_to_impact_s", num_frames, np.float32),
            time_from_impact_s=_frame_array(data, "time_from_impact_s", num_frames, np.float32),
            impact_flag=_frame_array(data, "impact_flag", num_frames, np.bool_),
        )
    if all(key in data for key in racket_keys):
        optional.update(
            racket_position_world=_frame_array(data, "racket_position_world", num_frames, np.float32, 3),
            racket_quaternion_world=_frame_array(data, "racket_quaternion_world", num_frames, np.float32, 4),
            racket_linear_velocity_world=_frame_array(data, "racket_linear_velocity_world", num_frames, np.float32, 3),
            racket_angular_velocity_world=_frame_array(
                data, "racket_angular_velocity_world", num_frames, np.float32, 3
            ),
            stringbed_normal_world=_frame_array(data, "stringbed_normal_world", num_frames, np.float32, 3),
            stringbed_center_world=_frame_array(data, "stringbed_center_world", num_frames, np.float32, 3),
            racket_reference_confidence=_frame_array(data, "racket_reference_confidence", num_frames, np.float32),
            racket_reference_source=str(np.asarray(data["racket_reference_source"]).reshape(-1)[0]),
        )
    if "reference_bundle_content_fingerprint" in data:
        optional["reference_bundle_content_fingerprint"] = str(
            np.asarray(data["reference_bundle_content_fingerprint"]).reshape(-1)[0]
        )
    return ContactTrackingData(
        stance_mask=stance_mask,
        foot_points=foot_points,
        body_laplacian=body_lap,
        foot_labels=foot_labels,
        num_frames=num_frames,
        reference_fps=ref_fps,
        control_dt=runtime_dt,
        effective_ref_stride=eff_stride,
        **optional,
    )


def load_contact_tracking_bank(
    manifest: str | Path,
    *,
    control_dt: float,
    motion_identity_map: MotionIdentityMap,
) -> ContactTrackingBank:
    """Load an exact multi-motion event bank into padded JIT-safe arrays."""

    lookup = EventReferenceLookup.from_manifest(
        manifest,
        motion_identity_map=motion_identity_map,
    )
    entries = tuple(sorted(lookup.entries, key=lambda entry: entry.traj_no))
    expected_trajectories = tuple(range(len(entries)))
    if tuple(entry.traj_no for entry in entries) != expected_trajectories:
        raise ValueError("contact tracking bank traj_no must be contiguous from zero")
    caches = tuple(
        load_contact_tracking_data(
            entry.cache_path,
            control_dt=float(control_dt),
            strict_contract=True,
        )
        for entry in entries
    )
    labels = caches[0].foot_labels
    if any(cache.foot_labels != labels for cache in caches):
        raise ValueError("contact tracking bank foot label order differs across motions")
    if any(cache.racket_reference_source not in {"measured", "fused"} for cache in caches):
        sources = [cache.racket_reference_source for cache in caches]
        raise ValueError(f"multi-motion event bank mainline requires measured/fused racket references; got {sources}")
    required_fields = (
        "phase_global",
        "phase_id",
        "phase_local",
        "time_to_impact_s",
        "time_from_impact_s",
        "impact_flag",
        "racket_position_world",
        "racket_quaternion_world",
        "racket_linear_velocity_world",
        "racket_angular_velocity_world",
        "stringbed_normal_world",
        "stringbed_center_world",
        "racket_reference_confidence",
        "reference_bundle_content_fingerprint",
    )
    for index, cache in enumerate(caches):
        missing = [field for field in required_fields if getattr(cache, field) is None]
        if missing:
            raise ValueError(f"event bank cache {index} is incomplete; missing {missing}")
    max_frames = max(cache.num_frames for cache in caches)

    def padded(field: str) -> np.ndarray:
        return np.stack([_pad_frames(np.asarray(getattr(cache, field)), max_frames) for cache in caches])

    return ContactTrackingBank(
        stance_mask=padded("stance_mask").astype(np.bool_),
        foot_points=padded("foot_points").astype(np.float32),
        body_laplacian=None,
        foot_labels=list(labels),
        num_frames=np.asarray([cache.num_frames for cache in caches], dtype=np.int32),
        reference_fps=np.asarray([cache.reference_fps for cache in caches], dtype=np.float32),
        control_dt=float(control_dt),
        effective_ref_stride=np.asarray([cache.effective_ref_stride for cache in caches], dtype=np.float32),
        phase_global=padded("phase_global").astype(np.float32),
        phase_id=padded("phase_id").astype(np.int16),
        phase_local=padded("phase_local").astype(np.float32),
        time_to_impact_s=padded("time_to_impact_s").astype(np.float32),
        time_from_impact_s=padded("time_from_impact_s").astype(np.float32),
        impact_flag=padded("impact_flag").astype(np.bool_),
        racket_position_world=padded("racket_position_world").astype(np.float32),
        racket_quaternion_world=padded("racket_quaternion_world").astype(np.float32),
        racket_linear_velocity_world=padded("racket_linear_velocity_world").astype(np.float32),
        racket_angular_velocity_world=padded("racket_angular_velocity_world").astype(np.float32),
        stringbed_normal_world=padded("stringbed_normal_world").astype(np.float32),
        stringbed_center_world=padded("stringbed_center_world").astype(np.float32),
        racket_reference_confidence=padded("racket_reference_confidence").astype(np.float32),
        racket_reference_source=tuple(str(cache.racket_reference_source) for cache in caches),
        reference_bundle_content_fingerprint=tuple(str(cache.reference_bundle_content_fingerprint) for cache in caches),
        motion_uids=np.asarray([entry.motion_uid for entry in entries], dtype=np.int64),
        motion_paths=tuple(entry.motion_path for entry in entries),
        event_reference_bank_fingerprint=lookup.fingerprint,
    )


def _require_complete_group(data, keys: tuple[str, ...], *, label: str) -> None:
    present = [key for key in keys if key in data]
    if present and len(present) != len(keys):
        missing = sorted(set(keys) - set(present))
        raise ValueError(f"partial {label} cache is forbidden; missing {missing}")


def _frame_array(data, key: str, frames: int, dtype, width: int | None = None) -> np.ndarray:
    value = np.asarray(data[key], dtype=dtype)
    expected = (frames,) if width is None else (frames, width)
    if value.shape != expected:
        raise ValueError(f"{key} must have shape {expected}, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return value


def _pad_frames(value: np.ndarray, frames: int) -> np.ndarray:
    if value.ndim < 1 or value.shape[0] <= 0 or value.shape[0] > int(frames):
        raise ValueError(f"cannot pad invalid frame array with shape {value.shape}")
    if value.shape[0] == int(frames):
        return value
    padding = [(0, int(frames) - value.shape[0]), *[(0, 0)] * (value.ndim - 1)]
    # Repeat the final valid frame.  Runtime indexes are still clipped by each
    # trajectory's true length; edge padding merely keeps traced gathers finite.
    return np.pad(value, padding, mode="edge")


def _find_cache_npz(path: Path) -> Path:
    if path.is_file() and path.suffix == ".npz":
        return path
    candidate = path / "tracking_reference_cache.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No tracking cache found at {path}")
