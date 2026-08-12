"""Pre-PPO QC for the canonical 60 Hz source / 100 Hz GMR cache contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.action_registry import DEFAULT_ACTION, action_choices, resolve
from musclemimic.badminton.data.event_schema import PHASE_NAMES, EventTimeline
from musclemimic.badminton.data.racket_reference import RacketReference, racket_reference_metrics

TRAIN_MOTIONS = (
    "6月2日(1)-10", "6月2日(1)-1", "6月2日(1)-2", "6月2日(1)-4", "6月2日(1)-6",
    "6月2日(1)-7", "6月2日(1)-8", "6月2日(1)-9", "6月2日-2", "6月2日-3", "6月2日-4",
    "6月2日-6", "6月2日-7", "video1", "video2", "video3", "video4", "video5", "video6",
    "video7", "video8", "video9",
)
VAL_MOTIONS = ("6月2日(1)-3", "6月2日(1)-5", "6月2日-1", "6月2日-5", "video10")

# A normal 60 -> 100 Hz resample can differ by a few endpoint frames.  Flag
# materially shorter/longer caches without declaring them corrupt: retargeting
# may intentionally crop an invalid source window, but that decision must be
# visible before PPO training.
MAX_DURATION_ALIGNMENT_ERROR_S = 0.05
MAX_DURATION_ALIGNMENT_ERROR_FRACTION = 0.05
MAX_RIGHT_HAND_ANGULAR_SPEED_RAD_S = 60.0
MAX_ISOLATED_ROOT_SPEED_M_S = 4.0
MIN_ISOLATED_ROOT_VELOCITY_CHANGE_M_S = 2.0
MAX_ABSOLUTE_ROOT_SPEED_M_S = 6.0
MIN_MEDIAN_ROOT_HEIGHT_M = 0.85
MIN_RIGHT_HAND_PATH_LENGTH_M = 0.75
MIN_RIGHT_HAND_MAX_DISPLACEMENT_M = 0.35
MAX_NONROOT_QPOS_STEP_RAD = 0.45
# A large step is a discontinuity only when it is entered *and* exited sharply.
# This rejects one-frame solver jumps while allowing a physically continuous
# high-speed ramp such as raw_smooth_v1/video5's shoulder acceleration.
MIN_ISOLATED_STEP_NEIGHBOR_CHANGE_RAD = 0.25
_SAFE_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LEGACY_SOURCE_VARIANTS = frozenset({"initial", "initiall_wham"})


@dataclass(frozen=True)
class MotionQC:
    motion: str
    source_frames: int
    cache_frames: int
    source_fps: float
    cache_fps: float
    source_duration_s: float
    cache_duration_s: float
    expected_cache_frames_from_source: int
    cache_frame_alignment_delta: int
    duration_alignment_error_s: float
    duration_alignment_error_fraction: float
    qpos_dim: int
    qvel_dim: int
    max_root_speed_m_s: float
    max_isolated_root_speed_m_s: float
    median_root_height_m: float
    p10_root_height_m: float
    min_root_height_m: float
    max_nonroot_qpos_step: float
    max_isolated_nonroot_qpos_jump: float
    max_right_hand_speed_m_s: float
    max_right_hand_angular_speed_rad_s: float
    right_hand_path_length_m: float
    max_right_hand_displacement_m: float
    max_right_wrist_qpos_step: float
    right_palm_proxy_site: str
    max_right_palm_proxy_speed_m_s: float
    max_right_palm_proxy_angular_speed_rad_s: float
    cache_contains_racket_reference_site: bool
    racket_reference_source: str
    max_racket_reference_angular_speed_rad_s: float
    warnings: tuple[str, ...]


def inspect_event_reference(timeline: EventTimeline) -> dict[str, Any]:
    """Return reusable QC metrics for an already validated event timeline."""

    phases = timeline.phase_arrays()
    counts = {
        PHASE_NAMES[int(phase)]: int(np.sum(phases.phase_id == int(phase)))
        for phase in np.unique(phases.phase_id)
    }
    missing_phases = [name for name in PHASE_NAMES if counts.get(name, 0) <= 0]
    hard_errors = [f"event phase {name!r} has no frames" for name in missing_phases]
    return {
        "schema_version": "forehand_clear_event_qc_v1",
        "passed": not hard_errors,
        "hard_errors": hard_errors,
        "warnings": [],
        "impact_frame": int(timeline.impact.frame),
        "impact_time_s": float(timeline.impact.time_s),
        "impact_confidence": float(timeline.impact.confidence),
        "phase_frame_counts": counts,
        "phase_global_monotone": bool(np.all(np.diff(phases.phase_global) >= -1e-7)),
    }


def inspect_racket_reference(
    reference: RacketReference,
    *,
    max_linear_speed_m_s: float = 80.0,
    max_angular_speed_rad_s: float = 250.0,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """QC an independent racket reference without changing canonical cache gates."""

    metrics = racket_reference_metrics(reference)
    hard_errors: list[str] = []
    warnings: list[str] = []
    if metrics["min_confidence"] < float(min_confidence):
        hard_errors.append(
            f"racket confidence {metrics['min_confidence']:.3f} is below {float(min_confidence):.3f}"
        )
    if metrics["max_linear_speed_m_s"] > float(max_linear_speed_m_s):
        warnings.append(
            f"racket linear speed spike {metrics['max_linear_speed_m_s']:.3f} m/s"
        )
    if metrics["max_angular_speed_rad_s"] > float(max_angular_speed_rad_s):
        warnings.append(
            f"racket angular speed spike {metrics['max_angular_speed_rad_s']:.3f} rad/s"
        )
    return {
        "schema_version": "forehand_clear_racket_reference_qc_v1",
        "passed": not hard_errors,
        "clean_passed": not hard_errors and not warnings,
        "hard_errors": hard_errors,
        "warnings": warnings,
        "source": reference.source,
        **metrics,
    }


def validate_session_split(
    train_records: list[Any] | tuple[Any, ...],
    val_records: list[Any] | tuple[Any, ...],
) -> dict[str, Any]:
    """Fail closed when subject/session recording blocks cross train and validation."""

    train_sessions, train_missing = _session_keys(train_records)
    val_sessions, val_missing = _session_keys(val_records)
    overlap = sorted(train_sessions & val_sessions)
    hard_errors = []
    if train_missing:
        hard_errors.append(f"train records missing subject_id/session_id: {train_missing}")
    if val_missing:
        hard_errors.append(f"validation records missing subject_id/session_id: {val_missing}")
    if overlap:
        hard_errors.append(f"train/validation session leakage: {overlap}")
    return {
        "schema_version": "forehand_clear_session_split_qc_v1",
        "passed": not hard_errors,
        "hard_errors": hard_errors,
        "train_sessions": [list(value) for value in sorted(train_sessions)],
        "validation_sessions": [list(value) for value in sorted(val_sessions)],
        "overlap": [list(value) for value in overlap],
    }


def inspect_event_racket_bundle(
    bundle: Any,
    *,
    min_racket_confidence: float = 0.0,
) -> dict[str, Any]:
    """Compose event and racket QC for a v2 ReferenceBundle-like object."""

    if getattr(bundle, "events", None) is None or getattr(bundle, "racket", None) is None:
        return {
            "schema_version": "forehand_clear_event_racket_bundle_qc_v1",
            "passed": False,
            "hard_errors": ["bundle has no event-aware racket reference"],
        }
    event_report = inspect_event_reference(bundle.events)
    racket_report = inspect_racket_reference(
        bundle.racket,
        min_confidence=float(min_racket_confidence),
    )
    hard_errors = [*event_report["hard_errors"], *racket_report["hard_errors"]]
    return {
        "schema_version": "forehand_clear_event_racket_bundle_qc_v1",
        "passed": not hard_errors,
        "clean_passed": not hard_errors and not racket_report["warnings"],
        "hard_errors": hard_errors,
        "warnings": list(racket_report["warnings"]),
        "event": event_report,
        "racket": racket_report,
        "content_fingerprint": getattr(bundle, "content_fingerprint", None),
    }


def _session_keys(records: list[Any] | tuple[Any, ...]) -> tuple[set[tuple[str, str]], list[int]]:
    keys: set[tuple[str, str]] = set()
    missing: list[int] = []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            provenance = record.get("provenance", record)
        else:
            provenance = getattr(record, "provenance", None)
        if not isinstance(provenance, dict):
            missing.append(index)
            continue
        subject = str(provenance.get("subject_id", "")).strip()
        session = str(provenance.get("session_id", "")).strip()
        if not subject or not session:
            missing.append(index)
            continue
        keys.add((subject, session))
    return keys, missing


def _validate_variant(value: str, *, name: str) -> str:
    variant = str(value).strip()
    if not _SAFE_VARIANT.fullmatch(variant) or variant in {".", ".."}:
        raise ValueError(
            f"{name} must be one safe namespace component matching "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return variant


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_variant_paths(
    dataset_root: str | Path,
    *,
    source_variant: str,
    cache_variant: str,
) -> tuple[Path, Path, Path, str]:
    """Resolve namespace paths without permitting traversal or symlink escape."""
    root = Path(dataset_root).expanduser().resolve()
    # A source may be given bare ("raw_smooth_v1", resolved under temp/) or
    # bucket-qualified ("wham/optimized_wham").  ChinaJump has no temp/ tree at
    # all, so the bucket form is the only way to QC its retarget source.
    source_bucket = "temp"
    if "/" in source_variant:
        source_bucket, _, source_variant = source_variant.partition("/")
        source_bucket = _validate_variant(source_bucket, name="source_bucket")
        if source_bucket not in {"temp", "wham"}:
            raise ValueError("source bucket must be 'temp' or 'wham'")
    source_variant = _validate_variant(source_variant, name="source_variant")
    cache_variant = _validate_variant(cache_variant, name="cache_variant")
    if source_variant in _LEGACY_SOURCE_VARIANTS:
        source_dir = root / "wham" / "initiall_wham"
        source_suffix = "__original.npz"
    else:
        source_dir = root / source_bucket / source_variant
        source_suffix = ".npz"
    cache_dir = root / "muscle_trajectory" / cache_variant
    expected_source = source_dir
    expected_cache = cache_dir
    resolved_source = source_dir.resolve()
    resolved_cache = cache_dir.resolve()
    if not _is_within(resolved_source, root):
        raise ValueError(f"source namespace escapes dataset root: {resolved_source}")
    if not _is_within(resolved_cache, root):
        raise ValueError(f"cache namespace escapes dataset root: {resolved_cache}")
    if resolved_source != expected_source or resolved_cache != expected_cache:
        raise ValueError(
            "source/cache namespace must not be a symlink alias: "
            f"source={resolved_source}, cache={resolved_cache}"
        )
    return root, resolved_source, resolved_cache, source_suffix


def inspect_canonical_dataset(
    dataset_root: str | Path,
    *,
    source_variant: str = "initial",
    cache_variant: str = "raw",
    action: str = DEFAULT_ACTION,
) -> dict[str, Any]:
    spec = resolve(action)
    train_motions = spec.train_motions
    val_motions = spec.val_motions
    root, source_dir, cache_dir, source_suffix = _resolve_variant_paths(
        dataset_root,
        source_variant=source_variant,
        cache_variant=cache_variant,
    )
    hard_errors: list[str] = []
    rows: list[MotionQC] = []
    try:
        spec.validate()
    except ValueError as exc:
        hard_errors.append(str(exc))
    expected_rows = len(train_motions) + len(val_motions)

    for motion in (*train_motions, *val_motions):
        source_path = source_dir / f"{motion}{source_suffix}"
        cache_path = cache_dir / f"{motion}.npz"
        if not source_path.is_file() or not cache_path.is_file():
            hard_errors.append(
                f"{motion}: missing source ({source_variant}) or cache ({cache_variant})"
            )
            continue
        try:
            row, errors = _inspect_motion(motion, source_path, cache_path)
        except Exception as exc:
            hard_errors.append(f"{motion}: unreadable data: {exc}")
            continue
        rows.append(row)
        hard_errors.extend(errors)

    warnings = [f"{row.motion}: {warning}" for row in rows for warning in row.warnings]
    passed = not hard_errors and len(rows) == expected_rows
    schema_action = "forehand_clear" if spec.slug == "forehand_clear" else spec.slug
    return {
        "schema_version": f"{schema_action}_data_qc_v3",
        "action": spec.action_id,
        "action_slug": spec.slug,
        "dataset_root": str(root),
        "source_variant": source_variant,
        "cache_variant": cache_variant,
        "resolved_source_dir": str(source_dir),
        "resolved_cache_dir": str(cache_dir),
        "expected_motion_count": expected_rows,
        "train_motions": list(train_motions),
        "validation_motions": list(val_motions),
        "expected_source_fps": 60.0,
        "expected_cache_fps": 100.0,
        "expected_qpos_dim": 89,
        "expected_qvel_dim": 88,
        "duration_alignment_warning_threshold_s": MAX_DURATION_ALIGNMENT_ERROR_S,
        "duration_alignment_warning_threshold_fraction": MAX_DURATION_ALIGNMENT_ERROR_FRACTION,
        "right_hand_angular_speed_warning_threshold_rad_s": MAX_RIGHT_HAND_ANGULAR_SPEED_RAD_S,
        "isolated_root_speed_warning_threshold_m_s": MAX_ISOLATED_ROOT_SPEED_M_S,
        "isolated_root_velocity_change_threshold_m_s": MIN_ISOLATED_ROOT_VELOCITY_CHANGE_M_S,
        "absolute_root_speed_safety_limit_m_s": MAX_ABSOLUTE_ROOT_SPEED_M_S,
        "minimum_median_root_height_m": MIN_MEDIAN_ROOT_HEIGHT_M,
        "minimum_right_hand_path_length_m": MIN_RIGHT_HAND_PATH_LENGTH_M,
        "minimum_right_hand_max_displacement_m": MIN_RIGHT_HAND_MAX_DISPLACEMENT_M,
        "nonroot_qpos_step_warning_threshold_rad": MAX_NONROOT_QPOS_STEP_RAD,
        "isolated_step_neighbor_change_threshold_rad": MIN_ISOLATED_STEP_NEIGHBOR_CHANGE_RAD,
        "reference_contract": {
            "right_palm_proxy_site": "right_hand_mimic",
            "right_wrist_joint_names": ["pro_sup_r", "deviation_r", "flexion_r"],
            "racket_reference_source_when_cache_site_absent": (
                "derived_from_right_hand_rigid_attachment_at_runtime"
            ),
            "frequency_semantics": (
                "gmr_config.target_fps=60 aligns source SMPL/IK frames; "
                "extend_motion resamples the retargeted result to the 100 Hz control cache"
            ),
            "note": (
                "Canonical human GMR caches do not contain a racket/stringbed site. "
                "Stage-2 derives its rigid racket pose from the right-hand attachment; "
                "the cache QC therefore audits hand pose/orientation and wrist joints."
            ),
        },
        "hard_errors": hard_errors,
        "warnings": warnings,
        "motions": [asdict(row) for row in rows],
        "passed": passed,
        "clean_passed": passed and not warnings,
    }


def _inspect_motion(motion: str, source_path: Path, cache_path: Path) -> tuple[MotionQC, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    with np.load(source_path, allow_pickle=True) as source:
        source_fps_values = [float(source[name]) for name in ("mocap_framerate", "mocap_frame_rate") if name in source]
        if not source_fps_values:
            errors.append(f"{motion}: source has no FPS field")
            source_fps = math.nan
        else:
            source_fps = source_fps_values[0]
            if any(not np.isclose(value, 60.0) for value in source_fps_values):
                errors.append(f"{motion}: source FPS must be 60, got {source_fps_values}")
        source_frames = int(np.asarray(source["poses"]).shape[0])
    with np.load(cache_path, allow_pickle=True) as cache:
        for field in (
            "qpos",
            "qvel",
            "site_xpos",
            "site_xmat",
            "site_names",
            "joint_names",
            "jnt_type",
            "frequency",
        ):
            if field not in cache:
                errors.append(f"{motion}: cache missing {field}")
        required_fields = (
            "qpos", "qvel", "site_xpos", "site_xmat", "site_names",
            "joint_names", "jnt_type", "frequency",
        )
        if errors and not all(field in cache for field in required_fields):
            return MotionQC(
                motion=motion,
                source_frames=source_frames,
                cache_frames=0,
                source_fps=source_fps,
                cache_fps=math.nan,
                source_duration_s=math.nan,
                cache_duration_s=math.nan,
                expected_cache_frames_from_source=0,
                cache_frame_alignment_delta=0,
                duration_alignment_error_s=math.nan,
                duration_alignment_error_fraction=math.nan,
                qpos_dim=0,
                qvel_dim=0,
                max_root_speed_m_s=math.nan,
                max_isolated_root_speed_m_s=math.nan,
                median_root_height_m=math.nan,
                p10_root_height_m=math.nan,
                min_root_height_m=math.nan,
                max_nonroot_qpos_step=math.nan,
                max_isolated_nonroot_qpos_jump=math.nan,
                max_right_hand_speed_m_s=math.nan,
                max_right_hand_angular_speed_rad_s=math.nan,
                right_hand_path_length_m=math.nan,
                max_right_hand_displacement_m=math.nan,
                max_right_wrist_qpos_step=math.nan,
                right_palm_proxy_site="right_hand_mimic",
                max_right_palm_proxy_speed_m_s=math.nan,
                max_right_palm_proxy_angular_speed_rad_s=math.nan,
                cache_contains_racket_reference_site=False,
                racket_reference_source="derived_from_right_hand_rigid_attachment_at_runtime",
                max_racket_reference_angular_speed_rad_s=math.nan,
                warnings=(),
            ), errors
        qpos = np.asarray(cache["qpos"], dtype=np.float64)
        qvel = np.asarray(cache["qvel"], dtype=np.float64)
        sites = np.asarray(cache["site_xpos"], dtype=np.float64)
        site_xmat = np.asarray(cache["site_xmat"], dtype=np.float64)
        site_names = [str(name) for name in np.asarray(cache.get("site_names", []))]
        joint_names = [str(name) for name in np.asarray(cache.get("joint_names", []))]
        joint_types = np.asarray(cache.get("jnt_type", []), dtype=np.int32).reshape(-1)
        cache_fps = float(cache["frequency"])
    cache_frames = int(qpos.shape[0])
    source_duration_s = (
        float(source_frames) / source_fps
        if source_frames >= 0 and math.isfinite(source_fps) and source_fps > 0.0
        else math.nan
    )
    cache_duration_s = (
        float(cache_frames) / cache_fps
        if cache_frames >= 0 and math.isfinite(cache_fps) and cache_fps > 0.0
        else math.nan
    )
    expected_cache_frames = (
        round(float(source_frames) * cache_fps / source_fps)
        if math.isfinite(source_fps)
        and source_fps > 0.0
        and math.isfinite(cache_fps)
        and cache_fps > 0.0
        else 0
    )
    cache_frame_delta = cache_frames - expected_cache_frames
    duration_alignment_error_s = (
        abs(cache_duration_s - source_duration_s)
        if math.isfinite(source_duration_s) and math.isfinite(cache_duration_s)
        else math.nan
    )
    duration_alignment_error_fraction = (
        duration_alignment_error_s / source_duration_s
        if math.isfinite(duration_alignment_error_s) and source_duration_s > 0.0
        else math.nan
    )
    if qpos.ndim != 2 or qpos.shape[1] != 89:
        errors.append(f"{motion}: qpos shape must be [T,89], got {qpos.shape}")
    if qvel.ndim != 2 or qvel.shape[1] != 88:
        errors.append(f"{motion}: qvel shape must be [T,88], got {qvel.shape}")
    if not np.isclose(cache_fps, 100.0):
        errors.append(f"{motion}: cache frequency must be 100, got {cache_fps}")
    if (
        cache_frames < 2
        or qvel.shape[0] != cache_frames
        or sites.shape[0] != cache_frames
        or site_xmat.shape[0] != cache_frames
    ):
        errors.append(f"{motion}: cache arrays have inconsistent/insufficient frame counts")
    if not all(np.isfinite(array).all() for array in (qpos, qvel, sites, site_xmat)):
        errors.append(f"{motion}: cache contains NaN/Inf")

    root_steps = np.diff(qpos[:, :3], axis=0)
    root_speed = _max_norm_step(qpos[:, :3], cache_fps)
    isolated_root_speed = _max_isolated_vector_step_speed(
        root_steps,
        frequency=cache_fps,
        neighbor_velocity_change_threshold=MIN_ISOLATED_ROOT_VELOCITY_CHANGE_M_S,
    )
    root_height = qpos[:, 2]
    median_root_height = float(np.median(root_height))
    p10_root_height = float(np.percentile(root_height, 10.0))
    min_root_height = float(np.min(root_height))
    nonroot_steps = np.diff(qpos[:, 7:], axis=0)
    nonroot_step = (
        float(np.max(np.abs(nonroot_steps))) if cache_frames > 1 else math.nan
    )
    isolated_nonroot_jump = _max_isolated_step(
        nonroot_steps,
        neighbor_change_threshold=MIN_ISOLATED_STEP_NEIGHBOR_CHANGE_RAD,
    )
    right_hand_speed = math.nan
    right_hand_angular_speed = math.nan
    right_hand_path_length = math.nan
    max_right_hand_displacement = math.nan
    if "right_hand_mimic" in site_names:
        right_hand_site_index = site_names.index("right_hand_mimic")
        right_hand_positions = sites[:, right_hand_site_index]
        right_hand_speed = _max_norm_step(right_hand_positions, cache_fps)
        right_hand_angular_speed = _max_rotation_angular_speed(
            site_xmat[:, right_hand_site_index], cache_fps
        )
        right_hand_path_length = float(
            np.sum(np.linalg.norm(np.diff(right_hand_positions, axis=0), axis=-1))
        )
        max_right_hand_displacement = float(
            np.max(
                np.linalg.norm(
                    right_hand_positions - right_hand_positions[0],
                    axis=-1,
                )
            )
        )
    else:
        errors.append(f"{motion}: cache missing right_hand_mimic site")
    wrist_names = ("pro_sup_r", "deviation_r", "flexion_r")
    wrist_step = math.nan
    try:
        qpos_slices = _joint_qpos_slices(joint_names, joint_types, int(qpos.shape[1]))
        missing_wrist = [name for name in wrist_names if name not in qpos_slices]
        if missing_wrist:
            errors.append(f"{motion}: cache missing right-wrist joints {missing_wrist}")
        else:
            wrist_indices = np.concatenate(
                [np.arange(*qpos_slices[name], dtype=np.int32) for name in wrist_names]
            )
            wrist_step = (
                float(np.max(np.abs(np.diff(qpos[:, wrist_indices], axis=0))))
                if cache_frames > 1
                else math.nan
            )
    except ValueError as exc:
        errors.append(f"{motion}: invalid joint/qpos schema: {exc}")

    racket_site_present = any(
        "racket" in name.lower() or "stringbed" in name.lower()
        for name in site_names
    )
    racket_reference_source = (
        "cache_site"
        if racket_site_present
        else "derived_from_right_hand_rigid_attachment_at_runtime"
    )
    if isolated_root_speed > MAX_ISOLATED_ROOT_SPEED_M_S:
        warnings.append(
            "isolated root position jump "
            f"{isolated_root_speed:.3f} m/s "
            f"(raw max speed {root_speed:.3f} m/s)"
        )
    elif root_speed > MAX_ABSOLUTE_ROOT_SPEED_M_S:
        warnings.append(
            "root speed exceeds absolute safety limit "
            f"{root_speed:.3f} m/s"
        )
    if median_root_height < MIN_MEDIAN_ROOT_HEIGHT_M:
        warnings.append(
            "persistent low-root/posture-collapse: "
            f"median root height {median_root_height:.3f} m < "
            f"{MIN_MEDIAN_ROOT_HEIGHT_M:.3f} m "
            f"(p10 {p10_root_height:.3f} m, min {min_root_height:.3f} m)"
        )
    if isolated_nonroot_jump > MAX_NONROOT_QPOS_STEP_RAD:
        warnings.append(
            "isolated non-root qpos jump "
            f"{isolated_nonroot_jump:.3f} rad "
            f"(raw max step {nonroot_step:.3f} rad)"
        )
    if right_hand_speed > 20.0:
        warnings.append(f"right-hand speed spike {right_hand_speed:.3f} m/s")
    if (
        right_hand_path_length < MIN_RIGHT_HAND_PATH_LENGTH_M
        or max_right_hand_displacement < MIN_RIGHT_HAND_MAX_DISPLACEMENT_M
    ):
        warnings.append(
            "insufficient forehand swing amplitude: "
            f"right-hand path {right_hand_path_length:.3f} m "
            f"(minimum {MIN_RIGHT_HAND_PATH_LENGTH_M:.3f} m), "
            f"max displacement {max_right_hand_displacement:.3f} m "
            f"(minimum {MIN_RIGHT_HAND_MAX_DISPLACEMENT_M:.3f} m)"
        )
    if wrist_step > 0.45:
        warnings.append(f"right-wrist qpos step spike {wrist_step:.3f} rad")
    if (
        duration_alignment_error_s > MAX_DURATION_ALIGNMENT_ERROR_S
        or duration_alignment_error_fraction > MAX_DURATION_ALIGNMENT_ERROR_FRACTION
    ):
        warnings.append(
            "source/cache duration misalignment "
            f"{duration_alignment_error_s:.3f} s "
            f"({100.0 * duration_alignment_error_fraction:.1f}%, "
            f"cache frame delta {cache_frame_delta:+d})"
        )
    if right_hand_angular_speed > MAX_RIGHT_HAND_ANGULAR_SPEED_RAD_S:
        warnings.append(
            "right-hand/racket-reference angular speed spike "
            f"{right_hand_angular_speed:.3f} rad/s"
        )
    return MotionQC(
        motion=motion,
        source_frames=source_frames,
        cache_frames=cache_frames,
        source_fps=source_fps,
        cache_fps=cache_fps,
        source_duration_s=source_duration_s,
        cache_duration_s=cache_duration_s,
        expected_cache_frames_from_source=expected_cache_frames,
        cache_frame_alignment_delta=cache_frame_delta,
        duration_alignment_error_s=duration_alignment_error_s,
        duration_alignment_error_fraction=duration_alignment_error_fraction,
        qpos_dim=int(qpos.shape[1]) if qpos.ndim == 2 else 0,
        qvel_dim=int(qvel.shape[1]) if qvel.ndim == 2 else 0,
        max_root_speed_m_s=root_speed,
        max_isolated_root_speed_m_s=isolated_root_speed,
        median_root_height_m=median_root_height,
        p10_root_height_m=p10_root_height,
        min_root_height_m=min_root_height,
        max_nonroot_qpos_step=nonroot_step,
        max_isolated_nonroot_qpos_jump=isolated_nonroot_jump,
        max_right_hand_speed_m_s=right_hand_speed,
        max_right_hand_angular_speed_rad_s=right_hand_angular_speed,
        right_hand_path_length_m=right_hand_path_length,
        max_right_hand_displacement_m=max_right_hand_displacement,
        max_right_wrist_qpos_step=wrist_step,
        right_palm_proxy_site="right_hand_mimic",
        max_right_palm_proxy_speed_m_s=right_hand_speed,
        max_right_palm_proxy_angular_speed_rad_s=right_hand_angular_speed,
        cache_contains_racket_reference_site=racket_site_present,
        racket_reference_source=racket_reference_source,
        max_racket_reference_angular_speed_rad_s=right_hand_angular_speed,
        warnings=tuple(warnings),
    ), errors


def _max_norm_step(values: np.ndarray, frequency: float) -> float:
    if values.shape[0] < 2:
        return math.nan
    return float(np.max(np.linalg.norm(np.diff(values, axis=0), axis=-1)) * frequency)


def _max_isolated_step(
    steps: np.ndarray,
    *,
    neighbor_change_threshold: float,
) -> float:
    """Return the largest step surrounded by sharp changes on both sides.

    A real one-frame jump has a large velocity step whose velocity changes
    abruptly both entering and leaving the frame.  A continuous fast ramp may
    have a large position step but at least one neighbouring velocity remains
    similar.  Boundary steps use their single available neighbour so short
    synthetic/corrupt sequences still fail closed.
    """
    values = np.asarray(steps, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"steps must be [T-1,D], got {values.shape}")
    if values.shape[0] == 0:
        return math.nan
    if values.shape[0] == 1:
        return float(np.max(np.abs(values)))

    changes = np.abs(np.diff(values, axis=0))
    isolation = np.empty_like(values)
    isolation[0] = changes[0]
    isolation[-1] = changes[-1]
    if values.shape[0] > 2:
        isolation[1:-1] = np.minimum(changes[:-1], changes[1:])
    candidates = np.abs(values)[isolation >= float(neighbor_change_threshold)]
    return float(np.max(candidates)) if candidates.size else 0.0


def _max_isolated_vector_step_speed(
    steps: np.ndarray,
    *,
    frequency: float,
    neighbor_velocity_change_threshold: float,
) -> float:
    """Largest vector step speed entered and exited through sharp velocity changes."""
    values = np.asarray(steps, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"vector steps must be [T-1,D], got {values.shape}")
    if values.shape[0] == 0:
        return math.nan
    velocity = values * float(frequency)
    speed = np.linalg.norm(velocity, axis=-1)
    if values.shape[0] == 1:
        return float(speed[0])
    changes = np.linalg.norm(np.diff(velocity, axis=0), axis=-1)
    isolation = np.empty(values.shape[0], dtype=np.float64)
    isolation[0] = changes[0]
    isolation[-1] = changes[-1]
    if values.shape[0] > 2:
        isolation[1:-1] = np.minimum(changes[:-1], changes[1:])
    candidates = speed[
        isolation >= float(neighbor_velocity_change_threshold)
    ]
    return float(np.max(candidates)) if candidates.size else 0.0


def _max_rotation_angular_speed(rotation_matrices: np.ndarray, frequency: float) -> float:
    """Maximum SO(3) geodesic step speed for a persisted site orientation."""
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.shape[0] < 2:
        return math.nan
    if matrices.shape[-1:] == (9,):
        matrices = matrices.reshape(*matrices.shape[:-1], 3, 3)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"site rotation matrices must end in [3,3], got {matrices.shape}")
    relative = matrices[1:] @ np.swapaxes(matrices[:-1], -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.max(np.arccos(cosine)) * float(frequency))


def _joint_qpos_slices(
    joint_names: list[str],
    joint_types: np.ndarray,
    qpos_dim: int,
) -> dict[str, tuple[int, int]]:
    """Reconstruct MuJoCo qpos addresses from persisted joint types."""
    if len(joint_names) != int(joint_types.size):
        raise ValueError(
            f"joint_names/jnt_type length mismatch: {len(joint_names)} vs {joint_types.size}"
        )
    widths = {0: 7, 1: 4, 2: 1, 3: 1}
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    for name, joint_type in zip(joint_names, joint_types, strict=True):
        if int(joint_type) not in widths:
            raise ValueError(f"unsupported MuJoCo joint type {int(joint_type)} for {name}")
        width = widths[int(joint_type)]
        result[name] = (cursor, cursor + width)
        cursor += width
    if cursor != int(qpos_dim):
        raise ValueError(f"joint qpos widths sum to {cursor}, expected {qpos_dim}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=action_choices(),
        default=DEFAULT_ACTION,
        help="selects the clip split; also supplies dataset/variant defaults",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=None,
        help="defaults to the selected action's dataset root",
    )
    parser.add_argument(
        "--source-variant",
        "--source_variant",
        dest="source_variant",
        default=None,
        help="initial (legacy wham/initiall_wham) or a safe temp/<variant> namespace",
    )
    parser.add_argument(
        "--cache-variant",
        "--cache_variant",
        dest="cache_variant",
        default=None,
        help="safe muscle_trajectory/<variant> cache namespace",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require-clean",
        "--require_clean",
        dest="require_clean",
        action="store_true",
        help="also fail on spike warnings",
    )
    args = parser.parse_args()
    spec = resolve(args.action)
    report = inspect_canonical_dataset(
        args.dataset_root or f"datasets/{spec.action_id}",
        source_variant=args.source_variant or spec.source_namespace,
        cache_variant=args.cache_variant or spec.cache_variant,
        action=args.action,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if (not report["passed"] or (args.require_clean and not report["clean_passed"])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
