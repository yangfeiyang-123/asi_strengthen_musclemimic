"""Host-side exact motion/step lookup for event-aware reference caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    normalize_relative_motion_path,
    stable_motion_uid,
)

EVENT_REFERENCE_BANK_VERSION = "forehand_clear_event_reference_bank_v1"
EVENT_LOOKUP_FIELDS = (
    "phase_global",
    "phase_id",
    "phase_local",
    "time_to_impact_s",
    "time_from_impact_s",
    "impact_flag",
    "reference_confidence",
)
RACKET_LOOKUP_FIELDS = (
    "racket_position_world",
    "racket_quaternion_world",
    "racket_linear_velocity_world",
    "racket_angular_velocity_world",
    "stringbed_normal_world",
    "stringbed_center_world",
    "racket_reference_confidence",
)


@dataclass(frozen=True)
class EventReferenceEntry:
    traj_no: int
    motion_uid: int
    motion_path: str
    cache_path: Path
    cache_sha256: str
    reference_bundle_content_fingerprint: str
    reference_fps: float
    control_dt: float
    effective_ref_stride: float
    arrays: dict[str, np.ndarray]

    @property
    def num_frames(self) -> int:
        return int(self.arrays["phase_id"].shape[0])


@dataclass(frozen=True)
class EventReferenceLookup:
    manifest_path: Path
    entries: tuple[EventReferenceEntry, ...]
    manifest: dict[str, Any]
    fingerprint: str

    def validate_control_dt(self, control_dt: float) -> float:
        """Require every cache to use the policy environment's control period."""

        runtime_dt = float(control_dt)
        if not np.isfinite(runtime_dt) or runtime_dt <= 0.0:
            raise ValueError("event reference runtime control_dt must be finite and positive")
        mismatches = [
            (entry.traj_no, entry.control_dt)
            for entry in self.entries
            if not np.isclose(entry.control_dt, runtime_dt, atol=1e-8, rtol=0.0)
        ]
        if mismatches:
            raise ValueError(
                "event reference cache control_dt differs from policy runtime: "
                f"runtime={runtime_dt:g} caches={mismatches}"
            )
        return runtime_dt

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        motion_identity_map: MotionIdentityMap | None = None,
    ) -> EventReferenceLookup:
        manifest_path = Path(path).resolve()
        payload = load_json_strict(manifest_path)
        if not isinstance(payload, dict):
            raise ValueError("event reference bank manifest must contain an object")
        if payload.get("schema_version") != EVENT_REFERENCE_BANK_VERSION:
            raise ValueError("unsupported event reference bank schema")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("event reference bank requires non-empty entries")
        entries = tuple(_load_entry(manifest_path.parent, raw_entry) for raw_entry in raw_entries)
        _validate_unique_entries(entries)
        if motion_identity_map is not None:
            _validate_identity_coverage(entries, motion_identity_map)
        expected = _manifest_fingerprint(payload)
        if payload.get("manifest_fingerprint") != expected:
            raise ValueError("event reference bank manifest_fingerprint mismatch")
        return cls(
            manifest_path=manifest_path,
            entries=entries,
            manifest=payload,
            fingerprint=expected,
        )

    def lookup_batch(
        self,
        *,
        traj_no: np.ndarray,
        subtraj_step_no: np.ndarray,
        motion_uid: np.ndarray | None = None,
        include_racket: bool = False,
    ) -> dict[str, np.ndarray]:
        trajectories = np.asarray(traj_no, dtype=np.int32)
        steps = np.asarray(subtraj_step_no, dtype=np.int64)
        if trajectories.ndim != 1 or steps.shape != trajectories.shape:
            raise ValueError("event lookup traj_no/subtraj_step_no must be same-shape rank-1 arrays")
        if np.any(steps < 0):
            raise ValueError("event lookup subtraj_step_no must be non-negative")
        uids = None if motion_uid is None else np.asarray(motion_uid, dtype=np.int64)
        if uids is not None and uids.shape != trajectories.shape:
            raise ValueError("event lookup motion_uid shape differs from traj_no")
        by_traj = {entry.traj_no: entry for entry in self.entries}
        unknown = sorted({int(value) for value in np.unique(trajectories)} - set(by_traj))
        if unknown:
            raise ValueError(f"event lookup has no exact trajectory entries for {unknown}")
        fields = EVENT_LOOKUP_FIELDS + (RACKET_LOOKUP_FIELDS if include_racket else ())
        result: dict[str, list[np.ndarray]] = {field: [] for field in fields}
        frame_indices = np.empty(trajectories.shape, dtype=np.int32)
        for row, (traj, step) in enumerate(zip(trajectories, steps, strict=True)):
            entry = by_traj[int(traj)]
            if uids is not None and int(uids[row]) != entry.motion_uid:
                raise ValueError(
                    "event lookup stable motion UID differs from local trajectory mapping: "
                    f"traj_no={int(traj)} supplied={int(uids[row])} expected={entry.motion_uid}"
                )
            frame = int(np.rint(float(step) * entry.effective_ref_stride))
            if frame < 0 or frame >= entry.num_frames:
                raise ValueError(
                    "event lookup frame lies outside cache; extrapolation/clipping is forbidden: "
                    f"traj_no={int(traj)} step={int(step)} frame={frame} frames={entry.num_frames}"
                )
            frame_indices[row] = frame
            for field in fields:
                result[field].append(np.asarray(entry.arrays[field][frame]))
        output = {field: np.asarray(values) for field, values in result.items()}
        output["event_reference_frame"] = frame_indices
        return output


def select_transition_coordinates(
    traj_no: Any,
    subtraj_step_no: Any,
    done: Any,
    *,
    final_traj_no: Any | None,
    final_subtraj_step_no: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose pre-reset coordinates for every transition, including terminal lanes."""

    trajectories = np.asarray(traj_no, dtype=np.int32)
    steps = np.asarray(subtraj_step_no, dtype=np.int32)
    terminal = np.asarray(done, dtype=bool)
    if trajectories.shape != steps.shape or trajectories.shape != terminal.shape:
        raise ValueError("transition traj/step/done shapes differ")
    if np.any(terminal) and (final_traj_no is None or final_subtraj_step_no is None):
        raise ValueError(
            "terminal event lookup requires final_traj_no and final_subtraj_step_no; "
            "post-reset coordinates cannot be substituted"
        )
    if final_traj_no is None:
        return trajectories, steps
    final_trajectories = np.asarray(final_traj_no, dtype=np.int32)
    final_steps = np.asarray(final_subtraj_step_no, dtype=np.int32)
    if final_trajectories.shape != trajectories.shape or final_steps.shape != steps.shape:
        raise ValueError("final transition coordinate shapes differ")
    return (
        np.where(terminal, final_trajectories, trajectories).astype(np.int32),
        np.where(terminal, final_steps, steps).astype(np.int32),
    )


def write_event_reference_bank_manifest(
    path: str | Path,
    *,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a content-addressed bank from explicit trajectory/cache entries."""

    manifest_path = Path(path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_event_reference_bank_payload(manifest_path, entries=entries)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Re-open through the strict loader and all cache validators before return.
    EventReferenceLookup.from_manifest(manifest_path)
    return manifest_path


def build_event_reference_bank_payload(
    output_path: str | Path,
    *,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate entries and return the exact manifest payload without writing it."""

    manifest_path = Path(output_path).resolve()
    normalized: list[dict[str, Any]] = []
    for raw in entries:
        cache = Path(str(raw["tracking_cache_npz"]))
        absolute = cache if cache.is_absolute() else (manifest_path.parent / cache)
        absolute = absolute.resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"event tracking cache does not exist: {absolute}")
        relative = os.path.relpath(absolute, manifest_path.parent)
        with np.load(absolute, allow_pickle=False) as payload:
            fingerprint = _scalar_string(payload, "reference_bundle_content_fingerprint")
            reference_fps = _scalar_float(payload, "reference_fps")
            control_dt = _scalar_float(payload, "control_dt")
            effective_ref_stride = _scalar_float(payload, "effective_ref_stride")
        normalized.append(
            {
                "traj_no": int(raw["traj_no"]),
                "motion_uid": int(raw["motion_uid"]),
                "motion_path": normalize_relative_motion_path(str(raw["motion_path"])),
                "tracking_cache_npz": relative,
                "cache_sha256": _file_sha256(absolute),
                "reference_bundle_content_fingerprint": fingerprint,
                "reference_fps": reference_fps,
                "control_dt": control_dt,
                "effective_ref_stride": effective_ref_stride,
            }
        )
    payload = {
        "schema_version": EVENT_REFERENCE_BANK_VERSION,
        "entries": normalized,
    }
    payload["manifest_fingerprint"] = _manifest_fingerprint(payload)
    validated = tuple(_load_entry(manifest_path.parent, entry) for entry in normalized)
    _validate_unique_entries(validated)
    return payload


def _load_entry(base: Path, raw: object) -> EventReferenceEntry:
    if not isinstance(raw, Mapping):
        raise ValueError("event reference bank entries must be objects")
    required = {
        "traj_no",
        "motion_uid",
        "motion_path",
        "tracking_cache_npz",
        "cache_sha256",
        "reference_bundle_content_fingerprint",
        "reference_fps",
        "control_dt",
        "effective_ref_stride",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"event reference bank entry is missing {missing}")
    cache_path = (base / str(raw["tracking_cache_npz"])).resolve()
    if not cache_path.is_file() or _file_sha256(cache_path) != str(raw["cache_sha256"]):
        raise ValueError(f"event reference cache hash mismatch: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as cache:
        required_arrays = EVENT_LOOKUP_FIELDS + RACKET_LOOKUP_FIELDS
        missing_arrays = sorted(set(required_arrays) - set(cache.files))
        if missing_arrays:
            raise ValueError(f"event reference cache lacks required arrays {missing_arrays}")
        arrays = {field: np.asarray(cache[field]) for field in required_arrays}
        reference_fps = _scalar_float(cache, "reference_fps")
        control_dt = _scalar_float(cache, "control_dt")
        stride = _scalar_float(cache, "effective_ref_stride")
        bundle_fingerprint = _scalar_string(cache, "reference_bundle_content_fingerprint")
        cache_motion_path = normalize_relative_motion_path(_scalar_string(cache, "reference_motion_path"))
        cache_motion_uid = _scalar_integer(cache, "reference_motion_uid")
    motion_path = normalize_relative_motion_path(str(raw["motion_path"]))
    motion_uid = int(raw["motion_uid"])
    if motion_uid != stable_motion_uid(motion_path):
        raise ValueError("event bank motion_uid does not match its stable motion_path identity")
    if cache_motion_path != motion_path or cache_motion_uid != motion_uid:
        raise ValueError(
            "event cache motion identity differs from bank manifest: "
            f"cache=({cache_motion_path!r}, {cache_motion_uid}) "
            f"bank=({motion_path!r}, {motion_uid})"
        )
    if bundle_fingerprint != str(raw["reference_bundle_content_fingerprint"]):
        raise ValueError("event cache reference bundle fingerprint differs from bank manifest")
    for field, cached, declared in (
        ("reference_fps", reference_fps, float(raw["reference_fps"])),
        ("control_dt", control_dt, float(raw["control_dt"])),
        ("effective_ref_stride", stride, float(raw["effective_ref_stride"])),
    ):
        if not np.isfinite(declared) or not np.isclose(cached, declared, atol=1e-8, rtol=0.0):
            raise ValueError(f"event cache {field} differs from bank manifest")
    lengths = {int(value.shape[0]) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("event/racket cache arrays have inconsistent frame counts")
    if reference_fps <= 0.0 or control_dt <= 0.0 or stride <= 0.0:
        raise ValueError("event cache fps/control_dt/effective_ref_stride must be positive")
    expected_stride = reference_fps * control_dt
    if not np.isclose(stride, expected_stride, atol=1e-6, rtol=0.0):
        raise ValueError("event cache effective_ref_stride differs from reference_fps * control_dt")
    for field, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"event cache field {field} contains NaN/Inf")
    return EventReferenceEntry(
        traj_no=int(raw["traj_no"]),
        motion_uid=motion_uid,
        motion_path=motion_path,
        cache_path=cache_path,
        cache_sha256=str(raw["cache_sha256"]),
        reference_bundle_content_fingerprint=bundle_fingerprint,
        reference_fps=reference_fps,
        control_dt=control_dt,
        effective_ref_stride=stride,
        arrays=arrays,
    )


def _validate_unique_entries(entries: Sequence[EventReferenceEntry]) -> None:
    trajectories = [entry.traj_no for entry in entries]
    motion_uids = [entry.motion_uid for entry in entries]
    if any(value < 0 for value in trajectories) or len(trajectories) != len(set(trajectories)):
        raise ValueError("event reference bank traj_no values must be unique and non-negative")
    if any(value < 0 for value in motion_uids) or len(motion_uids) != len(set(motion_uids)):
        raise ValueError("event reference bank motion_uid values must be unique and non-negative")


def _validate_identity_coverage(
    entries: Sequence[EventReferenceEntry],
    identity: MotionIdentityMap,
) -> None:
    if len(entries) != len(identity.motion_paths):
        raise ValueError("event reference bank does not cover every loaded motion exactly once")
    by_traj = {entry.traj_no: entry for entry in entries}
    for traj_no, (path, uid) in enumerate(zip(identity.motion_paths, identity.motion_uids, strict=True)):
        if traj_no not in by_traj:
            raise ValueError(f"event reference bank lacks traj_no={traj_no}")
        entry = by_traj[traj_no]
        if entry.motion_path != path or entry.motion_uid != int(uid):
            raise ValueError(f"event reference identity mismatch at traj_no={traj_no}")


def _manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {str(key): value for key, value in payload.items() if key != "manifest_fingerprint"}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_string(npz: Any, key: str) -> str:
    if key not in npz or np.asarray(npz[key]).size != 1:
        raise ValueError(f"event reference cache {key} must be scalar")
    value = str(np.asarray(npz[key]).reshape(-1)[0])
    if not value:
        raise ValueError(f"event reference cache {key} must be non-empty")
    return value


def _scalar_integer(npz: Any, key: str) -> int:
    if key not in npz or np.asarray(npz[key]).size != 1:
        raise ValueError(f"event reference cache {key} must be scalar")
    value = np.asarray(npz[key]).reshape(-1)[0]
    if isinstance(value, bool | np.bool_) or not np.issubdtype(np.asarray(value).dtype, np.integer):
        raise ValueError(f"event reference cache {key} must be an integer")
    return int(value)


def _scalar_float(npz: Any, key: str) -> float:
    if key not in npz or np.asarray(npz[key]).size != 1:
        raise ValueError(f"event reference cache {key} must be scalar")
    value = float(np.asarray(npz[key]).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"event reference cache {key} must be finite")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entries-json",
        required=True,
        help="Strict JSON list, or object with entries, containing explicit motion/cache mappings.",
    )
    parser.add_argument("--output", required=True, help="Output event reference bank manifest path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/cache-hash the complete contract and print it without writing output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries_path = Path(args.entries_json).resolve()
    raw = load_json_strict(entries_path)
    entries = raw.get("entries") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries JSON must be a non-empty list or contain a non-empty entries list")
    resolved_entries: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("entries JSON items must be objects")
        resolved = dict(item)
        cache = Path(str(resolved.get("tracking_cache_npz", "")))
        if not cache.is_absolute():
            resolved["tracking_cache_npz"] = str((entries_path.parent / cache).resolve())
        resolved_entries.append(resolved)
    payload = build_event_reference_bank_payload(args.output, entries=resolved_entries)
    if not args.dry_run:
        write_event_reference_bank_manifest(args.output, entries=resolved_entries)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "output": str(Path(args.output).resolve()),
                "manifest_fingerprint": payload["manifest_fingerprint"],
                "entry_count": len(payload["entries"]),
                "contract": payload if args.dry_run else None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
