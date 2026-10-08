"""Event-level forehand-clear annotations and phase construction.

The legacy tracking stack exposes a linear frame phase.  This module keeps that
ABI untouched and provides an opt-in, fail-closed event schema for research
artifacts that need a phase with stable biomechanical meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict

EVENT_SCHEMA_VERSION = "forehand_clear_events_v1"
REQUIRED_EVENTS = (
    "ready_start",
    "backswing_onset",
    "backswing_apex",
    "acceleration_onset",
    "impact",
    "followthrough_end",
    "recovery_end",
)
OPTIONAL_EVENTS = (
    "max_shoulder_external_rotation",
    "elbow_extension_onset",
)
_ORDERED_BOUNDARIES = (
    "ready_start",
    "backswing_onset",
    "backswing_apex",
    "acceleration_onset",
    "impact",
    "followthrough_end",
    "recovery_end",
)


class ForehandPhase(IntEnum):
    READY = 0
    BACKSWING = 1
    ACCELERATION = 2
    IMPACT = 3
    FOLLOWTHROUGH = 4
    RECOVERY = 5


PHASE_NAMES = tuple(phase.name.lower() for phase in ForehandPhase)


@dataclass(frozen=True)
class EventAnnotation:
    frame: int
    time_s: float
    confidence: float
    source: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> EventAnnotation:
        missing = sorted({"frame", "time_s", "confidence", "source"} - set(value))
        if missing:
            raise ValueError(f"event {name!r} is missing fields {missing}")
        event = cls(
            frame=int(value["frame"]),
            time_s=float(value["time_s"]),
            confidence=float(value["confidence"]),
            source=str(value["source"]).strip(),
        )
        if event.frame < 0:
            raise ValueError(f"event {name!r} frame must be non-negative")
        if not np.isfinite(event.time_s) or event.time_s < 0.0:
            raise ValueError(f"event {name!r} time_s must be finite and non-negative")
        if not np.isfinite(event.confidence) or not 0.0 <= event.confidence <= 1.0:
            raise ValueError(f"event {name!r} confidence must be in [0,1]")
        if not event.source:
            raise ValueError(f"event {name!r} source must be non-empty")
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "time_s": float(self.time_s),
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass(frozen=True)
class EventPhaseArrays:
    phase_global: np.ndarray
    phase_id: np.ndarray
    phase_local: np.ndarray
    time_to_impact_s: np.ndarray
    time_from_impact_s: np.ndarray
    impact_flag: np.ndarray


@dataclass(frozen=True)
class EventTimeline:
    events: dict[str, EventAnnotation]
    num_frames: int
    fps: float
    schema_version: str = EVENT_SCHEMA_VERSION

    @property
    def impact(self) -> EventAnnotation:
        return self.events["impact"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_frames": int(self.num_frames),
            "fps": float(self.fps),
            "events": {name: event.to_dict() for name, event in sorted(self.events.items())},
        }

    def phase_arrays(self) -> EventPhaseArrays:
        """Build six semantic phases while preserving impact as its own phase."""

        n = int(self.num_frames)
        ready = self.events["ready_start"].frame
        backswing = self.events["backswing_onset"].frame
        acceleration = self.events["acceleration_onset"].frame
        impact = self.events["impact"].frame
        follow_end = self.events["followthrough_end"].frame
        recovery_end = self.events["recovery_end"].frame

        phase_id = np.full(n, int(ForehandPhase.RECOVERY), dtype=np.int16)
        phase_id[:backswing] = int(ForehandPhase.READY)
        phase_id[backswing:acceleration] = int(ForehandPhase.BACKSWING)
        phase_id[acceleration:impact] = int(ForehandPhase.ACCELERATION)
        phase_id[impact : min(impact + 1, n)] = int(ForehandPhase.IMPACT)
        phase_id[min(impact + 1, n) : follow_end] = int(ForehandPhase.FOLLOWTHROUGH)
        phase_id[follow_end:] = int(ForehandPhase.RECOVERY)

        phase_local = np.zeros(n, dtype=np.float32)
        intervals = (
            (0, backswing, ForehandPhase.READY),
            (backswing, acceleration, ForehandPhase.BACKSWING),
            (acceleration, impact, ForehandPhase.ACCELERATION),
            (impact, min(impact + 1, n), ForehandPhase.IMPACT),
            (min(impact + 1, n), follow_end, ForehandPhase.FOLLOWTHROUGH),
            (follow_end, min(recovery_end + 1, n), ForehandPhase.RECOVERY),
        )
        for start, stop, phase in intervals:
            if stop <= start:
                continue
            idx = np.arange(start, stop, dtype=np.float64)
            denominator = max(stop - start - 1, 1)
            phase_local[start:stop] = ((idx - start) / denominator).astype(np.float32)
            phase_id[start:stop] = int(phase)
        # ``recovery_end`` is the semantic completion boundary.  Frames after
        # it remain in recovery but must not keep progressing toward completion.
        phase_local[recovery_end:] = 1.0

        # Piecewise anchors assign stable semantic capacity to every phase.  The
        # legacy global phase remains a single monotone scalar for ABI consumers.
        anchor_frames = np.asarray(
            [ready, backswing, acceleration, impact, follow_end, recovery_end],
            dtype=np.float64,
        )
        anchor_values = np.linspace(0.0, 1.0, len(anchor_frames), dtype=np.float64)
        frame_axis = np.arange(n, dtype=np.float64)
        phase_global = np.interp(frame_axis, anchor_frames, anchor_values).astype(np.float32)
        phase_global[:ready] = 0.0
        phase_global[recovery_end:] = 1.0

        time_axis = frame_axis / float(self.fps)
        time_from_impact = (time_axis - float(self.impact.time_s)).astype(np.float32)
        impact_flag = np.zeros(n, dtype=np.bool_)
        impact_flag[impact] = True
        return EventPhaseArrays(
            phase_global=phase_global,
            phase_id=phase_id,
            phase_local=phase_local,
            time_to_impact_s=(-time_from_impact).astype(np.float32),
            time_from_impact_s=time_from_impact,
            impact_flag=impact_flag,
        )


def load_event_timeline(
    source: str | Path | Mapping[str, Any],
    *,
    num_frames: int | None = None,
    fps: float | None = None,
) -> EventTimeline:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = load_json_strict(Path(source))
    version = str(payload.get("schema_version", payload.get("version", "")))
    if version != EVENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported event schema version: {version!r}")
    resolved_frames = int(payload.get("num_frames") if num_frames is None else num_frames)
    resolved_fps = float(payload.get("fps") if fps is None else fps)
    if resolved_frames <= 0 or not np.isfinite(resolved_fps) or resolved_fps <= 0.0:
        raise ValueError("event timeline requires positive num_frames and fps")
    if "num_frames" in payload and int(payload["num_frames"]) != resolved_frames:
        raise ValueError("event timeline num_frames does not match reference bundle")
    if "fps" in payload and not np.isclose(float(payload["fps"]), resolved_fps, atol=1e-6, rtol=0.0):
        raise ValueError("event timeline fps does not match reference bundle")
    event_payload = payload.get("events")
    if not isinstance(event_payload, Mapping):
        raise ValueError("event timeline events must be an object")
    missing = sorted(set(REQUIRED_EVENTS) - set(event_payload))
    if missing:
        raise ValueError(f"event timeline is missing required events {missing}")
    unknown = sorted(set(event_payload) - set(REQUIRED_EVENTS) - set(OPTIONAL_EVENTS))
    if unknown:
        raise ValueError(f"event timeline contains unknown events {unknown}")
    events = {str(name): EventAnnotation.from_mapping(value, name=str(name)) for name, value in event_payload.items()}
    _validate_event_order(events, num_frames=resolved_frames, fps=resolved_fps)
    return EventTimeline(events=events, num_frames=resolved_frames, fps=resolved_fps)


def _validate_event_order(
    events: Mapping[str, EventAnnotation],
    *,
    num_frames: int,
    fps: float,
) -> None:
    frames = [events[name].frame for name in _ORDERED_BOUNDARIES]
    times = [events[name].time_s for name in _ORDERED_BOUNDARIES]
    if any(frame >= int(num_frames) for frame in frames):
        raise ValueError("event frame lies outside the reference sequence")
    if any(b <= a for a, b in pairwise(frames)):
        raise ValueError("required event frames must be strictly increasing")
    if any(b <= a for a, b in pairwise(times)):
        raise ValueError("required event times must be strictly increasing")
    frame_tolerance = 1.0 / float(fps) + 1e-6
    for name, event in events.items():
        if event.frame >= int(num_frames):
            raise ValueError(f"event {name!r} lies outside the reference sequence")
        expected_time = event.frame / float(fps)
        if abs(event.time_s - expected_time) > frame_tolerance:
            raise ValueError(
                f"event {name!r} frame/time mismatch exceeds one frame: "
                f"frame={event.frame} time_s={event.time_s:g} fps={fps:g}"
            )
    for optional_name in OPTIONAL_EVENTS:
        if optional_name not in events:
            continue
        optional = events[optional_name]
        if not events["backswing_onset"].frame <= optional.frame <= events["impact"].frame:
            raise ValueError(f"optional event {optional_name!r} must occur between backswing and impact")
