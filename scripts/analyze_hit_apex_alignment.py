#!/usr/bin/env python3
"""Audit whether a badminton contact occurs in the reachable high-point window.

The script accepts either the continuous CPU audit trajectory or the v2 MJX
teacher trajectory produced by ``optimize_single_feed_hit_mjx.py``.  It never
treats event-only ``hit_*`` arrays as continuous kinematics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _array(payload: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in payload:
            return np.asarray(payload[name])
    raise ValueError("trajectory is missing all field aliases: " + ", ".join(names))


def _finite_vector(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [{rows}, 3] array")
    return result


def _sample_time(
    payload: Mapping[str, Any],
    time_to_intercept: np.ndarray,
    control_dt_s: float,
) -> tuple[np.ndarray, str]:
    timing = str(np.asarray(payload.get("kinematic_sample_timing", "legacy_post_control_step")).item())
    if "sample_time_s" in payload:
        return np.asarray(payload["sample_time_s"], dtype=np.float64), timing
    # The absolute feed intercept time is irrelevant for alignment.  Using
    # -TTI preserves every time delta and puts nominal intercept at t=0.
    relative = -np.asarray(time_to_intercept, dtype=np.float64)
    if timing in {"post_control_step", "legacy_post_control_step"}:
        relative = relative + float(control_dt_s)
    return relative, timing


def analyze_apex_alignment(
    payload: Mapping[str, Any],
    *,
    swing_duration_s: float = 1.2,
    configured_contact_phase: float = 0.76,
    configured_phase_advance_s: float = 0.0,
    stringbed_high_tolerance_m: float = 0.08,
    hand_high_tolerance_m: float = 0.06,
    max_downward_speed_m_s: float = 1.0,
) -> dict[str, Any]:
    """Return a JSON-serializable high-point/contact alignment report."""

    tti = np.asarray(_array(payload, "time_to_intercept_s"), dtype=np.float64)
    if tti.ndim != 1 or tti.size < 2 or not np.isfinite(tti).all():
        raise ValueError("time_to_intercept_s must be a finite 1-D trajectory")
    rows = int(tti.size)
    inferred_dt = float(np.median(tti[:-1] - tti[1:]))
    control_dt_s = float(np.asarray(payload.get("control_dt_s", inferred_dt)).item())
    if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    sample_time, sample_timing = _sample_time(payload, tti, control_dt_s)
    if sample_time.shape != (rows,) or not np.isfinite(sample_time).all():
        raise ValueError("sample time must be a finite 1-D trajectory")

    stringbed_position = _finite_vector(
        _array(payload, "stringbed_position_xyz_m", "stringbed_position"),
        name="stringbed position",
        rows=rows,
    )
    stringbed_velocity = _finite_vector(
        _array(
            payload,
            "stringbed_linear_velocity_xyz_m_s",
            "stringbed_linear_velocity",
        ),
        name="stringbed velocity",
        rows=rows,
    )
    shuttle_position = _finite_vector(
        _array(payload, "shuttle_position_xyz_m", "shuttle_position"),
        name="shuttle position",
        rows=rows,
    )
    hit_event = np.asarray(payload.get("hit_event", np.zeros(rows, dtype=bool)), dtype=bool)
    rebound_event = np.asarray(payload.get("event_rebound", np.zeros(rows, dtype=bool)), dtype=bool)
    if hit_event.shape != (rows,) or rebound_event.shape != (rows,):
        raise ValueError("hit/rebound event arrays must match trajectory length")
    alive = np.asarray(payload.get("alive", np.ones(rows, dtype=bool)), dtype=bool)
    if alive.shape != (rows,):
        raise ValueError("alive must match trajectory length")

    if "correction_window" in payload:
        correction_window = np.asarray(payload["correction_window"], dtype=np.float64)
        if correction_window.shape != (rows,):
            raise ValueError("correction_window must match trajectory length")
        swing_window = alive & (correction_window > 0.05)
    else:
        swing_window = alive & (tti <= 0.70) & (tti >= -0.10)
    if not swing_window.any():
        raise ValueError("trajectory has no live samples in the swing window")

    swing_indices = np.flatnonzero(swing_window)
    stringbed_apex_index = int(swing_indices[np.argmax(stringbed_position[swing_window, 2])])
    stringbed_apex_z = float(stringbed_position[stringbed_apex_index, 2])
    high_mask = swing_window & (
        stringbed_position[:, 2] >= stringbed_apex_z - float(stringbed_high_tolerance_m)
    )

    hand_position: np.ndarray | None = None
    hand_apex_index: int | None = None
    body_names: tuple[str, ...] = ()
    if "right_arm_body_position_xyz_m" in payload and "right_arm_body_names" in payload:
        body_names = tuple(str(value) for value in np.asarray(payload["right_arm_body_names"]).tolist())
        arm_position = np.asarray(payload["right_arm_body_position_xyz_m"], dtype=np.float64)
        if arm_position.shape != (rows, len(body_names), 3) or not np.isfinite(arm_position).all():
            raise ValueError("right-arm body positions are incompatible with right_arm_body_names")
        if "thirdmc_r" not in body_names:
            raise ValueError("right-arm audit trajectory does not include thirdmc_r")
        hand_position = arm_position[:, body_names.index("thirdmc_r")]
        hand_apex_index = int(swing_indices[np.argmax(hand_position[swing_window, 2])])
        hand_apex_z = float(hand_position[hand_apex_index, 2])
        high_mask &= hand_position[:, 2] >= hand_apex_z - float(hand_high_tolerance_m)

    # At the high region, prefer a fast outward-moving racket that has not
    # begun a steep downward chop.  The incoming x velocity determines the
    # opponent-facing sign, so this remains valid if court sides are swapped.
    shuttle_velocity = _finite_vector(
        _array(payload, "shuttle_velocity_xyz_m_s", "shuttle_velocity"),
        name="shuttle velocity",
        rows=rows,
    )
    nominal_index = int(np.argmin(np.abs(tti)))
    # The event solver can begin changing shuttle velocity just before the
    # reward event.  Infer court direction from clearly pre-contact samples,
    # not from the nominal-intercept frame itself.
    precontact_direction_mask = alive & (tti >= 0.20)
    if not precontact_direction_mask.any():
        precontact_direction_mask = alive & (~hit_event)
    incoming_x = float(np.median(shuttle_velocity[precontact_direction_mask, 0]))
    forward_sign = -1.0 if incoming_x > 0.0 else 1.0
    candidate_mask = high_mask & (
        stringbed_velocity[:, 2] >= -float(max_downward_speed_m_s)
    )
    if not candidate_mask.any():
        candidate_mask = high_mask
    candidate_indices = np.flatnonzero(candidate_mask)
    recommended_index = int(
        candidate_indices[
            np.argmax(forward_sign * stringbed_velocity[candidate_mask, 0])
        ]
    )

    distance = np.linalg.norm(shuttle_position - stringbed_position, axis=1)
    closest_index = int(np.flatnonzero(alive)[np.argmin(distance[alive])])
    hit_indices = np.flatnonzero(hit_event & alive)
    hit_index = None if hit_indices.size == 0 else int(hit_indices[0])
    event_contact_position = None
    if hit_index is not None and "event_stringbed_position_xyz_m" in payload:
        event_positions = np.asarray(payload["event_stringbed_position_xyz_m"], dtype=np.float64)
        if event_positions.shape == (rows, 3) and np.isfinite(event_positions).all():
            event_contact_position = event_positions[hit_index]
    contact_position = (
        event_contact_position
        if event_contact_position is not None and np.linalg.norm(event_contact_position) > 0.0
        else (None if hit_index is None else stringbed_position[hit_index])
    )

    phase = np.clip(
        configured_contact_phase
        + configured_phase_advance_s / swing_duration_s
        - tti / swing_duration_s,
        0.0,
        1.0,
    )

    def sample(index: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "index": int(index),
            "sample_time_s_relative_to_nominal_intercept": float(sample_time[index]),
            "time_to_intercept_s": float(tti[index]),
            "inherited_swing_phase": float(phase[index]),
            "stringbed_position_xyz_m": stringbed_position[index].tolist(),
            "stringbed_velocity_xyz_m_s": stringbed_velocity[index].tolist(),
            "shuttle_position_xyz_m": shuttle_position[index].tolist(),
            "ball_racket_distance_m": float(distance[index]),
            "stringbed_height_deficit_m": float(
                stringbed_apex_z - stringbed_position[index, 2]
            ),
        }
        if hand_position is not None and hand_apex_index is not None:
            result.update(
                {
                    "hand_position_xyz_m": hand_position[index].tolist(),
                    "hand_height_deficit_m": float(
                        hand_position[hand_apex_index, 2] - hand_position[index, 2]
                    ),
                }
            )
        return result

    high_indices = np.flatnonzero(high_mask)
    report: dict[str, Any] = {
        "schema_version": "stage3_hit_apex_alignment_report_v1",
        "trajectory_steps": rows,
        "control_dt_s": control_dt_s,
        "kinematic_sample_timing": sample_timing,
        "configured_timing": {
            "swing_duration_s": float(swing_duration_s),
            "contact_phase": float(configured_contact_phase),
            "phase_advance_s": float(configured_phase_advance_s),
        },
        "thresholds": {
            "stringbed_high_tolerance_m": float(stringbed_high_tolerance_m),
            "hand_high_tolerance_m": float(hand_high_tolerance_m),
            "max_downward_speed_m_s": float(max_downward_speed_m_s),
        },
        "events": {
            "hit": hit_index is not None,
            "event_rebound": bool(np.any(rebound_event & alive)),
            "hit_index": hit_index,
        },
        "stringbed_apex": sample(stringbed_apex_index),
        "hand_apex": None if hand_apex_index is None else sample(hand_apex_index),
        "nominal_intercept": sample(nominal_index),
        "closest_approach": sample(closest_index),
        "contact": None if hit_index is None else sample(hit_index),
        "recommended_high_contact": sample(recommended_index),
        "high_point_window": {
            "first_index": int(high_indices[0]),
            "last_index": int(high_indices[-1]),
            "duration_s": float(
                sample_time[high_indices[-1]] - sample_time[high_indices[0]] + control_dt_s
            ),
            "recommended_intercept_point_xyz_m": stringbed_position[recommended_index].tolist(),
            "recommended_contact_phase_if_phase_advance_zero": float(
                np.clip(
                    configured_contact_phase
                    + configured_phase_advance_s / swing_duration_s
                    - tti[recommended_index] / swing_duration_s,
                    0.0,
                    1.0,
                )
            ),
        },
    }
    if hit_index is not None and contact_position is not None:
        report["contact"].update(
            {
                "event_contact_position_xyz_m": contact_position.tolist(),
                "time_after_stringbed_apex_s": float(
                    sample_time[hit_index] - sample_time[stringbed_apex_index]
                ),
                "within_stringbed_high_window": bool(high_mask[hit_index]),
                "racket_not_steeply_downward": bool(
                    stringbed_velocity[hit_index, 2] >= -float(max_downward_speed_m_s)
                ),
            }
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--swing-duration-s", type=float, default=1.2)
    parser.add_argument("--contact-phase", type=float, default=0.76)
    parser.add_argument("--phase-advance-s", type=float, default=0.0)
    parser.add_argument("--stringbed-high-tolerance-m", type=float, default=0.08)
    parser.add_argument("--hand-high-tolerance-m", type=float, default=0.06)
    parser.add_argument("--max-downward-speed-m-s", type=float, default=1.0)
    args = parser.parse_args()

    trajectory = Path(args.trajectory).expanduser().resolve()
    with np.load(trajectory, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    report = analyze_apex_alignment(
        payload,
        swing_duration_s=args.swing_duration_s,
        configured_contact_phase=args.contact_phase,
        configured_phase_advance_s=args.phase_advance_s,
        stringbed_high_tolerance_m=args.stringbed_high_tolerance_m,
        hand_high_tolerance_m=args.hand_high_tolerance_m,
        max_downward_speed_m_s=args.max_downward_speed_m_s,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        output = Path(args.out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
