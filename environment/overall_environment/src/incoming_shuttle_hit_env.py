"""Incoming-shuttle hit RL environment.

A feeder launches a shuttle from the opposite half court toward the player,
who stands at the center of their own half with the racket rigidly attached to the right
hand. The policy drives the full muscle actuator set and is rewarded for
intercepting the shuttle with the string bed and returning it over the net
into the opponent court.

This environment is intentionally independent from the musclemimic trajectory
tracking pipeline: it owns its MuJoCo model/data and runs the badminton
physics substep loop (aero + stringbed + event rebound) itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics import (
    BadmintonPhysics,
    BadmintonPhysicsConfig,
)
from environment.overall_environment.src.control_scaling import normalized_action_to_model_ctrl
from environment.overall_environment.src.shuttle_feeder import (
    FeedConfig,
    FeedSample,
    HitWindow,
    launch_quat_from_velocity,
    sample_feed,
)
from environment.shuttlecock.src.shuttlecock_racket_impact import (
    ShuttlecockImpactConfig,
    compute_event_rebound_velocity,
)
from environment.overall_environment.src.static_forehand_clear_env import (
    FlightRegion,
    classify_landing_region,
)

READY_KEYFRAME = "overall_ready"
HUMAN_ROOT_FREEJOINT = "root"
SHUTTLE_FREEJOINT = "overall_shuttle_free"
RACKET_FREEJOINT = "overall_racket_free"
STRINGBED_CENTER_SITE = "overall_stringbed_center_site"
PALM_SITE = "rh_palm_grip_site"
GROUND_REST_HEIGHT_M = 0.035
BODY_FALL_ROOT_HEIGHT_M = 0.55
VACUUM_CLEARANCE_PREDICTION_MODE = "vacuum_ballistic_v1"
DRAG_AWARE_CLEARANCE_PREDICTION_MODE = "quadratic_drag_conservative_v1"
CLEARANCE_PREDICTION_MODES = frozenset(
    {
        VACUUM_CLEARANCE_PREDICTION_MODE,
        DRAG_AWARE_CLEARANCE_PREDICTION_MODE,
    }
)

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "approach": 1.0,
    # Time-aware contact acquisition terms.  Defaults stay zero so archived
    # legacy runs retain their exact objective unless a v3 spec opts in.
    "shuttle_proximity": 0.0,
    "timed_intercept": 0.0,
    # Direction is learned from task geometry, never copied from the (possibly
    # incorrect) reference racket orientation.
    "racket_direction": 0.0,
    "hit_bonus": 5.0,
    "hit_speed": 0.0,
    # Optional terminal penalty for never contacting the shuttle.  Keeping the
    # legacy default at zero preserves archived objectives; Stage-3 direction
    # repair enables it so avoiding a difficult return cannot beat learning it.
    "miss": 0.0,
    "return_direction": 0.0,
    "outgoing_vertical": 0.0,
    "outgoing_forward": 0.0,
    # One-shot, task-geometry-only shaping from the actual outgoing shuttle
    # state.  A zero default preserves every archived reward contract.
    "return_clearance": 0.0,
    "crossed_net": 2.0,
    # One-shot penalty when a post-impact shuttle crosses the net plane below
    # the configured legal clearance.  It is opt-in through the v4 spec.
    "invalid_net_crossing": 0.0,
    "landing_region": 4.0,
    "effort": 0.01,
    "posture": 0.5,
    "body_fall": 10.0,
    # residual-mode extra: penalty on deviation from the frozen base swing
    "residual": 0.0,
}

IMPACT_RECOVERY_PROFILE = "impact_recovery_v2"
LEGACY_PROFILE = "legacy_v1"
V2_REWARD_WEIGHTS: dict[str, float] = {
    "impact_position": 1.5,
    "impact_center": 2.0,
    "impact_time": 1.0,
    "impact_normal": 1.0,
    "impact_linear_velocity": 1.0,
    "impact_angular_velocity": 0.5,
    "precise_landing": 4.0,
    "apex": 1.0,
    "recovery_ready": 1.0,
    "recovery_balance": 0.5,
    "recovery_deceleration": 0.25,
}
V2_OBSERVATION_SIZE = 19

REGION_SCORES: dict[str, float] = {
    FlightRegion.OPPONENT_BACK.value: 1.0,
    FlightRegion.OPPONENT_MID.value: 0.5,
    FlightRegion.NET_FRONT.value: 0.2,
    FlightRegion.OWN_SIDE.value: -0.5,
    FlightRegion.OUT.value: -1.0,
}


def classify_return_net_crossing(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    *,
    player_half_sign: int,
    net_x_m: float = 0.0,
    net_height_m: float = 1.55,
    min_clearance_m: float = 0.0,
) -> dict[str, float | bool]:
    """Classify one player-to-opponent net-plane crossing.

    Height is linearly interpolated at the net plane.  Merely ending a step on
    the opponent half is not a legal return: the shuttle must have started on
    the player's half, travelled in the return direction, and cleared the
    configured absolute net height plus margin.
    """

    previous = np.asarray(previous_position, dtype=float)
    current = np.asarray(current_position, dtype=float)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError("return net crossing positions must be 3-vectors")
    sign = int(player_half_sign)
    if sign not in {-1, 1}:
        raise ValueError("player_half_sign must be -1 or 1")
    required_height = float(net_height_m) + float(min_clearance_m)
    previous_x = float(previous[0] - net_x_m)
    current_x = float(current[0] - net_x_m)
    travelled_to_opponent = (-sign) * (current_x - previous_x) > 0.0
    crossed = bool(sign * previous_x >= 0.0 and sign * current_x < 0.0 and travelled_to_opponent)
    denominator = current_x - previous_x
    alpha = 0.0 if abs(denominator) <= 1.0e-12 else float(np.clip(-previous_x / denominator, 0.0, 1.0))
    crossing_height = float(previous[2] + alpha * (current[2] - previous[2]))
    clearance = crossing_height - float(net_height_m)
    return {
        "crossed": crossed,
        "valid": bool(crossed and crossing_height >= required_height),
        "crossing_height_m": crossing_height,
        "clearance_m": clearance,
        "required_height_m": required_height,
    }


def ballistic_return_clearance_score(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    player_half_sign: int,
    net_x_m: float = 0.0,
    net_height_m: float = 1.55,
    min_clearance_m: float = 0.0,
    gravity_m_s2: float = 9.81,
    score_softness_m: float = 0.35,
    max_prediction_time_s: float = 1.5,
) -> dict[str, float]:
    """Legacy vacuum-parabola proxy for post-contact net clearance.

    This function is retained byte-for-byte for old experiment
    reproducibility.  It ignores the shuttle's strong quadratic drag and is
    therefore *not* a conservative reachability test.  New experiments should
    explicitly select ``quadratic_drag_conservative_v1``; the later legal-cross
    event from the full simulation remains authoritative in either mode.
    """

    xyz = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    if xyz.shape != (3,) or vel.shape != (3,):
        raise ValueError("ballistic return position and velocity must be 3-vectors")
    if not np.isfinite(xyz).all() or not np.isfinite(vel).all():
        raise ValueError("ballistic return position and velocity must be finite")
    sign = int(player_half_sign)
    if sign not in {-1, 1}:
        raise ValueError("player_half_sign must be -1 or 1")
    if min_clearance_m < 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("clearance must be non-negative and gravity positive")
    if score_softness_m <= 0.0 or max_prediction_time_s <= 0.0:
        raise ValueError("ballistic score softness and horizon must be positive")

    distance_to_net = max(0.0, sign * float(xyz[0] - net_x_m))
    forward_speed = -sign * float(vel[0])
    forward_gate = float(np.clip(forward_speed / 4.0, 0.0, 1.0))
    prediction_time = float(
        np.clip(
            distance_to_net / max(forward_speed, 0.25),
            0.0,
            max_prediction_time_s,
        )
    )
    predicted_height = float(xyz[2] + vel[2] * prediction_time - 0.5 * gravity_m_s2 * prediction_time**2)
    predicted_clearance = predicted_height - float(net_height_m)
    clearance_gap = predicted_clearance - float(min_clearance_m)
    scaled_gap = float(np.clip(clearance_gap / score_softness_m, -60.0, 60.0))
    height_score = 1.0 / (1.0 + math.exp(-scaled_gap))
    return {
        "score": forward_gate * height_score,
        "predicted_clearance_m": predicted_clearance,
        "forward_speed_m_s": forward_speed,
        "prediction_time_s": prediction_time,
    }


def drag_aware_return_clearance_score(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    player_half_sign: int,
    net_x_m: float = 0.0,
    net_height_m: float = 1.55,
    min_clearance_m: float = 0.0,
    gravity_m_s2: float = 9.81,
    terminal_velocity_m_s: float = 6.86,
    drag_multiplier: float = 1.20,
    score_softness_m: float = 0.35,
    max_prediction_time_s: float = 1.5,
    integration_steps: int = 128,
    ground_height_m: float = GROUND_REST_HEIGHT_M,
) -> dict[str, float | bool]:
    """Conservatively project a return through the shuttle's quadratic drag.

    The earlier dense proxy used a vacuum parabola even though the simulated
    shuttle is calibrated to a 6.86 m/s terminal velocity.  At realistic hit
    speeds that can predict a legal return for a shuttle which actually lands
    before the net.  This fixed-step semi-implicit projection mirrors the
    translational aero law used by :class:`BadmintonPhysics`; using the maximum
    configured angle-drag multiplier makes its positive clearance a
    conservative reachability certificate.

    If the shuttle lands short, ``predicted_clearance_m`` is a continuous net
    margin: ground height minus net height minus remaining horizontal distance.
    It is therefore always negative but still ranks near-net landings above
    returns that die far inside the player's court.
    """

    xyz = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    if xyz.shape != (3,) or vel.shape != (3,):
        raise ValueError("drag-aware return position and velocity must be 3-vectors")
    if not np.isfinite(xyz).all() or not np.isfinite(vel).all():
        raise ValueError("drag-aware return position and velocity must be finite")
    sign = int(player_half_sign)
    if sign not in {-1, 1}:
        raise ValueError("player_half_sign must be -1 or 1")
    scalars = (
        net_x_m,
        net_height_m,
        min_clearance_m,
        gravity_m_s2,
        terminal_velocity_m_s,
        drag_multiplier,
        score_softness_m,
        max_prediction_time_s,
        ground_height_m,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("drag-aware clearance parameters must be finite")
    if min_clearance_m < 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("clearance must be non-negative and gravity positive")
    if terminal_velocity_m_s <= 0.0 or drag_multiplier <= 0.0:
        raise ValueError("terminal velocity and drag multiplier must be positive")
    if score_softness_m <= 0.0 or max_prediction_time_s <= 0.0:
        raise ValueError("drag-aware score softness and horizon must be positive")
    if isinstance(integration_steps, bool) or int(integration_steps) <= 0:
        raise ValueError("drag-aware integration_steps must be a positive integer")
    if float(integration_steps) != float(int(integration_steps)):
        raise ValueError("drag-aware integration_steps must be a positive integer")

    distance_to_net = max(0.0, sign * float(xyz[0] - net_x_m))
    forward_speed = -sign * float(vel[0])
    forward_gate = float(np.clip(forward_speed / 4.0, 0.0, 1.0))
    if distance_to_net <= 0.0:
        predicted_clearance = float(xyz[2] - net_height_m)
        predicted_crosses = True
        prediction_time = 0.0
        landing_shortfall = 0.0
    else:
        pos = xyz.copy()
        current_velocity = vel.copy()
        dt = float(max_prediction_time_s) / int(integration_steps)
        drag_coefficient = float(gravity_m_s2) * float(drag_multiplier) / float(terminal_velocity_m_s) ** 2
        predicted_crosses = False
        prediction_time = float(max_prediction_time_s)
        landing_shortfall = max(0.0, sign * float(pos[0] - net_x_m))
        predicted_clearance = float(pos[2]) - float(net_height_m) - landing_shortfall
        for step_index in range(int(integration_steps)):
            previous = pos.copy()
            speed = float(np.linalg.norm(current_velocity))
            acceleration = -drag_coefficient * speed * current_velocity
            acceleration[2] -= float(gravity_m_s2)
            current_velocity = current_velocity + acceleration * dt
            pos = pos + current_velocity * dt

            previous_x = sign * float(previous[0] - net_x_m)
            current_x = sign * float(pos[0] - net_x_m)
            travelled_to_opponent = -sign * float(pos[0] - previous[0]) > 0.0
            plane_crossed = previous_x >= 0.0 and current_x < 0.0 and travelled_to_opponent
            x_denominator = current_x - previous_x
            cross_alpha = (
                0.0 if abs(x_denominator) <= 1.0e-12 else float(np.clip(-previous_x / x_denominator, 0.0, 1.0))
            )
            cross_height = float(previous[2] + cross_alpha * (pos[2] - previous[2]))

            ground_crossed = previous[2] > float(ground_height_m) and pos[2] <= float(ground_height_m)
            z_denominator = float(pos[2] - previous[2])
            ground_alpha = (
                0.0
                if abs(z_denominator) <= 1.0e-12
                else float(
                    np.clip(
                        (float(ground_height_m) - previous[2]) / z_denominator,
                        0.0,
                        1.0,
                    )
                )
            )
            cross_first = bool(
                plane_crossed
                and cross_height > float(ground_height_m)
                and (not ground_crossed or cross_alpha <= ground_alpha)
            )
            if cross_first:
                predicted_crosses = True
                predicted_clearance = cross_height - float(net_height_m)
                prediction_time = (step_index + cross_alpha) * dt
                landing_shortfall = 0.0
                break
            if ground_crossed:
                landing_x = float(previous[0] + ground_alpha * (pos[0] - previous[0]))
                landing_shortfall = max(0.0, sign * float(landing_x - net_x_m))
                predicted_clearance = float(ground_height_m) - float(net_height_m) - landing_shortfall
                prediction_time = (step_index + ground_alpha) * dt
                break
        else:
            landing_shortfall = max(0.0, sign * float(pos[0] - net_x_m))
            predicted_clearance = float(pos[2]) - float(net_height_m) - landing_shortfall

    clearance_gap = predicted_clearance - float(min_clearance_m)
    scaled_gap = float(np.clip(clearance_gap / score_softness_m, -60.0, 60.0))
    height_score = 1.0 / (1.0 + math.exp(-scaled_gap))
    return {
        "score": forward_gate * height_score,
        "predicted_clearance_m": predicted_clearance,
        "forward_speed_m_s": forward_speed,
        "prediction_time_s": prediction_time,
        "predicted_crosses_net": predicted_crosses,
        "predicted_landing_shortfall_m": landing_shortfall,
    }


def counterfactual_rebound_guidance_score(
    position: np.ndarray,
    shuttle_velocity: np.ndarray,
    racket_surface_velocity: np.ndarray,
    signed_face_normal: np.ndarray,
    desired_return_direction: np.ndarray,
    *,
    player_half_sign: int,
    impact_config: ShuttlecockImpactConfig,
    net_x_m: float = 0.0,
    net_height_m: float = 1.55,
    min_clearance_m: float = 0.0,
    clearance_softness_m: float = 0.35,
    closing_softness_m_s: float = 1.0,
    quality_mode: str = "balanced_shifted",
) -> dict[str, float | np.ndarray]:
    """Predict task quality from the same event-impact law used by simulation.

    This is contact-free, demonstration-free shaping: the current shuttle,
    racket-surface velocity, and signed physical face normal are substituted
    into the real rebound equation.  A smooth gate around the actual event
    threshold prevents a nearly static overlap from imitating a useful hit.
    The score remains non-negative so dense shaping cannot make avoiding the
    shuttle preferable to acquiring a real contact.
    """

    position = np.asarray(position, dtype=float)
    shuttle_velocity = np.asarray(shuttle_velocity, dtype=float)
    racket_surface_velocity = np.asarray(racket_surface_velocity, dtype=float)
    normal = np.asarray(signed_face_normal, dtype=float)
    desired = np.asarray(desired_return_direction, dtype=float)
    vectors = {
        "position": position,
        "shuttle_velocity": shuttle_velocity,
        "racket_surface_velocity": racket_surface_velocity,
        "signed_face_normal": normal,
        "desired_return_direction": desired,
    }
    if any(value.shape != (3,) for value in vectors.values()):
        raise ValueError("counterfactual rebound inputs must be three-vectors")
    if any(not np.isfinite(value).all() for value in vectors.values()):
        raise ValueError("counterfactual rebound inputs must be finite")
    normal_norm = float(np.linalg.norm(normal))
    desired_norm = float(np.linalg.norm(desired))
    if normal_norm <= 1.0e-9 or desired_norm <= 1.0e-9:
        raise ValueError("counterfactual normal and desired direction must be non-zero")
    if closing_softness_m_s <= 0.0:
        raise ValueError("closing_softness_m_s must be positive")
    if quality_mode not in {"balanced_shifted", "clearance_priority"}:
        raise ValueError("quality_mode must be balanced_shifted or clearance_priority")
    normal = normal / normal_norm
    desired = desired / desired_norm
    relative_normal_velocity = float(np.dot(shuttle_velocity - racket_surface_velocity, normal))
    closing_speed = -relative_normal_velocity
    scaled_closing = float(
        np.clip(
            (closing_speed - impact_config.min_speed_for_event_m_s) / closing_softness_m_s,
            -60.0,
            60.0,
        )
    )
    closing_gate = 1.0 / (1.0 + math.exp(-scaled_closing))
    predicted_velocity = compute_event_rebound_velocity(
        shuttle_velocity_world=shuttle_velocity,
        racket_surface_velocity_world=racket_surface_velocity,
        normal_world=normal,
        cfg=impact_config,
    )
    predicted_speed = float(np.linalg.norm(predicted_velocity))
    direction_signed_score = float(
        np.clip(
            np.dot(predicted_velocity / max(predicted_speed, 1.0e-9), desired),
            -1.0,
            1.0,
        )
    )
    if quality_mode == "clearance_priority":
        # A sideways or downward rebound must not receive the 0.5 baseline
        # used by v10.  Keeping the score non-negative still prevents the
        # agent from improving its return by avoiding the shuttle entirely.
        direction_score = max(direction_signed_score, 0.0)
        direction_fraction = 0.30
    else:
        direction_score = 0.5 * (direction_signed_score + 1.0)
        direction_fraction = 0.65
    clearance = ballistic_return_clearance_score(
        position,
        predicted_velocity,
        player_half_sign=player_half_sign,
        net_x_m=net_x_m,
        net_height_m=net_height_m,
        min_clearance_m=min_clearance_m,
        score_softness_m=clearance_softness_m,
    )
    score = closing_gate * (
        direction_fraction * direction_score + (1.0 - direction_fraction) * float(clearance["score"])
    )
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "closing_gate": closing_gate,
        "closing_speed_m_s": closing_speed,
        "direction_signed_score": direction_signed_score,
        "direction_score": direction_score,
        "clearance_score": float(clearance["score"]),
        "predicted_clearance_m": float(clearance["predicted_clearance_m"]),
        "predicted_velocity_m_s": predicted_velocity,
    }


def inverse_impact_guidance_score(
    shuttle_velocity: np.ndarray,
    racket_surface_velocity: np.ndarray,
    signed_face_normal: np.ndarray,
    desired_return_direction: np.ndarray,
    *,
    impact_config: ShuttlecockImpactConfig,
    target_outgoing_speed_m_s: float = 12.0,
    racket_velocity_softness_m_s: float = 6.0,
    racket_velocity_fraction: float = 0.5,
) -> dict[str, float | np.ndarray]:
    """Score the physical racket state that exactly produces a target return.

    For a purely normal relative impact, ``v_racket = v_in + c n`` and the
    event law gives ``v_out = v_in + (1 + e) c n``.  Therefore choosing ``n``
    along ``v_target - v_in`` and ``c = ||v_target-v_in|| / (1+e)`` is a
    closed-form inverse of the simulator's own rebound equation.  The target
    is derived from the live incoming shuttle and court return direction; it
    never uses a demonstration racket pose or velocity.
    """

    shuttle_velocity = np.asarray(shuttle_velocity, dtype=float)
    racket_surface_velocity = np.asarray(racket_surface_velocity, dtype=float)
    normal = np.asarray(signed_face_normal, dtype=float)
    desired = np.asarray(desired_return_direction, dtype=float)
    vectors = {
        "shuttle_velocity": shuttle_velocity,
        "racket_surface_velocity": racket_surface_velocity,
        "signed_face_normal": normal,
        "desired_return_direction": desired,
    }
    if any(value.shape != (3,) for value in vectors.values()):
        raise ValueError("inverse impact inputs must be three-vectors")
    if any(not np.isfinite(value).all() for value in vectors.values()):
        raise ValueError("inverse impact inputs must be finite")
    normal_norm = float(np.linalg.norm(normal))
    desired_norm = float(np.linalg.norm(desired))
    if normal_norm <= 1.0e-9 or desired_norm <= 1.0e-9:
        raise ValueError("inverse impact normal and desired direction must be non-zero")
    if not math.isfinite(target_outgoing_speed_m_s) or target_outgoing_speed_m_s <= 0.0:
        raise ValueError("target_outgoing_speed_m_s must be finite and positive")
    if not math.isfinite(racket_velocity_softness_m_s) or racket_velocity_softness_m_s <= 0.0:
        raise ValueError("racket_velocity_softness_m_s must be finite and positive")
    if not math.isfinite(racket_velocity_fraction) or not 0.0 <= racket_velocity_fraction <= 1.0:
        raise ValueError("racket_velocity_fraction must lie in [0, 1]")
    restitution_denominator = 1.0 + float(impact_config.event_restitution_normal)
    if restitution_denominator <= 1.0e-9:
        raise ValueError("event restitution must be greater than -1")

    normal = normal / normal_norm
    desired = desired / desired_norm
    target_outgoing_velocity = target_outgoing_speed_m_s * desired
    required_delta = target_outgoing_velocity - shuttle_velocity
    required_delta_norm = float(np.linalg.norm(required_delta))
    if required_delta_norm <= 1.0e-9:
        target_normal = desired
        target_closing_speed = 0.0
    else:
        target_normal = required_delta / required_delta_norm
        target_closing_speed = required_delta_norm / restitution_denominator
    target_racket_velocity = shuttle_velocity + target_closing_speed * target_normal
    signed_normal_alignment = float(np.clip(np.dot(normal, target_normal), -1.0, 1.0))
    normal_alignment = max(signed_normal_alignment, 0.0)
    shifted_normal_score = 0.5 * (signed_normal_alignment + 1.0)
    racket_velocity_error = float(np.linalg.norm(racket_surface_velocity - target_racket_velocity))
    scaled_velocity_error = racket_velocity_error / racket_velocity_softness_m_s
    # A Cauchy kernel retains a useful gradient even when the inherited swing
    # is far from the inverse-physics target; a Gaussian saturated in v10/v11.
    racket_velocity_score = 1.0 / (1.0 + scaled_velocity_error**2)
    score = normal_alignment * racket_velocity_score
    # The legacy product above has a dead zone: when the inherited racket face
    # points more than 90 degrees away from the target, ``normal_alignment`` is
    # exactly zero and masks both the normal and velocity gradients.  The
    # decomposed score is non-negative but keeps both factors independently
    # learnable.  It is exposed under a distinct control ABI and therefore
    # does not change old checkpoint semantics.
    decomposed_score = (
        1.0 - racket_velocity_fraction
    ) * shifted_normal_score + racket_velocity_fraction * racket_velocity_score
    target_rebound_velocity = compute_event_rebound_velocity(
        shuttle_velocity_world=shuttle_velocity,
        racket_surface_velocity_world=target_racket_velocity,
        normal_world=target_normal,
        cfg=impact_config,
    )
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "decomposed_score": float(np.clip(decomposed_score, 0.0, 1.0)),
        "signed_normal_alignment": signed_normal_alignment,
        "shifted_normal_score": shifted_normal_score,
        "normal_alignment": normal_alignment,
        "racket_velocity_score": racket_velocity_score,
        "racket_velocity_error_m_s": racket_velocity_error,
        "target_closing_speed_m_s": target_closing_speed,
        "target_face_normal": target_normal,
        "target_racket_velocity_m_s": target_racket_velocity,
        "target_outgoing_velocity_m_s": target_outgoing_velocity,
        "target_rebound_velocity_m_s": target_rebound_velocity,
    }


class IncomingHitState(str, Enum):
    INCOMING = "INCOMING"
    HIT = "HIT"
    FLIGHT = "FLIGHT"
    RECOVERY = "RECOVERY"
    DONE = "DONE"


def _validate_reward_weights(weights: dict[str, float], *, task_profile: str = LEGACY_PROFILE) -> dict[str, float]:
    if task_profile not in {LEGACY_PROFILE, IMPACT_RECOVERY_PROFILE}:
        raise ValueError(f"unsupported incoming-hit task_profile: {task_profile!r}")
    merged = dict(DEFAULT_REWARD_WEIGHTS)
    if task_profile == IMPACT_RECOVERY_PROFILE:
        merged.update(V2_REWARD_WEIGHTS)
    unknown = set(weights) - set(merged)
    if unknown:
        raise ValueError(f"unknown reward weight keys: {sorted(unknown)}")
    for key, value in weights.items():
        merged[key] = float(value)
    return merged


CONTACT_GUIDANCE_REWARD_MODES = frozenset(
    {
        "dense_per_step",
        "best_progress",
        "event_direction",
        "potential_event_direction",
        "closest_approach_event_direction",
    }
)


def _bounded_best_progress(current: float, previous_best: float) -> tuple[float, float]:
    """Return a non-negative telescoping increment and the updated best.

    Contact guidance potentials are in ``[0, 1]``.  Paying only improvements
    makes the total reward from one potential no larger than its configured
    weight, regardless of episode length or how long the racket parks nearby.
    """

    current_value = float(np.clip(current, 0.0, 1.0))
    best_value = float(np.clip(previous_best, 0.0, 1.0))
    updated_best = max(best_value, current_value)
    return updated_best - best_value, updated_best


def _discounted_event_direction_increment(
    previous_potential: float,
    current_potential: float,
    *,
    discount: float,
    event_score: float | None = None,
    terminal_without_event: bool = False,
) -> tuple[float, float]:
    """Potential shaping whose discounted sum is only the event score.

    On an ordinary transition this returns gamma * Phi(next) - Phi(prev).
    On a real hit, the next potential is terminal and the exact event score is
    added, yielding event_score - Phi(prev).  A miss simply returns
    -Phi(prev).  Starting from zero, the discounted shaping terms telescope
    exactly, so an early good racket pose that is lost at impact earns nothing.
    """

    previous = float(previous_potential)
    current = float(current_potential)
    gamma = float(discount)
    if not all(math.isfinite(value) for value in (previous, current, gamma)):
        raise ValueError("event-direction potentials and discount must be finite")
    if not -1.0 <= previous <= 1.0 or not -1.0 <= current <= 1.0:
        raise ValueError("event-direction potentials must lie in [-1, 1]")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("contact_guidance_discount must lie in (0, 1]")
    if event_score is not None:
        event = float(event_score)
        if not math.isfinite(event) or not -1.0 <= event <= 1.0:
            raise ValueError("event-direction score must lie in [-1, 1]")
        return event - previous, 0.0
    if terminal_without_event:
        return -previous, 0.0
    return gamma * current - previous, current


def _validate_contact_guidance_contract(mode: str, reward_weights: dict[str, float]) -> str:
    resolved = str(mode)
    if resolved not in CONTACT_GUIDANCE_REWARD_MODES:
        raise ValueError(
            "contact_guidance_reward_mode must be dense_per_step, best_progress, "
            "event_direction, potential_event_direction, or "
            "closest_approach_event_direction"
        )
    if resolved == "best_progress":
        guidance_cap = sum(
            max(0.0, float(reward_weights[name]))
            for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
        )
        miss_penalty = float(reward_weights["miss"])
        if miss_penalty <= guidance_cap:
            raise ValueError(
                "best_progress requires miss reward weight to exceed the bounded "
                f"contact-guidance cap ({miss_penalty} <= {guidance_cap})"
            )
        best_miss_return = guidance_cap - miss_penalty
        minimum_bad_hit_return = float(reward_weights["hit_bonus"]) - float(reward_weights["invalid_net_crossing"])
        if minimum_bad_hit_return <= best_miss_return:
            raise ValueError("best_progress reward hierarchy requires a real hit to beat the best possible miss")
    elif resolved in {"event_direction", "potential_event_direction"}:
        # Racket direction is paid only on the real rebound, so it is not part
        # of the profitable-miss cap.  The two pre-contact shaping potentials
        # remain telescoping and must still be dominated by the miss penalty.
        pre_contact_cap = sum(
            max(0.0, float(reward_weights[name])) for name in ("shuttle_proximity", "timed_intercept")
        )
        miss_penalty = float(reward_weights["miss"])
        if miss_penalty <= pre_contact_cap:
            raise ValueError(
                f"{resolved} requires miss reward weight to exceed the bounded "
                f"pre-contact guidance cap ({miss_penalty} <= {pre_contact_cap})"
            )
    elif resolved == "closest_approach_event_direction":
        # A miss receives the signed inverse-impact score from exactly one
        # physical state: the closest stringbed/cork approach in the contact
        # window.  Include that terminal signal in the profitable-miss cap.
        guidance_cap = sum(
            max(0.0, float(reward_weights[name]))
            for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
        )
        miss_penalty = float(reward_weights["miss"])
        if miss_penalty <= guidance_cap:
            raise ValueError(
                "closest_approach_event_direction requires miss reward weight "
                "to exceed the bounded terminal-guidance cap "
                f"({miss_penalty} <= {guidance_cap})"
            )
        best_miss_return = guidance_cap - miss_penalty
        minimum_bad_hit_return = (
            float(reward_weights["hit_bonus"])
            - float(reward_weights["racket_direction"])
            - float(reward_weights["return_direction"])
            - float(reward_weights["return_clearance"])
            - float(reward_weights["invalid_net_crossing"])
            - float(reward_weights["landing_region"])
        )
        if minimum_bad_hit_return <= best_miss_return:
            raise ValueError(
                "closest_approach_event_direction reward hierarchy requires "
                "the worst real hit to beat the best possible miss"
            )
    return resolved


def incoming_hit_policy_abi_hash(control_manifest: dict[str, Any]) -> str:
    """Hash policy-facing ABI while excluding train/eval dataset identity."""
    payload = dict(control_manifest)
    payload.pop("control_hash", None)
    payload.pop("policy_abi_hash", None)
    environment_abi = dict(payload.get("environment_abi", {}) or {})
    environment_abi.pop("target_bank_sha256", None)
    environment_abi.pop("task_curriculum_stage", None)
    payload["environment_abi"] = environment_abi
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _feed_difficulty(feed: FeedSample) -> float:
    """Match the MJX stable easy-to-hard curriculum feed ordering."""

    point = np.asarray(feed.intercept_point, dtype=float)
    center_penalty = abs(float(point[1])) + 0.6 * abs(float(point[2]) - 1.8)
    timing_penalty = 0.25 * abs(float(feed.intercept_time_s) - 0.75)
    speed_penalty = 0.02 * float(np.linalg.norm(feed.intercept_velocity))
    return center_penalty + timing_penalty + speed_penalty


class IncomingShuttleHitEnv:
    def __init__(
        self,
        xml: str | Path,
        *,
        feed_bank: list[FeedSample] | None = None,
        feed_config: FeedConfig | None = None,
        hit_window: HitWindow | None = None,
        physics_config: BadmintonPhysicsConfig | None = None,
        control_substeps: int = 10,
        max_episode_steps: int = 300,
        reward_weights: dict[str, float] | None = None,
        player_half_sign: int = -1,
        singles: bool = True,
        terminate_on_body_fall: bool = True,
        base_policy_artifact: str | Path | None = None,
        residual_scale: float = 0.3,
        residual_scale_overrides: dict[str, float] | None = None,
        residual_scale_schedule: dict[str, Any] | None = None,
        residual_authority_progress: float = 1.0,
        base_skill: str | None = None,
        lab_controller: Any | None = None,
        lab_state_builder: Any | None = None,
        curriculum: Any | None = None,
        curriculum_feed_order: str = "difficulty_sorted",
        seed_feed_fingerprints: tuple[str, ...] | list[str] = (),
        filter_finger_observation: bool | None = None,
        swing_duration_s: float = 1.2,
        contact_phase: float = 0.76,
        swing_phase_advance_s: float = 0.0,
        return_net_x_m: float = 0.0,
        return_net_height_m: float = 1.55,
        min_return_net_clearance_m: float | None = None,
        desired_return_up_component: float = 0.40,
        ballistic_return_score_softness_m: float = 0.35,
        clearance_prediction_mode: str = VACUUM_CLEARANCE_PREDICTION_MODE,
        shuttle_proximity_softness_m: float = 0.35,
        timed_intercept_softness_m: float = 0.30,
        direction_distance_softness_m: float = 0.45,
        contact_guidance_reward_mode: str = "dense_per_step",
        contact_guidance_discount: float = 1.0,
        racket_velocity_direction_fraction: float = 0.30,
        direction_reward_mode: str = "positive_projection",
        clearance_reward_mode: str = "positive_score",
        hit_event_mode: str = "any_stringbed_contact",
        racket_guidance_mode: str = "component_projection",
        inverse_target_speed_m_s: float = 12.0,
        inverse_velocity_softness_m_s: float = 6.0,
        task_profile: str = LEGACY_PROFILE,
        impact_target_bank: Any | None = None,
        recovery_horizon_steps: int = 60,
        task_curriculum_stage: str | None = None,
        racket_attachment_contract: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.xml_path = Path(xml)
        self.racket_attachment_contract_path = (
            None if racket_attachment_contract is None else Path(racket_attachment_contract).expanduser().resolve()
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.physics = BadmintonPhysics(physics_config)
        self.curriculum_feed_order = str(curriculum_feed_order)
        if self.curriculum_feed_order not in {
            "difficulty_sorted",
            "stored",
            "explicit_fingerprint_order",
        }:
            raise ValueError("curriculum_feed_order must be difficulty_sorted, stored, or explicit_fingerprint_order")
        self.seed_feed_fingerprints = tuple(str(value) for value in seed_feed_fingerprints)
        self.feed_bank = feed_bank
        if curriculum is not None and self.feed_bank is not None and self.curriculum_feed_order == "difficulty_sorted":
            self.feed_bank = sorted(self.feed_bank, key=_feed_difficulty)
        elif self.curriculum_feed_order == "explicit_fingerprint_order":
            if self.feed_bank is None:
                raise ValueError("explicit_fingerprint_order requires a deterministic feed bank")
            from environment.overall_environment.src.shuttle_feeder import (
                reorder_feed_bank_with_seed_fingerprints,
            )

            self.feed_bank = reorder_feed_bank_with_seed_fingerprints(
                self.feed_bank,
                self.seed_feed_fingerprints,
            )
        self.feed_config = feed_config if feed_config is not None else FeedConfig()
        self.hit_window = hit_window if hit_window is not None else HitWindow()
        self.control_substeps = int(control_substeps)
        self.max_episode_steps = int(max_episode_steps)
        if self.control_substeps <= 0:
            raise ValueError(f"control_substeps must be positive, got {control_substeps}")
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")
        self.task_profile = str(task_profile)
        self.reward_weights = _validate_reward_weights(reward_weights or {}, task_profile=self.task_profile)
        self.player_half_sign = int(player_half_sign)
        if self.player_half_sign not in {-1, 1}:
            raise ValueError("player_half_sign must be -1 or 1")
        self.singles = bool(singles)
        self.terminate_on_body_fall = bool(terminate_on_body_fall)
        self.rng = np.random.default_rng(seed)
        self.recovery_horizon_steps = int(recovery_horizon_steps)
        if self.recovery_horizon_steps <= 0:
            raise ValueError("recovery_horizon_steps must be positive")
        self.return_net_x_m = float(return_net_x_m)
        self.return_net_height_m = float(return_net_height_m)
        self.min_return_net_clearance_m = (
            None if min_return_net_clearance_m is None else float(min_return_net_clearance_m)
        )
        self.desired_return_up_component = float(desired_return_up_component)
        self.ballistic_return_score_softness_m = float(ballistic_return_score_softness_m)
        self.clearance_prediction_mode = str(clearance_prediction_mode)
        self.shuttle_proximity_softness_m = float(shuttle_proximity_softness_m)
        self.timed_intercept_softness_m = float(timed_intercept_softness_m)
        self.direction_distance_softness_m = float(direction_distance_softness_m)
        self.contact_guidance_reward_mode = _validate_contact_guidance_contract(
            contact_guidance_reward_mode,
            self.reward_weights,
        )
        self.contact_guidance_discount = float(contact_guidance_discount)
        self.racket_velocity_direction_fraction = float(racket_velocity_direction_fraction)
        self.direction_reward_mode = str(direction_reward_mode)
        self.clearance_reward_mode = str(clearance_reward_mode)
        self.hit_event_mode = str(hit_event_mode)
        self.racket_guidance_mode = str(racket_guidance_mode)
        self.inverse_target_speed_m_s = float(inverse_target_speed_m_s)
        self.inverse_velocity_softness_m_s = float(inverse_velocity_softness_m_s)
        if not all(
            math.isfinite(value)
            for value in (
                self.return_net_x_m,
                self.return_net_height_m,
                self.desired_return_up_component,
                self.ballistic_return_score_softness_m,
                self.shuttle_proximity_softness_m,
                self.timed_intercept_softness_m,
                self.direction_distance_softness_m,
                self.contact_guidance_discount,
                self.racket_velocity_direction_fraction,
                self.inverse_target_speed_m_s,
                self.inverse_velocity_softness_m_s,
            )
        ):
            raise ValueError("return constraints must be finite")
        if self.return_net_height_m <= 0.0:
            raise ValueError("return_net_height_m must be positive")
        if self.min_return_net_clearance_m is not None and (
            not math.isfinite(self.min_return_net_clearance_m) or self.min_return_net_clearance_m < 0.0
        ):
            raise ValueError("min_return_net_clearance_m must be finite and non-negative")
        if self.desired_return_up_component <= 0.0:
            raise ValueError("desired_return_up_component must be positive")
        if self.ballistic_return_score_softness_m <= 0.0:
            raise ValueError("ballistic_return_score_softness_m must be positive")
        if self.clearance_prediction_mode not in CLEARANCE_PREDICTION_MODES:
            raise ValueError(
                "clearance_prediction_mode must be one of "
                f"{sorted(CLEARANCE_PREDICTION_MODES)}, got "
                f"{self.clearance_prediction_mode!r}"
            )
        if self.shuttle_proximity_softness_m <= 0.0:
            raise ValueError("shuttle_proximity_softness_m must be positive")
        if self.timed_intercept_softness_m <= 0.0:
            raise ValueError("timed_intercept_softness_m must be positive")
        if self.direction_distance_softness_m <= 0.0:
            raise ValueError("direction_distance_softness_m must be positive")
        if not 0.0 < self.contact_guidance_discount <= 1.0:
            raise ValueError("contact_guidance_discount must lie in (0, 1]")
        if not 0.0 <= self.racket_velocity_direction_fraction <= 1.0:
            raise ValueError("racket_velocity_direction_fraction must lie in [0, 1]")
        if self.inverse_target_speed_m_s <= 0.0:
            raise ValueError("inverse_target_speed_m_s must be positive")
        if self.inverse_velocity_softness_m_s <= 0.0:
            raise ValueError("inverse_velocity_softness_m_s must be positive")
        if self.direction_reward_mode not in {
            "positive_projection",
            "signed_projection",
        }:
            raise ValueError("direction_reward_mode must be positive_projection or signed_projection")
        if self.clearance_reward_mode not in {
            "positive_score",
            "signed_centered",
        }:
            raise ValueError("clearance_reward_mode must be positive_score or signed_centered")
        if self.hit_event_mode not in {
            "any_stringbed_contact",
            "event_rebound",
        }:
            raise ValueError("hit_event_mode must be any_stringbed_contact or event_rebound")
        if self.racket_guidance_mode not in {
            "component_projection",
            "counterfactual_rebound",
            "counterfactual_clearance_priority",
            "inverse_impact_target",
            "inverse_impact_decomposed",
        }:
            raise ValueError(
                "racket_guidance_mode must be component_projection or "
                "counterfactual_rebound or counterfactual_clearance_priority "
                "or inverse_impact_target or inverse_impact_decomposed"
            )
        if self.contact_guidance_reward_mode in {
            "event_direction",
            "potential_event_direction",
            "closest_approach_event_direction",
        }:
            if self.hit_event_mode != "event_rebound":
                raise ValueError("event-based direction contact guidance requires hit_event_mode=event_rebound")
            if self.racket_guidance_mode != "inverse_impact_decomposed":
                raise ValueError(
                    "event-based direction contact guidance requires racket_guidance_mode=inverse_impact_decomposed"
                )

        self.impact_target_bank = None
        self._target_arrays: dict[str, np.ndarray] | None = None
        self.task_curriculum_stage = task_curriculum_stage
        self._v2_environment_mode_code = 3
        self._v2_reward_mask = dict.fromkeys(V2_REWARD_WEIGHTS, 1.0)
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            if not self.feed_bank:
                raise ValueError("impact_recovery_v2 requires a deterministic feed_bank")
            from environment.overall_environment.src.shuttle_feeder import (
                feed_sample_fingerprint,
            )
            from environment.overall_environment.src.stage3_target_bank_v2 import (
                Stage3TargetBank,
                load_target_bank,
                target_arrays,
            )

            feed_fingerprints = tuple(feed_sample_fingerprint(sample) for sample in self.feed_bank)
            if isinstance(impact_target_bank, str | Path):
                bank = load_target_bank(
                    impact_target_bank,
                    expected_feed_fingerprints=feed_fingerprints,
                )
            elif isinstance(impact_target_bank, Stage3TargetBank):
                bank = impact_target_bank.aligned_to_feeds(feed_fingerprints)
            else:
                raise ValueError("impact_recovery_v2 requires a validated Stage3TargetBank or path")
            self.impact_target_bank = bank
            self._target_arrays = target_arrays(bank)
            if self.task_curriculum_stage is not None:
                self.set_task_curriculum_stage(self.task_curriculum_stage)
        elif self.task_curriculum_stage is not None:
            raise ValueError("task_curriculum_stage is only valid for impact_recovery_v2")

        if lab_controller is not None and base_policy_artifact is not None:
            raise ValueError("LAB control and legacy full-action residual control are mutually exclusive")
        if (lab_controller is None) != (lab_state_builder is None):
            raise ValueError("lab_controller and lab_state_builder must be provided together")
        self.lab_controller = lab_controller
        self.lab_state_builder = lab_state_builder
        self.curriculum = curriculum
        self.filter_finger_observation = bool(
            lab_controller is not None if filter_finger_observation is None else filter_finger_observation
        )
        self.swing_duration_s = float(swing_duration_s)
        self.contact_phase = float(contact_phase)
        self.swing_phase_advance_s = float(swing_phase_advance_s)
        if self.swing_duration_s <= 0.0:
            raise ValueError("swing_duration_s must be positive")
        if not 0.0 <= self.contact_phase <= 1.0:
            raise ValueError("contact_phase must lie in [0, 1]")
        if not math.isfinite(self.swing_phase_advance_s) or self.swing_phase_advance_s < 0.0:
            raise ValueError("swing_phase_advance_s must be finite and non-negative")
        self._effective_ctrlrange_hash: str | None = None
        self._control_manifest_cache: dict[str, Any] | None = None
        if self.lab_controller is not None:
            if int(self.lab_controller.router.full_size) != int(self.model.nu):
                raise ValueError("Stage-3 LAB router does not cover every model actuator")
            if int(self.lab_state_builder.expected_state_dim) != int(self.lab_controller.lab_state_size):
                raise ValueError("LAB state builder and latent runtime dimensions differ")
            from environment.overall_environment.src.stage3_lab import (
                apply_teacher_body_ctrlrange,
            )

            self._effective_ctrlrange_hash = apply_teacher_body_ctrlrange(self.model, self.lab_controller)

        self.base_bridge = None
        if base_policy_artifact is None and (residual_scale_overrides or residual_scale_schedule):
            raise ValueError("residual scale overrides/schedule require base_policy_artifact")
        self.residual_authority_progress = float(residual_authority_progress)
        if not math.isfinite(self.residual_authority_progress) or not 0.0 <= self.residual_authority_progress <= 1.0:
            raise ValueError("residual_authority_progress must be finite and lie in [0, 1]")
        if base_policy_artifact is not None:
            from environment.overall_environment.src.base_swing_bridge import (
                BaseSwingBridge,
                SwingPhaseConfig,
            )

            self.base_bridge = BaseSwingBridge(
                base_policy_artifact,
                self.model,
                residual_scale=residual_scale,
                residual_scale_overrides=residual_scale_overrides,
                residual_scale_schedule=residual_scale_schedule,
                phase_config=SwingPhaseConfig(
                    swing_duration_s=self.swing_duration_s,
                    contact_phase=self.contact_phase,
                    phase_advance_s=self.swing_phase_advance_s,
                ),
                skill=base_skill,
            )

        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, READY_KEYFRAME)
        if self.keyframe_id < 0:
            raise ValueError(f"missing keyframe {READY_KEYFRAME!r} in {self.xml_path}")
        self._root_qadr = self._joint_qposadr(HUMAN_ROOT_FREEJOINT)
        self._root_dadr = self._joint_dofadr(HUMAN_ROOT_FREEJOINT)
        self._shuttle_qadr = self._joint_qposadr(SHUTTLE_FREEJOINT)
        self._shuttle_dadr = self._joint_dofadr(SHUTTLE_FREEJOINT)
        self._stringbed_site = self._site_id(STRINGBED_CENTER_SITE)
        self._cork_site = self._site_id(self.physics.cfg.shuttle_contact_site_name)
        self._palm_site = self._site_id(PALM_SITE)
        self._racket_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
        if self._racket_body < 0:
            raise ValueError("missing body 'overall_racket'")
        self._qpos_obs_index = self._build_qpos_obs_index()
        self._qvel_obs_index = self._build_qvel_obs_index()
        ready_keep = np.ones(self.model.nq, dtype=bool)
        ready_keep[self._root_qadr : self._root_qadr + 7] = False
        ready_keep[self._shuttle_qadr : self._shuttle_qadr + 7] = False
        racket_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            RACKET_FREEJOINT,
        )
        self._racket_qadr = None if racket_joint < 0 else int(self.model.jnt_qposadr[racket_joint])
        if self._racket_qadr is not None:
            # Compatibility with archived weld scenes.  The production
            # exact-child racket has no generalized coordinate.
            ready_keep[self._racket_qadr : self._racket_qadr + 7] = False
        self._ready_qpos_index = np.nonzero(ready_keep)[0]
        self._ready_qpos = np.asarray(self.model.key_qpos[self.keyframe_id], dtype=float)[self._ready_qpos_index]

        self.state = IncomingHitState.INCOMING
        self.step_index = 0
        self.termination_reason: str | None = None
        self.feed: FeedSample | None = None
        self._hit_closing_speed = 0.0
        self._best_shuttle_proximity_potential = 0.0
        self._best_timed_intercept_potential = 0.0
        self._best_racket_direction_potential = 0.0
        self._closest_racket_distance_m = float("inf")
        self._closest_racket_direction_score = 0.0
        self._closest_racket_direction_terminal_score = 0.0
        self._hit_rewarded = False
        self._crossed_net_rewarded = False
        self._invalid_net_crossed = False
        self._landing_region: str | None = None
        self._landing_rewarded = False
        self._feed_index = 0
        self._active_target_index = 0
        self._impact_diag: dict[str, Any] | None = None
        self._landing_xy: np.ndarray | None = None
        self._apex_height_m = 0.0
        self._recovery_step = 0
        self._recovery_active = False
        self._recovery_complete = False
        self._flight_resolved = False
        self._last_v2_metrics: dict[str, float] = {}
        self.lab_state: np.ndarray | None = None
        self._last_lab_output: Any | None = None
        self._last_lab_input_state: np.ndarray | None = None
        self._previous_raw_latent: np.ndarray | None = None

    # ---- id helpers -----------------------------------------------------

    def _joint_qposadr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name!r}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dofadr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name!r}")
        return int(self.model.jnt_dofadr[joint_id])

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"missing site {name!r}")
        return int(site_id)

    def _build_qpos_obs_index(self) -> np.ndarray:
        keep = np.ones(self.model.nq, dtype=bool)
        keep[self._shuttle_qadr : self._shuttle_qadr + 7] = False  # replaced by relative features
        keep[self._root_qadr : self._root_qadr + 2] = False  # drop absolute root x/y
        if self.filter_finger_observation:
            self._remove_finger_joint_coordinates(keep, qpos=True)
        return np.nonzero(keep)[0]

    def _build_qvel_obs_index(self) -> np.ndarray:
        keep = np.ones(self.model.nv, dtype=bool)
        keep[self._shuttle_dadr : self._shuttle_dadr + 6] = False
        if self.filter_finger_observation:
            self._remove_finger_joint_coordinates(keep, qpos=False)
        return np.nonzero(keep)[0]

    def _remove_finger_joint_coordinates(self, keep: np.ndarray, *, qpos: bool) -> None:
        from musclemimic.utils.finger_isolation import finger_joint_side

        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if finger_joint_side(name) is None:
                continue
            joint_type = int(self.model.jnt_type[joint_id])
            if qpos:
                address = int(self.model.jnt_qposadr[joint_id])
                width = (
                    7
                    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
                    else (4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
                )
            else:
                address = int(self.model.jnt_dofadr[joint_id])
                width = (
                    6
                    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
                    else (3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
                )
            keep[address : address + width] = False

    # ---- spaces ----------------------------------------------------------

    @property
    def action_size(self) -> int:
        if self.lab_controller is not None:
            return int(self.lab_controller.task_action_size)
        return int(self.model.nu)

    @property
    def full_action_size(self) -> int:
        return int(self.model.nu)

    @property
    def expects_raw_latent(self) -> bool:
        return self.lab_controller is not None

    @property
    def control_manifest(self) -> dict[str, Any]:
        if self._control_manifest_cache is not None:
            return self._control_manifest_cache
        if self.lab_controller is None:
            payload: dict[str, Any] = {
                "schema_version": (
                    "incoming_hit_direct_action_v1"
                    if self.task_profile == LEGACY_PROFILE
                    else "incoming_hit_direct_action_impact_recovery_v2"
                )
            }
        else:
            payload = dict(self.lab_controller.control_manifest)
            payload["lab_state_schema_hash"] = self.lab_state_builder.schema_hash
        base_bridge = getattr(self, "base_bridge", None)
        if base_bridge is not None:
            payload["frozen_base_residual"] = base_bridge.control_binding
        from environment.overall_environment.src.stage3_lab import (
            stage3_attachment_report,
        )

        payload["racket_attachment"] = stage3_attachment_report(
            self.model,
            self.xml_path,
            contract_path=getattr(self, "racket_attachment_contract_path", None),
        )
        payload["filter_finger_observation"] = self.filter_finger_observation
        min_return_clearance = getattr(self, "min_return_net_clearance_m", None)
        swing_phase_advance = float(getattr(self, "swing_phase_advance_s", 0.0))
        environment_abi = {
            "schema_version": (
                "incoming_hit_environment_v1"
                if self.task_profile == LEGACY_PROFILE
                else "incoming_hit_environment_impact_recovery_v2"
            ),
            "scene_sha256": hashlib.sha256(self.xml_path.read_bytes()).hexdigest(),
            "effective_ctrlrange_hash": self._effective_ctrlrange_hash,
            "full_action_size": self.full_action_size,
            "control_substeps": self.control_substeps,
            "max_episode_steps": self.max_episode_steps,
            "reward_weights": {
                name: value
                for name, value in self.reward_weights.items()
                if not (
                    value == 0.0
                    and (
                        name in {"return_clearance", "outgoing_vertical", "outgoing_forward"}
                        or (name == "invalid_net_crossing" and min_return_clearance is None)
                    )
                )
            },
            "player_half_sign": self.player_half_sign,
            "singles": self.singles,
            "terminate_on_body_fall": self.terminate_on_body_fall,
            "swing_duration_s": self.swing_duration_s,
            "contact_phase": self.contact_phase,
        }
        # Only event_rebound runs are governed by the single-impulse/cooldown
        # rule, so recording it unconditionally would both assert semantics an
        # any_stringbed_contact run never used and break byte-for-byte manifest
        # compatibility with checkpoints predating the declaration.
        if self.hit_event_mode == "event_rebound":
            environment_abi["event_rebound_contact_semantics"] = (
                "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
            )
        if swing_phase_advance != 0.0:
            environment_abi["swing_phase_advance_s"] = swing_phase_advance
        if min_return_clearance is not None:
            environment_abi["return_constraints"] = {
                "net_x_m": float(getattr(self, "return_net_x_m", 0.0)),
                "net_height_m": float(getattr(self, "return_net_height_m", 1.55)),
                "min_clearance_m": float(min_return_clearance),
                "desired_up_component": float(getattr(self, "desired_return_up_component", 0.40)),
            }
            if self.ballistic_return_score_softness_m != 0.35:
                environment_abi["return_constraints"]["ballistic_score_softness_m"] = (
                    self.ballistic_return_score_softness_m
                )
            if self.clearance_prediction_mode != VACUUM_CLEARANCE_PREDICTION_MODE:
                environment_abi["return_constraints"]["clearance_prediction_mode"] = self.clearance_prediction_mode
            if self.shuttle_proximity_softness_m != 0.35:
                environment_abi["return_constraints"]["shuttle_proximity_softness_m"] = (
                    self.shuttle_proximity_softness_m
                )
            if self.timed_intercept_softness_m != 0.30:
                environment_abi["return_constraints"]["timed_intercept_softness_m"] = self.timed_intercept_softness_m
            if self.direction_distance_softness_m != 0.45:
                environment_abi["return_constraints"]["direction_distance_softness_m"] = (
                    self.direction_distance_softness_m
                )
            if self.contact_guidance_reward_mode != "dense_per_step":
                environment_abi["return_constraints"]["contact_guidance_reward_mode"] = (
                    self.contact_guidance_reward_mode
                )
            if self.contact_guidance_reward_mode == "potential_event_direction":
                environment_abi["return_constraints"]["contact_guidance_discount"] = self.contact_guidance_discount
            if self.racket_velocity_direction_fraction != 0.30:
                environment_abi["return_constraints"]["racket_velocity_direction_fraction"] = (
                    self.racket_velocity_direction_fraction
                )
            if self.direction_reward_mode != "positive_projection":
                environment_abi["return_constraints"]["direction_reward_mode"] = self.direction_reward_mode
            if self.clearance_reward_mode != "positive_score":
                environment_abi["return_constraints"]["clearance_reward_mode"] = self.clearance_reward_mode
            if self.hit_event_mode != "any_stringbed_contact":
                environment_abi["return_constraints"]["hit_event_mode"] = self.hit_event_mode
            if self.racket_guidance_mode != "component_projection":
                environment_abi["return_constraints"]["racket_guidance_mode"] = self.racket_guidance_mode
            if self.racket_guidance_mode in {
                "inverse_impact_target",
                "inverse_impact_decomposed",
            }:
                environment_abi["return_constraints"].update(
                    {
                        "inverse_target_speed_m_s": self.inverse_target_speed_m_s,
                        "inverse_velocity_softness_m_s": (self.inverse_velocity_softness_m_s),
                    }
                )
        if self.reward_weights.get("return_clearance", 0.0) != 0.0:
            if self.contact_guidance_reward_mode == "closest_approach_event_direction":
                environment_abi["reward_semantics"] = (
                    "incoming_hit_drag_aware_closest_approach_event_direction_v30"
                    if self.clearance_prediction_mode == DRAG_AWARE_CLEARANCE_PREDICTION_MODE
                    else "incoming_hit_closest_approach_event_direction_v29"
                )
            elif self.contact_guidance_reward_mode == "potential_event_direction":
                environment_abi["reward_semantics"] = "incoming_hit_discounted_potential_event_direction_v27"
            elif self.contact_guidance_reward_mode == "event_direction":
                environment_abi["reward_semantics"] = "incoming_hit_event_direction_quality_v26"
            elif self.contact_guidance_reward_mode == "best_progress":
                environment_abi["reward_semantics"] = "incoming_hit_bounded_contact_progress_v23"
            elif (
                self.direction_reward_mode == "positive_projection"
                and self.clearance_reward_mode == "positive_score"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_decomposed"
                and self.reward_weights.get("miss", 0.0) != 0.0
            ):
                environment_abi["reward_semantics"] = "incoming_hit_wrist_hierarchical_quality_v19"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_decomposed"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_inverse_impact_decomposed_quality_v16"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_target"
            ):
                environment_abi["reward_semantics"] = (
                    "incoming_hit_quality_hierarchy_v15"
                    if self.clearance_reward_mode == "signed_centered" or self.reward_weights.get("miss", 0.0) != 0.0
                    else "incoming_hit_inverse_impact_target_guidance_v12"
                )
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "counterfactual_clearance_priority"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_counterfactual_clearance_priority_v11"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "counterfactual_rebound"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_counterfactual_rebound_guidance_v10"
            elif self.direction_reward_mode == "signed_projection" and self.hit_event_mode == "event_rebound":
                environment_abi["reward_semantics"] = "incoming_hit_effective_rebound_direction_v9"
            elif self.direction_reward_mode == "signed_projection":
                environment_abi["reward_semantics"] = "incoming_hit_signed_task_direction_v8"
            else:
                environment_abi["reward_semantics"] = (
                    "incoming_hit_non_saturating_ballistic_direction_v6"
                    if (
                        self.ballistic_return_score_softness_m != 0.35
                        or self.racket_velocity_direction_fraction != 0.30
                    )
                    else "incoming_hit_ballistic_legal_return_v5"
                )
        elif any(
            self.reward_weights.get(name, 0.0) != 0.0
            for name in (
                "shuttle_proximity",
                "timed_intercept",
                "racket_direction",
                "hit_speed",
                "return_direction",
            )
        ):
            environment_abi["reward_semantics"] = (
                "incoming_hit_legal_return_stability_v4"
                if min_return_clearance is not None
                else "incoming_hit_timed_cork_task_direction_v3"
            )
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            environment_abi.update(
                {
                    "task_profile": self.task_profile,
                    "target_bank_sha256": self.impact_target_bank.bank_sha256,
                    "v2_observation_size": V2_OBSERVATION_SIZE,
                    "recovery_horizon_steps": self.recovery_horizon_steps,
                    "task_curriculum_stage": self.task_curriculum_stage,
                }
            )
        payload["environment_abi"] = environment_abi
        payload["curriculum"] = None if self.curriculum is None else dict(vars(self.curriculum))
        if self.curriculum is not None and self.curriculum_feed_order != "difficulty_sorted":
            payload["curriculum_feed_order"] = self.curriculum_feed_order
        if self.curriculum_feed_order == "explicit_fingerprint_order":
            payload["seed_feed_fingerprints"] = list(self.seed_feed_fingerprints)
        # Keep the legacy LAB control manifest byte-for-byte compatible with
        # checkpoints produced before the impact/recovery task existed.  The
        # policy-only ABI is a v2 contract because train/eval target banks may
        # legitimately differ while the policy interface stays fixed.
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            payload["policy_abi_hash"] = incoming_hit_policy_abi_hash(payload)
        payload_without_hash = dict(payload)
        payload_without_hash.pop("control_hash", None)
        payload["control_hash"] = hashlib.sha256(
            json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._control_manifest_cache = payload
        return payload

    @property
    def control_hash(self) -> str | None:
        return self.control_manifest.get("control_hash")

    @property
    def policy_abi_hash(self) -> str | None:
        return self.control_manifest.get("policy_abi_hash")

    @property
    def observation_size(self) -> int:
        legacy_size = int(self._qpos_obs_index.size + self._qvel_obs_index.size + 12 + 9 + 8)
        return legacy_size + (V2_OBSERVATION_SIZE if self.task_profile == IMPACT_RECOVERY_PROFILE else 0)

    # ---- core API --------------------------------------------------------

    def set_task_curriculum_stage(self, stage_name: str) -> None:
        """Select one explicit CPU curriculum stage.

        GPU training carries these values in :class:`EnvState`; this setter is
        the CPU/evaluation counterpart used for deterministic contract checks
        and optional static-target rollouts.
        """
        if self.task_profile != IMPACT_RECOVERY_PROFILE:
            raise ValueError("Stage-3 v2 task curriculum requires impact_recovery_v2")
        from environment.overall_environment.src.stage3_task_curriculum_v2 import (
            V2_REWARD_TERM_ORDER,
            stage_by_name,
        )

        stage = stage_by_name(stage_name)
        self.task_curriculum_stage = stage.name
        self._v2_environment_mode_code = stage.environment_mode_code
        self._v2_reward_mask = dict(zip(V2_REWARD_TERM_ORDER, stage.reward_mask, strict=False))
        self._control_manifest_cache = None

    def _prepare_static_shuttle(self) -> None:
        if self.task_profile != IMPACT_RECOVERY_PROFILE or self._v2_environment_mode_code == 3:
            return
        qadr, dadr = self._shuttle_qadr, self._shuttle_dadr
        if self._v2_environment_mode_code in (0, 1):
            position = self._active_target_value("impact_position_world").copy()
            position[2] = 100.0
        elif self._hit_rewarded or self._recovery_active or self._recovery_complete:
            return
        else:
            position = self._active_target_value("impact_position_world")
        self.data.qpos[qadr : qadr + 3] = position
        self.data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, feed_index: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        if self.feed_bank:
            if feed_index is None:
                feed_index = int(self.rng.integers(len(self.feed_bank)))
            self._feed_index = int(feed_index % len(self.feed_bank))
            self.feed = self.feed_bank[self._feed_index]
        else:
            self.feed = sample_feed(self.rng, self.feed_config, self.hit_window)
            self._feed_index = 0
        self._active_target_index = self._feed_index

        qadr, dadr = self._shuttle_qadr, self._shuttle_dadr
        self.data.qpos[qadr : qadr + 3] = self.feed.launch_pos
        self.data.qpos[qadr + 3 : qadr + 7] = launch_quat_from_velocity(self.feed.launch_vel)
        self.data.qvel[dadr : dadr + 3] = self.feed.launch_vel
        self.data.qvel[dadr + 3 : dadr + 6] = 0.0

        self.physics.reset()
        self.state = IncomingHitState.INCOMING
        self.step_index = 0
        self.termination_reason = None
        self._hit_closing_speed = 0.0
        self._best_shuttle_proximity_potential = 0.0
        self._best_timed_intercept_potential = 0.0
        self._best_racket_direction_potential = 0.0
        self._closest_racket_distance_m = float("inf")
        self._closest_racket_direction_score = 0.0
        self._closest_racket_direction_terminal_score = 0.0
        self._predicted_net_clearance_m = 0.0
        self._return_clearance_score = 0.0
        self._return_direction_signed_score = 0.0
        self._hit_rewarded = False
        self._hit_event_direction_reward_score = 0.0
        self._crossed_net_rewarded = False
        self._invalid_net_crossed = False
        self._landing_region = None
        self._landing_rewarded = False
        self._impact_diag = None
        self._landing_xy = None
        self._apex_height_m = float(self.data.qpos[self._shuttle_qadr + 2])
        self._recovery_step = 0
        self._recovery_active = False
        self._recovery_complete = False
        self._flight_resolved = self._v2_environment_mode_code != 3
        self._last_v2_metrics = {}

        mujoco.mj_forward(self.model, self.data)
        self._prepare_static_shuttle()
        if self.lab_controller is not None:
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=self._swing_phase(),
            )
            self._last_lab_output = None
            self._last_lab_input_state = None
            self._previous_raw_latent = None
        obs = self._observation()
        return obs, self._info({})

    def step(
        self,
        action: np.ndarray,
        *,
        effective_latent_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        ``effective_latent_override`` is a fail-closed, CPU-evaluation-only
        intervention hook used by the post-Stage3 task-causal evaluator.  It
        is rejected outside LAB control and leaves the default training and
        evaluation path byte-for-byte unchanged when omitted.
        """
        if self.feed is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        swing_phase = 0.0
        applied_action = action
        if self.lab_controller is not None:
            swing_phase = self._swing_phase()
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=swing_phase,
            )
            self._last_lab_input_state = np.asarray(self.lab_state, dtype=float).copy()
            if effective_latent_override is None:
                self._last_lab_output = self.lab_controller.decode_task_numpy(
                    lab_state=self.lab_state,
                    task_action=action,
                )
            else:
                self._last_lab_output = self.lab_controller.decode_task_with_latent_override_numpy(
                    lab_state=self.lab_state,
                    task_action=action,
                    effective_latent=effective_latent_override,
                )
            applied_action = np.asarray(self._last_lab_output.full_action, dtype=float)
            ctrl = normalized_action_to_model_ctrl(self.model, applied_action)
        elif self.base_bridge is not None:
            if effective_latent_override is not None:
                raise ValueError("effective latent override requires Stage-3 LAB control")
            elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
            swing_phase = self.base_bridge.phase_config.phase_at(elapsed, float(self.feed.intercept_time_s))
            combined, _base = self.base_bridge.combined_action(
                self.model,
                self.data,
                action,
                phase=swing_phase,
                residual_authority_progress=self.residual_authority_progress,
            )
            applied_action = np.asarray(combined, dtype=float)
            ctrl = normalized_action_to_model_ctrl(self.model, combined)
        else:
            if effective_latent_override is not None:
                raise ValueError("effective latent override requires Stage-3 LAB control")
            ctrl = normalized_action_to_model_ctrl(self.model, action)
        self.data.ctrl[:] = ctrl

        self._prepare_static_shuttle()
        if self._landing_xy is not None:
            self.data.qpos[self._shuttle_qadr + 2] = GROUND_REST_HEIGHT_M
            self.data.qvel[self._shuttle_dadr : self._shuttle_dadr + 6] = 0.0

        previous_elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        previous_shuttle_position = np.asarray(
            self.data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3],
            dtype=float,
        ).copy()
        hit_this_step = False
        contact_this_step = False
        rebound_this_step = False
        max_closing_speed = 0.0
        best_contact: dict[str, Any] | None = None
        event_contact: dict[str, np.ndarray] | None = None
        for _ in range(self.control_substeps):
            diag = self.physics.substep(self.model, self.data)
            contact = diag["stringbed"]
            # A high-speed cork can cross the proxy plane between substeps.
            # The side-dependent normal then flips and reports a positive
            # separating velocity even though the stringbed is actively
            # applying the real rebound force.  Contact is the 0/1 event;
            # normal-speed sign is not a valid contact classifier.
            normal_speed = abs(float(contact.get("relative_normal_velocity", 0.0)))
            active_contact = bool(contact.get("active", False)) and normal_speed > 0.05
            contact_this_step = contact_this_step or active_contact
            if bool(diag["event_rebound_used"]):
                rebound_this_step = True
                max_closing_speed = max(max_closing_speed, normal_speed)
                if event_contact is None:
                    event_contact = {
                        "shuttle_velocity_before_world_m_s": np.asarray(
                            diag["event_shuttle_velocity_before_world_m_s"], dtype=float
                        ).copy(),
                        "shuttle_velocity_after_world_m_s": np.asarray(
                            diag["event_shuttle_velocity_after_world_m_s"], dtype=float
                        ).copy(),
                        "racket_surface_velocity_world_m_s": np.asarray(
                            diag["event_racket_surface_velocity_world_m_s"], dtype=float
                        ).copy(),
                        "stringbed_normal_world": np.asarray(diag["event_stringbed_normal_world"], dtype=float).copy(),
                    }
            elif active_contact:
                max_closing_speed = max(max_closing_speed, normal_speed)
            if (bool(diag["event_rebound_used"]) or active_contact) and (
                best_contact is None or float(contact.get("rho2", np.inf)) < float(best_contact.get("rho2", np.inf))
            ):
                best_contact = dict(contact)
                best_contact["position_world"] = np.asarray(
                    self.data.site_xpos[self._stringbed_site], dtype=float
                ).copy()

        hit_this_step = (
            rebound_this_step if self.hit_event_mode == "event_rebound" else (rebound_this_step or contact_this_step)
        )

        self.step_index += 1
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        virtual_hit = bool(
            self.task_profile == IMPACT_RECOVERY_PROFILE
            and self._v2_environment_mode_code == 1
            and self.state == IncomingHitState.INCOMING
            and previous_elapsed < float(self._active_target_value("impact_time_s")) <= elapsed
        )
        if virtual_hit:
            hit_this_step = True
            stringbed_position = np.asarray(self.data.site_xpos[self._stringbed_site], dtype=float)
            position_error = float(
                np.linalg.norm(stringbed_position - self._active_target_value("impact_position_world"))
            )
            best_contact = {
                "rho2": (position_error / 0.15) ** 2,
                "normal_world": self._stringbed_normal(),
            }
        if self.lab_controller is not None:
            self.lab_state = self.lab_state_builder.build_numpy(
                model=self.model,
                data=self.data,
                phase=self._swing_phase(),
            )
        if self.state == IncomingHitState.INCOMING and hit_this_step:
            self.state = IncomingHitState.HIT
            self._hit_closing_speed = max_closing_speed
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                self._impact_diag = {
                    "rho2": float((best_contact or {}).get("rho2", 1.0)),
                    "position_world": np.asarray(
                        (best_contact or {}).get(
                            "position_world",
                            self.data.site_xpos[self._stringbed_site],
                        ),
                        dtype=float,
                    ).copy(),
                    "normal_world": np.asarray(
                        (best_contact or {}).get("normal_world", self._stringbed_normal()),
                        dtype=float,
                    ),
                    "linear_velocity_world": self._stringbed_velocity(),
                    "angular_velocity_world": self._stringbed_angular_velocity(),
                    "impact_time_s": elapsed,
                }
                self._recovery_active = True
                self._recovery_complete = False
                self._recovery_step = 0
                if self._v2_environment_mode_code != 3:
                    self._flight_resolved = True
                    self.state = IncomingHitState.RECOVERY

        flight = self._flight_info()
        if self.min_return_net_clearance_m is None:
            raw_opponent_side = bool(
                np.sign(flight["shuttle_xyz"][0]) == self.player_half_sign * -1
                and abs(float(flight["shuttle_xyz"][0])) > 1.0e-9
            )
            crossing = {
                "crossed": raw_opponent_side,
                "valid": raw_opponent_side,
                "crossing_height_m": float(flight["shuttle_xyz"][2]),
                "clearance_m": float(flight["shuttle_xyz"][2]) - self.return_net_height_m,
            }
        else:
            crossing = classify_return_net_crossing(
                previous_shuttle_position,
                np.asarray(flight["shuttle_xyz"], dtype=float),
                player_half_sign=self.player_half_sign,
                net_x_m=self.return_net_x_m,
                net_height_m=self.return_net_height_m,
                min_clearance_m=self.min_return_net_clearance_m,
            )
        post_hit = self._hit_rewarded or self.state in (
            IncomingHitState.HIT,
            IncomingHitState.FLIGHT,
        )
        valid_crossing_event = bool(post_hit and crossing["valid"])
        invalid_crossing_event = bool(post_hit and crossing["crossed"] and not crossing["valid"])
        flight.update(
            {
                "crossed_net": bool(self._crossed_net_rewarded or valid_crossing_event),
                "valid_net_crossing_event": valid_crossing_event,
                "invalid_net_crossing_event": invalid_crossing_event,
                "net_crossing_height_m": float(crossing["crossing_height_m"]),
                "net_clearance_m": float(crossing["clearance_m"]),
            }
        )
        if self._hit_rewarded or self.state in (
            IncomingHitState.HIT,
            IncomingHitState.FLIGHT,
        ):
            self._apex_height_m = max(self._apex_height_m, float(flight["shuttle_xyz"][2]))
        if self.state == IncomingHitState.HIT and valid_crossing_event:
            self.state = IncomingHitState.FLIGHT

        terminated = False
        if bool(flight["landed"]) and self._landing_xy is None:
            if self.state == IncomingHitState.INCOMING:
                terminated = True
                self.termination_reason = "miss"
                self.state = IncomingHitState.DONE
            else:
                self._landing_region = str(flight["region"])
                self._landing_xy = np.asarray(flight["shuttle_xyz"][:2], dtype=float)
                if self.task_profile == IMPACT_RECOVERY_PROFILE:
                    self._flight_resolved = True
                    if not self._recovery_complete:
                        self.state = IncomingHitState.RECOVERY
                else:
                    terminated = True
                    self.termination_reason = "landed"
                    self.state = IncomingHitState.DONE

        if self._recovery_active and not hit_this_step:
            self._recovery_step += 1
            if self._recovery_step >= self._active_recovery_horizon():
                self._recovery_active = False
                self._recovery_complete = True

        if (
            self.task_profile == IMPACT_RECOVERY_PROFILE
            and self._flight_resolved
            and self._recovery_complete
            and not terminated
        ):
            terminated = True
            self.termination_reason = "recovery_and_flight_complete"
            self.state = IncomingHitState.DONE

        body_fall = self._root_height() < BODY_FALL_ROOT_HEIGHT_M
        if body_fall and self.terminate_on_body_fall and not terminated:
            terminated = True
            self.termination_reason = "body_fall"
            self.state = IncomingHitState.DONE

        obs = self._observation()
        if not np.isfinite(obs).all():
            terminated = True
            self.termination_reason = "non_finite"
            self.state = IncomingHitState.DONE
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        truncated = False
        if not terminated and self.step_index >= self.max_episode_steps:
            truncated = True
            self.termination_reason = "time_limit"
            self.state = IncomingHitState.DONE

        reward_terms = self._reward_terms(
            applied_action,
            residual_action=action,
            flight=flight,
            hit_this_step=hit_this_step,
            event_contact=event_contact,
            body_fall=body_fall,
        )
        # ``flight`` is sampled before reward construction, while the first
        # real hit computes these return-quality values inside
        # ``_reward_terms``. Refresh the transition payload afterwards so a
        # CPU teacher audit sees the same event metrics as MJX rather than
        # stale reset-time zeros.
        flight.update(
            {
                "predicted_net_clearance_m": self._predicted_net_clearance_m,
                "return_clearance_score": self._return_clearance_score,
                "return_direction_signed_score": (self._return_direction_signed_score),
            }
        )
        reward = float(sum(reward_terms.values()))

        info = self._info(
            {
                "reward_terms": reward_terms,
                "flight": flight,
                "hit_this_step": hit_this_step,
                "stringbed_contact_this_step": contact_this_step,
                "event_rebound_this_step": rebound_this_step,
                "event_shuttle_velocity_before_world_m_s": (
                    np.zeros(3, dtype=float)
                    if event_contact is None
                    else event_contact["shuttle_velocity_before_world_m_s"].copy()
                ),
                "event_impulse_velocity_after_world_m_s": (
                    np.zeros(3, dtype=float)
                    if event_contact is None
                    else event_contact["shuttle_velocity_after_world_m_s"].copy()
                ),
                "event_racket_surface_velocity_world_m_s": (
                    np.zeros(3, dtype=float)
                    if event_contact is None
                    else event_contact["racket_surface_velocity_world_m_s"].copy()
                ),
                "event_stringbed_normal_world": (
                    np.zeros(3, dtype=float)
                    if event_contact is None
                    else event_contact["stringbed_normal_world"].copy()
                ),
                "hit_closing_speed_m_s": self._hit_closing_speed,
                "hit_contact_speed_m_s": self._hit_closing_speed,
                "hit_event_direction_reward_score": (
                    self._hit_event_direction_reward_score
                    if hit_this_step
                    and self.contact_guidance_reward_mode
                    in {
                        "event_direction",
                        "potential_event_direction",
                        "closest_approach_event_direction",
                    }
                    else 0.0
                ),
                "closest_approach_distance_m": (
                    self._closest_racket_distance_m if math.isfinite(self._closest_racket_distance_m) else 0.0
                ),
                "closest_approach_direction_score": (self._closest_racket_direction_score),
                "closest_approach_terminal_direction_score": (self._closest_racket_direction_terminal_score),
                "body_fall": bool(body_fall),
                "landing_region": self._landing_region,
                "invalid_net_crossed": self._invalid_net_crossed,
                "predicted_net_clearance_m": self._predicted_net_clearance_m,
                "return_clearance_score": self._return_clearance_score,
                "return_direction_signed_score": (self._return_direction_signed_score),
                "crossed_net": bool(flight["crossed_net"]),
                "valid_net_cross_event": bool(flight["valid_net_crossing_event"]),
                "swing_phase": swing_phase,
                **self._lab_diagnostics(action, full_action=applied_action),
            }
        )
        return obs, reward, terminated, truncated, info

    def _swing_phase(self) -> float:
        if self.feed is None:
            return 0.0
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        start = (
            float(self.feed.intercept_time_s) - self.swing_phase_advance_s - self.contact_phase * self.swing_duration_s
        )
        return float(np.clip((elapsed - start) / self.swing_duration_s, 0.0, 1.0))

    def _lab_diagnostics(
        self,
        raw_latent: np.ndarray,
        *,
        full_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self._last_lab_output is None or self._last_lab_input_state is None:
            # The pure 354-D policy has no latent/LAB diagnostics, but it must
            # still expose the physical-control quantities shared by both
            # branches of the Stage-3 comparison.  Keeping these names and
            # definitions identical prevents the direct branch from either
            # bypassing the energy/saturation gates or receiving fabricated
            # latent/OOD values.
            direct_action = np.asarray(
                raw_latent if full_action is None else full_action,
                dtype=float,
            )
            diagnostics = {
                "control_finite": float(np.all(np.isfinite(direct_action))),
                "body_action_rms": float(np.sqrt(np.mean(np.square(direct_action)))),
                "normalized_control_energy": float(np.mean(np.square(direct_action))),
                "body_action_saturation_fraction": float(np.mean(np.abs(direct_action) > 0.98)),
                "full_action_saturation_fraction": float(np.mean(np.abs(direct_action) > 0.98)),
                "muscle_power_abs_mean": float(
                    np.mean(
                        np.abs(
                            np.asarray(self.data.actuator_force, dtype=float)
                            * np.asarray(self.data.actuator_velocity, dtype=float)
                        )
                    )
                ),
                "control_hash": self.control_hash,
            }
            base_bridge = getattr(self, "base_bridge", None)
            if base_bridge is not None and base_bridge.residual_override_indices.size:
                override_ids = base_bridge.residual_override_indices
                raw_residual = np.asarray(raw_latent, dtype=float)[override_ids]
                override_action = direct_action[override_ids]
                diagnostics.update(
                    {
                        "residual_override_action_rms": float(np.sqrt(np.mean(np.square(raw_residual)))),
                        "residual_override_composed_saturation_fraction": float(
                            np.mean(np.abs(override_action) > 0.98)
                        ),
                    }
                )
            return diagnostics
        output = self._last_lab_output
        normalizer = getattr(self.lab_controller.runtime, "normalizer", None)
        if normalizer is None:
            # Lightweight test runtimes may omit normalization.  Emit
            # non-finite diagnostics so any production promotion report fails
            # closed instead of inventing an in-distribution score.
            unclipped_state_z = np.full_like(np.asarray(self._last_lab_input_state, dtype=float), np.nan)
        else:
            unclipped_state_z = (
                np.asarray(self._last_lab_input_state, dtype=float) - np.asarray(normalizer.mean, dtype=float)
            ) / np.asarray(normalizer.std, dtype=float)
        task_action = np.asarray(raw_latent, dtype=float)
        raw = np.asarray(output.raw_latent, dtype=float)
        right_grip = np.asarray(output.right_grip_action, dtype=float)
        raw_rate = (
            0.0
            if self._previous_raw_latent is None
            else float(np.sqrt(np.mean(np.square(task_action - self._previous_raw_latent))))
        )
        self._previous_raw_latent = task_action.copy()
        diagnostics = {
            "control_finite": float(np.all(np.isfinite(np.asarray(output.full_action, dtype=float)))),
            "raw_latent_rms": float(np.sqrt(np.mean(np.square(raw)))),
            "raw_latent_saturation": float(np.mean(np.abs(raw) > 2.0)),
            "latent_norm": float(np.linalg.norm(output.latent)),
            "prior_sigma_mean": float(np.mean(output.prior_sigma)),
            "lab_state_unclipped_z_rms": float(np.sqrt(np.mean(np.square(unclipped_state_z)))),
            "lab_state_ood_fraction": float(np.mean(np.abs(unclipped_state_z) > 5.0)),
            "body_action_rms": float(np.sqrt(np.mean(np.square(output.body_action)))),
            "right_grip_action_rms": (0.0 if right_grip.size == 0 else float(np.sqrt(np.mean(np.square(right_grip))))),
            "lambda_lab": float(output.lambda_lab),
            "raw_action_rate_rms": raw_rate,
            "normalized_control_energy": float(np.mean(np.square(np.asarray(output.full_action, dtype=float)))),
            "body_action_saturation_fraction": float(
                np.mean(np.abs(np.asarray(output.body_action, dtype=float)) > 0.98)
            ),
            "full_action_saturation_fraction": float(
                np.mean(np.abs(np.asarray(output.full_action, dtype=float)) > 0.98)
            ),
            "muscle_power_abs_mean": float(
                np.mean(
                    np.abs(
                        np.asarray(self.data.actuator_force, dtype=float)
                        * np.asarray(self.data.actuator_velocity, dtype=float)
                    )
                )
            ),
            "lab_state_schema_hash": self.lab_state_builder.schema_hash,
            "control_hash": self.control_hash,
        }
        if output.raw_bounded_residual is not None:
            diagnostics["bounded_residual_rms"] = float(np.sqrt(np.mean(np.square(output.raw_bounded_residual))))
        return diagnostics

    # ---- observation -----------------------------------------------------

    def _observation(self) -> np.ndarray:
        data = self.data
        qpos = np.asarray(data.qpos, dtype=float)[self._qpos_obs_index]
        qvel = np.asarray(data.qvel, dtype=float)[self._qvel_obs_index]

        root_pos = np.asarray(data.qpos[self._root_qadr : self._root_qadr + 3], dtype=float)
        shuttle_pos = np.asarray(data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3], dtype=float)
        shuttle_vel = np.asarray(data.qvel[self._shuttle_dadr : self._shuttle_dadr + 3], dtype=float)
        stringbed_pos = np.asarray(data.site_xpos[self._stringbed_site], dtype=float)
        stringbed_mat = np.asarray(data.site_xmat[self._stringbed_site], dtype=float).reshape(3, 3)
        face_normal = stringbed_mat[:, 2]
        face_vel = self._stringbed_velocity()

        shuttle_features = np.concatenate(
            [
                shuttle_pos - root_pos,
                shuttle_vel,
                shuttle_pos - stringbed_pos,
                shuttle_vel - face_vel,
            ]
        )
        racket_features = np.concatenate([stringbed_pos - root_pos, face_normal, face_vel])

        intercept = np.asarray(self.feed.intercept_point, dtype=float) if self.feed is not None else np.zeros(3)
        elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
        time_to_intercept = max(0.0, (self.feed.intercept_time_s if self.feed is not None else 0.0) - elapsed)
        # Task policy and frozen LAB prior must share the same intercept-aligned
        # swing clock.  Episode progress is not a swing phase when feed timing
        # varies and previously made the task policy condition on a conflicting
        # signal.
        phase = self._swing_phase()
        task_features = np.concatenate(
            [
                intercept - stringbed_pos,
                intercept - root_pos,
                [time_to_intercept, phase],
            ]
        )
        legacy = np.concatenate([qpos, qvel, shuttle_features, racket_features, task_features])
        if self.task_profile == LEGACY_PROFILE:
            return legacy
        target_position = self._active_target_value("impact_position_world")
        target_normal = self._active_target_value("stringbed_normal_world")
        target_linear = self._active_target_value("racket_linear_velocity_world")
        target_angular = self._active_target_value("racket_angular_velocity_world")
        landing_target = self._active_target_value("landing_target_xy")
        desired_apex = float(self._active_target_value("apex_height_m"))
        target_time = float(self._active_target_value("impact_time_s"))
        angular_velocity = self._stringbed_angular_velocity()
        recovery_progress = min(1.0, self._recovery_step / max(1, self._active_recovery_horizon()))
        v2_features = np.concatenate(
            [
                target_position - stringbed_pos,
                target_normal,
                target_linear - face_vel,
                target_angular - angular_velocity,
                landing_target - shuttle_pos[:2],
                [desired_apex - self._apex_height_m],
                [max(0.0, target_time - elapsed)],
                [recovery_progress],
                [self._ready_pose_error()],
                [float(self._recovery_active)],
            ]
        )
        if v2_features.shape != (V2_OBSERVATION_SIZE,):
            raise RuntimeError(f"Stage-3 v2 observation ABI drift: {v2_features.shape}")
        return np.concatenate([legacy, v2_features])

    def _stringbed_velocity(self) -> np.ndarray:
        vel6 = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._racket_body, vel6, 0)
        omega, v_origin = vel6[:3], vel6[3:]
        origin = np.asarray(self.data.xpos[self._racket_body], dtype=float)
        point = np.asarray(self.data.site_xpos[self._stringbed_site], dtype=float)
        return v_origin + np.cross(omega, point - origin)

    def _stringbed_angular_velocity(self) -> np.ndarray:
        vel6 = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._racket_body,
            vel6,
            0,
        )
        return vel6[:3]

    def _stringbed_normal(self) -> np.ndarray:
        return np.asarray(self.data.site_xmat[self._stringbed_site], dtype=float).reshape(3, 3)[:, 2]

    def _desired_return_direction(self, shuttle_position: np.ndarray) -> np.ndarray:
        """Broad high-clear direction toward the opponent, independent of demonstrations."""
        position = np.asarray(shuttle_position, dtype=float)
        direction = np.array(
            [
                -float(self.player_half_sign),
                -0.15 * float(position[1]),
                self.desired_return_up_component,
            ],
            dtype=float,
        )
        return direction / max(float(np.linalg.norm(direction)), 1.0e-9)

    def _active_target_value(self, name: str) -> np.ndarray:
        if self._target_arrays is None:
            raise RuntimeError("Stage-3 v2 target arrays are unavailable")
        return np.asarray(self._target_arrays[name][self._active_target_index])

    def _active_recovery_horizon(self) -> int:
        if self._target_arrays is None:
            return self.recovery_horizon_steps
        return int(self._target_arrays["recovery_horizon_steps"][self._active_target_index])

    def _ready_pose_error(self) -> float:
        current = np.asarray(self.data.qpos, dtype=float)[self._ready_qpos_index]
        return float(np.sqrt(np.mean(np.square(current - self._ready_qpos))))

    # ---- reward / termination helpers -------------------------------------

    def _reward_terms(
        self,
        action: np.ndarray,
        *,
        residual_action: np.ndarray | None = None,
        flight: dict[str, Any],
        hit_this_step: bool,
        event_contact: dict[str, np.ndarray] | None = None,
        body_fall: bool,
    ) -> dict[str, float]:
        w = self.reward_weights
        reward_keys = dict(DEFAULT_REWARD_WEIGHTS)
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            reward_keys.update(V2_REWARD_WEIGHTS)
        terms = dict.fromkeys(reward_keys, 0.0)
        first_hit = hit_this_step and not self._hit_rewarded
        dynamic_feed = self._v2_environment_mode_code == 3
        shuttle_proximity_potential = 0.0
        timed_intercept_potential = 0.0
        racket_direction_potential = 0.0

        if (
            self.state == IncomingHitState.INCOMING
            and self.feed is not None
            and (self.task_profile == LEGACY_PROFILE or dynamic_feed)
        ):
            stringbed_pos = np.asarray(self.data.site_xpos[self._stringbed_site], dtype=float)
            dist = float(np.linalg.norm(self.feed.intercept_point - stringbed_pos))
            terms["approach"] = w["approach"] * float(np.exp(-2.0 * dist))
            elapsed = self.step_index * self.control_substeps * float(self.model.opt.timestep)
            time_to_intercept = float(self.feed.intercept_time_s) - elapsed
            # Event hits in the fixed feed bank can occur as late as 0.14 s
            # after the nominal intercept.  The closest-approach mode keeps a
            # narrow late-contact margin while all legacy modes preserve their
            # original -0.08 s ABI.
            guidance_lower_bound = (
                -0.25 if self.contact_guidance_reward_mode == "closest_approach_event_direction" else -0.08
            )
            if guidance_lower_bound <= time_to_intercept <= 0.70:
                time_gate = float(np.exp(-0.5 * (time_to_intercept / 0.28) ** 2))
                shuttle_pos = np.asarray(
                    self.data.site_xpos[self._cork_site],
                    dtype=float,
                )
                shuttle_distance = float(np.linalg.norm(shuttle_pos - stringbed_pos))
                shuttle_proximity_potential = time_gate * float(
                    np.exp(-0.5 * (shuttle_distance / self.shuttle_proximity_softness_m) ** 2)
                )
                timed_intercept_potential = time_gate * float(
                    np.exp(-0.5 * (dist / self.timed_intercept_softness_m) ** 2)
                )
                if time_to_intercept <= 0.30:
                    face_normal = self._stringbed_normal()
                    ball_side = float(np.dot(shuttle_pos - stringbed_pos, face_normal))
                    signed_normal = face_normal * (1.0 if ball_side >= 0.0 else -1.0)
                    desired = self._desired_return_direction(shuttle_pos)
                    normal_projection = float(np.clip(np.dot(signed_normal, desired), -1.0, 1.0))
                    velocity_projection = float(
                        np.clip(
                            np.dot(self._stringbed_velocity(), desired) / 8.0,
                            -1.0,
                            1.0,
                        )
                    )
                    direction_gate = float(
                        np.exp(-0.5 * (time_to_intercept / 0.15) ** 2)
                        * np.exp(-0.5 * (shuttle_distance / self.direction_distance_softness_m) ** 2)
                    )
                    if self.racket_guidance_mode in {
                        "inverse_impact_target",
                        "inverse_impact_decomposed",
                    }:
                        inverse = inverse_impact_guidance_score(
                            np.asarray(flight["shuttle_velocity"], dtype=float),
                            self._stringbed_velocity(),
                            signed_normal,
                            desired,
                            impact_config=self.physics.cfg.impact,
                            target_outgoing_speed_m_s=(self.inverse_target_speed_m_s),
                            racket_velocity_softness_m_s=(self.inverse_velocity_softness_m_s),
                            racket_velocity_fraction=(self.racket_velocity_direction_fraction),
                        )
                        direction_score = float(
                            inverse[
                                "decomposed_score"
                                if self.racket_guidance_mode == "inverse_impact_decomposed"
                                else "score"
                            ]
                        )
                    elif self.racket_guidance_mode in {
                        "counterfactual_rebound",
                        "counterfactual_clearance_priority",
                    }:
                        counterfactual = counterfactual_rebound_guidance_score(
                            shuttle_pos,
                            np.asarray(flight["shuttle_velocity"], dtype=float),
                            self._stringbed_velocity(),
                            signed_normal,
                            desired,
                            player_half_sign=self.player_half_sign,
                            impact_config=self.physics.cfg.impact,
                            net_x_m=self.return_net_x_m,
                            net_height_m=self.return_net_height_m,
                            min_clearance_m=(
                                0.0 if self.min_return_net_clearance_m is None else self.min_return_net_clearance_m
                            ),
                            clearance_softness_m=(self.ballistic_return_score_softness_m),
                            quality_mode=(
                                "clearance_priority"
                                if self.racket_guidance_mode == "counterfactual_clearance_priority"
                                else "balanced_shifted"
                            ),
                        )
                        direction_score = float(counterfactual["score"])
                    else:
                        if self.direction_reward_mode == "signed_projection":
                            normal_score = normal_projection
                            velocity_score = velocity_projection
                        else:
                            normal_score = max(normal_projection, 0.0)
                            velocity_score = max(velocity_projection, 0.0)
                        direction_score = (
                            1.0 - self.racket_velocity_direction_fraction
                        ) * normal_score + self.racket_velocity_direction_fraction * velocity_score
                    if (
                        self.contact_guidance_reward_mode == "closest_approach_event_direction"
                        and shuttle_distance < self._closest_racket_distance_m
                    ):
                        self._closest_racket_distance_m = shuttle_distance
                        distance_gate = float(
                            np.exp(-0.5 * (shuttle_distance / self.direction_distance_softness_m) ** 2)
                        )
                        self._closest_racket_direction_score = float(
                            np.clip(
                                distance_gate * (2.0 * direction_score - 1.0),
                                -1.0,
                                1.0,
                            )
                        )
                    if self.contact_guidance_reward_mode == "potential_event_direction":
                        racket_direction_potential = direction_gate * (2.0 * direction_score - 1.0)
                    else:
                        racket_direction_potential = direction_gate * direction_score

        event_direction_score: float | None = None
        if (
            first_hit
            and self.contact_guidance_reward_mode
            in {
                "event_direction",
                "potential_event_direction",
                "closest_approach_event_direction",
            }
            and (self.task_profile == LEGACY_PROFILE or dynamic_feed)
        ):
            if event_contact is None:
                raise RuntimeError("event-direction hit is missing the exact event-rebound snapshot")
            event_inverse = inverse_impact_guidance_score(
                event_contact["shuttle_velocity_before_world_m_s"],
                event_contact["racket_surface_velocity_world_m_s"],
                event_contact["stringbed_normal_world"],
                self._desired_return_direction(flight["shuttle_xyz"]),
                impact_config=self.physics.cfg.impact,
                target_outgoing_speed_m_s=self.inverse_target_speed_m_s,
                racket_velocity_softness_m_s=self.inverse_velocity_softness_m_s,
                racket_velocity_fraction=self.racket_velocity_direction_fraction,
            )
            event_direction_score = float(np.clip(2.0 * float(event_inverse["decomposed_score"]) - 1.0, -1.0, 1.0))
            self._hit_event_direction_reward_score = event_direction_score

        if self.contact_guidance_reward_mode in {
            "best_progress",
            "event_direction",
            "potential_event_direction",
            "closest_approach_event_direction",
        }:
            proximity_progress, self._best_shuttle_proximity_potential = _bounded_best_progress(
                shuttle_proximity_potential,
                self._best_shuttle_proximity_potential,
            )
            intercept_progress, self._best_timed_intercept_potential = _bounded_best_progress(
                timed_intercept_potential,
                self._best_timed_intercept_potential,
            )
            terms["shuttle_proximity"] = w["shuttle_proximity"] * proximity_progress
            terms["timed_intercept"] = w["timed_intercept"] * intercept_progress
            if self.contact_guidance_reward_mode == "best_progress":
                direction_progress, self._best_racket_direction_potential = _bounded_best_progress(
                    racket_direction_potential,
                    self._best_racket_direction_potential,
                )
                terms["racket_direction"] = w["racket_direction"] * direction_progress
            elif self.contact_guidance_reward_mode == "event_direction":
                if event_direction_score is not None:
                    terms["racket_direction"] = w["racket_direction"] * event_direction_score
            elif self.contact_guidance_reward_mode == "closest_approach_event_direction":
                if event_direction_score is not None:
                    terms["racket_direction"] = w["racket_direction"] * event_direction_score
                elif self.state == IncomingHitState.DONE and not self._hit_rewarded:
                    self._closest_racket_direction_terminal_score = self._closest_racket_direction_score
                    terms["racket_direction"] = w["racket_direction"] * self._closest_racket_direction_terminal_score
            else:
                terminal_without_event = bool(
                    self.state == IncomingHitState.DONE and not first_hit and not self._hit_rewarded
                )
                direction_increment, self._best_racket_direction_potential = _discounted_event_direction_increment(
                    self._best_racket_direction_potential,
                    racket_direction_potential,
                    discount=self.contact_guidance_discount,
                    event_score=event_direction_score,
                    terminal_without_event=terminal_without_event,
                )
                terms["racket_direction"] = w["racket_direction"] * direction_increment
        else:
            terms["shuttle_proximity"] = w["shuttle_proximity"] * shuttle_proximity_potential
            terms["timed_intercept"] = w["timed_intercept"] * timed_intercept_potential
            terms["racket_direction"] = w["racket_direction"] * racket_direction_potential

        if first_hit:
            self._hit_rewarded = True
            if self.task_profile == LEGACY_PROFILE or dynamic_feed:
                # The main task signal is a literal real-contact 0/1 event.
                # Speed quality remains a separate continuous term so a soft
                # first contact still teaches PPO which state/action caused it.
                terms["hit_bonus"] = w["hit_bonus"]
                terms["hit_speed"] = w["hit_speed"] * min(1.0, self._hit_closing_speed / 8.0)
                outgoing = np.asarray(flight["shuttle_velocity"], dtype=float)
                outgoing_speed = float(np.linalg.norm(outgoing))
                terms["outgoing_vertical"] = w["outgoing_vertical"] * float(np.clip(outgoing[2] / 6.0, -1.0, 1.0))
                terms["outgoing_forward"] = w["outgoing_forward"] * float(
                    np.clip((-self.player_half_sign * outgoing[0]) / 10.0, -1.0, 1.0)
                )
                if outgoing_speed > 1.0e-9:
                    desired = self._desired_return_direction(flight["shuttle_xyz"])
                    direction_projection = float(
                        np.clip(
                            np.dot(outgoing / outgoing_speed, desired),
                            -1.0,
                            1.0,
                        )
                    )
                    self._return_direction_signed_score = direction_projection
                    direction_score = (
                        direction_projection
                        if self.direction_reward_mode == "signed_projection"
                        else max(direction_projection, 0.0)
                    )
                    terms["return_direction"] = (
                        w["return_direction"] * direction_score * min(1.0, outgoing_speed / 10.0)
                    )
                clearance_kwargs = {
                    "player_half_sign": self.player_half_sign,
                    "net_x_m": self.return_net_x_m,
                    "net_height_m": self.return_net_height_m,
                    "min_clearance_m": (
                        0.0 if self.min_return_net_clearance_m is None else self.min_return_net_clearance_m
                    ),
                    "score_softness_m": self.ballistic_return_score_softness_m,
                }
                if self.clearance_prediction_mode == DRAG_AWARE_CLEARANCE_PREDICTION_MODE:
                    ballistic = drag_aware_return_clearance_score(
                        np.asarray(flight["shuttle_xyz"], dtype=float),
                        outgoing,
                        terminal_velocity_m_s=(self.physics.cfg.aero.terminal_velocity_m_s),
                        drag_multiplier=(1.0 + self.physics.cfg.aero.angle_drag_gain),
                        **clearance_kwargs,
                    )
                else:
                    ballistic = ballistic_return_clearance_score(
                        np.asarray(flight["shuttle_xyz"], dtype=float),
                        outgoing,
                        **clearance_kwargs,
                    )
                self._predicted_net_clearance_m = float(ballistic["predicted_clearance_m"])
                self._return_clearance_score = float(ballistic["score"])
                clearance_reward_score = (
                    2.0 * self._return_clearance_score - 1.0
                    if self.clearance_reward_mode == "signed_centered"
                    else self._return_clearance_score
                )
                terms["return_clearance"] = w["return_clearance"] * clearance_reward_score

            if self.task_profile == IMPACT_RECOVERY_PROFILE and self._impact_diag is not None:
                target_position = self._active_target_value("impact_position_world")
                target_time = float(self._active_target_value("impact_time_s"))
                target_normal = self._active_target_value("stringbed_normal_world")
                target_linear = self._active_target_value("racket_linear_velocity_world")
                target_angular = self._active_target_value("racket_angular_velocity_world")
                rho2 = max(0.0, float(self._impact_diag["rho2"]))
                position_error = float(np.linalg.norm(self._impact_diag["position_world"] - target_position))
                timing_error = abs(float(self._impact_diag["impact_time_s"]) - target_time)
                normal_dot = float(
                    np.clip(
                        np.dot(self._impact_diag["normal_world"], target_normal),
                        -1.0,
                        1.0,
                    )
                )
                normal_error = float(np.arccos(normal_dot))
                linear_error = float(
                    np.sqrt(np.mean(np.square(self._impact_diag["linear_velocity_world"] - target_linear)))
                )
                angular_error = float(
                    np.sqrt(np.mean(np.square(self._impact_diag["angular_velocity_world"] - target_angular)))
                )
                terms["impact_position"] = w["impact_position"] * float(np.exp(-0.5 * (position_error / 0.12) ** 2))
                terms["impact_center"] = w["impact_center"] * float(np.exp(-2.0 * rho2))
                terms["impact_time"] = w["impact_time"] * float(np.exp(-0.5 * (timing_error / 0.08) ** 2))
                terms["impact_normal"] = w["impact_normal"] * float(np.exp(-2.0 * normal_error**2))
                terms["impact_linear_velocity"] = w["impact_linear_velocity"] * float(np.exp(-0.25 * linear_error**2))
                terms["impact_angular_velocity"] = w["impact_angular_velocity"] * float(
                    np.exp(-0.02 * angular_error**2)
                )
                self._last_v2_metrics.update(
                    {
                        "impact_position_error_m": position_error,
                        "impact_rho2": rho2,
                        "impact_timing_error_s": timing_error,
                        "stringbed_normal_error_rad": normal_error,
                        "racket_linear_velocity_error_m_s": linear_error,
                        "racket_angular_velocity_error_rad_s": angular_error,
                    }
                )

        if (
            self.state in (IncomingHitState.FLIGHT, IncomingHitState.DONE)
            and self._hit_rewarded
            and dynamic_feed
            and bool(flight.get("valid_net_crossing_event", False))
            and not self._crossed_net_rewarded
        ):
            self._crossed_net_rewarded = True
            terms["crossed_net"] = w["crossed_net"]

        if (
            self._hit_rewarded
            and dynamic_feed
            and bool(flight.get("invalid_net_crossing_event", False))
            and not self._invalid_net_crossed
        ):
            self._invalid_net_crossed = True
            terms["invalid_net_crossing"] = -w["invalid_net_crossing"]

        if (
            self.task_profile == LEGACY_PROFILE
            and self._landing_region is not None
            and self.termination_reason == "landed"
            and self._hit_rewarded
            and not self._landing_rewarded
        ):
            landing_score = REGION_SCORES.get(self._landing_region, 0.0) if self._crossed_net_rewarded else -1.0
            terms["landing_region"] = w["landing_region"] * landing_score
            self._landing_rewarded = True

        if (
            self.task_profile == IMPACT_RECOVERY_PROFILE
            and self._landing_xy is not None
            and self._hit_rewarded
            and not self._landing_rewarded
        ):
            landing_target = self._active_target_value("landing_target_xy")
            landing_error = float(np.linalg.norm(self._landing_xy - landing_target))
            apex_error = abs(self._apex_height_m - float(self._active_target_value("apex_height_m")))
            terms["precise_landing"] = w["precise_landing"] * float(np.exp(-0.5 * (landing_error / 0.75) ** 2))
            terms["apex"] = w["apex"] * float(np.exp(-0.5 * (apex_error / 0.35) ** 2))
            self._landing_rewarded = True
            self._last_v2_metrics.update(
                {
                    "landing_error_m": landing_error,
                    "apex_error_m": apex_error,
                    "apex_height_m": self._apex_height_m,
                }
            )

        if self.task_profile == IMPACT_RECOVERY_PROFILE and (
            self._recovery_active or self._v2_environment_mode_code == 0
        ):
            ready_error = self._ready_pose_error()
            root_speed = float(
                np.linalg.norm(
                    np.asarray(
                        self.data.qvel[self._root_dadr : self._root_dadr + 6],
                        dtype=float,
                    )
                )
            )
            racket_speed = float(
                np.linalg.norm(self._stringbed_velocity()) + 0.15 * np.linalg.norm(self._stringbed_angular_velocity())
            )
            terms["recovery_ready"] = w["recovery_ready"] * float(np.exp(-8.0 * ready_error**2))
            terms["recovery_balance"] = w["recovery_balance"] * float(np.exp(-2.0 * root_speed**2))
            terms["recovery_deceleration"] = w["recovery_deceleration"] * float(np.exp(-0.5 * racket_speed**2))
            self._last_v2_metrics.update(
                {
                    "ready_pose_error": ready_error,
                    "recovery_root_speed": root_speed,
                    "recovery_racket_speed": racket_speed,
                    "recovery_progress": self._recovery_step / max(1, self._active_recovery_horizon()),
                }
            )

        terms["effort"] = -w["effort"] * float(np.mean(np.square(action)))
        if self.base_bridge is not None and w.get("residual", 0.0) != 0.0:
            residual = action if residual_action is None else residual_action
            terms["residual"] = -w["residual"] * float(np.mean(np.square(residual)))
        terms["posture"] = -w["posture"] * max(0.0, 0.85 - self._root_height())
        if self.termination_reason == "miss" and (self.task_profile == LEGACY_PROFILE or dynamic_feed):
            terms["miss"] = -w["miss"]
        if body_fall:
            terms["body_fall"] = -w["body_fall"]
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            for name, multiplier in self._v2_reward_mask.items():
                terms[name] *= float(multiplier)
        return terms

    def _root_height(self) -> float:
        return float(self.data.qpos[self._root_qadr + 2])

    def _flight_info(self) -> dict[str, Any]:
        shuttle_pos = np.asarray(self.data.qpos[self._shuttle_qadr : self._shuttle_qadr + 3], dtype=float)
        shuttle_vel = np.asarray(self.data.qvel[self._shuttle_dadr : self._shuttle_dadr + 3], dtype=float)
        landed = bool(shuttle_pos[2] <= GROUND_REST_HEIGHT_M)
        crossed_net = bool(self._crossed_net_rewarded)
        region = classify_landing_region(
            shuttle_pos[:2],
            player_half_sign=self.player_half_sign,
            singles=self.singles,
        )
        return {
            "shuttle_xyz": shuttle_pos.copy(),
            "shuttle_velocity": shuttle_vel.copy(),
            "crossed_net": crossed_net,
            "invalid_net_crossed": self._invalid_net_crossed,
            "predicted_net_clearance_m": self._predicted_net_clearance_m,
            "return_clearance_score": self._return_clearance_score,
            "return_direction_signed_score": (self._return_direction_signed_score),
            "landed": landed,
            "region": region.value,
        }

    def _info(self, extra: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state": self.state.value,
            "step_count": self.step_index,
            "feed_intercept_point": None
            if self.feed is None
            else np.asarray(self.feed.intercept_point, dtype=float).copy(),
            "feed_intercept_time_s": None if self.feed is None else float(self.feed.intercept_time_s),
        }
        if self.termination_reason is not None:
            info["termination_reason"] = self.termination_reason
        if self._landing_region is not None:
            info["landing_region"] = self._landing_region
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            info.update(
                {
                    "task_profile": self.task_profile,
                    "target_bank_sha256": self.impact_target_bank.bank_sha256,
                    "target_index": self._active_target_index,
                    "desired_impact_position_world": self._active_target_value("impact_position_world").copy(),
                    "desired_landing_target_xy": self._active_target_value("landing_target_xy").copy(),
                    "recovery_step": self._recovery_step,
                    "recovery_horizon_steps": self._active_recovery_horizon(),
                    "recovery_active": self._recovery_active,
                    "recovery_complete": self._recovery_complete,
                    "flight_resolved": self._flight_resolved,
                    "task_curriculum_stage": self.task_curriculum_stage,
                    "stage3_v2_metrics": dict(self._last_v2_metrics),
                }
            )
        info.update(extra)
        return info
