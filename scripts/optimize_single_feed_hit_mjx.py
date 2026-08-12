#!/usr/bin/env python3
"""Find a real upward badminton rebound with batched MJX CEM.

The inherited Stage-2 swing and the learned v24c Phase-A adapter are frozen.
CEM searches a time-varying anatomical-synergy trajectory for the canonical
32 right-arm physical corrections.  The correction has its own tanh and is
added after the inherited residual has been squashed, matching the Stage-3
selected-physical-correction PPO ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.base_swing_bridge import (  # noqa: E402
    compose_selected_physical_correction,
    selected_correction_window,
)
from environment.overall_environment.src.incoming_shuttle_hit_env import (  # noqa: E402
    IncomingShuttleHitEnv,
)
from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    IncomingHitMjxEnv,
)
from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    feed_sample_fingerprint,
)
from environment.overall_environment.src.train_incoming_hit_mjx import (  # noqa: E402
    _inherited_policy_mean,
    init_agent,
    load_training_actor_checkpoint,
    load_training_checkpoint_metadata,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (  # noqa: E402
    _ensure_feed_bank_artifact,
    _ensure_scene,
    _policy_update_contract,
    _residual_scale_overrides,
    _residual_scale_schedule,
    _return_constraints,
    load_incoming_hit_spec,
)

RIGHT_ARM_AUDIT_BODY_NAMES = (
    "clavicle_r",
    "scapula_r",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "scaphoid_r",
    "thirdmc_r",
)

# Outside the broad high-contact window, each metre of combined hand and
# stringbed height deficit adds only this fraction of a metre to the contact
# acquisition cost.  This keeps high contact as a soft preference while the
# policy remains free to choose the outcome-optimal instant inside the window.
SOFT_HIGH_REGION_EXCESS_WEIGHT = 0.10
REPLICA_FRACTION_INTEGER_TOLERANCE = 1.0e-6
SEARCH_SAFE_NO_FALL_RATE = 0.98
TIME_KNOT_REBIND_PHASE_SAMPLES = 1001
TIME_KNOT_REBIND_MAX_PHYSICAL_RMS_ERROR = 0.002
TIME_KNOT_REBIND_MAX_PHYSICAL_ABS_ERROR = 0.010
CPU_ACTOR_INFERENCE_SEMANTICS = "explicit_jax_cpu_device_v1"
CPU_AUDIT_TRACE_SCHEMA = "stage3_cem_cpu_audit_trajectory_v5"
# A custom stringbed event must be the first meaningful racket impulse in its
# control interval.  Comparing the recorded pre-event velocity with the prior
# settled control state catches a cork/frame collision that happened earlier
# in the same interval and would otherwise masquerade as a clean string hit.
MAX_PRE_EVENT_VELOCITY_DELTA_M_S = 0.75
# Aero and gravity can change the event impulse slightly during the remaining
# physics substeps of one control interval.  A large discontinuity, however,
# means a native collision (typically shuttle versus racket frame) resolved the
# same nominal hit again.  Such a trajectory is not a stringbed teacher.
MAX_EVENT_SETTLED_VELOCITY_DELTA_M_S = 0.75


def _required_replica_count(replicas: int, min_replica_fraction: float) -> int:
    """Convert a requested fraction to a stable integer replica gate.

    CLI decimal spellings such as ``0.6666667`` must mean two of three, not
    accidentally three of three because the rounded decimal is a few ulps
    larger than the exact ratio.
    """

    return max(
        1,
        min(
            int(replicas),
            int(math.ceil(float(min_replica_fraction) * int(replicas) - REPLICA_FRACTION_INTEGER_TOLERANCE)),
        ),
    )


def _verification_group_indices(
    *,
    population: int,
    repeats: int,
    anchor_group: int,
) -> tuple[int, ...]:
    """Spread final candidate replays across deterministic Warp batch lanes.

    A one-feed rollout has an identical physical reset in every replica.  Its
    remaining cross-replica variation is therefore a backend/batch-lane
    sensitivity, not an environmental perturbation.  Replaying every final
    batch in the discovery lane would repeatedly sample the same numerical
    context.  Relocating the candidate group across the population makes the
    advertised verification count cover distinct lane positions without
    increasing the compiled batch size.
    """

    if population <= 0 or repeats <= 0:
        raise ValueError("verification population and repeats must be positive")
    if repeats > population:
        raise ValueError("verification repeats cannot exceed the CEM population")
    if not 0 <= anchor_group < population:
        raise ValueError("verification anchor group is outside the CEM population")
    offsets = (np.arange(repeats, dtype=np.int64) * population) // repeats
    groups = tuple(int((anchor_group + int(offset)) % population) for offset in offsets)
    if len(set(groups)) != repeats:
        raise RuntimeError("verification lane placement unexpectedly contains duplicates")
    return groups


def _stratified_candidate_lane_indices(
    *,
    population: int,
    replicas: int,
) -> np.ndarray:
    """Map each candidate to replicas spread across the full Warp batch.

    Contiguous ``np.repeat`` placement couples candidate identity to one local
    group of Warp lanes.  That is a biased comparison when contact is
    numerically lane-sensitive: a candidate can win because its contiguous
    group is favourable, not because its racket-face correction is robust.
    Put one replica in each of ``replicas`` population-sized blocks and rotate
    the within-block lane by a deterministic, evenly spaced offset.  The
    returned array is candidate-major so host metrics can be gathered back
    into the ABI expected by ``_aggregate_replica_metrics``.
    """

    if population <= 0 or replicas <= 0:
        raise ValueError("stratified lane population and replicas must be positive")
    if replicas > population:
        raise ValueError("stratified lane replicas cannot exceed the population")
    offsets = (np.arange(replicas, dtype=np.int64) * population) // replicas
    candidate_indices = np.arange(population, dtype=np.int64)[:, None]
    block_indices = np.arange(replicas, dtype=np.int64)[None, :]
    lane_indices = block_indices * population + (candidate_indices + offsets[None, :]) % population
    flattened = lane_indices.reshape(-1)
    if len(np.unique(flattened)) != population * replicas:
        raise RuntimeError("stratified candidate lane placement is not bijective")
    return lane_indices


def _expand_candidates_across_stratified_lanes(
    candidates: np.ndarray,
    *,
    replicas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand candidates into Warp lanes and return candidate-major indices."""

    candidate_array = np.asarray(candidates)
    if candidate_array.ndim != 2:
        raise ValueError("CEM candidates must be a rank-2 array")
    population = int(candidate_array.shape[0])
    lane_indices = _stratified_candidate_lane_indices(
        population=population,
        replicas=int(replicas),
    )
    expanded = np.empty(
        (population * int(replicas), int(candidate_array.shape[1])),
        dtype=candidate_array.dtype,
    )
    expanded[lane_indices.reshape(-1)] = np.repeat(
        candidate_array,
        int(replicas),
        axis=0,
    )
    return expanded, lane_indices


def _gather_candidate_major_lane_values(
    values: np.ndarray,
    *,
    lane_indices: np.ndarray,
) -> np.ndarray:
    """Undo stratified lane placement before candidate-level aggregation."""

    value_array = np.asarray(values)
    candidate_lane_indices = np.asarray(lane_indices, dtype=np.int64)
    if candidate_lane_indices.ndim != 2:
        raise ValueError("candidate lane indices must be rank 2")
    expected_lanes = int(candidate_lane_indices.size)
    if value_array.ndim < 1 or value_array.shape[0] != expected_lanes:
        raise ValueError("Warp lane values do not match the stratified lane layout")
    return value_array[candidate_lane_indices.reshape(-1)]


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _cross_backend_promotion_passes(
    backend_metrics: dict[str, Any],
    cpu_audit: dict[str, Any] | None,
) -> bool:
    """Require both the vectorized backend and independent CPU quality gates."""

    return bool(
        backend_metrics.get("teacher_success") is True
        and isinstance(cpu_audit, dict)
        and cpu_audit.get("cpu_quality_passed") is True
    )


def _cpu_unqualified_search_progress_key(
    audit: dict[str, Any],
    *,
    min_outgoing_z_m_s: float,
    min_forward_m_s: float,
    min_predicted_clearance_m: float,
    min_return_direction_signed_score: float,
    real_net_cross_authoritative: bool = False,
) -> tuple[float, ...]:
    """Rank CPU-audited search progress without granting teacher status.

    Warp is useful for proposing many trajectories, but a grazing racket hit
    can rank very differently in the independent CPU MuJoCo replay.  This key
    lets a *still unqualified* CPU-consistent candidate guide the next search
    mean.  Physical validity is lexicographically ahead of every velocity or
    clearance value so a duplicate native collision can never win by showing
    an attractive but discontinuous post-contact velocity.
    """

    def finite_metric(name: str) -> float:
        value = audit.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return -math.inf
        result = float(value)
        return result if math.isfinite(result) else -math.inf

    outgoing_z = finite_metric("outgoing_z_m_s")
    outgoing_forward = finite_metric("outgoing_forward_m_s")
    predicted_clearance = finite_metric("predicted_net_clearance_m")
    return_direction = finite_metric("return_direction_signed_score")
    settled_delta = finite_metric("event_settled_velocity_delta_m_s")
    common_margins = (
        (outgoing_z - float(min_outgoing_z_m_s))
        / max(1.0, abs(float(min_outgoing_z_m_s))),
        (outgoing_forward - float(min_forward_m_s))
        / max(1.0, abs(float(min_forward_m_s))),
        (return_direction - float(min_return_direction_signed_score))
        / max(1.0, abs(float(min_return_direction_signed_score))),
    )
    authoritative_cross = bool(
        real_net_cross_authoritative and audit.get("crossed_net") is True
    )
    # Before an observed crossing, the drag-aware projection remains the best
    # continuous signal for net reachability.  Once an authoritative real
    # crossing exists, however, predicted clearance is no longer a teacher
    # constraint and must not outrank the still-required direction gate.
    margins = (
        common_margins
        if authoritative_cross
        else (
            *common_margins,
            (predicted_clearance - float(min_predicted_clearance_m))
            / max(1.0, abs(float(min_predicted_clearance_m))),
        )
    )
    finite_margins = tuple(value if math.isfinite(value) else -math.inf for value in margins)
    min_margin = min(finite_margins)
    margin_sum = sum(finite_margins) if all(math.isfinite(value) for value in finite_margins) else -math.inf
    continuous_tiebreak = (
        (return_direction, outgoing_z, outgoing_forward, predicted_clearance)
        if authoritative_cross
        else (predicted_clearance, return_direction, outgoing_z, outgoing_forward)
    )
    return (
        float(audit.get("body_fall") is False),
        float(audit.get("hit") is True),
        float(audit.get("event_rebound") is True),
        float(audit.get("pre_event_velocity_consistent") is True),
        float(audit.get("event_settled_velocity_consistent") is True),
        float(audit.get("high_region_contact") is True),
        float(audit.get("cpu_quality_passed") is True),
        float(audit.get("crossed_net") is True),
        float(audit.get("legal_return") is True),
        min_margin,
        margin_sum,
        *continuous_tiebreak,
        (-settled_delta if math.isfinite(settled_delta) else -math.inf),
    )


def _cpu_unqualified_search_improves(
    incumbent: dict[str, Any] | None,
    challenger: dict[str, Any],
    **thresholds: float | bool,
) -> bool:
    """Return whether a CPU audit is strictly better search guidance."""

    challenger_key = _cpu_unqualified_search_progress_key(challenger, **thresholds)
    # A search mean must never be reanchored to a fall, miss, non-rebound,
    # duplicate collision, or contact outside the broad high region.  Keeping
    # these as hard prerequisites also means ``None`` is preferable to an
    # invalid first audit.
    if challenger_key[:6] != (1.0, 1.0, 1.0, 1.0, 1.0, 1.0):
        return False
    if incumbent is None:
        return True
    return challenger_key > _cpu_unqualified_search_progress_key(incumbent, **thresholds)


def _retain_cpu_search_frontier_mean(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    frontier_parameters: np.ndarray,
    initial_parameters: np.ndarray,
    trainable_parameter_mask: np.ndarray,
    initial_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last CPU-consistent frontier as the next proposal center.

    A Warp-ranked CEM update may move away from a narrow contact that was
    independently verified in CPU MuJoCo.  The frontier must therefore remain
    the proposal mean even on an iteration that does not set a new CPU record;
    otherwise one non-improving batch silently discards the trusted search
    state.  This is search guidance only and grants no teacher status.
    """

    current_mean = np.asarray(mean, dtype=np.float32)
    current_std = np.asarray(std, dtype=np.float32)
    frontier = np.asarray(frontier_parameters, dtype=np.float32)
    initial = np.asarray(initial_parameters, dtype=np.float32)
    mask = np.asarray(trainable_parameter_mask, dtype=bool)
    if (
        current_mean.ndim != 1
        or current_std.shape != current_mean.shape
        or frontier.shape != current_mean.shape
        or initial.shape != current_mean.shape
        or mask.shape != current_mean.shape
    ):
        raise ValueError("CPU frontier retention arrays are incompatible")
    if (
        not np.isfinite(current_mean).all()
        or not np.isfinite(current_std).all()
        or not np.isfinite(frontier).all()
        or not np.isfinite(initial).all()
        or np.any(current_std < 0.0)
        or not math.isfinite(float(initial_std))
        or float(initial_std) <= 0.0
    ):
        raise ValueError("CPU frontier retention requires finite search state")
    if not bool(mask.any()):
        raise ValueError("CPU frontier retention requires trainable parameters")

    retained_mean = frontier.copy()
    retained_mean[~mask] = initial[~mask]
    std_floor = np.where(mask, 0.5 * float(initial_std), 0.0).astype(np.float32)
    retained_std = np.maximum(current_std, std_floor).astype(np.float32)
    retained_std[~mask] = 0.0
    return retained_mean, retained_std


def _interpolate_numpy_knots(
    knots: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Interpolate a knot matrix with the same uniform convention as JAX."""

    values = np.asarray(knots, dtype=np.float64)
    query = np.asarray(phases, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("time-knot interpolation requires a rank-2 knot matrix")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("time-knot interpolation phases must be a finite vector")
    position = np.clip(query, 0.0, 1.0) * float(values.shape[0] - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, values.shape[0] - 1)
    fraction = position - lower
    return (1.0 - fraction[:, None]) * values[lower] + fraction[:, None] * values[upper]


def _rebind_unqualified_anatomical_time_knots(
    parameters: np.ndarray,
    *,
    source_contract: dict[str, Any],
    target_contract: dict[str, Any],
    synergy_basis: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand an unqualified anatomical seed without changing its swing.

    A scale change cannot in general be represented by applying the inverse
    ``tanh`` independently at six knots: latent interpolation occurs before
    ``tanh``, so a small mid-segment action jump remains and can turn a narrow
    CPU contact into a miss.  A nested, denser knot grid can approximate the
    same complete physical correction trajectory while providing additional
    authority for the subsequent search.

    This is only an initialization intervention.  It never promotes a teacher,
    and it fails closed if the rebound trajectory is not physically close.
    """

    from scipy.optimize import least_squares

    source_time_knots = int(source_contract.get("time_knots", 0))
    target_time_knots = int(target_contract.get("time_knots", 0))
    source_latent_size = int(source_contract.get("latent_size", 0))
    target_latent_size = int(target_contract.get("latent_size", 0))
    if source_time_knots < 2 or target_time_knots <= source_time_knots:
        raise ValueError("time-knot rebind requires a strictly denser target knot grid")
    if (target_time_knots - 1) % (source_time_knots - 1) != 0:
        raise ValueError("time-knot rebind target grid must contain every source knot")
    if source_latent_size <= 0 or source_latent_size != target_latent_size:
        raise ValueError("time-knot rebind changed the anatomical latent size")
    if source_contract.get("parameterization") != "anatomical_synergies" or (
        target_contract.get("parameterization") != "anatomical_synergies"
    ):
        raise ValueError("time-knot rebind requires anatomical_synergies")

    basis = np.asarray(synergy_basis, dtype=np.float64)
    source_scales = np.asarray(source_contract.get("physical_scales"), dtype=np.float64)
    target_scales = np.asarray(target_contract.get("physical_scales"), dtype=np.float64)
    if (
        basis.ndim != 2
        or basis.shape[0] != source_latent_size
        or source_scales.shape != (basis.shape[1],)
        or target_scales.shape != source_scales.shape
        or not np.isfinite(basis).all()
        or not np.isfinite(source_scales).all()
        or not np.isfinite(target_scales).all()
        or np.any(source_scales <= 0.0)
        or np.any(target_scales <= 0.0)
    ):
        raise ValueError("time-knot rebind has incompatible basis or scales")

    source_parameters = np.asarray(parameters, dtype=np.float32)
    if source_parameters.shape != (source_time_knots * source_latent_size,):
        raise ValueError("time-knot rebind source parameters are incompatible")
    source_knots = source_parameters.astype(np.float64).reshape(source_time_knots, source_latent_size)
    dense_phases = np.linspace(
        0.0,
        1.0,
        TIME_KNOT_REBIND_PHASE_SAMPLES,
        dtype=np.float64,
    )

    def physical(knots: np.ndarray, scales: np.ndarray, phases: np.ndarray) -> np.ndarray:
        raw = np.clip(
            _interpolate_numpy_knots(knots, phases) @ basis,
            -3.0,
            3.0,
        )
        return np.tanh(raw) * scales[None, :]

    source_physical = physical(source_knots, source_scales, dense_phases)
    target_phases = np.linspace(0.0, 1.0, target_time_knots, dtype=np.float64)
    target_knots = _interpolate_numpy_knots(source_knots, target_phases)
    changed_actuators = ~np.isclose(
        source_scales,
        target_scales,
        rtol=0.0,
        atol=1.0e-8,
    )
    affected_latents = np.any(
        np.abs(basis[:, changed_actuators]) > 1.0e-8,
        axis=1,
    )
    affected_indices = np.flatnonzero(affected_latents)

    if affected_indices.size:
        # First solve every nested target knot independently.  This gives the
        # global fit a close, deterministic initialization despite redundant
        # antagonist rows in the proposal basis.
        for knot_index, phase in enumerate(target_phases):
            desired = physical(
                source_knots,
                source_scales,
                np.asarray([phase], dtype=np.float64),
            )[0]
            fixed = target_knots[knot_index].copy()

            def local_residual(
                values: np.ndarray,
                *,
                fixed_row: np.ndarray = fixed.copy(),
                desired_physical: np.ndarray = desired.copy(),
            ) -> np.ndarray:
                row = fixed_row.copy()
                row[affected_indices] = values
                raw = np.clip(row @ basis, -3.0, 3.0)
                return np.tanh(raw) * target_scales - desired_physical

            local = least_squares(
                local_residual,
                fixed[affected_indices],
                bounds=(-3.0, 3.0),
                xtol=1.0e-13,
                ftol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=5000,
                x_scale="jac",
            )
            if not local.success:
                raise ValueError("time-knot rebind local physical fit failed")
            target_knots[knot_index, affected_indices] = local.x

        initial = target_knots[:, affected_indices].reshape(-1).copy()

        def dense_residual(values: np.ndarray) -> np.ndarray:
            rebound = target_knots.copy()
            rebound[:, affected_indices] = values.reshape(target_time_knots, affected_indices.size)
            return (physical(rebound, target_scales, dense_phases) - source_physical).reshape(-1)

        global_fit = least_squares(
            dense_residual,
            initial,
            bounds=(-3.0, 3.0),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=5000,
            x_scale="jac",
        )
        if not global_fit.success:
            raise ValueError("time-knot rebind dense physical fit failed")
        target_knots[:, affected_indices] = global_fit.x.reshape(target_time_knots, affected_indices.size)

    rebound_parameters = target_knots.astype(np.float32).reshape(-1)
    rebound_physical = physical(
        rebound_parameters.astype(np.float64).reshape(target_time_knots, target_latent_size),
        target_scales,
        dense_phases,
    )
    error = rebound_physical - source_physical
    rms_error = float(np.sqrt(np.mean(np.square(error))))
    max_abs_error = float(np.max(np.abs(error)))
    if (
        not math.isfinite(rms_error)
        or not math.isfinite(max_abs_error)
        or rms_error > TIME_KNOT_REBIND_MAX_PHYSICAL_RMS_ERROR
        or max_abs_error > TIME_KNOT_REBIND_MAX_PHYSICAL_ABS_ERROR
    ):
        raise ValueError(
            "time-knot rebind physical error exceeds the fail-closed bound: "
            f"rms={rms_error:.9g}, max={max_abs_error:.9g}"
        )
    synergy_names = tuple(str(name) for name in target_contract.get("synergy_names", ()))
    optimized_names = [
        synergy_names[index] if index < len(synergy_names) else str(index) for index in affected_indices.tolist()
    ]
    report = {
        "schema_version": "stage3_unqualified_time_knot_physical_rebind_v1",
        "source_time_knots": source_time_knots,
        "target_time_knots": target_time_knots,
        "nested_grid_factor": (target_time_knots - 1) // (source_time_knots - 1),
        "dense_phase_samples": TIME_KNOT_REBIND_PHASE_SAMPLES,
        "optimized_synergy_names": optimized_names,
        "physical_rms_error": rms_error,
        "physical_max_abs_error": max_abs_error,
        "max_allowed_physical_rms_error": (TIME_KNOT_REBIND_MAX_PHYSICAL_RMS_ERROR),
        "max_allowed_physical_abs_error": (TIME_KNOT_REBIND_MAX_PHYSICAL_ABS_ERROR),
        "target_parameter_f32_sha256": hashlib.sha256(rebound_parameters.tobytes(order="C")).hexdigest(),
    }
    return rebound_parameters, report


def _load_initial_candidate(
    path: str | Path,
    *,
    dimension: int,
    expected_source_contract: dict[str, Any],
    allow_unqualified_physical_scale_rebind: bool = False,
    allow_unqualified_time_knot_rebind: bool = False,
    synergy_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a teacher candidate or an explicitly unqualified CEM search seed.

    Search seeds are optimizer means exported from an interrupted/diagnostic
    run.  They are useful starting points, but are deliberately not accepted
    as teachers and may cross MJX implementations, feeds, or base-swing timing
    because the latent-to-action mapping is backend independent.  Qualified
    teacher candidates remain fail-closed on every physical/control mapping
    field.
    """

    candidate_path = Path(path).expanduser().resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"initial CEM candidate is missing: {candidate_path}")
    source_contract_path = candidate_path.parent / "cem_contract.json"
    if not source_contract_path.is_file():
        raise ValueError("initial CEM candidate has no sibling cem_contract.json")
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("initial CEM candidate or contract is unreadable") from exc
    if not isinstance(candidate, dict):
        raise ValueError("initial CEM candidate must be a JSON object")
    candidate_schema = candidate.get("schema_version")
    if candidate_schema not in {
        "stage3_cem_teacher_candidate_v1",
        "stage3_cem_search_seed_v1",
    }:
        raise ValueError("initial CEM candidate schema is incompatible")
    is_search_seed = candidate_schema == "stage3_cem_search_seed_v1"
    allowed_search_seed_roles = {
        "unqualified_optimizer_mean",
        "unqualified_parameter_intervention",
        "unqualified_snapshot_candidate",
    }
    if is_search_seed and (
        candidate.get("qualified_teacher") is not False or candidate.get("seed_role") not in allowed_search_seed_roles
    ):
        raise ValueError("initial CEM search seed is not explicitly unqualified")
    if not isinstance(source_contract, dict):
        raise ValueError("initial CEM source contract must be a JSON object")
    recorded_contract_sha = source_contract.get("contract_sha256")
    unhashed_contract = dict(source_contract)
    unhashed_contract.pop("contract_sha256", None)
    if recorded_contract_sha != _json_hash(unhashed_contract):
        raise ValueError("initial CEM source contract hash mismatch")
    if candidate.get("contract_sha256") != recorded_contract_sha:
        raise ValueError("initial CEM candidate is detached from its source contract")
    search_seed_intervention_fields = {
        "mjx_impl",
        "feed_fingerprint",
        "swing_phase_advance_s",
        # These labels are redundant once the source checkpoint, selected
        # actuators, exact physical scales, correction window and horizon all
        # match.  An explicitly unqualified optimizer seed may therefore be
        # re-ranked under a repaired reward spec or an equivalent authority
        # spelling without inheriting any teacher-quality claim.
        "spec",
        "authority_multiplier",
    }
    time_knot_rebind_fields = {"time_knots", "parameter_count"}
    for key, expected in expected_source_contract.items():
        if is_search_seed and key in search_seed_intervention_fields:
            # A search mean has no teacher-quality claim.  Its complete latent
            # vector can safely seed a different MJX implementation or an
            # explicitly changed feed/timing condition while the new run
            # records both source and target conditions.
            continue
        if is_search_seed and key == "physical_scales" and bool(allow_unqualified_physical_scale_rebind):
            # Changing the physical scale changes the latent-to-action map.
            # Permit it only as an explicit *unqualified* search intervention;
            # the target run seals the new scales and must rediscover and
            # independently certify any teacher under that target contract.
            continue
        if is_search_seed and key in time_knot_rebind_fields and bool(allow_unqualified_time_knot_rebind):
            # A denser nested temporal grid is an explicit initialization
            # intervention.  The source vector is expanded below and the
            # complete physical-trajectory error is sealed in the new run.
            continue
        if source_contract.get(key) != expected:
            raise ValueError(f"initial CEM source contract changed field: {key}")
    parameters = np.asarray(candidate.get("parameters"), dtype=np.float32)
    source_parameter_count = int(source_contract.get("parameter_count", -1))
    if parameters.shape != (source_parameter_count,) or not np.isfinite(parameters).all():
        raise ValueError("initial CEM candidate parameters have an incompatible shape or values")
    parameter_sha = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    if is_search_seed and candidate.get("parameter_f32_sha256") != parameter_sha:
        raise ValueError("initial CEM search seed parameter hash mismatch")
    time_knot_rebind_report: dict[str, Any] | None = None
    source_time_knots = int(source_contract.get("time_knots", 0))
    target_time_knots = int(expected_source_contract.get("time_knots", source_time_knots))
    if source_time_knots != target_time_knots:
        if not is_search_seed or not bool(allow_unqualified_time_knot_rebind):
            raise ValueError("initial CEM source contract changed field: time_knots")
        if synergy_basis is None:
            raise ValueError("time-knot rebind requires the exact target synergy basis")
        parameters, time_knot_rebind_report = _rebind_unqualified_anatomical_time_knots(
            parameters,
            source_contract=source_contract,
            target_contract=expected_source_contract,
            synergy_basis=np.asarray(synergy_basis, dtype=np.float32),
        )
    if parameters.shape != (int(dimension),):
        raise ValueError("initial CEM candidate parameters have an incompatible target shape")
    bound_parameter_sha = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    binding = {
        "candidate_schema_version": candidate_schema,
        "candidate_role": (str(candidate["seed_role"]) if is_search_seed else "qualified_candidate"),
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "candidate_parameter_f32_sha256": parameter_sha,
        "bound_parameter_f32_sha256": bound_parameter_sha,
        "source_contract_path": str(source_contract_path.resolve()),
        "source_contract_file_sha256": hashlib.sha256(source_contract_path.read_bytes()).hexdigest(),
        "source_contract_sha256": recorded_contract_sha,
        "source_mjx_impl": source_contract.get("mjx_impl"),
        "source_feed_fingerprint": source_contract.get("feed_fingerprint"),
        "source_swing_phase_advance_s": source_contract.get("swing_phase_advance_s"),
        "source_spec": source_contract.get("spec"),
        "source_authority_multiplier": source_contract.get("authority_multiplier"),
        "source_physical_scales": source_contract.get("physical_scales"),
        "source_synergy_basis_sha256": source_contract.get("synergy_basis_sha256"),
        "source_time_knots": source_contract.get("time_knots"),
        "target_mjx_impl": expected_source_contract.get("mjx_impl"),
        "target_feed_fingerprint": expected_source_contract.get("feed_fingerprint"),
        "target_swing_phase_advance_s": expected_source_contract.get("swing_phase_advance_s"),
        "target_spec": expected_source_contract.get("spec"),
        "target_authority_multiplier": expected_source_contract.get("authority_multiplier"),
        "target_physical_scales": expected_source_contract.get("physical_scales"),
        "target_synergy_basis_sha256": expected_source_contract.get("synergy_basis_sha256"),
        "target_time_knots": target_time_knots,
        "unqualified_physical_scale_rebind": bool(is_search_seed and allow_unqualified_physical_scale_rebind),
        "unqualified_time_knot_rebind": bool(time_knot_rebind_report is not None),
        "time_knot_rebind": time_knot_rebind_report,
    }
    return parameters, binding


def _save_iteration_snapshot(
    *,
    out_dir: Path,
    iteration: int,
    contract_sha256: str,
    candidates: np.ndarray,
    rank_order: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    candidate_lane_indices: np.ndarray,
    replica_metrics: dict[str, np.ndarray],
    backend_best_index: int,
    search_frontier_candidate_indices: tuple[int, ...] = (),
    coordinate_probe_candidate_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Persist enough host-side evidence to replay every CEM ranking decision."""

    snapshot_dir = out_dir / "iteration_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"iteration_{int(iteration):04d}.npz"
    temporary_path = snapshot_path.with_suffix(".npz.tmp")
    arrays: dict[str, np.ndarray] = {
        "iteration": np.asarray(iteration, dtype=np.int32),
        "contract_sha256": np.asarray(contract_sha256),
        "candidates": np.asarray(candidates, dtype=np.float32),
        "rank_order_low_to_high": np.asarray(rank_order, dtype=np.int32),
        "mean_after_update": np.asarray(mean, dtype=np.float32),
        "std_after_update": np.asarray(std, dtype=np.float32),
        "candidate_lane_indices": np.asarray(candidate_lane_indices, dtype=np.int32),
        "backend_best_index": np.asarray(backend_best_index, dtype=np.int32),
        "search_frontier_candidate_indices": np.asarray(
            search_frontier_candidate_indices,
            dtype=np.int32,
        ),
        "coordinate_probe_candidate_indices": np.asarray(
            coordinate_probe_candidate_indices,
            dtype=np.int32,
        ),
    }
    population = int(np.asarray(candidates).shape[0])
    replicas = int(np.asarray(candidate_lane_indices).shape[1])
    for name, values in replica_metrics.items():
        value_array = np.asarray(values)
        if value_array.shape != (population * replicas,):
            raise ValueError(
                f"replica metric {name} has shape {value_array.shape}, expected {(population * replicas,)}"
            )
        arrays[f"replica_metric__{name}"] = value_array.reshape(population, replicas)
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_path, snapshot_path)
    best_parameters = np.asarray(candidates[int(backend_best_index)], dtype=np.float32)
    report = {
        "schema_version": "stage3_cem_iteration_snapshot_v2",
        "iteration": int(iteration),
        "contract_sha256": str(contract_sha256),
        "population": population,
        "replicas": replicas,
        "backend_best_index": int(backend_best_index),
        "backend_best_parameter_f32_sha256": hashlib.sha256(best_parameters.tobytes(order="C")).hexdigest(),
        "search_frontier_candidate_indices": list(search_frontier_candidate_indices),
        "coordinate_probe_candidate_indices": list(coordinate_probe_candidate_indices),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    report_path = snapshot_path.with_suffix(".json")
    temporary_report_path = report_path.with_suffix(".json.tmp")
    temporary_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report_path, report_path)
    return report


def _inject_coordinate_probe_candidates(
    candidates: np.ndarray,
    *,
    center: np.ndarray,
    trainable_parameter_mask: np.ndarray,
    radius: float,
    start_index: int = 2,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Place deterministic positive/negative axis probes in a CEM population."""

    candidate_array = np.asarray(candidates, dtype=np.float32).copy()
    center_array = np.asarray(center, dtype=np.float32)
    mask = np.asarray(trainable_parameter_mask, dtype=bool)
    if candidate_array.ndim != 2 or center_array.shape != candidate_array.shape[1:]:
        raise ValueError("coordinate probe candidates and center have incompatible shapes")
    if mask.shape != center_array.shape:
        raise ValueError("coordinate probe trainable mask has an incompatible shape")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("coordinate probe radius must be finite and positive")
    if start_index < 0:
        raise ValueError("coordinate probe start index must be non-negative")
    trainable_indices = np.flatnonzero(mask)
    required_stop = int(start_index) + 2 * int(trainable_indices.size)
    if required_stop > candidate_array.shape[0]:
        raise ValueError(
            "population is too small for positive/negative coordinate probes: "
            f"need at least {required_stop}, got {candidate_array.shape[0]}"
        )
    probe_indices: list[int] = []
    cursor = int(start_index)
    for parameter_index in trainable_indices:
        for direction in (1.0, -1.0):
            candidate_array[cursor] = center_array
            candidate_array[cursor, int(parameter_index)] += np.float32(direction * float(radius))
            probe_indices.append(cursor)
            cursor += 1
    return candidate_array, tuple(probe_indices)


def _inject_search_frontier_copies(
    candidates: np.ndarray,
    *,
    frontier: np.ndarray,
    copies: int,
    start_index: int = 1,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Replay one optimizer frontier in several independent batch contexts."""

    candidate_array = np.asarray(candidates, dtype=np.float32).copy()
    frontier_array = np.asarray(frontier, dtype=np.float32)
    if candidate_array.ndim != 2 or frontier_array.shape != candidate_array.shape[1:]:
        raise ValueError("search-frontier copies and candidates have incompatible shapes")
    if isinstance(copies, bool) or int(copies) <= 0:
        raise ValueError("search-frontier copy count must be positive")
    if isinstance(start_index, bool) or int(start_index) < 0:
        raise ValueError("search-frontier copy start index must be non-negative")
    stop_index = int(start_index) + int(copies)
    if stop_index > int(candidate_array.shape[0]):
        raise ValueError("search-frontier copies do not fit in the CEM population")
    candidate_array[int(start_index) : stop_index] = frontier_array
    return candidate_array, tuple(range(int(start_index), stop_index))


def _build_search_frontier_challenge_batch(
    template: np.ndarray,
    *,
    incumbent: np.ndarray,
    challenger: np.ndarray,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Put an incumbent and challenger in equal halves of one backend batch."""

    batch = np.asarray(template, dtype=np.float32).copy()
    incumbent_array = np.asarray(incumbent, dtype=np.float32)
    challenger_array = np.asarray(challenger, dtype=np.float32)
    if batch.ndim != 2:
        raise ValueError("search-frontier challenge template must be rank 2")
    if incumbent_array.shape != batch.shape[1:] or challenger_array.shape != batch.shape[1:]:
        raise ValueError("search-frontier challenge parameters have incompatible shapes")
    if batch.shape[0] < 2 or batch.shape[0] % 2 != 0:
        raise ValueError("search-frontier challenge requires an even batch size >= 2")
    midpoint = int(batch.shape[0] // 2)
    incumbent_slice = slice(0, midpoint)
    challenger_slice = slice(midpoint, int(batch.shape[0]))
    batch[incumbent_slice] = incumbent_array
    batch[challenger_slice] = challenger_array
    return batch, (incumbent_slice, challenger_slice)


def _source_actor(checkpoint: Path):
    metadata = load_training_checkpoint_metadata(checkpoint)
    config = dict(metadata.get("config", {}) or {})
    hidden = tuple(int(value) for value in metadata.get("hidden", config.get("hidden", ())))
    if not hidden:
        raise ValueError("source checkpoint has no actor hidden-size contract")
    template = init_agent(
        jax.random.PRNGKey(0),
        obs_size=int(metadata["obs_size"]),
        action_size=int(metadata["action_size"]),
        hidden=hidden,
        action_std_init=float(config.get("action_std_init", 0.35)),
        policy_delta_hidden=tuple(config.get("policy_delta_hidden", ())),
        policy_refinement_delta_hidden=tuple(config.get("policy_refinement_delta_hidden", ())),
        policy_correction_hidden=tuple(config.get("policy_correction_hidden", ())),
        correction_action_size=len(tuple(config.get("policy_trainable_action_indices", ()))),
        correction_std_init=tuple(config.get("correction_std_init", ())),
    )
    restored = load_training_actor_checkpoint(checkpoint, agent_template=template)
    return restored, metadata


def _build_explicit_cpu_actor_fn(actor: Any) -> tuple[Any, Any]:
    """Place the frozen actor and its compiled mean function on JAX CPU.

    A MuJoCo CPU replay is not an independent CPU audit if the inherited
    policy still executes on whichever GPU happens to run CEM.  CPU and GPU
    GEMM can differ by a few float32 ulps; a grazing racket collision can then
    amplify that tiny difference into hit versus miss.  Bind both parameters
    and executable explicitly so the audit is reproducible across search
    GPUs and launch environments.
    """

    cpu_devices = tuple(jax.devices("cpu"))
    if not cpu_devices:
        raise RuntimeError("CPU audit requires an available JAX CPU device")
    cpu_device = cpu_devices[0]
    cpu_actor = jax.tree_util.tree_map(
        lambda value: jax.device_put(value, cpu_device),
        actor,
    )
    actor_fn = jax.jit(lambda value: _inherited_policy_mean(cpu_actor, value))
    return actor_fn, cpu_device


def _rank_components(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """High-to-low staged teacher objective.

    Before any useful return exists, replica-robust contact dominates the
    continuous acquisition objective.  Once a candidate produces a strict
    upward-and-forward return on at least one replica, however, its joint
    return-quality rate must guide robustification.  Requiring the contact gate
    first made six downward rebounds outrank five useful returns and pulled CEM
    back into the stable side/downward local optimum.  Final teacher promotion
    remains fail-closed behind the configured replica fraction, legal-return,
    real-cross and CPU gates.  The broad high-region preference contributes
    only outside the allowed window.
    """

    rebound = metrics["event_rebound"].astype(np.float64)
    rebound_rate = metrics.get("event_rebound_rate", rebound).astype(np.float64)
    stringbed_contact = metrics.get("stringbed_contact", rebound).astype(np.float64)
    stringbed_contact_rate = metrics.get(
        "stringbed_contact_rate",
        stringbed_contact,
    ).astype(np.float64)
    positive_z = (metrics["outgoing_z_m_s"] > 0.5).astype(np.float64)
    positive_z_rate = metrics.get("positive_outgoing_z_rate", positive_z).astype(np.float64)
    positive_forward = (metrics["outgoing_forward_m_s"] > 2.0).astype(np.float64)
    positive_forward_rate = metrics.get(
        "positive_outgoing_forward_rate",
        positive_forward,
    ).astype(np.float64)
    high_region_rate = metrics.get(
        "high_region_contact_rate",
        metrics.get("high_region_contact", rebound),
    ).astype(np.float64)
    high_region = metrics.get("high_region_contact", rebound).astype(np.float64)
    teacher_success_rate = metrics.get(
        "teacher_success_rate",
        rebound * positive_z * positive_forward * high_region,
    ).astype(np.float64)
    return_quality_rate = metrics.get(
        "return_quality_rate",
        np.minimum(positive_z_rate, positive_forward_rate),
    ).astype(np.float64)
    any_return_quality = (return_quality_rate > 0.0).astype(np.float64)
    no_fall_rate = metrics.get(
        "no_fall_rate",
        metrics["no_fall"].astype(np.float64),
    ).astype(np.float64)
    safe_no_fall = (no_fall_rate >= SEARCH_SAFE_NO_FALL_RATE - REPLICA_FRACTION_INTEGER_TOLERANCE).astype(np.float64)
    teacher_success = metrics.get(
        "teacher_success",
        rebound * positive_z * positive_forward * high_region,
    ).astype(np.float64)
    any_teacher_success = (teacher_success_rate > 0.0).astype(np.float64)
    soft_high_excess = metrics.get(
        "soft_high_region_excess_m",
        np.zeros_like(rebound),
    ).astype(np.float64)
    contact_acquisition_cost = metrics.get(
        "contact_acquisition_cost_m",
        metrics["min_ball_racket_distance_m"] + SOFT_HIGH_REGION_EXCESS_WEIGHT * soft_high_excess,
    ).astype(np.float64)
    robust_positive_z_rate = np.where(rebound > 0.0, positive_z_rate, 0.0)
    robust_positive_forward_rate = np.where(
        rebound > 0.0,
        positive_forward_rate,
        0.0,
    )
    robust_high_region_rate = np.where(rebound > 0.0, high_region_rate, 0.0)
    stringbed_contact_closing_speed = metrics.get(
        "stringbed_contact_closing_speed_m_s",
        metrics.get(
            "stringbed_contact_speed_m_s",
            metrics.get("hit_contact_speed_m_s", np.zeros_like(rebound)),
        ),
    ).astype(np.float64)
    legal_prediction = (metrics["predicted_clearance_m"] > 0.20).astype(np.float64)
    legal_return_rate = metrics.get(
        "legal_return_rate",
        np.zeros_like(rebound),
    ).astype(np.float64)
    ballistic_return_progress = metrics.get(
        "ballistic_return_progress_score",
        np.zeros_like(rebound),
    ).astype(np.float64)
    ballistic_return_progress_mean = metrics.get(
        "ballistic_return_progress_mean_score",
        ballistic_return_progress,
    ).astype(np.float64)
    return_direction_signed_score = metrics.get(
        "return_direction_signed_score",
        np.zeros_like(rebound),
    ).astype(np.float64)
    outgoing_lateral_abs = metrics.get(
        "outgoing_lateral_abs_m_s",
        np.zeros_like(rebound),
    ).astype(np.float64)
    return (
        teacher_success,
        # Once at least one complete success exists, prefer candidates whose
        # other perturbations clear the explicit safety floor before buying
        # more unsafe return-quality replicas.  Above that floor, tiny
        # one-lane no-fall differences must not block ballistic learning.
        any_teacher_success,
        np.where(any_teacher_success > 0.0, safe_no_fall, 0.0),
        teacher_success_rate,
        np.where(any_teacher_success > 0.0, no_fall_rate, 0.0),
        # A strict upward-and-forward return is a better robustification seed
        # than a larger number of side/downward rebounds.  Reward it before
        # the robust contact boolean, while keeping safety ahead of buying
        # more useful replicas.  Separate marginal rates cannot satisfy this
        # same-replica quantity.
        any_return_quality,
        np.where(any_return_quality > 0.0, safe_no_fall, 0.0),
        # Optimize the actual deployment objective before buying more coarse
        # upward+forward events.  This mean already weights every replica
        # (misses contribute zero) and combines forward/vertical margin,
        # drag-aware clearance, and signed direction.  Putting the coarse
        # count first let v43v double lateral speed while progress decreased.
        np.where(
            any_return_quality > 0.0,
            ballistic_return_progress_mean,
            0.0,
        ),
        return_quality_rate,
        np.where(any_return_quality > 0.0, no_fall_rate, 0.0),
        np.where(any_return_quality > 0.0, rebound_rate, 0.0),
        # If no candidate has a useful return, retain the original staged
        # search: robust rebound/contact before proximity and impact quality.
        rebound,
        # Opt-in drag-aware searches expose these fields.  Historical search
        # metrics omit them and therefore preserve their exact ordering.
        legal_return_rate,
        ballistic_return_progress,
        robust_positive_z_rate,
        # Before any replica has an upward return, retain the real rebound
        # and climb continuously in vertical velocity.  Putting forward rate
        # here first created a stable but downward smash-like local optimum.
        np.where(rebound > 0.0, metrics["outgoing_z_m_s"], 0.0),
        robust_positive_forward_rate,
        metrics["no_fall"].astype(np.float64),
        # The high region is intentionally behind physical return quality.  It
        # is a broad window, not an exact-apex timing target.
        robust_high_region_rate,
        # Stable stringbed contact is an intermediate stage.  A one-replica
        # touch cannot outrank proximity, but a replica-robust touch can guide
        # closing speed and face orientation toward a real rebound.
        np.where(rebound > 0.0, 1.0, stringbed_contact),
        np.where(
            rebound > 0.0,
            1.0,
            np.where(stringbed_contact > 0.0, stringbed_contact_rate, 0.0),
        ),
        np.where(
            rebound > 0.0,
            rebound_rate,
            np.where(stringbed_contact > 0.0, rebound_rate, -contact_acquisition_cost),
        ),
        np.where(
            rebound > 0.0,
            positive_z,
            metrics.get(
                "closest_inverse_impact_decomposed_score",
                -metrics["min_ball_racket_distance_m"],
            ),
        ),
        np.where(
            rebound > 0.0,
            0.0,
            np.where(
                stringbed_contact > 0.0,
                stringbed_contact_closing_speed,
                0.0,
            ),
        ),
        np.where(
            rebound > 0.0,
            0.0,
            metrics.get("closest_inverse_impact_normal_alignment", np.zeros_like(rebound)),
        ),
        np.where(
            rebound > 0.0,
            0.0,
            -metrics.get("closest_inverse_impact_racket_velocity_error_m_s", np.zeros_like(rebound)),
        ),
        np.where(rebound > 0.0, positive_z, -contact_acquisition_cost),
        np.where(rebound > 0.0, positive_forward, 0.0),
        np.where(rebound > 0.0, metrics["outgoing_forward_m_s"], 0.0),
        legal_prediction,
        metrics["crossed_net"].astype(np.float64),
        metrics["opponent_back_landing"].astype(np.float64),
        metrics["predicted_clearance_m"],
        return_direction_signed_score,
        -outgoing_lateral_abs,
        -metrics["correction_rate_cost"],
        -metrics["correction_rms"],
    )


def _rank_order(metrics: dict[str, np.ndarray]) -> np.ndarray:
    # np.lexsort uses its final key as the primary key.
    return np.lexsort(tuple(reversed(_rank_components(metrics))))


def _candidate_metrics_improve_frontier(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any],
    *,
    metric_names: tuple[str, ...],
) -> bool:
    """Compare one unqualified search result against the cumulative frontier."""

    if reference is None:
        return True
    missing = [name for name in metric_names if name not in reference or name not in candidate]
    if missing:
        raise ValueError("search-frontier comparison is missing metrics: " + ", ".join(missing))
    pair_metrics = {name: np.asarray([reference[name], candidate[name]]) for name in metric_names}
    return int(_rank_order(pair_metrics)[-1]) == 1


def _candidate_summary(
    metrics: dict[str, np.ndarray],
    index: int,
    *,
    min_replica_fraction: float = 1.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in metrics.items():
        value = np.asarray(values)[int(index)]
        result[name] = bool(value) if np.issubdtype(np.asarray(value).dtype, np.bool_) else float(value)
    if "teacher_success" in result:
        result["teacher_success"] = bool(result["teacher_success"] and result["no_fall"])
    elif "teacher_success_rate" in result:
        result["teacher_success"] = bool(
            result["teacher_success_rate"] >= float(min_replica_fraction) - REPLICA_FRACTION_INTEGER_TOLERANCE
            and result["no_fall"]
        )
    else:
        result["teacher_success"] = bool(
            result["event_rebound"]
            and result["outgoing_z_m_s"] >= 0.5
            and result["outgoing_forward_m_s"] >= 2.0
            and result.get("high_region_contact", False)
            and result["no_fall"]
        )
    result["preferred_teacher_success"] = bool(
        result["teacher_success"] and (result["predicted_clearance_m"] >= 0.20 or result["crossed_net"])
    )
    return result


def _wandb_iteration_best_payload(
    metrics: dict[str, Any],
) -> dict[str, float]:
    """Expose the current search frontier even before teacher promotion.

    A strict legal-return search can spend many iterations without a promoted
    global best.  Logging only ``global_best_metrics`` made those healthy
    iterations look empty in W&B and hid whether contact, velocity, or
    clearance was improving.  Candidate summaries are scalar by contract; be
    conservative here and ignore any future non-scalar diagnostic fields.
    """

    payload: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            payload[f"cem/iteration_best_{name}"] = float(value)
    return payload


def _cem_failure_reason(total_cpu_audits: int) -> str:
    """Report which fail-closed promotion layer rejected the search."""

    if int(total_cpu_audits) < 0:
        raise ValueError("total CPU audit count must be non-negative")
    if int(total_cpu_audits) == 0:
        return "no_backend_candidate_passed_strict_teacher_gate"
    return "no_candidate_passed_independent_cpu_quality_gate"


def _aggregate_replica_metrics(
    metrics: dict[str, np.ndarray],
    *,
    population: int,
    replicas: int,
    min_replica_fraction: float,
    min_outgoing_z_m_s: float = 0.5,
    min_forward_m_s: float = 2.0,
    require_legal_return_for_teacher: bool = False,
    require_real_net_cross_for_teacher: bool = False,
    real_net_cross_authoritative_for_teacher: bool = False,
    min_predicted_clearance_m: float = 0.20,
    min_return_direction_signed_score: float = 0.65,
    max_event_settled_velocity_delta_m_s: float = MAX_EVENT_SETTLED_VELOCITY_DELTA_M_S,
) -> dict[str, np.ndarray]:
    """Reduce repeated simulations into robust candidate-level metrics."""

    shaped = {name: np.asarray(values).reshape(population, replicas) for name, values in metrics.items()}
    raw_event = shaped["event_rebound"].astype(bool)
    if "event_settled_velocity_delta_m_s" not in shaped:
        raise ValueError("CEM metrics lack event-to-settled velocity consistency")
    event_settled_delta = shaped["event_settled_velocity_delta_m_s"]
    event_consistent = (
        raw_event
        & np.isfinite(event_settled_delta)
        & (event_settled_delta <= float(max_event_settled_velocity_delta_m_s))
    )
    event = raw_event & event_consistent
    stringbed_contact = shaped["stringbed_contact"].astype(bool)
    no_fall = shaped["no_fall"].astype(bool)
    high_region = shaped["high_region_contact"].astype(bool)
    positive_z = event & (shaped["outgoing_z_m_s"] >= float(min_outgoing_z_m_s))
    positive_forward = event & (shaped["outgoing_forward_m_s"] >= float(min_forward_m_s))
    return_quality = positive_z & positive_forward
    if bool(real_net_cross_authoritative_for_teacher) and not bool(require_real_net_cross_for_teacher):
        raise ValueError("authoritative real-cross mode requires a real net-cross gate")
    if bool(require_legal_return_for_teacher):
        if "return_direction_signed_score" not in shaped:
            raise ValueError("legal-return CEM requires return-direction metrics")
        legal_return = event & (shaped["return_direction_signed_score"] >= float(min_return_direction_signed_score))
        if not bool(real_net_cross_authoritative_for_teacher):
            legal_return &= shaped["predicted_clearance_m"] >= float(min_predicted_clearance_m)
        if bool(require_real_net_cross_for_teacher):
            legal_return = legal_return & shaped["crossed_net"].astype(bool)
    else:
        legal_return = event & ((shaped["predicted_clearance_m"] >= 0.20) | shaped["crossed_net"].astype(bool))
    teacher_success = positive_z & positive_forward & no_fall & high_region
    if bool(require_legal_return_for_teacher):
        teacher_success = teacher_success & legal_return
    preferred = teacher_success & legal_return

    if bool(require_legal_return_for_teacher):
        if "return_clearance_score" not in shaped:
            raise ValueError("legal-return CEM requires drag-aware clearance scores")

        def sigmoid(value: np.ndarray) -> np.ndarray:
            clipped = np.clip(value, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-clipped))

        ballistic_progress = event.astype(np.float64)
        ballistic_progress *= sigmoid((shaped["outgoing_forward_m_s"] - float(min_forward_m_s)) / 1.0)
        ballistic_progress *= sigmoid((shaped["outgoing_z_m_s"] - float(min_outgoing_z_m_s)) / 0.75)
        ballistic_progress *= np.clip(shaped["return_clearance_score"], 0.0, 1.0)
        ballistic_progress *= sigmoid(
            (shaped["return_direction_signed_score"] - float(min_return_direction_signed_score)) / 0.10
        )
    else:
        ballistic_progress = np.zeros_like(shaped["outgoing_z_m_s"], dtype=np.float64)
    required_replica_count = _required_replica_count(
        int(replicas),
        float(min_replica_fraction),
    )
    event_rate = event.mean(axis=1)
    stringbed_contact_rate = stringbed_contact.mean(axis=1)
    result: dict[str, np.ndarray] = {
        "min_ball_racket_distance_m": np.quantile(
            shaped["min_ball_racket_distance_m"],
            0.75,
            axis=1,
        ),
        "event_rebound": event.sum(axis=1) >= required_replica_count,
        "event_rebound_rate": event_rate,
        "raw_event_rebound_rate": raw_event.mean(axis=1),
        "event_settled_velocity_consistency_rate": event_consistent.mean(axis=1),
        "event_settled_velocity_delta_m_s": np.quantile(
            event_settled_delta,
            0.75,
            axis=1,
        ),
        "stringbed_contact": stringbed_contact.sum(axis=1) >= required_replica_count,
        "stringbed_contact_rate": stringbed_contact_rate,
        "high_region_contact": high_region.sum(axis=1) >= required_replica_count,
        "high_region_contact_rate": high_region.mean(axis=1),
        "soft_high_region_excess_m": np.quantile(shaped["soft_high_region_excess_m"], 0.75, axis=1),
        "positive_outgoing_z_rate": positive_z.mean(axis=1),
        "positive_outgoing_forward_rate": positive_forward.mean(axis=1),
        "return_quality": return_quality.sum(axis=1) >= required_replica_count,
        "return_quality_rate": return_quality.mean(axis=1),
        # A production teacher must not buy robust hit statistics by allowing
        # the remaining perturbations to fall.  Keep the fractional signal
        # for search shaping, but reserve the boolean success tier for a
        # robust return whose complete replica set remains upright.
        "teacher_success": ((teacher_success.sum(axis=1) >= required_replica_count) & no_fall.all(axis=1)),
        "teacher_success_rate": teacher_success.mean(axis=1),
        "preferred_teacher_success": preferred.sum(axis=1) >= required_replica_count,
        "preferred_teacher_success_rate": preferred.mean(axis=1),
        # A lower quartile makes a candidate improve the whole replica set,
        # instead of winning on one numerically brittle collision.
        "outgoing_z_m_s": np.quantile(shaped["outgoing_z_m_s"], 0.25, axis=1),
        "outgoing_forward_m_s": np.quantile(
            shaped["outgoing_forward_m_s"],
            0.25,
            axis=1,
        ),
        "predicted_clearance_m": np.quantile(
            shaped["predicted_clearance_m"],
            0.25,
            axis=1,
        ),
        "crossed_net": shaped["crossed_net"].astype(bool).sum(axis=1) >= required_replica_count,
        "opponent_back_landing": (shaped["opponent_back_landing"].astype(bool).sum(axis=1) >= required_replica_count),
        "no_fall": no_fall.all(axis=1),
        "no_fall_rate": no_fall.mean(axis=1),
        "correction_rms": shaped["correction_rms"].mean(axis=1),
        "correction_rate_cost": shaped["correction_rate_cost"].mean(axis=1),
        "stringbed_height_deficit_at_hit_m": np.quantile(shaped["stringbed_height_deficit_at_hit_m"], 0.75, axis=1),
        "hand_height_deficit_at_hit_m": np.quantile(shaped["hand_height_deficit_at_hit_m"], 0.75, axis=1),
        "hit_racket_vertical_velocity_m_s": np.quantile(shaped["hit_racket_vertical_velocity_m_s"], 0.25, axis=1),
        "hit_contact_speed_m_s": np.quantile(shaped["hit_contact_speed_m_s"], 0.25, axis=1),
        "stringbed_contact_speed_m_s": np.quantile(shaped["stringbed_contact_speed_m_s"], 0.25, axis=1),
        "stringbed_contact_closing_speed_m_s": np.quantile(shaped["stringbed_contact_closing_speed_m_s"], 0.25, axis=1),
        "closest_inverse_impact_decomposed_score": np.quantile(
            shaped["closest_inverse_impact_decomposed_score"], 0.25, axis=1
        ),
        "closest_inverse_impact_normal_alignment": np.quantile(
            shaped["closest_inverse_impact_normal_alignment"], 0.25, axis=1
        ),
        "closest_inverse_impact_racket_velocity_error_m_s": np.quantile(
            shaped["closest_inverse_impact_racket_velocity_error_m_s"], 0.75, axis=1
        ),
    }
    if bool(require_legal_return_for_teacher):
        result["legal_return_rate"] = legal_return.mean(axis=1)
        result["ballistic_return_progress_score"] = np.quantile(
            ballistic_progress,
            0.25,
            axis=1,
        )
        result["ballistic_return_progress_mean_score"] = np.mean(
            ballistic_progress,
            axis=1,
        )
        result["outgoing_lateral_abs_m_s"] = np.quantile(
            np.abs(shaped["outgoing_lateral_m_s"]),
            0.75,
            axis=1,
        )
        result["return_direction_signed_score"] = np.quantile(
            shaped["return_direction_signed_score"],
            0.25,
            axis=1,
        )
        result["return_clearance_score"] = np.quantile(
            shaped["return_clearance_score"],
            0.25,
            axis=1,
        )
    result["contact_acquisition_cost_m"] = (
        result["min_ball_racket_distance_m"] + SOFT_HIGH_REGION_EXCESS_WEIGHT * result["soft_high_region_excess_m"]
    )
    return result


def _anatomical_synergy_basis(
    actuator_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return signed right-arm correction synergies in actuator-name order.

    The entries are *corrections* around the inherited policy, so negative
    values are meaningful: they reduce an antagonist's inherited excitation.
    Keeping the basis name-driven makes the CEM artifact independent of the
    absolute 354-D actuator indices while the policy contract still seals the
    exact mapping back into the model.
    """

    expected = {
        "DELT1",
        "DELT2",
        "DELT3",
        "SUPSP",
        "INFSP",
        "SUBSC",
        "TMIN",
        "TMAJ",
        "PECM1",
        "PECM2",
        "PECM3",
        "LAT1",
        "LAT2",
        "LAT3",
        "CORB",
        "TRIlong",
        "TRIlat",
        "TRImed",
        "ANC",
        "SUP",
        "BIClong",
        "BICshort",
        "BRA",
        "BRD",
        "ECRL",
        "ECRB",
        "ECU",
        "FCR",
        "FCU",
        "PL",
        "PT",
        "PQ",
    }
    if len(actuator_names) != 32 or set(actuator_names) != expected:
        raise ValueError("anatomical CEM basis requires the canonical 32 right-arm actuators")
    index = {name: i for i, name in enumerate(actuator_names)}

    def row(positive: tuple[str, ...], negative: tuple[str, ...]) -> np.ndarray:
        value = np.zeros((len(actuator_names),), dtype=np.float32)
        for name in positive:
            value[index[name]] = 1.0
        for name in negative:
            value[index[name]] = -1.0
        return value

    shoulder_internal = (
        "SUBSC",
        "TMAJ",
        "PECM1",
        "PECM2",
        "PECM3",
        "LAT1",
        "LAT2",
        "LAT3",
    )
    shoulder_external = ("INFSP", "TMIN", "DELT3")
    elbow_extensors = ("TRIlong", "TRIlat", "TRImed", "ANC")
    elbow_flexors = ("BIClong", "BICshort", "BRA", "BRD")
    wrist_extensors = ("ECRL", "ECRB", "ECU")
    wrist_flexors = ("FCR", "FCU", "PL")
    radial = ("ECRL", "ECRB", "FCR")
    ulnar = ("ECU", "FCU")
    definitions = (
        ("shoulder_elevation", ("DELT1", "DELT2", "SUPSP", "CORB"), ("LAT1", "LAT2", "LAT3")),
        ("shoulder_retraction", ("DELT3", "INFSP", "TMIN"), ("PECM1", "PECM2", "PECM3")),
        ("shoulder_internal_rotation", shoulder_internal, shoulder_external),
        ("shoulder_external_rotation", shoulder_external, shoulder_internal),
        ("elbow_extension", elbow_extensors, elbow_flexors),
        ("elbow_flexion", elbow_flexors, elbow_extensors),
        ("forearm_pronation", ("PT", "PQ"), ("SUP",)),
        ("forearm_supination", ("SUP",), ("PT", "PQ")),
        ("wrist_extension", wrist_extensors, wrist_flexors),
        ("wrist_flexion", wrist_flexors, wrist_extensors),
        ("wrist_radial_deviation", radial, ulnar),
        ("wrist_ulnar_deviation", ulnar, radial),
    )
    names = tuple(item[0] for item in definitions)
    basis = np.stack([row(item[1], item[2]) for item in definitions], axis=0)
    # A coefficient should have comparable scale regardless of how many
    # overlapping synergies contain a muscle.
    column_norm = np.maximum(np.sqrt(np.square(basis).sum(axis=0)), 1.0)
    basis = basis / column_norm[None, :]
    return basis.astype(np.float32), names


def _trainable_parameter_mask(
    *,
    parameterization: str,
    synergy_names: tuple[str, ...],
    time_knots: int,
    requested_synergies: tuple[str, ...] | None,
    requested_knot_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """Select anatomical synergies without changing the teacher artifact ABI.

    Candidate artifacts always retain the complete ``time_knots x latent``
    parameter vector.  A local repair can nevertheless freeze proximal body
    synergies and search only forearm/wrist coefficients.  Keeping the full
    vector makes such a candidate replayable by the existing loader while the
    search contract records exactly which degrees of freedom were allowed to
    move.
    """

    if int(time_knots) <= 0:
        raise ValueError("time_knots must be positive")

    if requested_synergies is None:
        names = synergy_names
    else:
        if parameterization != "anatomical_synergies":
            raise ValueError("--trainable-synergies requires anatomical_synergies parameterization")
        if not requested_synergies:
            raise ValueError("--trainable-synergies must not be empty")
        if len(set(requested_synergies)) != len(requested_synergies):
            raise ValueError("--trainable-synergies contains duplicate names")
        unknown = sorted(set(requested_synergies) - set(synergy_names))
        if unknown:
            raise ValueError("unknown trainable anatomical synergies: " + ", ".join(unknown))
        names = tuple(name for name in synergy_names if name in requested_synergies)

    if parameterization == "anatomical_synergies":
        latent_mask = np.asarray([name in names for name in synergy_names], dtype=bool)
    else:
        # Muscle-knot searches have no named anatomical latent basis.  The
        # default remains the historical all-parameter search.
        latent_mask = np.ones((len(synergy_names),), dtype=bool)
    if latent_mask.size == 0 or not bool(latent_mask.any()):
        raise ValueError("at least one CEM latent dimension must remain trainable")

    if requested_knot_indices is None:
        knot_indices = tuple(range(int(time_knots)))
    else:
        if not requested_knot_indices:
            raise ValueError("--trainable-knot-indices must not be empty")
        if len(set(requested_knot_indices)) != len(requested_knot_indices):
            raise ValueError("--trainable-knot-indices contains duplicates")
        invalid = sorted(index for index in requested_knot_indices if index < 0 or index >= int(time_knots))
        if invalid:
            raise ValueError(
                "trainable knot indices outside [0, time_knots): " + ", ".join(str(index) for index in invalid)
            )
        knot_indices = tuple(sorted(int(index) for index in requested_knot_indices))
    knot_mask = np.asarray(
        [index in knot_indices for index in range(int(time_knots))],
        dtype=bool,
    )
    parameter_mask = (knot_mask[:, None] & latent_mask[None, :]).reshape(-1)
    if not bool(parameter_mask.any()):
        raise ValueError("at least one CEM knot/latent parameter must remain trainable")
    return parameter_mask, names, knot_indices


def _interpolate_correction_knots(
    parameters: jax.Array,
    trajectory_phase: jax.Array,
    *,
    num_envs: int,
    time_knots: int,
    output_size: int,
    synergy_basis: jax.Array | None,
) -> jax.Array:
    """Interpolate temporal CEM knots and optionally expand synergies."""

    latent_size = output_size if synergy_basis is None else int(synergy_basis.shape[0])
    knots = parameters.reshape(num_envs, time_knots, latent_size)
    knot_position = jnp.clip(trajectory_phase, 0.0, 1.0) * float(time_knots - 1)
    lower = jnp.floor(knot_position).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, time_knots - 1)
    fraction = knot_position - lower.astype(knot_position.dtype)
    rows = jnp.arange(num_envs, dtype=jnp.int32)
    latent = (1.0 - fraction[:, None]) * knots[rows, lower] + fraction[:, None] * knots[rows, upper]
    if synergy_basis is None:
        return latent
    return jnp.clip(latent @ synergy_basis, -3.0, 3.0)


def _make_rollout(
    env: IncomingHitMjxEnv,
    *,
    num_envs: int,
    actor: Any,
    obs_mean: jax.Array,
    obs_var: jax.Array,
    selected_indices: tuple[int, ...],
    physical_scales: jax.Array,
    residual_scale_vector: jax.Array,
    open_s: float,
    close_s: float,
    smoothing_s: float,
    time_knots: int,
    synergy_basis: jax.Array | None,
    max_stringbed_height_deficit_m: float,
    max_hand_height_deficit_m: float,
):
    mx = env.put_model(num_envs)
    template = env.make_batched_template(num_envs)
    reset = env.make_reset_fn(mx, num_envs)
    step_env = env.make_step_fn(mx, num_envs)
    right_arm_body_ids = jnp.asarray(
        [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in RIGHT_ARM_AUDIT_BODY_NAMES],
        dtype=jnp.int32,
    )
    if bool(np.any(np.asarray(right_arm_body_ids) < 0)):
        raise ValueError("MJX scene is missing one or more right-arm audit bodies")

    @jax.jit
    def rollout(parameters: jax.Array, key: jax.Array, trace_index: jax.Array):
        state = reset(key, template)
        initial = {
            "alive": jnp.ones((num_envs,), dtype=jnp.bool_),
            "min_distance": jnp.full((num_envs,), jnp.inf, dtype=jnp.float32),
            "stringbed_contact": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "event_rebound": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "event_settled_velocity_delta": jnp.full(
                (num_envs,),
                1.0e3,
                dtype=jnp.float32,
            ),
            "outgoing_z": jnp.zeros((num_envs,), dtype=jnp.float32),
            "outgoing_forward": jnp.zeros((num_envs,), dtype=jnp.float32),
            "outgoing_lateral": jnp.zeros((num_envs,), dtype=jnp.float32),
            "predicted_clearance": jnp.full((num_envs,), -1.0e3, dtype=jnp.float32),
            "return_clearance_score": jnp.zeros((num_envs,), dtype=jnp.float32),
            "return_direction_signed_score": jnp.zeros((num_envs,), dtype=jnp.float32),
            "crossed": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "back": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "fall": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "correction_sq_sum": jnp.zeros((num_envs,), dtype=jnp.float32),
            "correction_rate_sq_sum": jnp.zeros((num_envs,), dtype=jnp.float32),
            "correction_count": jnp.zeros((num_envs,), dtype=jnp.float32),
            "previous_correction": jnp.zeros((num_envs, len(selected_indices)), dtype=jnp.float32),
            "max_stringbed_height": jnp.full((num_envs,), -jnp.inf, dtype=jnp.float32),
            "max_hand_height": jnp.full((num_envs,), -jnp.inf, dtype=jnp.float32),
            "hit_stringbed_height": jnp.zeros((num_envs,), dtype=jnp.float32),
            "hit_hand_height": jnp.zeros((num_envs,), dtype=jnp.float32),
            "hit_racket_vertical_velocity": jnp.zeros((num_envs,), dtype=jnp.float32),
            "hit_contact_speed": jnp.zeros((num_envs,), dtype=jnp.float32),
            "stringbed_contact_speed": jnp.zeros((num_envs,), dtype=jnp.float32),
            "stringbed_contact_closing_speed": jnp.zeros((num_envs,), dtype=jnp.float32),
            "closest_stringbed_height": jnp.zeros((num_envs,), dtype=jnp.float32),
            "closest_hand_height": jnp.zeros((num_envs,), dtype=jnp.float32),
            "closest_inverse_score": jnp.zeros((num_envs,), dtype=jnp.float32),
            "closest_inverse_normal_alignment": jnp.zeros((num_envs,), dtype=jnp.float32),
            "closest_inverse_velocity_error": jnp.full((num_envs,), 1.0e3, dtype=jnp.float32),
        }

        def body(carry, _unused):
            state, stats = carry
            obs_norm = jnp.clip(
                (state.obs - obs_mean) / jnp.sqrt(obs_var + 1.0e-8),
                -10.0,
                10.0,
            )
            inherited = jnp.tanh(_inherited_policy_mean(actor, obs_norm))
            elapsed = state.step_index.astype(jnp.float32) * env.control_substeps * env.timestep
            time_to_intercept = env.intercept_times[state.feed_idx] - elapsed
            trajectory_phase = jnp.clip(
                (open_s - time_to_intercept) / (open_s - close_s),
                0.0,
                1.0,
            )
            raw_correction = _interpolate_correction_knots(
                parameters,
                trajectory_phase,
                num_envs=num_envs,
                time_knots=time_knots,
                output_size=len(selected_indices),
                synergy_basis=synergy_basis,
            )
            window = selected_correction_window(
                time_to_intercept,
                open_s=open_s,
                close_s=close_s,
                smoothing_s=smoothing_s,
                array_module=jnp,
            )
            residual = compose_selected_physical_correction(
                inherited,
                raw_correction,
                selected_indices=selected_indices,
                physical_scales=physical_scales,
                inherited_residual_scale=residual_scale_vector,
                window=window,
                array_module=jnp,
            )
            # These are continuous pre-control-step kinematics.  Keep them
            # separate from transition["hit_*"] fields, which are zero on
            # every non-contact step and therefore cannot establish the true
            # reach apex or timing of a swing.
            stringbed_position = state.data.site_xpos[:, env._stringbed_site]
            stringbed_matrix = state.data.site_xmat[:, env._stringbed_site].reshape(-1, 3, 3)
            stringbed_normal = stringbed_matrix[:, :, 2]
            racket_cvel = state.data.cvel[:, env._racket_body]
            racket_offset = stringbed_position - state.data.subtree_com[:, env._racket_root]
            stringbed_linear_velocity = racket_cvel[:, 3:] + jnp.cross(racket_cvel[:, :3], racket_offset)
            shuttle_position = state.data.site_xpos[:, env._cork_site]
            shuttle_cvel = state.data.cvel[:, env.ids.shuttle_body]
            shuttle_offset = shuttle_position - state.data.subtree_com[:, env.ids.shuttle_root]
            shuttle_velocity = shuttle_cvel[:, 3:] + jnp.cross(shuttle_cvel[:, :3], shuttle_offset)
            right_arm_body_position = state.data.xpos[:, right_arm_body_ids]
            next_state, transition = step_env(state, residual)
            alive = stats["alive"]
            correction = window[:, None] * jnp.tanh(raw_correction)
            hit = alive & transition["hit_event"]
            stringbed_contact = alive & transition["stringbed_contact_event"]
            swing_sample = alive & (window > 0.05)
            hand_height = right_arm_body_position[:, -1, 2]
            closer = alive & (transition["ball_racket_distance_m"] < stats["min_distance"])
            stats = {
                **stats,
                "alive": alive & (~transition["done"]),
                "min_distance": jnp.minimum(
                    stats["min_distance"],
                    jnp.where(alive, transition["ball_racket_distance_m"], jnp.inf),
                ),
                "stringbed_contact": stats["stringbed_contact"] | stringbed_contact,
                "event_rebound": stats["event_rebound"] | (alive & transition["event_rebound_event"]),
                "event_settled_velocity_delta": jnp.where(
                    hit,
                    jnp.linalg.norm(
                        transition["hit_outgoing_velocity_xyz_m_s"]
                        - transition["hit_event_impulse_velocity_after_xyz_m_s"],
                        axis=-1,
                    ),
                    stats["event_settled_velocity_delta"],
                ),
                "outgoing_z": jnp.where(hit, transition["hit_outgoing_velocity_z_m_s"], stats["outgoing_z"]),
                "outgoing_forward": jnp.where(
                    hit,
                    transition["hit_outgoing_forward_velocity_m_s"],
                    stats["outgoing_forward"],
                ),
                "outgoing_lateral": jnp.where(
                    hit,
                    transition["hit_outgoing_velocity_y_m_s"],
                    stats["outgoing_lateral"],
                ),
                "predicted_clearance": jnp.where(
                    hit,
                    transition["predicted_net_clearance_m"],
                    stats["predicted_clearance"],
                ),
                "return_clearance_score": jnp.where(
                    hit,
                    transition["return_clearance_score"],
                    stats["return_clearance_score"],
                ),
                "return_direction_signed_score": jnp.where(
                    hit,
                    transition["return_direction_signed_score"],
                    stats["return_direction_signed_score"],
                ),
                "crossed": stats["crossed"] | (alive & transition["valid_net_cross_event"]),
                "back": stats["back"] | (alive & transition["opponent_back_landing"]),
                "fall": stats["fall"] | (alive & transition["body_fall"]),
                "correction_sq_sum": stats["correction_sq_sum"]
                + jnp.where(alive, jnp.mean(jnp.square(correction), axis=-1), 0.0),
                "correction_rate_sq_sum": stats["correction_rate_sq_sum"]
                + jnp.where(
                    alive,
                    jnp.mean(jnp.square(correction - stats["previous_correction"]), axis=-1),
                    0.0,
                ),
                "correction_count": stats["correction_count"] + alive.astype(jnp.float32),
                "previous_correction": jnp.where(alive[:, None], correction, stats["previous_correction"]),
                "max_stringbed_height": jnp.maximum(
                    stats["max_stringbed_height"],
                    jnp.where(swing_sample, stringbed_position[:, 2], -jnp.inf),
                ),
                "max_hand_height": jnp.maximum(
                    stats["max_hand_height"],
                    jnp.where(swing_sample, hand_height, -jnp.inf),
                ),
                "hit_stringbed_height": jnp.where(
                    hit,
                    transition["hit_stringbed_position_xyz_m"][:, 2],
                    stats["hit_stringbed_height"],
                ),
                "hit_hand_height": jnp.where(hit, hand_height, stats["hit_hand_height"]),
                "hit_racket_vertical_velocity": jnp.where(
                    hit,
                    transition["hit_racket_linear_velocity_xyz_m_s"][:, 2],
                    stats["hit_racket_vertical_velocity"],
                ),
                "hit_contact_speed": jnp.where(
                    hit,
                    transition["hit_contact_speed_m_s"],
                    stats["hit_contact_speed"],
                ),
                # A contact can span several control steps.  Keep the peak
                # physically closing speed for the whole episode; overwriting
                # it with a later separating/slow contact hides the margin to
                # the event-rebound threshold from CEM.
                "stringbed_contact_speed": jnp.where(
                    stringbed_contact,
                    jnp.maximum(
                        stats["stringbed_contact_speed"],
                        transition["stringbed_contact_speed_m_s"],
                    ),
                    stats["stringbed_contact_speed"],
                ),
                "stringbed_contact_closing_speed": jnp.where(
                    stringbed_contact,
                    jnp.maximum(
                        stats["stringbed_contact_closing_speed"],
                        transition["stringbed_contact_closing_speed_m_s"],
                    ),
                    stats["stringbed_contact_closing_speed"],
                ),
                "closest_stringbed_height": jnp.where(
                    closer, stringbed_position[:, 2], stats["closest_stringbed_height"]
                ),
                "closest_hand_height": jnp.where(closer, hand_height, stats["closest_hand_height"]),
                "closest_inverse_score": jnp.where(
                    closer,
                    transition["inverse_impact_decomposed_score"],
                    stats["closest_inverse_score"],
                ),
                "closest_inverse_normal_alignment": jnp.where(
                    closer,
                    transition["inverse_impact_normal_alignment"],
                    stats["closest_inverse_normal_alignment"],
                ),
                "closest_inverse_velocity_error": jnp.where(
                    closer,
                    transition["inverse_impact_racket_velocity_error_m_s"],
                    stats["closest_inverse_velocity_error"],
                ),
            }
            trace_step = {
                "observation": state.obs[trace_index],
                "observation_normalized": obs_norm[trace_index],
                "sample_time_s": elapsed[trace_index],
                "time_to_intercept_s": time_to_intercept[trace_index],
                "correction_window": window[trace_index],
                "inherited_residual": inherited[trace_index],
                "correction_raw": raw_correction[trace_index],
                "correction": correction[trace_index],
                "full_residual": residual[trace_index],
                "alive": alive[trace_index],
                "done": transition["done"][trace_index],
                "hit_event": transition["hit_event"][trace_index],
                "stringbed_contact_event": transition["stringbed_contact_event"][trace_index],
                "stringbed_contact_speed_m_s": transition["stringbed_contact_speed_m_s"][trace_index],
                "stringbed_contact_closing_speed_m_s": transition["stringbed_contact_closing_speed_m_s"][trace_index],
                "event_rebound": transition["event_rebound_event"][trace_index],
                "body_fall": transition["body_fall"][trace_index],
                "ball_racket_distance_m": transition["ball_racket_distance_m"][trace_index],
                "predicted_net_clearance_m": transition["predicted_net_clearance_m"][trace_index],
                "return_clearance_score": transition["return_clearance_score"][trace_index],
                "return_direction_signed_score": transition["return_direction_signed_score"][trace_index],
                "valid_net_cross_event": transition["valid_net_cross_event"][trace_index],
                "shuttle_position_xyz_m": shuttle_position[trace_index],
                "shuttle_velocity_xyz_m_s": shuttle_velocity[trace_index],
                "stringbed_position_xyz_m": stringbed_position[trace_index],
                "stringbed_normal_xyz": stringbed_normal[trace_index],
                "stringbed_linear_velocity_xyz_m_s": stringbed_linear_velocity[trace_index],
                "racket_angular_velocity_xyz_rad_s": racket_cvel[trace_index, :3],
                "right_arm_body_position_xyz_m": right_arm_body_position[trace_index],
                "incoming_shuttle_velocity_xyz_m_s": transition["hit_incoming_velocity_xyz_m_s"][trace_index],
                "outgoing_shuttle_velocity_xyz_m_s": transition["hit_outgoing_velocity_xyz_m_s"][trace_index],
                "event_impulse_velocity_after_xyz_m_s": transition["hit_event_impulse_velocity_after_xyz_m_s"][
                    trace_index
                ],
                "event_stringbed_position_xyz_m": transition["hit_stringbed_position_xyz_m"][trace_index],
                "event_stringbed_normal_xyz": transition["hit_stringbed_normal_xyz"][trace_index],
                "event_racket_linear_velocity_xyz_m_s": transition["hit_racket_linear_velocity_xyz_m_s"][trace_index],
                "event_racket_angular_velocity_xyz_rad_s": transition["hit_racket_angular_velocity_xyz_rad_s"][
                    trace_index
                ],
            }
            return (next_state, stats), trace_step

        (_state, stats), trace = jax.lax.scan(
            body,
            (state, initial),
            None,
            length=env.max_episode_steps,
        )
        count = jnp.maximum(stats["correction_count"], 1.0)
        stringbed_height_deficit = jnp.where(
            stats["event_rebound"],
            jnp.maximum(
                0.0,
                stats["max_stringbed_height"] - stats["hit_stringbed_height"],
            ),
            jnp.maximum(
                0.0,
                stats["max_stringbed_height"] - stats["closest_stringbed_height"],
            ),
        )
        hand_height_deficit = jnp.where(
            stats["event_rebound"],
            jnp.maximum(0.0, stats["max_hand_height"] - stats["hit_hand_height"]),
            jnp.maximum(0.0, stats["max_hand_height"] - stats["closest_hand_height"]),
        )
        high_region_contact = (
            stats["event_rebound"]
            & (stringbed_height_deficit <= float(max_stringbed_height_deficit_m))
            & (hand_height_deficit <= float(max_hand_height_deficit_m))
        )
        soft_high_region_excess = jnp.maximum(
            0.0,
            stringbed_height_deficit - float(max_stringbed_height_deficit_m),
        ) + jnp.maximum(
            0.0,
            hand_height_deficit - float(max_hand_height_deficit_m),
        )
        metrics = {
            "min_ball_racket_distance_m": stats["min_distance"],
            "stringbed_contact": stats["stringbed_contact"],
            "event_rebound": stats["event_rebound"],
            "event_settled_velocity_delta_m_s": stats[
                "event_settled_velocity_delta"
            ],
            "outgoing_z_m_s": stats["outgoing_z"],
            "outgoing_forward_m_s": stats["outgoing_forward"],
            "outgoing_lateral_m_s": stats["outgoing_lateral"],
            "predicted_clearance_m": stats["predicted_clearance"],
            "return_clearance_score": stats["return_clearance_score"],
            "return_direction_signed_score": stats["return_direction_signed_score"],
            "crossed_net": stats["crossed"],
            "opponent_back_landing": stats["back"],
            "no_fall": ~stats["fall"],
            "correction_rms": jnp.sqrt(stats["correction_sq_sum"] / count),
            "correction_rate_cost": stats["correction_rate_sq_sum"] / count,
            "high_region_contact": high_region_contact,
            "soft_high_region_excess_m": soft_high_region_excess,
            "stringbed_height_deficit_at_hit_m": stringbed_height_deficit,
            "hand_height_deficit_at_hit_m": hand_height_deficit,
            "hit_racket_vertical_velocity_m_s": stats["hit_racket_vertical_velocity"],
            "hit_contact_speed_m_s": stats["hit_contact_speed"],
            "stringbed_contact_speed_m_s": stats["stringbed_contact_speed"],
            "stringbed_contact_closing_speed_m_s": stats["stringbed_contact_closing_speed"],
            "closest_inverse_impact_decomposed_score": stats["closest_inverse_score"],
            "closest_inverse_impact_normal_alignment": stats["closest_inverse_normal_alignment"],
            "closest_inverse_impact_racket_velocity_error_m_s": stats["closest_inverse_velocity_error"],
        }
        return metrics, trace

    return rollout


def _save_cpu_teacher_trace(
    *,
    path: Path,
    feed: Any,
    paths: Any,
    actor: Any,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    parameters: np.ndarray,
    selected_indices: tuple[int, ...],
    physical_scales: np.ndarray,
    base_policy_artifact: Path,
    residual_scale: float,
    open_s: float,
    close_s: float,
    smoothing_s: float,
    max_episode_steps: int,
    time_knots: int,
    synergy_basis: np.ndarray | None,
    swing_phase_advance_s: float | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    constraints = _return_constraints(paths)
    env = IncomingShuttleHitEnv(
        paths.scene_xml,
        feed_bank=[feed],
        control_substeps=paths.control_substeps,
        max_episode_steps=max_episode_steps,
        reward_weights=paths.reward_weights,
        return_net_x_m=constraints["net_x_m"],
        return_net_height_m=constraints["net_height_m"],
        min_return_net_clearance_m=constraints["min_clearance_m"],
        desired_return_up_component=constraints["desired_up_component"],
        ballistic_return_score_softness_m=constraints["ballistic_score_softness_m"],
        clearance_prediction_mode=constraints["clearance_prediction_mode"],
        shuttle_proximity_softness_m=constraints["shuttle_proximity_softness_m"],
        timed_intercept_softness_m=constraints["timed_intercept_softness_m"],
        direction_distance_softness_m=constraints["direction_distance_softness_m"],
        contact_guidance_reward_mode=constraints["contact_guidance_reward_mode"],
        contact_guidance_discount=constraints["contact_guidance_discount"],
        racket_velocity_direction_fraction=constraints["racket_velocity_direction_fraction"],
        direction_reward_mode=constraints["direction_reward_mode"],
        clearance_reward_mode=constraints["clearance_reward_mode"],
        hit_event_mode=constraints["hit_event_mode"],
        racket_guidance_mode=constraints["racket_guidance_mode"],
        inverse_target_speed_m_s=constraints["inverse_target_speed_m_s"],
        inverse_velocity_softness_m_s=constraints["inverse_velocity_softness_m_s"],
        base_policy_artifact=base_policy_artifact,
        residual_scale=residual_scale,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.76)),
        swing_phase_advance_s=(
            float(paths.stage3_direct.get("swing_phase_advance_s", 0.0))
            if swing_phase_advance_s is None
            else float(swing_phase_advance_s)
        ),
        seed=0,
    )
    obs, _ = env.reset(feed_index=0)
    actor_fn, actor_cpu_device = _build_explicit_cpu_actor_fn(actor)
    latent_size = len(selected_indices) if synergy_basis is None else int(synergy_basis.shape[0])
    knots = np.asarray(parameters, dtype=np.float32).reshape(time_knots, latent_size)
    right_arm_body_ids = np.asarray(
        [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in RIGHT_ARM_AUDIT_BODY_NAMES],
        dtype=np.int32,
    )
    if bool(np.any(right_arm_body_ids < 0)):
        raise ValueError("CPU scene is missing one or more right-arm audit bodies")
    shuttle_body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        env.physics.cfg.shuttle_body_name,
    )
    if shuttle_body_id < 0:
        raise ValueError("CPU scene is missing the shuttle body")
    trace: dict[str, list[Any]] = {
        "observation": [],
        "observation_normalized": [],
        "time_to_intercept_s": [],
        "correction_window": [],
        "inherited_residual": [],
        "correction_raw": [],
        "correction": [],
        "full_residual": [],
        "shuttle_position": [],
        "shuttle_velocity": [],
        "stringbed_position": [],
        "stringbed_normal": [],
        "stringbed_linear_velocity": [],
        "stringbed_angular_velocity": [],
        "right_arm_body_position_xyz_m": [],
        "hit_event": [],
        "event_rebound": [],
        "event_shuttle_velocity_before_world_m_s": [],
        "event_impulse_velocity_after_world_m_s": [],
        "event_racket_surface_velocity_world_m_s": [],
        "event_stringbed_normal_world": [],
        "predicted_net_clearance_m": [],
        "return_direction_signed_score": [],
        "valid_net_cross_event": [],
        "body_fall": [],
    }
    renderer = None
    camera = None
    background_floor_geom_id = -1
    video_frames: list[np.ndarray] = []
    if video_path is not None:
        env.model.vis.global_.offwidth = 1280
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=1280)
        camera = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "overall_view",
        )
        if camera < 0:
            renderer.close()
            raise ValueError("CPU scene is missing the overall_view camera")
        background_floor_geom_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
    final_info: dict[str, Any] = {}
    for _step in range(max_episode_steps):
        obs_norm = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1.0e-8), -10.0, 10.0)
        actor_input = jax.device_put(
            np.asarray(obs_norm, dtype=np.float32),
            actor_cpu_device,
        )
        actor_logits = actor_fn(actor_input)
        if any(device.platform != "cpu" for device in actor_logits.devices()):
            raise RuntimeError("CPU audit actor executable escaped to a non-CPU device")
        inherited = np.tanh(np.asarray(jax.device_get(actor_logits)))
        elapsed = env.step_index * env.control_substeps * float(env.model.opt.timestep)
        tti = float(feed.intercept_time_s) - elapsed
        phase = float(np.clip((open_s - tti) / (open_s - close_s), 0.0, 1.0))
        knot_position = phase * float(time_knots - 1)
        lower = int(np.floor(knot_position))
        upper = min(lower + 1, time_knots - 1)
        fraction = knot_position - float(lower)
        latent = (1.0 - fraction) * knots[lower] + fraction * knots[upper]
        raw = latent if synergy_basis is None else np.clip(latent @ synergy_basis, -3.0, 3.0)
        window = float(
            selected_correction_window(
                tti,
                open_s=open_s,
                close_s=close_s,
                smoothing_s=smoothing_s,
            )
        )
        correction = window * np.tanh(raw)
        residual = compose_selected_physical_correction(
            inherited,
            raw,
            selected_indices=selected_indices,
            physical_scales=physical_scales,
            inherited_residual_scale=residual_scale,
            window=window,
        )
        next_obs, _reward, terminated, truncated, info = env.step(residual)
        trace["observation"].append(np.asarray(obs, dtype=np.float32))
        trace["observation_normalized"].append(np.asarray(obs_norm, dtype=np.float32))
        trace["time_to_intercept_s"].append(tti)
        trace["correction_window"].append(window)
        trace["inherited_residual"].append(np.asarray(inherited, dtype=np.float32))
        trace["correction_raw"].append(np.asarray(raw, dtype=np.float32))
        trace["correction"].append(np.asarray(correction, dtype=np.float32))
        trace["full_residual"].append(np.asarray(residual, dtype=np.float32))
        cork_position = np.asarray(env.data.site_xpos[env._cork_site], dtype=np.float32)
        shuttle_velocity_6d = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            env.model,
            env.data,
            mujoco.mjtObj.mjOBJ_BODY,
            shuttle_body_id,
            shuttle_velocity_6d,
            0,
        )
        shuttle_origin = np.asarray(env.data.xpos[shuttle_body_id], dtype=float)
        cork_velocity = shuttle_velocity_6d[3:] + np.cross(
            shuttle_velocity_6d[:3],
            np.asarray(cork_position, dtype=float) - shuttle_origin,
        )
        trace["shuttle_position"].append(cork_position)
        trace["shuttle_velocity"].append(np.asarray(cork_velocity, dtype=np.float32))
        trace["stringbed_position"].append(np.asarray(env.data.site_xpos[env._stringbed_site], dtype=np.float32))
        trace["stringbed_normal"].append(np.asarray(env._stringbed_normal(), dtype=np.float32))
        trace["stringbed_linear_velocity"].append(np.asarray(env._stringbed_velocity(), dtype=np.float32))
        trace["stringbed_angular_velocity"].append(np.asarray(env._stringbed_angular_velocity(), dtype=np.float32))
        trace["right_arm_body_position_xyz_m"].append(np.asarray(env.data.xpos[right_arm_body_ids], dtype=np.float32))
        trace["hit_event"].append(bool(info.get("hit_this_step", False)))
        trace["event_rebound"].append(bool(info.get("event_rebound_this_step", False)))
        trace["event_shuttle_velocity_before_world_m_s"].append(
            np.asarray(
                info.get("event_shuttle_velocity_before_world_m_s", np.zeros(3)),
                dtype=np.float32,
            )
        )
        trace["event_impulse_velocity_after_world_m_s"].append(
            np.asarray(
                info.get("event_impulse_velocity_after_world_m_s", np.zeros(3)),
                dtype=np.float32,
            )
        )
        trace["event_racket_surface_velocity_world_m_s"].append(
            np.asarray(
                info.get("event_racket_surface_velocity_world_m_s", np.zeros(3)),
                dtype=np.float32,
            )
        )
        trace["event_stringbed_normal_world"].append(
            np.asarray(
                info.get("event_stringbed_normal_world", np.zeros(3)),
                dtype=np.float32,
            )
        )
        flight_info = info.get("flight", {})
        trace["predicted_net_clearance_m"].append(
            float(
                info.get(
                    "predicted_net_clearance_m",
                    flight_info.get("predicted_net_clearance_m", 0.0),
                )
            )
        )
        trace["return_direction_signed_score"].append(
            float(
                info.get(
                    "return_direction_signed_score",
                    flight_info.get("return_direction_signed_score", 0.0),
                )
            )
        )
        trace["valid_net_cross_event"].append(
            bool(
                info.get(
                    "valid_net_cross_event",
                    flight_info.get("valid_net_crossing_event", False),
                )
            )
        )
        trace["body_fall"].append(bool(info.get("body_fall", False)))
        if renderer is not None:
            renderer.update_scene(env.data, camera=camera)
            # Software/offscreen MuJoCo shadow maps can produce large jagged
            # court artifacts on headless machines.  Shadows are presentation
            # only; disabling them leaves the replayed physics untouched.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            # The imported body scene contains an infinite grey plane exactly
            # coplanar with the green badminton-court box.  Hiding only the
            # plane in the copied render scene removes z-fighting without
            # changing either collision geom or the replayed dynamics.
            for scene_geom_index in range(renderer.scene.ngeom):
                scene_geom = renderer.scene.geoms[scene_geom_index]
                if (
                    scene_geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and scene_geom.objid == background_floor_geom_id
                ):
                    scene_geom.rgba[3] = 0.0
            video_frames.append(renderer.render().copy())
        final_info = info
        obs = next_obs
        if terminated or truncated:
            break
    video_report: dict[str, Any] = {}
    if renderer is not None:
        renderer.close()
        if not video_frames:
            raise ValueError("CPU candidate replay produced no video frames")
        import imageio.v2 as imageio

        resolved_video_path = Path(video_path).expanduser().resolve()
        resolved_video_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_video_path.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing demo video: {resolved_video_path}"
            )
        control_dt_s = env.control_substeps * float(env.model.opt.timestep)
        imageio.mimsave(
            resolved_video_path,
            video_frames,
            fps=max(1, int(round(1.0 / control_dt_s))),
            macro_block_size=None,
        )
        video_report = {
            "video_path": str(resolved_video_path),
            "video_sha256": hashlib.sha256(resolved_video_path.read_bytes()).hexdigest(),
            "video_frame_count": len(video_frames),
            "video_fps": max(1, int(round(1.0 / control_dt_s))),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: np.asarray(values) for name, values in trace.items()},
        selected_action_indices=np.asarray(selected_indices, dtype=np.int32),
        physical_scales=np.asarray(physical_scales, dtype=np.float32),
        feed_fingerprint=np.asarray(feed_sample_fingerprint(feed)),
        swing_phase_advance_s=np.asarray(
            float(env.swing_phase_advance_s),
            dtype=np.float32,
        ),
        trace_schema_version=np.asarray(CPU_AUDIT_TRACE_SCHEMA),
        actor_inference_semantics=np.asarray(CPU_ACTOR_INFERENCE_SEMANTICS),
        actor_inference_platform=np.asarray(actor_cpu_device.platform),
        actor_inference_device_id=np.asarray(int(actor_cpu_device.id), dtype=np.int32),
        actor_inference_device_kind=np.asarray(actor_cpu_device.device_kind),
        outgoing_velocity_semantics=np.asarray("post_control_step_after_all_physics_substeps"),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
        kinematic_sample_timing=np.asarray("post_control_step"),
        control_dt_s=np.asarray(
            env.control_substeps * float(env.model.opt.timestep),
            dtype=np.float32,
        ),
        right_arm_body_names=np.asarray(RIGHT_ARM_AUDIT_BODY_NAMES),
    )
    return {
        "steps": len(trace["time_to_intercept_s"]),
        "hit": bool(any(trace["hit_event"])),
        "event_rebound": bool(any(trace["event_rebound"])),
        "body_fall": bool(any(trace["body_fall"])),
        "termination_reason": final_info.get("termination_reason"),
        "feed_fingerprint": feed_sample_fingerprint(feed),
        "swing_phase_advance_s": float(env.swing_phase_advance_s),
        "actor_inference_semantics": CPU_ACTOR_INFERENCE_SEMANTICS,
        "actor_inference_platform": str(actor_cpu_device.platform),
        "actor_inference_device_id": int(actor_cpu_device.id),
        "actor_inference_device_kind": str(actor_cpu_device.device_kind),
        "trace_path": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **video_report,
    }


def _summarize_cpu_quality_trace(
    path: str | Path,
    *,
    player_half_sign: int,
    min_outgoing_z_m_s: float,
    min_forward_m_s: float,
    max_stringbed_height_deficit_m: float,
    max_hand_height_deficit_m: float,
    min_predicted_clearance_m: float | None = None,
    min_return_direction_signed_score: float | None = None,
    require_real_net_cross: bool = False,
    real_net_cross_authoritative: bool = False,
    max_pre_event_velocity_delta_m_s: float = MAX_PRE_EVENT_VELOCITY_DELTA_M_S,
    max_event_settled_velocity_delta_m_s: float = MAX_EVENT_SETTLED_VELOCITY_DELTA_M_S,
) -> dict[str, Any]:
    """Measure the independent CPU return gate from a saved audit trace."""

    if player_half_sign not in {-1, 1}:
        raise ValueError("player_half_sign must be -1 or 1")
    if (
        not math.isfinite(max_pre_event_velocity_delta_m_s)
        or max_pre_event_velocity_delta_m_s <= 0.0
    ):
        raise ValueError("pre-event velocity threshold must be finite and positive")
    if (
        not math.isfinite(max_event_settled_velocity_delta_m_s)
        or max_event_settled_velocity_delta_m_s <= 0.0
    ):
        raise ValueError("event-to-settled velocity threshold must be finite and positive")
    trace_path = Path(path).expanduser().resolve()
    with np.load(trace_path, allow_pickle=False) as payload:
        required_cpu_actor_fields = {
            "trace_schema_version",
            "actor_inference_semantics",
            "actor_inference_platform",
        }
        missing_cpu_actor_fields = sorted(required_cpu_actor_fields - set(payload.files))
        if missing_cpu_actor_fields:
            raise ValueError(
                "CPU audit trace lacks explicit CPU actor provenance: " + ", ".join(missing_cpu_actor_fields)
            )
        trace_schema_version = str(np.asarray(payload["trace_schema_version"]).item())
        actor_inference_semantics = str(np.asarray(payload["actor_inference_semantics"]).item())
        actor_inference_platform = str(np.asarray(payload["actor_inference_platform"]).item())
        if trace_schema_version != CPU_AUDIT_TRACE_SCHEMA:
            raise ValueError("CPU audit trace lacks the explicit CPU actor schema")
        if actor_inference_semantics != CPU_ACTOR_INFERENCE_SEMANTICS:
            raise ValueError("CPU audit trace has incompatible actor inference semantics")
        if actor_inference_platform != "cpu":
            raise ValueError("CPU audit trace actor did not execute on CPU")
        outgoing_velocity_semantics = str(np.asarray(payload["outgoing_velocity_semantics"]).item())
        if outgoing_velocity_semantics != ("post_control_step_after_all_physics_substeps"):
            raise ValueError("CPU audit trace has incompatible outgoing-velocity semantics")
        event_rebound_contact_semantics = str(np.asarray(payload["event_rebound_contact_semantics"]).item())
        if event_rebound_contact_semantics != (
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ):
            raise ValueError("CPU audit trace permits a double-applied stringbed impact")
        event = np.asarray(payload["event_rebound"], dtype=bool)
        fall = np.asarray(payload["body_fall"], dtype=bool)
        velocity = np.asarray(payload["shuttle_velocity"], dtype=np.float64)
        event_impulse_velocity_after = np.asarray(
            payload["event_impulse_velocity_after_world_m_s"],
            dtype=np.float64,
        )
        if "event_shuttle_velocity_before_world_m_s" not in payload.files:
            raise ValueError("CPU audit trace lacks pre-event shuttle velocity")
        event_shuttle_velocity_before = np.asarray(
            payload["event_shuttle_velocity_before_world_m_s"],
            dtype=np.float64,
        )
        window = np.asarray(payload["correction_window"], dtype=np.float64)
        tti = np.asarray(payload["time_to_intercept_s"], dtype=np.float64)
        stringbed_height = np.asarray(payload["stringbed_position"], dtype=np.float64)[:, 2]
        hand_height = np.asarray(payload["right_arm_body_position_xyz_m"], dtype=np.float64)[:, -1, 2]
        predicted_clearance = (
            np.asarray(payload["predicted_net_clearance_m"], dtype=np.float64)
            if "predicted_net_clearance_m" in payload.files
            else None
        )
        return_direction = (
            np.asarray(payload["return_direction_signed_score"], dtype=np.float64)
            if "return_direction_signed_score" in payload.files
            else None
        )
        valid_cross_event = (
            np.asarray(payload["valid_net_cross_event"], dtype=bool)
            if "valid_net_cross_event" in payload.files
            else None
        )
    event_indices = np.flatnonzero(event)
    event_index = int(event_indices[0]) if event_indices.size else None
    active = window > 0.05
    outgoing_velocity = None if event_index is None else velocity[event_index]
    event_impulse_outgoing_velocity = None if event_index is None else event_impulse_velocity_after[event_index]
    pre_event_velocity = (
        None
        if event_index is None
        else event_shuttle_velocity_before[event_index]
    )
    previous_settled_velocity = (
        None
        if event_index is None or event_index <= 0
        else velocity[event_index - 1]
    )
    pre_event_velocity_delta = (
        None
        if pre_event_velocity is None or previous_settled_velocity is None
        else float(np.linalg.norm(pre_event_velocity - previous_settled_velocity))
    )
    pre_event_velocity_consistent = bool(
        pre_event_velocity_delta is not None
        and math.isfinite(pre_event_velocity_delta)
        and pre_event_velocity_delta <= float(max_pre_event_velocity_delta_m_s)
    )
    event_settled_velocity_delta = (
        None
        if outgoing_velocity is None or event_impulse_outgoing_velocity is None
        else float(
            np.linalg.norm(
                outgoing_velocity - event_impulse_outgoing_velocity,
            )
        )
    )
    event_settled_velocity_consistent = bool(
        event_settled_velocity_delta is not None
        and math.isfinite(event_settled_velocity_delta)
        and event_settled_velocity_delta
        <= float(max_event_settled_velocity_delta_m_s)
    )
    outgoing_z = None if outgoing_velocity is None else float(outgoing_velocity[2])
    outgoing_forward = None if outgoing_velocity is None else float(-int(player_half_sign) * outgoing_velocity[0])
    event_predicted_clearance = (
        None if event_index is None or predicted_clearance is None else float(predicted_clearance[event_index])
    )
    event_return_direction = (
        None if event_index is None or return_direction is None else float(return_direction[event_index])
    )
    crossed_net = bool(valid_cross_event is not None and valid_cross_event.any())
    stringbed_deficit = (
        None
        if event_index is None or not active.any()
        else float(
            max(
                0.0,
                float(stringbed_height[active].max() - stringbed_height[event_index]),
            )
        )
    )
    hand_deficit = (
        None
        if event_index is None or not active.any()
        else float(max(0.0, float(hand_height[active].max() - hand_height[event_index])))
    )
    high_region = bool(
        stringbed_deficit is not None
        and hand_deficit is not None
        and stringbed_deficit <= float(max_stringbed_height_deficit_m)
        and hand_deficit <= float(max_hand_height_deficit_m)
    )
    if bool(real_net_cross_authoritative) and not bool(require_real_net_cross):
        raise ValueError("authoritative real-cross CPU gate requires a real net cross")
    legal_return_required = (
        min_predicted_clearance_m is not None
        or min_return_direction_signed_score is not None
        or bool(require_real_net_cross)
    )
    if legal_return_required and min_return_direction_signed_score is None:
        raise ValueError("CPU legal-return gate requires a direction threshold")
    if legal_return_required and not bool(real_net_cross_authoritative) and min_predicted_clearance_m is None:
        raise ValueError("CPU legal-return gate requires a clearance threshold")
    legal_return = bool(
        not legal_return_required
        or (
            event_return_direction is not None
            and event_return_direction >= float(min_return_direction_signed_score)
            and (
                bool(real_net_cross_authoritative)
                or (
                    event_predicted_clearance is not None
                    and event_predicted_clearance >= float(min_predicted_clearance_m)
                )
            )
            and (not bool(require_real_net_cross) or crossed_net)
        )
    )
    quality = bool(
        event_index is not None
        and pre_event_velocity_consistent
        and event_settled_velocity_consistent
        and not fall.any()
        and high_region
        and outgoing_z is not None
        and outgoing_z >= float(min_outgoing_z_m_s)
        and outgoing_forward is not None
        and outgoing_forward >= float(min_forward_m_s)
        and legal_return
    )
    return {
        "cpu_quality_passed": quality,
        "actor_inference_semantics": actor_inference_semantics,
        "actor_inference_platform": actor_inference_platform,
        "event_rebound": event_index is not None,
        "body_fall": bool(fall.any()),
        "event_step": event_index,
        "event_tti_s": None if event_index is None else float(tti[event_index]),
        "event_shuttle_velocity_before_xyz_m_s": (
            None if pre_event_velocity is None else pre_event_velocity.tolist()
        ),
        "previous_settled_shuttle_velocity_xyz_m_s": (
            None
            if previous_settled_velocity is None
            else previous_settled_velocity.tolist()
        ),
        "pre_event_velocity_delta_m_s": pre_event_velocity_delta,
        "pre_event_velocity_consistent": pre_event_velocity_consistent,
        "max_pre_event_velocity_delta_m_s": float(
            max_pre_event_velocity_delta_m_s
        ),
        "outgoing_velocity_xyz_m_s": (None if outgoing_velocity is None else outgoing_velocity.tolist()),
        "outgoing_velocity_semantics": outgoing_velocity_semantics,
        "event_rebound_contact_semantics": event_rebound_contact_semantics,
        "event_impulse_velocity_after_xyz_m_s": (
            None if event_impulse_outgoing_velocity is None else event_impulse_outgoing_velocity.tolist()
        ),
        "event_settled_velocity_delta_m_s": event_settled_velocity_delta,
        "event_settled_velocity_consistent": event_settled_velocity_consistent,
        "max_event_settled_velocity_delta_m_s": float(
            max_event_settled_velocity_delta_m_s
        ),
        "outgoing_z_m_s": outgoing_z,
        "outgoing_forward_m_s": outgoing_forward,
        "predicted_net_clearance_m": event_predicted_clearance,
        "return_direction_signed_score": event_return_direction,
        "crossed_net": crossed_net,
        "legal_return": legal_return,
        "real_net_cross_authoritative": bool(real_net_cross_authoritative),
        "high_region_contact": high_region,
        "stringbed_height_deficit_m": stringbed_deficit,
        "hand_height_deficit_m": hand_deficit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feed-fingerprint", default=None)
    parser.add_argument(
        "--swing-phase-advance-s",
        type=float,
        default=None,
        help=(
            "optional frozen base-swing timing override; when omitted, use "
            "stage3_direct.swing_phase_advance_s from the spec"
        ),
    )
    parser.add_argument("--population", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.08)
    parser.add_argument("--initial-std", type=float, default=0.45)
    parser.add_argument("--min-std", type=float, default=0.04)
    parser.add_argument(
        "--search-frontier-copies",
        type=int,
        default=1,
        help=(
            "number of candidate groups reserved every iteration for the "
            "current unqualified search frontier; copies occupy independent "
            "batch lanes and make promising returns influence robustification"
        ),
    )
    parser.add_argument(
        "--verify-search-frontier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compare every new unqualified frontier challenger against the "
            "incumbent in equal halves of an additional full backend batch"
        ),
    )
    parser.add_argument(
        "--coordinate-probe-radius",
        type=float,
        default=0.0,
        help=(
            "on iteration 1, reserve positive/negative axis probes around the "
            "initial mean for every trainable parameter; zero disables the probe"
        ),
    )
    parser.add_argument(
        "--initial-candidate",
        default=None,
        help=(
            "sealed best_teacher.json or explicitly unqualified search_seed.json "
            "from a compatible CEM run; starts a fresh local search around those "
            "parameters without reusing optimizer state"
        ),
    )
    parser.add_argument(
        "--allow-unqualified-physical-scale-rebind",
        action="store_true",
        help=(
            "allow only an explicitly unqualified search seed to initialize "
            "a target run with different physical correction scales; the new "
            "run remains unqualified until all target-contract gates pass"
        ),
    )
    parser.add_argument(
        "--allow-unqualified-time-knot-rebind",
        action="store_true",
        help=(
            "allow only an explicitly unqualified anatomical search seed to "
            "expand onto a denser nested time-knot grid; initialization fails "
            "closed unless the complete physical correction remains close"
        ),
    )
    parser.add_argument(
        "--additional-candidate",
        action="append",
        default=[],
        help=(
            "repeatable compatible unqualified seed/teacher candidate to place "
            "verbatim in iteration 1 after the mean/frontier-copy anchors"
        ),
    )
    parser.add_argument(
        "--parameterization",
        choices=("anatomical_synergies", "muscle_knots"),
        default="anatomical_synergies",
    )
    parser.add_argument(
        "--trainable-synergies",
        nargs="+",
        default=None,
        help=(
            "optional anatomical synergy names to optimize; all unlisted "
            "synergies remain exactly equal to the initial candidate"
        ),
    )
    parser.add_argument(
        "--trainable-knot-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "optional zero-based temporal knot indices to optimize; all other "
            "knots remain bitwise equal to the initial candidate"
        ),
    )
    parser.add_argument("--time-knots", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--min-replica-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--verification-repeats",
        type=int,
        default=2,
        help="independent final replay batches required before sealing a teacher",
    )
    parser.add_argument(
        "--require-cpu-quality-for-best",
        action="store_true",
        help=(
            "allow a backend candidate to become the global teacher only after "
            "an independent CPU upward-forward, high-region, no-fall replay"
        ),
    )
    parser.add_argument("--cpu-min-outgoing-z-m-s", type=float, default=0.5)
    parser.add_argument("--cpu-min-forward-m-s", type=float, default=2.0)
    parser.add_argument(
        "--max-pre-event-velocity-delta-m-s",
        type=float,
        default=MAX_PRE_EVENT_VELOCITY_DELTA_M_S,
        help=(
            "maximum allowed norm between the prior settled shuttle velocity "
            "and the velocity recorded immediately before the custom stringbed "
            "event; larger changes indicate an earlier native frame collision"
        ),
    )
    parser.add_argument(
        "--max-event-settled-velocity-delta-m-s",
        type=float,
        default=MAX_EVENT_SETTLED_VELOCITY_DELTA_M_S,
        help=(
            "maximum allowed norm between the instantaneous event rebound "
            "and the settled post-control shuttle velocity; larger changes "
            "indicate a duplicate native collision"
        ),
    )
    parser.add_argument(
        "--cpu-promotion-audit-limit",
        type=int,
        default=4,
        help="maximum ranked backend improvements to CPU-audit per CEM iteration",
    )
    parser.add_argument(
        "--cpu-audit-coordinate-probes",
        action="store_true",
        help=(
            "on the coordinate-probe iteration, expand the CPU audit budget to "
            "cover every axis probe in backend rank order unless a passing "
            "teacher is found first"
        ),
    )
    parser.add_argument(
        "--cpu-guide-unqualified-mean",
        action="store_true",
        help=(
            "when no teacher has passed, move the next CEM mean to the best "
            "physically consistent CPU-audited candidate; this changes only "
            "search guidance and never bypasses backend or CPU promotion gates"
        ),
    )
    parser.add_argument("--authority-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--min-outgoing-z-m-s",
        type=float,
        default=0.5,
        help=(
            "replica success threshold used during search; raise above the "
            "deployment gate to search for vertical-velocity margin"
        ),
    )
    parser.add_argument(
        "--min-forward-m-s",
        type=float,
        default=2.0,
        help=(
            "replica success threshold used during search; raise above the "
            "deployment gate to search for forward-velocity margin"
        ),
    )
    parser.add_argument(
        "--require-legal-return-for-teacher",
        action="store_true",
        help=(
            "require the robust teacher to satisfy drag-aware clearance and "
            "return-direction gates; also enables their continuous CEM score"
        ),
    )
    parser.add_argument(
        "--require-real-net-cross-for-teacher",
        action="store_true",
        help=(
            "in addition to drag-aware clearance, require a real valid net "
            "cross in the search backend and independent CPU audit"
        ),
    )
    parser.add_argument(
        "--real-net-cross-authoritative-for-teacher",
        action="store_true",
        help=(
            "treat the simulator's real valid net crossing, which already "
            "enforces the configured net-height clearance, as the teacher "
            "clearance gate; retain conservative drag projection for ranking "
            "and diagnostics"
        ),
    )
    parser.add_argument("--min-predicted-clearance-m", type=float, default=0.20)
    parser.add_argument(
        "--min-return-direction-signed-score",
        type=float,
        default=0.65,
    )
    parser.add_argument("--max-stringbed-height-deficit-m", type=float, default=0.10)
    parser.add_argument("--max-hand-height-deficit-m", type=float, default=0.10)
    parser.add_argument("--max-episode-steps", type=int, default=420)
    parser.add_argument("--impl", choices=("warp", "jax"), default="warp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="musclemimic-stage3-hit")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.population < 32 or args.iterations <= 0:
        parser.error("population must be >=32 and iterations must be positive")
    if not 1 <= int(args.search_frontier_copies) < int(args.population):
        parser.error("search-frontier-copies must lie in [1, population)")
    if args.verify_search_frontier and (int(args.population) * int(args.replicas)) % 2 != 0:
        parser.error("verified search-frontier comparison requires an even simulation batch")
    if not 0.0 < args.elite_fraction < 0.5:
        parser.error("elite-fraction must lie in (0, 0.5)")
    if not 0.0 < args.min_std <= args.initial_std:
        parser.error("require 0 < min-std <= initial-std")
    if (
        not math.isfinite(args.coordinate_probe_radius)
        or args.coordinate_probe_radius < 0.0
        or args.coordinate_probe_radius > 3.0
    ):
        parser.error("coordinate-probe-radius must be finite and lie in [0, 3]")
    if args.time_knots < 2 or args.time_knots > 12:
        parser.error("time-knots must lie in [2, 12]")
    if args.replicas < 1 or args.replicas > 8:
        parser.error("replicas must lie in [1, 8]")
    if args.verification_repeats < 1 or args.verification_repeats > 16:
        parser.error("verification-repeats must lie in [1, 16]")
    if args.require_cpu_quality_for_best and args.initial_candidate is None:
        parser.error("--require-cpu-quality-for-best requires --initial-candidate")
    if args.allow_unqualified_physical_scale_rebind and args.initial_candidate is None:
        parser.error("--allow-unqualified-physical-scale-rebind requires --initial-candidate")
    if args.allow_unqualified_time_knot_rebind and args.initial_candidate is None:
        parser.error("--allow-unqualified-time-knot-rebind requires --initial-candidate")
    if args.allow_unqualified_time_knot_rebind and args.parameterization != "anatomical_synergies":
        parser.error("--allow-unqualified-time-knot-rebind requires anatomical_synergies")
    if not 1 <= args.cpu_promotion_audit_limit <= 128:
        parser.error("--cpu-promotion-audit-limit must lie in [1, 128]")
    if args.cpu_audit_coordinate_probes and (
        not args.require_cpu_quality_for_best or float(args.coordinate_probe_radius) <= 0.0
    ):
        parser.error(
            "--cpu-audit-coordinate-probes requires a positive coordinate probe and --require-cpu-quality-for-best"
        )
    if args.cpu_guide_unqualified_mean and not args.require_cpu_quality_for_best:
        parser.error(
            "--cpu-guide-unqualified-mean requires --require-cpu-quality-for-best"
        )
    if args.additional_candidate and float(args.coordinate_probe_radius) > 0.0:
        parser.error("--additional-candidate cannot be combined with a coordinate probe")
    if not 0.0 < args.min_replica_fraction <= 1.0:
        parser.error("min-replica-fraction must lie in (0, 1]")
    if not math.isfinite(args.authority_multiplier) or args.authority_multiplier <= 0.0:
        parser.error("authority-multiplier must be finite and positive")
    if args.swing_phase_advance_s is not None and (
        not math.isfinite(args.swing_phase_advance_s) or not 0.0 <= float(args.swing_phase_advance_s) <= 1.0
    ):
        parser.error("swing-phase-advance-s must be finite and lie in [0, 1]")
    if (
        not math.isfinite(args.min_outgoing_z_m_s)
        or args.min_outgoing_z_m_s <= 0.0
        or not math.isfinite(args.min_forward_m_s)
        or args.min_forward_m_s <= 0.0
    ):
        parser.error("return-quality search thresholds must be finite and positive")
    if args.require_real_net_cross_for_teacher and not args.require_legal_return_for_teacher:
        parser.error("--require-real-net-cross-for-teacher requires --require-legal-return-for-teacher")
    if args.real_net_cross_authoritative_for_teacher and not args.require_real_net_cross_for_teacher:
        parser.error("--real-net-cross-authoritative-for-teacher requires --require-real-net-cross-for-teacher")
    if (
        not math.isfinite(args.min_predicted_clearance_m)
        or not -5.0 <= float(args.min_predicted_clearance_m) <= 5.0
        or not math.isfinite(args.min_return_direction_signed_score)
        or not -1.0 <= float(args.min_return_direction_signed_score) <= 1.0
    ):
        parser.error("legal-return thresholds require clearance in [-5, 5] and direction in [-1, 1]")
    if (
        not math.isfinite(args.cpu_min_outgoing_z_m_s)
        or args.cpu_min_outgoing_z_m_s <= 0.0
        or not math.isfinite(args.cpu_min_forward_m_s)
        or args.cpu_min_forward_m_s <= 0.0
    ):
        parser.error("CPU promotion thresholds must be finite and positive")
    if (
        not math.isfinite(args.max_pre_event_velocity_delta_m_s)
        or not 0.0 < float(args.max_pre_event_velocity_delta_m_s) <= 3.0
    ):
        parser.error(
            "max-pre-event-velocity-delta-m-s must be finite and lie in (0, 3]"
        )
    if (
        not math.isfinite(args.max_event_settled_velocity_delta_m_s)
        or not 0.0 < float(args.max_event_settled_velocity_delta_m_s) <= 3.0
    ):
        parser.error(
            "max-event-settled-velocity-delta-m-s must be finite and lie in (0, 3]"
        )
    if (
        not math.isfinite(args.max_stringbed_height_deficit_m)
        or not 0.0 < args.max_stringbed_height_deficit_m <= 0.30
        or not math.isfinite(args.max_hand_height_deficit_m)
        or not 0.0 < args.max_hand_height_deficit_m <= 0.30
    ):
        parser.error("high-region height deficits must be finite and lie in (0, 0.30]")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = load_incoming_hit_spec(args.spec)
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    configured_seeds = tuple(paths.stage3_direct.get("seed_feed_fingerprints", ()))
    requested_fingerprint = args.feed_fingerprint or (configured_seeds[0] if configured_seeds else None)
    if requested_fingerprint is None:
        raise ValueError("provide --feed-fingerprint or configure stage3_direct.seed_feed_fingerprints")
    by_fingerprint = {feed_sample_fingerprint(sample): sample for sample in feed_artifact.bank}
    if requested_fingerprint not in by_fingerprint:
        raise ValueError("requested feed fingerprint is absent from the training bank")
    feed = by_fingerprint[requested_fingerprint]
    configured_swing_phase_advance_s = float(paths.stage3_direct.get("swing_phase_advance_s", 0.0))
    effective_swing_phase_advance_s = (
        configured_swing_phase_advance_s if args.swing_phase_advance_s is None else float(args.swing_phase_advance_s)
    )
    if not math.isfinite(effective_swing_phase_advance_s) or not 0.0 <= effective_swing_phase_advance_s <= 1.0:
        raise ValueError("effective swing phase advance must be finite and lie in [0, 1]")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    restored, source_metadata = _source_actor(checkpoint)
    if "policy_delta" not in restored.agent:
        raise ValueError("CEM requires the inherited v24c Phase-A policy_delta")
    if "policy_refinement_delta" in restored.agent:
        raise ValueError("CEM must start before the coupled refinement adapter")
    base_policy = source_metadata.get("base_policy_artifact")
    if not base_policy:
        raise ValueError("source checkpoint has no frozen base policy artifact")
    base_policy = Path(base_policy).expanduser().resolve()

    constraints = _return_constraints(paths)
    if (
        args.require_legal_return_for_teacher
        and constraints["clearance_prediction_mode"] != "quadratic_drag_conservative_v1"
    ):
        raise ValueError("legal-return CEM requires quadratic_drag_conservative_v1")
    residual_scale = float(paths.stage3_direct.get("residual_scale", 0.25))
    if _residual_scale_overrides(paths) or _residual_scale_schedule(paths):
        raise ValueError("CEM physical correction requires constant inherited residual authority")
    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=[feed],
        control_substeps=paths.control_substeps,
        max_episode_steps=int(args.max_episode_steps),
        reward_weights=paths.reward_weights,
        return_net_x_m=constraints["net_x_m"],
        return_net_height_m=constraints["net_height_m"],
        min_return_net_clearance_m=constraints["min_clearance_m"],
        desired_return_up_component=constraints["desired_up_component"],
        ballistic_return_score_softness_m=constraints["ballistic_score_softness_m"],
        clearance_prediction_mode=constraints["clearance_prediction_mode"],
        shuttle_proximity_softness_m=constraints["shuttle_proximity_softness_m"],
        timed_intercept_softness_m=constraints["timed_intercept_softness_m"],
        direction_distance_softness_m=constraints["direction_distance_softness_m"],
        contact_guidance_reward_mode=constraints["contact_guidance_reward_mode"],
        contact_guidance_discount=constraints["contact_guidance_discount"],
        racket_velocity_direction_fraction=constraints["racket_velocity_direction_fraction"],
        direction_reward_mode=constraints["direction_reward_mode"],
        clearance_reward_mode=constraints["clearance_reward_mode"],
        hit_event_mode=constraints["hit_event_mode"],
        racket_guidance_mode=constraints["racket_guidance_mode"],
        inverse_target_speed_m_s=constraints["inverse_target_speed_m_s"],
        inverse_velocity_softness_m_s=constraints["inverse_velocity_softness_m_s"],
        impl=args.impl,
        base_policy_artifact=base_policy,
        residual_scale=residual_scale,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.76)),
        swing_phase_advance_s=effective_swing_phase_advance_s,
    )
    contract = _policy_update_contract(paths, env.model)
    if contract["mode"] != "selected_physical_correction":
        raise ValueError("CEM spec must use policy_update_mode=selected_physical_correction")
    selected_indices = tuple(int(value) for value in contract["trainable_action_indices"])
    selected_names = tuple(str(value) for value in contract["trainable_actuator_names"])
    synergy_basis_np: np.ndarray | None = None
    synergy_names: tuple[str, ...] = ()
    if args.parameterization == "anatomical_synergies":
        synergy_basis_np, synergy_names = _anatomical_synergy_basis(selected_names)
    synergy_basis_sha256 = (
        None if synergy_basis_np is None else hashlib.sha256(synergy_basis_np.tobytes(order="C")).hexdigest()
    )
    physical_scales_np = np.asarray(contract["correction_physical_scales"], dtype=np.float32) * float(
        args.authority_multiplier
    )
    window = dict(contract["correction_window"])
    residual_scale_vector = jnp.full((env.action_size,), residual_scale, dtype=jnp.float32)

    rollout = _make_rollout(
        env,
        num_envs=int(args.population) * int(args.replicas),
        actor=restored.agent,
        obs_mean=restored.obs_rms.mean,
        obs_var=restored.obs_rms.var,
        selected_indices=selected_indices,
        physical_scales=jnp.asarray(physical_scales_np),
        residual_scale_vector=residual_scale_vector,
        open_s=float(window["time_to_intercept_open_s"]),
        close_s=float(window["time_to_intercept_close_s"]),
        smoothing_s=float(window["smoothing_s"]),
        time_knots=int(args.time_knots),
        synergy_basis=(None if synergy_basis_np is None else jnp.asarray(synergy_basis_np)),
        max_stringbed_height_deficit_m=float(args.max_stringbed_height_deficit_m),
        max_hand_height_deficit_m=float(args.max_hand_height_deficit_m),
    )

    latent_size = len(selected_indices) if synergy_basis_np is None else int(synergy_basis_np.shape[0])
    dimension = int(args.time_knots) * latent_size
    (
        trainable_parameter_mask,
        trainable_synergy_names,
        trainable_knot_indices,
    ) = _trainable_parameter_mask(
        parameterization=str(args.parameterization),
        synergy_names=(
            synergy_names if synergy_basis_np is not None else tuple(str(index) for index in range(latent_size))
        ),
        time_knots=int(args.time_knots),
        requested_synergies=(
            None if args.trainable_synergies is None else tuple(str(name) for name in args.trainable_synergies)
        ),
        requested_knot_indices=(
            None if args.trainable_knot_indices is None else tuple(int(index) for index in args.trainable_knot_indices)
        ),
    )
    if trainable_parameter_mask.shape != (dimension,):
        raise RuntimeError("CEM trainable-parameter mask has an incompatible shape")
    coordinate_probe_required_population = (
        0
        if float(args.coordinate_probe_radius) == 0.0
        else 1 + int(args.search_frontier_copies) + 2 * int(trainable_parameter_mask.sum())
    )
    if coordinate_probe_required_population > int(args.population):
        parser.error(
            "coordinate probe requires population >= "
            f"{coordinate_probe_required_population} for "
            f"{int(trainable_parameter_mask.sum())} trainable parameters"
        )
    initial_parameters = np.zeros(dimension, dtype=np.float32)
    initial_candidate_binding: dict[str, Any] | None = None
    source_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    candidate_compatibility_contract = {
        # Compare only fields that determine how a saved latent vector maps
        # into this rollout.  Optimizer/search-policy settings are not part of
        # the physical candidate ABI.
        "spec": str(paths.spec_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "feed_fingerprint": requested_fingerprint,
        "swing_phase_advance_s": effective_swing_phase_advance_s,
        "parameterization": str(args.parameterization),
        "time_knots": int(args.time_knots),
        "latent_size": latent_size,
        "parameter_count": dimension,
        "selected_actuator_names": list(selected_names),
        "synergy_names": list(synergy_names),
        # The latent width and names alone do not define how a saved vector
        # maps into the 32 physical corrections.  Seal the actual basis so a
        # future cleanup of redundant/collinear synergies cannot silently
        # reinterpret an old search seed.
        "synergy_basis_sha256": synergy_basis_sha256,
        "authority_multiplier": float(args.authority_multiplier),
        "physical_scales": physical_scales_np.tolist(),
        "correction_window": window,
        "max_episode_steps": int(args.max_episode_steps),
        "mjx_impl": str(args.impl),
    }
    if args.initial_candidate is not None:
        initial_parameters, initial_candidate_binding = _load_initial_candidate(
            args.initial_candidate,
            dimension=dimension,
            expected_source_contract=candidate_compatibility_contract,
            allow_unqualified_physical_scale_rebind=bool(args.allow_unqualified_physical_scale_rebind),
            allow_unqualified_time_knot_rebind=bool(args.allow_unqualified_time_knot_rebind),
            synergy_basis=synergy_basis_np,
        )
    reserved_anchor_count = 1 + int(args.search_frontier_copies)
    if len(args.additional_candidate) > int(args.population) - reserved_anchor_count:
        parser.error("additional candidates must fit after the mean/frontier-copy anchors")
    additional_candidate_parameters: list[np.ndarray] = []
    additional_candidate_bindings: list[dict[str, Any]] = []
    for additional_path in args.additional_candidate:
        parameters, binding = _load_initial_candidate(
            additional_path,
            dimension=dimension,
            expected_source_contract=candidate_compatibility_contract,
            allow_unqualified_physical_scale_rebind=bool(args.allow_unqualified_physical_scale_rebind),
            allow_unqualified_time_knot_rebind=bool(args.allow_unqualified_time_knot_rebind),
            synergy_basis=synergy_basis_np,
        )
        if not np.array_equal(
            parameters[~trainable_parameter_mask],
            initial_parameters[~trainable_parameter_mask],
        ):
            raise ValueError("additional CEM candidate changes a frozen parameter relative to the initial candidate")
        additional_candidate_parameters.append(parameters)
        additional_candidate_bindings.append(binding)

    search_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "spec": str(paths.spec_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "scene_sha256": hashlib.sha256(paths.scene_xml.read_bytes()).hexdigest(),
        "scene_collision_semantics": (
            "cork_only_native_racket_frame_contact_with_ground_support_excluded_v1"
        ),
        "feed_fingerprint": requested_fingerprint,
        "configured_swing_phase_advance_s": configured_swing_phase_advance_s,
        "swing_phase_advance_s": effective_swing_phase_advance_s,
        "swing_phase_timing_semantics": (
            "frozen_base_swing_phase_advance_applied_identically_to_search_backend_and_cpu_replays"
        ),
        "population": int(args.population),
        "replicas_per_candidate": int(args.replicas),
        "simulation_envs": int(args.population) * int(args.replicas),
        "min_replica_fraction": float(args.min_replica_fraction),
        "required_replica_count": _required_replica_count(
            int(args.replicas),
            float(args.min_replica_fraction),
        ),
        "search_frontier_copies": int(args.search_frontier_copies),
        "search_frontier_candidate_indices": list(range(1, 1 + int(args.search_frontier_copies))),
        "search_frontier_replay_semantics": (
            "same_unqualified_frontier_replayed_across_independent_candidate_groups_every_iteration"
        ),
        "search_frontier_verification": {
            "enabled": bool(args.verify_search_frontier),
            "replicas_per_side": (int(args.population) * int(args.replicas) // 2 if args.verify_search_frontier else 0),
            "semantics": (
                "incumbent_and_challenger_share_equal_halves_of_one_full_backend_batch_before_frontier_replacement"
            ),
        },
        "allow_unqualified_physical_scale_rebind": bool(args.allow_unqualified_physical_scale_rebind),
        "allow_unqualified_time_knot_rebind": bool(args.allow_unqualified_time_knot_rebind),
        "search_replica_context_semantics": (
            "candidate_replicas_stratified_across_deterministic_warp_batch_lanes"
            if str(args.impl) == "warp"
            else ("candidate_replicas_stratified_across_deterministic_standard_mjx_jax_batch_lanes")
        ),
        "search_replica_lane_offsets": (
            (np.arange(int(args.replicas), dtype=np.int64) * int(args.population)) // int(args.replicas)
        ).tolist(),
        "verification_repeats": int(args.verification_repeats),
        "verification_replica_count": int(args.replicas) * int(args.verification_repeats),
        "verification_required_count": _required_replica_count(
            int(args.replicas) * int(args.verification_repeats),
            float(args.min_replica_fraction),
        ),
        "verification_context_semantics": (
            "same_candidate_relocated_across_deterministic_warp_batch_lanes"
            if str(args.impl) == "warp"
            else ("same_candidate_relocated_across_deterministic_standard_mjx_jax_batch_lanes")
        ),
        "cpu_best_promotion_gate": {
            "enabled": bool(args.require_cpu_quality_for_best),
            "min_outgoing_z_m_s": float(args.cpu_min_outgoing_z_m_s),
            "min_forward_m_s": float(args.cpu_min_forward_m_s),
            "require_legal_return": bool(args.require_legal_return_for_teacher),
            "require_real_net_cross": bool(args.require_real_net_cross_for_teacher),
            "real_net_cross_authoritative": bool(args.real_net_cross_authoritative_for_teacher),
            "min_predicted_clearance_m": float(args.min_predicted_clearance_m),
            "min_return_direction_signed_score": float(args.min_return_direction_signed_score),
            "max_stringbed_height_deficit_m": float(args.max_stringbed_height_deficit_m),
            "max_hand_height_deficit_m": float(args.max_hand_height_deficit_m),
            "max_pre_event_velocity_delta_m_s": float(
                args.max_pre_event_velocity_delta_m_s
            ),
            "max_event_settled_velocity_delta_m_s": float(
                args.max_event_settled_velocity_delta_m_s
            ),
            "max_ranked_candidate_audits_per_iteration": int(args.cpu_promotion_audit_limit),
            "semantics": ("explicit_jax_cpu_actor_and_cpu_mujoco_quality_before_global_best_promotion"),
            "actor_inference_semantics": CPU_ACTOR_INFERENCE_SEMANTICS,
            "clearance_gate_semantics": (
                "simulated_valid_cross_at_configured_clearance"
                if args.real_net_cross_authoritative_for_teacher
                else "conservative_drag_projection"
            ),
            "event_settled_velocity_semantics": (
                "reject_duplicate_native_collision_after_custom_stringbed_event"
            ),
            "pre_event_velocity_semantics": (
                "reject_native_racket_collision_before_custom_stringbed_event"
            ),
        },
        "cpu_unqualified_mean_guidance": {
            "enabled": bool(args.cpu_guide_unqualified_mean),
            "semantics": (
                "best_cpu_consistent_frontier_remains_next_cem_mean_until_strictly_improved_without_teacher_claim"
            ),
            "physical_validity_precedes_continuous_progress": True,
            "continuous_progress_semantics": (
                "maximin_active_teacher_constraint_margin_with_predicted_clearance_only_until_authoritative_real_cross"
            ),
            "exploration_std_floor_fraction_of_initial": 0.5,
        },
        "iterations": int(args.iterations),
        "elite_fraction": float(args.elite_fraction),
        "initial_std": float(args.initial_std),
        "min_std": float(args.min_std),
        "parameterization": str(args.parameterization),
        "time_knots": int(args.time_knots),
        "latent_size": latent_size,
        "parameter_count": dimension,
        "selected_actuator_names": list(selected_names),
        "synergy_names": list(synergy_names),
        "trainable_synergy_names": list(trainable_synergy_names),
        "frozen_synergy_names": [name for name in synergy_names if name not in trainable_synergy_names],
        "trainable_knot_indices": list(trainable_knot_indices),
        "frozen_knot_indices": [index for index in range(int(args.time_knots)) if index not in trainable_knot_indices],
        "trainable_parameter_count": int(trainable_parameter_mask.sum()),
        "frozen_parameter_count": int(dimension - trainable_parameter_mask.sum()),
        "coordinate_probe": {
            "enabled": bool(float(args.coordinate_probe_radius) > 0.0),
            "radius": float(args.coordinate_probe_radius),
            "iteration": 1 if float(args.coordinate_probe_radius) > 0.0 else None,
            "required_population": coordinate_probe_required_population,
            "semantics": "positive_and_negative_single_parameter_axis_probes",
            "audit_all_probes_until_first_passing_teacher": bool(args.cpu_audit_coordinate_probes),
        },
        "synergy_basis_sha256": synergy_basis_sha256,
        "authority_multiplier": float(args.authority_multiplier),
        "high_region_contact": {
            "max_stringbed_height_deficit_m": float(args.max_stringbed_height_deficit_m),
            "max_hand_height_deficit_m": float(args.max_hand_height_deficit_m),
            "semantics": "soft_window_teacher_gate_not_exact_apex",
        },
        "return_quality_search_margin": {
            "min_outgoing_z_m_s": float(args.min_outgoing_z_m_s),
            "min_forward_m_s": float(args.min_forward_m_s),
            "require_legal_return": bool(args.require_legal_return_for_teacher),
            "require_real_net_cross": bool(args.require_real_net_cross_for_teacher),
            "real_net_cross_authoritative": bool(args.real_net_cross_authoritative_for_teacher),
            "min_predicted_clearance_m": float(args.min_predicted_clearance_m),
            "min_return_direction_signed_score": float(args.min_return_direction_signed_score),
            "max_pre_event_velocity_delta_m_s": float(
                args.max_pre_event_velocity_delta_m_s
            ),
            "max_event_settled_velocity_delta_m_s": float(
                args.max_event_settled_velocity_delta_m_s
            ),
            "clearance_prediction_mode": str(constraints["clearance_prediction_mode"]),
            "semantics": (
                "same_replica_simulated_valid_cross_direction_gate_with_drag_ranking"
                if args.real_net_cross_authoritative_for_teacher
                else (
                    "same_replica_real_cross_drag_clearance_direction_gate"
                    if args.require_real_net_cross_for_teacher
                    else (
                        "same_replica_drag_clearance_direction_gate"
                        if args.require_legal_return_for_teacher
                        else "same_replica_training_backend_margin_gate"
                    )
                )
            ),
        },
        "outgoing_velocity_semantics": ("post_control_step_after_all_physics_substeps"),
        "event_impulse_velocity_semantics": ("instantaneous_event_restitution_before_remaining_physics_substeps"),
        "event_rebound_contact_semantics": ("single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"),
        "pre_event_velocity_consistency": {
            "max_delta_m_s": float(args.max_pre_event_velocity_delta_m_s),
            "semantics": (
                "event_input_velocity_must_remain_close_to_prior_settled_control_state"
            ),
        },
        "event_settled_velocity_consistency": {
            "max_delta_m_s": float(args.max_event_settled_velocity_delta_m_s),
            "semantics": (
                "post_control_velocity_must_remain_close_to_event_impulse_after_aero_and_gravity"
            ),
        },
        "badminton_physics_mjx_sha256": hashlib.sha256(
            (REPO_ROOT / "environment/overall_environment/src/badminton_physics_mjx.py").read_bytes()
        ).hexdigest(),
        "search_objective": {
            "semantics": (
                "proximity_then_joint_return_quality_then_event_weighted_drag_progress_then_robust_legal_return"
                if args.require_legal_return_for_teacher
                else "proximity_then_joint_return_quality_robustification_then_rebound"
            ),
            "stages": [
                "soft_high_region_proximity",
                "replica_robust_stringbed_contact",
                "single_event_pre_velocity_consistency",
                "single_event_settled_velocity_consistency",
                "same_replica_upward_forward_return_quality",
                *(
                    [
                        "event_weighted_continuous_drag_clearance_direction_progress",
                        "replica_robust_event_rebound",
                        "continuous_drag_clearance_direction_progress",
                        "robust_real_valid_net_cross",
                    ]
                    if args.require_real_net_cross_for_teacher
                    else (
                        ["continuous_drag_clearance_direction_progress"]
                        if args.require_legal_return_for_teacher
                        else []
                    )
                ),
                *([] if args.require_legal_return_for_teacher else ["replica_robust_event_rebound"]),
            ],
            "pre_contact_primary": "contact_acquisition_cost_m",
            "soft_high_region_excess_weight": SOFT_HIGH_REGION_EXCESS_WEIGHT,
            "inverse_impact_role": "secondary_after_proximity",
            "timing_semantics": "policy_selects_outcome_optimal_time_within_soft_high_window",
            "search_safe_no_fall_rate": SEARCH_SAFE_NO_FALL_RATE,
            "teacher_no_fall_semantics": "all_replicas_must_remain_upright",
        },
        "physical_scales": physical_scales_np.tolist(),
        "correction_window": window,
        "max_episode_steps": int(args.max_episode_steps),
        "seed": int(args.seed),
        "initial_candidate": initial_candidate_binding,
        "additional_candidates": additional_candidate_bindings,
        "mjx_impl": str(args.impl),
        "optimizer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_update_contract_sha256": contract["contract_sha256"],
    }
    search_contract["contract_sha256"] = _json_hash(search_contract)
    (out_dir / "cem_contract.json").write_text(
        json.dumps(search_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"cem_feed12_a{args.authority_multiplier:g}_s{args.seed}",
            config=search_contract,
            dir=str(out_dir / "wandb"),
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    except Exception as exc:  # pragma: no cover - network credentials are host-specific
        print(f"[cem] W&B unavailable: {exc}", flush=True)

    def audit_cpu_promotion(parameters: np.ndarray) -> dict[str, Any]:
        parameter_array = np.asarray(parameters, dtype=np.float32)
        parameter_sha = hashlib.sha256(parameter_array.tobytes(order="C")).hexdigest()
        audit_root = out_dir / "cpu_candidate_audits"
        trace_path = audit_root / f"candidate_paramsha{parameter_sha[:16]}.npz"
        report_path = trace_path.with_suffix(".json")
        if report_path.is_file():
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if cached.get("schema_version") != "stage3_cem_inline_cpu_quality_gate_v3":
                raise ValueError(
                    "cached CPU candidate audit predates the bidirectional "
                    "event-velocity consistency gate"
                )
            if cached.get("quality_gate") != search_contract["cpu_best_promotion_gate"]:
                raise ValueError("cached CPU candidate audit has a different quality gate")
            if cached.get("candidate_parameter_f32_sha256") != parameter_sha:
                raise ValueError("cached CPU candidate audit has a parameter hash mismatch")
            cached_parameters = np.asarray(
                cached.get("candidate_parameters", ()),
                dtype=np.float32,
            )
            if not np.array_equal(cached_parameters, parameter_array):
                raise ValueError("cached CPU candidate audit detached from its parameters")
            return cached
        if trace_path.exists():
            raise FileExistsError(f"CPU candidate trace exists without its immutable audit report: {trace_path}")
        trace_report = _save_cpu_teacher_trace(
            path=trace_path,
            feed=feed,
            paths=paths,
            actor=restored.agent,
            obs_mean=np.asarray(restored.obs_rms.mean),
            obs_var=np.asarray(restored.obs_rms.var),
            parameters=parameter_array,
            selected_indices=selected_indices,
            physical_scales=physical_scales_np,
            base_policy_artifact=base_policy,
            residual_scale=residual_scale,
            open_s=float(window["time_to_intercept_open_s"]),
            close_s=float(window["time_to_intercept_close_s"]),
            smoothing_s=float(window["smoothing_s"]),
            max_episode_steps=int(args.max_episode_steps),
            time_knots=int(args.time_knots),
            synergy_basis=synergy_basis_np,
            swing_phase_advance_s=effective_swing_phase_advance_s,
        )
        quality = _summarize_cpu_quality_trace(
            trace_path,
            player_half_sign=int(env.player_half_sign),
            min_outgoing_z_m_s=float(args.cpu_min_outgoing_z_m_s),
            min_forward_m_s=float(args.cpu_min_forward_m_s),
            max_stringbed_height_deficit_m=float(args.max_stringbed_height_deficit_m),
            max_hand_height_deficit_m=float(args.max_hand_height_deficit_m),
            min_predicted_clearance_m=(
                float(args.min_predicted_clearance_m) if args.require_legal_return_for_teacher else None
            ),
            min_return_direction_signed_score=(
                float(args.min_return_direction_signed_score) if args.require_legal_return_for_teacher else None
            ),
            require_real_net_cross=bool(args.require_real_net_cross_for_teacher),
            real_net_cross_authoritative=bool(args.real_net_cross_authoritative_for_teacher),
            max_pre_event_velocity_delta_m_s=float(
                args.max_pre_event_velocity_delta_m_s
            ),
            max_event_settled_velocity_delta_m_s=float(
                args.max_event_settled_velocity_delta_m_s
            ),
        )
        report = {
            "schema_version": "stage3_cem_inline_cpu_quality_gate_v3",
            "candidate_parameter_f32_sha256": parameter_sha,
            "candidate_parameters": parameter_array.tolist(),
            "quality_gate": search_contract["cpu_best_promotion_gate"],
            **trace_report,
            **quality,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    mean = initial_parameters.copy()
    std = np.where(
        trainable_parameter_mask,
        float(args.initial_std),
        0.0,
    ).astype(np.float32)
    start_iteration = 0
    global_best_parameters = mean.copy()
    global_best_metrics: dict[str, Any] | None = None
    global_best_context = np.broadcast_to(
        mean,
        (args.population * args.replicas, dimension),
    ).copy()
    global_best_context_index = 0
    global_best_cpu_audit: dict[str, Any] | None = None
    # Teacher promotion and optimizer progress are deliberately separate.
    # A strict search may need many iterations before any candidate is legal;
    # retaining only a promoted global best made candidate slot 1 stay at the
    # initial seed and allowed a useful, still-unqualified frontier to vanish.
    search_frontier_parameters = mean.copy()
    search_frontier_metrics: dict[str, Any] | None = None
    search_frontier_iteration = 0
    search_frontier_candidate_index = 0
    search_frontier_seed_path: str | None = None
    cpu_search_frontier_parameters = mean.copy()
    cpu_search_frontier_audit: dict[str, Any] | None = None
    cpu_search_frontier_iteration = 0
    cpu_search_frontier_candidate_index = 0
    cpu_search_frontier_seed_path: str | None = None
    state_path = out_dir / "cem_state.npz"
    if state_path.is_file():
        with np.load(state_path) as state:
            if str(state["contract_sha256"].item()) != search_contract["contract_sha256"]:
                raise ValueError("existing CEM state belongs to a different search contract")
            mean = np.asarray(state["mean"], dtype=np.float32)
            std = np.asarray(state["std"], dtype=np.float32)
            global_best_parameters = np.asarray(state["best_parameters"], dtype=np.float32)
            if not np.array_equal(
                mean[~trainable_parameter_mask],
                initial_parameters[~trainable_parameter_mask],
            ) or not np.array_equal(
                global_best_parameters[~trainable_parameter_mask],
                initial_parameters[~trainable_parameter_mask],
            ):
                raise ValueError("existing CEM state changed a frozen synergy parameter")
            std[~trainable_parameter_mask] = 0.0
            start_iteration = int(state["iteration"])
            if "best_context" in state and "best_context_index" in state:
                global_best_context = np.asarray(state["best_context"], dtype=np.float32)
                global_best_context_index = int(state["best_context_index"])
            else:
                global_best_context = np.broadcast_to(
                    global_best_parameters,
                    (args.population * args.replicas, dimension),
                ).copy()
                global_best_context_index = 0
            if "search_frontier_parameters" in state:
                search_frontier_parameters = np.asarray(state["search_frontier_parameters"], dtype=np.float32)
                if (
                    search_frontier_parameters.shape != (dimension,)
                    or not np.isfinite(search_frontier_parameters).all()
                    or not np.array_equal(
                        search_frontier_parameters[~trainable_parameter_mask],
                        initial_parameters[~trainable_parameter_mask],
                    )
                ):
                    raise ValueError("existing CEM state has an incompatible search frontier")
                try:
                    search_frontier_metrics = json.loads(str(state["search_frontier_metrics_json"].item()))
                except (KeyError, json.JSONDecodeError, TypeError) as exc:
                    raise ValueError("existing CEM state has no readable search-frontier metrics") from exc
                if not isinstance(search_frontier_metrics, dict):
                    raise ValueError("existing CEM search-frontier metrics must be an object")
                search_frontier_iteration = int(state["search_frontier_iteration"].item())
                search_frontier_candidate_index = int(state["search_frontier_candidate_index"].item())
                recorded_seed_path = str(state["search_frontier_seed_path"].item())
                search_frontier_seed_path = recorded_seed_path or None
                if (
                    search_frontier_iteration <= 0
                    or not 0 <= search_frontier_candidate_index < int(args.population)
                    or search_frontier_seed_path is None
                    or not Path(search_frontier_seed_path).is_file()
                ):
                    raise ValueError("existing CEM state has an incomplete search-frontier binding")
            if args.cpu_guide_unqualified_mean:
                required_cpu_frontier_fields = {
                    "cpu_search_frontier_parameters",
                    "cpu_search_frontier_audit_json",
                    "cpu_search_frontier_iteration",
                    "cpu_search_frontier_candidate_index",
                    "cpu_search_frontier_seed_path",
                }
                missing_cpu_frontier_fields = required_cpu_frontier_fields - set(state.files)
                if missing_cpu_frontier_fields:
                    raise ValueError(
                        "existing CEM state has no CPU-guided search frontier: "
                        + ", ".join(sorted(missing_cpu_frontier_fields))
                    )
                cpu_search_frontier_parameters = np.asarray(
                    state["cpu_search_frontier_parameters"],
                    dtype=np.float32,
                )
                if (
                    cpu_search_frontier_parameters.shape != (dimension,)
                    or not np.isfinite(cpu_search_frontier_parameters).all()
                    or not np.array_equal(
                        cpu_search_frontier_parameters[~trainable_parameter_mask],
                        initial_parameters[~trainable_parameter_mask],
                    )
                ):
                    raise ValueError("existing CEM state has an incompatible CPU-guided frontier")
                try:
                    parsed_cpu_frontier = json.loads(
                        str(state["cpu_search_frontier_audit_json"].item())
                    )
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError("existing CEM state has no readable CPU-guided audit") from exc
                if parsed_cpu_frontier is not None and not isinstance(parsed_cpu_frontier, dict):
                    raise ValueError("existing CPU-guided audit must be an object or null")
                cpu_search_frontier_audit = parsed_cpu_frontier
                cpu_search_frontier_iteration = int(
                    state["cpu_search_frontier_iteration"].item()
                )
                cpu_search_frontier_candidate_index = int(
                    state["cpu_search_frontier_candidate_index"].item()
                )
                recorded_cpu_seed_path = str(
                    state["cpu_search_frontier_seed_path"].item()
                )
                cpu_search_frontier_seed_path = recorded_cpu_seed_path or None
                if cpu_search_frontier_audit is not None and (
                    cpu_search_frontier_iteration <= 0
                    or not 0 <= cpu_search_frontier_candidate_index < int(args.population)
                    or cpu_search_frontier_seed_path is None
                    or not Path(cpu_search_frontier_seed_path).is_file()
                ):
                    raise ValueError("existing CEM state has an incomplete CPU-guided frontier binding")
        best_report_path = out_dir / "best_teacher.json"
        if best_report_path.is_file():
            resumed_best = json.loads(best_report_path.read_text(encoding="utf-8"))
            global_best_metrics = resumed_best["metrics"]
            global_best_cpu_audit = resumed_best.get("cpu_quality_audit")
            if args.require_cpu_quality_for_best and (
                not isinstance(global_best_cpu_audit, dict)
                or global_best_cpu_audit.get("cpu_quality_passed") is not True
            ):
                raise ValueError("resumed CEM best lacks a passing CPU promotion audit")

    rng = np.random.default_rng(int(args.seed) + 1009 * start_iteration)
    elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
    history_path = out_dir / "cem_metrics.jsonl"
    total_cpu_audited_candidate_count = 0
    if history_path.is_file():
        total_cpu_audited_candidate_count = sum(
            int(json.loads(line).get("cpu_audited_candidate_count", 0))
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    t0 = time.time()
    for iteration in range(start_iteration + 1, int(args.iterations) + 1):
        candidates = rng.normal(mean, std, size=(args.population, dimension)).astype(np.float32)
        candidates = np.clip(candidates, -3.0, 3.0)
        candidates[:, ~trainable_parameter_mask] = initial_parameters[~trainable_parameter_mask]
        candidates[0] = mean
        candidates, search_frontier_candidate_indices = _inject_search_frontier_copies(
            candidates,
            frontier=search_frontier_parameters,
            copies=int(args.search_frontier_copies),
        )
        candidate_anchor_stop = 1 + int(args.search_frontier_copies)
        if iteration == 1:
            for offset, parameters in enumerate(
                additional_candidate_parameters,
                start=candidate_anchor_stop,
            ):
                candidates[offset] = parameters
        coordinate_probe_candidate_indices: tuple[int, ...] = ()
        if iteration == 1 and float(args.coordinate_probe_radius) > 0.0:
            candidates, coordinate_probe_candidate_indices = _inject_coordinate_probe_candidates(
                candidates,
                center=mean,
                trainable_parameter_mask=trainable_parameter_mask,
                radius=float(args.coordinate_probe_radius),
                start_index=candidate_anchor_stop,
            )
            candidates = np.clip(candidates, -3.0, 3.0)
            candidates[:, ~trainable_parameter_mask] = initial_parameters[~trainable_parameter_mask]
        # A single-feed reachability search must compare candidates from the
        # exact same initial state on every iteration.
        key = jax.random.PRNGKey(int(args.seed))
        expanded_candidates, candidate_lane_indices = _expand_candidates_across_stratified_lanes(
            candidates,
            replicas=int(args.replicas),
        )
        device_metrics, _device_trace = rollout(
            jnp.asarray(expanded_candidates),
            key,
            jnp.zeros((int(args.replicas),), dtype=jnp.int32),
        )
        replica_metrics = {
            name: _gather_candidate_major_lane_values(
                np.asarray(jax.device_get(value)),
                lane_indices=candidate_lane_indices,
            )
            for name, value in device_metrics.items()
        }
        metrics = _aggregate_replica_metrics(
            replica_metrics,
            population=int(args.population),
            replicas=int(args.replicas),
            min_replica_fraction=float(args.min_replica_fraction),
            min_outgoing_z_m_s=float(args.min_outgoing_z_m_s),
            min_forward_m_s=float(args.min_forward_m_s),
            require_legal_return_for_teacher=bool(args.require_legal_return_for_teacher),
            require_real_net_cross_for_teacher=bool(args.require_real_net_cross_for_teacher),
            real_net_cross_authoritative_for_teacher=bool(args.real_net_cross_authoritative_for_teacher),
            min_predicted_clearance_m=float(args.min_predicted_clearance_m),
            min_return_direction_signed_score=float(args.min_return_direction_signed_score),
            max_event_settled_velocity_delta_m_s=float(
                args.max_event_settled_velocity_delta_m_s
            ),
        )
        order = _rank_order(metrics)
        elites = candidates[order[-elite_count:]]
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        mean = (0.20 * mean + 0.80 * elite_mean).astype(np.float32)
        mean[~trainable_parameter_mask] = initial_parameters[~trainable_parameter_mask]
        std = np.maximum(
            np.where(trainable_parameter_mask, float(args.min_std), 0.0),
            0.20 * std + 0.80 * elite_std,
        ).astype(np.float32)
        std[~trainable_parameter_mask] = 0.0
        best_index = int(order[-1])
        backend_best_index = best_index
        best_metrics = _candidate_summary(
            metrics,
            best_index,
            min_replica_fraction=float(args.min_replica_fraction),
        )
        backend_iteration_best_metrics = best_metrics
        search_frontier_challenge: dict[str, Any] | None = None
        search_frontier_snapshot_candidate_index = int(backend_best_index)
        if args.verify_search_frontier:
            challenger_parameters = candidates[backend_best_index]
            same_as_incumbent = np.array_equal(
                challenger_parameters,
                search_frontier_parameters,
            )
            challenge_required = search_frontier_metrics is None or not same_as_incumbent
            search_frontier_improved = False
            if challenge_required:
                challenge_batch, challenge_slices = _build_search_frontier_challenge_batch(
                    expanded_candidates,
                    incumbent=search_frontier_parameters,
                    challenger=challenger_parameters,
                )
                challenge_key = jax.random.fold_in(
                    key,
                    int(10_000 + iteration),
                )
                challenge_device_metrics, _challenge_trace = rollout(
                    jnp.asarray(challenge_batch),
                    challenge_key,
                    jnp.zeros((int(args.replicas),), dtype=jnp.int32),
                )
                challenge_replica_metrics = {
                    name: np.asarray(jax.device_get(value)) for name, value in challenge_device_metrics.items()
                }

                def summarize_challenge_slice(
                    challenge_slice: slice,
                ) -> dict[str, Any]:
                    sliced = {name: values[challenge_slice] for name, values in challenge_replica_metrics.items()}
                    replica_count = int(challenge_slice.stop - challenge_slice.start)
                    aggregated = _aggregate_replica_metrics(
                        sliced,
                        population=1,
                        replicas=replica_count,
                        min_replica_fraction=float(args.min_replica_fraction),
                        min_outgoing_z_m_s=float(args.min_outgoing_z_m_s),
                        min_forward_m_s=float(args.min_forward_m_s),
                        require_legal_return_for_teacher=bool(args.require_legal_return_for_teacher),
                        require_real_net_cross_for_teacher=bool(args.require_real_net_cross_for_teacher),
                        real_net_cross_authoritative_for_teacher=bool(args.real_net_cross_authoritative_for_teacher),
                        min_predicted_clearance_m=float(args.min_predicted_clearance_m),
                        min_return_direction_signed_score=float(args.min_return_direction_signed_score),
                        max_event_settled_velocity_delta_m_s=float(
                            args.max_event_settled_velocity_delta_m_s
                        ),
                    )
                    return _candidate_summary(
                        aggregated,
                        0,
                        min_replica_fraction=float(args.min_replica_fraction),
                    )

                incumbent_verified_metrics = summarize_challenge_slice(challenge_slices[0])
                challenger_verified_metrics = summarize_challenge_slice(challenge_slices[1])
                recorded_reference = (
                    incumbent_verified_metrics if search_frontier_metrics is None else search_frontier_metrics
                )
                improves_recorded_frontier = _candidate_metrics_improve_frontier(
                    recorded_reference,
                    challenger_verified_metrics,
                    metric_names=tuple(metrics),
                )
                improves_concurrent_incumbent = not same_as_incumbent and _candidate_metrics_improve_frontier(
                    incumbent_verified_metrics,
                    challenger_verified_metrics,
                    metric_names=tuple(metrics),
                )
                challenger_accepted = bool(improves_recorded_frontier and improves_concurrent_incumbent)
                search_frontier_challenge = {
                    "evaluated": True,
                    "replicas_per_side": int(challenge_slices[0].stop - challenge_slices[0].start),
                    "same_as_incumbent": bool(same_as_incumbent),
                    "improves_recorded_frontier": bool(improves_recorded_frontier),
                    "improves_concurrent_incumbent": bool(improves_concurrent_incumbent),
                    "challenger_accepted": challenger_accepted,
                    "incumbent_metrics": incumbent_verified_metrics,
                    "challenger_metrics": challenger_verified_metrics,
                }
                if search_frontier_metrics is None:
                    search_frontier_improved = True
                    if challenger_accepted:
                        chosen_index = int(backend_best_index)
                        chosen_metrics = challenger_verified_metrics
                    else:
                        chosen_index = int(search_frontier_candidate_indices[0])
                        chosen_metrics = incumbent_verified_metrics
                    search_frontier_parameters = candidates[chosen_index].copy()
                    search_frontier_metrics = chosen_metrics
                    search_frontier_snapshot_candidate_index = chosen_index
                elif challenger_accepted:
                    search_frontier_improved = True
                    search_frontier_parameters = challenger_parameters.copy()
                    search_frontier_metrics = challenger_verified_metrics
                    search_frontier_snapshot_candidate_index = int(backend_best_index)
            else:
                search_frontier_challenge = {
                    "evaluated": False,
                    "reason": "iteration_best_equals_verified_incumbent",
                    "challenger_accepted": False,
                }
        else:
            search_frontier_improved = _candidate_metrics_improve_frontier(
                search_frontier_metrics,
                backend_iteration_best_metrics,
                metric_names=tuple(metrics),
            )
            if search_frontier_improved:
                search_frontier_parameters = candidates[backend_best_index].copy()
                search_frontier_metrics = backend_iteration_best_metrics
        if search_frontier_improved:
            search_frontier_iteration = int(iteration)
            search_frontier_candidate_index = int(search_frontier_snapshot_candidate_index)
            search_frontier_seed_path = None
        iteration_best_cpu_audit: dict[str, Any] | None = None
        cpu_rejected_improvement = False
        cpu_audited_candidate_count = 0
        cpu_audited_candidate_indices: list[int] = []
        cpu_audits_by_candidate: dict[int, dict[str, Any]] = {}
        cpu_candidate_order = [int(index) for index in order[::-1]]
        cpu_audit_limit = int(args.cpu_promotion_audit_limit)
        if args.cpu_audit_coordinate_probes and coordinate_probe_candidate_indices:
            coordinate_cpu_set = {0, *coordinate_probe_candidate_indices}
            cpu_candidate_order = [index for index in cpu_candidate_order if index in coordinate_cpu_set] + [
                index for index in cpu_candidate_order if index not in coordinate_cpu_set
            ]
            cpu_audit_limit = max(cpu_audit_limit, len(coordinate_cpu_set))
        if args.cpu_guide_unqualified_mean:
            # Always replay the current mean as the CPU baseline, then spend
            # the remaining budget on backend proposals.  The report cache
            # makes repeated unchanged anchors inexpensive.
            cpu_candidate_order = [0] + [index for index in cpu_candidate_order if index != 0]
        if global_best_metrics is None:
            if not args.require_cpu_quality_for_best:
                replace_best = True
            else:
                # A physically corrected run may intentionally start from an
                # old contact-only/downward candidate.  Search must be allowed
                # to improve away from it, while promotion remains fail-closed:
                # no global teacher exists until a ranked candidate passes the
                # independent CPU quality gate.
                replace_best = False
                promoted_index = best_index
                promoted_metrics = best_metrics
                for ranked_index in cpu_candidate_order:
                    if cpu_audited_candidate_count >= cpu_audit_limit:
                        break
                    candidate_index = int(ranked_index)
                    candidate_metrics = _candidate_summary(
                        metrics,
                        candidate_index,
                        min_replica_fraction=float(args.min_replica_fraction),
                    )
                    backend_quality_passed = bool(candidate_metrics.get("teacher_success") is True)
                    if not backend_quality_passed and not (
                        args.cpu_audit_coordinate_probes
                        or args.cpu_guide_unqualified_mean
                    ):
                        continue
                    candidate_cpu_audit = audit_cpu_promotion(candidates[candidate_index])
                    cpu_audited_candidate_count += 1
                    cpu_audited_candidate_indices.append(candidate_index)
                    cpu_audits_by_candidate[candidate_index] = candidate_cpu_audit
                    if _cross_backend_promotion_passes(
                        candidate_metrics,
                        candidate_cpu_audit,
                    ):
                        promoted_index = candidate_index
                        promoted_metrics = candidate_metrics
                        iteration_best_cpu_audit = candidate_cpu_audit
                        replace_best = True
                        break
                    cpu_rejected_improvement = True
        elif not args.require_cpu_quality_for_best:
            pair_metrics = {name: np.asarray([global_best_metrics[name], best_metrics[name]]) for name in metrics}
            replace_best = int(_rank_order(pair_metrics)[-1]) == 1
        else:
            replace_best = False
            promoted_index = best_index
            promoted_metrics = best_metrics
            for ranked_index in cpu_candidate_order:
                candidate_index = int(ranked_index)
                candidate_metrics = _candidate_summary(
                    metrics,
                    candidate_index,
                    min_replica_fraction=float(args.min_replica_fraction),
                )
                pair_metrics = {
                    name: np.asarray([global_best_metrics[name], candidate_metrics[name]]) for name in metrics
                }
                if int(_rank_order(pair_metrics)[-1]) != 1:
                    break
                if candidate_metrics.get("teacher_success") is not True:
                    continue
                if np.array_equal(candidates[candidate_index], global_best_parameters):
                    candidate_cpu_audit = global_best_cpu_audit
                else:
                    if cpu_audited_candidate_count >= cpu_audit_limit:
                        break
                    candidate_cpu_audit = audit_cpu_promotion(candidates[candidate_index])
                    cpu_audited_candidate_count += 1
                    cpu_audited_candidate_indices.append(candidate_index)
                    cpu_audits_by_candidate[candidate_index] = candidate_cpu_audit
                if _cross_backend_promotion_passes(
                    candidate_metrics,
                    candidate_cpu_audit,
                ):
                    promoted_index = candidate_index
                    promoted_metrics = candidate_metrics
                    iteration_best_cpu_audit = candidate_cpu_audit
                    replace_best = True
                    break
                cpu_rejected_improvement = True
        if replace_best:
            if args.require_cpu_quality_for_best:
                best_index = promoted_index
                best_metrics = promoted_metrics
            global_best_parameters = candidates[best_index].copy()
            global_best_metrics = best_metrics
            global_best_context = expanded_candidates.copy()
            global_best_context_index = int(candidate_lane_indices[best_index, 0])
            if args.require_cpu_quality_for_best:
                global_best_cpu_audit = iteration_best_cpu_audit

        cpu_guidance_improved = False
        cpu_guidance_anchor_retained = False
        cpu_guidance_candidate_index: int | None = None
        cpu_guidance_iteration_audit: dict[str, Any] | None = None
        if (
            args.cpu_guide_unqualified_mean
            and global_best_metrics is None
            and cpu_audits_by_candidate
        ):
            cpu_progress_thresholds = {
                "min_outgoing_z_m_s": float(args.cpu_min_outgoing_z_m_s),
                "min_forward_m_s": float(args.cpu_min_forward_m_s),
                "min_predicted_clearance_m": float(args.min_predicted_clearance_m),
                "min_return_direction_signed_score": float(
                    args.min_return_direction_signed_score
                ),
                "real_net_cross_authoritative": bool(
                    args.real_net_cross_authoritative_for_teacher
                ),
            }
            for candidate_index, candidate_audit in cpu_audits_by_candidate.items():
                if _cpu_unqualified_search_improves(
                    cpu_guidance_iteration_audit,
                    candidate_audit,
                    **cpu_progress_thresholds,
                ):
                    cpu_guidance_candidate_index = int(candidate_index)
                    cpu_guidance_iteration_audit = candidate_audit
            if (
                cpu_guidance_candidate_index is not None
                and cpu_guidance_iteration_audit is not None
                and _cpu_unqualified_search_improves(
                    cpu_search_frontier_audit,
                    cpu_guidance_iteration_audit,
                    **cpu_progress_thresholds,
                )
            ):
                cpu_guidance_improved = True
                cpu_search_frontier_parameters = candidates[
                    cpu_guidance_candidate_index
                ].copy()
                cpu_search_frontier_audit = cpu_guidance_iteration_audit
                cpu_search_frontier_iteration = int(iteration)
                cpu_search_frontier_candidate_index = int(
                    cpu_guidance_candidate_index
                )
                cpu_search_frontier_seed_path = None
        if (
            args.cpu_guide_unqualified_mean
            and global_best_metrics is None
            and cpu_search_frontier_parameters is not None
        ):
            mean, std = _retain_cpu_search_frontier_mean(
                mean,
                std,
                frontier_parameters=cpu_search_frontier_parameters,
                initial_parameters=initial_parameters,
                trainable_parameter_mask=trainable_parameter_mask,
                initial_std=float(args.initial_std),
            )
            cpu_guidance_anchor_retained = True

        total_cpu_audited_candidate_count += cpu_audited_candidate_count

        iteration_snapshot = _save_iteration_snapshot(
            out_dir=out_dir,
            iteration=iteration,
            contract_sha256=search_contract["contract_sha256"],
            candidates=candidates,
            rank_order=order,
            mean=mean,
            std=std,
            candidate_lane_indices=candidate_lane_indices,
            replica_metrics=replica_metrics,
            backend_best_index=backend_best_index,
            search_frontier_candidate_indices=(search_frontier_candidate_indices),
            coordinate_probe_candidate_indices=coordinate_probe_candidate_indices,
        )
        if search_frontier_improved:
            from scripts.export_cem_search_seed import (
                export_snapshot_candidate_seed,
            )

            frontier_seed = out_dir / (
                f"search_seed_frontier_iter{iteration:04d}_"
                f"candidate{search_frontier_snapshot_candidate_index:04d}_"
                "unqualified.json"
            )
            export_snapshot_candidate_seed(
                snapshot_path=iteration_snapshot["snapshot_path"],
                contract_path=out_dir / "cem_contract.json",
                output_path=frontier_seed,
                candidate_index=search_frontier_snapshot_candidate_index,
            )
            search_frontier_seed_path = str(frontier_seed.resolve())
        if cpu_guidance_improved:
            from scripts.export_cem_search_seed import (
                export_snapshot_candidate_seed,
            )

            if cpu_guidance_candidate_index is None:
                raise RuntimeError("CPU guidance improved without a candidate index")
            cpu_frontier_seed = out_dir / (
                f"search_seed_cpu_frontier_iter{iteration:04d}_"
                f"candidate{cpu_guidance_candidate_index:04d}_"
                "unqualified.json"
            )
            export_snapshot_candidate_seed(
                snapshot_path=iteration_snapshot["snapshot_path"],
                contract_path=out_dir / "cem_contract.json",
                output_path=cpu_frontier_seed,
                candidate_index=cpu_guidance_candidate_index,
            )
            cpu_search_frontier_seed_path = str(cpu_frontier_seed.resolve())

        row = {
            "iteration": iteration,
            "elapsed_seconds": time.time() - t0,
            "elite_count": elite_count,
            "distribution_std_mean": float(std.mean()),
            "distribution_std_max": float(std.max()),
            "distribution_trainable_std_mean": float(std[trainable_parameter_mask].mean()),
            "distribution_trainable_std_max": float(std[trainable_parameter_mask].max()),
            "iteration_best": backend_iteration_best_metrics,
            "promoted_iteration_metrics": (best_metrics if replace_best else None),
            "iteration_best_cpu_audit": iteration_best_cpu_audit,
            "cpu_rejected_improvement": cpu_rejected_improvement,
            "cpu_audited_candidate_count": cpu_audited_candidate_count,
            "cpu_audited_candidate_indices": cpu_audited_candidate_indices,
            "cpu_audit_limit_for_iteration": cpu_audit_limit,
            "global_best": global_best_metrics,
            "global_best_cpu_audit": global_best_cpu_audit,
            "cpu_guidance_improved": cpu_guidance_improved,
            "cpu_guidance_anchor_retained": cpu_guidance_anchor_retained,
            "cpu_guidance_candidate_index": cpu_guidance_candidate_index,
            "cpu_guidance_iteration_audit": cpu_guidance_iteration_audit,
            "cpu_search_frontier_audit": cpu_search_frontier_audit,
            "cpu_search_frontier_seed_path": cpu_search_frontier_seed_path,
            "search_frontier_best": search_frontier_metrics,
            "search_frontier_seed_path": search_frontier_seed_path,
            "search_frontier_challenge": search_frontier_challenge,
            "search_frontier_candidate_indices": list(search_frontier_candidate_indices),
            "iteration_snapshot": iteration_snapshot,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if global_best_metrics is not None:
            best_report = {
                "schema_version": "stage3_cem_teacher_candidate_v1",
                "contract_sha256": search_contract["contract_sha256"],
                "iteration": iteration,
                "parameters": global_best_parameters.tolist(),
                "metrics": global_best_metrics,
                "cpu_quality_audit": global_best_cpu_audit,
            }
            (out_dir / "best_teacher.json").write_text(
                json.dumps(best_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        np.savez(
            state_path,
            contract_sha256=np.asarray(search_contract["contract_sha256"]),
            iteration=np.asarray(iteration, dtype=np.int32),
            mean=mean,
            std=std,
            best_parameters=global_best_parameters,
            best_context=global_best_context,
            best_context_index=np.asarray(global_best_context_index, dtype=np.int32),
            search_frontier_parameters=search_frontier_parameters,
            search_frontier_metrics_json=np.asarray(
                json.dumps(
                    search_frontier_metrics,
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
            search_frontier_iteration=np.asarray(search_frontier_iteration, dtype=np.int32),
            search_frontier_candidate_index=np.asarray(search_frontier_candidate_index, dtype=np.int32),
            search_frontier_seed_path=np.asarray(search_frontier_seed_path or ""),
            cpu_search_frontier_parameters=cpu_search_frontier_parameters,
            cpu_search_frontier_audit_json=np.asarray(
                json.dumps(
                    cpu_search_frontier_audit,
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
            cpu_search_frontier_iteration=np.asarray(
                cpu_search_frontier_iteration,
                dtype=np.int32,
            ),
            cpu_search_frontier_candidate_index=np.asarray(
                cpu_search_frontier_candidate_index,
                dtype=np.int32,
            ),
            cpu_search_frontier_seed_path=np.asarray(
                cpu_search_frontier_seed_path or ""
            ),
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "cem/iteration": iteration,
                    "cem/std_mean": float(std.mean()),
                    "cem/trainable_std_mean": float(std[trainable_parameter_mask].mean()),
                    "cem/cpu_rejected_improvement": float(cpu_rejected_improvement),
                    "cem/cpu_audited_candidate_count": float(cpu_audited_candidate_count),
                    "cem/cpu_best_outgoing_z_m_s": (
                        0.0
                        if (global_best_cpu_audit or cpu_search_frontier_audit) is None
                        else float(
                            (global_best_cpu_audit or cpu_search_frontier_audit)[
                                "outgoing_z_m_s"
                            ]
                        )
                    ),
                    "cem/cpu_guidance_improved": float(cpu_guidance_improved),
                    "cem/cpu_guidance_anchor_retained": float(
                        cpu_guidance_anchor_retained
                    ),
                    "cem/cpu_search_frontier_outgoing_forward_m_s": (
                        0.0
                        if cpu_search_frontier_audit is None
                        else float(cpu_search_frontier_audit["outgoing_forward_m_s"])
                    ),
                    "cem/cpu_search_frontier_predicted_clearance_m": (
                        -1000.0
                        if cpu_search_frontier_audit is None
                        else float(cpu_search_frontier_audit["predicted_net_clearance_m"])
                    ),
                    "cem/cpu_search_frontier_return_direction_signed_score": (
                        0.0
                        if cpu_search_frontier_audit is None
                        else float(cpu_search_frontier_audit["return_direction_signed_score"])
                    ),
                    "cem/qualified_best_found": float(global_best_metrics is not None),
                    **_wandb_iteration_best_payload(backend_iteration_best_metrics),
                    **{
                        key.replace(
                            "cem/iteration_best_",
                            "cem/search_frontier_best_",
                        ): value
                        for key, value in _wandb_iteration_best_payload(search_frontier_metrics or {}).items()
                    },
                    **(
                        {}
                        if global_best_metrics is None
                        else {f"cem/best_{name}": value for name, value in global_best_metrics.items()}
                    ),
                },
                step=iteration,
            )
        display_metrics = backend_iteration_best_metrics if global_best_metrics is None else global_best_metrics
        print(
            f"[cem] iter={iteration}/{args.iterations} "
            f"qualified={int(global_best_metrics is not None)} "
            f"distance={display_metrics['min_ball_racket_distance_m']:.4f} "
            f"rebound={int(display_metrics['event_rebound'])} "
            f"vz={display_metrics['outgoing_z_m_s']:.3f} "
            f"forward={display_metrics['outgoing_forward_m_s']:.3f} "
            f"success={int(display_metrics['teacher_success'])}",
            flush=True,
        )
        if cpu_search_frontier_audit is not None and global_best_metrics is None:
            print(
                f"[cem-cpu-guide] improved={int(cpu_guidance_improved)} "
                f"vz={float(cpu_search_frontier_audit['outgoing_z_m_s']):.3f} "
                f"forward={float(cpu_search_frontier_audit['outgoing_forward_m_s']):.3f} "
                f"clearance={float(cpu_search_frontier_audit['predicted_net_clearance_m']):.3f} "
                f"direction={float(cpu_search_frontier_audit['return_direction_signed_score']):.3f}",
                flush=True,
            )

    if global_best_metrics is None:
        failure_report = {
            "schema_version": "stage3_single_feed_mjx_cem_failure_v1",
            "reason": _cem_failure_reason(total_cpu_audited_candidate_count),
            "total_cpu_audited_candidate_count": (total_cpu_audited_candidate_count),
            "best_unqualified_search_frontier": {
                "iteration": search_frontier_iteration,
                "candidate_index": search_frontier_candidate_index,
                "seed_path": search_frontier_seed_path,
                "metrics": search_frontier_metrics,
            },
            "best_cpu_audited_unqualified_search_frontier": {
                "iteration": cpu_search_frontier_iteration,
                "candidate_index": cpu_search_frontier_candidate_index,
                "seed_path": cpu_search_frontier_seed_path,
                "audit": cpu_search_frontier_audit,
            },
            "iterations_completed": int(args.iterations),
            "contract": search_contract,
            "wall_seconds": time.time() - t0,
            "wandb": (None if wandb_run is None else {"run_id": wandb_run.id, "url": wandb_run.url}),
        }
        (out_dir / "cem_failure.json").write_text(
            json.dumps(failure_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if wandb_run is not None:
            wandb_run.finish(exit_code=2)
        return 2

    verification_metric_batches: list[dict[str, np.ndarray]] = []
    verification_trace_batches: list[dict[str, np.ndarray]] = []
    verification_groups = _verification_group_indices(
        population=int(args.population),
        repeats=int(args.verification_repeats),
        anchor_group=global_best_context_index // int(args.replicas),
    )
    base_verification_key = jax.random.PRNGKey(int(args.seed))
    for verification_repeat, verification_group in enumerate(verification_groups):
        verification_key = (
            base_verification_key
            if verification_repeat == 0
            else jax.random.fold_in(base_verification_key, verification_repeat)
        )
        verification_start = int(verification_group) * int(args.replicas)
        verification_stop = verification_start + int(args.replicas)
        verification_context = global_best_context.copy()
        verification_context[verification_start:verification_stop] = global_best_parameters
        verification_trace_indices = jnp.arange(
            verification_start,
            verification_stop,
            dtype=jnp.int32,
        )
        device_metrics, device_trace = rollout(
            jnp.asarray(verification_context),
            verification_key,
            verification_trace_indices,
        )
        verification_metric_batches.append(
            {
                name: np.asarray(jax.device_get(value))[verification_start:verification_stop]
                for name, value in device_metrics.items()
            }
        )
        verification_trace_batches.append(
            {name: np.asarray(jax.device_get(value)) for name, value in device_trace.items()}
        )
    final_replica_metrics = {
        name: np.concatenate(
            [batch[name] for batch in verification_metric_batches],
            axis=0,
        )
        for name in verification_metric_batches[0]
    }
    verification_replica_count = int(args.replicas) * int(args.verification_repeats)
    final_aggregated_metrics = _aggregate_replica_metrics(
        final_replica_metrics,
        population=1,
        replicas=verification_replica_count,
        min_replica_fraction=float(args.min_replica_fraction),
        min_outgoing_z_m_s=float(args.min_outgoing_z_m_s),
        min_forward_m_s=float(args.min_forward_m_s),
        require_legal_return_for_teacher=bool(args.require_legal_return_for_teacher),
        require_real_net_cross_for_teacher=bool(args.require_real_net_cross_for_teacher),
        real_net_cross_authoritative_for_teacher=bool(args.real_net_cross_authoritative_for_teacher),
        min_predicted_clearance_m=float(args.min_predicted_clearance_m),
        min_return_direction_signed_score=float(args.min_return_direction_signed_score),
        max_event_settled_velocity_delta_m_s=float(
            args.max_event_settled_velocity_delta_m_s
        ),
    )
    verified_metrics = _candidate_summary(
        final_aggregated_metrics,
        0,
        min_replica_fraction=float(args.min_replica_fraction),
    )
    final_trace = {
        name: np.concatenate(
            [batch[name] for batch in verification_trace_batches],
            axis=1,
        )
        for name in verification_trace_batches[0]
    }
    replica_order = _rank_order(final_replica_metrics)
    selected_success = (
        np.asarray(final_replica_metrics["event_rebound"], dtype=bool)
        & (
            np.asarray(
                final_replica_metrics["event_settled_velocity_delta_m_s"],
                dtype=float,
            )
            <= float(args.max_event_settled_velocity_delta_m_s)
        )
        & np.asarray(final_replica_metrics["no_fall"], dtype=bool)
        & np.asarray(final_replica_metrics["high_region_contact"], dtype=bool)
        & (np.asarray(final_replica_metrics["outgoing_z_m_s"], dtype=float) >= float(args.min_outgoing_z_m_s))
        & (np.asarray(final_replica_metrics["outgoing_forward_m_s"], dtype=float) >= float(args.min_forward_m_s))
    )
    if args.require_legal_return_for_teacher:
        if not args.real_net_cross_authoritative_for_teacher:
            selected_success &= np.asarray(
                final_replica_metrics["predicted_clearance_m"],
                dtype=float,
            ) >= float(args.min_predicted_clearance_m)
        selected_success &= np.asarray(
            final_replica_metrics["return_direction_signed_score"],
            dtype=float,
        ) >= float(args.min_return_direction_signed_score)
    if args.require_real_net_cross_for_teacher:
        selected_success &= np.asarray(final_replica_metrics["crossed_net"], dtype=bool)
    successful_replica_order = [int(index) for index in replica_order if selected_success[int(index)]]
    selected_replica = int(successful_replica_order[-1] if successful_replica_order else replica_order[-1])
    selected_trace = {name: value[:, selected_replica] for name, value in final_trace.items()}
    alive_length = int(np.asarray(selected_trace["alive"], dtype=bool).sum())
    teacher_path = out_dir / "teacher_trajectory_mjx.npz"
    np.savez_compressed(
        teacher_path,
        **{name: value[:alive_length] for name, value in selected_trace.items()},
        selected_action_indices=np.asarray(selected_indices, dtype=np.int32),
        physical_scales=np.asarray(physical_scales_np, dtype=np.float32),
        feed_fingerprint=np.asarray(requested_fingerprint),
        swing_phase_advance_s=np.asarray(
            effective_swing_phase_advance_s,
            dtype=np.float32,
        ),
        source_checkpoint_sha256=np.asarray(search_contract["source_checkpoint_sha256"]),
        search_contract_sha256=np.asarray(search_contract["contract_sha256"]),
        search_context_index=np.asarray(global_best_context_index, dtype=np.int32),
        selected_replica=np.asarray(selected_replica, dtype=np.int32),
        selected_verification_repeat=np.asarray(
            selected_replica // int(args.replicas),
            dtype=np.int32,
        ),
        selected_replica_within_repeat=np.asarray(
            selected_replica % int(args.replicas),
            dtype=np.int32,
        ),
        verification_group_indices=np.asarray(verification_groups, dtype=np.int32),
        selected_batch_group=np.asarray(
            verification_groups[selected_replica // int(args.replicas)],
            dtype=np.int32,
        ),
        parameterization=np.asarray(str(args.parameterization)),
        time_knots=np.asarray(int(args.time_knots), dtype=np.int32),
        synergy_names=np.asarray(synergy_names),
        synergy_basis=(
            np.empty((0, len(selected_indices)), dtype=np.float32) if synergy_basis_np is None else synergy_basis_np
        ),
        trace_schema_version=np.asarray("stage3_cem_teacher_trajectory_v3"),
        outgoing_velocity_semantics=np.asarray("post_control_step_after_all_physics_substeps"),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
        kinematic_sample_timing=np.asarray("pre_control_step"),
        control_dt_s=np.asarray(
            env.control_substeps * env.timestep,
            dtype=np.float32,
        ),
        right_arm_body_names=np.asarray(RIGHT_ARM_AUDIT_BODY_NAMES),
    )
    selected_replica_metrics = _candidate_summary(
        final_replica_metrics,
        selected_replica,
    )
    if args.require_legal_return_for_teacher:
        selected_replica_metrics["teacher_success"] = bool(selected_success[selected_replica])
        selected_replica_metrics["preferred_teacher_success"] = bool(selected_success[selected_replica])
    mjx_trace_report = {
        "steps": alive_length,
        "hit": bool(np.asarray(selected_trace["hit_event"][:alive_length]).any()),
        "event_rebound": bool(np.asarray(selected_trace["event_rebound"][:alive_length]).any()),
        "body_fall": bool(np.asarray(selected_trace["body_fall"][:alive_length]).any()),
        "selected_replica": selected_replica,
        "selected_verification_repeat": selected_replica // int(args.replicas),
        "selected_replica_within_repeat": selected_replica % int(args.replicas),
        "verification_group_indices": list(verification_groups),
        "verification_context_semantics": search_contract["verification_context_semantics"],
        "selected_batch_group": verification_groups[selected_replica // int(args.replicas)],
        "selected_replica_metrics": selected_replica_metrics,
        "robust_candidate_metrics": verified_metrics,
        "trace_path": str(teacher_path),
        "trace_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
    }
    cpu_trace_report = _save_cpu_teacher_trace(
        path=out_dir / "teacher_trajectory_cpu_audit.npz",
        feed=feed,
        paths=paths,
        actor=restored.agent,
        obs_mean=np.asarray(restored.obs_rms.mean),
        obs_var=np.asarray(restored.obs_rms.var),
        parameters=global_best_parameters,
        selected_indices=selected_indices,
        physical_scales=physical_scales_np,
        base_policy_artifact=base_policy,
        residual_scale=residual_scale,
        open_s=float(window["time_to_intercept_open_s"]),
        close_s=float(window["time_to_intercept_close_s"]),
        smoothing_s=float(window["smoothing_s"]),
        max_episode_steps=int(args.max_episode_steps),
        time_knots=int(args.time_knots),
        synergy_basis=synergy_basis_np,
        swing_phase_advance_s=effective_swing_phase_advance_s,
    )
    cpu_replay_event_equivalent = bool(cpu_trace_report["event_rebound"] == mjx_trace_report["event_rebound"])
    cpu_replay_passed = bool(
        cpu_replay_event_equivalent
        and cpu_trace_report["hit"]
        and cpu_trace_report["event_rebound"]
        and not cpu_trace_report["body_fall"]
        and (
            not args.require_cpu_quality_for_best
            or (isinstance(global_best_cpu_audit, dict) and global_best_cpu_audit.get("cpu_quality_passed") is True)
        )
    )
    mjx_teacher_passed = bool(verified_metrics["teacher_success"])
    report = {
        "schema_version": "stage3_single_feed_mjx_cem_report_v3",
        "passed": bool(mjx_teacher_passed and cpu_replay_passed),
        "mjx_teacher_passed": mjx_teacher_passed,
        "cpu_replay_passed": cpu_replay_passed,
        "preferred_teacher_success": bool(verified_metrics["preferred_teacher_success"]),
        "iterations_completed": int(args.iterations),
        "wall_seconds": time.time() - t0,
        "best_search_metrics": global_best_metrics,
        "verified_metrics": verified_metrics,
        "teacher_trace": mjx_trace_report,
        "cpu_replay_audit": cpu_trace_report,
        "cpu_gated_best_audit": global_best_cpu_audit,
        "cpu_replay_event_equivalent": cpu_replay_event_equivalent,
        "contract": search_contract,
        "wandb": (None if wandb_run is None else {"run_id": wandb_run.id, "url": wandb_run.url}),
    }
    (out_dir / "cem_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary.update(report)
        wandb_run.finish(exit_code=0 if report["passed"] else 1)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
