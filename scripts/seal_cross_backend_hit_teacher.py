#!/usr/bin/env python3
"""Seal a CPU-validated teacher when accelerator search quality is replay-sensitive.

The source CEM run must have found a replica-robust successful candidate during
search, retained a robust real rebound in its independent MJX replay, and
produced an event-equivalent, upright CPU MuJoCo replay.  The CPU replay then
becomes the supervised trajectory only when its measured outgoing velocity and
high-region contact also pass the task gates.  This does not turn a contact-only
or downward trajectory into a teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SCHEMA = "stage3_single_feed_mjx_cem_report_v3"
OUTPUT_SCHEMA = "stage3_cross_backend_quality_teacher_report_v3"
OUTPUT_TRACE_SCHEMA = "stage3_cross_backend_training_backend_and_cpu_quality_teacher_v3"
HIGH_REGION_SEMANTICS = "soft_window_teacher_gate_not_exact_apex"
RETURN_QUALITY_MARGIN_SEMANTICS = "same_replica_training_backend_margin_gate"
OUTGOING_VELOCITY_SEMANTICS = (
    "post_control_step_after_all_physics_substeps"
)
EVENT_REBOUND_CONTACT_SEMANTICS = (
    "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
)
VERIFICATION_CONTEXT_SEMANTICS_BY_BACKEND = {
    "warp": "same_candidate_relocated_across_deterministic_warp_batch_lanes",
    "jax": (
        "same_candidate_relocated_across_deterministic_"
        "standard_mjx_jax_batch_lanes"
    ),
}
VERIFICATION_SOURCE_BY_BACKEND = {
    "warp": "warp_training_backend_plus_independent_cpu_mujoco_quality_replay",
    "jax": (
        "standard_mjx_jax_training_backend_plus_independent_"
        "cpu_mujoco_quality_replay"
    ),
}
SWING_PHASE_TIMING_SEMANTICS = (
    "frozen_base_swing_phase_advance_applied_identically_to_search_"
    "backend_and_cpu_replays"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _json_safe_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def seal_cross_backend_teacher(
    source_report_path: str | Path,
    out_dir: str | Path,
    *,
    player_half_sign: int = -1,
    min_outgoing_z_m_s: float = 0.5,
    min_forward_m_s: float = 2.0,
    source_base_phase_advance_s: float | None = None,
    runtime_base_phase_advance_s: float | None = None,
) -> dict[str, Any]:
    source_path = Path(source_report_path).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    if player_half_sign not in {-1, 1}:
        raise ValueError("player_half_sign must be -1 or 1")
    if (source_base_phase_advance_s is None) != (runtime_base_phase_advance_s is None):
        raise ValueError("source/runtime base phase advances must be provided together")
    if not source_path.is_file():
        raise FileNotFoundError(f"source CEM report is missing: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source CEM report schema is incompatible")

    contract = dict(source.get("contract", {}) or {})
    training_backend = str(contract.get("mjx_impl", ""))
    if training_backend not in VERIFICATION_CONTEXT_SEMANTICS_BY_BACKEND:
        raise ValueError("source CEM report has an unsupported training backend")
    verification_context_semantics = VERIFICATION_CONTEXT_SEMANTICS_BY_BACKEND[
        training_backend
    ]
    contract_feed_fingerprint = str(contract.get("feed_fingerprint", ""))
    if not contract_feed_fingerprint:
        raise ValueError("source CEM report has no bound feed fingerprint")
    contract_phase_advance_s = _json_safe_float(
        contract.get("swing_phase_advance_s"),
        name="source contract swing phase advance",
    )
    if contract_phase_advance_s < 0.0:
        raise ValueError("source contract swing phase advance must be non-negative")
    if contract.get("swing_phase_timing_semantics") != SWING_PHASE_TIMING_SEMANTICS:
        raise ValueError("source CEM report does not bind search/CPU swing timing")
    if contract.get("outgoing_velocity_semantics") != OUTGOING_VELOCITY_SEMANTICS:
        raise ValueError(
            "source CEM report does not measure the settled post-control outgoing velocity"
        )
    if contract.get("event_rebound_contact_semantics") != (
        EVENT_REBOUND_CONTACT_SEMANTICS
    ):
        raise ValueError("source CEM report permits a double-applied stringbed impact")
    high_contract = dict(contract.get("high_region_contact", {}) or {})
    if high_contract.get("semantics") != HIGH_REGION_SEMANTICS:
        raise ValueError("source CEM report does not use the broad high-region semantics")
    max_stringbed_deficit = _json_safe_float(
        high_contract.get("max_stringbed_height_deficit_m"),
        name="max stringbed height deficit",
    )
    max_hand_deficit = _json_safe_float(
        high_contract.get("max_hand_height_deficit_m"),
        name="max hand height deficit",
    )
    min_replica_fraction = _json_safe_float(
        contract.get("min_replica_fraction"),
        name="minimum replica fraction",
    )
    search_margin = dict(contract.get("return_quality_search_margin", {}) or {})
    if search_margin.get("semantics") != RETURN_QUALITY_MARGIN_SEMANTICS:
        raise ValueError(
            "source CEM report does not bind the training-backend return-quality margin"
        )
    search_min_outgoing_z = _json_safe_float(
        search_margin.get("min_outgoing_z_m_s"),
        name="search minimum outgoing z",
    )
    search_min_forward = _json_safe_float(
        search_margin.get("min_forward_m_s"),
        name="search minimum forward velocity",
    )
    if (
        search_min_outgoing_z < float(min_outgoing_z_m_s)
        or search_min_forward < float(min_forward_m_s)
    ):
        raise ValueError(
            "source CEM training-backend search margin is below the deployment quality gate"
        )
    replicas_per_candidate = int(contract.get("replicas_per_candidate", 0))
    required_replica_count = int(contract.get("required_replica_count", 0))
    if replicas_per_candidate <= 0 or not 1 <= required_replica_count <= replicas_per_candidate:
        raise ValueError("source CEM replica-count contract is invalid")
    # The serialized decimal threshold may be 0.6666667 while an observed
    # two-of-three rate is represented as 0.666666666....  Reconstruct the
    # exact discrete gate instead of rejecting the same result on formatting.
    required_replica_rate = required_replica_count / replicas_per_candidate
    if required_replica_rate + 1.0e-6 < min_replica_fraction:
        raise ValueError("source CEM replica-count and fraction gates disagree")
    population = int(contract.get("population", 0))
    verification_repeats = int(contract.get("verification_repeats", 0))
    source_teacher_trace = dict(source.get("teacher_trace", {}) or {})
    verification_groups = source_teacher_trace.get("verification_group_indices")
    if (
        contract.get("verification_context_semantics")
        != verification_context_semantics
        or source_teacher_trace.get("verification_context_semantics")
        != verification_context_semantics
        or population <= 0
        or verification_repeats <= 0
        or not isinstance(verification_groups, list)
        or len(verification_groups) != verification_repeats
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < population
            for value in verification_groups
        )
        or len(set(verification_groups)) != verification_repeats
        or source_teacher_trace.get("selected_batch_group")
        not in verification_groups
    ):
        backend_label = "Warp" if training_backend == "warp" else "standard MJX JAX"
        raise ValueError(
            "source training-backend verification did not relocate the candidate "
            f"across distinct {backend_label} batch lanes"
        )

    best = dict(source.get("best_search_metrics", {}) or {})
    final_mjx = dict(source.get("verified_metrics", {}) or {})
    if (
        best.get("teacher_success") is not True
        or float(best.get("teacher_success_rate", 0.0)) + 1.0e-9
        < required_replica_rate
        or best.get("high_region_contact") is not True
        or best.get("no_fall") is not True
    ):
        raise ValueError("source search never found a robust upright quality teacher")
    if source.get("passed") is not True or source.get("mjx_teacher_passed") is not True:
        raise ValueError(
            "source CEM report did not pass its independent training-backend and CPU replay gates"
        )
    if (
        final_mjx.get("event_rebound") is not True
        or float(final_mjx.get("event_rebound_rate", 0.0)) + 1.0e-9
        < required_replica_rate
        or final_mjx.get("high_region_contact") is not True
        or final_mjx.get("no_fall") is not True
        or final_mjx.get("teacher_success") is not True
        or float(final_mjx.get("teacher_success_rate", 0.0)) + 1.0e-9
        < required_replica_rate
        or final_mjx.get("return_quality") is not True
        or float(final_mjx.get("return_quality_rate", 0.0)) + 1.0e-9
        < required_replica_rate
        or float(final_mjx.get("positive_outgoing_z_rate", 0.0)) + 1.0e-9
        < required_replica_rate
        or float(final_mjx.get("positive_outgoing_forward_rate", 0.0)) + 1.0e-9
        < required_replica_rate
    ):
        raise ValueError(
            "independent training-backend replay did not retain a robust upright, "
            "high-region, upward-forward return"
        )

    cpu_audit = dict(source.get("cpu_replay_audit", {}) or {})
    if (
        source.get("cpu_replay_event_equivalent") is not True
        or cpu_audit.get("hit") is not True
        or cpu_audit.get("event_rebound") is not True
        or cpu_audit.get("body_fall") is not False
    ):
        raise ValueError("source CPU replay did not retain an upright real rebound")
    cpu_trace_path = Path(str(cpu_audit.get("trace_path", ""))).expanduser().resolve()
    if not cpu_trace_path.is_file() or cpu_audit.get("trace_sha256") != _sha256(cpu_trace_path):
        raise ValueError("source CPU trace is missing or detached from its report")

    required = {
        "observation_normalized",
        "correction_raw",
        "correction_window",
        "time_to_intercept_s",
        "event_rebound",
        "shuttle_velocity",
        "stringbed_position",
        "right_arm_body_position_xyz_m",
        "body_fall",
        "selected_action_indices",
        "physical_scales",
        "feed_fingerprint",
        "swing_phase_advance_s",
        "outgoing_velocity_semantics",
        "event_rebound_contact_semantics",
    }
    with np.load(cpu_trace_path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError("CPU teacher trace is missing fields: " + ", ".join(missing))
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    if str(np.asarray(arrays["outgoing_velocity_semantics"]).item()) != (
        OUTGOING_VELOCITY_SEMANTICS
    ):
        raise ValueError("CPU teacher trace has incompatible outgoing-velocity semantics")
    if str(np.asarray(arrays["event_rebound_contact_semantics"]).item()) != (
        EVENT_REBOUND_CONTACT_SEMANTICS
    ):
        raise ValueError("CPU teacher trace permits a double-applied stringbed impact")
    trace_feed_fingerprint = str(np.asarray(arrays["feed_fingerprint"]).item())
    if trace_feed_fingerprint != contract_feed_fingerprint:
        raise ValueError("CPU teacher trace feed is detached from the CEM contract")
    trace_phase_advance_s = _json_safe_float(
        np.asarray(arrays["swing_phase_advance_s"]).item(),
        name="CPU teacher trace swing phase advance",
    )
    if trace_phase_advance_s < 0.0:
        raise ValueError("CPU teacher trace swing phase advance must be non-negative")
    if cpu_audit.get("feed_fingerprint") != trace_feed_fingerprint:
        raise ValueError("CPU replay report feed is detached from its trajectory")
    cpu_audit_phase_advance_s = _json_safe_float(
        cpu_audit.get("swing_phase_advance_s"),
        name="CPU replay report swing phase advance",
    )
    if not math.isclose(
        cpu_audit_phase_advance_s,
        trace_phase_advance_s,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("CPU replay report timing is detached from its trajectory")
    if source_base_phase_advance_s is None and not math.isclose(
        trace_phase_advance_s,
        contract_phase_advance_s,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("CPU teacher trace timing is detached from the CEM contract")

    event = np.asarray(arrays["event_rebound"], dtype=bool)
    event_indices = np.flatnonzero(event)
    if event_indices.size == 0 or np.asarray(arrays["body_fall"], dtype=bool).any():
        raise ValueError("CPU teacher trace has no upright real rebound")
    event_index = int(event_indices[0])
    shuttle_velocity = np.asarray(arrays["shuttle_velocity"], dtype=np.float32)
    if shuttle_velocity.ndim != 2 or shuttle_velocity.shape[1] != 3:
        raise ValueError("CPU shuttle velocity trace must have shape [T, 3]")
    outgoing_velocity = shuttle_velocity[event_index]
    outgoing_z = float(outgoing_velocity[2])
    outgoing_forward = float(-player_half_sign * outgoing_velocity[0])
    if outgoing_z < float(min_outgoing_z_m_s) or outgoing_forward < float(min_forward_m_s):
        raise ValueError("CPU replay return is not simultaneously upward and forward")

    window = np.asarray(arrays["correction_window"], dtype=np.float32)
    time_to_intercept_s = np.asarray(
        arrays["time_to_intercept_s"], dtype=np.float32
    )
    active = window > 0.05
    if (
        active.shape != event.shape
        or time_to_intercept_s.shape != event.shape
        or not active.any()
    ):
        raise ValueError("CPU teacher correction window is incompatible or empty")
    stringbed_height = np.asarray(arrays["stringbed_position"], dtype=np.float32)[:, 2]
    right_arm = np.asarray(arrays["right_arm_body_position_xyz_m"], dtype=np.float32)
    if right_arm.ndim != 3 or right_arm.shape[0] != event.shape[0]:
        raise ValueError("CPU right-arm audit trace has an incompatible shape")
    hand_height = right_arm[:, -1, 2]
    stringbed_deficit = float(
        max(0.0, float(np.max(stringbed_height[active])) - float(stringbed_height[event_index]))
    )
    hand_deficit = float(
        max(0.0, float(np.max(hand_height[active])) - float(hand_height[event_index]))
    )
    if stringbed_deficit > max_stringbed_deficit or hand_deficit > max_hand_deficit:
        raise ValueError("CPU replay contact lies outside the broad high-region window")

    observations = np.asarray(arrays["observation_normalized"], dtype=np.float32)
    corrections = np.asarray(arrays["correction_raw"], dtype=np.float32)
    selected_indices = np.asarray(arrays["selected_action_indices"], dtype=np.int32)
    physical_scales = np.asarray(arrays["physical_scales"], dtype=np.float32)
    if observations.ndim != 2 or corrections.shape != (
        observations.shape[0],
        selected_indices.size,
    ):
        raise ValueError("CPU teacher observation/correction arrays are incompatible")
    if (
        physical_scales.shape != selected_indices.shape
        or not np.isfinite(physical_scales).all()
        or np.any(physical_scales <= 0.0)
    ):
        raise ValueError("CPU teacher correction physical scales are incompatible")
    if not all(
        np.isfinite(value).all()
        for value in (
            observations,
            corrections,
            window,
            time_to_intercept_s,
            shuttle_velocity,
        )
    ):
        raise ValueError("CPU teacher trace contains non-finite values")

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "teacher_trajectory_cpu_quality.npz"
    arrays.update(
        {
            "observation_normalized": observations,
            "correction_raw": corrections,
            "correction_window": window,
            "event_rebound": event,
            "outgoing_shuttle_velocity_xyz_m_s": shuttle_velocity,
            "selected_action_indices": selected_indices,
            "source_checkpoint_sha256": np.asarray(
                str(contract["source_checkpoint_sha256"])
            ),
            "search_contract_sha256": np.asarray(str(contract["contract_sha256"])),
            "trace_schema_version": np.asarray(OUTPUT_TRACE_SCHEMA),
        }
    )
    base_timing_transfer: dict[str, Any] | None = None
    if source_base_phase_advance_s is not None:
        source_phase = _json_safe_float(
            source_base_phase_advance_s,
            name="source base phase advance",
        )
        runtime_phase = _json_safe_float(
            runtime_base_phase_advance_s,
            name="runtime base phase advance",
        )
        if math.isclose(source_phase, runtime_phase, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("base timing transfer requires a real phase change")
        if not math.isclose(
            runtime_phase,
            contract_phase_advance_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("timing-transfer runtime phase is detached from the CEM contract")
        if not math.isclose(
            runtime_phase,
            trace_phase_advance_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("timing-transfer runtime phase lacks a matching CPU replay")
        base_timing_transfer = {
            "schema_version": "stage3_teacher_verified_base_timing_transfer_v1",
            "source_phase_advance_s": source_phase,
            "runtime_phase_advance_s": runtime_phase,
            "verification_source": "independent_cpu_mujoco_quality_replay_at_runtime_timing",
            "cpu_quality_verified": True,
            "source_cem_report_sha256": _sha256(source_path),
            "source_cpu_trace_sha256": _sha256(cpu_trace_path),
        }
        base_timing_transfer["evidence_sha256"] = _json_hash(base_timing_transfer)
        arrays["source_base_phase_advance_s"] = np.asarray(source_phase, dtype=np.float64)
        arrays["runtime_base_phase_advance_s"] = np.asarray(runtime_phase, dtype=np.float64)
    np.savez_compressed(dataset_path, **arrays)

    robust_rate = float(final_mjx["teacher_success_rate"])
    accepted_metrics = {
        "teacher_success": True,
        "teacher_success_rate": robust_rate,
        "high_region_contact": True,
        "high_region_contact_rate": robust_rate,
        "event_rebound": True,
        "event_rebound_rate": float(final_mjx["event_rebound_rate"]),
        "no_fall": True,
        "outgoing_z_m_s": outgoing_z,
        "outgoing_forward_m_s": outgoing_forward,
        "training_backend_outgoing_z_m_s": float(final_mjx["outgoing_z_m_s"]),
        "training_backend_outgoing_forward_m_s": float(
            final_mjx["outgoing_forward_m_s"]
        ),
        "training_backend_teacher_success_rate": float(
            final_mjx["teacher_success_rate"]
        ),
        "stringbed_height_deficit_at_hit_m": stringbed_deficit,
        "hand_height_deficit_at_hit_m": hand_deficit,
    }
    enriched_cpu_audit = {
        **cpu_audit,
        "outgoing_velocity_xyz_m_s": outgoing_velocity.tolist(),
        "outgoing_z_m_s": outgoing_z,
        "outgoing_forward_m_s": outgoing_forward,
        "stringbed_height_deficit_at_hit_m": stringbed_deficit,
        "hand_height_deficit_at_hit_m": hand_deficit,
        "high_region_contact": True,
        "quality_success": True,
    }
    report = {
        "schema_version": OUTPUT_SCHEMA,
        "passed": True,
        "mjx_teacher_passed": True,
        "cpu_replay_passed": True,
        "verification_source": VERIFICATION_SOURCE_BY_BACKEND[training_backend],
        "verified_metrics": accepted_metrics,
        "teacher_trace": {
            "trace_path": str(dataset_path),
            "trace_sha256": _sha256(dataset_path),
            "selected_replica_metrics": accepted_metrics,
            "source_cpu_trace_path": str(cpu_trace_path),
            "source_cpu_trace_sha256": _sha256(cpu_trace_path),
        },
        "cpu_replay_audit": enriched_cpu_audit,
        "cpu_replay_event_equivalent": True,
        "contract": contract,
        "cross_backend_evidence": {
            "training_backend_quality_verified": True,
            "training_backend": training_backend,
            "feed_fingerprint": contract_feed_fingerprint,
            "training_backend_swing_phase_advance_s": contract_phase_advance_s,
            "cpu_replay_swing_phase_advance_s": trace_phase_advance_s,
            "outgoing_velocity_semantics": OUTGOING_VELOCITY_SEMANTICS,
            "event_rebound_contact_semantics": EVENT_REBOUND_CONTACT_SEMANTICS,
            "deployment_quality_gate": {
                "min_outgoing_z_m_s": float(min_outgoing_z_m_s),
                "min_forward_m_s": float(min_forward_m_s),
                "min_replica_rate": required_replica_rate,
            },
            "search_quality_margin": search_margin,
            "verification_context_semantics": verification_context_semantics,
            "verification_group_indices": verification_groups,
            "source_cem_report_path": str(source_path),
            "source_cem_report_sha256": _sha256(source_path),
            "best_search_metrics": best,
            "independent_mjx_replay_metrics": final_mjx,
        },
    }
    if base_timing_transfer is not None:
        report["base_timing_transfer"] = base_timing_transfer
    report_path = output_root / "cem_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cem-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--player-half-sign", type=int, default=-1)
    parser.add_argument("--min-outgoing-z-m-s", type=float, default=0.5)
    parser.add_argument("--min-forward-m-s", type=float, default=2.0)
    parser.add_argument("--source-base-phase-advance-s", type=float, default=None)
    parser.add_argument("--runtime-base-phase-advance-s", type=float, default=None)
    args = parser.parse_args()
    report = seal_cross_backend_teacher(
        args.source_cem_report,
        args.out_dir,
        player_half_sign=args.player_half_sign,
        min_outgoing_z_m_s=args.min_outgoing_z_m_s,
        min_forward_m_s=args.min_forward_m_s,
        source_base_phase_advance_s=args.source_base_phase_advance_s,
        runtime_base_phase_advance_s=args.runtime_base_phase_advance_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
