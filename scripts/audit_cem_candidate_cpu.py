#!/usr/bin/env python3
"""Replay a saved Stage-3 CEM candidate in independent CPU MuJoCo.

This is an intermediate audit utility.  It does not seal a teacher: it binds
the exact candidate/contract hashes, saves the full CPU trajectory, and reports
the physical return, broad high-region timing, landing termination, and body
stability.  Production sealing still requires the final relocated-lane Warp
verification in ``cem_report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.shuttle_feeder import feed_sample_fingerprint
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _ensure_feed_bank_artifact,
    _ensure_scene,
    _policy_update_contract,
    _residual_scale_overrides,
    _residual_scale_schedule,
    load_incoming_hit_spec,
)
from scripts.optimize_single_feed_hit_mjx import (
    _anatomical_synergy_basis,
    _save_cpu_teacher_trace,
    _summarize_cpu_quality_trace,
    _source_actor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_sha256(parameters: Any) -> str:
    values = np.asarray(parameters, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("candidate parameters must be a finite vector")
    # Match the CEM candidate ABI exactly.  Hashing JSON text here produced a
    # different digest for the same vector and made audit provenance appear
    # detached from the sealed parameter_f32_sha256.
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _validated_trace_path(path: str | Path) -> Path:
    trace_path = Path(path).expanduser().resolve()
    if trace_path.suffix.lower() != ".npz":
        raise ValueError("CPU audit output path must end in .npz")
    return trace_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--record-video",
        default=None,
        help=(
            "optional MP4 path for an immutable engineering replay; this does "
            "not promote the candidate or relax the CPU quality gate"
        ),
    )
    parser.add_argument(
        "--engineering-demo-only",
        action="store_true",
        help=(
            "allow replaying an immutable inline CPU audit solely to render a "
            "non-promotable engineering demo"
        ),
    )
    parser.add_argument(
        "--feed-fingerprint",
        default=None,
        help=(
            "optional alternate feed for an explicitly unqualified search seed; "
            "teacher candidates remain bound to their source feed"
        ),
    )
    parser.add_argument(
        "--swing-phase-advance-s",
        type=float,
        default=None,
        help=(
            "optional base-swing timing intervention for an explicitly "
            "unqualified search seed"
        ),
    )
    parser.add_argument("--player-half-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--deployment-min-outgoing-z-m-s", type=float, default=0.5)
    parser.add_argument("--deployment-min-forward-m-s", type=float, default=2.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    candidate_path = (
        run_dir / "best_teacher.json"
        if args.candidate is None
        else Path(args.candidate).expanduser().resolve()
    )
    contract_path = run_dir / "cem_contract.json"
    if not candidate_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("candidate or sibling CEM contract is missing")
    candidate_raw = candidate_path.read_bytes()
    candidate = json.loads(candidate_raw)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    candidate_schema = candidate.get("schema_version")
    if candidate_schema not in {
        "stage3_cem_teacher_candidate_v1",
        "stage3_cem_search_seed_v1",
        "stage3_cem_inline_cpu_quality_gate_v3",
    }:
        raise ValueError("candidate schema is incompatible")
    if candidate_schema == "stage3_cem_search_seed_v1" and (
        candidate.get("qualified_teacher") is not False
        or candidate.get("seed_role")
        not in {
            "unqualified_optimizer_mean",
            "unqualified_parameter_intervention",
            "unqualified_snapshot_candidate",
        }
    ):
        raise ValueError("search seed is not explicitly unqualified")
    inline_cpu_audit = candidate_schema == "stage3_cem_inline_cpu_quality_gate_v3"
    if inline_cpu_audit:
        if not args.engineering_demo_only or args.record_video is None:
            raise ValueError(
                "inline CPU audits may only be replayed with "
                "--engineering-demo-only and --record-video"
            )
        source_trace = Path(str(candidate.get("trace_path", ""))).expanduser().resolve()
        if (
            not source_trace.is_file()
            or candidate.get("trace_sha256") != _sha256(source_trace)
            or candidate.get("feed_fingerprint") != contract.get("feed_fingerprint")
            or not math.isclose(
                float(candidate.get("swing_phase_advance_s", math.nan)),
                float(contract.get("swing_phase_advance_s", math.nan)),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise ValueError("inline CPU audit is detached from its saved replay contract")
        parameter_values = candidate.get("candidate_parameters")
        if candidate.get("candidate_parameter_f32_sha256") != _parameter_sha256(
            parameter_values
        ):
            raise ValueError("inline CPU audit candidate parameter hash is invalid")
    else:
        if candidate.get("contract_sha256") != contract.get("contract_sha256"):
            raise ValueError("candidate is detached from the CEM contract")
        parameter_values = candidate.get("parameters")
    parameters = np.asarray(parameter_values, dtype=np.float32)
    expected_shape = (int(contract["parameter_count"]),)
    if parameters.shape != expected_shape or not np.isfinite(parameters).all():
        raise ValueError("candidate parameter vector is incompatible")
    iteration = int(
        candidate.get("iteration", candidate.get("source_iteration", -1))
    )
    parameter_sha = _parameter_sha256(parameter_values)

    paths = load_incoming_hit_spec(contract["spec"])
    source_phase_advance_s = float(
        contract.get(
            "swing_phase_advance_s",
            paths.stage3_direct.get("swing_phase_advance_s", 0.0),
        )
    )
    audited_phase_advance_s = (
        source_phase_advance_s
        if args.swing_phase_advance_s is None
        else float(args.swing_phase_advance_s)
    )
    if (
        not math.isfinite(audited_phase_advance_s)
        or not 0.0 <= audited_phase_advance_s <= 1.0
    ):
        raise ValueError("audited swing phase advance must be finite and lie in [0, 1]")
    if (
        audited_phase_advance_s != source_phase_advance_s
        and candidate_schema != "stage3_cem_search_seed_v1"
    ):
        raise ValueError("a teacher candidate cannot be detached from its source timing")
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    by_fingerprint = {
        feed_sample_fingerprint(sample): sample for sample in feed_artifact.bank
    }
    requested_feed_fingerprint = (
        str(contract["feed_fingerprint"])
        if args.feed_fingerprint is None
        else str(args.feed_fingerprint)
    )
    if (
        requested_feed_fingerprint != str(contract["feed_fingerprint"])
        and candidate_schema != "stage3_cem_search_seed_v1"
    ):
        raise ValueError("a teacher candidate cannot be detached from its source feed")
    feed = by_fingerprint.get(requested_feed_fingerprint)
    if feed is None:
        raise ValueError("requested feed is absent from the configured feed bank")

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    policy_contract = _policy_update_contract(paths, model)
    selected_indices = tuple(
        int(value) for value in policy_contract["trainable_action_indices"]
    )
    selected_names = tuple(
        str(value) for value in policy_contract["trainable_actuator_names"]
    )
    parameterization = str(contract["parameterization"])
    if parameterization == "anatomical_synergies":
        synergy_basis, _synergy_names = _anatomical_synergy_basis(selected_names)
    elif parameterization == "muscle_knots":
        synergy_basis = None
    else:
        raise ValueError("candidate parameterization is unsupported")

    checkpoint = Path(contract["source_checkpoint"]).expanduser().resolve()
    restored, metadata = _source_actor(checkpoint)
    base_policy = metadata.get("base_policy_artifact")
    if not base_policy:
        raise ValueError("source checkpoint has no frozen base policy binding")
    if _residual_scale_overrides(paths) or _residual_scale_schedule(paths):
        raise ValueError("CPU CEM audit requires constant inherited residual authority")
    correction_window = dict(policy_contract["correction_window"])

    if args.output is None:
        trace_path = _validated_trace_path(
            run_dir
            / f"intermediate_cpu_audit_iter{iteration}_paramsha{parameter_sha[:12]}.npz"
        )
    else:
        trace_path = _validated_trace_path(args.output)
    if trace_path.exists() or trace_path.with_suffix(".json").exists():
        raise FileExistsError(
            "refusing to overwrite an existing immutable CPU audit: "
            f"{trace_path}"
        )
    trace_report = _save_cpu_teacher_trace(
        path=trace_path,
        feed=feed,
        paths=paths,
        actor=restored.agent,
        obs_mean=np.asarray(restored.obs_rms.mean),
        obs_var=np.asarray(restored.obs_rms.var),
        parameters=parameters,
        selected_indices=selected_indices,
        physical_scales=np.asarray(contract["physical_scales"], dtype=np.float32),
        base_policy_artifact=Path(base_policy).expanduser().resolve(),
        residual_scale=float(paths.stage3_direct.get("residual_scale", 0.25)),
        open_s=float(correction_window["time_to_intercept_open_s"]),
        close_s=float(correction_window["time_to_intercept_close_s"]),
        smoothing_s=float(correction_window["smoothing_s"]),
        max_episode_steps=int(contract["max_episode_steps"]),
        time_knots=int(contract["time_knots"]),
        synergy_basis=synergy_basis,
        swing_phase_advance_s=audited_phase_advance_s,
        video_path=(
            None
            if args.record_video is None
            else Path(args.record_video).expanduser().resolve()
        ),
    )
    quality_contract = dict(contract.get("return_quality_search_margin", {}) or {})
    strict_cpu_quality = _summarize_cpu_quality_trace(
        trace_path,
        player_half_sign=int(args.player_half_sign),
        min_outgoing_z_m_s=float(quality_contract["min_outgoing_z_m_s"]),
        min_forward_m_s=float(quality_contract["min_forward_m_s"]),
        max_stringbed_height_deficit_m=float(
            contract["high_region_contact"]["max_stringbed_height_deficit_m"]
        ),
        max_hand_height_deficit_m=float(
            contract["high_region_contact"]["max_hand_height_deficit_m"]
        ),
        min_predicted_clearance_m=float(
            quality_contract["min_predicted_clearance_m"]
        ),
        min_return_direction_signed_score=float(
            quality_contract["min_return_direction_signed_score"]
        ),
        require_real_net_cross=bool(
            quality_contract.get("require_real_net_cross", False)
        ),
        real_net_cross_authoritative=bool(
            quality_contract.get("real_net_cross_authoritative", False)
        ),
        max_pre_event_velocity_delta_m_s=float(
            quality_contract["max_pre_event_velocity_delta_m_s"]
        ),
        max_event_settled_velocity_delta_m_s=float(
            quality_contract["max_event_settled_velocity_delta_m_s"]
        ),
    )

    with np.load(trace_path, allow_pickle=False) as payload:
        outgoing_velocity_semantics = str(
            np.asarray(payload["outgoing_velocity_semantics"]).item()
        )
        if outgoing_velocity_semantics != (
            "post_control_step_after_all_physics_substeps"
        ):
            raise ValueError("CPU audit trace has incompatible outgoing-velocity semantics")
        event_rebound_contact_semantics = str(
            np.asarray(payload["event_rebound_contact_semantics"]).item()
        )
        if event_rebound_contact_semantics != (
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ):
            raise ValueError("CPU audit trace permits a double-applied stringbed impact")
        event = np.asarray(payload["event_rebound"], dtype=bool)
        hit = np.asarray(payload["hit_event"], dtype=bool)
        fall = np.asarray(payload["body_fall"], dtype=bool)
        velocity = np.asarray(payload["shuttle_velocity"], dtype=np.float64)
        event_impulse_velocity_after = np.asarray(
            payload["event_impulse_velocity_after_world_m_s"],
            dtype=np.float64,
        )
        tti = np.asarray(payload["time_to_intercept_s"], dtype=np.float64)
        window = np.asarray(payload["correction_window"], dtype=np.float64)
        stringbed_height = np.asarray(
            payload["stringbed_position"], dtype=np.float64
        )[:, 2]
        hand_height = np.asarray(
            payload["right_arm_body_position_xyz_m"], dtype=np.float64
        )[:, -1, 2]
    event_indices = np.flatnonzero(event)
    event_index = int(event_indices[0]) if event_indices.size else None
    active = window > 0.05
    outgoing_velocity = None if event_index is None else velocity[event_index]
    event_impulse_outgoing_velocity = (
        None
        if event_index is None
        else event_impulse_velocity_after[event_index]
    )
    outgoing_z = None if outgoing_velocity is None else float(outgoing_velocity[2])
    outgoing_forward = (
        None
        if outgoing_velocity is None
        else float(-int(args.player_half_sign) * outgoing_velocity[0])
    )
    stringbed_deficit = (
        None
        if event_index is None or not active.any()
        else float(
            max(0.0, float(stringbed_height[active].max() - stringbed_height[event_index]))
        )
    )
    hand_deficit = (
        None
        if event_index is None or not active.any()
        else float(max(0.0, float(hand_height[active].max() - hand_height[event_index])))
    )
    search_margin = dict(contract.get("return_quality_search_margin", {}) or {})
    search_min_z = float(search_margin.get("min_outgoing_z_m_s", math.inf))
    search_min_forward = float(search_margin.get("min_forward_m_s", math.inf))
    high_contract = dict(contract.get("high_region_contact", {}) or {})
    max_stringbed_deficit = float(
        high_contract.get("max_stringbed_height_deficit_m", -math.inf)
    )
    max_hand_deficit = float(high_contract.get("max_hand_height_deficit_m", -math.inf))
    upright_event = bool(event_index is not None and not fall.any())
    high_region = bool(
        stringbed_deficit is not None
        and hand_deficit is not None
        and stringbed_deficit <= max_stringbed_deficit
        and hand_deficit <= max_hand_deficit
    )
    deployment_quality = bool(
        upright_event
        and high_region
        and outgoing_z is not None
        and outgoing_z >= float(args.deployment_min_outgoing_z_m_s)
        and outgoing_forward is not None
        and outgoing_forward >= float(args.deployment_min_forward_m_s)
    )
    search_quality = bool(
        upright_event
        and high_region
        and outgoing_z is not None
        and outgoing_z >= search_min_z
        and outgoing_forward is not None
        and outgoing_forward >= search_min_forward
    )
    current_parameter_sha = None
    if candidate_path.is_file():
        current = json.loads(candidate_path.read_text(encoding="utf-8"))
        current_parameter_sha = _parameter_sha256(
            current.get(
                "candidate_parameters"
                if current.get("schema_version")
                == "stage3_cem_inline_cpu_quality_gate_v3"
                else "parameters"
            )
        )
    report = {
        "schema_version": "stage3_cem_intermediate_cpu_quality_audit_v1",
        "engineering_demo_only": bool(args.engineering_demo_only),
        "promotable": not inline_cpu_audit,
        "claim_scope": (
            "legacy_visual_engineering_demo_only"
            if inline_cpu_audit
            else "intermediate_cpu_quality_audit"
        ),
        "strict_cpu_quality": strict_cpu_quality,
        "candidate_path": str(candidate_path),
        "candidate_schema_version": candidate_schema,
        "candidate_role": candidate.get("seed_role", "teacher_candidate"),
        "candidate_file_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "candidate_iteration": iteration,
        "candidate_parameter_sha256": parameter_sha,
        "candidate_changed_during_audit": current_parameter_sha != parameter_sha,
        "contract_path": str(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "source_feed_fingerprint": str(contract["feed_fingerprint"]),
        "audited_feed_fingerprint": requested_feed_fingerprint,
        "alternate_feed_for_unqualified_seed": bool(
            requested_feed_fingerprint != str(contract["feed_fingerprint"])
        ),
        "source_swing_phase_advance_s": source_phase_advance_s,
        "audited_swing_phase_advance_s": audited_phase_advance_s,
        "alternate_timing_for_unqualified_seed": bool(
            audited_phase_advance_s != source_phase_advance_s
        ),
        **trace_report,
        "hit_step": int(np.flatnonzero(hit)[0]) if hit.any() else None,
        "event_step": event_index,
        "event_tti_s": None if event_index is None else float(tti[event_index]),
        "outgoing_velocity_xyz_m_s": (
            None if outgoing_velocity is None else outgoing_velocity.tolist()
        ),
        "outgoing_velocity_semantics": outgoing_velocity_semantics,
        "event_rebound_contact_semantics": event_rebound_contact_semantics,
        "event_impulse_velocity_after_xyz_m_s": (
            None
            if event_impulse_outgoing_velocity is None
            else event_impulse_outgoing_velocity.tolist()
        ),
        "outgoing_forward_m_s": outgoing_forward,
        "outgoing_z_m_s": outgoing_z,
        "fall_step": int(np.flatnonzero(fall)[0]) if fall.any() else None,
        "stringbed_height_deficit_m": stringbed_deficit,
        "hand_height_deficit_m": hand_deficit,
        "high_region_contact": high_region,
        "deployment_quality_passed": deployment_quality,
        "search_margin_quality_passed": search_quality,
        "deployment_gate": {
            "min_outgoing_z_m_s": float(args.deployment_min_outgoing_z_m_s),
            "min_forward_m_s": float(args.deployment_min_forward_m_s),
        },
        "search_margin_gate": {
            "min_outgoing_z_m_s": search_min_z,
            "min_forward_m_s": search_min_forward,
        },
    }
    report_path = trace_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
