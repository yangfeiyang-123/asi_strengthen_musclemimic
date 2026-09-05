#!/usr/bin/env python3
"""Replay a frozen Stage-3 policy across phase advances and audit reach."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_hit_apex_alignment import analyze_apex_alignment
from optimize_single_feed_hit_mjx import (
    _anatomical_synergy_basis,
    _save_cpu_teacher_trace,
    _source_actor,
)
from environment.overall_environment.src.shuttle_feeder import feed_sample_fingerprint
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _ensure_feed_bank_artifact,
    _ensure_scene,
    _residual_scale_overrides,
    _residual_scale_schedule,
    load_incoming_hit_spec,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_advances(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not 0.0 <= item <= 1.2 for item in result):
        raise ValueError("phase advances must be a non-empty comma list in [0, 1.2]")
    if len(set(result)) != len(result):
        raise ValueError("phase advances must be unique")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cem-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--phase-advances",
        default="0.00,0.10,0.20,0.30,0.40,0.50,0.58,0.65,0.75",
    )
    parser.add_argument(
        "--trajectory",
        choices=("zero_correction", "best_teacher"),
        default="zero_correction",
    )
    parser.add_argument("--max-episode-steps", type=int, default=260)
    parser.add_argument("--stringbed-high-tolerance-m", type=float, default=0.08)
    parser.add_argument("--hand-high-tolerance-m", type=float, default=0.06)
    parser.add_argument("--max-downward-speed-m-s", type=float, default=1.0)
    args = parser.parse_args()

    source_report_path = Path(args.source_cem_report).expanduser().resolve()
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    contract = dict(source_report.get("contract", {}) or {})
    if contract.get("schema_version") != "stage3_single_feed_mjx_cem_v3":
        raise ValueError("source report does not contain a compatible CEM contract")

    spec = Path(str(contract["spec"])).expanduser().resolve()
    checkpoint = Path(str(contract["source_checkpoint"])).expanduser().resolve()
    if _sha256(checkpoint) != str(contract["source_checkpoint_sha256"]):
        raise ValueError("source checkpoint hash differs from the CEM contract")
    paths = load_incoming_hit_spec(spec)
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    requested_fingerprint = str(contract["feed_fingerprint"])
    by_fingerprint = {feed_sample_fingerprint(sample): sample for sample in feed_artifact.bank}
    if requested_fingerprint not in by_fingerprint:
        raise ValueError("source feed is absent from the runtime feed bank")
    feed = by_fingerprint[requested_fingerprint]

    actor, source_metadata = _source_actor(checkpoint)
    base_policy_value = source_metadata.get("base_policy_artifact")
    if not base_policy_value:
        raise ValueError("source checkpoint has no frozen base-policy artifact")
    base_policy = Path(base_policy_value).expanduser().resolve()
    if _residual_scale_overrides(paths) or _residual_scale_schedule(paths):
        raise ValueError("calibration requires a constant inherited residual scale")
    residual_scale = float(paths.stage3_direct.get("residual_scale", 0.25))

    teacher_trace_path = source_report_path.parent / "teacher_trajectory_mjx.npz"
    with np.load(teacher_trace_path, allow_pickle=False) as teacher_trace:
        selected_indices = tuple(
            int(value)
            for value in np.asarray(teacher_trace["selected_action_indices"]).tolist()
        )
    selected_names = tuple(str(value) for value in contract["selected_actuator_names"])
    if len(selected_indices) != len(selected_names):
        raise ValueError("source selected-index/name mappings have different lengths")
    physical_scales = np.asarray(contract["physical_scales"], dtype=np.float32)
    if physical_scales.shape != (len(selected_indices),):
        raise ValueError("source physical scales do not match selected actions")

    parameterization = str(contract["parameterization"])
    if parameterization == "anatomical_synergies":
        synergy_basis, _synergy_names = _anatomical_synergy_basis(selected_names)
        latent_size = int(synergy_basis.shape[0])
    elif parameterization == "muscle_knots":
        synergy_basis = None
        latent_size = len(selected_indices)
    else:
        raise ValueError(f"unsupported source parameterization: {parameterization}")
    time_knots = int(contract["time_knots"])
    parameter_count = time_knots * latent_size
    if args.trajectory == "best_teacher":
        teacher_path = source_report_path.parent / "best_teacher.json"
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        parameters = np.asarray(teacher["parameters"], dtype=np.float32)
        parameter_source_sha256 = _sha256(teacher_path)
    else:
        parameters = np.zeros((parameter_count,), dtype=np.float32)
        parameter_source_sha256 = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    if parameters.shape != (parameter_count,) or not np.isfinite(parameters).all():
        raise ValueError("trajectory parameters are incompatible with the source CEM contract")

    correction_window = dict(contract["correction_window"])
    swing_duration_s = float(paths.stage3_lab.get("swing_duration_s", 1.2))
    contact_phase = float(paths.stage3_lab.get("contact_phase", 0.76))
    advances = _parse_advances(args.phase_advances)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for advance in advances:
        tag = f"phase_advance_{advance:.3f}".replace(".", "p")
        trace_path = out_dir / f"{tag}.npz"
        replay = _save_cpu_teacher_trace(
            path=trace_path,
            feed=feed,
            paths=paths,
            actor=actor.agent,
            obs_mean=np.asarray(actor.obs_rms.mean),
            obs_var=np.asarray(actor.obs_rms.var),
            parameters=parameters,
            selected_indices=selected_indices,
            physical_scales=physical_scales,
            base_policy_artifact=base_policy,
            residual_scale=residual_scale,
            open_s=float(correction_window["time_to_intercept_open_s"]),
            close_s=float(correction_window["time_to_intercept_close_s"]),
            smoothing_s=float(correction_window["smoothing_s"]),
            max_episode_steps=int(args.max_episode_steps),
            time_knots=time_knots,
            synergy_basis=synergy_basis,
            swing_phase_advance_s=advance,
        )
        with np.load(trace_path, allow_pickle=False) as loaded:
            payload = {name: loaded[name] for name in loaded.files}
        alignment = analyze_apex_alignment(
            payload,
            swing_duration_s=swing_duration_s,
            configured_contact_phase=contact_phase,
            configured_phase_advance_s=advance,
            stringbed_high_tolerance_m=float(args.stringbed_high_tolerance_m),
            hand_high_tolerance_m=float(args.hand_high_tolerance_m),
            max_downward_speed_m_s=float(args.max_downward_speed_m_s),
        )
        alignment_path = out_dir / f"{tag}.alignment.json"
        alignment_path.write_text(
            json.dumps(alignment, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        recommendation = alignment["recommended_high_contact"]
        nominal = alignment["nominal_intercept"]
        row = {
            "phase_advance_s": advance,
            "trace": replay,
            "alignment_report": str(alignment_path),
            "recommended_high_contact": recommendation,
            "nominal_intercept": nominal,
            "absolute_recommended_timing_error_s": abs(
                float(recommendation["time_to_intercept_s"])
            ),
            "nominal_stringbed_height_deficit_m": float(
                nominal["stringbed_height_deficit_m"]
            ),
            "nominal_hand_height_deficit_m": float(
                nominal.get("hand_height_deficit_m", nominal["stringbed_height_deficit_m"])
            ),
            "nominal_downward_speed_m_s": max(
                0.0, -float(nominal["stringbed_velocity_xyz_m_s"][2])
            ),
        }
        rows.append(row)
        print(
            "[high-point] "
            f"advance={advance:.3f} "
            f"recommended_tti={recommendation['time_to_intercept_s']:+.3f}s "
            f"point={np.round(recommendation['stringbed_position_xyz_m'], 3).tolist()} "
            f"nominal_height_deficit={row['nominal_stringbed_height_deficit_m']:.3f}m",
            flush=True,
        )

    # The user's biomechanical constraint is explicit: contact at the highest
    # hand/arm point.  Treat hand-height alignment as the primary criterion;
    # forward racket speed is left to the independent wrist/forearm teacher.
    best = min(
        rows,
        key=lambda item: (
            item["nominal_hand_height_deficit_m"],
            item["nominal_stringbed_height_deficit_m"],
            item["nominal_downward_speed_m_s"],
            item["absolute_recommended_timing_error_s"],
        ),
    )
    summary = {
        "schema_version": "stage3_high_point_phase_calibration_v1",
        "source": {
            "cem_report": str(source_report_path),
            "cem_report_sha256": _sha256(source_report_path),
            "spec": str(spec),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "feed_fingerprint": requested_fingerprint,
            "trajectory": args.trajectory,
            "parameter_source_sha256": parameter_source_sha256,
        },
        "timing_contract": {
            "swing_duration_s": swing_duration_s,
            "contact_phase": contact_phase,
            "phase_advances_s": list(advances),
            "max_episode_steps": int(args.max_episode_steps),
        },
        "best_phase_advance": best,
        "sweeps": rows,
    }
    summary_path = out_dir / "high_point_calibration.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "best": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
