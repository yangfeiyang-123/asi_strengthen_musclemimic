"""Run and compare real clean/perturbed Stage-1R held-out rollouts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.promotion_artifact import checkpoint_identity
from musclemimic.badminton.stage1r_artifact import (
    LEGACY_EVIDENCE_KIND,
    SOURCE_SCHEMA_VERSION,
    VERIFIED_EVIDENCE_KIND,
    build_evaluation_contract,
    build_verified_report,
)
from musclemimic.utils.finger_isolation import PairedMetricRule, compare_paired_metrics

RULES = (
    PairedMetricRule("body_site_error", lower_is_better=True, max_relative_degradation=0.05),
    PairedMetricRule("right_hand_site_error", lower_is_better=True, max_relative_degradation=0.05),
    PairedMetricRule("racket_head_position_error", lower_is_better=True, max_relative_degradation=0.05),
    PairedMetricRule("racket_head_rotation_error", lower_is_better=True, max_relative_degradation=0.05),
    PairedMetricRule("early_termination", lower_is_better=True, max_absolute_degradation=0.02),
)

ROLLOUT_METRIC_MAP = {
    "body_site_error": "val_err_rpos",
    "right_hand_site_error": "val_err_right_hand_pos",
    "racket_head_position_error": "val_err_racket_pos",
    "racket_head_rotation_error": "val_err_racket_rot",
    "early_termination": "val_early_termination_rate",
}
SPIKE_METRIC_MAP = {
    "root": "val_max_err_root_xyz",
    "right_hand": "val_max_err_right_hand_pos",
    "racket_position": "val_max_err_racket_pos",
    "racket_rotation": "val_max_err_racket_rot",
}


def compare_finger_robustness(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    clean_metrics = dict(clean.get("metrics", clean))
    perturbed_metrics = dict(perturbed.get("metrics", perturbed))
    report = compare_paired_metrics(
        clean_metrics,
        perturbed_metrics,
        RULES,
        clean_seeds=clean.get("seeds"),
        perturbed_seeds=perturbed.get("seeds"),
    )
    spike_count, spike_details = _new_spike_count(clean_metrics, perturbed_metrics)
    payload = {
        "schema_version": "stage1r_paired_robustness_v2",
        "evidence_kind": LEGACY_EVIDENCE_KIND,
        "production_eligible": False,
        **asdict(report),
        "clean_provenance": dict(clean.get("provenance", {}) or {}),
        "perturbed_provenance": dict(perturbed.get("provenance", {}) or {}),
        "new_root_hand_racket_spike_count": spike_count,
        "spike_checks": spike_details,
    }
    perturbation_scale = payload["perturbed_provenance"].get(
        "finger_qpos_perturb_scale"
    )
    payload["finger_qpos_perturb_scale"] = (
        None if perturbation_scale is None else float(perturbation_scale)
    )
    payload["passed"] = bool(report.passed and spike_count == 0)
    return payload


def _new_spike_count(
    clean_metrics: dict[str, Any],
    perturbed_metrics: dict[str, Any],
    *,
    relative_tolerance: float = 0.05,
    absolute_epsilon: float = 1e-6,
) -> tuple[int, dict[str, Any]]:
    count = 0
    details: dict[str, Any] = {}
    for label, metric_name in SPIKE_METRIC_MAP.items():
        if metric_name not in clean_metrics or metric_name not in perturbed_metrics:
            raise KeyError(f"paired spike metric {metric_name!r} is missing")
        clean = np.asarray(clean_metrics[metric_name], dtype=np.float64)
        perturbed = np.asarray(perturbed_metrics[metric_name], dtype=np.float64)
        if clean.ndim != 1 or clean.shape != perturbed.shape or clean.size == 0:
            raise ValueError(f"paired spike metric {metric_name!r} must have matched non-empty 1D arrays")
        if not np.isfinite(clean).all() or not np.isfinite(perturbed).all():
            raise ValueError(f"paired spike metric {metric_name!r} contains non-finite values")
        limit = clean * (1.0 + float(relative_tolerance)) + float(absolute_epsilon)
        new_spike = perturbed > limit
        metric_count = int(np.sum(new_spike))
        count += metric_count
        details[label] = {
            "metric": metric_name,
            "new_spike_count": metric_count,
            "clean_max": float(np.max(clean)),
            "perturbed_max": float(np.max(perturbed)),
            "relative_tolerance": float(relative_tolerance),
        }
    return count, details


def collect_paired_rollouts(
    *,
    checkpoint: str,
    motion_paths: list[str],
    seeds: list[int],
    perturb_qpos_scale: float,
    perturb_qvel_scale: float,
    metrics_envs: int,
    metrics_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_contract = build_evaluation_contract(
        motion_paths=motion_paths,
        seeds=seeds,
        perturb_qpos_scale=perturb_qpos_scale,
        perturb_qvel_scale=perturb_qvel_scale,
        metrics_envs=metrics_envs,
        metrics_steps=metrics_steps,
    )
    identity = checkpoint_identity(checkpoint)
    clean_rows = []
    perturbed_rows = []
    for seed in seeds:
        clean_rows.append(
            _run_eval_once(
                checkpoint=checkpoint,
                motion_paths=motion_paths,
                seed=seed,
                qpos_scale=0.0,
                qvel_scale=0.0,
                metrics_envs=metrics_envs,
                metrics_steps=metrics_steps,
            )
        )
        perturbed_rows.append(
            _run_eval_once(
                checkpoint=checkpoint,
                motion_paths=motion_paths,
                seed=seed,
                qpos_scale=perturb_qpos_scale,
                qvel_scale=perturb_qvel_scale,
                metrics_envs=metrics_envs,
                metrics_steps=metrics_steps,
            )
        )
    provenance = {
        "checkpoint": identity["checkpoint_path"],
        "checkpoint_identity": identity,
        "motion_paths": list(motion_paths),
        "heldout_motion_identity": evaluation_contract["heldout_motion_identity"],
        "deterministic_policy": True,
        "evaluate_all": True,
        "finger_perturb_rng_mode": "fold_in",
        "finger_perturb_side": "right",
        "metrics_envs": int(metrics_envs),
        "metrics_steps": int(metrics_steps),
    }
    return (
        _rows_to_payload(
            clean_rows,
            seeds,
            provenance
            | {
                "finger_qpos_perturb_scale": 0.0,
                "finger_qvel_perturb_scale": 0.0,
            },
            condition="clean",
            checkpoint_identity_payload=identity,
            evaluation_contract=evaluation_contract,
        ),
        _rows_to_payload(
            perturbed_rows,
            seeds,
            provenance
            | {
                "finger_qpos_perturb_scale": float(perturb_qpos_scale),
                "finger_qvel_perturb_scale": float(perturb_qvel_scale),
            },
            condition="perturbed",
            checkpoint_identity_payload=identity,
            evaluation_contract=evaluation_contract,
        ),
    )


def _rows_to_payload(
    rows: list[dict[str, float]],
    seeds: list[int],
    provenance: dict[str, Any],
    *,
    condition: str,
    checkpoint_identity_payload: dict[str, Any],
    evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        output_name: [float(row[source_name]) for row in rows]
        for output_name, source_name in ROLLOUT_METRIC_MAP.items()
    }
    metrics.update(
        {
            source_name: [float(row[source_name]) for row in rows]
            for source_name in SPIKE_METRIC_MAP.values()
        }
    )
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "evidence_kind": VERIFIED_EVIDENCE_KIND,
        "condition": condition,
        "checkpoint_identity": dict(checkpoint_identity_payload),
        "evaluation_contract": dict(evaluation_contract),
        "seeds": list(seeds),
        "metrics": metrics,
        "rollouts": rows,
        "provenance": provenance,
    }


def _run_eval_once(
    *,
    checkpoint: str,
    motion_paths: list[str],
    seed: int,
    qpos_scale: float,
    qvel_scale: float,
    metrics_envs: int,
    metrics_steps: int,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="stage1r_eval_") as tmpdir:
        output = Path(tmpdir) / "metrics.json"
        command = [
            sys.executable,
            "-m",
            "fullbody.eval",
            "--path",
            checkpoint,
            "--evaluate_all",
            "--metrics",
            "--metrics_only",
            "--metrics_deterministic",
            "--metrics_envs",
            str(int(metrics_envs)),
            "--metrics_steps",
            str(int(metrics_steps)),
            "--eval_seed",
            str(int(seed)),
            "--finger_perturb_qpos_scale",
            str(float(qpos_scale)),
            "--finger_perturb_qvel_scale",
            str(float(qvel_scale)),
            "--finger_perturb_side",
            "right",
            "--metrics_output_json",
            str(output),
            "--motion_path",
            *motion_paths,
        ]
        subprocess.run(command, check=True, text=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
    required = set(ROLLOUT_METRIC_MAP.values()) | set(SPIKE_METRIC_MAP.values())
    missing = sorted(required - set(payload))
    nonfinite = sorted(key for key in required if key in payload and not np.isfinite(float(payload[key])))
    if missing or nonfinite:
        raise RuntimeError(f"invalid Stage-1R eval metrics: missing={missing}, nonfinite={nonfinite}")
    return {str(key): float(value) for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--clean", help="Precomputed clean rollout JSON (legacy/offline mode).")
    parser.add_argument("--perturbed", help="Precomputed perturbed rollout JSON.")
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--perturb_qpos_scale", type=float, default=0.03)
    parser.add_argument("--perturb_qvel_scale", type=float, default=0.0)
    parser.add_argument("--metrics_envs", type=int, default=5)
    parser.add_argument("--metrics_steps", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.checkpoint:
        if not args.motion_path:
            parser.error("--checkpoint mode requires explicit held-out --motion_path")
        clean, perturbed = collect_paired_rollouts(
            checkpoint=args.checkpoint,
            motion_paths=args.motion_path,
            seeds=args.seeds,
            perturb_qpos_scale=args.perturb_qpos_scale,
            perturb_qvel_scale=args.perturb_qvel_scale,
            metrics_envs=args.metrics_envs,
            metrics_steps=args.metrics_steps,
        )
        output_parent = Path(args.output).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        (output_parent / "clean_rollouts.json").write_text(
            json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_parent / "perturbed_rollouts.json").write_text(
            json.dumps(perturbed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        if not args.perturbed:
            parser.error("--clean offline mode also requires --perturbed")
        clean = json.loads(Path(args.clean).read_text(encoding="utf-8"))
        perturbed = json.loads(Path(args.perturbed).read_text(encoding="utf-8"))
    result = compare_finger_robustness(clean, perturbed)
    if args.checkpoint:
        result = build_verified_report(
            result,
            checkpoint=args.checkpoint,
            evaluation_contract=clean["evaluation_contract"],
            clean_source_path=output_parent / "clean_rollouts.json",
            perturbed_source_path=output_parent / "perturbed_rollouts.json",
        )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if args.require_pass and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
