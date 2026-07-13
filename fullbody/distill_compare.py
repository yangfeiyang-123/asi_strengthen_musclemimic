"""Compare teacher/student checkpoints with fullbody eval metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musclemimic.distill.eval_student import (
    DEFAULT_DISTILL_ACCEPTANCE_THRESHOLDS,
    DistillAcceptanceThresholds,
    load_convergence_evidence,
    run_checkpoint_temporal_audit,
    run_eval_metrics,
    write_acceptance_outputs,
    write_comparison_outputs,
    write_summary_report,
    write_temporal_audit_outputs,
)
from musclemimic.distill.motion_identity import normalize_motion_path, stable_motion_uid
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    file_sha256,
    validate_dataset_manifest,
    validate_direct_acceptance_record,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare distillation teacher/student metrics.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--student_dagger_ckpt", default=None)
    parser.add_argument("--student_ppo_ckpt", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Direct-distillation dataset containing strict held-out val sequence shards.",
    )
    parser.add_argument(
        "--convergence_metrics",
        default=None,
        help="distill_metadata.json/convergence.json from the final supervised BC/DAgger stage.",
    )
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--metrics_envs", type=int, default=20)
    parser.add_argument("--metrics_steps", type=int, default=500)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic held-out evaluation (required for promotion).",
    )
    parser.add_argument(
        "--promotion_policy",
        choices=("student_bc", "student_bc_dagger", "student_bc_ppo"),
        default=None,
        help="Policy whose acceptance result controls --require_pass.",
    )
    defaults = DEFAULT_DISTILL_ACCEPTANCE_THRESHOLDS
    parser.add_argument("--min_return_ratio", type=float, default=defaults.min_return_ratio)
    parser.add_argument(
        "--target_return_ratio", type=float, default=defaults.target_return_ratio
    )
    parser.add_argument("--min_completion_ratio", type=float, default=defaults.min_completion_ratio)
    parser.add_argument(
        "--max_early_termination_delta",
        type=float,
        default=defaults.max_early_termination_delta,
    )
    parser.add_argument(
        "--max_tracking_error_relative_degradation",
        type=float,
        default=defaults.max_tracking_error_relative_degradation,
    )
    parser.add_argument("--plateau_min_points", type=int, default=defaults.plateau_min_points)
    parser.add_argument(
        "--plateau_window_points", type=int, default=defaults.plateau_window_points
    )
    parser.add_argument(
        "--max_plateau_normalized_abs_slope",
        type=float,
        default=defaults.max_plateau_normalized_abs_slope,
    )
    parser.add_argument(
        "--max_plateau_normalized_span",
        type=float,
        default=defaults.max_plateau_normalized_span,
    )
    parser.add_argument(
        "--temporal_search_max_lag_steps",
        type=int,
        default=defaults.temporal_search_max_lag_steps,
    )
    parser.add_argument(
        "--max_abs_temporal_best_lag_steps",
        type=int,
        default=defaults.max_abs_temporal_best_lag_steps,
    )
    parser.add_argument(
        "--max_temporal_lag_mse_improvement_fraction",
        type=float,
        default=defaults.max_temporal_lag_mse_improvement_fraction,
    )
    parser.add_argument(
        "--min_temporal_sequences", type=int, default=defaults.min_temporal_sequences
    )
    parser.add_argument("--temporal_batch_size", type=int, default=4096)
    parser.add_argument("--require_pass", action="store_true", default=False)
    args = parser.parse_args()
    if args.require_pass and not args.deterministic:
        parser.error("--require_pass requires deterministic held-out evaluation")
    if args.require_pass and not args.motion_path:
        parser.error("--require_pass requires explicit held-out --motion_path inputs")
    if args.require_pass and not args.dataset_dir:
        parser.error("--require_pass requires --dataset_dir for held-out temporal audit")
    if args.require_pass and not args.convergence_metrics:
        parser.error("--require_pass requires --convergence_metrics from BC/DAgger")

    thresholds = DistillAcceptanceThresholds(
        min_return_ratio=float(args.min_return_ratio),
        target_return_ratio=float(args.target_return_ratio),
        min_completion_ratio=float(args.min_completion_ratio),
        max_early_termination_delta=float(args.max_early_termination_delta),
        max_tracking_error_relative_degradation=float(
            args.max_tracking_error_relative_degradation
        ),
        plateau_min_points=int(args.plateau_min_points),
        plateau_window_points=int(args.plateau_window_points),
        max_plateau_normalized_abs_slope=float(
            args.max_plateau_normalized_abs_slope
        ),
        max_plateau_normalized_span=float(args.max_plateau_normalized_span),
        temporal_search_max_lag_steps=int(args.temporal_search_max_lag_steps),
        max_abs_temporal_best_lag_steps=int(
            args.max_abs_temporal_best_lag_steps
        ),
        max_temporal_lag_mse_improvement_fraction=float(
            args.max_temporal_lag_mse_improvement_fraction
        ),
        min_temporal_sequences=int(args.min_temporal_sequences),
    )

    eval_kwargs = {
        "motion_paths": args.motion_path,
        "metrics_envs": args.metrics_envs,
        "metrics_steps": args.metrics_steps,
        "eval_seed": args.eval_seed,
        "deterministic": bool(args.deterministic),
        "evaluate_all": True,
    }

    results = {
        "teacher": run_eval_metrics(
            args.teacher_ckpt,
            **eval_kwargs,
        ),
        "student_bc": run_eval_metrics(
            args.student_ckpt,
            **eval_kwargs,
        ),
    }
    if args.student_ppo_ckpt:
        results["student_bc_ppo"] = run_eval_metrics(
            args.student_ppo_ckpt,
            **eval_kwargs,
        )
    if args.student_dagger_ckpt:
        results["student_bc_dagger"] = run_eval_metrics(
            args.student_dagger_ckpt,
            **eval_kwargs,
        )

    promotion_policy = args.promotion_policy
    if promotion_policy is None:
        promotion_policy = (
            "student_bc_ppo"
            if args.student_ppo_ckpt
            else "student_bc_dagger"
            if args.student_dagger_ckpt
            else "student_bc"
        )
    checkpoint_by_policy = {
        "student_bc": args.student_ckpt,
        "student_bc_dagger": args.student_dagger_ckpt,
        "student_bc_ppo": args.student_ppo_ckpt,
    }
    temporal_audits = {}
    if args.dataset_dir:
        checkpoint = checkpoint_by_policy.get(promotion_policy)
        if not checkpoint:
            parser.error(
                f"promotion policy {promotion_policy!r} has no supplied checkpoint for temporal audit"
            )
        temporal_audits[promotion_policy] = run_checkpoint_temporal_audit(
            checkpoint,
            dataset_dir=args.dataset_dir,
            expected_motion_paths=args.motion_path,
            thresholds=thresholds,
            batch_size=args.temporal_batch_size,
        )
    temporal_path = write_temporal_audit_outputs(temporal_audits, args.output_dir)
    convergence = (
        None
        if args.convergence_metrics is None
        else load_convergence_evidence(args.convergence_metrics)
    )

    json_path, csv_path = write_comparison_outputs(results, args.output_dir)
    summary_path = write_summary_report(results, args.output_dir)
    acceptance_path = write_acceptance_outputs(
        results,
        args.output_dir,
        thresholds,
        convergence=convergence,
        temporal_audits=temporal_audits,
    )
    direct_evidence_path = None
    if args.require_pass:
        checkpoint_by_policy = {
            "student_bc": args.student_ckpt,
            "student_bc_dagger": args.student_dagger_ckpt,
            "student_bc_ppo": args.student_ppo_ckpt,
        }
        direct_evidence_path = _write_direct_promotion_evidence(
            output_dir=Path(args.output_dir),
            teacher_ckpt=args.teacher_ckpt,
            policy_checkpoints=checkpoint_by_policy,
            promotion_policy=promotion_policy,
            motion_paths=args.motion_path,
            deterministic=bool(args.deterministic),
            metrics_envs=int(args.metrics_envs),
            metrics_steps=int(args.metrics_steps),
            eval_seed=int(args.eval_seed),
            dataset_dir=args.dataset_dir,
            comparison_path=Path(json_path),
            acceptance_path=Path(acceptance_path),
            convergence_path=Path(args.convergence_metrics),
            temporal_path=Path(temporal_path),
        )
    print(f"comparison_metrics_json: {json_path}")
    print(f"comparison_table_csv: {csv_path}")
    print(f"summary_markdown: {summary_path}")
    print(f"acceptance_json: {acceptance_path}")
    print(f"temporal_audit_json: {temporal_path}")
    if direct_evidence_path is not None:
        print(f"direct_promotion_evidence_json: {direct_evidence_path}")
    for policy, metrics in results.items():
        print(f"\n[{policy}]")
        for key in sorted(metrics):
            print(f"{key}: {metrics[key]:.6f}")
    if args.require_pass:
        acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
        if promotion_policy not in acceptance:
            parser.error(
                f"promotion policy {promotion_policy!r} was not evaluated; "
                "supply its checkpoint argument"
            )
        failed = [] if acceptance[promotion_policy]["passed"] else [promotion_policy]
        if failed:
            print(f"distill promotion failed: {failed}")
            return 2
    return 0


def _write_direct_promotion_evidence(
    *,
    output_dir: Path,
    teacher_ckpt: str,
    policy_checkpoints: dict[str, str | None],
    promotion_policy: str,
    motion_paths: list[str],
    deterministic: bool,
    metrics_envs: int,
    metrics_steps: int,
    eval_seed: int,
    dataset_dir: str,
    comparison_path: Path,
    acceptance_path: Path,
    convergence_path: Path,
    temporal_path: Path,
) -> Path:
    """Bind direct acceptance to exact checkpoints, motions, and artifacts."""
    selected_checkpoint = policy_checkpoints.get(promotion_policy)
    if not selected_checkpoint:
        raise ValueError(f"promotion policy {promotion_policy!r} has no checkpoint")
    teacher = checkpoint_content_fingerprint(teacher_ckpt)
    student = checkpoint_content_fingerprint(selected_checkpoint)
    dataset = validate_dataset_manifest(
        dataset_dir,
        expected_teacher=teacher,
        require_promoted_teacher=True,
    )
    normalized = [normalize_motion_path(path) for path in motion_paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError("direct promotion held-out motion paths must be unique")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    selected_acceptance = acceptance.get(promotion_policy)
    validate_direct_acceptance_record(selected_acceptance)
    payload = {
        "schema_version": "direct_distill_promotion_evidence_v2",
        "promotion_policy": promotion_policy,
        "deterministic": bool(deterministic),
        "teacher_checkpoint": teacher,
        "student_checkpoint": student,
        "heldout": {
            "motion_paths": normalized,
            "motion_uids": [int(stable_motion_uid(path)) for path in normalized],
            "motion_set_fingerprint": canonical_json_sha256(normalized),
            "num_motions": len(normalized),
            "metrics_envs": int(metrics_envs),
            "metrics_steps": int(metrics_steps),
            "eval_seed": int(eval_seed),
        },
        "dataset_manifest_fingerprint": dataset["manifest_fingerprint"],
        "dataset_run_uid": dataset["run_uid"],
        "teacher_promotion": dataset["teacher_promotion"],
        "artifacts": {
            "comparison_metrics": {
                "path": str(comparison_path.resolve()),
                "sha256": file_sha256(comparison_path),
            },
            "acceptance": {
                "path": str(acceptance_path.resolve()),
                "sha256": file_sha256(acceptance_path),
            },
            "convergence": {
                "path": str(convergence_path.resolve()),
                "sha256": file_sha256(convergence_path),
            },
            "temporal_audit": {
                "path": str(temporal_path.resolve()),
                "sha256": file_sha256(temporal_path),
            },
        },
    }
    payload["evidence_fingerprint"] = canonical_json_sha256(payload)
    target = output_dir / "direct_promotion_evidence.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    raise SystemExit(main())
