"""Compare teacher/student checkpoints with fullbody eval metrics."""

from __future__ import annotations

import argparse

from musclemimic.distill.eval_student import run_eval_metrics, write_comparison_outputs, write_summary_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare distillation teacher/student metrics.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--student_dagger_ckpt", default=None)
    parser.add_argument("--student_ppo_ckpt", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--metrics_envs", type=int, default=20)
    parser.add_argument("--metrics_steps", type=int, default=500)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    results = {
        "teacher": run_eval_metrics(
            args.teacher_ckpt,
            motion_paths=args.motion_path,
            metrics_envs=args.metrics_envs,
            metrics_steps=args.metrics_steps,
            eval_seed=args.eval_seed,
            deterministic=args.deterministic,
        ),
        "student_bc": run_eval_metrics(
            args.student_ckpt,
            motion_paths=args.motion_path,
            metrics_envs=args.metrics_envs,
            metrics_steps=args.metrics_steps,
            eval_seed=args.eval_seed,
            deterministic=args.deterministic,
        ),
    }
    if args.student_ppo_ckpt:
        results["student_bc_ppo"] = run_eval_metrics(
            args.student_ppo_ckpt,
            motion_paths=args.motion_path,
            metrics_envs=args.metrics_envs,
            metrics_steps=args.metrics_steps,
            eval_seed=args.eval_seed,
            deterministic=args.deterministic,
        )
    if args.student_dagger_ckpt:
        results["student_bc_dagger"] = run_eval_metrics(
            args.student_dagger_ckpt,
            motion_paths=args.motion_path,
            metrics_envs=args.metrics_envs,
            metrics_steps=args.metrics_steps,
            eval_seed=args.eval_seed,
            deterministic=args.deterministic,
        )

    json_path, csv_path = write_comparison_outputs(results, args.output_dir)
    summary_path = write_summary_report(results, args.output_dir)
    print(f"comparison_metrics_json: {json_path}")
    print(f"comparison_table_csv: {csv_path}")
    print(f"summary_markdown: {summary_path}")
    for policy, metrics in results.items():
        print(f"\n[{policy}]")
        for key in sorted(metrics):
            print(f"{key}: {metrics[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
