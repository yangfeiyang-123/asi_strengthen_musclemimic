"""ForehandClear wrapper for teacher-vs-student distillation evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ForehandClear teacher and student checkpoints.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--dagger-student-path", default=None)
    parser.add_argument("--ppo-student-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Strict direct-distillation dataset used for the held-out temporal audit.",
    )
    parser.add_argument(
        "--convergence-metrics",
        default=None,
        help="Final BC/DAgger convergence evidence used by production acceptance.",
    )
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--promotion-policy",
        choices=("student_bc", "student_bc_dagger", "student_bc_ppo"),
        default=None,
    )
    parser.add_argument("--require-pass", action="store_true", default=False)
    args = parser.parse_args()
    if args.require_pass and not args.dataset_dir:
        parser.error("--require-pass requires --dataset-dir")
    if args.require_pass and not args.convergence_metrics:
        parser.error("--require-pass requires --convergence-metrics")

    cmd = [
        sys.executable,
        "-m",
        "fullbody.distill_compare",
        "--teacher_ckpt",
        args.teacher_path,
        "--student_ckpt",
        args.student_path,
        "--output_dir",
        args.output_dir,
        "--metrics_envs",
        str(args.num_envs),
        "--metrics_steps",
        str(args.num_steps),
        "--eval_seed",
        str(args.seed),
    ]
    if args.dagger_student_path:
        cmd.extend(["--student_dagger_ckpt", args.dagger_student_path])
    if args.ppo_student_path:
        cmd.extend(["--student_ppo_ckpt", args.ppo_student_path])
    if args.dataset_dir:
        cmd.extend(["--dataset_dir", args.dataset_dir])
    if args.convergence_metrics:
        cmd.extend(["--convergence_metrics", args.convergence_metrics])
    if args.motion_path:
        cmd.append("--motion_path")
        cmd.extend(args.motion_path)
    if args.promotion_policy:
        cmd.extend(["--promotion_policy", args.promotion_policy])
    if args.deterministic:
        cmd.append("--deterministic")
    if args.require_pass:
        cmd.append("--require_pass")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
