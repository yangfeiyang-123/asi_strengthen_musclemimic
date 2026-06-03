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
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    if args.motion_path:
        cmd.append("--motion_path")
        cmd.extend(args.motion_path)
    if args.deterministic:
        cmd.append("--deterministic")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
