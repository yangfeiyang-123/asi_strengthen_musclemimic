"""ForehandClear wrapper for iterative DAgger distillation."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForehandClear iterative DAgger loop.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--config-name", default="config_specific_task/conf_fullbody_badminton_student_gmr")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix-teacher-action-prob", type=float, default=0.1)
    parser.add_argument("--value-distill-weight", type=float, default=0.1)
    parser.add_argument("--gaussian-kl-weight", type=float, default=0.0)
    parser.add_argument("--save-reference-features", action="store_true", default=False)
    parser.add_argument("--include-reference-phase", action="store_true", default=False)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--freeze-run-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "fullbody.distill_run_dagger",
        "--teacher_ckpt",
        args.teacher_path,
        "--initial_student_ckpt",
        args.student_path,
        "--student_config",
        args.config_name,
        "--dataset_dir",
        args.dataset_dir,
        "--output_dir",
        args.output_dir,
        "--num_iters",
        str(args.num_iters),
        "--num_envs",
        str(args.num_envs),
        "--num_steps",
        str(args.num_steps),
        "--train_steps",
        str(args.train_steps),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--mix_teacher_action_prob",
        str(args.mix_teacher_action_prob),
        "--value_distill_weight",
        str(args.value_distill_weight),
        "--gaussian_kl_weight",
        str(args.gaussian_kl_weight),
        "--split",
        args.split,
    ]
    if args.save_reference_features:
        cmd.append("--save_reference_features")
    if args.include_reference_phase:
        cmd.append("--include_reference_phase")
    cmd.append("--freeze_run_stats" if args.freeze_run_stats else "--no-freeze_run_stats")
    if args.dry_run:
        cmd.append("--dry_run")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
