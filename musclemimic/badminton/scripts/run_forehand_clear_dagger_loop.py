"""ForehandClear wrapper for iterative DAgger distillation."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForehandClear iterative DAgger loop.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--teacher-promotion-manifest", default=None)
    parser.add_argument(
        "--test-only-allow-unpromoted-teacher",
        action="store_true",
        default=False,
    )
    parser.add_argument("--student-path", required=True)
    parser.add_argument(
        "--config-name",
        default="config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num-transitions",
        type=int,
        default=None,
        help="Exact total DAgger samples per iteration (default: 500,000).",
    )
    budget.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Legacy vector-step count per iteration; samples = num_steps * num_envs.",
    )
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
    parser.add_argument("--run-uid", default=None)
    args = parser.parse_args()
    if args.teacher_promotion_manifest is None and not args.test_only_allow_unpromoted_teacher:
        parser.error("--teacher-promotion-manifest is required for production DAgger loop")

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
    if args.teacher_promotion_manifest is not None:
        cmd.extend(
            ["--teacher-promotion-manifest", args.teacher_promotion_manifest]
        )
    else:
        cmd.append("--test-only-allow-unpromoted-teacher")
    if args.run_uid is not None:
        cmd.extend(["--run-uid", args.run_uid])
    if args.num_steps is not None:
        cmd.extend(["--num_steps", str(args.num_steps)])
    else:
        transitions = 500_000 if args.num_transitions is None else args.num_transitions
        cmd.extend(["--num_transitions", str(transitions)])
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
