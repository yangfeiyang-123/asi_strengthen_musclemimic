"""ForehandClear wrapper for DAgger student-rollout relabel collection."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ForehandClear DAgger correction shards.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--teacher-promotion-manifest", default=None)
    parser.add_argument(
        "--test-only-allow-unpromoted-teacher",
        action="store_true",
        default=False,
    )
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num-transitions",
        type=int,
        default=None,
        help="Exact total DAgger samples across vector environments (default: 500,000).",
    )
    budget.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Legacy vector-step count; samples = num_steps * num_envs.",
    )
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--motion-group", default=None)
    parser.add_argument("--traj-index", type=int, default=None)
    parser.add_argument("--traj-start-step", type=int, default=None)
    parser.add_argument("--mix-teacher-action-prob", type=float, default=0.0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--dagger-iteration", type=int, required=True)
    parser.add_argument("--resume-dataset", action="store_true", required=True)
    parser.add_argument("--run-uid", default=None)
    parser.add_argument("--save-full-obs", action="store_true", default=False)
    parser.add_argument("--save-reference-features", action="store_true", default=False)
    parser.add_argument("--include-reference-phase", action="store_true", default=False)
    parser.add_argument("--freeze-run-stats", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.teacher_promotion_manifest is None and not args.test_only_allow_unpromoted_teacher:
        parser.error("--teacher-promotion-manifest is required for production DAgger")

    cmd = [
        sys.executable,
        "-m",
        "fullbody.distill_collect_dagger",
        "--teacher_ckpt",
        args.teacher_path,
        "--student_ckpt",
        args.student_path,
        "--output_dir",
        args.output_dir,
        "--num_envs",
        str(args.num_envs),
        "--shard_size",
        str(args.shard_size),
        "--seed",
        str(args.seed),
        "--mix_teacher_action_prob",
        str(args.mix_teacher_action_prob),
        "--split",
        args.split,
        "--dagger-iteration",
        str(args.dagger_iteration),
        "--resume-dataset",
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
    if args.motion_path:
        cmd.append("--motion_path")
        cmd.extend(args.motion_path)
    if args.motion_group:
        cmd.extend(["--motion_group", args.motion_group])
    if args.traj_index is not None:
        cmd.extend(["--traj_index", str(args.traj_index)])
    if args.traj_start_step is not None:
        cmd.extend(["--traj_start_step", str(args.traj_start_step)])
    if args.save_full_obs:
        cmd.append("--save_full_obs")
    if args.save_reference_features:
        cmd.append("--save_reference_features")
    if args.include_reference_phase:
        cmd.append("--include_reference_phase")
    cmd.append("--freeze_run_stats" if args.freeze_run_stats else "--no-freeze_run_stats")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
