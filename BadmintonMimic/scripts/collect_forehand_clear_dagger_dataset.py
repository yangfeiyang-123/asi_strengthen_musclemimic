"""ForehandClear wrapper for DAgger student-rollout relabel collection."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ForehandClear DAgger correction shards.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--motion-group", default=None)
    parser.add_argument("--traj-index", type=int, default=None)
    parser.add_argument("--traj-start-step", type=int, default=None)
    parser.add_argument("--mix-teacher-action-prob", type=float, default=0.0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-obs", action="store_true", default=False)
    parser.add_argument("--save-reference-features", action="store_true", default=False)
    parser.add_argument("--include-reference-phase", action="store_true", default=False)
    parser.add_argument("--freeze-run-stats", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
        "--num_steps",
        str(args.num_steps),
        "--shard_size",
        str(args.shard_size),
        "--seed",
        str(args.seed),
        "--mix_teacher_action_prob",
        str(args.mix_teacher_action_prob),
        "--split",
        args.split,
    ]
    if args.motion_path:
        cmd.append("--motion_path")
        cmd.extend(args.motion_path)
    if args.motion_group:
        cmd.extend(["--motion_group", args.motion_group])
    if args.traj_index is not None:
        cmd.extend(["--traj_index", str(args.traj_index)])
    if args.traj_start_step is not None:
        cmd.extend(["--traj_start_step", str(args.traj_start_step)])
    if args.append:
        cmd.append("--append")
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
