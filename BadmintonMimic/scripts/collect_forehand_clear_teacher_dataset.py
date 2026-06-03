"""ForehandClear wrapper for collecting teacher distillation shards."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ForehandClear teacher dataset for student distillation.")
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--config-name", default="config_specific_task/conf_fullbody_badminton_gmr")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=200_000)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--wandb", choices=["disabled", "online"], default="disabled")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--deterministic-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-obs", action="store_true", default=False)
    parser.add_argument("--freeze-run-stats", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "fullbody/distill_collect.py",
        "--teacher_ckpt",
        args.teacher_path,
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
        "--split",
        args.split,
    ]
    if args.deterministic_teacher:
        cmd.append("--deterministic_teacher")
    if args.save_full_obs:
        cmd.append("--save_full_obs")
    cmd.append("--freeze_run_stats" if args.freeze_run_stats else "--no-freeze_run_stats")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
