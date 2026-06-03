"""ForehandClear wrapper for training a no-lookahead student with BC/KD."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ForehandClear student BC checkpoint.")
    parser.add_argument("--config-name", default="config_specific_task/conf_fullbody_badminton_student_gmr")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-distill-weight", type=float, default=0.1)
    parser.add_argument("--gaussian-kl-weight", type=float, default=0.0)
    parser.add_argument("--resume-student", default=None)
    parser.add_argument("--wandb", choices=["disabled", "online"], default="disabled")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "fullbody/distill_train_bc.py",
        "--dataset_dir",
        args.dataset_dir,
        "--student_config",
        args.config_name,
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--num_steps",
        str(args.num_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--value_distill_weight",
        str(args.value_distill_weight),
        "--gaussian_kl_weight",
        str(args.gaussian_kl_weight),
    ]
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
