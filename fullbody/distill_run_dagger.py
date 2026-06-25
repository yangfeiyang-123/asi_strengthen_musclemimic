"""Run an iterative ForehandClear DAgger distillation loop."""

from __future__ import annotations

import argparse

from musclemimic.distill.dagger_loop import DaggerLoopConfig, run_dagger_loop


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iterative DAgger collection and BC retraining.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--initial_student_ckpt", required=True)
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=50_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_distill_weight", type=float, default=0.1)
    parser.add_argument("--gaussian_kl_weight", type=float, default=0.0)
    parser.add_argument("--mix_teacher_action_prob", type=float, default=0.0)
    parser.add_argument("--save_reference_features", action="store_true", default=False)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument("--freeze_run_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    manifest = run_dagger_loop(
        DaggerLoopConfig(
            teacher_ckpt=args.teacher_ckpt,
            initial_student_ckpt=args.initial_student_ckpt,
            student_config=args.student_config,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_iters=args.num_iters,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            shard_size=args.shard_size,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            value_distill_weight=args.value_distill_weight,
            gaussian_kl_weight=args.gaussian_kl_weight,
            mix_teacher_action_prob=args.mix_teacher_action_prob,
            save_reference_features=bool(args.save_reference_features),
            include_reference_phase=bool(args.include_reference_phase),
            freeze_run_stats=bool(args.freeze_run_stats),
            split=args.split,
            seed=args.seed,
        ),
        dry_run=args.dry_run,
    )
    print(f"dagger_loop_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
