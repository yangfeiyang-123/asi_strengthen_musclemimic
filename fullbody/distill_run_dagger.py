"""Run an iterative ForehandClear DAgger distillation loop."""

from __future__ import annotations

import argparse
import os

from musclemimic.distill.dagger_loop import DaggerLoopConfig, run_dagger_loop


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iterative DAgger collection and BC retraining.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument(
        "--teacher_promotion_manifest",
        "--teacher-promotion-manifest",
        dest="teacher_promotion_manifest",
        default=None,
    )
    parser.add_argument(
        "--test_only_allow_unpromoted_teacher",
        "--test-only-allow-unpromoted-teacher",
        dest="test_only_allow_unpromoted_teacher",
        action="store_true",
        default=False,
    )
    parser.add_argument("--initial_student_ckpt", required=True)
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num_transitions",
        type=int,
        default=None,
        help="Total DAgger samples per iteration (default: 500,000).",
    )
    budget.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Legacy vector-step budget; samples=steps*num_envs.",
    )
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
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion_field", default="motion_uid")
    parser.add_argument("--strict_motion_identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume_dataset",
        "--resume-dataset",
        dest="resume_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run_uid", "--run-uid", dest="run_uid", default=None)
    parser.add_argument(
        "--physical_gpu",
        default=None,
        help=(
            "Physical GPU index inherited by each canonical BC retrain; defaults "
            "to MM_CUDA_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES from the outer launcher."
        ),
    )
    parser.add_argument(
        "--jax_cache_key_prefix",
        default=None,
        help="Stable prefix; each DAgger BC iteration receives its own suffix.",
    )
    parser.add_argument(
        "--train_log_dir",
        default=None,
        help="Directory for append-only per-iteration canonical launcher logs.",
    )
    parser.add_argument(
        "--source_train_dataset_dir",
        default=None,
        help="Immutable shared physical train dataset from which direct data was derived.",
    )
    parser.add_argument(
        "--source_val_dataset_dir",
        default=None,
        help="Immutable shared physical validation dataset from which direct data was derived.",
    )
    parser.add_argument(
        "--dataset_derivation_manifest",
        default=None,
        help="Seed-specific direct_dataset_derivation.json; required in production.",
    )
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
            num_transitions=(
                500_000 if args.num_transitions is None else args.num_transitions
            ),
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
            motion_path=args.motion_path,
            seed=args.seed,
            motion_field=args.motion_field,
            strict_motion_identity=bool(args.strict_motion_identity),
            resume_dataset=bool(args.resume_dataset),
            run_uid=args.run_uid,
            teacher_promotion_manifest=args.teacher_promotion_manifest,
            test_only_allow_unpromoted_teacher=bool(
                args.test_only_allow_unpromoted_teacher
            ),
            physical_gpu=(
                args.physical_gpu
                or os.environ.get("MM_CUDA_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
            jax_cache_key_prefix=(
                args.jax_cache_key_prefix
                or os.environ.get("MUSCLEMIMIC_JAX_CACHE_KEY")
            ),
            train_log_dir=args.train_log_dir,
            source_train_dataset_dir=args.source_train_dataset_dir,
            source_val_dataset_dir=args.source_val_dataset_dir,
            dataset_derivation_manifest=args.dataset_derivation_manifest,
        ),
        dry_run=args.dry_run,
    )
    print(f"dagger_loop_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
