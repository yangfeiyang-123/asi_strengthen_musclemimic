"""Command line entrypoint for ForehandClear student behavior cloning."""

from __future__ import annotations

import argparse

from musclemimic.distill.train_bc import load_config_for_bc, train_bc
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env


def main() -> int:
    reexec_with_configured_cuda_env()
    parser = argparse.ArgumentParser(description="Train a distilled student policy with behavior cloning.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value_distill_weight", type=float, default=0.1)
    parser.add_argument("--gaussian_kl_weight", type=float, default=0.0)
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--convergence_eval_interval",
        type=int,
        default=10_000,
        help="BC optimizer steps between deterministic evaluations on the fixed held-out motion split.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--motion_field", default="motion_uid")
    parser.add_argument("--strict_motion_identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require_dataset_manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    config = load_config_for_bc(args.student_config)
    result = train_bc(
        config=config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        value_distill_weight=args.value_distill_weight,
        gaussian_kl_weight=args.gaussian_kl_weight,
        init_ckpt=args.init_ckpt,
        log_interval=args.log_interval,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        motion_field=args.motion_field,
        strict_motion_identity=args.strict_motion_identity,
        convergence_eval_interval=args.convergence_eval_interval,
        require_dataset_manifest=bool(args.require_dataset_manifest),
    )
    print(f"checkpoint_path: {result.checkpoint_path}")
    print(f"metadata_path: {result.metadata_path}")
    print(f"final_train_action_mse: {result.final_train_action_mse:.6f}")
    if result.final_val_action_mse is not None:
        print(f"final_val_action_mse: {result.final_val_action_mse:.6f}")
    print(f"convergence_path: {result.convergence_path}")
    print(f"convergence_passed: {str(result.convergence_passed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
