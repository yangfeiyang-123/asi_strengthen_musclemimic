"""Train latent posterior/prior/decoder modules from strict distill shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musclemimic.latent_muscle.train_latent import LatentTrainConfig, train_latent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_layer_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_steps", type=int, default=10_000)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--bound_weight", type=float, default=0.0)
    parser.add_argument("--action_min", type=float, default=-1.0)
    parser.add_argument("--action_max", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--sigma_max", type=float, default=2.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--action_mask_json",
        type=Path,
        default=None,
        help="Optional JSON with all_actuator_names and correction_actuator_names.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    action_mask = None
    if args.action_mask_json is not None:
        action_mask = json.loads(args.action_mask_json.read_text(encoding="utf-8"))
    result = train_latent(
        LatentTrainConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            latent_dim=int(args.latent_dim),
            hidden_layer_dims=tuple(int(value) for value in args.hidden_layer_dims),
            batch_size=int(args.batch_size),
            horizon=int(args.horizon),
            num_steps=int(args.num_steps),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            kl_weight=float(args.kl_weight),
            kl_warmup_steps=int(args.kl_warmup_steps),
            free_bits=float(args.free_bits),
            smooth_weight=float(args.smooth_weight),
            bound_weight=float(args.bound_weight),
            action_min=float(args.action_min),
            action_max=float(args.action_max),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            log_interval=int(args.log_interval),
            action_mask=action_mask,
        )
    )
    print(f"latent_checkpoint_dir: {result.checkpoint_dir}")
    print(f"final_total_loss: {result.final_total_loss:.6f}")
    print(f"final_action_mse: {result.final_action_mse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
