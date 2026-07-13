"""Train latent posterior/prior/decoder modules from strict distill shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from musclemimic.latent_muscle.train_latent import LatentTrainConfig, train_latent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Production latent YAML config.")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
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
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--motion_field", default="motion_uid")
    parser.add_argument("--strict_motion_identity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalizer_epsilon", type=float, default=1e-6)
    parser.add_argument("--normalizer_clip", type=float, default=10.0)
    parser.add_argument("--max_eval_samples", type=int, default=65_536)
    parser.add_argument("--direct_bc_action_mse", type=float, default=None)
    parser.add_argument(
        "--direct_bc_metrics",
        type=Path,
        default=None,
        help="Direct-BC held-out metrics JSON; action MSE is injected into the latent gate.",
    )
    parser.add_argument("--teacher_ckpt", default=None, help="Stage-2 teacher provenance/closed-loop source.")
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
    parser.add_argument(
        "--closed_loop_metrics",
        type=Path,
        default=None,
        help="Held-out prior-mean/LAB sweep metrics JSON produced by the closed-loop evaluator.",
    )
    parser.add_argument(
        "--test_only_closed_loop_metrics",
        action="store_true",
        default=False,
        help="Allow legacy injected metrics for tests; they can never satisfy production promotion.",
    )
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
    direct_bc_action_mse = args.direct_bc_action_mse
    if args.direct_bc_metrics is not None:
        direct_bc_action_mse = _direct_bc_action_mse(args.direct_bc_metrics)
    closed_loop_metrics = (
        None
        if args.closed_loop_metrics is None
        else json.loads(args.closed_loop_metrics.read_text(encoding="utf-8"))
    )
    if closed_loop_metrics is not None and not args.test_only_closed_loop_metrics:
        raise ValueError(
            "--closed_loop_metrics is legacy test-only input; pass "
            "--test_only_closed_loop_metrics explicitly. Production promotion must use "
            "fullbody.latent_closed_loop_eval."
        )
    if args.config is not None:
        payload = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        payload = dict(payload.get("latent_distill", payload))
        if args.dataset_dir is not None:
            payload["dataset_dir"] = args.dataset_dir
        if args.output_dir is not None:
            payload["output_dir"] = args.output_dir
        if action_mask is not None:
            payload["action_mask"] = action_mask
        if direct_bc_action_mse is not None:
            payload["direct_bc_action_mse"] = float(direct_bc_action_mse)
            payload["direct_bc_metrics_path"] = (
                None if args.direct_bc_metrics is None else str(args.direct_bc_metrics)
            )
        if args.teacher_ckpt is not None:
            payload["teacher_ckpt"] = str(args.teacher_ckpt)
        if args.teacher_promotion_manifest is not None:
            payload["teacher_promotion_manifest"] = str(
                args.teacher_promotion_manifest
            )
        if args.test_only_allow_unpromoted_teacher:
            payload["test_only_allow_unpromoted_teacher"] = True
        if closed_loop_metrics is not None:
            payload["closed_loop_evaluator"] = lambda _context: closed_loop_metrics
        config = LatentTrainConfig(**payload)
    else:
        if args.dataset_dir is None or args.output_dir is None:
            raise SystemExit("--dataset_dir and --output_dir are required unless --config is supplied")
        config = LatentTrainConfig(
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
            val_fraction=float(args.val_fraction),
            motion_field=str(args.motion_field),
            strict_motion_identity=bool(args.strict_motion_identity),
            normalizer_epsilon=float(args.normalizer_epsilon),
            normalizer_clip=float(args.normalizer_clip),
            max_eval_samples=int(args.max_eval_samples),
            direct_bc_action_mse=direct_bc_action_mse,
            direct_bc_metrics_path=(
                None if args.direct_bc_metrics is None else str(args.direct_bc_metrics)
            ),
            teacher_ckpt=args.teacher_ckpt,
            teacher_promotion_manifest=args.teacher_promotion_manifest,
            test_only_allow_unpromoted_teacher=bool(
                args.test_only_allow_unpromoted_teacher
            ),
            closed_loop_evaluator=(
                None if closed_loop_metrics is None else lambda _context: closed_loop_metrics
            ),
        )
    result = train_latent(config)
    print(f"latent_checkpoint_dir: {result.checkpoint_dir}")
    print(f"final_total_loss: {result.final_total_loss:.6f}")
    print(f"final_action_mse: {result.final_action_mse:.6f}")
    return 0


def _direct_bc_action_mse(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = (
        # Direct BC and latent decoder both target the clipped normalized
        # action actually applied by DefaultControl.  Raw Gaussian-mean MSE is
        # retained only as a legacy fallback because it may be unreachable.
        "mse_to_teacher_action",
        "action_mse",
        "val_action_mse",
        "mse_to_teacher_mu",
    )
    mappings = [payload]
    for key in ("val", "validation", "eval_metrics", "metrics"):
        nested = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(nested, dict):
            mappings.append(nested)
    for mapping in mappings:
        for key in candidates:
            if key in mapping:
                value = float(mapping[key])
                if value < 0.0 or not np.isfinite(value):
                    raise ValueError(f"direct BC action MSE must be finite and non-negative, got {value}")
                return value
    raise ValueError(
        f"direct BC metrics {path} has none of the supported keys: {', '.join(candidates)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
