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
    parser.add_argument(
        "--val_dataset_dir",
        default=None,
        help="Independent immutable validation collection; production synergy sweeps use the canonical five motions.",
    )
    parser.add_argument("--expected_val_motion_count", type=int, default=None)
    parser.add_argument(
        "--closed-loop-correction-dataset-dir",
        dest="closed_loop_correction_dataset_dir",
        default=None,
        help=(
            "Optional pre-collected student closed-loop states relabeled by the "
            "Stage-2 teacher. No rollout collection is performed by this command."
        ),
    )
    parser.add_argument(
        "--closed-loop-correction-manifest",
        dest="closed_loop_correction_manifest",
        default=None,
        help="Immutable distill_dataset_manifest_v2 for the correction dataset.",
    )
    parser.add_argument("--output_dir", default=None)
    # None means "leave YAML untouched".  This makes dimension/seed sweeps
    # explicit while preserving the standalone defaults below.
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_layer_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument(
        "--decoder_type",
        choices=("direct", "fixed_synergy", "synergy_residual"),
        default=None,
    )
    parser.add_argument("--synergy_basis_path", type=Path, default=None)
    parser.add_argument("--synergy_basis_expected_fingerprint", default=None)
    parser.add_argument(
        "--test_only_allow_legacy_synergy_basis",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--synergy_include_baseline",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--synergy_baseline_init", type=float, default=None)
    parser.add_argument("--synergy_residual_actuator_names", nargs="+", default=None)
    parser.add_argument("--synergy_residual_alpha", type=float, default=None)
    parser.add_argument("--synergy_residual_l1_weight", type=float, default=None)
    parser.add_argument("--synergy_residual_l2_weight", type=float, default=None)
    parser.add_argument("--synergy_residual_smooth_weight", type=float, default=None)
    parser.add_argument(
        "--disable_synergy_residual",
        action="store_true",
        default=False,
        help="Clear residual names/alpha/losses when sweeping direct or fixed-synergy ablations.",
    )
    parser.add_argument("--synergy_baseline_l1_weight", type=float, default=None)
    parser.add_argument("--synergy_baseline_l2_weight", type=float, default=None)
    parser.add_argument(
        "--disable_synergy_baseline",
        action="store_true",
        default=False,
        help="Disable baseline and its losses for the direct-decoder ablation.",
    )
    parser.add_argument("--phase_field", default=None)
    parser.add_argument(
        "--phase_balance_weights_json",
        type=Path,
        default=None,
        help="JSON mapping ready/backswing/... phase names (or IDs) to loss weights.",
    )
    parser.add_argument("--physical_excitation_field", default=None)
    parser.add_argument("--physical_excitation_weight", type=float, default=None)
    parser.add_argument("--physical_excitation_min", type=float, default=None)
    parser.add_argument("--physical_excitation_max", type=float, default=None)
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
    phase_balance_weights = (
        None
        if args.phase_balance_weights_json is None
        else json.loads(args.phase_balance_weights_json.read_text(encoding="utf-8"))
    )
    if phase_balance_weights is not None and not isinstance(phase_balance_weights, dict):
        raise ValueError("--phase_balance_weights_json must contain a JSON object")
    if args.config is not None:
        payload = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        payload = dict(payload.get("latent_distill", payload))
        if args.dataset_dir is not None:
            payload["dataset_dir"] = args.dataset_dir
        if args.val_dataset_dir is not None:
            payload["val_dataset_dir"] = args.val_dataset_dir
        if args.expected_val_motion_count is not None:
            payload["expected_val_motion_count"] = int(
                args.expected_val_motion_count
            )
        if args.closed_loop_correction_dataset_dir is not None:
            payload["closed_loop_correction_dataset_dir"] = str(
                args.closed_loop_correction_dataset_dir
            )
        if args.closed_loop_correction_manifest is not None:
            payload["closed_loop_correction_manifest"] = str(
                args.closed_loop_correction_manifest
            )
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
        _apply_latent_cli_overrides(
            payload,
            args,
            phase_balance_weights=phase_balance_weights,
        )
        if closed_loop_metrics is not None:
            payload["closed_loop_evaluator"] = lambda _context: closed_loop_metrics
        config = LatentTrainConfig(**payload)
    else:
        if args.dataset_dir is None or args.output_dir is None:
            raise SystemExit("--dataset_dir and --output_dir are required unless --config is supplied")
        config = LatentTrainConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            val_dataset_dir=args.val_dataset_dir,
            expected_val_motion_count=args.expected_val_motion_count,
            closed_loop_correction_dataset_dir=(
                None
                if args.closed_loop_correction_dataset_dir is None
                else str(args.closed_loop_correction_dataset_dir)
            ),
            closed_loop_correction_manifest=(
                None
                if args.closed_loop_correction_manifest is None
                else str(args.closed_loop_correction_manifest)
            ),
            latent_dim=32 if args.latent_dim is None else int(args.latent_dim),
            hidden_layer_dims=tuple(int(value) for value in args.hidden_layer_dims),
            batch_size=int(args.batch_size),
            horizon=int(args.horizon),
            num_steps=int(args.num_steps),
            learning_rate=float(args.learning_rate),
            seed=0 if args.seed is None else int(args.seed),
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
            decoder_type=args.decoder_type or "direct",
            synergy_basis_path=(
                None if args.synergy_basis_path is None else str(args.synergy_basis_path)
            ),
            synergy_basis_expected_fingerprint=args.synergy_basis_expected_fingerprint,
            test_only_allow_legacy_synergy_basis=bool(
                args.test_only_allow_legacy_synergy_basis
            ),
            synergy_include_baseline=(
                False
                if args.disable_synergy_baseline
                else (
                    True
                    if args.synergy_include_baseline is None
                    else bool(args.synergy_include_baseline)
                )
            ),
            synergy_baseline_init=(
                0.01
                if args.synergy_baseline_init is None
                else float(args.synergy_baseline_init)
            ),
            synergy_residual_actuator_names=(
                ()
                if args.disable_synergy_residual
                else tuple(args.synergy_residual_actuator_names or ())
            ),
            synergy_residual_alpha=(
                0.0 if args.disable_synergy_residual else float(args.synergy_residual_alpha or 0.0)
            ),
            synergy_residual_l1_weight=(
                0.0 if args.disable_synergy_residual else float(args.synergy_residual_l1_weight or 0.0)
            ),
            synergy_residual_l2_weight=(
                0.0 if args.disable_synergy_residual else float(args.synergy_residual_l2_weight or 0.0)
            ),
            synergy_residual_smooth_weight=float(
                0.0
                if args.disable_synergy_residual
                else args.synergy_residual_smooth_weight or 0.0
            ),
            synergy_baseline_l1_weight=(
                0.0 if args.disable_synergy_baseline else float(args.synergy_baseline_l1_weight or 0.0)
            ),
            synergy_baseline_l2_weight=(
                0.0 if args.disable_synergy_baseline else float(args.synergy_baseline_l2_weight or 0.0)
            ),
            phase_field=args.phase_field or "phase_id",
            phase_balance_weights=phase_balance_weights,
            physical_excitation_field=args.physical_excitation_field or "muscle_excitation",
            physical_excitation_weight=float(args.physical_excitation_weight or 0.0),
            physical_excitation_min=(
                0.0
                if args.physical_excitation_min is None
                else float(args.physical_excitation_min)
            ),
            physical_excitation_max=(
                1.0
                if args.physical_excitation_max is None
                else float(args.physical_excitation_max)
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


def _apply_latent_cli_overrides(
    payload: dict,
    args: argparse.Namespace,
    *,
    phase_balance_weights: dict | None,
) -> None:
    """Apply only explicitly supplied sweep/decoder options to a YAML payload."""

    scalar_fields = (
        "latent_dim",
        "seed",
        "decoder_type",
        "synergy_basis_expected_fingerprint",
        "synergy_include_baseline",
        "synergy_baseline_init",
        "synergy_residual_alpha",
        "synergy_residual_l1_weight",
        "synergy_residual_l2_weight",
        "synergy_residual_smooth_weight",
        "synergy_baseline_l1_weight",
        "synergy_baseline_l2_weight",
        "phase_field",
        "physical_excitation_field",
        "physical_excitation_weight",
        "physical_excitation_min",
        "physical_excitation_max",
    )
    for field in scalar_fields:
        value = getattr(args, field)
        if value is not None:
            payload[field] = value
    if args.synergy_basis_path is not None:
        payload["synergy_basis_path"] = str(args.synergy_basis_path)
    if args.test_only_allow_legacy_synergy_basis:
        payload["test_only_allow_legacy_synergy_basis"] = True
    if args.synergy_residual_actuator_names is not None:
        payload["synergy_residual_actuator_names"] = list(
            args.synergy_residual_actuator_names
        )
    if args.disable_synergy_residual:
        payload.update(
            {
                "synergy_residual_actuator_names": [],
                "synergy_residual_alpha": 0.0,
                "synergy_residual_l1_weight": 0.0,
                "synergy_residual_l2_weight": 0.0,
                "synergy_residual_smooth_weight": 0.0,
            }
        )
    if args.disable_synergy_baseline:
        payload.update(
            {
                "synergy_include_baseline": False,
                "synergy_baseline_l1_weight": 0.0,
                "synergy_baseline_l2_weight": 0.0,
            }
        )
    if phase_balance_weights is not None:
        payload["phase_balance_weights"] = phase_balance_weights


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
