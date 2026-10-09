"""Generate reproducible latent-dimension/decoder/seed training jobs."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_DIMENSIONS = (2, 4, 8, 16, 32)
DEFAULT_DECODERS = ("direct", "fixed_synergy", "synergy_residual")


def build_sweep_specs(
    *,
    base_config: str | Path,
    output_root: str | Path,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    decoder_types: Sequence[str] = DEFAULT_DECODERS,
    seeds: Sequence[int] = (0, 1, 2),
    dataset_dir: str | Path | None = None,
    val_dataset_dir: str | Path | None = None,
    teacher_ckpt: str | Path | None = None,
    teacher_promotion_manifest: str | Path | None = None,
    direct_bc_metrics: str | Path | None = None,
    direct_rollout_metrics: str | Path | None = None,
    direct_promotion_evidence: str | Path | None = None,
    require_direct_bc_baseline: bool = True,
    closed_loop_correction_root: str | Path | None = None,
    heldout_motion_paths: Sequence[str] = (),
    expected_validation_motion_count: int = 5,
    synergy_basis_path: str | Path | None = None,
    synergy_basis_expected_fingerprint: str | None = None,
    frozen_body_decoder_path: str | Path | None = None,
    frozen_body_decoder_expected_fingerprint: str | None = None,
    body_synergy_contract_expected_fingerprint: str | None = None,
    body_synergy_portable_core_expected_fingerprint: str | None = None,
    residual_actuator_names: Sequence[str] = (),
    residual_alpha: float = 0.05,
    emg_privileged_enabled: bool = False,
    emg_synergy_dim: int = 0,
    emg_reference_manifest: str | Path | None = None,
    emg_context_dropout: float | None = None,
    emg_synergy_loss_weight: float | None = None,
    emg_tube_kappa: float | None = None,
    emg_shuffle_context_ablation: bool = False,
    phase_field: str | None = "phase_id",
    require_all_phases: bool = True,
    phase_contract_path: str | Path | None = None,
    max_analysis_samples: int = 1024,
    max_intervention_directions: int = 8,
    require_causal_interventions: bool = False,
    python_executable: str = "python",
    training_launcher: str | Path = "scripts/run_fullbody_training.sh",
) -> list[dict[str, Any]]:
    config = Path(base_config)
    dims = tuple(int(value) for value in dimensions)
    decoders = tuple(str(value) for value in decoder_types)
    seed_values = tuple(int(value) for value in seeds)
    if not dims or any(value <= 0 for value in dims) or len(set(dims)) != len(dims):
        raise ValueError("sweep dimensions must be unique positive integers")
    if not seed_values or any(value < 0 for value in seed_values) or len(set(seed_values)) != len(seed_values):
        raise ValueError("sweep seeds must be unique non-negative integers")
    unknown = sorted(set(decoders) - set(DEFAULT_DECODERS))
    if not decoders or unknown or len(set(decoders)) != len(decoders):
        raise ValueError(f"unsupported or duplicate sweep decoder types: {unknown}")
    has_synergy = any(name != "direct" for name in decoders)
    if has_synergy and synergy_basis_path is None:
        raise ValueError("synergy sweep analysis requires synergy_basis_path")
    if (
        synergy_basis_expected_fingerprint is None
        or len(str(synergy_basis_expected_fingerprint)) != 64
        or any(character not in "0123456789abcdef" for character in str(synergy_basis_expected_fingerprint).lower())
    ):
        raise ValueError("latent sweep analysis requires a 64-hex synergy_basis_expected_fingerprint")
    portable_inputs = {
        "frozen_body_decoder_path": frozen_body_decoder_path,
        "frozen_body_decoder_expected_fingerprint": (frozen_body_decoder_expected_fingerprint),
        "body_synergy_contract_expected_fingerprint": (body_synergy_contract_expected_fingerprint),
        "body_synergy_portable_core_expected_fingerprint": (body_synergy_portable_core_expected_fingerprint),
    }
    if has_synergy:
        missing_portable = sorted(
            name for name, value in portable_inputs.items() if value is None or not str(value).strip()
        )
        if missing_portable:
            raise ValueError(f"synergy sweep requires portable frozen decoder inputs: {missing_portable}")
        for name, value in portable_inputs.items():
            if name.endswith("fingerprint") and (
                len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value).lower())
            ):
                raise ValueError(f"{name} must be a 64-hex fingerprint")
    production_inputs = {
        "dataset_dir": dataset_dir,
        "val_dataset_dir": val_dataset_dir,
        "teacher_ckpt": teacher_ckpt,
        "teacher_promotion_manifest": teacher_promotion_manifest,
        "synergy_basis_path": synergy_basis_path,
        "frozen_body_decoder_path": (frozen_body_decoder_path if has_synergy else "not_required"),
    }
    if require_direct_bc_baseline:
        production_inputs.update(
            {
                "direct_bc_metrics": direct_bc_metrics,
                "direct_rollout_metrics": direct_rollout_metrics,
                "direct_promotion_evidence": direct_promotion_evidence,
            }
        )
    missing_inputs = sorted(
        name for name, value in production_inputs.items() if value is None or not str(value).strip()
    )
    if missing_inputs:
        raise ValueError(f"production latent sweep requires lifecycle inputs: {missing_inputs}")
    heldout = tuple(str(value) for value in heldout_motion_paths)
    expected_val_count = int(expected_validation_motion_count)
    if expected_val_count <= 0:
        raise ValueError("expected_validation_motion_count must be positive")
    if len(heldout) != expected_val_count or len(set(heldout)) != expected_val_count:
        raise ValueError(f"production latent sweep requires exactly {expected_val_count} unique heldout_motion_paths")
    if int(max_analysis_samples) < 5:
        raise ValueError("max_analysis_samples must be at least five")
    if int(max_intervention_directions) <= 0:
        raise ValueError("max_intervention_directions must be positive")
    if require_all_phases and not phase_field:
        raise ValueError("require_all_phases requires phase_field")
    emg_enabled = bool(emg_privileged_enabled)
    emg_dim = int(emg_synergy_dim)
    if emg_enabled:
        # A privileged sweep without a reviewed tube would produce checkpoints
        # whose EMG provenance cannot be audited, so the sweep refuses it here
        # rather than relying on the per-run trainer to notice.
        if emg_dim <= 0:
            raise ValueError("privileged EMG sweep requires a positive emg_synergy_dim")
        if emg_reference_manifest is None or not str(emg_reference_manifest).strip():
            raise ValueError("privileged EMG sweep requires a reviewed emg_reference_manifest")
        if emg_context_dropout is not None and not 0.0 <= float(emg_context_dropout) <= 1.0:
            raise ValueError("emg_context_dropout must lie in [0, 1]")
        for name, value in (
            ("emg_synergy_loss_weight", emg_synergy_loss_weight),
            ("emg_tube_kappa", emg_tube_kappa),
        ):
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
    elif emg_dim or emg_reference_manifest or emg_shuffle_context_ablation:
        raise ValueError("EMG sweep inputs require emg_privileged_enabled")
    root = Path(output_root)
    specs: list[dict[str, Any]] = []
    for latent_dim in dims:
        for decoder_type in decoders:
            for seed in seed_values:
                # The marker keeps a privileged run distinguishable from its
                # EMG-free baseline even when records from separate sweep roots
                # are collated into one comparison table.  The shuffled control
                # needs its own marker: it shares every other spec field with
                # the real privileged arm, so a common name would collide on
                # ``output_dir`` and let one arm overwrite the other.  An
                # explicit dropout=0 arm (S2-E, "privileged without context
                # dropout") also gets its own marker for the same reason.
                if not emg_enabled:
                    emg_marker = ""
                elif emg_shuffle_context_ablation:
                    emg_marker = "_peasd_shuffled"
                elif emg_context_dropout is not None and float(emg_context_dropout) == 0.0:
                    emg_marker = "_peasd_nodropout"
                else:
                    emg_marker = "_peasd"
                run_name = f"d{latent_dim}_{decoder_type}{emg_marker}_seed{seed}"
                output_dir = root / run_name
                command = [
                    str(training_launcher),
                    "--latent",
                    "--config",
                    str(config),
                    "--output_dir",
                    str(output_dir),
                    "--latent_dim",
                    str(latent_dim),
                    "--seed",
                    str(seed),
                    "--decoder_type",
                    decoder_type,
                    "--dataset_dir",
                    str(dataset_dir),
                    "--val_dataset_dir",
                    str(val_dataset_dir),
                    "--expected_val_motion_count",
                    str(expected_val_count),
                    "--teacher_ckpt",
                    str(teacher_ckpt),
                    "--teacher_promotion_manifest",
                    str(teacher_promotion_manifest),
                ]
                if require_direct_bc_baseline:
                    command += ["--direct_bc_metrics", str(direct_bc_metrics)]
                if decoder_type != "direct":
                    command += [
                        "--frozen_body_decoder_path",
                        str(frozen_body_decoder_path),
                        "--frozen_body_decoder_expected_fingerprint",
                        str(frozen_body_decoder_expected_fingerprint),
                        "--body_synergy_contract_expected_fingerprint",
                        str(body_synergy_contract_expected_fingerprint),
                        "--body_synergy_portable_core_expected_fingerprint",
                        str(body_synergy_portable_core_expected_fingerprint),
                    ]
                if decoder_type in {"direct", "fixed_synergy"}:
                    command.append("--disable_synergy_residual")
                if decoder_type == "direct":
                    command.append("--disable_synergy_baseline")
                if emg_enabled:
                    command += [
                        "--emg_privileged_enabled",
                        "--emg_synergy_dim",
                        str(emg_dim),
                        "--emg_reference_manifest",
                        str(emg_reference_manifest),
                    ]
                    if emg_context_dropout is not None:
                        command += ["--emg_context_dropout", str(float(emg_context_dropout))]
                    if emg_synergy_loss_weight is not None:
                        command += ["--emg_synergy_loss_weight", str(float(emg_synergy_loss_weight))]
                    if emg_tube_kappa is not None:
                        command += ["--emg_tube_kappa", str(float(emg_tube_kappa))]
                    if emg_shuffle_context_ablation:
                        command.append("--emg_shuffle_context_ablation")
                correction_dataset = None
                correction_manifest = None
                if closed_loop_correction_root is not None:
                    correction_dataset = Path(closed_loop_correction_root) / run_name / "dataset"
                    correction_manifest = correction_dataset / "dataset_manifest.json"
                    command += [
                        "--closed-loop-correction-dataset-dir",
                        str(correction_dataset),
                        "--closed-loop-correction-manifest",
                        str(correction_manifest),
                    ]
                checkpoint_dir = output_dir / "latent_checkpoint"
                closed_loop_command = [
                    str(python_executable),
                    "-m",
                    "fullbody.latent_closed_loop_eval",
                    "--latent_checkpoint",
                    str(checkpoint_dir),
                    "--teacher_ckpt",
                    str(teacher_ckpt),
                ]
                if require_direct_bc_baseline:
                    closed_loop_command += [
                        "--direct_rollout_metrics",
                        str(direct_rollout_metrics),
                        "--direct_promotion_evidence",
                        str(direct_promotion_evidence),
                    ]
                closed_loop_command += [
                    "--motion_path",
                    *heldout,
                    "--collect_decoder_usage",
                    "--collect_jacobian_alignment",
                    "--alignment_synergy_basis",
                    str(synergy_basis_path),
                    "--require_pass",
                ]
                if phase_field:
                    closed_loop_command += ["--phase_field", str(phase_field)]
                if require_all_phases:
                    closed_loop_command.append("--require_all_phases")
                if phase_contract_path is not None:
                    closed_loop_command += ["--phase_contract_json", str(phase_contract_path)]
                analysis_export_command = [
                    str(python_executable),
                    "-m",
                    "fullbody.latent_synergy_export",
                    "--latent-checkpoint",
                    str(checkpoint_dir),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--val-dataset-dir",
                    str(val_dataset_dir),
                    "--synergy-basis",
                    str(synergy_basis_path),
                    "--synergy-basis-fingerprint",
                    str(synergy_basis_expected_fingerprint),
                    "--output-npz",
                    str(output_dir / "analysis_inputs.npz"),
                    "--max-samples",
                    str(int(max_analysis_samples)),
                    "--max-intervention-directions",
                    str(int(max_intervention_directions)),
                ]
                if require_all_phases:
                    analysis_export_command.append("--require-all-phases")
                if phase_contract_path is not None:
                    analysis_export_command += ["--phase-contract-json", str(phase_contract_path)]
                # Causal evidence has an unavoidable two-pass lifecycle: the
                # bootstrap export creates stable sample UIDs/directions for
                # the environment rollout, then the finalized export seals
                # those measured outcomes into the same analysis input.
                causal_finalize_command = [
                    *analysis_export_command,
                    "--causal-interventions-npz",
                    str(output_dir / "causal_interventions.npz"),
                    "--causal-interventions-manifest",
                    str(output_dir / "causal_interventions.json"),
                    "--require-causal-interventions",
                ]
                specs.append(
                    {
                        "run_name": run_name,
                        "latent_dim": latent_dim,
                        "decoder_type": decoder_type,
                        "seed": seed,
                        "emg_privileged_enabled": emg_enabled,
                        "emg_synergy_dim": (emg_dim if emg_enabled else 0),
                        "emg_reference_manifest": (str(emg_reference_manifest) if emg_enabled else None),
                        "emg_shuffle_context_ablation": (bool(emg_shuffle_context_ablation) if emg_enabled else False),
                        "synergy_basis_expected_fingerprint": (str(synergy_basis_expected_fingerprint)),
                        "frozen_body_decoder_fingerprint": (
                            None if decoder_type == "direct" else str(frozen_body_decoder_expected_fingerprint)
                        ),
                        "body_synergy_contract_fingerprint": (
                            None if decoder_type == "direct" else str(body_synergy_contract_expected_fingerprint)
                        ),
                        "body_synergy_portable_core_fingerprint": (
                            None if decoder_type == "direct" else str(body_synergy_portable_core_expected_fingerprint)
                        ),
                        "output_dir": str(output_dir),
                        "checkpoint_dir": str(checkpoint_dir),
                        "closed_loop_correction_dataset_dir": (
                            None if correction_dataset is None else str(correction_dataset)
                        ),
                        "closed_loop_correction_manifest": (
                            None if correction_manifest is None else str(correction_manifest)
                        ),
                        "command": command,
                        "command_shell": shlex.join(command),
                        "training_command": command,
                        "training_command_shell": shlex.join(command),
                        "closed_loop_command": closed_loop_command,
                        "closed_loop_command_shell": shlex.join(closed_loop_command),
                        "analysis_export_command": analysis_export_command,
                        "analysis_export_command_shell": shlex.join(analysis_export_command),
                        "causal_finalize_command": causal_finalize_command,
                        "causal_finalize_command_shell": shlex.join(causal_finalize_command),
                        "causal_interventions_npz": str(output_dir / "causal_interventions.npz"),
                        "causal_interventions_manifest": str(output_dir / "causal_interventions.json"),
                    }
                )
    return specs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--val-dataset-dir", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--teacher-promotion-manifest", type=Path, required=True)
    parser.add_argument("--direct-bc-metrics", type=Path, required=True)
    parser.add_argument("--direct-rollout-metrics", type=Path, required=True)
    parser.add_argument("--direct-promotion-evidence", type=Path, required=True)
    parser.add_argument(
        "--closed-loop-correction-root",
        type=Path,
        default=None,
        help=(
            "Optional pre-collected per-run latent DAgger root. Each job reads "
            "<root>/<run_name>/dataset and its immutable dataset_manifest.json."
        ),
    )
    parser.add_argument("--heldout-motion-path", action="append", required=True)
    parser.add_argument("--expected-validation-motion-count", type=int, default=5)
    parser.add_argument("--synergy-basis-path", type=Path, default=None)
    parser.add_argument("--synergy-basis-expected-fingerprint", default=None)
    parser.add_argument("--frozen-body-decoder-path", type=Path, default=None)
    parser.add_argument("--frozen-body-decoder-expected-fingerprint", default=None)
    parser.add_argument("--body-synergy-contract-expected-fingerprint", default=None)
    parser.add_argument("--body-synergy-portable-core-expected-fingerprint", default=None)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--decoder-types", nargs="+", default=list(DEFAULT_DECODERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--residual-actuator-names", nargs="*", default=[])
    parser.add_argument("--residual-alpha", type=float, default=0.05)
    parser.add_argument("--phase-contract-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shell", type=Path, default=None)
    parser.add_argument(
        "--require-causal-interventions",
        action="store_true",
        default=False,
        help=(
            "Register a fail-closed causal artifact input for every analysis export. "
            "The sweep never creates rollout evidence itself."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    specs = build_sweep_specs(
        base_config=args.base_config,
        output_root=args.output_root,
        dimensions=args.dimensions,
        decoder_types=args.decoder_types,
        seeds=args.seeds,
        dataset_dir=args.dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        teacher_ckpt=args.teacher_ckpt,
        teacher_promotion_manifest=args.teacher_promotion_manifest,
        direct_bc_metrics=args.direct_bc_metrics,
        direct_rollout_metrics=args.direct_rollout_metrics,
        direct_promotion_evidence=args.direct_promotion_evidence,
        closed_loop_correction_root=args.closed_loop_correction_root,
        heldout_motion_paths=args.heldout_motion_path,
        expected_validation_motion_count=int(args.expected_validation_motion_count),
        synergy_basis_path=args.synergy_basis_path,
        synergy_basis_expected_fingerprint=args.synergy_basis_expected_fingerprint,
        frozen_body_decoder_path=args.frozen_body_decoder_path,
        frozen_body_decoder_expected_fingerprint=(args.frozen_body_decoder_expected_fingerprint),
        body_synergy_contract_expected_fingerprint=(args.body_synergy_contract_expected_fingerprint),
        body_synergy_portable_core_expected_fingerprint=(args.body_synergy_portable_core_expected_fingerprint),
        residual_actuator_names=args.residual_actuator_names,
        residual_alpha=float(args.residual_alpha),
        phase_contract_path=args.phase_contract_json,
        require_causal_interventions=bool(args.require_causal_interventions),
    )
    payload = {
        "schema_version": "latent_dimension_sweep_v1",
        "num_jobs": len(specs),
        "jobs": specs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.output_shell is not None:
        args.output_shell.parent.mkdir(parents=True, exist_ok=True)
        args.output_shell.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(spec["command_shell"] for spec in specs) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
