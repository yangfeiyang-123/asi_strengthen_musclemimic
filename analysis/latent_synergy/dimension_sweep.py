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
    closed_loop_correction_root: str | Path | None = None,
    heldout_motion_paths: Sequence[str] = (),
    synergy_basis_path: str | Path | None = None,
    synergy_basis_expected_fingerprint: str | None = None,
    residual_actuator_names: Sequence[str] = (),
    residual_alpha: float = 0.05,
    phase_field: str | None = "phase_id",
    require_all_phases: bool = True,
    max_analysis_samples: int = 1024,
    max_intervention_directions: int = 8,
    require_causal_interventions: bool = False,
    python_executable: str = "python",
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
    if any(name != "direct" for name in decoders) and synergy_basis_path is None:
        raise ValueError("synergy sweep variants require synergy_basis_path")
    if (
        synergy_basis_expected_fingerprint is None
        or len(str(synergy_basis_expected_fingerprint)) != 64
        or any(character not in "0123456789abcdef" for character in str(synergy_basis_expected_fingerprint).lower())
    ):
        raise ValueError("latent sweep analysis requires a 64-hex synergy_basis_expected_fingerprint")
    if "synergy_residual" in decoders and (not residual_actuator_names or float(residual_alpha) <= 0.0):
        raise ValueError("synergy_residual sweep requires residual actuator names and residual_alpha > 0")
    production_inputs = {
        "dataset_dir": dataset_dir,
        "val_dataset_dir": val_dataset_dir,
        "teacher_ckpt": teacher_ckpt,
        "teacher_promotion_manifest": teacher_promotion_manifest,
        "direct_bc_metrics": direct_bc_metrics,
        "direct_rollout_metrics": direct_rollout_metrics,
        "direct_promotion_evidence": direct_promotion_evidence,
        "synergy_basis_path": synergy_basis_path,
    }
    missing_inputs = sorted(
        name for name, value in production_inputs.items() if value is None or not str(value).strip()
    )
    if missing_inputs:
        raise ValueError(f"production latent sweep requires lifecycle inputs: {missing_inputs}")
    heldout = tuple(str(value) for value in heldout_motion_paths)
    if len(heldout) != 5 or len(set(heldout)) != 5:
        raise ValueError("production latent sweep requires exactly five unique heldout_motion_paths")
    if int(max_analysis_samples) < 5:
        raise ValueError("max_analysis_samples must be at least five")
    if int(max_intervention_directions) <= 0:
        raise ValueError("max_intervention_directions must be positive")
    if require_all_phases and not phase_field:
        raise ValueError("require_all_phases requires phase_field")
    root = Path(output_root)
    specs: list[dict[str, Any]] = []
    for latent_dim in dims:
        for decoder_type in decoders:
            for seed in seed_values:
                run_name = f"d{latent_dim}_{decoder_type}_seed{seed}"
                output_dir = root / run_name
                command = [
                    str(python_executable),
                    "-m",
                    "fullbody.latent_train",
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
                    "5",
                    "--teacher_ckpt",
                    str(teacher_ckpt),
                    "--teacher_promotion_manifest",
                    str(teacher_promotion_manifest),
                    "--direct_bc_metrics",
                    str(direct_bc_metrics),
                    # Every variant records the exact formal analysis basis in
                    # its config.  Direct checkpoints do not embed W, while
                    # synergy checkpoints additionally bind the runtime W.
                    "--synergy_basis_path",
                    str(synergy_basis_path),
                    "--synergy_basis_expected_fingerprint",
                    str(synergy_basis_expected_fingerprint),
                ]
                if decoder_type in {"direct", "fixed_synergy"}:
                    command.append("--disable_synergy_residual")
                if decoder_type == "direct":
                    command.append("--disable_synergy_baseline")
                if decoder_type == "synergy_residual":
                    command += [
                        "--synergy_residual_alpha",
                        str(float(residual_alpha)),
                        "--synergy_residual_actuator_names",
                        *[str(name) for name in residual_actuator_names],
                    ]
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
                    "--direct_rollout_metrics",
                    str(direct_rollout_metrics),
                    "--direct_promotion_evidence",
                    str(direct_promotion_evidence),
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
                        "synergy_basis_expected_fingerprint": (str(synergy_basis_expected_fingerprint)),
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
    parser.add_argument("--synergy-basis-path", type=Path, default=None)
    parser.add_argument("--synergy-basis-expected-fingerprint", default=None)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--decoder-types", nargs="+", default=list(DEFAULT_DECODERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--residual-actuator-names", nargs="*", default=[])
    parser.add_argument("--residual-alpha", type=float, default=0.05)
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
        synergy_basis_path=args.synergy_basis_path,
        synergy_basis_expected_fingerprint=args.synergy_basis_expected_fingerprint,
        residual_actuator_names=args.residual_actuator_names,
        residual_alpha=float(args.residual_alpha),
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
