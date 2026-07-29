"""Plan, explicitly execute, and summarize latent-synergy dimension sweeps.

``plan`` is command-only: it writes a manifest and shell scripts and can never
start training.  The separate ``execute`` command is the explicit mutation
boundary.  ``analyze`` consumes completed, fingerprint-bound lifecycle evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.latent_synergy.build_report import (
    build_comparison_report,
    report_markdown,
)
from analysis.latent_synergy.cross_seed import cross_seed_report
from analysis.latent_synergy.dimension_sweep import build_sweep_specs
from analysis.latent_synergy.effective_dimension import effective_dimension_report
from analysis.latent_synergy.intervention import summarize_intervention_effects
from analysis.latent_synergy.jacobian_alignment import jacobian_alignment_report
from analysis.latent_synergy.representation_similarity import representation_report
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.provenance import validate_dataset_manifest
from musclemimic.latent_muscle.analysis_export import (
    ANALYSIS_INPUT_SCHEMA_VERSION,
    CORE_ARRAY_FIELDS,
    STAGE2_DIAGNOSTIC_CAUSAL_OUTCOMES,
    validate_runtime_basis_binding,
)
from musclemimic.latent_muscle.causal_rollout_artifact import (
    validate_causal_rollout_artifact,
)
from musclemimic.latent_muscle.causal_rollout_driver import (
    BASELINE_FILENAME,
    JOB_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    PERTURBED_FILENAME,
)
from musclemimic.latent_muscle.checkpoint import latent_checkpoint_fingerprint
from musclemimic.latent_muscle.closed_loop_eval import (
    validate_closed_loop_promotion_report,
)
from musclemimic.latent_muscle.runtime import load_latent_runtime
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.frozen_decoder import load_frozen_body_decoder
from musclemimic.synergy.hybrid_basis import HYBRID_BASIS_SCHEMA_VERSION
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

DEFAULT_CONFIG = Path("fullbody/config_specific_task/distill/latent_forehandclear_synergy_v3.yaml")
DEFAULT_RESIDUAL_NAMES = (
    "SUP",
    "BRA",
    "BRD",
    "ECRL",
    "ECRB",
    "ECU",
    "FCR",
    "FCU",
    "PT",
    "PQ",
)
ANALYSIS_METRICS = (
    "posterior_action_mse",
    "prior_mean_action_mse",
    "physical_excitation_mse",
    "residual_energy_ratio",
)
ANALYSIS_INPUT_FIELDS = CORE_ARRAY_FIELDS
MAX_RESIDUAL_ENERGY_RATIO = 0.10
MAX_QUIESCENT_PHASE_RESIDUAL_ENERGY_RATIO = 0.05
CAUSAL_ADAPTER_SHARED_CONFIG_SCHEMA_VERSION = "latent_causal_adapter_shared_config_v1"
CAUSAL_EVALUATION_MANIFEST_SCHEMA_VERSION = "latent_causal_sweep_evaluation_v1"
DEFAULT_CAUSAL_ADAPTER_IMPORT = "musclemimic.latent_muscle.stage2_causal_adapter:create_adapter"
_CAUSAL_ADAPTER_PLAN_BOUND_FIELDS = frozenset(
    {
        "latent_checkpoint",
        "teacher_ckpt",
        "dataset_dir",
        "val_dataset_dir",
        "analysis_inputs",
        "analysis_manifest",
    }
)
_COMPLETE_SWEEP_BASIS_REGIONS = frozenset(
    {
        "regional_composite",
        "whole_body",
        "hybrid_global_regional",
    }
)
_PRIMARY_HYBRID_ARTIFACT_ROLE = "primary_hybrid_global_regional"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write the job matrix; do not train by default")
    plan.add_argument("--dataset-dir", type=Path, required=True)
    plan.add_argument("--val-dataset-dir", type=Path, required=True)
    plan.add_argument("--teacher-ckpt", type=Path, required=True)
    plan.add_argument("--teacher-promotion-manifest", type=Path, required=True)
    plan.add_argument("--direct-bc-metrics", type=Path, required=True)
    plan.add_argument("--direct-rollout-metrics", type=Path, required=True)
    plan.add_argument("--direct-promotion-evidence", type=Path, required=True)
    plan.add_argument(
        "--closed-loop-correction-root",
        type=Path,
        default=None,
        help=(
            "Optional root of pre-collected per-run student-rollout/teacher-relabel "
            "datasets. Planning and execution never collect them implicitly."
        ),
    )
    plan.add_argument("--synergy-basis", type=Path, required=True)
    plan.add_argument("--synergy-basis-fingerprint", default=None)
    plan.add_argument("--frozen-body-decoder", type=Path, default=None)
    plan.add_argument("--frozen-body-decoder-fingerprint", default=None)
    plan.add_argument("--body-synergy-contract-fingerprint", default=None)
    plan.add_argument("--body-synergy-portable-core-fingerprint", default=None)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--dimensions", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument(
        "--decoder-types",
        nargs="+",
        default=["direct", "fixed_synergy", "synergy_residual"],
    )
    plan.add_argument(
        "--require-causal-interventions",
        action="store_true",
        default=False,
        help=(
            "Register causal rollout artifacts as mandatory analysis inputs. "
            "Planning never creates or runs those rollouts."
        ),
    )

    execute = subparsers.add_parser(
        "execute",
        help="explicitly run registered train -> closed-loop -> analysis-export jobs",
    )
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--run-name", action="append", default=[])
    execute.add_argument(
        "--stage",
        choices=("train", "full"),
        default="full",
        help="Run training only or the complete lifecycle (default: full).",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run checkpoint-bound closed-loop evaluation and offline analysis export",
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--run-name", action="append", default=[])
    evaluate.add_argument("--skip-closed-loop", action="store_true", default=False)
    evaluate.add_argument("--skip-analysis-export", action="store_true", default=False)

    causal_evaluate = subparsers.add_parser(
        "causal-evaluate",
        help=(
            "run the registered real-environment causal driver and seal its paired "
            "artifact for every selected sweep run"
        ),
    )
    causal_evaluate.add_argument("--output-dir", type=Path, required=True)
    causal_evaluate.add_argument("--shared-config", type=Path, required=True)
    causal_evaluate.add_argument("--run-name", action="append", default=[])

    finalize = subparsers.add_parser(
        "finalize-causal",
        help="second-pass export after every registered environment causal artifact exists",
    )
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--run-name", action="append", default=[])

    analyze = subparsers.add_parser("analyze", help="summarize completed sweep checkpoints")
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the exact registered dimension/decoder/seed matrix (default: true).",
    )
    analyze.add_argument("--num-bootstrap", type=int, default=2000)
    analyze.add_argument("--max-analysis-samples", type=int, default=1024)
    analyze.add_argument("--require-all-phases", action="store_true", default=False)
    analyze.add_argument(
        "--require-causal-interventions",
        action="store_true",
        default=False,
        help="Require separately fingerprinted environment-rollout causal effects.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        return _plan(args)
    if args.command == "execute":
        return _execute(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "causal-evaluate":
        return _causal_evaluate(args)
    if args.command == "finalize-causal":
        return _finalize_causal(args)
    return _analyze(args)


def _plan(args: argparse.Namespace) -> int:
    artifact = load_synergy_basis(args.synergy_basis)
    if artifact.manifest.get("signal_kind") != EXCITATION_SIGNAL_KIND:
        raise ValueError("latent sweep requires a physical_excitation_unit basis")
    region = str(artifact.manifest.get("region", ""))
    if region not in _COMPLETE_SWEEP_BASIS_REGIONS:
        raise ValueError(
            "latent sweep requires a complete regional_composite or "
            "primary hybrid_global_regional basis (whole_body is retained as "
            "an explicit comparator), never one regional component"
        )
    if region == "hybrid_global_regional":
        if artifact.manifest.get("hybrid_schema_version") != HYBRID_BASIS_SCHEMA_VERSION:
            raise ValueError("latent sweep hybrid_global_regional artifact has an unsupported schema")
        if artifact.manifest.get("artifact_role") != _PRIMARY_HYBRID_ARTIFACT_ROLE:
            raise ValueError(
                "latent sweep requires the primary hybrid_global_regional artifact, not an unpromoted hybrid candidate"
            )
    supplied = args.synergy_basis_fingerprint
    if supplied is not None and str(supplied) != artifact.fingerprint:
        raise ValueError("supplied synergy basis fingerprint differs from the formal artifact")
    frozen = None
    if any(str(value) != "direct" for value in args.decoder_types):
        if args.frozen_body_decoder is None:
            raise ValueError("synergy sweep requires --frozen-body-decoder from Stage-1")
        frozen = load_frozen_body_decoder(
            args.frozen_body_decoder,
            expected_artifact_fingerprint=(args.frozen_body_decoder_fingerprint),
            expected_portable_decoder_core_fingerprint=(args.body_synergy_portable_core_fingerprint),
        )
        contract = frozen.body_synergy_contract
        if args.body_synergy_contract_fingerprint != contract.contract_fingerprint:
            raise ValueError("supplied BodySynergyContractV2 fingerprint differs from frozen artifact")
        if contract.basis_fingerprint != artifact.fingerprint:
            raise ValueError("frozen decoder formal W fingerprint differs from analysis basis")
    validation_manifest = validate_dataset_manifest(args.val_dataset_dir)
    heldout_motion_paths = _manifest_validation_motion_paths(validation_manifest)
    specs = build_sweep_specs(
        base_config=args.base_config.resolve(),
        output_root=args.output_dir.resolve(),
        dimensions=args.dimensions,
        decoder_types=args.decoder_types,
        seeds=args.seeds,
        dataset_dir=args.dataset_dir.resolve(),
        val_dataset_dir=args.val_dataset_dir.resolve(),
        teacher_ckpt=args.teacher_ckpt.resolve(),
        teacher_promotion_manifest=args.teacher_promotion_manifest.resolve(),
        direct_bc_metrics=args.direct_bc_metrics.resolve(),
        direct_rollout_metrics=args.direct_rollout_metrics.resolve(),
        direct_promotion_evidence=args.direct_promotion_evidence.resolve(),
        closed_loop_correction_root=(
            None if args.closed_loop_correction_root is None else args.closed_loop_correction_root.resolve()
        ),
        heldout_motion_paths=heldout_motion_paths,
        synergy_basis_path=artifact.path.resolve(),
        synergy_basis_expected_fingerprint=artifact.fingerprint,
        frozen_body_decoder_path=(None if frozen is None else args.frozen_body_decoder.resolve()),
        frozen_body_decoder_expected_fingerprint=(None if frozen is None else frozen.artifact_fingerprint),
        body_synergy_contract_expected_fingerprint=(
            None if frozen is None else frozen.body_synergy_contract.contract_fingerprint
        ),
        body_synergy_portable_core_expected_fingerprint=(
            None if frozen is None else frozen.body_synergy_contract.portable_decoder_core_fingerprint
        ),
        residual_actuator_names=DEFAULT_RESIDUAL_NAMES,
        residual_alpha=0.05,
        require_causal_interventions=bool(args.require_causal_interventions),
        python_executable=sys.executable,
    )
    payload = {
        "schema_version": "latent_synergy_sweep_plan_v2",
        "synergy_basis_path": str(artifact.path.resolve()),
        "synergy_basis_fingerprint": artifact.fingerprint,
        "frozen_body_decoder_path": (None if frozen is None else str(args.frozen_body_decoder.resolve())),
        "frozen_body_decoder_fingerprint": (None if frozen is None else frozen.artifact_fingerprint),
        "body_synergy_contract_fingerprint": (
            None if frozen is None else frozen.body_synergy_contract.contract_fingerprint
        ),
        "body_synergy_portable_core_fingerprint": (
            None if frozen is None else frozen.body_synergy_contract.portable_decoder_core_fingerprint
        ),
        "source_dataset_fingerprint": artifact.manifest["source_dataset_fingerprint"],
        "teacher_checkpoint_fingerprint": artifact.manifest["teacher_checkpoint_fingerprint"],
        "lifecycle_inputs": {
            "train_dataset_dir": str(args.dataset_dir.resolve()),
            "validation_dataset_dir": str(args.val_dataset_dir.resolve()),
            "expected_validation_motion_count": 5,
            "heldout_motion_paths": heldout_motion_paths,
            "teacher_checkpoint": str(args.teacher_ckpt.resolve()),
            "teacher_promotion_manifest": str(args.teacher_promotion_manifest.resolve()),
            "direct_bc_metrics": str(args.direct_bc_metrics.resolve()),
            "direct_rollout_metrics": str(args.direct_rollout_metrics.resolve()),
            "direct_promotion_evidence": str(args.direct_promotion_evidence.resolve()),
            "closed_loop_correction_root": (
                None if args.closed_loop_correction_root is None else str(args.closed_loop_correction_root.resolve())
            ),
        },
        "closed_loop_correction_contract": {
            "enabled": args.closed_loop_correction_root is not None,
            "evidence_kind": "student_closed_loop_rollout_stage2_teacher_relabel",
            "dataset_schema": "distill_dataset_manifest_v2",
            "per_run_path": "<root>/<run_name>/dataset",
            "required_collection_request": {
                "student_policy_kind": "latent_checkpoint_prior_mean_lab",
                "teacher_relabel_target": "normalized_body_action",
                "closed_loop_state_source": "environment_student_visited_state",
            },
            "collection_is_implicit": False,
            "training_behavior": "append verified correction rows to the offline train split only",
        },
        "num_jobs": len(specs),
        "analysis_contract": {
            "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
            "path_per_run": "analysis_inputs.npz",
            "required_fields": list(ANALYSIS_INPUT_FIELDS),
            "optional_fields": [
                "decoder_synergy_coefficients",
                "causal_effects",
                "causal_effect_names",
            ],
            "causal_evidence_policy": (
                "optional by default; when supplied it must be a separately fingerprinted "
                "environment_rollout artifact. Offline decoder perturbations are not causal task evidence."
            ),
            "causal_interventions_required": bool(args.require_causal_interventions),
            "causal_lifecycle": (
                "bootstrap analysis export -> exact environment paired rollout -> "
                "causal artifact seal -> finalize-causal export -> analyze"
            ),
            "shapes": {
                "latents": "[N, latent_dim]",
                "synergy_coefficients": "[N, synergy_rank] NNLS target from teacher excitation and formal W",
                "target_synergy_coefficients": "[N, synergy_rank]",
                "decoder_jacobians": "[N, action_dim, latent_dim]",
                "phase_ids": "[N] integer 0..5",
                "train_mask": "[N] bool; false rows form held-out test",
                "sample_uids": "[N] stable aligned IDs shared across seeds",
                "teacher_physical_excitation": "[N, action_dim]",
                "baseline_physical_excitation": "[N, action_dim]",
                "perturbed_physical_excitation": "[N, directions, epsilons, action_dim]",
                "intervention_epsilons": "[epsilons] finite non-zero",
                "intervention_directions": "[directions, latent_dim] covariance-PC directions",
                "intervention_direction_names": "[directions]",
            },
        },
        "jobs": specs,
    }
    payload["plan_fingerprint"] = _canonical_json_sha256(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_train.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(item["training_command_shell"] for item in specs)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_closed_loop.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(item["closed_loop_command_shell"] for item in specs)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_analysis_export.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(item["analysis_export_command_shell"] for item in specs)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_causal_finalize.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(item["causal_finalize_command_shell"] for item in specs)
        + "\n",
        encoding="utf-8",
    )
    full_lifecycle = []
    for item in specs:
        full_lifecycle.extend(
            [
                item["training_command_shell"],
                item["closed_loop_command_shell"],
                item["analysis_export_command_shell"],
            ]
        )
    (args.output_dir / "run_full_lifecycle.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(full_lifecycle) + "\n",
        encoding="utf-8",
    )
    # Historical name remains the explicit training-only entrypoint.
    (args.output_dir / "run_sweep.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(item["training_command_shell"] for item in specs)
        + "\n",
        encoding="utf-8",
    )
    return 0


def _manifest_validation_motion_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for collection in manifest.get("collections") or []:
        contract = collection.get("contract") if isinstance(collection, dict) else None
        if not isinstance(contract, dict) or contract.get("split") != "val":
            continue
        for value in contract.get("motion_paths") or []:
            name = str(value)
            if name not in paths:
                paths.append(name)
    if len(paths) != 5:
        raise ValueError("validation dataset manifest must bind exactly five unique val motion paths")
    return paths


def _selected_plan_jobs(
    plan: dict[str, Any],
    requested_names: list[str],
) -> list[dict[str, Any]]:
    requested = {str(value) for value in requested_names}
    known = {str(job["run_name"]) for job in plan["jobs"]}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown sweep run names: {unknown}")
    return [job for job in plan["jobs"] if not requested or str(job["run_name"]) in requested]


def _execute(args: argparse.Namespace) -> int:
    plan = _load_and_validate_plan(args.output_dir)
    jobs = _selected_plan_jobs(plan, args.run_name)
    for job in jobs:
        checkpoint = Path(job["checkpoint_dir"])
        if checkpoint.exists():
            raise FileExistsError(
                f"registered checkpoint already exists for {job['run_name']}; "
                "use the evaluate subcommand for an existing trained checkpoint"
            )
        subprocess.run(job["training_command"], check=True)
        if args.stage == "full":
            subprocess.run(job["closed_loop_command"], check=True)
            subprocess.run(job["analysis_export_command"], check=True)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    plan = _load_and_validate_plan(args.output_dir)
    jobs = _selected_plan_jobs(plan, args.run_name)
    if args.skip_closed_loop and args.skip_analysis_export:
        raise ValueError("evaluate cannot skip both lifecycle stages")
    for job in jobs:
        checkpoint = Path(job["checkpoint_dir"])
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"training checkpoint is missing for {job['run_name']}: {checkpoint}")
        if not args.skip_closed_loop:
            subprocess.run(job["closed_loop_command"], check=True)
        if not args.skip_analysis_export:
            subprocess.run(job["analysis_export_command"], check=True)
    return 0


def _causal_evaluate(args: argparse.Namespace) -> int:
    """Run and seal exact paired environment interventions for registered jobs.

    The shared file contains only adapter-wide options.  Checkpoint, dataset,
    bootstrap-analysis, and teacher paths are injected from the fingerprinted
    sweep plan for each run, so a hand-edited job cannot silently evaluate a
    different policy or sample set.
    """

    plan = _load_and_validate_plan(args.output_dir)
    if (plan.get("analysis_contract") or {}).get("causal_interventions_required") is not True:
        raise ValueError("causal-evaluate requires a sweep planned with --require-causal-interventions")
    shared = _load_causal_adapter_shared_config(args.shared_config)
    jobs = _selected_plan_jobs(plan, args.run_name)
    completed: list[dict[str, Any]] = []
    for job in jobs:
        completed.append(
            _evaluate_and_seal_causal_job(
                job,
                plan=plan,
                shared=shared,
            )
        )
    report = {
        "schema_version": CAUSAL_EVALUATION_MANIFEST_SCHEMA_VERSION,
        "sweep_plan_fingerprint": plan["plan_fingerprint"],
        "shared_config_path": str(Path(args.shared_config).resolve()),
        "shared_config_sha256": _file_sha256(Path(args.shared_config)),
        "adapter_import": shared["adapter_import"],
        "base_seed": int(shared["base_seed"]),
        "num_registered_runs": len(plan["jobs"]),
        "num_evaluated_runs": len(completed),
        "selected_run_names": [str(job["run_name"]) for job in jobs],
        "runs": completed,
    }
    report["manifest_fingerprint"] = _canonical_json_sha256(report)
    _write_generated_json(
        Path(args.output_dir) / "causal_evaluation_manifest.json",
        report,
        require_exact_if_exists=False,
    )
    return 0


def _load_causal_adapter_shared_config(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("causal adapter shared config must be a JSON object")
    allowed = {"schema_version", "adapter_import", "base_seed", "adapter_config"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"causal adapter shared config has unknown fields: {unknown}")
    if payload.get("schema_version") != CAUSAL_ADAPTER_SHARED_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"causal adapter shared config schema must be {CAUSAL_ADAPTER_SHARED_CONFIG_SCHEMA_VERSION}")
    adapter_config = payload.get("adapter_config")
    if not isinstance(adapter_config, dict):
        raise ValueError("causal adapter shared config requires adapter_config object")
    ambiguous = sorted(set(adapter_config) & _CAUSAL_ADAPTER_PLAN_BOUND_FIELDS)
    if ambiguous:
        raise ValueError(f"causal adapter plan-bound fields cannot be supplied in shared config: {ambiguous}")
    adapter_import = str(payload.get("adapter_import", DEFAULT_CAUSAL_ADAPTER_IMPORT)).strip()
    if not adapter_import or ":" not in adapter_import or adapter_import == "replay-record":
        raise ValueError("causal-evaluate requires a real module:factory adapter, never replay-record")
    base_seed = int(payload.get("base_seed", 20260713))
    if base_seed < 0:
        raise ValueError("causal adapter base_seed must be non-negative")
    return {
        "adapter_import": adapter_import,
        "base_seed": base_seed,
        "adapter_config": dict(adapter_config),
    }


def _evaluate_and_seal_causal_job(
    job: dict[str, Any],
    *,
    plan: dict[str, Any],
    shared: dict[str, Any],
) -> dict[str, Any]:
    run_name = str(job["run_name"])
    run_dir = Path(job["output_dir"])
    checkpoint = Path(job["checkpoint_dir"])
    analysis_inputs = run_dir / "analysis_inputs.npz"
    analysis_manifest = analysis_inputs.with_suffix(".json")
    required = (checkpoint, analysis_inputs, analysis_manifest)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"causal evaluation inputs are incomplete for {run_name}: {missing}")
    analysis_sidecar = load_json_strict(analysis_manifest)
    if not isinstance(analysis_sidecar, dict):
        raise ValueError(f"bootstrap analysis manifest is not an object for {run_name}")
    sidecar_unsigned = {key: value for key, value in analysis_sidecar.items() if key != "manifest_fingerprint"}
    if (
        analysis_sidecar.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION
        or analysis_sidecar.get("manifest_fingerprint") != _canonical_json_sha256(sidecar_unsigned)
        or analysis_sidecar.get("npz_sha256") != _file_sha256(analysis_inputs)
        or Path(str(analysis_sidecar.get("checkpoint_dir", ""))).resolve() != checkpoint.resolve()
    ):
        raise ValueError(f"bootstrap analysis binding is invalid for {run_name}")
    causal_status = analysis_sidecar.get("causal_evidence")
    if not isinstance(causal_status, dict):
        raise ValueError(f"bootstrap analysis manifest lacks causal policy for {run_name}")
    if causal_status.get("causal_rollout_verified") is True:
        raise ValueError(
            f"analysis inputs are already causal-finalized for {run_name}; "
            "do not rerun causal-evaluate over finalized inputs"
        )

    lifecycle = plan.get("lifecycle_inputs") or {}
    adapter_config = dict(shared["adapter_config"])
    adapter_config.update(
        {
            "latent_checkpoint": str(checkpoint.resolve()),
            "teacher_ckpt": str(Path(lifecycle["teacher_checkpoint"]).resolve()),
            "dataset_dir": str(Path(lifecycle["train_dataset_dir"]).resolve()),
            "val_dataset_dir": str(Path(lifecycle["validation_dataset_dir"]).resolve()),
            "analysis_inputs": str(analysis_inputs.resolve()),
            "analysis_manifest": str(analysis_manifest.resolve()),
        }
    )
    rollout_dir = run_dir / "causal_rollouts"
    job_path = run_dir / "causal_rollout_job.json"
    job_payload = {
        "schema_version": JOB_SCHEMA_VERSION,
        "analysis_inputs": str(analysis_inputs.resolve()),
        "analysis_manifest": str(analysis_manifest.resolve()),
        "output_dir": str(rollout_dir.resolve()),
        "base_seed": int(shared["base_seed"]),
        "adapter_import": str(shared["adapter_import"]),
        "adapter_config": adapter_config,
    }
    _write_generated_json(job_path, job_payload, require_exact_if_exists=True)

    baseline = rollout_dir / BASELINE_FILENAME
    perturbed = rollout_dir / PERTURBED_FILENAME
    rollout_manifest = rollout_dir / MANIFEST_FILENAME
    rollout_outputs = (baseline, perturbed, rollout_manifest)
    if rollout_dir.exists():
        missing_rollout = [str(path) for path in rollout_outputs if not path.is_file()]
        if missing_rollout:
            raise ValueError(f"partial causal rollout output exists for {run_name}: {missing_rollout}")
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "musclemimic.latent_muscle.causal_rollout_driver",
                "evaluate",
                "--job-config",
                str(job_path),
            ],
            check=True,
        )
        missing_rollout = [str(path) for path in rollout_outputs if not path.is_file()]
        if missing_rollout:
            raise ValueError(
                f"causal rollout driver did not publish complete outputs for {run_name}: {missing_rollout}"
            )

    causal_npz = Path(job["causal_interventions_npz"])
    causal_manifest = Path(job["causal_interventions_manifest"])
    if causal_npz.exists() != causal_manifest.exists():
        raise ValueError(f"partial sealed causal artifact exists for {run_name}")
    if not causal_npz.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "musclemimic.latent_muscle.causal_rollout_artifact",
                "--analysis-inputs",
                str(analysis_inputs),
                "--analysis-manifest",
                str(analysis_manifest),
                "--baseline-records",
                str(baseline),
                "--perturbed-records",
                str(perturbed),
                "--rollout-manifest",
                str(rollout_manifest),
                "--output-npz",
                str(causal_npz),
                "--output-manifest",
                str(causal_manifest),
            ],
            check=True,
        )
    sealed = validate_causal_rollout_artifact(causal_npz, causal_manifest)
    availability = sealed.get("outcome_availability")
    missing_diagnostic = [
        name
        for name in STAGE2_DIAGNOSTIC_CAUSAL_OUTCOMES
        if not isinstance(availability, dict) or availability.get(name) is not True
    ]
    if missing_diagnostic:
        raise ValueError(f"Stage-2 causal diagnostic outcomes are unavailable for {run_name}: {missing_diagnostic}")
    if sealed.get("analysis_inputs_sha256") != _file_sha256(analysis_inputs):
        raise ValueError(f"sealed causal evidence differs from bootstrap inputs for {run_name}")
    if sealed.get("analysis_manifest_fingerprint") != analysis_sidecar.get("manifest_fingerprint"):
        raise ValueError(f"sealed causal evidence differs from bootstrap manifest for {run_name}")
    source_manifest = load_json_strict(rollout_manifest)
    source_unsigned = {key: value for key, value in source_manifest.items() if key != "manifest_fingerprint"}
    if (
        source_manifest.get("manifest_fingerprint") != _canonical_json_sha256(source_unsigned)
        or sealed.get("paired_rollout_source_manifest_fingerprint") != source_manifest.get("manifest_fingerprint")
        or sealed.get("baseline_records_sha256") != _file_sha256(baseline)
        or sealed.get("perturbed_records_sha256") != _file_sha256(perturbed)
    ):
        raise ValueError(f"sealed causal evidence source records changed for {run_name}")
    return {
        "run_name": run_name,
        "job_config_path": str(job_path.resolve()),
        "job_config_fingerprint": _canonical_json_sha256(job_payload),
        "rollout_manifest_path": str(rollout_manifest.resolve()),
        "rollout_manifest_sha256": _file_sha256(rollout_manifest),
        "causal_interventions_path": str(causal_npz.resolve()),
        "causal_interventions_sha256": _file_sha256(causal_npz),
        "causal_manifest_path": str(causal_manifest.resolve()),
        "causal_manifest_fingerprint": sealed["manifest_fingerprint"],
        "checkpoint_fingerprint": sealed["checkpoint_fingerprint"],
        "environment_fingerprint": sealed["environment_fingerprint"],
        "policy_abi_hash": sealed["policy_abi_hash"],
        "outcome_availability": availability,
        "stage2_diagnostic_outcomes_complete": True,
        "task_outcomes_complete": all(value is True for value in availability.values()),
        "passed": True,
    }


def _write_generated_json(
    path: Path,
    payload: dict[str, Any],
    *,
    require_exact_if_exists: bool,
) -> None:
    if path.exists() and require_exact_if_exists:
        current = load_json_strict(path)
        if current != payload:
            raise ValueError(f"generated causal job config changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finalize_causal(args: argparse.Namespace) -> int:
    """Merge sealed environment-rollout effects after bootstrap sample export."""

    plan = _load_and_validate_plan(args.output_dir)
    if (plan.get("analysis_contract") or {}).get("causal_interventions_required") is not True:
        raise ValueError("finalize-causal requires a sweep planned with --require-causal-interventions")
    jobs = _selected_plan_jobs(plan, args.run_name)
    for job in jobs:
        checkpoint = Path(job["checkpoint_dir"])
        bootstrap = Path(job["output_dir"]) / "analysis_inputs.npz"
        causal_npz = Path(job["causal_interventions_npz"])
        causal_manifest = Path(job["causal_interventions_manifest"])
        missing = [
            str(path)
            for path in (checkpoint, bootstrap, bootstrap.with_suffix(".json"), causal_npz, causal_manifest)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"causal finalize inputs are incomplete for {job['run_name']}: {missing}")
        subprocess.run(job["causal_finalize_command"], check=True)
        finalized = json.loads(bootstrap.with_suffix(".json").read_text(encoding="utf-8"))
        evidence = finalized.get("causal_evidence") or {}
        if (
            evidence.get("status") != "verified_environment_rollout"
            or evidence.get("causal_rollout_verified") is not True
        ):
            raise ValueError(f"causal finalize did not seal verified evidence for {job['run_name']}")
    return 0


def _load_and_validate_plan(output_dir: Path) -> dict[str, Any]:
    plan_path = Path(output_dir) / "sweep_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"sweep plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "latent_synergy_sweep_plan_v2":
        raise ValueError("sweep lifecycle requires latent_synergy_sweep_plan_v2")
    supplied_fingerprint = plan.get("plan_fingerprint")
    fingerprint_payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if supplied_fingerprint != _canonical_json_sha256(fingerprint_payload):
        raise ValueError("sweep plan fingerprint mismatch")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs or int(plan.get("num_jobs", -1)) != len(jobs):
        raise ValueError("sweep plan contains no exact job matrix")
    names = [str(job.get("run_name")) for job in jobs]
    keys = [
        (
            int(job.get("latent_dim", -1)),
            str(job.get("decoder_type", "")),
            int(job.get("seed", -1)),
        )
        for job in jobs
    ]
    if len(set(names)) != len(names) or len(set(keys)) != len(keys):
        raise ValueError("sweep plan has duplicate run names or matrix cells")
    dimensions = sorted({key[0] for key in keys})
    decoders = sorted({key[1] for key in keys})
    seeds = sorted({key[2] for key in keys})
    expected = {(dimension, decoder, seed) for dimension in dimensions for decoder in decoders for seed in seeds}
    if set(keys) != expected or any(value <= 0 for value in dimensions) or any(seed < 0 for seed in seeds):
        raise ValueError("sweep plan is not a complete Cartesian job matrix")
    analysis_contract = plan.get("analysis_contract")
    if not isinstance(analysis_contract, dict) or not isinstance(
        analysis_contract.get("causal_interventions_required", False), bool
    ):
        raise ValueError("sweep plan causal evidence policy is malformed")
    lifecycle = plan.get("lifecycle_inputs")
    required_lifecycle_fields = {
        "train_dataset_dir",
        "validation_dataset_dir",
        "teacher_checkpoint",
    }
    if not isinstance(lifecycle, dict) or any(
        not str(lifecycle.get(name, "")).strip() for name in required_lifecycle_fields
    ):
        raise ValueError("sweep plan lacks causal adapter lifecycle inputs")
    required_job_fields = {
        "training_command",
        "closed_loop_command",
        "analysis_export_command",
        "causal_finalize_command",
        "causal_interventions_npz",
        "causal_interventions_manifest",
        "checkpoint_dir",
        "output_dir",
        "synergy_basis_expected_fingerprint",
    }
    for job in jobs:
        missing = sorted(required_job_fields - set(job))
        if missing:
            raise ValueError(f"sweep job {job.get('run_name')} is missing {missing}")
        if job["synergy_basis_expected_fingerprint"] != plan.get("synergy_basis_fingerprint"):
            raise ValueError("sweep job formal basis fingerprint differs from plan")
    return plan


def _analyze(args: argparse.Namespace) -> int:
    plan = _load_and_validate_plan(args.output_dir)
    require_causal = bool(args.require_causal_interventions) or bool(
        (plan.get("analysis_contract") or {}).get("causal_interventions_required", False)
    )
    jobs = plan["jobs"]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    analysis_records: list[dict[str, Any]] = []
    cross_seed_inputs: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    basis_artifact = load_synergy_basis(plan["synergy_basis_path"])
    if basis_artifact.fingerprint != plan.get("synergy_basis_fingerprint"):
        raise ValueError("formal synergy basis content no longer matches the sweep plan")
    if basis_artifact.manifest.get("source_dataset_fingerprint") != plan.get(
        "source_dataset_fingerprint"
    ) or basis_artifact.manifest.get("teacher_checkpoint_fingerprint") != plan.get("teacher_checkpoint_fingerprint"):
        raise ValueError("formal synergy basis source provenance differs from the sweep plan")
    for job in jobs:
        try:
            record = _load_completed_run_record(
                job,
                plan=plan,
                basis_artifact=basis_artifact,
            )
            analysis_path = Path(job["output_dir"]) / "analysis_inputs.npz"
            analysis_manifest = _validate_analysis_input_manifest(
                analysis_path,
                record=record,
                plan=plan,
                require_causal=require_causal,
            )
            causal_evidence = analysis_manifest.get("causal_evidence") or {}
            record["stage2_diagnostic_outcomes_complete"] = bool(
                causal_evidence.get("stage2_diagnostic_outcomes_complete")
            )
            record["task_outcomes_complete"] = bool(causal_evidence.get("task_outcomes_complete"))
            analysis_payload, cross_payload = _analyze_run_inputs(
                analysis_path,
                basis=basis_artifact.basis,
                checkpoint_fingerprint=record["checkpoint_fingerprint"],
                synergy_basis_fingerprint=basis_artifact.fingerprint,
                max_samples=int(args.max_analysis_samples),
                require_all_phases=bool(args.require_all_phases),
                require_causal=require_causal,
            )
            record["analysis_complete"] = True
            record["analysis_input_manifest_fingerprint"] = analysis_manifest["manifest_fingerprint"]
            record["analysis"] = analysis_payload
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append({"run_name": str(job["run_name"]), "reason": str(exc)})
            continue
        analysis_payload.update(
            {
                "run_name": str(job["run_name"]),
                "latent_dim": int(job["latent_dim"]),
                "decoder_type": str(job["decoder_type"]),
                "seed": int(job["seed"]),
            }
        )
        analysis_payload["analysis_fingerprint"] = _canonical_json_sha256(analysis_payload)
        analysis_output = Path(job["output_dir"]) / "analysis_metrics.json"
        analysis_output.write_text(
            json.dumps(analysis_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        analysis_records.append(analysis_payload)
        group_key = (int(job["latent_dim"]), str(job["decoder_type"]))
        cross_seed_inputs.setdefault(group_key, {})[str(job["seed"])] = cross_payload
    if not records:
        raise ValueError("sweep analysis found no completed checkpoints")
    if args.require_complete and failures:
        raise ValueError(f"sweep analysis has incomplete jobs: {failures}")
    if len(records) != len(jobs):
        raise ValueError(
            "promotion requires every registered run to have checkpoint, closed-loop, and analysis evidence"
        )
    split_fingerprints = {record["motion_split_fingerprint"] for record in records}
    validation_fingerprints = {record["validation_dataset_fingerprint"] for record in records}
    if len(split_fingerprints) != 1 or None in split_fingerprints:
        raise ValueError("all sweep runs must share one seed-independent motion split")
    if len(validation_fingerprints) != 1 or None in validation_fingerprints:
        raise ValueError("all sweep runs must share one immutable validation dataset")
    cross_seed_analysis = _cross_seed_reports(
        cross_seed_inputs,
        expected_seeds={str(seed) for seed in sorted({int(job["seed"]) for job in jobs})},
        require_causal=require_causal,
    )
    report = build_comparison_report(
        records,
        required_metrics=ANALYSIS_METRICS,
        num_bootstrap=int(args.num_bootstrap),
    )
    report.pop("report_fingerprint", None)
    report["sweep_plan_fingerprint"] = plan["plan_fingerprint"]
    report["failed_or_missing_runs"] = failures
    report["num_failed_or_missing_runs"] = len(failures)
    report["analysis_contract"] = plan.get("analysis_contract")
    report["per_run_analysis"] = analysis_records
    report["cross_seed_analysis"] = cross_seed_analysis
    promotion_metrics = _select_promotion_model(
        records,
        plan,
        cross_seed_analysis=cross_seed_analysis,
        failures=failures,
        require_causal_interventions=require_causal,
    )
    report["evidence_policy"] = {
        "offline_intervention_verified": True,
        "causal_rollout_required": require_causal,
        "causal_rollout_verified": bool(promotion_metrics["causal_rollout_verified"]),
    }
    report["selected_models"] = promotion_metrics["selected_models"]
    report["selected_model"] = promotion_metrics["selected_model"]
    report["selected_artifact"] = _materialize_selected_artifact(
        args.output_dir,
        promotion_metrics=promotion_metrics,
        plan=plan,
    )
    report["report_fingerprint"] = _canonical_json_sha256(report)
    output_json = args.output_dir / "latent_synergy_report.json"
    output_md = args.output_dir / "latent_synergy_report.md"
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(report_markdown(report), encoding="utf-8")
    return 0


def _materialize_selected_artifact(
    output_dir: Path,
    *,
    promotion_metrics: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Expose independently selected direct and synergy checkpoints atomically."""

    supplied_promotion_fingerprint = promotion_metrics.get("promotion_metrics_fingerprint")
    promotion_payload = {
        key: value for key, value in promotion_metrics.items() if key != "promotion_metrics_fingerprint"
    }
    if supplied_promotion_fingerprint != _canonical_json_sha256(promotion_payload):
        raise ValueError("promotion metrics fingerprint mismatch")
    selected_models = promotion_metrics.get("selected_models")
    if not isinstance(selected_models, dict) or not selected_models:
        # Compatibility for pre-v2 callers; production plans containing both
        # families are required to provide the explicit two-entry mapping.
        selected = promotion_metrics.get("selected_model")
        if not isinstance(selected, dict):
            raise ValueError("promotion metrics contain no selected checkpoints")
        family = "best_direct" if selected.get("decoder_type") == "direct" else "best_synergy"
        selected_models = {family: selected}
    unknown = sorted(set(selected_models) - {"best_direct", "best_synergy"})
    if unknown:
        raise ValueError(f"unknown selected checkpoint families: {unknown}")
    planned_decoders = {str(job["decoder_type"]) for job in plan.get("jobs", [])}
    if "direct" in planned_decoders and "best_direct" not in selected_models:
        raise ValueError("selection manifest requires the registered best_direct comparator")
    if any(name != "direct" for name in planned_decoders) and ("best_synergy" not in selected_models):
        raise ValueError("selection manifest requires the registered best_synergy checkpoint")

    identity_fields = (
        "dataset_fingerprint",
        "validation_dataset_fingerprint",
        "motion_split_fingerprint",
    )
    if len(selected_models) > 1:
        identities = {tuple(model.get(field) for field in identity_fields) for model in selected_models.values()}
        if len(identities) != 1:
            raise ValueError("best_direct and best_synergy do not share one dataset/validation/split identity")

    root = Path(output_dir) / "selected"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_entries: dict[str, dict[str, Any]] = {}
    for family in ("best_direct", "best_synergy"):
        if family not in selected_models:
            continue
        selected = selected_models[family]
        source = Path(selected["checkpoint_dir"]).resolve()
        if not source.is_dir():
            raise ValueError(f"selected checkpoint directory is missing: {source}")
        expected_fingerprint = str(selected["checkpoint_fingerprint"])
        if latent_checkpoint_fingerprint(source) != expected_fingerprint:
            raise ValueError(f"{family} checkpoint fingerprint changed before materialization")
        stable = root / family
        _replace_verified_checkpoint_link(
            stable,
            source=source,
            expected_fingerprint=expected_fingerprint,
        )
        checkpoint_entries[family] = {
            "stable_checkpoint_path": str(stable.absolute()),
            "checkpoint_path": str(stable.resolve()),
            "checkpoint_fingerprint": expected_fingerprint,
            "run_name": selected["run_name"],
            "latent_dim": int(selected["latent_dim"]),
            "decoder_type": selected["decoder_type"],
            "seed": int(selected["seed"]),
            "formal_synergy_basis_fingerprint": selected["formal_synergy_basis_fingerprint"],
            "runtime_synergy_basis_fingerprint": selected.get("runtime_synergy_basis_fingerprint"),
            "runtime_synergy_basis_source_fingerprint": selected.get("runtime_synergy_basis_source_fingerprint"),
            "dataset_fingerprint": selected["dataset_fingerprint"],
            "validation_dataset_fingerprint": selected["validation_dataset_fingerprint"],
            "motion_split_fingerprint": selected["motion_split_fingerprint"],
        }

    alias_family = "best_synergy" if "best_synergy" in checkpoint_entries else "best_direct"
    alias = root / "latent_checkpoint"
    _replace_verified_checkpoint_link(
        alias,
        source=Path(checkpoint_entries[alias_family]["checkpoint_path"]),
        expected_fingerprint=checkpoint_entries[alias_family]["checkpoint_fingerprint"],
    )

    promotion_path = Path(output_dir) / "promotion_metrics.json"
    temporary_promotion = Path(output_dir) / f".promotion_metrics.tmp-{os.getpid()}"
    temporary_promotion.write_text(
        json.dumps(promotion_metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_promotion, promotion_path)
    manifest = {
        "schema_version": "latent_synergy_selected_checkpoints_v2",
        "checkpoints": checkpoint_entries,
        "compatibility_alias": {
            "stable_checkpoint_path": str(alias.absolute()),
            "target_family": alias_family,
            "checkpoint_fingerprint": checkpoint_entries[alias_family]["checkpoint_fingerprint"],
        },
        "sweep_plan_fingerprint": plan["plan_fingerprint"],
        "promotion_metrics_path": str(promotion_path.resolve()),
        "promotion_metrics_sha256": _file_sha256(promotion_path),
        "promotion_metrics_fingerprint": supplied_promotion_fingerprint,
        "selection_rule": promotion_metrics["selection_rule"],
    }
    manifest["selection_manifest_fingerprint"] = _canonical_json_sha256(manifest)
    manifest_path = root / "selection_manifest.json"
    temporary_manifest = root / f".selection_manifest.tmp-{os.getpid()}"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    validated = validate_selected_artifact(manifest_path)
    return {
        "stable_checkpoint_paths": {
            family: entry["stable_checkpoint_path"] for family, entry in checkpoint_entries.items()
        },
        "stable_checkpoint_path": str(alias.absolute()),
        "compatibility_alias_target": alias_family,
        "selection_manifest_path": str(manifest_path.resolve()),
        "selection_manifest_fingerprint": validated["selection_manifest_fingerprint"],
    }


def _replace_verified_checkpoint_link(
    link: Path,
    *,
    source: Path,
    expected_fingerprint: str,
) -> None:
    """Atomically replace a stable symlink without touching its source tree."""

    source = source.resolve()
    if not source.is_dir() or latent_checkpoint_fingerprint(source) != str(expected_fingerprint):
        raise ValueError(f"checkpoint source is missing or changed: {source}")
    if os.path.lexists(link) and not link.is_symlink():
        if not link.is_dir() or latent_checkpoint_fingerprint(link) != str(expected_fingerprint):
            raise ValueError(f"{link} is a non-matching non-symlink; refusing to replace it")
        return
    temporary = link.parent / f".{link.name}.tmp-{os.getpid()}"
    if os.path.lexists(temporary):
        temporary.unlink()
    relative_source = os.path.relpath(source, start=link.parent.resolve())
    os.symlink(relative_source, temporary, target_is_directory=True)
    os.replace(temporary, link)
    if not link.is_dir() or latent_checkpoint_fingerprint(link) != str(expected_fingerprint):
        raise ValueError(f"materialized checkpoint failed verification: {link}")


def validate_selected_artifact(manifest_path: str | Path) -> dict[str, Any]:
    """Validate the complete v2 selection seal, both checkpoints, and alias."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "latent_synergy_selected_checkpoints_v2":
        raise ValueError("unsupported latent selection manifest schema")
    supplied = manifest.get("selection_manifest_fingerprint")
    payload = {key: value for key, value in manifest.items() if key != "selection_manifest_fingerprint"}
    if supplied != _canonical_json_sha256(payload):
        raise ValueError("latent selection manifest fingerprint mismatch")
    promotion_path = Path(manifest["promotion_metrics_path"])
    if not promotion_path.is_file() or manifest.get("promotion_metrics_sha256") != _file_sha256(promotion_path):
        raise ValueError("latent promotion metrics content changed after selection")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion_payload = {key: value for key, value in promotion.items() if key != "promotion_metrics_fingerprint"}
    promotion_fingerprint = _canonical_json_sha256(promotion_payload)
    if not (
        promotion.get("promotion_metrics_fingerprint")
        == promotion_fingerprint
        == manifest.get("promotion_metrics_fingerprint")
    ):
        raise ValueError("latent selection promotion fingerprint mismatch")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("latent selection manifest has no checkpoints")
    for family, entry in checkpoints.items():
        if family not in {"best_direct", "best_synergy"}:
            raise ValueError(f"unknown latent selection family: {family}")
        stable = Path(entry["stable_checkpoint_path"])
        expected = str(entry["checkpoint_fingerprint"])
        if not stable.is_dir() or latent_checkpoint_fingerprint(stable) != expected:
            raise ValueError(f"selected {family} checkpoint no longer matches its seal")
        promoted = (promotion.get("selected_models") or {}).get(family)
        if not isinstance(promoted, dict) or promoted.get("checkpoint_fingerprint") != expected:
            raise ValueError(f"selected {family} is not bound to promotion metrics")
    alias = manifest.get("compatibility_alias") or {}
    family = alias.get("target_family")
    if family not in checkpoints:
        raise ValueError("latent compatibility alias target is absent")
    alias_path = Path(alias["stable_checkpoint_path"])
    expected = checkpoints[family]["checkpoint_fingerprint"]
    if not alias_path.is_dir() or latent_checkpoint_fingerprint(alias_path) != expected:
        raise ValueError("latent compatibility alias does not match its selected target")
    return manifest


def _load_completed_run_record(
    job: dict[str, Any],
    *,
    plan: dict[str, Any],
    basis_artifact: Any,
) -> dict[str, Any]:
    checkpoint = Path(job["checkpoint_dir"])
    required = (
        checkpoint / "eval_metrics.json",
        checkpoint / "latent_config.yaml",
        checkpoint / "training_provenance.json",
        checkpoint / "motion_split.json",
        checkpoint / "closed_loop_metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing lifecycle artifacts {missing}")
    runtime = load_latent_runtime(checkpoint)
    config = dict(runtime.config)
    provenance = dict(runtime.training_provenance or {})
    metrics = json.loads((checkpoint / "eval_metrics.json").read_text(encoding="utf-8"))
    closed = json.loads((checkpoint / "closed_loop_metrics.json").read_text(encoding="utf-8"))
    validate_closed_loop_promotion_report(
        closed,
        checkpoint_dir=checkpoint,
        require_seal=True,
    )
    if (
        int(config.get("latent_dim", -1)) != int(job["latent_dim"])
        or str(config.get("decoder_type", "direct")) != str(job["decoder_type"])
        or int(config.get("seed", -1)) != int(job["seed"])
    ):
        raise ValueError("checkpoint config differs from its registered sweep job")
    lifecycle = plan["lifecycle_inputs"]
    for config_key, lifecycle_key in (
        ("dataset_dir", "train_dataset_dir"),
        ("val_dataset_dir", "validation_dataset_dir"),
        ("teacher_ckpt", "teacher_checkpoint"),
        ("teacher_promotion_manifest", "teacher_promotion_manifest"),
        ("direct_bc_metrics_path", "direct_bc_metrics"),
    ):
        configured = config.get(config_key)
        if configured is None or Path(configured).resolve() != Path(lifecycle[lifecycle_key]).resolve():
            raise ValueError(f"checkpoint config {config_key} differs from the sweep lifecycle input")
    correction_dir = job.get("closed_loop_correction_dataset_dir")
    correction_manifest_path = job.get("closed_loop_correction_manifest")
    configured_correction_dir = config.get("closed_loop_correction_dataset_dir")
    configured_correction_manifest = config.get("closed_loop_correction_manifest")
    if correction_dir is None:
        if configured_correction_dir is not None or configured_correction_manifest is not None:
            raise ValueError("latent checkpoint used unregistered closed-loop correction data")
    else:
        if (
            configured_correction_dir is None
            or configured_correction_manifest is None
            or Path(configured_correction_dir).resolve() != Path(correction_dir).resolve()
            or Path(configured_correction_manifest).resolve() != Path(correction_manifest_path).resolve()
        ):
            raise ValueError("latent checkpoint closed-loop correction paths differ from its sweep job")
        correction_file = Path(correction_manifest_path)
        correction_provenance = provenance.get("closed_loop_correction_dataset_manifest") or {}
        if not correction_file.is_file() or correction_provenance.get("manifest_fingerprint") != provenance.get(
            "closed_loop_correction_dataset_manifest_fingerprint"
        ):
            raise ValueError("latent checkpoint correction dataset provenance is incomplete")
    teacher_sha = (provenance.get("teacher_checkpoint") or {}).get("sha256")
    if teacher_sha != plan.get("teacher_checkpoint_fingerprint"):
        raise ValueError("latent checkpoint teacher differs from the formal synergy basis")
    direct_bc = provenance.get("direct_bc_metrics") or {}
    direct_path = Path(lifecycle["direct_bc_metrics"])
    if (
        direct_bc.get("path") != str(direct_path.resolve())
        or not direct_path.is_file()
        or direct_bc.get("sha256") != _file_sha256(direct_path)
    ):
        raise ValueError("latent checkpoint direct-BC baseline binding is invalid")
    split = json.loads((checkpoint / "motion_split.json").read_text(encoding="utf-8"))
    if (
        split.get("schema_version") != "motion_split_v2"
        or split.get("mode") != "explicit_dataset_directories"
        or split.get("split_seed") is not None
        or len(split.get("val_motion_ids") or []) != int(lifecycle["expected_validation_motion_count"])
    ):
        raise ValueError("checkpoint did not use the fixed five-motion validation split")
    split_payload = {key: value for key, value in split.items() if key != "split_fingerprint"}
    if split.get("split_fingerprint") != _canonical_json_sha256(split_payload):
        raise ValueError("checkpoint motion split fingerprint mismatch")
    validation_fingerprint = provenance.get("validation_dataset_manifest_fingerprint")
    if (
        split.get("train_dataset_manifest_fingerprint") != provenance.get("dataset_manifest_fingerprint")
        or split.get("val_dataset_manifest_fingerprint") != validation_fingerprint
    ):
        raise ValueError("checkpoint split and dataset provenance fingerprints differ")
    basis_binding = validate_runtime_basis_binding(
        runtime,
        formal_basis_fingerprint=basis_artifact.fingerprint,
    )
    if closed.get("analysis_synergy_basis_fingerprint") != basis_artifact.fingerprint:
        raise ValueError("closed-loop Jacobian basis differs from the sweep formal basis")
    if closed.get("heldout_motion_paths") != lifecycle.get("heldout_motion_paths"):
        raise ValueError("closed-loop held-out motion paths differ from the canonical validation manifest")
    if closed.get("promotion", {}).get("passed") is not True:
        raise ValueError("closed-loop production promotion did not pass")
    return {
        "run_name": str(job["run_name"]),
        "latent_dim": int(job["latent_dim"]),
        "decoder_type": str(job["decoder_type"]),
        "seed": int(job["seed"]),
        "checkpoint_fingerprint": runtime.checkpoint_fingerprint,
        "dataset_fingerprint": provenance.get("dataset_manifest_fingerprint"),
        "validation_dataset_fingerprint": validation_fingerprint,
        "motion_split_fingerprint": split.get("split_fingerprint"),
        "teacher_checkpoint_sha256": teacher_sha,
        "runtime_synergy_basis_fingerprint": basis_binding.get("runtime_synergy_basis_fingerprint"),
        "runtime_synergy_basis_source_fingerprint": basis_binding.get("runtime_synergy_basis_source_fingerprint"),
        "synergy_basis_fingerprint": (None if str(job["decoder_type"]) == "direct" else basis_artifact.fingerprint),
        "synergy_basis_expected_fingerprint": basis_binding["config_synergy_basis_expected_fingerprint"],
        "basis_binding_verified": bool(basis_binding.get("verified")),
        "checkpoint_dir": str(checkpoint.resolve()),
        "closed_loop": closed,
        "metrics": metrics,
        "analysis_complete": False,
    }


def _validate_analysis_input_manifest(
    path: Path,
    *,
    record: dict[str, Any],
    plan: dict[str, Any],
    require_causal: bool,
) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"missing analysis input or sidecar: {path}, {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        raise ValueError("analysis input sidecar has an unsupported schema")
    supplied = manifest.get("manifest_fingerprint")
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    if supplied != _canonical_json_sha256(payload):
        raise ValueError("analysis input sidecar fingerprint mismatch")
    if manifest.get("npz_sha256") != _file_sha256(path):
        raise ValueError("analysis input NPZ hash mismatch")
    expected = {
        "checkpoint_fingerprint": record["checkpoint_fingerprint"],
        "decoder_type": record["decoder_type"],
        "dataset_manifest_fingerprint": record["dataset_fingerprint"],
        "validation_dataset_manifest_fingerprint": record["validation_dataset_fingerprint"],
        "teacher_checkpoint_sha256": record["teacher_checkpoint_sha256"],
        "formal_synergy_basis_fingerprint": plan["synergy_basis_fingerprint"],
    }
    mismatched = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatched:
        raise ValueError(f"analysis input provenance binding mismatch: {mismatched}")
    binding = manifest.get("basis_binding") or {}
    if (
        binding.get("verified") is not True
        or binding.get("formal_synergy_basis_fingerprint") != plan["synergy_basis_fingerprint"]
    ):
        raise ValueError("analysis input lacks verified runtime/formal basis binding")
    causal_evidence = manifest.get("causal_evidence") or {}
    causal_status = causal_evidence.get("status")
    if causal_evidence.get("offline_intervention_verified") is not True:
        raise ValueError("analysis input sidecar lacks verified offline perturbations")
    if causal_status == "verified_environment_rollout" and causal_evidence.get("causal_rollout_verified") is not True:
        raise ValueError("analysis input causal status and verification flag disagree")
    if require_causal and not (
        causal_status == "verified_environment_rollout" and causal_evidence.get("causal_rollout_verified") is True
    ):
        raise ValueError("required environment-rollout causal evidence is absent")
    if require_causal and causal_evidence.get("stage2_diagnostic_outcomes_complete") is not True:
        raise ValueError("required Stage-2 causal diagnostic outcomes are incomplete")
    return manifest


def _analyze_run_inputs(
    path: Path,
    *,
    basis,
    checkpoint_fingerprint: str,
    synergy_basis_fingerprint: str,
    max_samples: int,
    require_all_phases: bool,
    require_causal: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    if max_samples <= 4:
        raise ValueError("max_analysis_samples must be greater than four")
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(ANALYSIS_INPUT_FIELDS) - set(data.files))
        if missing:
            raise ValueError(f"analysis_inputs.npz is missing fields {missing}")
        sample_count = int(np.asarray(data["latents"]).shape[0])
        if sample_count <= 4:
            raise ValueError("analysis_inputs.npz has too few samples")
        selected = np.arange(sample_count)
        if sample_count > max_samples:
            selected = np.arange(max_samples, dtype=np.int64)
        latents = np.asarray(data["latents"])[selected]
        coefficients = np.asarray(data["synergy_coefficients"])[selected]
        target_coefficients = np.asarray(data["target_synergy_coefficients"])[selected]
        jacobians = np.asarray(data["decoder_jacobians"])[selected]
        phases = np.asarray(data["phase_ids"])[selected]
        train_mask = np.asarray(data["train_mask"])[selected]
        sample_uids = np.asarray(data["sample_uids"])[selected].astype(str)
        teacher_physical = np.asarray(data["teacher_physical_excitation"])[selected]
        baseline = np.asarray(data["baseline_physical_excitation"])[selected]
        perturbed = np.asarray(data["perturbed_physical_excitation"])[selected]
        epsilons = np.asarray(data["intervention_epsilons"])
        directions = np.asarray(data["intervention_directions"])
        direction_names = np.asarray(data["intervention_direction_names"]).astype(str)
        causal_effects = np.asarray(data["causal_effects"])[selected] if "causal_effects" in data.files else None
    if not np.array_equal(coefficients, target_coefficients):
        raise ValueError("synergy_coefficients alias differs from the formal NNLS target")
    if (
        latents.ndim != 2
        or coefficients.ndim != 2
        or jacobians.shape != (latents.shape[0], np.asarray(basis).shape[0], latents.shape[1])
        or phases.shape != (latents.shape[0],)
        or train_mask.shape != (latents.shape[0],)
        or sample_uids.shape != (latents.shape[0],)
        or teacher_physical.shape != baseline.shape
        or baseline.shape != (latents.shape[0], np.asarray(basis).shape[0])
        or perturbed.shape
        != (
            latents.shape[0],
            directions.shape[0],
            epsilons.shape[0],
            np.asarray(basis).shape[0],
        )
        or directions.shape != (direction_names.shape[0], latents.shape[1])
    ):
        raise ValueError("analysis_inputs.npz arrays violate the v2 shape contract")
    numeric = (
        latents,
        coefficients,
        jacobians,
        phases,
        teacher_physical,
        baseline,
        perturbed,
        epsilons,
        directions,
    )
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("analysis_inputs.npz contains non-finite core evidence")
    if len(set(sample_uids.tolist())) != len(sample_uids):
        raise ValueError("analysis sample_uids must be unique")
    if train_mask.dtype != np.bool_:
        if not np.all(np.isin(train_mask, [0, 1])):
            raise ValueError("analysis train_mask must be boolean")
        train_mask = train_mask.astype(bool)
    if np.sum(train_mask) < 2 or np.sum(~train_mask) < 2:
        raise ValueError("analysis requires at least two train and validation samples")
    if require_causal and causal_effects is None:
        raise ValueError("required environment-rollout causal_effects are absent")
    if causal_effects is not None and (
        causal_effects.shape[:3] != (latents.shape[0], directions.shape[0], epsilons.shape[0])
        or causal_effects.ndim < 4
        or not np.all(np.isfinite(causal_effects))
    ):
        raise ValueError("optional causal_effects violate the aligned rollout contract")
    effective = effective_dimension_report(latents, decoder_jacobians=jacobians)
    alignment = jacobian_alignment_report(
        jacobians,
        basis,
        phase_ids=phases,
        require_all_phases=require_all_phases,
    )
    representation = representation_report(
        latents,
        coefficients,
        train_mask=train_mask,
        phase_ids=phases,
        require_all_phases=require_all_phases,
    )
    intervention = summarize_intervention_effects(
        {"physical_excitation": baseline},
        {"physical_excitation": perturbed},
        epsilons=epsilons.tolist(),
        direction_names=direction_names.tolist(),
        phase_ids=phases,
        require_metrics=("physical_excitation",),
        require_all_phases=require_all_phases,
    )
    payload = {
        "schema_version": "latent_synergy_analysis_metrics_v1",
        "input_path": str(path.resolve()),
        "input_sha256": _file_sha256(path),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "synergy_basis_fingerprint": synergy_basis_fingerprint,
        "num_source_samples": sample_count,
        "num_analyzed_samples": int(latents.shape[0]),
        "effective_dimension": effective,
        "jacobian_alignment": alignment,
        "representation_similarity": representation,
        "intervention": intervention,
        # Decoder-space perturbations are valid offline evidence, but are not
        # an environment-level causal rollout.  Keep the two claims separate.
        "offline_intervention_verified": True,
        "causal_rollout_verified": causal_effects is not None,
        "causal_evidence_status": (
            "verified_environment_rollout" if causal_effects is not None else "not_provided_optional"
        ),
    }
    cross = {
        "representations": latents,
        "jacobian_span": np.mean(jacobians, axis=0),
        "causal_effects": causal_effects,
        "sample_uids": sample_uids,
    }
    return payload, cross


def _cross_seed_reports(
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]],
    *,
    expected_seeds: set[str],
    require_causal: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    reports: list[dict[str, Any]] = []
    for (latent_dim, decoder_type), seed_payloads in sorted(grouped.items()):
        if set(seed_payloads) != expected_seeds:
            raise ValueError(
                f"cross-seed group d={latent_dim}, decoder={decoder_type} has seeds "
                f"{sorted(seed_payloads)}, expected {sorted(expected_seeds)}"
            )
        ordered = [seed_payloads[name]["sample_uids"] for name in sorted(seed_payloads)]
        if any(not np.array_equal(ordered[0], value) for value in ordered[1:]):
            raise ValueError(f"cross-seed sample_uids differ for d={latent_dim}, decoder={decoder_type}")
        causal_values = [payload.get("causal_effects") for payload in seed_payloads.values()]
        if any(value is None for value in causal_values) and not all(value is None for value in causal_values):
            raise ValueError(f"cross-seed causal evidence is mixed for d={latent_dim}, decoder={decoder_type}")
        if require_causal and any(value is None for value in causal_values):
            raise ValueError("required causal evidence is absent from a cross-seed group")
        causal_mapping = (
            None
            if all(value is None for value in causal_values)
            else {name: payload["causal_effects"] for name, payload in seed_payloads.items()}
        )
        report = cross_seed_report(
            {name: payload["representations"] for name, payload in seed_payloads.items()},
            jacobian_spans={name: payload["jacobian_span"] for name, payload in seed_payloads.items()},
            causal_effects=causal_mapping,
        )
        reports.append(
            {
                "latent_dim": latent_dim,
                "decoder_type": decoder_type,
                "seed_set": sorted(seed_payloads),
                "causal_evidence_status": (
                    "verified_environment_rollout" if causal_mapping is not None else "not_provided_optional"
                ),
                "offline_intervention_verified": True,
                "causal_rollout_verified": causal_mapping is not None,
                "report": report,
            }
        )
    return reports


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_promotion_model(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    cross_seed_analysis: list[dict[str, Any]],
    failures: list[dict[str, str]],
    require_causal_interventions: bool = False,
) -> dict[str, Any]:
    import math

    if failures or len(records) != len(plan["jobs"]):
        raise ValueError("promotion requires the exact complete registered sweep matrix")
    expected_seeds = sorted({int(job["seed"]) for job in plan["jobs"]})
    expected_groups = {(int(job["latent_dim"]), str(job["decoder_type"])) for job in plan["jobs"]}
    cross_by_group = {(int(item["latent_dim"]), str(item["decoder_type"])): item for item in cross_seed_analysis}
    if set(cross_by_group) != expected_groups:
        raise ValueError("promotion requires cross-seed analysis for every matrix group")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((int(record["latent_dim"]), str(record["decoder_type"])), []).append(record)
    candidates: dict[str, list[tuple[tuple[float, ...], dict[str, Any]]]] = {
        "direct": [],
        "synergy": [],
    }
    group_summaries: list[dict[str, Any]] = []
    matrix_causal_verified = True
    decoder_order = {"direct": 0.0, "fixed_synergy": 0.0, "synergy_residual": 1.0}
    for (latent_dim, decoder_type), group_records in sorted(grouped.items()):
        if decoder_type not in decoder_order:
            raise ValueError(f"unsupported promotion decoder type: {decoder_type}")
        ordered = sorted(group_records, key=lambda item: int(item["seed"]))
        if [int(item["seed"]) for item in ordered] != expected_seeds:
            raise ValueError(f"promotion group d={latent_dim}, decoder={decoder_type} lacks the registered seed set")
        heldout: list[int] = []
        nrmse: list[float] = []
        success: list[float] = []
        residual: list[float] = []
        residual_max: list[float] = []
        residual_ready_max: list[float] = []
        residual_recovery_max: list[float] = []
        alignment: list[float] = []
        run_causal_verified: list[bool] = []
        for record in ordered:
            metrics = record["metrics"]
            closed = record["closed_loop"]
            analysis = record.get("analysis") or {}
            try:
                heldout_value = int(metrics["num_eval_samples"])
                physical_mse = float(metrics["physical_excitation_mse"])
                success_value = float(closed["prior_mean_no_fall_rate"])
                residual_value = float(metrics.get("residual_energy_ratio", 0.0))
                offline_ready = float(metrics["residual_energy_ratio_ready"])
                offline_recovery = float(metrics["residual_energy_ratio_recovery"])
                closed_by_lambda = closed["by_lambda"]
                closed_total = [float(value["residual_energy_ratio"]) for value in closed_by_lambda.values()]
                closed_ready = [
                    float(value["by_phase"]["ready"]["residual_energy_ratio"]) for value in closed_by_lambda.values()
                ]
                closed_recovery = [
                    float(value["by_phase"]["recovery"]["residual_energy_ratio"]) for value in closed_by_lambda.values()
                ]
                alignment_value = float(analysis["jacobian_alignment"]["projection_score_mean"])
                intervention_samples = int(analysis["intervention"]["num_samples"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"promotion group {latent_dim}/{decoder_type} has incomplete analysis metrics"
                ) from exc
            run_residual_max = max(residual_value, *closed_total)
            run_ready_max = max(offline_ready, *closed_ready)
            run_recovery_max = max(offline_recovery, *closed_recovery)
            values = (
                physical_mse,
                success_value,
                residual_value,
                run_residual_max,
                run_ready_max,
                run_recovery_max,
                alignment_value,
            )
            offline_verified = bool(analysis.get("offline_intervention_verified", intervention_samples > 0))
            causal_verified = bool(
                analysis.get(
                    "causal_rollout_verified",
                    analysis.get("causal_evidence_status") == "verified_environment_rollout",
                )
                and record.get("stage2_diagnostic_outcomes_complete") is True
            )
            runtime_basis_valid = (
                record["runtime_synergy_basis_source_fingerprint"] is None
                and record["runtime_synergy_basis_fingerprint"] is None
                if decoder_type == "direct"
                else record["runtime_synergy_basis_source_fingerprint"] == plan["synergy_basis_fingerprint"]
                and record["runtime_synergy_basis_fingerprint"] is not None
            )
            if (
                heldout_value <= 0
                or intervention_samples <= 0
                or not offline_verified
                or physical_mse < 0.0
                or not all(math.isfinite(value) for value in values)
                or record.get("analysis_complete") is not True
                or record.get("basis_binding_verified") is not True
                or record["synergy_basis_expected_fingerprint"] != plan["synergy_basis_fingerprint"]
                or not runtime_basis_valid
                or record["checkpoint_fingerprint"] != closed.get("checkpoint_fingerprint")
                or closed.get("promotion", {}).get("passed") is not True
            ):
                raise ValueError(
                    f"promotion group d={latent_dim}, decoder={decoder_type} failed a bound evidence contract"
                )
            heldout.append(heldout_value)
            nrmse.append(math.sqrt(physical_mse))
            success.append(success_value)
            residual.append(residual_value)
            residual_max.append(run_residual_max)
            residual_ready_max.append(run_ready_max)
            residual_recovery_max.append(run_recovery_max)
            alignment.append(alignment_value)
            run_causal_verified.append(causal_verified)
        cross = cross_by_group[(latent_dim, decoder_type)]
        cross_report = cross.get("report") or {}
        cross_causal_verified = bool(
            cross.get(
                "causal_rollout_verified",
                cross.get("causal_evidence_status") == "verified_environment_rollout",
            )
        )
        if (
            cross.get("seed_set") != [str(seed) for seed in expected_seeds]
            or int(cross_report.get("num_pairs", 0)) <= 0
            or not math.isfinite(float(cross_report.get("linear_cka_mean", math.nan)))
            or not math.isfinite(float(cross_report.get("jacobian_projection_score_mean", math.nan)))
        ):
            raise ValueError(f"promotion group d={latent_dim}, decoder={decoder_type} lacks valid cross-seed stability")
        group_summary = {
            "latent_dim": latent_dim,
            "decoder_type": decoder_type,
            "seed_set": expected_seeds,
            "num_seeds": len(expected_seeds),
            "heldout_sample_count_min": min(heldout),
            "reconstruction_nrmse_mean": float(sum(nrmse) / len(nrmse)),
            "reconstruction_nrmse_std": float(np.std(nrmse)),
            "closed_loop_success_rate_mean": float(sum(success) / len(success)),
            "closed_loop_success_rate_std": float(np.std(success)),
            "residual_energy_ratio_mean": float(sum(residual) / len(residual)),
            "residual_energy_ratio_max": max(residual_max),
            "residual_energy_ratio_ready_max": max(residual_ready_max),
            "residual_energy_ratio_recovery_max": max(residual_recovery_max),
            "residual_bypass_gate_passed": bool(
                max(residual_max) <= MAX_RESIDUAL_ENERGY_RATIO
                and max(residual_ready_max) <= MAX_QUIESCENT_PHASE_RESIDUAL_ENERGY_RATIO
                and max(residual_recovery_max) <= MAX_QUIESCENT_PHASE_RESIDUAL_ENERGY_RATIO
            ),
            "jacobian_projection_score_mean": float(sum(alignment) / len(alignment)),
            "cross_seed_linear_cka_mean": float(cross_report["linear_cka_mean"]),
            "cross_seed_jacobian_projection_score_mean": float(cross_report["jacobian_projection_score_mean"]),
            "offline_intervention_verified": True,
            "causal_rollout_verified": bool(cross_causal_verified and all(run_causal_verified)),
        }
        matrix_causal_verified = bool(matrix_causal_verified and group_summary["causal_rollout_verified"])
        group_summaries.append(group_summary)
        score = (
            -group_summary["closed_loop_success_rate_mean"],
            group_summary["reconstruction_nrmse_mean"],
            group_summary["residual_energy_ratio_mean"],
            float(latent_dim),
            decoder_order[decoder_type],
        )
        # Deployment seed is pre-registered as the smallest seed in the chosen
        # group; no per-seed metric is used to cherry-pick the checkpoint.
        deployment = ordered[0]
        selected = {
            "run_name": deployment["run_name"],
            "checkpoint_dir": deployment["checkpoint_dir"],
            "checkpoint_fingerprint": deployment["checkpoint_fingerprint"],
            "formal_synergy_basis_fingerprint": plan["synergy_basis_fingerprint"],
            "runtime_synergy_basis_fingerprint": deployment["runtime_synergy_basis_fingerprint"],
            "runtime_synergy_basis_source_fingerprint": deployment["runtime_synergy_basis_source_fingerprint"],
            "dataset_fingerprint": deployment["dataset_fingerprint"],
            "validation_dataset_fingerprint": deployment["validation_dataset_fingerprint"],
            "motion_split_fingerprint": deployment["motion_split_fingerprint"],
            "latent_dim": latent_dim,
            "decoder_type": decoder_type,
            "seed": int(deployment["seed"]),
            "deployment_seed_rule": "smallest_registered_seed_within_selected_group",
            "checkpoint_binding_verified": 1.0,
        }
        family = "direct" if decoder_type == "direct" else "synergy"
        candidates[family].append((score, {"group": group_summary, "model": selected}))
    expects_direct = any(decoder == "direct" for _, decoder in expected_groups)
    expects_synergy = any(decoder != "direct" for _, decoder in expected_groups)
    if expects_direct and not candidates["direct"]:
        raise ValueError("no complete multi-seed direct comparator has promotion evidence")
    if expects_synergy and not candidates["synergy"]:
        raise ValueError("no complete multi-seed synergy group has promotion evidence")
    if require_causal_interventions and not matrix_causal_verified:
        raise ValueError("promotion requires causal environment rollouts for every registered run")

    chosen_direct = None if not candidates["direct"] else min(candidates["direct"], key=lambda item: item[0])[1]
    chosen_synergy = None if not candidates["synergy"] else min(candidates["synergy"], key=lambda item: item[0])[1]
    primary = chosen_synergy or chosen_direct
    if primary is None:
        raise ValueError("promotion found no complete multi-seed candidate")
    selected_group = primary["group"]
    selected = primary["model"]
    selected_models = {
        key: value["model"]
        for key, value in (
            ("best_direct", chosen_direct),
            ("best_synergy", chosen_synergy),
        )
        if value is not None
    }
    selected_groups = {
        key: value["group"]
        for key, value in (
            ("best_direct", chosen_direct),
            ("best_synergy", chosen_synergy),
        )
        if value is not None
    }
    payload = {
        "schema_version": "latent_synergy_promotion_metrics_v2",
        "heldout_sample_count": selected_group["heldout_sample_count_min"],
        "reconstruction_nrmse": selected_group["reconstruction_nrmse_mean"],
        "closed_loop_success_rate": selected_group["closed_loop_success_rate_mean"],
        "residual_energy_ratio": selected_group["residual_energy_ratio_max"],
        "residual_energy_ratio_ready": selected_group["residual_energy_ratio_ready_max"],
        "residual_energy_ratio_recovery": selected_group["residual_energy_ratio_recovery_max"],
        "residual_bypass_gate_passed": 1.0 if selected_group["residual_bypass_gate_passed"] else 0.0,
        "latent_dimension_selected": 1.0,
        "checkpoint_binding_verified": 1.0,
        "analysis_complete": 1.0,
        "cross_seed_stability_verified": 1.0,
        "alignment_evidence_verified": 1.0,
        "offline_intervention_verified": 1.0,
        "causal_rollout_required": bool(require_causal_interventions),
        "causal_rollout_verified": 1.0 if matrix_causal_verified else 0.0,
        "stage2_diagnostic_outcomes_complete": 1.0 if matrix_causal_verified else 0.0,
        # Backward-compatible key: it now means causal environment-rollout
        # intervention evidence only, never offline decoder perturbations.
        "intervention_evidence_verified": (1.0 if matrix_causal_verified else 0.0),
        "full_matrix_complete": 1.0,
        "selected_group": selected_group,
        "selected_model": selected,
        "selected_groups": selected_groups,
        "selected_models": selected_models,
        "group_summaries": group_summaries,
        "selection_rule": {
            "group_ranking": [
                "highest mean closed-loop success",
                "lowest mean physical-excitation NRMSE",
                "lowest mean residual energy ratio",
                "smallest latent dimension",
                "fixed_synergy before synergy_residual on an exact tie",
            ],
            "families_selected_independently": ["best_direct", "best_synergy"],
            "primary_alias": ("best_synergy" if chosen_synergy is not None else "best_direct"),
            "deployment_seed": "smallest registered seed within selected group",
            "seed_metrics_used_for_deployment_selection": False,
        },
        "metric_mapping": {
            "heldout_sample_count": "minimum eval_metrics.num_eval_samples across selected seed group",
            "reconstruction_nrmse": "group mean sqrt(eval_metrics.physical_excitation_mse) for unit excitation",
            "closed_loop_success_rate": "group mean closed_loop_metrics.prior_mean_no_fall_rate",
            "residual_energy_ratio": (
                "worst held-out or closed-loop residual/physical excitation energy ratio "
                "across every selected-group seed and lambda"
            ),
            "residual_energy_ratio_ready": (
                "worst held-out or closed-loop ready-phase residual energy ratio across selected-group seeds/lambdas"
            ),
            "residual_energy_ratio_recovery": (
                "worst held-out or closed-loop recovery-phase residual energy ratio across selected-group seeds/lambdas"
            ),
            "residual_bypass_gate_passed": (
                f"total <= {MAX_RESIDUAL_ENERGY_RATIO:g}; ready/recovery <= "
                f"{MAX_QUIESCENT_PHASE_RESIDUAL_ENERGY_RATIO:g}"
            ),
            "latent_dimension_selected": "pre-registered complete multi-seed group selection",
            "checkpoint_binding_verified": "runtime + embedded checkpoint + config + plan basis and dataset fingerprints verified",
            "analysis_complete": "every registered run has bound Jacobian/representation/intervention evidence",
            "cross_seed_stability_verified": "every (dimension,decoder) group has the exact registered seed set",
            "offline_intervention_verified": "decoder-space perturbations were computed for every registered run",
            "causal_rollout_verified": "all registered runs and cross-seed groups have separately sealed environment-rollout effects",
            "intervention_evidence_verified": "deprecated compatibility alias for causal_rollout_verified",
        },
    }
    payload["promotion_metrics_fingerprint"] = _canonical_json_sha256(payload)
    return payload


def _canonical_json_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
