"""Iterative DAgger orchestration for student distillation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from musclemimic.distill.provenance import canonical_json_sha256


@dataclass(frozen=True)
class DaggerLoopConfig:
    teacher_ckpt: str
    initial_student_ckpt: str
    student_config: str
    dataset_dir: str
    output_dir: str
    num_iters: int = 3
    num_envs: int = 256
    num_transitions: int = 500_000
    num_steps: int | None = None
    shard_size: int = 50_000
    train_steps: int = 200_000
    batch_size: int = 4096
    lr: float = 3e-4
    value_distill_weight: float = 0.1
    gaussian_kl_weight: float = 0.0
    mix_teacher_action_prob: float = 0.0
    save_reference_features: bool = False
    include_reference_phase: bool = False
    freeze_run_stats: bool = True
    split: str = "train"
    motion_path: list[str] | None = None
    seed: int = 0
    motion_field: str = "motion_uid"
    strict_motion_identity: bool = True
    resume_dataset: bool = True
    run_uid: str | None = None
    teacher_promotion_manifest: str | None = None
    test_only_allow_unpromoted_teacher: bool = False
    # Every production BC retrain is a separate canonical launcher job.  The
    # outer DAgger driver inherits the selected physical GPU, while each
    # iteration receives a stable, non-overlapping compilation cache and log.
    physical_gpu: str | None = None
    jax_cache_key_prefix: str | None = None
    train_log_dir: str | None = None
    canonical_launcher: str = "scripts/run_fullbody_training.sh"
    source_train_dataset_dir: str | None = None
    source_val_dataset_dir: str | None = None
    dataset_derivation_manifest: str | None = None


@dataclass(frozen=True)
class DaggerIterationPlan:
    dagger_iteration: int
    student_ckpt_in: str
    student_ckpt_out: str
    train_output_dir: str
    collect_command: list[str]
    train_command: list[str]
    train_environment: dict[str, str]


def _planned_checkpoint(train_output_dir: str, train_steps: int) -> str:
    return str(Path(train_output_dir) / "checkpoints" / f"checkpoint_{int(train_steps)}")


def _collection_budget_args(config: DaggerLoopConfig) -> list[str]:
    if config.num_steps is not None:
        return ["--num_steps", str(int(config.num_steps))]
    return ["--num_transitions", str(int(config.num_transitions))]


def build_iteration_plan(config: DaggerLoopConfig) -> list[DaggerIterationPlan]:
    """Build deterministic subprocess commands for each DAgger iteration."""
    plans: list[DaggerIterationPlan] = []
    current_student = config.initial_student_ckpt
    for iteration in range(int(config.num_iters)):
        item = _iteration_plan_item(config, iteration, current_student)
        plans.append(item)
        current_student = item.student_ckpt_out
    return plans


def write_loop_manifest(config: DaggerLoopConfig, plan: list[DaggerIterationPlan], path: str | Path) -> Path:
    """Write a reproducibility manifest for a DAgger loop."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dagger_loop_v2",
        "config": asdict(config),
        "iterations": [
            {
                "dagger_iteration": item.dagger_iteration,
                "student_checkpoint_in": item.student_ckpt_in,
                "student_checkpoint_out": item.student_ckpt_out,
                "train_output_dir": item.train_output_dir,
                "collect_command": item.collect_command,
                "train_command": item.train_command,
                "train_environment": item.train_environment,
            }
            for item in plan
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _checkpoint_from_stdout(stdout: str, fallback: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("checkpoint_path:"):
            return line.split(":", 1)[1].strip()
    return fallback


def _train_state_step_from_stdout(stdout: str, fallback: int | None = None) -> int | None:
    for line in stdout.splitlines():
        if line.startswith("train_state_step:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return fallback
    return fallback


def _iteration_plan_item(config: DaggerLoopConfig, iteration: int, current_student: str) -> DaggerIterationPlan:
    train_output_dir = str(Path(config.output_dir) / f"iter_{iteration:03d}")
    student_out = _planned_checkpoint(train_output_dir, config.train_steps)
    collect_cmd = [
        sys.executable,
        "-m",
        "fullbody.distill_collect_dagger",
        "--teacher_ckpt",
        config.teacher_ckpt,
        "--student_ckpt",
        current_student,
        "--output_dir",
        config.dataset_dir,
        "--num_envs",
        str(int(config.num_envs)),
        *_collection_budget_args(config),
        "--shard_size",
        str(int(config.shard_size)),
        "--seed",
        str(int(config.seed) + iteration),
        "--mix_teacher_action_prob",
        str(float(config.mix_teacher_action_prob)),
        "--dagger_iteration",
        str(iteration),
        "--rollout_policy",
        "student_with_optional_teacher_mix",
        "--split",
        config.split,
    ]
    if config.resume_dataset:
        collect_cmd.append("--resume_dataset")
    if config.run_uid is not None:
        collect_cmd.extend(["--run_uid", str(config.run_uid)])
    if config.teacher_promotion_manifest is not None:
        collect_cmd.extend(["--teacher_promotion_manifest", str(config.teacher_promotion_manifest)])
    elif config.test_only_allow_unpromoted_teacher:
        collect_cmd.append("--test_only_allow_unpromoted_teacher")
    else:
        raise ValueError("DAgger loop requires a Stage-2 teacher promotion manifest")
    if config.save_reference_features:
        collect_cmd.append("--save_reference_features")
    if config.include_reference_phase:
        collect_cmd.append("--include_reference_phase")
    if config.motion_path:
        collect_cmd.append("--motion_path")
        collect_cmd.extend(config.motion_path)
    collect_cmd.append("--freeze_run_stats" if config.freeze_run_stats else "--no-freeze_run_stats")
    train_cmd = [
        config.canonical_launcher,
        "--distill-bc",
        "--dataset_dir",
        config.dataset_dir,
        "--student_config",
        config.student_config,
        "--output_dir",
        train_output_dir,
        "--batch_size",
        str(int(config.batch_size)),
        "--num_steps",
        str(int(config.train_steps)),
        "--lr",
        str(float(config.lr)),
        "--seed",
        str(int(config.seed) + iteration),
        "--value_distill_weight",
        str(float(config.value_distill_weight)),
        "--gaussian_kl_weight",
        str(float(config.gaussian_kl_weight)),
        "--init_ckpt",
        current_student,
        "--motion_field",
        config.motion_field,
        "--strict_motion_identity" if config.strict_motion_identity else "--no-strict_motion_identity",
        "--require_dataset_manifest",
    ]
    return DaggerIterationPlan(
        dagger_iteration=iteration,
        student_ckpt_in=current_student,
        student_ckpt_out=student_out,
        train_output_dir=train_output_dir,
        collect_command=collect_cmd,
        train_command=train_cmd,
        train_environment=_iteration_train_environment(config, iteration),
    )


def _iteration_train_environment(config: DaggerLoopConfig, iteration: int) -> dict[str, str]:
    """Return the explicit canonical-launch contract for one BC retrain."""

    gpu = config.physical_gpu or os.environ.get("MM_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    prefix = config.jax_cache_key_prefix or os.environ.get("MUSCLEMIMIC_JAX_CACHE_KEY")
    log_root = Path(config.train_log_dir or (Path(config.output_dir) / "logs"))
    result: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": "<required:physical_gpu>" if gpu is None else str(gpu),
        "MUSCLEMIMIC_JAX_CACHE_KEY": (
            "<required:jax_cache_key_prefix>" if prefix is None else f"{prefix}_bc_iter_{int(iteration):03d}"
        ),
        "MUSCLEMIMIC_TRAIN_LOG": str((log_root / f"dagger_bc_iter_{int(iteration):03d}.log").resolve()),
        "MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB": "4",
        "MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB": "4",
    }
    if prefix is not None:
        cache_root = Path(
            os.environ.get(
                "MUSCLEMIMIC_JAX_CACHE_ROOT",
                "/data3/yangfeiyang/WorkSpace/ENV/jax-cache",
            )
        )
        result["JAX_COMPILATION_CACHE_DIR"] = str(cache_root / f"{prefix}_bc_iter_{int(iteration):03d}")
    return result


def write_iteration_result(
    item: DaggerIterationPlan,
    *,
    checkpoint_out_actual: str,
    train_state_step: int | None,
    collect_stdout: str,
    train_stdout: str,
    dataset_manifest_before: dict[str, object] | None = None,
    dataset_manifest_after: dict[str, object] | None = None,
    checkpoint_in: dict[str, object] | None = None,
    checkpoint_out: dict[str, object] | None = None,
) -> Path:
    """Write per-iteration DAgger result metadata with planned and actual ckpts."""
    result_path = Path(item.train_output_dir) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dagger_iteration_result_v2",
        "dagger_iteration": int(item.dagger_iteration),
        "checkpoint_in": item.student_ckpt_in,
        "checkpoint_out_actual": checkpoint_out_actual,
        "checkpoint_out_planned": item.student_ckpt_out,
        "train_state_step": train_state_step,
        "num_train_steps_this_iter": int(
            Path(item.student_ckpt_out).name.removeprefix("checkpoint_")
            if Path(item.student_ckpt_out).name.startswith("checkpoint_")
            else 0
        ),
        "collect_stdout": collect_stdout,
        "train_stdout": train_stdout,
        "dataset_manifest_before": dataset_manifest_before,
        "dataset_manifest_after": dataset_manifest_after,
        "checkpoint_in_content": checkpoint_in,
        "checkpoint_out_content": checkpoint_out,
        "train_command": item.train_command,
        "train_environment": item.train_environment,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def _resolved_launcher(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    resolved = candidate.resolve(strict=True)
    if resolved.name != "run_fullbody_training.sh":
        raise ValueError("production DAgger BC retraining must use scripts/run_fullbody_training.sh")
    return resolved


def _validate_execution_contract(config: DaggerLoopConfig) -> None:
    if int(config.num_iters) <= 0:
        raise ValueError("DAgger num_iters must be positive")
    if config.teacher_promotion_manifest is not None and int(config.num_iters) != 3:
        raise ValueError("production S2-A requires exactly three DAgger iterations")
    if config.teacher_promotion_manifest is None and not config.test_only_allow_unpromoted_teacher:
        raise ValueError("DAgger loop requires a Stage-2 teacher promotion manifest")
    if not config.resume_dataset:
        raise ValueError("DAgger collection must append to an immutable derived direct dataset")
    _resolved_launcher(config.canonical_launcher)

    environment = _iteration_train_environment(config, 0)
    gpu = environment["CUDA_VISIBLE_DEVICES"]
    cache_key = environment["MUSCLEMIMIC_JAX_CACHE_KEY"]
    if re.fullmatch(r"[0-9]+", gpu) is None:
        raise ValueError("DAgger execution requires one explicit physical GPU index")
    if not cache_key or cache_key.startswith("<required:"):
        raise ValueError("DAgger execution requires a stable JAX cache-key prefix")

    dataset = Path(config.dataset_dir).expanduser().resolve()
    output = Path(config.output_dir).expanduser().resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("DAgger dataset and checkpoint roots must be disjoint")
    if config.teacher_promotion_manifest is not None:
        if (
            config.source_train_dataset_dir is None
            or config.source_val_dataset_dir is None
            or config.dataset_derivation_manifest is None
        ):
            raise ValueError(
                "production S2-A DAgger requires an immutable source dataset and direct-dataset derivation manifest"
            )
        sources = {
            "train": Path(config.source_train_dataset_dir).expanduser().resolve(strict=True),
            "validation": Path(config.source_val_dataset_dir).expanduser().resolve(strict=True),
        }
        for source in set(sources.values()):
            if source == dataset or source in dataset.parents or dataset in source.parents:
                raise ValueError(
                    "DAgger must append only to a seed-specific direct dataset outside "
                    "both immutable shared train/validation roots"
                )
        from musclemimic.distill.stage2_direct_lifecycle import (
            validate_direct_dataset_derivation,
        )

        derivation = validate_direct_dataset_derivation(config.dataset_dir)
        requested = Path(config.dataset_derivation_manifest).expanduser().resolve(strict=True)
        if Path(str(derivation["artifact_path"])).resolve(strict=True) != requested:
            raise ValueError("DAgger selected a different direct-dataset derivation")
        records = derivation.get("source_datasets") or {}
        for name, source in sources.items():
            if Path(str((records.get(name) or {}).get("path", ""))).resolve(strict=True) != source:
                raise ValueError(f"DAgger derivation belongs to a different shared {name} dataset")


def run_dagger_loop(config: DaggerLoopConfig, *, dry_run: bool = False) -> Path:
    """Run or plan the iterative DAgger loop and write iteration metadata."""
    _validate_execution_contract(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_iteration_plan(config)
    manifest_path = write_loop_manifest(config, plan, output_dir / "dagger_loop_manifest.json")
    if dry_run:
        return manifest_path

    from musclemimic.distill.provenance import (
        checkpoint_content_fingerprint,
        validate_dataset_manifest,
    )

    executed: list[dict[str, object]] = []
    current_student = config.initial_student_ckpt
    for iteration in range(int(config.num_iters)):
        item = _iteration_plan_item(config, iteration, current_student)
        dataset_before = validate_dataset_manifest(
            config.dataset_dir,
            require_promoted_teacher=config.teacher_promotion_manifest is not None,
        )
        checkpoint_in = checkpoint_content_fingerprint(current_student)
        print(f"[dagger_loop] iteration={item.dagger_iteration} collect")
        subprocess.run(item.collect_command, check=True, text=True)
        dataset_after = validate_dataset_manifest(
            config.dataset_dir,
            require_promoted_teacher=config.teacher_promotion_manifest is not None,
        )
        if dataset_before["manifest_fingerprint"] == dataset_after["manifest_fingerprint"]:
            # An exact idempotent rerun is valid only when this iteration had
            # already been committed.  The immutable collection contracts in
            # validate_dataset_manifest make that distinction auditable.
            matching = [
                entry
                for entry in dataset_after.get("collections", [])
                if int((entry.get("contract") or {}).get("dagger_iteration", -1)) == int(iteration)
            ]
            if len(matching) != 1:
                raise ValueError(
                    "DAgger collection produced no new manifest and has no exact idempotent iteration record"
                )
            existing_result_path = Path(item.train_output_dir) / "result.json"
            if existing_result_path.is_file() and Path(item.student_ckpt_out).is_dir():
                existing = json.loads(existing_result_path.read_text(encoding="utf-8"))
                existing_out = checkpoint_content_fingerprint(item.student_ckpt_out)
                if (
                    existing.get("schema_version") != "dagger_iteration_result_v2"
                    or int(existing.get("dagger_iteration", -1)) != iteration
                    or existing.get("checkpoint_in_content") != checkpoint_in
                    or existing.get("checkpoint_out_content") != existing_out
                    or existing.get("train_command") != item.train_command
                    or existing.get("train_environment") != item.train_environment
                ):
                    raise ValueError(f"existing DAgger iteration {iteration} result differs from the current contract")
                print(f"[dagger_loop] iteration={item.dagger_iteration} idempotent checkpoint/result already complete")
                executed.append(
                    {
                        "dagger_iteration": item.dagger_iteration,
                        "student_checkpoint_in": item.student_ckpt_in,
                        "student_checkpoint_out": item.student_ckpt_out,
                        "student_checkpoint_out_planned": item.student_ckpt_out,
                        "train_state_step": existing.get("train_state_step"),
                        "train_output_dir": item.train_output_dir,
                        "result_json": str(existing_result_path),
                        "collect_stdout": existing.get("collect_stdout"),
                        "train_stdout": existing.get("train_stdout"),
                        "dataset_manifest_before": existing.get("dataset_manifest_before"),
                        "dataset_manifest_after": existing.get("dataset_manifest_after"),
                        "checkpoint_in_content": checkpoint_in,
                        "checkpoint_out_content": existing_out,
                    }
                )
                current_student = item.student_ckpt_out
                continue

        print(f"[dagger_loop] iteration={item.dagger_iteration} train")
        launch_env = os.environ.copy()
        launch_env.update(item.train_environment)
        subprocess.run(item.train_command, check=True, text=True, env=launch_env)
        checkpoint_out = item.student_ckpt_out
        if not Path(checkpoint_out).exists():
            raise FileNotFoundError(f"DAgger training completed but planned checkpoint is missing: {checkpoint_out}")
        train_state_step = int(config.train_steps)
        checkpoint_out_content = checkpoint_content_fingerprint(checkpoint_out)
        result_path = write_iteration_result(
            item,
            checkpoint_out_actual=checkpoint_out,
            train_state_step=train_state_step,
            collect_stdout="streamed_to_parent_terminal",
            train_stdout="streamed_to_parent_terminal",
            dataset_manifest_before={
                "manifest_fingerprint": dataset_before["manifest_fingerprint"],
                "num_collections": len(dataset_before.get("collections", [])),
                "num_samples": int((dataset_before.get("totals") or {}).get("num_samples", 0)),
            },
            dataset_manifest_after={
                "manifest_fingerprint": dataset_after["manifest_fingerprint"],
                "num_collections": len(dataset_after.get("collections", [])),
                "num_samples": int((dataset_after.get("totals") or {}).get("num_samples", 0)),
            },
            checkpoint_in=checkpoint_in,
            checkpoint_out=checkpoint_out_content,
        )
        executed.append(
            {
                "dagger_iteration": item.dagger_iteration,
                "student_checkpoint_in": item.student_ckpt_in,
                "student_checkpoint_out": checkpoint_out,
                "student_checkpoint_out_planned": item.student_ckpt_out,
                "train_state_step": train_state_step,
                "train_output_dir": item.train_output_dir,
                "result_json": str(result_path),
                "collect_stdout": "streamed_to_parent_terminal",
                "train_stdout": "streamed_to_parent_terminal",
                "dataset_manifest_before": {
                    "manifest_fingerprint": dataset_before["manifest_fingerprint"],
                    "num_collections": len(dataset_before.get("collections", [])),
                    "num_samples": int((dataset_before.get("totals") or {}).get("num_samples", 0)),
                },
                "dataset_manifest_after": {
                    "manifest_fingerprint": dataset_after["manifest_fingerprint"],
                    "num_collections": len(dataset_after.get("collections", [])),
                    "num_samples": int((dataset_after.get("totals") or {}).get("num_samples", 0)),
                },
                "checkpoint_in_content": checkpoint_in,
                "checkpoint_out_content": checkpoint_out_content,
            }
        )
        current_student = checkpoint_out
    executed_path = output_dir / "dagger_loop_results.json"
    results_payload = {
        "schema_version": "dagger_loop_results_v2",
        "loop_manifest": str(manifest_path.resolve()),
        "loop_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "iterations": executed,
    }
    results_payload["binding_sha256"] = canonical_json_sha256(results_payload)
    executed_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
