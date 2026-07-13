"""Iterative DAgger orchestration for student distillation."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class DaggerIterationPlan:
    dagger_iteration: int
    student_ckpt_in: str
    student_ckpt_out: str
    train_output_dir: str
    collect_command: list[str]
    train_command: list[str]


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
            collect_cmd.extend(
                ["--teacher_promotion_manifest", str(config.teacher_promotion_manifest)]
            )
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
            sys.executable,
            "-m",
            "fullbody.distill_train_bc",
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
        plans.append(
            DaggerIterationPlan(
                dagger_iteration=iteration,
                student_ckpt_in=current_student,
                student_ckpt_out=student_out,
                train_output_dir=train_output_dir,
                collect_command=collect_cmd,
                train_command=train_cmd,
            )
        )
        current_student = student_out
    return plans


def write_loop_manifest(config: DaggerLoopConfig, plan: list[DaggerIterationPlan], path: str | Path) -> Path:
    """Write a reproducibility manifest for a DAgger loop."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dagger_loop_v1",
        "config": asdict(config),
        "iterations": [
            {
                "dagger_iteration": item.dagger_iteration,
                "student_checkpoint_in": item.student_ckpt_in,
                "student_checkpoint_out": item.student_ckpt_out,
                "train_output_dir": item.train_output_dir,
                "collect_command": item.collect_command,
                "train_command": item.train_command,
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
        collect_cmd.extend(
            ["--teacher_promotion_manifest", str(config.teacher_promotion_manifest)]
        )
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
        sys.executable,
        "-m",
        "fullbody.distill_train_bc",
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
    )


def write_iteration_result(
    item: DaggerIterationPlan,
    *,
    checkpoint_out_actual: str,
    train_state_step: int | None,
    collect_stdout: str,
    train_stdout: str,
) -> Path:
    """Write per-iteration DAgger result metadata with planned and actual ckpts."""
    result_path = Path(item.train_output_dir) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def run_dagger_loop(config: DaggerLoopConfig, *, dry_run: bool = False) -> Path:
    """Run or plan the iterative DAgger loop and write iteration metadata."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_iteration_plan(config)
    manifest_path = write_loop_manifest(config, plan, output_dir / "dagger_loop_manifest.json")
    if dry_run:
        return manifest_path

    executed: list[dict[str, object]] = []
    current_student = config.initial_student_ckpt
    for iteration in range(int(config.num_iters)):
        item = _iteration_plan_item(config, iteration, current_student)
        print(f"[dagger_loop] iteration={item.dagger_iteration} collect")
        subprocess.run(item.collect_command, check=True, text=True)

        print(f"[dagger_loop] iteration={item.dagger_iteration} train")
        subprocess.run(item.train_command, check=True, text=True)
        checkpoint_out = item.student_ckpt_out
        if not Path(checkpoint_out).exists():
            raise FileNotFoundError(
                f"DAgger training completed but planned checkpoint is missing: {checkpoint_out}"
            )
        train_state_step = int(config.train_steps)
        result_path = write_iteration_result(
            item,
            checkpoint_out_actual=checkpoint_out,
            train_state_step=train_state_step,
            collect_stdout="streamed_to_parent_terminal",
            train_stdout="streamed_to_parent_terminal",
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
            }
        )
        current_student = checkpoint_out
    executed_path = output_dir / "dagger_loop_results.json"
    executed_path.write_text(json.dumps(executed, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
