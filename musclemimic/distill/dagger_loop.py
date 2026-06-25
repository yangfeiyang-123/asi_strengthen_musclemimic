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
    num_steps: int = 50_000
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
    seed: int = 0


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
            "--num_steps",
            str(int(config.num_steps)),
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
            "--append",
        ]
        if config.save_reference_features:
            collect_cmd.append("--save_reference_features")
        if config.include_reference_phase:
            collect_cmd.append("--include_reference_phase")
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


def run_dagger_loop(config: DaggerLoopConfig, *, dry_run: bool = False) -> Path:
    """Run or plan the iterative DAgger loop and write iteration metadata."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_iteration_plan(config)
    manifest_path = write_loop_manifest(config, plan, output_dir / "dagger_loop_manifest.json")
    if dry_run:
        return manifest_path

    executed: list[dict[str, object]] = []
    for item in plan:
        print(f"[dagger_loop] iteration={item.dagger_iteration} collect")
        collect = subprocess.run(item.collect_command, check=True, text=True, capture_output=True)
        print(collect.stdout, end="")

        print(f"[dagger_loop] iteration={item.dagger_iteration} train")
        train = subprocess.run(item.train_command, check=True, text=True, capture_output=True)
        print(train.stdout, end="")
        checkpoint_out = _checkpoint_from_stdout(train.stdout, item.student_ckpt_out)
        executed.append(
            {
                "dagger_iteration": item.dagger_iteration,
                "student_checkpoint_in": item.student_ckpt_in,
                "student_checkpoint_out": checkpoint_out,
                "train_output_dir": item.train_output_dir,
                "collect_stdout": collect.stdout,
                "train_stdout": train.stdout,
            }
        )
    executed_path = output_dir / "dagger_loop_results.json"
    executed_path.write_text(json.dumps(executed, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
