"""Unified direct BC / DAgger distillation experiment runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DistillExperimentConfig:
    teacher_ckpt: str
    student_config: str
    out_dir: str
    motion_path: list[str] | None = None
    initial_student_ckpt: str | None = None
    collect_train: bool = False
    train_bc: bool = False
    run_dagger: int = 0
    compare: bool = False
    num_envs: int = 256
    num_steps: int = 200_000
    shard_size: int = 50_000
    train_steps: int = 200_000
    batch_size: int = 4096
    lr: float = 3e-4
    seed: int = 0
    save_reference_features: bool = True
    include_reference_phase: bool = False
    metrics_envs: int = 20
    metrics_steps: int = 500


@dataclass(frozen=True)
class DistillExperimentPlan:
    dataset_dir: str
    bc_dir: str
    dagger_dir: str
    compare_dir: str
    commands: dict[str, list[str]]


def build_distill_experiment_plan(config: DistillExperimentConfig) -> DistillExperimentPlan:
    out_dir = Path(config.out_dir)
    dataset_dir = out_dir / "dataset"
    bc_dir = out_dir / "bc"
    dagger_dir = out_dir / "dagger"
    compare_dir = out_dir / "compare"
    planned_bc_ckpt = str(bc_dir / "checkpoints" / f"checkpoint_{int(config.train_steps)}")
    planned_dagger_ckpt = str(dagger_dir / f"iter_{int(config.run_dagger) - 1:03d}" / "checkpoints" / f"checkpoint_{int(config.train_steps)}")

    commands: dict[str, list[str]] = {}
    if config.collect_train:
        cmd = [
            sys.executable,
            "-m",
            "fullbody.distill_collect",
            "--teacher_ckpt",
            config.teacher_ckpt,
            "--output_dir",
            str(dataset_dir),
            "--num_envs",
            str(int(config.num_envs)),
            "--num_steps",
            str(int(config.num_steps)),
            "--shard_size",
            str(int(config.shard_size)),
            "--seed",
            str(int(config.seed)),
            "--deterministic_teacher",
            "--teacher_action_target",
            "mean",
            "--split",
            "train",
        ]
        if config.save_reference_features:
            cmd.append("--save_reference_features")
        if config.include_reference_phase:
            cmd.append("--include_reference_phase")
        if config.motion_path:
            cmd.append("--motion_path")
            cmd.extend(config.motion_path)
        commands["collect_train"] = cmd

    if config.train_bc:
        commands["train_bc"] = [
            sys.executable,
            "-m",
            "fullbody.distill_train_bc",
            "--dataset_dir",
            str(dataset_dir),
            "--student_config",
            config.student_config,
            "--output_dir",
            str(bc_dir),
            "--batch_size",
            str(int(config.batch_size)),
            "--num_steps",
            str(int(config.train_steps)),
            "--lr",
            str(float(config.lr)),
            "--seed",
            str(int(config.seed)),
        ]

    if int(config.run_dagger) > 0:
        initial_student = config.initial_student_ckpt or planned_bc_ckpt
        commands["dagger"] = [
            sys.executable,
            "-m",
            "fullbody.distill_run_dagger",
            "--teacher_ckpt",
            config.teacher_ckpt,
            "--initial_student_ckpt",
            initial_student,
            "--student_config",
            config.student_config,
            "--dataset_dir",
            str(dataset_dir),
            "--output_dir",
            str(dagger_dir),
            "--num_iters",
            str(int(config.run_dagger)),
            "--num_envs",
            str(int(config.num_envs)),
            "--num_steps",
            str(int(config.num_steps)),
            "--shard_size",
            str(int(config.shard_size)),
            "--train_steps",
            str(int(config.train_steps)),
            "--batch_size",
            str(int(config.batch_size)),
            "--lr",
            str(float(config.lr)),
            "--seed",
            str(int(config.seed)),
        ]
        if config.save_reference_features:
            commands["dagger"].append("--save_reference_features")
        if config.include_reference_phase:
            commands["dagger"].append("--include_reference_phase")

    if config.compare:
        student_ckpt = planned_dagger_ckpt if int(config.run_dagger) > 0 else planned_bc_ckpt
        cmd = [
            sys.executable,
            "-m",
            "fullbody.distill_compare",
            "--teacher_ckpt",
            config.teacher_ckpt,
            "--student_ckpt",
            student_ckpt,
            "--output_dir",
            str(compare_dir),
            "--metrics_envs",
            str(int(config.metrics_envs)),
            "--metrics_steps",
            str(int(config.metrics_steps)),
            "--eval_seed",
            str(int(config.seed)),
        ]
        if config.motion_path:
            cmd.append("--motion_path")
            cmd.extend(config.motion_path)
        commands["compare"] = cmd

    return DistillExperimentPlan(
        dataset_dir=str(dataset_dir),
        bc_dir=str(bc_dir),
        dagger_dir=str(dagger_dir),
        compare_dir=str(compare_dir),
        commands=commands,
    )


def run_distill_experiment(config: DistillExperimentConfig, *, dry_run: bool = False) -> Path:
    plan = build_distill_experiment_plan(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = _manifest_payload(config, plan)
    manifest["dry_run"] = bool(dry_run)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(out_dir / "final_report.md", manifest)
    if dry_run:
        return manifest_path
    for name, command in plan.commands.items():
        print(f"[distill_experiment] {name}")
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        print(completed.stdout, end="")
    return manifest_path


def _manifest_payload(config: DistillExperimentConfig, plan: DistillExperimentPlan) -> dict[str, Any]:
    payload = {
        "schema_version": "distill_experiment_v1",
        "git_commit": _git_commit(),
        "teacher_ckpt": config.teacher_ckpt,
        "student_config": config.student_config,
        "motion_path": list(config.motion_path or []),
        "output_layout": {
            "dataset": plan.dataset_dir,
            "bc": plan.bc_dir,
            "dagger": plan.dagger_dir,
            "compare": plan.compare_dir,
        },
        "commands": plan.commands,
    }
    payload.update({f"config_{key}": value for key, value in asdict(config).items()})
    return payload


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Distill Experiment Report",
        "",
        f"- git_commit: {manifest['git_commit']}",
        f"- teacher_ckpt: {manifest['teacher_ckpt']}",
        f"- student_config: {manifest['student_config']}",
        f"- motion_path: {manifest['motion_path']}",
        "",
        "## Commands",
        "",
    ]
    for name, command in manifest["commands"].items():
        lines.extend([f"### {name}", "", "```bash", " ".join(command), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--initial_student_ckpt", default=None)
    parser.add_argument("--collect_train", action="store_true", default=False)
    parser.add_argument("--train_bc", action="store_true", default=False)
    parser.add_argument("--run_dagger", type=int, default=0)
    parser.add_argument("--compare", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=200_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_reference_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument("--metrics_envs", type=int, default=20)
    parser.add_argument("--metrics_steps", type=int, default=500)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = run_distill_experiment(
        DistillExperimentConfig(
            teacher_ckpt=args.teacher_ckpt,
            student_config=args.student_config,
            motion_path=args.motion_path,
            out_dir=args.out_dir,
            initial_student_ckpt=args.initial_student_ckpt,
            collect_train=bool(args.collect_train),
            train_bc=bool(args.train_bc),
            run_dagger=int(args.run_dagger),
            compare=bool(args.compare),
            num_envs=int(args.num_envs),
            num_steps=int(args.num_steps),
            shard_size=int(args.shard_size),
            train_steps=int(args.train_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            save_reference_features=bool(args.save_reference_features),
            include_reference_phase=bool(args.include_reference_phase),
            metrics_envs=int(args.metrics_envs),
            metrics_steps=int(args.metrics_steps),
        ),
        dry_run=bool(args.dry_run),
    )
    print(f"distill_experiment_manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
