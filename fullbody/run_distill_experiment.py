"""Unified direct BC / DAgger distillation experiment runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    validate_stage2_teacher_promotion,
)


@dataclass(frozen=True)
class DistillExperimentConfig:
    teacher_ckpt: str
    student_config: str
    out_dir: str
    motion_path: list[str] | None = None
    train_motion_path: list[str] | None = None
    val_motion_path: list[str] | None = None
    initial_student_ckpt: str | None = None
    convergence_metrics_path: str | None = None
    student_ppo_config: str | None = None
    collect_train: bool = False
    collect_val: bool = False
    train_bc: bool = False
    run_dagger: int = 0
    run_ppo: bool = False
    compare: bool = False
    num_envs: int = 256
    # Production budgets are total samples across vector lanes.  Legacy
    # ``*_num_steps`` remain explicit opt-ins because 200k vector steps with
    # 256 envs would silently create 51.2M very wide samples.
    num_transitions: int = 1_000_000
    val_num_transitions: int = 200_000
    dagger_num_transitions: int = 500_000
    num_steps: int | None = None
    val_num_steps: int | None = None
    dagger_num_steps: int | None = None
    shard_size: int = 50_000
    train_steps: int = 200_000
    ppo_total_timesteps: int | None = None
    batch_size: int = 4096
    lr: float = 3e-4
    seed: int = 0
    save_reference_features: bool = True
    include_reference_phase: bool = False
    metrics_envs: int = 20
    metrics_steps: int = 500
    run_uid: str | None = None
    resume_dataset: bool = False
    teacher_promotion_manifest: str | None = None
    test_only_allow_unpromoted_teacher: bool = False


@dataclass(frozen=True)
class DistillExperimentPlan:
    dataset_dir: str
    bc_dir: str
    dagger_dir: str
    ppo_dir: str
    compare_dir: str
    commands: dict[str, list[str]]


def build_distill_experiment_plan(config: DistillExperimentConfig) -> DistillExperimentPlan:
    _validate_distill_experiment_config(config)
    out_dir = Path(config.out_dir)
    dataset_dir = out_dir / "dataset"
    bc_dir = out_dir / "bc"
    dagger_dir = out_dir / "dagger"
    ppo_dir = out_dir / "ppo"
    compare_dir = out_dir / "compare"
    planned_bc_ckpt = str(bc_dir / "checkpoints" / f"checkpoint_{int(config.train_steps)}")
    planned_dagger_ckpt = str(dagger_dir / f"iter_{int(config.run_dagger) - 1:03d}" / "checkpoints" / f"checkpoint_{int(config.train_steps)}")
    generated_convergence_path = (
        str(dagger_dir / f"iter_{int(config.run_dagger) - 1:03d}" / "distill_metadata.json")
        if int(config.run_dagger) > 0
        else str(bc_dir / "distill_metadata.json")
        if config.train_bc
        else config.convergence_metrics_path
    )
    run_uid = _distill_run_uid(config)

    promotion_cli = (
        ["--teacher_promotion_manifest", str(config.teacher_promotion_manifest)]
        if config.teacher_promotion_manifest is not None
        else ["--test_only_allow_unpromoted_teacher"]
    )

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
            *_budget_cli(config.num_transitions, config.num_steps),
            "--shard_size",
            str(int(config.shard_size)),
            "--seed",
            str(int(config.seed)),
            "--deterministic_teacher",
            "--teacher_action_target",
            "mean",
            "--split",
            "train",
            "--run_uid",
            run_uid,
            *promotion_cli,
        ]
        if config.save_reference_features:
            cmd.append("--save_reference_features")
        if config.include_reference_phase:
            cmd.append("--include_reference_phase")
        train_motions = config.train_motion_path or config.motion_path
        if train_motions:
            cmd.append("--motion_path")
            cmd.extend(train_motions)
        if config.resume_dataset:
            cmd.append("--resume_dataset")
        commands["collect_train"] = cmd

    if config.collect_val:
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
            *_budget_cli(config.val_num_transitions, config.val_num_steps),
            "--shard_size",
            str(int(config.shard_size)),
            "--seed",
            str(int(config.seed) + 10_000),
            "--deterministic_teacher",
            "--teacher_action_target",
            "mean",
            "--split",
            "val",
            "--resume_dataset",
            "--run_uid",
            run_uid,
            *promotion_cli,
        ]
        if config.save_reference_features:
            cmd.append("--save_reference_features")
        if config.include_reference_phase:
            cmd.append("--include_reference_phase")
        val_motions = config.val_motion_path
        if val_motions:
            cmd.append("--motion_path")
            cmd.extend(val_motions)
        commands["collect_val"] = cmd

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
            "--require_dataset_manifest",
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
            *_budget_cli(config.dagger_num_transitions, config.dagger_num_steps),
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
            "--resume_dataset",
            "--run_uid",
            run_uid,
            *promotion_cli,
        ]
        if config.save_reference_features:
            commands["dagger"].append("--save_reference_features")
        if config.include_reference_phase:
            commands["dagger"].append("--include_reference_phase")
        train_motions = config.train_motion_path or config.motion_path
        if train_motions:
            commands["dagger"].append("--motion_path")
            commands["dagger"].extend(train_motions)

    base_student_ckpt = (
        planned_dagger_ckpt
        if int(config.run_dagger) > 0
        else config.initial_student_ckpt or planned_bc_ckpt
    )
    ppo_checkpoint_root = str(ppo_dir / "checkpoints" / "direct_student_ppo")
    if config.run_ppo:
        ppo_cmd = [
            sys.executable,
            "-m",
            "fullbody.experiment",
            f"--config-name={_hydra_config_name(config.student_ppo_config)}",
            f"experiment.resume_from={base_student_ckpt}",
            "experiment.auto_resume=true",
            "experiment.run_id=direct_student_ppo",
            f"experiment.training_root={ppo_dir}",
        ]
        if config.ppo_total_timesteps is not None:
            ppo_cmd.append(
                f"experiment.total_timesteps={int(config.ppo_total_timesteps)}"
            )
        commands["ppo"] = ppo_cmd

    if config.compare:
        student_ckpt = config.initial_student_ckpt or planned_bc_ckpt
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
            "--dataset_dir",
            str(dataset_dir),
            "--convergence_metrics",
            str(generated_convergence_path),
            "--metrics_envs",
            str(int(config.metrics_envs)),
            "--metrics_steps",
            str(int(config.metrics_steps)),
            "--eval_seed",
            str(int(config.seed)),
            "--deterministic",
        ]
        if int(config.run_dagger) > 0:
            cmd.extend(["--student_dagger_ckpt", planned_dagger_ckpt])
        if config.run_ppo:
            cmd.extend(
                [
                    "--student_ppo_ckpt",
                    ppo_checkpoint_root,
                    "--promotion_policy",
                    "student_bc_ppo",
                ]
            )
        elif int(config.run_dagger) > 0:
            cmd.extend(["--promotion_policy", "student_bc_dagger"])
        else:
            cmd.extend(["--promotion_policy", "student_bc"])
        cmd.append("--motion_path")
        cmd.extend(config.val_motion_path)
        commands["compare"] = cmd
        commands["compare"].append("--require_pass")

    return DistillExperimentPlan(
        dataset_dir=str(dataset_dir),
        bc_dir=str(bc_dir),
        dagger_dir=str(dagger_dir),
        ppo_dir=str(ppo_dir),
        compare_dir=str(compare_dir),
        commands=commands,
    )


def _validate_distill_experiment_config(config: DistillExperimentConfig) -> None:
    """Reject plans that cannot produce the requested downstream checkpoint.

    Planning deliberately permits pre-existing datasets, but every policy stage
    must either be produced earlier in this plan or be supplied explicitly.
    Promotion motions must also be disjoint from known training motions.
    """

    if int(config.run_dagger) < 0:
        raise ValueError("run_dagger must be non-negative")
    if config.teacher_promotion_manifest is None and not config.test_only_allow_unpromoted_teacher:
        raise ValueError(
            "production distillation requires teacher_promotion_manifest; "
            "only tests may use test_only_allow_unpromoted_teacher"
        )
    if config.teacher_promotion_manifest is not None and config.test_only_allow_unpromoted_teacher:
        raise ValueError(
            "teacher_promotion_manifest and test_only_allow_unpromoted_teacher are mutually exclusive"
        )
    for label, transitions, legacy_steps in (
        ("train", config.num_transitions, config.num_steps),
        ("validation", config.val_num_transitions, config.val_num_steps),
        ("DAgger", config.dagger_num_transitions, config.dagger_num_steps),
    ):
        selected = legacy_steps if legacy_steps is not None else transitions
        if int(selected) <= 0:
            raise ValueError(f"{label} collection budget must be positive")

    needs_student_source = bool(config.run_dagger or config.run_ppo or config.compare)
    if needs_student_source and not (config.train_bc or config.initial_student_ckpt):
        raise ValueError(
            "DAgger/PPO/compare requires either train_bc=True or an explicit "
            "initial_student_ckpt"
        )

    if config.run_ppo and not config.student_ppo_config:
        raise ValueError("run_ppo=True requires student_ppo_config")
    if config.ppo_total_timesteps is not None and int(config.ppo_total_timesteps) <= 0:
        raise ValueError("ppo_total_timesteps must be positive when provided")

    if config.collect_val and not config.val_motion_path:
        raise ValueError("collect_val=True requires explicit val_motion_path")
    if config.compare and not config.val_motion_path:
        raise ValueError(
            "compare=True requires explicit val_motion_path; promotion may not "
            "fall back to the training motions"
        )
    if config.compare and not (
        config.train_bc or int(config.run_dagger) > 0 or config.convergence_metrics_path
    ):
        raise ValueError(
            "compare=True requires generated BC/DAgger convergence history or an explicit "
            "convergence_metrics_path"
        )

    train_motions = {str(path) for path in (config.train_motion_path or config.motion_path or [])}
    val_motions = {str(path) for path in (config.val_motion_path or [])}
    overlap = sorted(train_motions & val_motions)
    if overlap:
        raise ValueError(
            "train_motion_path and val_motion_path must be disjoint; overlapping "
            f"motions: {overlap}"
        )


def run_distill_experiment(config: DistillExperimentConfig, *, dry_run: bool = False) -> Path:
    plan = build_distill_experiment_plan(config)
    if not dry_run and config.teacher_promotion_manifest is not None:
        teacher = checkpoint_content_fingerprint(config.teacher_ckpt)
        validate_stage2_teacher_promotion(
            config.teacher_promotion_manifest,
            teacher_checkpoint=teacher,
        )
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
        # Inherit the terminal so multi-hour BC/DAgger/PPO jobs stream progress
        # and never accumulate hundreds of MB of stdout in RAM.
        subprocess.run(command, check=True, text=True)
    return manifest_path


def _manifest_payload(config: DistillExperimentConfig, plan: DistillExperimentPlan) -> dict[str, Any]:
    payload = {
        "schema_version": "distill_experiment_v1",
        "run_uid": _distill_run_uid(config),
        "git_commit": _git_commit(),
        "teacher_ckpt": config.teacher_ckpt,
        "student_config": config.student_config,
        "motion_path": list(config.motion_path or []),
        "train_motion_path": list(config.train_motion_path or config.motion_path or []),
        "val_motion_path": list(config.val_motion_path or []),
        "output_layout": {
            "dataset": plan.dataset_dir,
            "bc": plan.bc_dir,
            "dagger": plan.dagger_dir,
            "ppo": plan.ppo_dir,
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


def _hydra_config_name(config_path: str) -> str:
    """Convert a fullbody YAML path into Hydra's config-name syntax."""
    value = Path(config_path)
    fullbody_dir = Path(__file__).resolve().parent
    if value.is_absolute():
        try:
            value = value.resolve().relative_to(fullbody_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "student_ppo_config must live below the repository fullbody config root"
            ) from exc
    else:
        parts = value.parts
        if parts and parts[0] == "fullbody":
            value = Path(*parts[1:])
    return value.with_suffix("").as_posix()


def _distill_run_uid(config: DistillExperimentConfig) -> str:
    if config.run_uid is not None:
        value = str(config.run_uid).strip()
        if not value:
            raise ValueError("run_uid cannot be empty")
        return value
    return canonical_json_sha256(
        {
            "kind": "forehandclear_distill_run_v2",
            "out_dir": str(Path(config.out_dir).expanduser().resolve()),
            "teacher_ckpt": str(config.teacher_ckpt),
            "teacher_promotion_manifest": config.teacher_promotion_manifest,
            "student_config": str(config.student_config),
            "train_motion_path": list(config.train_motion_path or config.motion_path or []),
            "val_motion_path": list(config.val_motion_path or []),
            "seed": int(config.seed),
        }
    )[:24]


def _budget_cli(total_transitions: int, legacy_vector_steps: int | None) -> list[str]:
    if legacy_vector_steps is not None:
        return ["--num_steps", str(int(legacy_vector_steps))]
    return ["--num_transitions", str(int(total_transitions))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_ckpt", required=True)
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
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--train_motion_path", nargs="+", default=None)
    parser.add_argument("--val_motion_path", nargs="+", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--initial_student_ckpt", default=None)
    parser.add_argument("--convergence_metrics_path", default=None)
    parser.add_argument("--student_ppo_config", default=None)
    parser.add_argument("--collect_train", action="store_true", default=False)
    parser.add_argument("--collect_val", action="store_true", default=False)
    parser.add_argument("--train_bc", action="store_true", default=False)
    parser.add_argument("--run_dagger", type=int, default=0)
    parser.add_argument("--run_ppo", action="store_true", default=False)
    parser.add_argument("--compare", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_transitions", type=int, default=1_000_000)
    parser.add_argument("--val_num_transitions", type=int, default=200_000)
    parser.add_argument("--dagger_num_transitions", type=int, default=500_000)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Legacy train vector-step budget (samples = steps*num_envs).",
    )
    parser.add_argument("--val_num_steps", type=int, default=None, help="Legacy validation vector steps.")
    parser.add_argument("--dagger_num_steps", type=int, default=None, help="Legacy DAgger vector steps.")
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--ppo_total_timesteps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_reference_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument("--metrics_envs", type=int, default=20)
    parser.add_argument("--metrics_steps", type=int, default=500)
    parser.add_argument("--run_uid", default=None)
    parser.add_argument(
        "--resume_dataset",
        action="store_true",
        default=False,
        help="Explicitly resume the train collection after exact manifest validation.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = run_distill_experiment(
        DistillExperimentConfig(
            teacher_ckpt=args.teacher_ckpt,
            student_config=args.student_config,
            motion_path=args.motion_path,
            train_motion_path=args.train_motion_path,
            val_motion_path=args.val_motion_path,
            out_dir=args.out_dir,
            initial_student_ckpt=args.initial_student_ckpt,
            convergence_metrics_path=args.convergence_metrics_path,
            student_ppo_config=args.student_ppo_config,
            collect_train=bool(args.collect_train),
            collect_val=bool(args.collect_val),
            train_bc=bool(args.train_bc),
            run_dagger=int(args.run_dagger),
            run_ppo=bool(args.run_ppo),
            compare=bool(args.compare),
            num_envs=int(args.num_envs),
            num_transitions=int(args.num_transitions),
            val_num_transitions=int(args.val_num_transitions),
            dagger_num_transitions=int(args.dagger_num_transitions),
            num_steps=None if args.num_steps is None else int(args.num_steps),
            val_num_steps=None if args.val_num_steps is None else int(args.val_num_steps),
            dagger_num_steps=None if args.dagger_num_steps is None else int(args.dagger_num_steps),
            shard_size=int(args.shard_size),
            train_steps=int(args.train_steps),
            ppo_total_timesteps=args.ppo_total_timesteps,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            save_reference_features=bool(args.save_reference_features),
            include_reference_phase=bool(args.include_reference_phase),
            metrics_envs=int(args.metrics_envs),
            metrics_steps=int(args.metrics_steps),
            run_uid=args.run_uid,
            resume_dataset=bool(args.resume_dataset),
            teacher_promotion_manifest=args.teacher_promotion_manifest,
            test_only_allow_unpromoted_teacher=bool(
                args.test_only_allow_unpromoted_teacher
            ),
        ),
        dry_run=bool(args.dry_run),
    )
    print(f"distill_experiment_manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
