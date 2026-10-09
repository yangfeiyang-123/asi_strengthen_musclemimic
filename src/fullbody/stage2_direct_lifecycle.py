"""Plan and seal the componentized Stage-2 S2-A direct lifecycle.

This CLI never starts training.  Commands emitted by ``plan`` must be run as
separate pipeline steps with their recorded environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musclemimic.badminton.action_registry import action_choices
from musclemimic.distill.stage2_direct_lifecycle import (
    Stage2DirectFamilyConfig,
    _write_immutable,
    build_stage2_direct_family_plan,
    build_stage2_direct_family_promotion,
    build_stage2_direct_seed_evidence,
    derive_direct_dataset,
    validate_stage2_direct_family_promotion,
)


def _shared_dataset_paths(path: str | Path) -> tuple[str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") or {}
    train = str((datasets.get("train") or {}).get("path", ""))
    validation = str((datasets.get("validation") or {}).get("path", ""))
    if not train or not validation:
        raise ValueError("Stage-2 shared inputs lack train/validation dataset paths")
    return train, validation


def _source_paths(args: argparse.Namespace) -> tuple[str, str]:
    shared_train, shared_val = _shared_dataset_paths(args.shared_inputs)
    return (
        args.source_train_dataset_dir or shared_train,
        args.source_val_dataset_dir or shared_val,
    )


def _add_action_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action", choices=action_choices(), required=True)
    parser.add_argument("--shared-inputs", "--shared_inputs", dest="shared_inputs", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Write the exact non-executing S2-A plan.")
    _add_action_shared(plan)
    plan.add_argument("--source-train-dataset-dir", "--source_train_dataset_dir", dest="source_train_dataset_dir")
    plan.add_argument("--source-val-dataset-dir", "--source_val_dataset_dir", dest="source_val_dataset_dir")
    plan.add_argument("--teacher-checkpoint", "--teacher_checkpoint", dest="teacher_checkpoint", required=True)
    plan.add_argument("--teacher-promotion-manifest", "--teacher_promotion_manifest", dest="teacher_promotion_manifest", required=True)
    plan.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    plan.add_argument("--physical-gpu", "--physical_gpu", dest="physical_gpu", type=int, required=True)
    plan.add_argument("--cache-key-prefix", "--cache_key_prefix", dest="cache_key_prefix", required=True)
    plan.add_argument("--student-bc-config", "--student_bc_config", dest="student_bc_config")
    plan.add_argument("--student-ppo-config", "--student_ppo_config", dest="student_ppo_config")
    plan.add_argument("--train-steps", type=int, default=200_000)
    plan.add_argument("--dagger-num-transitions", type=int, default=500_000)
    plan.add_argument("--ppo-total-timesteps", type=int)
    plan.add_argument("--num-envs", type=int, default=256)
    plan.add_argument("--shard-size", type=int, default=50_000)
    plan.add_argument("--batch-size", type=int, default=4096)
    plan.add_argument("--bc-lr", type=float, default=3e-4)

    derive = commands.add_parser(
        "derive-dataset", help="Byte-copy shared train data into one isolated seed root."
    )
    _add_action_shared(derive)
    derive.add_argument("--seed", type=int, required=True)
    derive.add_argument("--source-train-dataset-dir", "--source_train_dataset_dir", dest="source_train_dataset_dir")
    derive.add_argument("--source-val-dataset-dir", "--source_val_dataset_dir", dest="source_val_dataset_dir")
    derive.add_argument("--output-dataset-dir", "--output_dataset_dir", dest="output_dataset_dir", required=True)

    seal = commands.add_parser("seal-seed", help="Seal one completed seed endpoint.")
    _add_action_shared(seal)
    seal.add_argument("--seed", type=int, required=True)
    seal.add_argument("--direct-dataset-dir", required=True)
    seal.add_argument("--bc-checkpoint", required=True)
    seal.add_argument("--dagger-dir", required=True)
    seal.add_argument("--ppo-checkpoint", required=True)
    seal.add_argument("--compare-dir", required=True)
    seal.add_argument("--output", required=True)

    promote = commands.add_parser(
        "promote-family", help="Apply the exact seed-paired S2-A family gate."
    )
    _add_action_shared(promote)
    promote.add_argument(
        "--seed-evidence",
        action="append",
        required=True,
        help="Repeat exactly three times as SEED:/absolute/path.json.",
    )
    promote.add_argument("--output", required=True)

    validate = commands.add_parser("validate-family", help="Revalidate a sealed family.")
    validate.add_argument("--promotion", required=True)
    validate.add_argument("--action", choices=action_choices())
    validate.add_argument("--shared-inputs")
    return parser


def _parse_seed_evidence(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path = value.partition(":")
        if not separator:
            raise ValueError("--seed-evidence must use SEED:PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate seed evidence: {seed}")
        result[seed] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        train, validation = _source_paths(args)
        payload, _steps = build_stage2_direct_family_plan(
            Stage2DirectFamilyConfig(
                action=args.action,
                shared_inputs=args.shared_inputs,
                source_train_dataset_dir=train,
                source_val_dataset_dir=validation,
                teacher_checkpoint=args.teacher_checkpoint,
                teacher_promotion_manifest=args.teacher_promotion_manifest,
                output_dir=args.output_dir,
                physical_gpu=args.physical_gpu,
                cache_key_prefix=args.cache_key_prefix,
                student_bc_config=args.student_bc_config,
                student_ppo_config=args.student_ppo_config,
                train_steps=args.train_steps,
                dagger_num_transitions=args.dagger_num_transitions,
                ppo_total_timesteps=args.ppo_total_timesteps,
                num_envs=args.num_envs,
                shard_size=args.shard_size,
                batch_size=args.batch_size,
                bc_lr=args.bc_lr,
            )
        )
        target = _write_immutable(
            Path(args.output_dir) / "stage2_direct_family_plan.json", payload
        )
        print(f"stage2_direct_family_plan: {target}")
        return 0
    if args.command == "derive-dataset":
        train, validation = _source_paths(args)
        target = derive_direct_dataset(
            action=args.action,
            seed=args.seed,
            shared_inputs=args.shared_inputs,
            source_train_dataset_dir=train,
            source_val_dataset_dir=validation,
            output_dataset_dir=args.output_dataset_dir,
        )
        print(f"direct_dataset_derivation: {target}")
        return 0
    if args.command == "seal-seed":
        payload = build_stage2_direct_seed_evidence(
            action=args.action,
            seed=args.seed,
            shared_inputs=args.shared_inputs,
            direct_dataset_dir=args.direct_dataset_dir,
            bc_checkpoint=args.bc_checkpoint,
            dagger_dir=args.dagger_dir,
            ppo_checkpoint=args.ppo_checkpoint,
            compare_dir=args.compare_dir,
        )
        target = _write_immutable(args.output, payload)
        print(f"stage2_direct_seed_evidence: {target}")
        return 0
    if args.command == "promote-family":
        payload = build_stage2_direct_family_promotion(
            action=args.action,
            shared_inputs=args.shared_inputs,
            seed_evidence=_parse_seed_evidence(args.seed_evidence),
        )
        target = _write_immutable(args.output, payload)
        print(f"stage2_direct_family_promotion: {target}")
        return 0
    payload = validate_stage2_direct_family_promotion(
        args.promotion,
        expected_action=args.action,
        expected_shared_inputs=args.shared_inputs,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
