"""Collect DAgger-style student rollout states relabeled by a teacher policy."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.collection_budget import resolve_collection_budget
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.dagger import collect_dagger_dataset
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    resolve_config_motion_paths,
    validate_environment_motion_identity,
)
from musclemimic.distill.provenance import (
    begin_collection,
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    validate_stage2_teacher_promotion,
)
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect DAgger relabel shards for student distillation.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num_transitions",
        type=int,
        default=None,
        help="Exact total samples across vector environments (default: 500,000).",
    )
    budget.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Legacy vector-step count; produces num_steps*num_envs samples.",
    )
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix_teacher_action_prob", type=float, default=0.0)
    parser.add_argument("--save_full_obs", action="store_true", default=False)
    parser.add_argument("--save_reference_features", action="store_true", default=False)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument("--freeze_run_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--motion_group", default=None)
    parser.add_argument("--traj_index", type=int, default=None)
    parser.add_argument("--traj_start_step", type=int, default=None)
    parser.add_argument(
        "--dagger_iteration",
        "--dagger-iteration",
        dest="dagger_iteration",
        type=int,
        default=None,
    )
    parser.add_argument("--rollout_policy", default=None)
    parser.add_argument(
        "--resume_dataset",
        "--resume-dataset",
        action="store_true",
        default=False,
        help="Append only after exact immutable-manifest validation.",
    )
    parser.add_argument("--run_uid", "--run-uid", dest="run_uid", default=None)
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
    args = parser.parse_args()
    collection_budget = resolve_collection_budget(
        num_envs=args.num_envs,
        num_transitions=args.num_transitions,
        num_steps=args.num_steps,
        default_transitions=500_000,
    )
    print(
        "[distill_dagger] budget "
        f"transitions={collection_budget.requested_transitions} "
        f"vector_steps={collection_budget.vector_steps} num_envs={collection_budget.num_envs} "
        f"pretrim={collection_budget.planned_transitions_before_trim}"
    )

    teacher_config, teacher_state, teacher_metadata = load_checkpoint(args.teacher_ckpt)
    student_config, student_state, student_metadata = load_checkpoint(args.student_ckpt)
    OmegaConf.set_struct(teacher_config, False)
    OmegaConf.set_struct(student_config, False)
    teacher_config.experiment.env_params["headless"] = True
    teacher_config.experiment.env_params["num_envs"] = int(args.num_envs)
    apply_collection_overrides(
        teacher_config,
        motion_path=args.motion_path,
        motion_group=args.motion_group,
        traj_index=args.traj_index,
        traj_start_step=args.traj_start_step,
    )
    apply_temporal_params(teacher_config)
    motion_identity_map = MotionIdentityMap.from_paths(resolve_config_motion_paths(teacher_config))

    teacher_fingerprint = checkpoint_content_fingerprint(args.teacher_ckpt)
    student_fingerprint = checkpoint_content_fingerprint(args.student_ckpt)
    teacher_promotion = (
        None
        if args.teacher_promotion_manifest is None
        else validate_stage2_teacher_promotion(
            args.teacher_promotion_manifest,
            teacher_checkpoint=teacher_fingerprint,
        )
    )
    transaction = begin_collection(
        dataset_dir=args.output_dir,
        teacher_checkpoint=teacher_fingerprint,
        student_checkpoint=student_fingerprint,
        collector="dagger_student_rollout_teacher_relabel",
        split=args.split,
        seed=int(args.seed),
        motion_paths=motion_identity_map.motion_paths,
        config_payload={
            "teacher": OmegaConf.to_container(teacher_config, resolve=True),
            "student": OmegaConf.to_container(student_config, resolve=True),
        },
        request_payload={
            "num_envs": int(args.num_envs),
            "num_transitions": int(collection_budget.requested_transitions),
            "shard_size": int(args.shard_size),
            "mix_teacher_action_prob": float(args.mix_teacher_action_prob),
            "save_full_obs": bool(args.save_full_obs),
            "save_reference_features": bool(args.save_reference_features),
            "include_reference_phase": bool(args.include_reference_phase),
            "freeze_run_stats": bool(args.freeze_run_stats),
            "rollout_policy": args.rollout_policy,
        },
        resume=bool(args.resume_dataset),
        run_uid=args.run_uid,
        dagger_iteration=args.dagger_iteration,
        teacher_promotion=teacher_promotion,
        allow_test_only_unpromoted_teacher=bool(
            args.test_only_allow_unpromoted_teacher
        ),
    )
    if transaction.already_complete:
        print(f"[distill_dagger] idempotent collection already complete: {transaction.collection_id}")
        for shard in transaction.existing_paths:
            print(shard)
        return 0

    factory = TaskFactory.get_factory_cls(teacher_config.experiment.task_factory.name)
    env = factory.make(
        **OmegaConf.to_container(teacher_config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(teacher_config.experiment.task_factory.params, resolve=True),
    )
    validate_environment_motion_identity(env, motion_identity_map)
    if getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    teacher_agent_conf = PPOJax.init_agent_conf(env, teacher_config)
    student_agent_conf = PPOJax.init_agent_conf(env, student_config)
    shards = collect_dagger_dataset(
        env=env,
        teacher_agent_conf=teacher_agent_conf,
        teacher_agent_state=teacher_state,
        student_agent_conf=student_agent_conf,
        student_agent_state=student_state,
        output_dir=transaction.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_transitions=args.num_transitions,
        shard_size=args.shard_size,
        seed=args.seed,
        student_obs_filter=OmegaConf.to_container(
            student_config.experiment.get("student_obs_filter", {}),
            resolve=True,
        ),
        mix_teacher_action_prob=args.mix_teacher_action_prob,
        # The immutable transaction stages into an empty directory and assigns
        # final append indices exactly once during commit.
        append=False,
        save_full_obs=args.save_full_obs,
        save_reference_features=bool(args.save_reference_features),
        include_reference_phase=bool(args.include_reference_phase),
        freeze_run_stats=bool(args.freeze_run_stats),
        split=args.split,
        metadata={
            "teacher_ckpt": args.teacher_ckpt,
            "student_ckpt": args.student_ckpt,
            "teacher_checkpoint_fingerprint": teacher_fingerprint,
            "student_checkpoint_fingerprint": student_fingerprint,
            "teacher_promotion": transaction.manifest["teacher_promotion"],
            "distill_run_uid": transaction.manifest["run_uid"],
            "collection_contract_fingerprint": canonical_json_sha256(transaction.contract),
            "student_ckpt_in": args.student_ckpt,
            "dagger_iteration": args.dagger_iteration,
            "rollout_policy": args.rollout_policy,
            "teacher_checkpoint_step": int(getattr(teacher_metadata, "step", 0) or 0),
            "student_checkpoint_step": int(getattr(student_metadata, "step", 0) or 0),
            "teacher_config": OmegaConf.to_container(teacher_config, resolve=True),
            "student_config": OmegaConf.to_container(student_config, resolve=True),
        },
        motion_identity_map=motion_identity_map,
    )
    shards = transaction.commit(shards)
    for shard in shards:
        print(shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
