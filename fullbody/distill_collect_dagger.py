"""Collect DAgger-style student rollout states relabeled by a teacher policy."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.dagger import collect_dagger_dataset
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect DAgger relabel shards for student distillation.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=50_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix_teacher_action_prob", type=float, default=0.0)
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--save_full_obs", action="store_true", default=False)
    parser.add_argument("--save_reference_features", action="store_true", default=False)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument("--freeze_run_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--motion_group", default=None)
    parser.add_argument("--traj_index", type=int, default=None)
    parser.add_argument("--traj_start_step", type=int, default=None)
    parser.add_argument("--dagger_iteration", type=int, default=None)
    parser.add_argument("--rollout_policy", default=None)
    args = parser.parse_args()

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

    factory = TaskFactory.get_factory_cls(teacher_config.experiment.task_factory.name)
    env = factory.make(
        **OmegaConf.to_container(teacher_config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(teacher_config.experiment.task_factory.params, resolve=True),
    )
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
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        shard_size=args.shard_size,
        seed=args.seed,
        student_obs_filter=OmegaConf.to_container(
            student_config.experiment.get("student_obs_filter", {}),
            resolve=True,
        ),
        mix_teacher_action_prob=args.mix_teacher_action_prob,
        append=args.append,
        save_full_obs=args.save_full_obs,
        save_reference_features=bool(args.save_reference_features),
        include_reference_phase=bool(args.include_reference_phase),
        freeze_run_stats=bool(args.freeze_run_stats),
        split=args.split,
        metadata={
            "teacher_ckpt": args.teacher_ckpt,
            "student_ckpt": args.student_ckpt,
            "student_ckpt_in": args.student_ckpt,
            "dagger_iteration": args.dagger_iteration,
            "rollout_policy": args.rollout_policy,
            "teacher_checkpoint_step": int(getattr(teacher_metadata, "step", 0) or 0),
            "student_checkpoint_step": int(getattr(student_metadata, "step", 0) or 0),
            "teacher_config": OmegaConf.to_container(teacher_config, resolve=True),
            "student_config": OmegaConf.to_container(student_config, resolve=True),
        },
    )
    for shard in shards:
        print(shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
