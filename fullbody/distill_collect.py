"""Command line entrypoint for collecting teacher rollouts for distillation."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.collect_teacher import collect_teacher_dataset
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect teacher rollout shards for student distillation.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=200_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic_teacher", action="store_true", default=False)
    parser.add_argument("--save_full_obs", action="store_true", default=False)
    parser.add_argument("--freeze_run_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--motion_group", default=None)
    parser.add_argument("--traj_index", type=int, default=None)
    parser.add_argument("--traj_start_step", type=int, default=None)
    args = parser.parse_args()

    config, agent_state, metadata = load_checkpoint(args.teacher_ckpt)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    config.experiment.env_params["num_envs"] = int(args.num_envs)
    apply_collection_overrides(
        config,
        motion_path=args.motion_path,
        motion_group=args.motion_group,
        traj_index=args.traj_index,
        traj_start_step=args.traj_start_step,
    )
    apply_temporal_params(config)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **OmegaConf.to_container(config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(config.experiment.task_factory.params, resolve=True),
    )
    if getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    agent_conf = PPOJax.init_agent_conf(env, config)
    shards = collect_teacher_dataset(
        env=env,
        agent_conf=agent_conf,
        agent_state=agent_state,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        shard_size=args.shard_size,
        deterministic_teacher=bool(args.deterministic_teacher),
        seed=args.seed,
        save_full_obs=bool(args.save_full_obs),
        freeze_run_stats=bool(args.freeze_run_stats),
        split=args.split,
        metadata={
            "teacher_ckpt": args.teacher_ckpt,
            "teacher_checkpoint_step": int(getattr(metadata, "step", 0) or 0),
            "teacher_config": OmegaConf.to_container(config, resolve=True),
        },
    )
    for shard in shards:
        print(shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
