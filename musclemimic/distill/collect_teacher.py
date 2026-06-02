"""Teacher rollout collection for student policy distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.distill.dataset import write_distill_shard
from musclemimic.distill.losses import distribution_mean
from musclemimic.distill.obs_filter import build_student_obs_indices, filter_student_obs


def _tree_get_info(info: dict[str, Any], key: str, shape, dtype):
    value = info.get(key)
    if value is None:
        return np.zeros(shape, dtype=dtype)
    return np.asarray(jax.device_get(value), dtype=dtype)


def collect_teacher_dataset(
    *,
    env: Any,
    agent_conf: Any,
    agent_state: Any,
    output_dir: str | Path,
    num_envs: int,
    num_steps: int,
    shard_size: int = 50_000,
    deterministic_teacher: bool = True,
    seed: int = 0,
    student_obs_filter: dict[str, Any] | None = None,
    save_full_obs: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Collect student_obs -> teacher_action samples from a PPO teacher."""
    if agent_conf.config.experiment.get("len_obs_history", 1) > 1:
        raise NotImplementedError("distill collection currently supports len_obs_history=1 teacher policies")

    filter_cfg = {
        "enabled": True,
        "keep_motion_phase": True,
        "require_goal_group": True,
        "require_motion_phase": True,
    }
    if student_obs_filter:
        filter_cfg.update(student_obs_filter)
    spec = build_student_obs_indices(env, filter_cfg)

    exp_cfg = OmegaConf.create(OmegaConf.to_container(agent_conf.config.experiment, resolve=True))
    exp_cfg.num_envs = int(num_envs)
    teacher_env = wrap_env(env, exp_cfg)

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = teacher_env.reset(jax.random.split(reset_rng, int(num_envs)))
    train_state = agent_state.train_state

    @jax.jit
    def policy_step(ts, cur_obs, cur_env_state, cur_rng):
        cur_rng, action_rng = jax.random.split(cur_rng)
        (pi, value), updates = agent_conf.network.apply(
            {"params": ts.params, "run_stats": ts.run_stats},
            cur_obs,
            mutable=["run_stats"],
        )
        mean_action = distribution_mean(pi)
        action = mean_action if deterministic_teacher else pi.sample(seed=action_rng)
        log_prob = pi.log_prob(action)
        next_obs, reward, absorbing, done, info, next_env_state, _transition_state = teacher_env.step_with_transition(
            cur_env_state,
            action,
        )
        ts = ts.replace(run_stats=updates["run_stats"])
        return ts, next_obs, next_env_state, cur_rng, mean_action, action, value, log_prob, reward, absorbing, done, info

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    buffers: dict[str, list[np.ndarray]] = {}
    total_written = 0
    shard_idx = 0

    def append(name: str, value):
        buffers.setdefault(name, []).append(np.asarray(jax.device_get(value)))

    def flush(force: bool = False):
        nonlocal shard_idx, total_written
        if not buffers:
            return
        current_n = sum(part.shape[0] for part in buffers["student_obs"])
        if current_n < int(shard_size) and not force:
            return
        data = {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}
        shard = write_distill_shard(
            output_path / f"shard_{shard_idx:06d}.npz",
            data,
            metadata={
                **(metadata or {}),
                "num_envs": int(num_envs),
                "requested_num_steps": int(num_steps),
                "student_obs_filter": filter_cfg,
                "student_obs_dim": int(spec.student_obs_dim),
                "action_dim": int(data["teacher_action"].shape[-1]),
            },
        )
        written.append(shard)
        total_written += int(data["student_obs"].shape[0])
        shard_idx += 1
        buffers.clear()

    for _ in range(int(num_steps)):
        train_state, next_obs, env_state, rng, mean_action, action, value, log_prob, reward, absorbing, done, info = policy_step(
            train_state,
            obs,
            env_state,
            rng,
        )
        student_obs = filter_student_obs(obs, spec)
        phase_idx = spec.phase_student_index
        phase = student_obs[..., phase_idx] if phase_idx is not None else jnp.zeros(student_obs.shape[0])

        append("student_obs", student_obs)
        append("teacher_action", mean_action if deterministic_teacher else action)
        append("teacher_mu", mean_action)
        append("teacher_value", value)
        append("teacher_log_prob", log_prob)
        append("reward", reward)
        append("done", done)
        append("absorbing", absorbing)
        append("traj_no", _tree_get_info(info, "traj_no", (int(num_envs),), np.int32))
        append("subtraj_step_no", _tree_get_info(info, "subtraj_step_no", (int(num_envs),), np.int32))
        append("phase", phase)
        if save_full_obs:
            append("full_obs", obs)

        obs = next_obs
        flush(force=False)

    flush(force=True)
    print(f"[distill_collect] wrote {total_written} samples in {len(written)} shards to {output_path}")
    return written
