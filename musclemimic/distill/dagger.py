"""DAgger-style student rollout collection with teacher relabeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.distill.dataset import write_distill_shard, write_split_shard
from musclemimic.distill.losses import distribution_log_std, distribution_mean
from musclemimic.distill.obs_filter import StudentObsSpec, build_student_obs_indices, filter_student_obs


def _info_array(info: dict[str, Any], key: str, n: int, dtype):
    value = info.get(key)
    if value is None:
        return np.zeros((n,), dtype=dtype)
    return np.asarray(jax.device_get(value), dtype=dtype)


def build_dagger_shard_data(
    *,
    full_obs,
    teacher_mu,
    student_action,
    reward,
    done,
    absorbing,
    info: dict[str, Any],
    spec: StudentObsSpec,
    rollout_action=None,
    used_teacher_action=None,
    teacher_value=None,
    teacher_log_prob=None,
    teacher_log_std=None,
    teacher_log_prob_teacher_mu=None,
    teacher_log_prob_student_action=None,
    teacher_log_prob_rollout_action=None,
    save_full_obs: bool = False,
) -> dict[str, np.ndarray]:
    """Build a shard batch from student-visited states labeled by teacher mean."""
    full_obs_np = np.asarray(jax.device_get(full_obs), dtype=np.float32)
    student_obs = np.asarray(jax.device_get(filter_student_obs(jnp.asarray(full_obs_np), spec)), dtype=np.float32)
    n = int(student_obs.shape[0])
    phase_idx = spec.phase_student_index
    phase = (
        np.asarray(student_obs[..., phase_idx], dtype=np.float32)
        if phase_idx is not None
        else np.zeros((n,), dtype=np.float32)
    )

    data = {
        "student_obs": student_obs,
        "teacher_action": np.asarray(jax.device_get(teacher_mu), dtype=np.float32),
        "teacher_mu": np.asarray(jax.device_get(teacher_mu), dtype=np.float32),
        "student_action": np.asarray(jax.device_get(student_action), dtype=np.float32),
        "rollout_action": np.asarray(
            jax.device_get(rollout_action if rollout_action is not None else student_action),
            dtype=np.float32,
        ),
        "used_teacher_action": np.asarray(
            jax.device_get(
                used_teacher_action if used_teacher_action is not None else np.zeros((n,), dtype=bool)
            ),
            dtype=bool,
        ),
        "reward": np.asarray(jax.device_get(reward), dtype=np.float32),
        "done": np.asarray(jax.device_get(done), dtype=bool),
        "absorbing": np.asarray(jax.device_get(absorbing), dtype=bool),
        "traj_no": _info_array(info, "traj_no", n, np.int32),
        "subtraj_step_no": _info_array(info, "subtraj_step_no", n, np.int32),
        "phase": phase,
    }
    if teacher_value is not None:
        data["teacher_value"] = np.asarray(jax.device_get(teacher_value), dtype=np.float32)
    if teacher_log_prob is not None:
        data["teacher_log_prob"] = np.asarray(jax.device_get(teacher_log_prob), dtype=np.float32)
    if teacher_log_std is not None:
        data["teacher_log_std"] = np.asarray(jax.device_get(teacher_log_std), dtype=np.float32)
    if teacher_log_prob_teacher_mu is not None:
        data["teacher_log_prob_teacher_mu"] = np.asarray(jax.device_get(teacher_log_prob_teacher_mu), dtype=np.float32)
    if teacher_log_prob_student_action is not None:
        data["teacher_log_prob_student_action"] = np.asarray(
            jax.device_get(teacher_log_prob_student_action),
            dtype=np.float32,
        )
    if teacher_log_prob_rollout_action is not None:
        data["teacher_log_prob_rollout_action"] = np.asarray(
            jax.device_get(teacher_log_prob_rollout_action),
            dtype=np.float32,
        )
    if save_full_obs:
        data["full_obs"] = full_obs_np
    return data


def collect_dagger_dataset(
    *,
    env: Any,
    teacher_agent_conf: Any,
    teacher_agent_state: Any,
    student_agent_conf: Any,
    student_agent_state: Any,
    output_dir: str | Path,
    num_envs: int,
    num_steps: int,
    shard_size: int = 50_000,
    seed: int = 0,
    student_obs_filter: dict[str, Any] | None = None,
    mix_teacher_action_prob: float = 0.0,
    append: bool = False,
    save_full_obs: bool = False,
    freeze_run_stats: bool = True,
    split: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Roll out student actions while labeling visited states with teacher mean.

    First version supports non-history student policies. This matches the
    provided student phase configs and keeps relabeling unambiguous.
    """
    teacher_exp = teacher_agent_conf.config.experiment
    student_exp = student_agent_conf.config.experiment
    if teacher_exp.get("len_obs_history", 1) > 1:
        raise NotImplementedError("DAgger collection currently supports len_obs_history=1 teacher policies")
    if student_exp.get("len_obs_history", 1) > 1:
        raise NotImplementedError("DAgger collection currently supports len_obs_history=1 student policies")

    filter_cfg = {
        "enabled": True,
        "keep_motion_phase": True,
        "require_goal_group": True,
        "require_motion_phase": True,
    }
    if student_obs_filter:
        filter_cfg.update(student_obs_filter)
    spec = build_student_obs_indices(env, filter_cfg)

    rollout_cfg = OmegaConf.create(OmegaConf.to_container(teacher_exp, resolve=True))
    rollout_cfg.num_envs = int(num_envs)
    if "student_obs_filter" in rollout_cfg:
        rollout_cfg.student_obs_filter.enabled = False
    rollout_env = wrap_env(env, rollout_cfg)

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    full_obs, env_state = rollout_env.reset(jax.random.split(reset_rng, int(num_envs)))
    teacher_ts = teacher_agent_state.train_state
    student_ts = student_agent_state.train_state

    @jax.jit
    def policy_step(t_ts, s_ts, cur_full_obs, cur_env_state, cur_rng):
        cur_rng, student_rng, mix_rng = jax.random.split(cur_rng, 3)
        student_obs = filter_student_obs(cur_full_obs, spec)
        (student_pi, _student_value), s_updates = student_agent_conf.network.apply(
            {"params": s_ts.params, "run_stats": s_ts.run_stats},
            student_obs,
            mutable=["run_stats"],
        )
        student_action = student_pi.sample(seed=student_rng)

        (teacher_pi, teacher_value), t_updates = teacher_agent_conf.network.apply(
            {"params": t_ts.params, "run_stats": t_ts.run_stats},
            cur_full_obs,
            mutable=["run_stats"],
        )
        teacher_mu = distribution_mean(teacher_pi)
        teacher_log_std = jnp.broadcast_to(distribution_log_std(teacher_pi), teacher_mu.shape)
        teacher_log_prob_teacher_mu = teacher_pi.log_prob(teacher_mu)
        teacher_log_prob_student_action = teacher_pi.log_prob(student_action)
        use_teacher = jax.random.uniform(mix_rng, shape=(int(num_envs),)) < float(mix_teacher_action_prob)
        rollout_action = jnp.where(use_teacher[:, None], teacher_mu, student_action)
        teacher_log_prob_rollout_action = teacher_pi.log_prob(rollout_action)
        next_obs, reward, absorbing, done, info, next_env_state, _transition_state = rollout_env.step_with_transition(
            cur_env_state,
            rollout_action,
        )
        t_ts = t_ts if freeze_run_stats else t_ts.replace(run_stats=t_updates["run_stats"])
        s_ts = s_ts if freeze_run_stats else s_ts.replace(run_stats=s_updates["run_stats"])
        return (
            t_ts,
            s_ts,
            next_obs,
            next_env_state,
            cur_rng,
            teacher_mu,
            teacher_value,
            teacher_log_std,
            teacher_log_prob_teacher_mu,
            teacher_log_prob_student_action,
            teacher_log_prob_rollout_action,
            student_action,
            rollout_action,
            use_teacher,
            reward,
            absorbing,
            done,
            info,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start_idx = 0
    if append:
        pattern = f"{split}_*.npz" if split else "shard_*.npz"
        existing = sorted(output_path.glob(pattern))
        if existing:
            start_idx = max(int(path.stem.split("_")[-1]) for path in existing) + 1

    written: list[Path] = []
    buffers: dict[str, list[np.ndarray]] = {}
    shard_idx = start_idx
    total_written = 0

    def append_buffer(data: dict[str, np.ndarray]):
        for name, value in data.items():
            buffers.setdefault(name, []).append(value)

    def flush(force: bool = False):
        nonlocal shard_idx, total_written
        if not buffers:
            return
        current_n = sum(part.shape[0] for part in buffers["student_obs"])
        if current_n < int(shard_size) and not force:
            return
        data = {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}
        shard_metadata = {
            **(metadata or {}),
            "collector": "dagger_student_rollout_teacher_relabel",
            "num_envs": int(num_envs),
            "requested_num_steps": int(num_steps),
            "student_obs_filter": filter_cfg,
            "student_obs_dim": int(spec.student_obs_dim),
            "action_dim": int(data["teacher_action"].shape[-1]),
            "mix_teacher_action_prob": float(mix_teacher_action_prob),
            "freeze_run_stats": bool(freeze_run_stats),
        }
        if split:
            shard = write_split_shard(output_path, data, split=split, shard_idx=shard_idx, metadata=shard_metadata)
        else:
            shard = write_distill_shard(output_path / f"shard_{shard_idx:06d}.npz", data, metadata=shard_metadata)
        written.append(shard)
        total_written += int(data["student_obs"].shape[0])
        shard_idx += 1
        buffers.clear()

    for _ in range(int(num_steps)):
        (
            teacher_ts,
            student_ts,
            next_full_obs,
            env_state,
            rng,
            teacher_mu,
            teacher_value,
            teacher_log_std,
            teacher_log_prob_teacher_mu,
            teacher_log_prob_student_action,
            teacher_log_prob_rollout_action,
            student_action,
            rollout_action,
            used_teacher_action,
            reward,
            absorbing,
            done,
            info,
        ) = policy_step(teacher_ts, student_ts, full_obs, env_state, rng)
        append_buffer(
            build_dagger_shard_data(
                full_obs=full_obs,
                teacher_mu=teacher_mu,
                student_action=student_action,
                rollout_action=rollout_action,
                used_teacher_action=used_teacher_action,
                reward=reward,
                done=done,
                absorbing=absorbing,
                info=info,
                spec=spec,
                teacher_value=teacher_value,
                teacher_log_prob=teacher_log_prob_teacher_mu,
                teacher_log_std=teacher_log_std,
                teacher_log_prob_teacher_mu=teacher_log_prob_teacher_mu,
                teacher_log_prob_student_action=teacher_log_prob_student_action,
                teacher_log_prob_rollout_action=teacher_log_prob_rollout_action,
                save_full_obs=save_full_obs,
            )
        )
        full_obs = next_full_obs
        flush(force=False)

    flush(force=True)
    print(f"[distill_dagger] wrote {total_written} relabeled samples in {len(written)} shards to {output_path}")
    return written
