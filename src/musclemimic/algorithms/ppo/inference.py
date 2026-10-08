"""
Policy inference and visualization for PPO.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers, wrap_env
from musclemimic.distill.obs_filter import build_student_obs_indices, filter_student_obs

if TYPE_CHECKING:
    from musclemimic.algorithms.ppo.config import PPOAgentConf, PPOAgentState


class ObservationHistoryBuffer:
    """Observation history buffer for MuJoCo inference.

    Mirrors the NStepWrapper behavior for single-env NumPy-based inference.
    Maintains a rolling buffer of the last n_steps observations.

    When split_goal=True, only state indices are stacked and goal indices
    stay at the current timestep. Output: [state_hist, goal].
    """

    def __init__(
        self,
        n_steps: int,
        split_goal: bool = False,
        state_indices: np.ndarray | None = None,
        goal_indices: np.ndarray | None = None,
    ):
        self.n_steps = n_steps
        self.split_goal = split_goal
        self.state_indices = None if state_indices is None else np.asarray(state_indices, dtype=int)
        self.goal_indices = None if goal_indices is None else np.asarray(goal_indices, dtype=int)
        self._buffer: np.ndarray | None = None
        if self.split_goal and (self.state_indices is None or self.goal_indices is None):
            raise ValueError("split_goal=True requires state_indices and goal_indices")

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Initialize buffer with zeros and set last entry to initial observation."""
        obs = np.asarray(obs).flatten()

        if self.split_goal:
            state_obs = obs[self.state_indices]
            goal_obs = obs[self.goal_indices]

            self._buffer = np.zeros((self.n_steps, state_obs.shape[0]), dtype=obs.dtype)
            self._buffer[-1] = state_obs
            return np.concatenate([self._buffer.flatten(), goal_obs])
        else:
            self._buffer = np.zeros((self.n_steps, obs.shape[0]), dtype=obs.dtype)
            self._buffer[-1] = obs
            return self._buffer.flatten()

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Roll buffer left and append new observation."""
        obs = np.asarray(obs).flatten()

        if self.split_goal:
            state_obs = obs[self.state_indices]
            goal_obs = obs[self.goal_indices]

            self._buffer = np.roll(self._buffer, shift=-1, axis=0)
            self._buffer[-1] = state_obs
            return np.concatenate([self._buffer.flatten(), goal_obs])
        else:
            self._buffer = np.roll(self._buffer, shift=-1, axis=0)
            self._buffer[-1] = obs
            return self._buffer.flatten()


def play_policy(
    env: Any,
    agent_conf: PPOAgentConf,
    agent_state: PPOAgentState,
    n_envs: int,
    n_steps: int | None = None,
    render: bool = True,
    record: bool = False,
    rng: jax.Array | None = None,
    deterministic: bool = False,
    use_mujoco: bool = False,
    do_wrap_env: bool = True,
    train_state_seed: int | None = None,
    sequential_mjx: bool = False,
    debug_overlay: bool = True,
    freeze_run_stats: bool = False,
) -> None:
    """
    Run policy in environment for visualization or evaluation.

    Args:
        env: environment instance
        agent_conf: agent configuration with network
        agent_state: agent state with trained parameters
        n_envs: number of parallel environments
        n_steps: max steps to run (None = infinite)
        render: whether to render
        record: whether to record video
        rng: random key (default: key(0))
        deterministic: use mean actions (no sampling)
        use_mujoco: use cpu mujoco backend
        do_wrap_env: apply standard wrappers
        train_state_seed: seed index for multi-seed checkpoints
        sequential_mjx: use sequential mjx mode (single env, manual resets)
        freeze_run_stats: keep checkpoint observation-normalization statistics
            immutable for evidence-producing deterministic evaluation
    """
    if sequential_mjx:
        n_envs = 1
        do_wrap_env = False

    config = agent_conf.config.experiment
    sequential_interface = None
    interface_env = env
    if use_mujoco and not sequential_mjx:
        # CPU evaluation cannot use Vec/Log wrappers, but it must use the same
        # policy-facing finger action/observation contract as training.
        env = apply_policy_interface_wrappers(env, config, include_student=False)
        interface_env = env
    elif sequential_mjx:
        # Sequential MJX calls mjx_reset/mjx_step directly. Keep the raw env for
        # those calls and use this object only to project observations/actions.
        sequential_interface = apply_policy_interface_wrappers(
            env, config, include_student=False
        )
        interface_env = sequential_interface

    # Determine observation history length for MuJoCo inference
    len_obs_history = 1
    split_goal = False
    state_indices = None
    goal_indices = None
    student_obs_spec = None

    if use_mujoco or sequential_mjx:
        student_cfg = config.get("student_obs_filter", {})
        if student_cfg.get("enabled", False):
            student_obs_spec = build_student_obs_indices(interface_env, student_cfg)
        if hasattr(config, "len_obs_history"):
            len_obs_history = config.len_obs_history
        if hasattr(config, "split_goal"):
            split_goal = config.split_goal
        if split_goal:
            obs_group_env = apply_policy_interface_wrappers(interface_env, config)
            if not hasattr(obs_group_env, "obs_container"):
                raise ValueError("split_goal=True requires env.obs_container with goal group indices")
            goal_indices = obs_group_env.obs_container.get_obs_ind_by_group("goal")
            if goal_indices.size == 0:
                raise ValueError("split_goal=True requires goal observations grouped as 'goal'")
            raw_obs_dim = obs_group_env.info.observation_space.shape[0]
            goal_indices = np.asarray(goal_indices, dtype=int)
            state_mask = np.ones(raw_obs_dim, dtype=bool)
            state_mask[goal_indices] = False
            state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]

    # Create observation history buffer if needed
    if len_obs_history > 1:
        obs_buffer = ObservationHistoryBuffer(
            len_obs_history,
            split_goal=split_goal,
            state_indices=state_indices,
            goal_indices=goal_indices,
        )
    else:
        obs_buffer = None

    if use_mujoco:
        assert n_envs == 1, "only one mujoco env can run at a time"

    train_state = agent_state.train_state

    if config.n_seeds > 1:
        assert train_state_seed is not None, "loaded train state has multiple seeds; specify train_state_seed"
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    if not use_mujoco and getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    if not render and n_steps is None and not record:
        warnings.warn(
            "no rendering, no record, no n_steps; will run forever with no effect",
            stacklevel=2,
        )

    if do_wrap_env and not use_mujoco:
        env = wrap_env(env, config)

    if rng is None:
        rng = jax.random.key(0)

    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]

    def sample_actions(ts, obs, _rng):
        obs_b = jnp.atleast_2d(obs) if hasattr(obs, "ndim") and obs.ndim == 1 else obs
        vars_in = {"params": ts.params, "run_stats": ts.run_stats}
        if freeze_run_stats:
            y = agent_conf.network.apply(vars_in, obs_b)
            ts_out = ts
        else:
            y, updates = agent_conf.network.apply(vars_in, obs_b, mutable=["run_stats"])
            ts_out = ts.replace(run_stats=updates["run_stats"])
        pi, _ = y
        a = pi.mode() if deterministic else pi.sample(seed=_rng)
        if hasattr(a, "ndim") and a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return a, ts_out

    policy_fn = jax.jit(sample_actions)

    def project_manual_observation(raw_obs):
        projected = raw_obs
        if sequential_interface is not None and hasattr(
            sequential_interface, "filter_observation"
        ):
            projected = sequential_interface.filter_observation(projected)
        if student_obs_spec is not None:
            projected = filter_student_obs(jnp.asarray(projected), student_obs_spec)
        return projected

    # reset environment
    if use_mujoco and not sequential_mjx:
        obs = env.reset()
        obs = np.asarray(project_manual_observation(obs))
        if obs_buffer is not None:
            obs = obs_buffer.reset(obs)
        env_state = None
    elif sequential_mjx:
        env_state = env.mjx_reset(env_keys[0])
        obs = project_manual_observation(env_state.observation)
        if obs_buffer is not None:
            obs = obs_buffer.reset(np.asarray(obs))
    else:
        obs, env_state = env.reset(env_keys)

    if n_steps is None:
        n_steps = np.iinfo(np.int32).max

    # Counters for debug overlay
    global_step = 0
    episode_step = 0
    episode_count = 0
    episode_reward = 0.0
    prev_done = False  # Track previous step's done for debugging

    for _ in range(n_steps):
        rng, _rng = jax.random.split(rng)
        action, train_state = policy_fn(train_state, obs, _rng)
        action = jnp.atleast_2d(action)

        # Initialize defaults for non-mujoco paths
        _absorbing = False
        _info = {}

        _reward = 0.0
        done = False
        episode_done = False
        if use_mujoco and not sequential_mjx:
            obs, _reward, _absorbing, done, _info = env.step(action)
            obs = np.asarray(project_manual_observation(obs))
            if obs_buffer is not None:
                obs = obs_buffer.step(obs)
            _reward = float(np.asarray(_reward).item())
            _absorbing = bool(np.asarray(_absorbing).item())
            done = bool(np.asarray(done).item())
            episode_done = done
        elif sequential_mjx:
            action_single = jnp.squeeze(action, axis=0)
            if sequential_interface is not None and hasattr(
                sequential_interface, "decode_action"
            ):
                action_single = sequential_interface.decode_action(
                    action_single
                ).body_action
            if sequential_interface is not None and hasattr(
                sequential_interface, "expand_body_action"
            ):
                action_single = sequential_interface.expand_body_action(action_single)
            env_state = env.mjx_step(env_state, action_single)
            obs = project_manual_observation(env_state.observation)
            if obs_buffer is not None:
                obs = obs_buffer.step(np.asarray(obs))
            done = env_state.done
            done = bool(np.asarray(done).item())
            episode_done = done
        else:
            obs, _reward, _absorbing, done, _info, env_state = env.step(env_state, action)

        # Extract termination info
        terminated = _info.get('terminated', False) if isinstance(_info, dict) else False
        truncated = _info.get('truncated', False) if isinstance(_info, dict) else False
        traj_no = _info.get('traj_no', '?') if isinstance(_info, dict) else '?'
        subtraj_step = _info.get('subtraj_step_no', '?') if isinstance(_info, dict) else '?'
        traj_len = _info.get('traj_len', '?') if isinstance(_info, dict) else '?'

        debug_info = None
        if debug_overlay:
            debug_info = {
                'global_step': global_step,
                'episode_step': episode_step,
                'done': int(done) if isinstance(done, (bool, np.bool_)) else done,
                'prev_done': int(prev_done) if isinstance(prev_done, (bool, np.bool_)) else prev_done,
                'absorbing': int(_absorbing) if isinstance(_absorbing, (bool, np.bool_)) else _absorbing,
                'terminated': int(terminated) if isinstance(terminated, (bool, np.bool_)) else terminated,
                'truncated': int(truncated) if isinstance(truncated, (bool, np.bool_)) else truncated,
                'traj_no': traj_no,
                'subtraj_step': subtraj_step,
                'traj_len': traj_len,
            }

        if render:
            if use_mujoco and not sequential_mjx:
                env.render(record=record, debug_info=debug_info)
            elif sequential_mjx:

                def _add_batch(x):
                    if hasattr(x, "shape") and x.shape:
                        return x[None]
                    elif isinstance(x, (int, float, bool, np.number)):
                        return jnp.array([x])
                    return x

                env_state_batched = jax.tree_util.tree_map(_add_batch, env_state)
                env.mjx_render(env_state_batched, record=record, debug_info=debug_info)
            else:
                env.mjx_render(env_state, record=record, debug_info=debug_info)

        global_step += 1
        episode_step += 1

        if use_mujoco or sequential_mjx:
            episode_reward += _reward
            if episode_done:
                episode_count += 1
                if use_mujoco and not sequential_mjx and isinstance(_info, dict):
                    # Determine termination reason
                    terminated = _info.get('terminated', False)
                    truncated = _info.get('truncated', False)
                    if terminated:
                        reason = "TERMINATED (early)"
                    elif truncated:
                        reason = "TRUNCATED (time limit)"
                    else:
                        reason = "DONE"

                    # Trajectory progress
                    traj_no = _info.get('traj_no', '?')
                    subtraj_step = _info.get('subtraj_step_no', '?')
                    traj_len = _info.get('traj_len', '?')
                    progress = ""
                    if isinstance(subtraj_step, (int, float)) and isinstance(traj_len, (int, float)) and traj_len > 0:
                        pct = subtraj_step / traj_len * 100
                        progress = f" ({pct:.0f}%)"

                    print(f"\n--- Episode {episode_count} {reason} | steps={episode_step}, reward={episode_reward:.2f} ---")
                    print(f"  trajectory: #{traj_no}, step {subtraj_step}/{traj_len}{progress}")

                    # Reward breakdown
                    reward_keys = [k for k in _info if k.startswith('reward_')]
                    penalty_keys = [k for k in _info if k.startswith('penalty_')]
                    if reward_keys:
                        parts = [f"{k.removeprefix('reward_')}={_info[k]:.3f}" for k in sorted(reward_keys)]
                        print(f"  rewards: {', '.join(parts)}")
                    if penalty_keys:
                        parts = [f"{k.removeprefix('penalty_')}={_info[k]:.3f}" for k in sorted(penalty_keys)]
                        print(f"  penalties: {', '.join(parts)}")

                    # Error metrics
                    err_keys = [k for k in _info if k.startswith('err_')]
                    if err_keys:
                        parts = [f"{k.removeprefix('err_')}={_info[k]:.4f}" for k in sorted(err_keys)]
                        print(f"  errors: {', '.join(parts)}")

                episode_step = 0
                episode_reward = 0.0
                if use_mujoco:
                    obs = env.reset()
                    obs = np.asarray(project_manual_observation(obs))
                    if obs_buffer is not None:
                        obs = obs_buffer.reset(obs)
                else:
                    rng, reset_key = jax.random.split(rng)
                    env_state = env.mjx_reset(reset_key)
                    obs = project_manual_observation(env_state.observation)
                    if obs_buffer is not None:
                        obs = obs_buffer.reset(np.asarray(obs))

        # Update prev_done for next iteration
        prev_done = episode_done if (use_mujoco or sequential_mjx) else done

    env.stop()


def play_policy_mujoco(
    env: Any,
    agent_conf: PPOAgentConf,
    agent_state: PPOAgentState,
    n_steps: int | None = None,
    render: bool = True,
    record: bool = False,
    rng: jax.Array | None = None,
    deterministic: bool = False,
    train_state_seed: int | None = None,
    debug_overlay: bool = True,
    freeze_run_stats: bool = False,
) -> None:
    """
    Convenience wrapper for play_policy with mujoco backend.
    """
    play_policy(
        env,
        agent_conf,
        agent_state,
        1,
        n_steps,
        render,
        record,
        rng,
        deterministic,
        True,
        False,
        train_state_seed,
        False,
        debug_overlay,
        freeze_run_stats,
    )
