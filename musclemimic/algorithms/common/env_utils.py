"""
Environment wrapping and observation utilities for RL training.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

from musclemimic.core.wrappers import (
    AutoResetWrapper,
    BodyFingerIsolationWrapper,
    LogWrapper,
    NormalizeVecReward,
    NStepWrapper,
    SynergyActionWrapper,
    VecEnv,
)
from musclemimic.distill.obs_filter import StudentObservationFilterWrapper


def apply_policy_interface_wrappers(
    env: Any,
    config: DictConfig,
    *,
    include_student: bool = True,
) -> Any:
    """Apply wrappers that define the policy's observation/action contract.

    This function is intentionally shared by network construction, training,
    validation and inference.  Interface wrappers must be applied before
    history/vector/log wrappers so the network dimensions and runtime tensors
    cannot drift apart.
    """
    finger_cfg = config.get("finger_isolation", {})
    if finger_cfg.get("enabled", False) and _find_wrapper(
        env, BodyFingerIsolationWrapper
    ) is None:
        env = BodyFingerIsolationWrapper(env, finger_cfg)

    action_cfg = config.get("action_representation", {})
    if action_cfg.get("enabled", False):
        synergy_wrapper = _find_wrapper(env, SynergyActionWrapper)
        if synergy_wrapper is None:
            env = SynergyActionWrapper(env, action_cfg)
            synergy_wrapper = env
        _bind_early_synergy_runtime_config(config, synergy_wrapper)

    student_cfg = config.get("student_obs_filter", {})
    if (
        include_student
        and student_cfg.get("enabled", False)
        and _find_wrapper(env, StudentObservationFilterWrapper) is None
    ):
        env = StudentObservationFilterWrapper(env, student_cfg)
    return env


def _find_wrapper(env: Any, wrapper_type: type) -> Any | None:
    """Find an interface wrapper through an existing wrapper chain."""

    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, wrapper_type):
            return current
        visited.add(id(current))
        next_env = getattr(current, "env", None)
        if next_env is current:
            break
        current = next_env
    return None


def _bind_early_synergy_runtime_config(
    config: DictConfig,
    env: SynergyActionWrapper,
) -> None:
    """Bind runtime-resolved decoder/exploration identities before config hash."""

    manifest = dict(env.action_manifest)
    exploration_cfg = config.action_representation.get("exploration", {}) or {}
    calibrate = bool(exploration_cfg.get("calibrate_in_physical_space", False))
    if calibrate:
        from musclemimic.synergy.exploration_scaling import calibrate_exploration_std

        std_vector = calibrate_exploration_std(
            env.decoder_jacobian_at_zero,
            float(exploration_cfg.get("target_initial_excitation_rms", 0.08)),
            mode=str(exploration_cfg.get("std_mode", "per_dimension")),
            residual_dim=env.action_interface.residual_dim,
            residual_std_scale=float(exploration_cfg.get("residual_std_scale", 0.25)),
            gram_epsilon=float(exploration_cfg.get("gram_epsilon", 1e-6)),
            min_std=float(exploration_cfg.get("min_std", 1e-4)),
            max_std=float(exploration_cfg.get("max_std", 2.0)),
        )
        with open_dict(config):
            config.init_std_vector = [float(value) for value in std_vector]
        exploration_manifest = {
            "kind": "physical_decoder_jacobian_calibration_v1",
            "mode": str(exploration_cfg.get("std_mode", "per_dimension")),
            "target_initial_excitation_rms": float(
                exploration_cfg.get("target_initial_excitation_rms", 0.08)
            ),
            "residual_std_scale": float(
                exploration_cfg.get("residual_std_scale", 0.25)
            ),
            "init_std_vector": [float(value) for value in std_vector],
        }
    else:
        configured_vector = config.get("init_std_vector", None)
        exploration_manifest = {
            "kind": "configured_policy_std_v1",
            "init_std": float(config.get("init_std", 1.0)),
            "init_std_vector": (
                None
                if configured_vector is None
                else [float(value) for value in configured_vector]
            ),
        }
    exploration_manifest["fingerprint"] = hashlib.sha256(
        json.dumps(
            exploration_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest["exploration"] = exploration_manifest
    with open_dict(config):
        config.action_manifest = OmegaConf.create(manifest)


def expand_obs_indices_for_history(
    obs_ind: jnp.ndarray,
    env: Any,
    config: DictConfig,
) -> jnp.ndarray:
    """
    Expand observation indices for history stacking.

    When split_goal=True, the observation layout is [state_hist, goal],
    where only non-goal indices are stacked and goal indices stay current.

    Args:
        obs_ind: Original observation indices (before history expansion)
        env: Environment instance (needs obs_container with goal group for split_goal)
        config: Experiment config with len_obs_history and split_goal

    Returns:
        Expanded observation indices for the history-stacked observation
    """
    n_steps = config.len_obs_history
    split_goal = config.get("split_goal", False)

    if split_goal:
        if not hasattr(env, "obs_container"):
            raise ValueError("split_goal=True requires env.obs_container with goal group indices")
        goal_indices = env.obs_container.get_obs_ind_by_group("goal")
        if goal_indices.size == 0:
            raise ValueError("split_goal=True requires goal observations grouped as 'goal'")

        raw_obs_dim = env.info.observation_space.shape[0]
        goal_indices = np.asarray(goal_indices, dtype=int)
        raw_indices = np.arange(raw_obs_dim, dtype=int)
        state_mask = np.ones(raw_obs_dim, dtype=bool)
        state_mask[goal_indices] = False
        state_indices = raw_indices[state_mask]

        state_dim = state_indices.size

        state_index_map = np.full(raw_obs_dim, -1, dtype=int)
        state_index_map[state_indices] = np.arange(state_dim, dtype=int)
        goal_index_map = np.full(raw_obs_dim, -1, dtype=int)
        goal_index_map[goal_indices] = np.arange(goal_indices.size, dtype=int)

        obs_ind_np = np.asarray(obs_ind)
        state_positions = state_index_map[obs_ind_np]
        goal_positions = goal_index_map[obs_ind_np]
        state_positions = state_positions[state_positions >= 0]
        goal_positions = goal_positions[goal_positions >= 0]

        dtype = obs_ind_np.dtype
        if state_positions.size:
            state_expanded = np.concatenate(
                [state_positions + i * state_dim for i in range(n_steps)]
            ).astype(dtype)
        else:
            state_expanded = np.array([], dtype=dtype)

        if goal_positions.size:
            goal_expanded = (goal_positions + n_steps * state_dim).astype(dtype)
        else:
            goal_expanded = np.array([], dtype=dtype)

        return jnp.array(np.concatenate([state_expanded, goal_expanded]), dtype=obs_ind_np.dtype)
    else:
        # Original behavior: full obs stacking
        obs_len = env.info.observation_space.shape[0]
        return jnp.concatenate([obs_ind + i * obs_len for i in range(n_steps)])


def wrap_env(env: Any, config: DictConfig) -> Any:
    """
    Apply standard wrappers for RL training.

    Wrapper order for MJX: VecEnv -> LogWrapper -> AutoResetWrapper
    LogWrapper must see real done flags before AutoResetWrapper clears them.

    Args:
        env: base environment
        config: experiment config with normalize_env, gamma, len_obs_history

    Returns:
        wrapped environment
    """
    env = apply_policy_interface_wrappers(env, config)

    if "len_obs_history" in config and config.len_obs_history > 1:
        split_goal = config.get("split_goal", False)
        env = NStepWrapper(env, config.len_obs_history, split_goal=split_goal)

    if hasattr(env, "mjx_env") and bool(env.mjx_env):
        env = VecEnv(env)
        env = LogWrapper(env)
        env = AutoResetWrapper(env)
    else:
        env = LogWrapper(env)
        env = VecEnv(env)

    if config.normalize_env:
        env = NormalizeVecReward(env, config.gamma)

    return env
