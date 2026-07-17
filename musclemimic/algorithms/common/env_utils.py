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
    model_action_names,
)
from musclemimic.core.wrappers.synergy_action import (
    _resolve_runtime_ctrlrange,
    _resolve_runtime_model_hash,
)
from musclemimic.distill.obs_filter import StudentObservationFilterWrapper
from musclemimic.synergy.multistage_contract import (
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    FULL_354_MODE,
    BodySynergyContractV2,
    build_full_354_action_manifest,
    canonical_action_mode,
)


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

    raw_action_cfg = config.get("action_representation", None)
    action_cfg = raw_action_cfg or {}
    # Historical full-finger environments had no action-representation block.
    # Preserve their native 416-D interface without falsely naming it
    # ``full_354``. Any explicit direct-mode request remains strict below.
    runtime_action_dim = _runtime_policy_action_dim(env)
    implicit_legacy_native = raw_action_cfg is None and runtime_action_dim != 354
    action_mode = None if implicit_legacy_native else canonical_action_mode(action_cfg)
    if action_mode is not None:
        _bind_canonical_action_mode(config, action_mode)
        action_cfg = config.get("action_representation", {}) or {}
    if action_mode == FULL_354_MODE:
        if _find_wrapper(env, SynergyActionWrapper) is not None:
            raise ValueError("full_354 action mode cannot reuse an environment already wrapped for synergy")
        _bind_full_354_runtime_config(config, env, action_cfg)
    elif action_mode in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}:
        synergy_wrapper = _find_wrapper(env, SynergyActionWrapper)
        if synergy_wrapper is None:
            env = SynergyActionWrapper(env, action_cfg)
            synergy_wrapper = env
        if synergy_wrapper.action_interface.mode != action_mode:
            raise ValueError(
                "existing synergy wrapper mode differs from canonical action mode: "
                f"wrapper={synergy_wrapper.action_interface.mode} config={action_mode}"
            )
        _bind_early_synergy_runtime_config(config, synergy_wrapper)
    elif action_mode is not None:  # pragma: no cover - canonical_action_mode is the single mode gate.
        raise AssertionError(f"unhandled canonical action mode {action_mode!r}")

    student_cfg = config.get("student_obs_filter", {})
    if (
        include_student
        and student_cfg.get("enabled", False)
        and _find_wrapper(env, StudentObservationFilterWrapper) is None
    ):
        env = StudentObservationFilterWrapper(env, student_cfg)
    return env


def _runtime_policy_action_dim(env: Any) -> int | None:
    """Return the native policy action dimension when the environment exposes it.

    Lightweight inference and wrapper-order test environments may intentionally
    omit action-space metadata.  A missing implicit action configuration must
    preserve those environments' native interface; only an observed 354-D
    interface is eligible for implicit ``full_354`` contract binding.  Explicit
    action modes remain strict in ``_bind_full_354_runtime_config`` and the
    synergy wrapper constructors.
    """

    for info_attr in ("info", "mdp_info"):
        info = getattr(env, info_attr, None)
        action_space = getattr(info, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape:
            return int(shape[0])
    return None


def _bind_canonical_action_mode(config: DictConfig, mode: str) -> None:
    """Persist legacy mode resolution before the experiment config is hashed."""

    current = config.get("action_representation", {}) or {}
    if isinstance(current, DictConfig):
        payload = OmegaConf.to_container(current, resolve=False)
    else:
        payload = dict(current)
    payload["mode"] = mode
    payload["enabled"] = mode != FULL_354_MODE
    if isinstance(config, DictConfig):
        with open_dict(config):
            config.action_representation = OmegaConf.create(payload)
    else:
        config["action_representation"] = payload


def _bind_full_354_runtime_config(
    config: DictConfig,
    env: Any,
    action_cfg: Any,
) -> None:
    """Bind a direct head to its exact ordered body-action ABI."""

    action_dim = int(env.info.action_space.shape[0])
    if action_dim != 354:
        raise ValueError(
            "full_354 requires exactly 354 ordered body actuators; "
            f"runtime action dimension is {action_dim}"
        )
    names, name_source = _resolve_direct_action_names(env, action_dim)
    expected_dim = action_cfg.get("expected_underlying_action_dim", None)
    if expected_dim is not None and int(expected_dim) != action_dim:
        raise ValueError(
            "full_354 runtime action dimension differs from expected_underlying_action_dim: "
            f"expected={int(expected_dim)} actual={action_dim}"
        )
    runtime_ctrlrange = _resolve_runtime_ctrlrange(env, names)
    if runtime_ctrlrange is None:
        low = np.broadcast_to(np.asarray(env.info.action_space.low, dtype=np.float64), (action_dim,))
        high = np.broadcast_to(np.asarray(env.info.action_space.high, dtype=np.float64), (action_dim,))
        runtime_ctrlrange = np.stack([low, high], axis=1)
        control_range_source = "policy_action_space_bounds"
    else:
        control_range_source = "runtime_mujoco_actuator_ctrlrange"
    runtime_model_hash = _resolve_runtime_model_hash(env)
    manifest = build_full_354_action_manifest(
        actuator_names=names,
        ctrlrange=runtime_ctrlrange,
        runtime_model_hash=runtime_model_hash,
        policy_action_dim=action_dim,
        source_binding={
            "kind": "runtime_direct_body_action_interface",
            "actuator_name_source": name_source,
            "control_range_source": control_range_source,
        },
    )
    _bind_full_354_exploration(
        config,
        manifest,
        action_cfg=action_cfg,
        runtime_ctrlrange=runtime_ctrlrange,
    )
    expected_schema_hash = action_cfg.get("expected_actuator_schema_hash", None)
    if expected_schema_hash not in (None, "") and str(expected_schema_hash) != manifest["actuator_schema_hash"]:
        raise ValueError("full_354 actuator schema hash differs from expected_actuator_schema_hash")
    expected_interface_hash = action_cfg.get("expected_physical_action_interface_hash", None)
    if expected_interface_hash not in (None, "") and (
        str(expected_interface_hash) != manifest["physical_action_interface_hash"]
    ):
        raise ValueError("full_354 physical action interface hash differs from expected value")
    _bind_action_contract(config, manifest, actuator_names=names)


def _bind_full_354_exploration(
    config: DictConfig,
    manifest: dict[str, Any],
    *,
    action_cfg: Any,
    runtime_ctrlrange: np.ndarray,
) -> None:
    """Calibrate direct-policy exploration in the same physical units as W."""

    exploration_cfg = action_cfg.get("exploration", {}) or {}
    calibrate = bool(exploration_cfg.get("calibrate_in_physical_space", False))
    if calibrate:
        from musclemimic.synergy.exploration_scaling import (
            calibrate_exploration_std,
            physical_exploration_rms,
        )

        ranges = np.asarray(runtime_ctrlrange, dtype=np.float64)
        # The direct policy emits normalized [-1, 1] coordinates and unit
        # excitation is (ctrl-low)/(high-low). Composing both affine maps gives
        # d(excitation)/d(action)=0.5 for every actuator, independent of whether
        # its physical ctrlrange is [0,1], [-1,1], or another finite interval.
        physical_jacobian = 0.5 * np.eye(ranges.shape[0], dtype=np.float64)
        target_rms = float(exploration_cfg.get("target_initial_excitation_rms", 0.08))
        std_mode = str(exploration_cfg.get("std_mode", "per_dimension"))
        std_vector = calibrate_exploration_std(
            physical_jacobian,
            target_rms,
            mode=std_mode,
            min_std=float(exploration_cfg.get("min_std", 1e-4)),
            max_std=float(exploration_cfg.get("max_std", 2.0)),
        )
        achieved_rms = physical_exploration_rms(physical_jacobian, std_vector)
        if isinstance(config, DictConfig):
            with open_dict(config):
                config.init_std_vector = [float(value) for value in std_vector]
        else:
            config["init_std_vector"] = [float(value) for value in std_vector]
        exploration_manifest = {
            "kind": "direct_unit_excitation_jacobian_calibration_v1",
            "mode": std_mode,
            "target_initial_excitation_rms": target_rms,
            "achieved_initial_excitation_rms": achieved_rms,
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


def _resolve_direct_action_names(env: Any, action_dim: int) -> tuple[tuple[str, ...], str]:
    for attribute in ("policy_actuator_names", "policy_action_names"):
        value = getattr(env, attribute, None)
        if value is not None:
            names = tuple(str(name) for name in value)
            if len(names) != action_dim:
                raise ValueError(f"{attribute} length differs from the direct policy action dimension")
            return names, attribute
    try:
        return model_action_names(env), "runtime_mujoco_action_order"
    except ValueError:
        # Network-only test environments have no MuJoCo model.  Their manifest
        # remains explicitly marked as coordinate-only and has no model hash;
        # a persisted unbound contract cannot later validate against a model.
        return tuple(f"policy_action_{index:03d}" for index in range(action_dim)), "unbound_policy_coordinates"


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
        from musclemimic.synergy.exploration_scaling import (
            calibrate_exploration_std,
            physical_exploration_rms,
        )

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
            "achieved_initial_excitation_rms": physical_exploration_rms(
                env.decoder_jacobian_at_zero,
                std_vector,
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
    _bind_action_contract(
        config,
        manifest,
        actuator_names=env.body_actuator_names,
    )


def _bind_action_contract(
    config: DictConfig,
    manifest: dict[str, Any],
    *,
    actuator_names: tuple[str, ...],
) -> None:
    """Bind both the runtime manifest and its stage-portable canonical contract."""

    contract = BodySynergyContractV2.from_action_manifest(
        manifest,
        actuator_names=actuator_names,
    )
    if isinstance(config, DictConfig):
        with open_dict(config):
            config.action_manifest = OmegaConf.create(manifest)
            config.body_synergy_contract = OmegaConf.create(contract.to_manifest())
    else:
        config["action_manifest"] = manifest
        config["body_synergy_contract"] = contract.to_manifest()


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
