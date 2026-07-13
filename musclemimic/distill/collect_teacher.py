"""Teacher rollout collection for student policy distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers, wrap_env
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.collection_budget import resolve_collection_budget
from musclemimic.distill.dataset import write_distill_shard, write_split_shard
from musclemimic.distill.losses import distribution_log_std, distribution_mean
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    RolloutIdentityTracker,
    select_transition_traj_no,
    stable_collection_uid,
)
from musclemimic.distill.obs_filter import (
    build_student_obs_indices,
    extract_reference_features,
    filter_student_obs,
    reference_feature_indices,
)
from musclemimic.runner.export_metadata import model_actuator_names


def _tree_get_info(info: dict[str, Any], key: str, shape, dtype):
    value = info.get(key)
    if value is None:
        return np.zeros(shape, dtype=dtype)
    return np.asarray(jax.device_get(value), dtype=dtype)


def build_teacher_rollout_config(experiment_config: Any, *, num_envs: int):
    """Build a full-observation rollout config for lookahead teacher collection."""
    exp_cfg = OmegaConf.create(OmegaConf.to_container(experiment_config, resolve=True))
    exp_cfg.num_envs = int(num_envs)
    if "student_obs_filter" in exp_cfg:
        exp_cfg.student_obs_filter.enabled = False
    return exp_cfg


def collect_teacher_dataset(
    *,
    env: Any,
    agent_conf: Any,
    agent_state: Any,
    output_dir: str | Path,
    num_envs: int,
    num_steps: int | None = None,
    num_transitions: int | None = None,
    shard_size: int = 50_000,
    deterministic_teacher: bool = True,
    seed: int = 0,
    student_obs_filter: dict[str, Any] | None = None,
    save_full_obs: bool = False,
    save_reference_features: bool = False,
    include_reference_phase: bool = False,
    freeze_run_stats: bool = True,
    split: str | None = None,
    metadata: dict[str, Any] | None = None,
    actuator_names: list[str] | None = None,
    motion_identity_map: MotionIdentityMap | None = None,
) -> list[Path]:
    """Collect student_obs -> teacher_action samples from a PPO teacher."""
    budget = resolve_collection_budget(
        num_envs=num_envs,
        num_transitions=num_transitions,
        num_steps=num_steps,
        default_transitions=1_000_000,
    )
    print(
        "[distill_collect] resolved budget "
        f"transitions={budget.requested_transitions} vector_steps={budget.vector_steps} "
        f"num_envs={budget.num_envs} pretrim={budget.planned_transitions_before_trim}"
    )
    if agent_conf.config.experiment.get("len_obs_history", 1) > 1:
        raise NotImplementedError("distill collection currently supports len_obs_history=1 teacher policies")

    exp_cfg = build_teacher_rollout_config(agent_conf.config.experiment, num_envs=num_envs)
    policy_env = apply_policy_interface_wrappers(env, exp_cfg, include_student=False)

    filter_cfg = {
        "enabled": True,
        "keep_motion_phase": True,
        "require_goal_group": True,
        "require_motion_phase": True,
    }
    if student_obs_filter:
        filter_cfg.update(student_obs_filter)
    spec = build_student_obs_indices(policy_env, filter_cfg)
    ref_indices = reference_feature_indices(spec, include_motion_phase=include_reference_phase)
    if save_reference_features and ref_indices.size == 0:
        raise ValueError("save_reference_features=True requires non-phase goal lookahead features")

    resolved_actuator_names = _resolve_actuator_names(policy_env, actuator_names)
    if resolved_actuator_names is None:
        raise ValueError("distillation collector could not resolve ordered policy actuator names")
    actuator_ctrlrange = _resolve_actuator_ctrlrange(policy_env, resolved_actuator_names)
    ctrlrange_schema_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={"actuator_names": resolved_actuator_names, "ctrlrange": actuator_ctrlrange.tolist()},
    )
    state_schema = _student_state_schema(spec, filter_cfg, metadata or {}, env=policy_env)
    body_obs_schema = build_body_obs_schema(
        env=policy_env,
        spec=spec,
        actuator_names=resolved_actuator_names,
        channels=state_schema["channels"],
        provenance={"teacher_ckpt": (metadata or {}).get("teacher_ckpt")},
    )

    teacher_env = wrap_env(policy_env, exp_cfg)

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
        raw_mean_action = distribution_mean(pi)
        teacher_log_std = jnp.broadcast_to(distribution_log_std(pi), raw_mean_action.shape)
        raw_action = raw_mean_action if deterministic_teacher else pi.sample(seed=action_rng)
        # DefaultControl applies this normalized clip before actuator scaling.
        # Persist a reachable target while retaining the raw mean for optional KL.
        action = jnp.clip(raw_action, -1.0, 1.0)
        log_prob = pi.log_prob(raw_action)
        next_obs, reward, absorbing, done, info, next_env_state, _transition_state = teacher_env.step_with_transition(
            cur_env_state,
            action,
        )
        next_ts = ts if freeze_run_stats else ts.replace(run_stats=updates["run_stats"])
        return next_ts, next_obs, next_env_state, cur_rng, raw_mean_action, raw_action, teacher_log_std, action, value, log_prob, reward, absorbing, done, info

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    identity_tracker = None
    collection_uid = None
    if motion_identity_map is not None:
        collection_uid = stable_collection_uid(
            motion_identity_map.motion_paths,
            split=split,
            seed=int(seed),
            collector="teacher_lookahead_rollout",
            run_tag=(metadata or {}).get("teacher_checkpoint_step", (metadata or {}).get("teacher_ckpt")),
        )
        identity_tracker = RolloutIdentityTracker(num_envs=int(num_envs), collection_uid=collection_uid)
    written: list[Path] = []
    buffers: dict[str, list[np.ndarray]] = {}
    total_written = 0
    shard_idx = 0

    def append(name: str, value, keep: int):
        buffers.setdefault(name, []).append(np.asarray(jax.device_get(value))[:keep])

    def flush(force: bool = False):
        nonlocal shard_idx, total_written
        if not buffers:
            return
        current_n = sum(part.shape[0] for part in buffers["student_obs"])
        if current_n < int(shard_size) and not force:
            return
        data = {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}
        teacher_action = np.asarray(data["teacher_action"])
        if not np.isfinite(teacher_action).all() or np.any(np.abs(teacher_action) > 1.0 + 1e-6):
            raise ValueError("persisted teacher_action must be finite normalized applied action in [-1,1]")
        shard_metadata = {
            **(metadata or {}),
            "collector": "teacher_lookahead_rollout",
            "collector_obs_mode": "teacher_full_obs",
            "teacher_action_target": "mean" if deterministic_teacher else "sample",
            "teacher_action_semantics": "clipped_normalized_applied_action",
            "teacher_mu_semantics": "raw_unbounded_gaussian_mean",
            "normalized_action_bounds": [-1.0, 1.0],
            "freeze_run_stats": bool(freeze_run_stats),
            "num_envs": int(num_envs),
            "requested_num_steps": budget.legacy_num_steps,
            "requested_num_transitions": budget.requested_transitions,
            "planned_vector_steps": budget.vector_steps,
            "planned_transitions_before_trim": budget.planned_transitions_before_trim,
            "student_obs_filter": filter_cfg,
            "student_obs_dim": int(spec.student_obs_dim),
            "action_dim": int(data["teacher_action"].shape[-1]),
            "student_state_schema": state_schema,
            "student_state_schema_hash": state_schema["schema_hash"],
            "body_obs_schema": body_obs_schema,
            "body_obs_schema_hash": body_obs_schema["semantic_hash"],
        }
        if motion_identity_map is not None:
            shard_metadata["motion_identity"] = motion_identity_map.to_manifest()
            shard_metadata["collection_uid"] = int(collection_uid)
        if resolved_actuator_names is not None:
            if len(resolved_actuator_names) != int(data["teacher_action"].shape[-1]):
                raise ValueError(
                    "resolved actuator name count does not match collected teacher action: "
                    f"names={len(resolved_actuator_names)} action_dim={data['teacher_action'].shape[-1]}"
                )
            shard_metadata["actuator_names"] = resolved_actuator_names
            shard_metadata["action_schema_hash"] = actuator_schema_hash(resolved_actuator_names)
            shard_metadata["actuator_ctrlrange"] = actuator_ctrlrange.tolist()
            shard_metadata["ctrlrange_schema_hash"] = ctrlrange_schema_hash
        if save_reference_features:
            shard_metadata["reference_features_source"] = "goal_lookahead"
            shard_metadata["reference_features_include_phase"] = bool(include_reference_phase)
            shard_metadata["reference_features_indices"] = ref_indices.tolist()
        if split:
            shard = write_split_shard(output_path, data, split=split, shard_idx=shard_idx, metadata=shard_metadata)
        else:
            shard = write_distill_shard(output_path / f"shard_{shard_idx:06d}.npz", data, metadata=shard_metadata)
        written.append(shard)
        total_written += int(data["student_obs"].shape[0])
        shard_idx += 1
        buffers.clear()

    collected = 0
    for _ in range(budget.vector_steps):
        batch_keep = min(int(num_envs), budget.requested_transitions - collected)
        train_state, next_obs, env_state, rng, raw_mean_action, raw_action, teacher_log_std, action, value, log_prob, reward, absorbing, done, info = policy_step(
            train_state,
            obs,
            env_state,
            rng,
        )
        student_obs = filter_student_obs(obs, spec)
        phase_idx = spec.phase_student_index
        phase = student_obs[..., phase_idx] if phase_idx is not None else jnp.zeros(student_obs.shape[0])

        append("student_obs", student_obs, batch_keep)
        append("teacher_action", action, batch_keep)
        append("teacher_mu", raw_mean_action, batch_keep)
        append(
            "teacher_raw_mean_saturation_fraction",
            jnp.mean(jnp.abs(raw_mean_action) > 1.0, axis=-1),
            batch_keep,
        )
        append(
            "teacher_raw_target_saturation_fraction",
            jnp.mean(jnp.abs(raw_action) > 1.0, axis=-1),
            batch_keep,
        )
        append("teacher_log_std", teacher_log_std, batch_keep)
        append("teacher_value", value, batch_keep)
        append("teacher_log_prob", log_prob, batch_keep)
        append("reward", reward, batch_keep)
        append("done", done, batch_keep)
        append("absorbing", absorbing, batch_keep)
        done_np = np.asarray(jax.device_get(done), dtype=bool)
        traj_no = _tree_get_info(info, "traj_no", (int(num_envs),), np.int32)
        final_traj_no = (
            _tree_get_info(info, "final_traj_no", (int(num_envs),), np.int32)
            if "final_traj_no" in info
            else None
        )
        traj_no = select_transition_traj_no(traj_no, done_np, final_traj_no=final_traj_no)
        append("traj_no", traj_no, batch_keep)
        append(
            "subtraj_step_no",
            _tree_get_info(info, "subtraj_step_no", (int(num_envs),), np.int32),
            batch_keep,
        )
        if motion_identity_map is not None and identity_tracker is not None:
            rollout_uid, rollout_step, env_index = identity_tracker.current()
            append("motion_uid", motion_identity_map.map_traj_no(traj_no), batch_keep)
            append("rollout_uid", rollout_uid, batch_keep)
            append("rollout_step", rollout_step, batch_keep)
            append("env_index", env_index, batch_keep)
        append("phase", phase, batch_keep)
        if save_reference_features:
            append(
                "reference_features",
                extract_reference_features(obs, spec, include_motion_phase=include_reference_phase),
                batch_keep,
            )
        if save_full_obs:
            append("full_obs", obs, batch_keep)

        if identity_tracker is not None:
            identity_tracker.advance(done_np)
        obs = next_obs
        collected += batch_keep
        flush(force=False)

    flush(force=True)
    if total_written != budget.requested_transitions:
        raise RuntimeError(
            f"collector wrote {total_written} samples; expected exactly {budget.requested_transitions}"
        )
    print(f"[distill_collect] wrote {total_written} samples in {len(written)} shards to {output_path}")
    return written


def _resolve_actuator_names(env: Any, explicit_names: list[str] | None) -> list[str] | None:
    if explicit_names is not None:
        return [str(name) for name in explicit_names]
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        policy_names = getattr(current, "policy_actuator_names", None)
        if policy_names is not None:
            return [str(name) for name in policy_names]
        current = getattr(current, "env", None)
    model = getattr(env, "_model", None)
    if model is None:
        model = getattr(env, "model", None)
    if model is None:
        return None
    try:
        return model_actuator_names(model)
    except Exception:
        return None


def _resolve_actuator_ctrlrange(env: Any, actuator_names: list[str]) -> np.ndarray:
    current = env
    seen: set[int] = set()
    model = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        model = getattr(current, "_model", getattr(current, "model", None))
        if model is not None:
            break
        current = getattr(current, "env", None)
    if model is None:
        raise ValueError("distillation collector cannot resolve MuJoCo actuator ctrlrange")
    rows = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
        if actuator_id < 0:
            raise ValueError(f"policy actuator {name!r} is missing from physics model")
        rows.append(np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64))
    ctrlrange = np.asarray(rows, dtype=np.float64)
    if ctrlrange.shape != (len(actuator_names), 2) or not np.all(np.isfinite(ctrlrange)):
        raise ValueError(f"invalid ordered actuator ctrlrange shape/content: {ctrlrange.shape}")
    if np.any(ctrlrange[:, 0] >= ctrlrange[:, 1]):
        raise ValueError("actuator ctrlrange lower bounds must be strictly below upper bounds")
    return ctrlrange


def _student_state_schema(
    spec,
    filter_cfg: dict[str, Any],
    metadata: dict[str, Any],
    *,
    env: Any | None = None,
) -> dict[str, Any]:
    """Persist the exact teacher-to-student observation projection for Stage 3."""
    payload = {
        "schema_version": "student_state_v1",
        "state_dim": int(spec.student_obs_dim),
        "raw_obs_dim": int(spec.raw_obs_dim),
        "goal_indices": np.asarray(spec.goal_indices, dtype=int).tolist(),
        "state_indices": np.asarray(spec.state_indices, dtype=int).tolist(),
        "student_indices": np.asarray(spec.student_indices, dtype=int).tolist(),
        "phase_index": None if spec.phase_index is None else int(spec.phase_index),
        "phase_student_index": None if spec.phase_student_index is None else int(spec.phase_student_index),
        "condition_group_indices": {
            str(name): np.asarray(indices, dtype=int).tolist()
            for name, indices in (spec.condition_group_indices or {}).items()
        },
        "student_obs_filter": dict(filter_cfg),
    }
    payload["channels"] = _named_observation_channels(env, spec.student_indices)
    return payload | {
        "schema_hash": ordered_schema_hash(kind="student_state", payload=payload),
        "provenance": {"teacher_ckpt": metadata.get("teacher_ckpt")},
    }


def _named_observation_channels(env: Any | None, student_indices: np.ndarray) -> list[dict[str, Any]]:
    """Name every flattened student-state channel in exact policy order."""
    by_raw_index: dict[int, dict[str, Any]] = {}
    named_schema = getattr(getattr(env, "observation_filter", None), "target_schema", None)
    if named_schema is None and env is not None:
        try:
            from musclemimic.core.wrappers.finger_isolation import build_named_observation_schema

            named_schema = build_named_observation_schema(env)
        except (AttributeError, TypeError, ValueError):
            named_schema = None
    if named_schema is not None:
        cursor = 0
        for field in named_schema.fields:
            entry_name = str(field.feature_name).split(":", 1)[0]
            for local_index in range(int(field.width)):
                channel = {
                    "name": f"{field.feature_name}[{local_index}]",
                    "entry": entry_name,
                    "entry_type": "NamedObservationField",
                    "entry_offset": int(local_index),
                    "raw_index": int(cursor),
                    "groups": [],
                }
                if field.joint_name is not None:
                    channel["joint_name"] = str(field.joint_name)
                if field.actuator_name is not None:
                    channel["actuator_name"] = str(field.actuator_name)
                by_raw_index[cursor] = channel
                cursor += 1
    container = None if env is None else getattr(env, "obs_container", None)
    entries = [] if container is None or not hasattr(container, "entries") else list(container.entries())
    for entry in entries:
        indices = np.asarray(getattr(entry, "obs_ind", ()), dtype=int)
        entry_name = str(getattr(entry, "name", type(entry).__name__))
        groups = [str(group) for group in (getattr(entry, "group", ()) or ()) if group is not None]
        xml_name = getattr(entry, "xml_name", None)
        xml_names = getattr(entry, "_xml_names", None)
        for local_index, raw_index in enumerate(indices.tolist()):
            channel = {
                "name": f"{entry_name}[{local_index}]",
                "entry": entry_name,
                "entry_type": type(entry).__name__,
                "entry_offset": int(local_index),
                "raw_index": int(raw_index),
                "groups": groups,
            }
            if xml_name is not None:
                channel["xml_name"] = str(xml_name)
            elif xml_names is not None:
                if len(xml_names) == len(indices):
                    channel["xml_name"] = str(xml_names[local_index])
                else:
                    channel["xml_name_count"] = int(len(xml_names))
            existing = by_raw_index.get(int(raw_index), {})
            for key, value in channel.items():
                existing.setdefault(key, value)
            # The concrete observation container carries group/xml provenance
            # that the generic named field may not know (notably touch sensors).
            existing["groups"] = channel["groups"]
            if "xml_name" in channel:
                existing.setdefault("xml_name", channel["xml_name"])
            by_raw_index[int(raw_index)] = existing
    result = []
    for student_index, raw_index in enumerate(np.asarray(student_indices, dtype=int).tolist()):
        channel = dict(
            by_raw_index.get(
                int(raw_index),
                {
                    "name": f"raw_obs[{int(raw_index)}]",
                    "entry": "unknown",
                    "entry_type": "unknown",
                    "entry_offset": 0,
                    "raw_index": int(raw_index),
                    "groups": [],
                },
            )
        )
        channel["student_index"] = int(student_index)
        result.append(channel)
    return result
