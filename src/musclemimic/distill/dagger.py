"""DAgger-style student rollout collection with teacher relabeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers, wrap_env
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.collect_teacher import (
    _find_synergy_action_wrapper,
    _resolve_actuator_ctrlrange,
    _resolve_actuator_names,
    _student_state_schema,
)
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
    StudentObsSpec,
    build_student_obs_indices,
    extract_reference_features,
    filter_student_obs,
    reference_feature_indices,
)
from musclemimic.synergy.multistage_contract import (
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    FULL_354_MODE,
    BodySynergyContractV2,
    canonical_action_mode,
)


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
    teacher_action=None,
    rollout_action=None,
    used_teacher_action=None,
    teacher_value=None,
    teacher_log_prob=None,
    teacher_log_std=None,
    teacher_log_prob_teacher_mu=None,
    teacher_log_prob_student_action=None,
    teacher_log_prob_rollout_action=None,
    teacher_policy_mu=None,
    teacher_policy_action=None,
    teacher_policy_log_std=None,
    student_policy_action=None,
    rollout_policy_action=None,
    teacher_synergy_coefficients=None,
    teacher_residual_coefficients=None,
    save_full_obs: bool = False,
    save_reference_features: bool = False,
    include_reference_phase: bool = False,
    motion_uid=None,
    rollout_uid=None,
    rollout_step=None,
    env_index=None,
    traj_no_override=None,
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

    raw_teacher_mu = np.asarray(jax.device_get(teacher_mu), dtype=np.float32)
    applied_teacher_action = np.asarray(
        jax.device_get(
            jnp.clip(jnp.asarray(teacher_mu), -1.0, 1.0)
            if teacher_action is None
            else teacher_action
        ),
        dtype=np.float32,
    )
    data = {
        "student_obs": student_obs,
        "teacher_action": applied_teacher_action,
        "teacher_mu": raw_teacher_mu,
        "teacher_raw_mean_saturation_fraction": np.mean(
            np.abs(raw_teacher_mu) > 1.0, axis=-1, dtype=np.float32
        ),
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
        "traj_no": (
            _info_array(info, "traj_no", n, np.int32)
            if traj_no_override is None
            else np.asarray(jax.device_get(traj_no_override), dtype=np.int32)
        ),
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
    for name, value in (
        ("teacher_policy_mu", teacher_policy_mu),
        ("teacher_policy_action", teacher_policy_action),
        ("teacher_policy_log_std", teacher_policy_log_std),
        ("student_policy_action", student_policy_action),
        ("rollout_policy_action", rollout_policy_action),
        ("teacher_synergy_coefficients", teacher_synergy_coefficients),
        ("teacher_residual_coefficients", teacher_residual_coefficients),
    ):
        if value is not None:
            data[name] = np.asarray(jax.device_get(value), dtype=np.float32)
    if save_full_obs:
        data["full_obs"] = full_obs_np
    if save_reference_features:
        reference_features = extract_reference_features(
            jnp.asarray(full_obs_np),
            spec,
            include_motion_phase=include_reference_phase,
        )
        data["reference_features"] = np.asarray(jax.device_get(reference_features), dtype=np.float32)
    for name, value, dtype in (
        ("motion_uid", motion_uid, np.int64),
        ("rollout_uid", rollout_uid, np.int64),
        ("rollout_step", rollout_step, np.int32),
        ("env_index", env_index, np.int32),
    ):
        if value is not None:
            array = np.asarray(jax.device_get(value), dtype=dtype)
            if array.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
            data[name] = array
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
    num_steps: int | None = None,
    num_transitions: int | None = None,
    shard_size: int = 50_000,
    seed: int = 0,
    student_obs_filter: dict[str, Any] | None = None,
    mix_teacher_action_prob: float = 0.0,
    append: bool = False,
    save_full_obs: bool = False,
    save_reference_features: bool = False,
    include_reference_phase: bool = False,
    freeze_run_stats: bool = True,
    split: str | None = None,
    metadata: dict[str, Any] | None = None,
    actuator_names: list[str] | None = None,
    motion_identity_map: MotionIdentityMap | None = None,
) -> list[Path]:
    """Roll out student actions while labeling visited states with teacher mean.

    First version supports non-history student policies. This matches the
    provided student phase configs and keeps relabeling unambiguous.
    """
    budget = resolve_collection_budget(
        num_envs=num_envs,
        num_transitions=num_transitions,
        num_steps=num_steps,
        default_transitions=500_000,
    )
    print(
        "[distill_dagger] resolved budget "
        f"transitions={budget.requested_transitions} vector_steps={budget.vector_steps} "
        f"num_envs={budget.num_envs} pretrim={budget.planned_transitions_before_trim}"
    )
    teacher_exp = teacher_agent_conf.config.experiment
    student_exp = student_agent_conf.config.experiment
    if teacher_exp.get("len_obs_history", 1) > 1:
        raise NotImplementedError("DAgger collection currently supports len_obs_history=1 teacher policies")
    if student_exp.get("len_obs_history", 1) > 1:
        raise NotImplementedError("DAgger collection currently supports len_obs_history=1 student policies")
    teacher_action_mode = canonical_action_mode(
        teacher_exp.get("action_representation", {}) or {}
    )
    student_action_mode = canonical_action_mode(
        student_exp.get("action_representation", {}) or {}
    )
    if teacher_action_mode != student_action_mode:
        raise ValueError(
            "DAgger teacher/student action modes differ; mixing actions requires "
            "one shared policy-coordinate ABI"
        )
    synergy_mode = teacher_action_mode in {
        FIXED_SYNERGY_MODE,
        FIXED_SYNERGY_RESIDUAL_MODE,
    }
    if teacher_action_mode != FULL_354_MODE and not synergy_mode:
        raise AssertionError(
            f"unhandled DAgger action mode {teacher_action_mode!r}"
        )

    rollout_cfg = OmegaConf.create(OmegaConf.to_container(teacher_exp, resolve=True))
    rollout_cfg.num_envs = int(num_envs)
    if "student_obs_filter" in rollout_cfg:
        rollout_cfg.student_obs_filter.enabled = False
    policy_env = apply_policy_interface_wrappers(env, rollout_cfg, include_student=False)
    synergy_wrapper = _find_synergy_action_wrapper(policy_env)
    if synergy_mode:
        if synergy_wrapper is None:
            raise ValueError("early-synergy DAgger did not construct SynergyActionWrapper")
        teacher_contract = synergy_wrapper.action_interface.body_synergy_contract
        for role, experiment in (
            ("teacher", teacher_exp),
            ("student", student_exp),
        ):
            payload = experiment.get("body_synergy_contract", None)
            if payload is None:
                raise ValueError(
                    f"early-synergy DAgger {role} lacks BodySynergyContractV2"
                )
            if OmegaConf.is_config(payload):
                payload = OmegaConf.to_container(payload, resolve=True)
            contract = BodySynergyContractV2.from_manifest(payload)
            teacher_contract.assert_exact_runtime_compatible(contract)
    elif synergy_wrapper is not None:
        raise ValueError("full_354 DAgger unexpectedly contains a synergy wrapper")

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
    if synergy_wrapper is None:
        resolved_actuator_names = _resolve_actuator_names(policy_env, actuator_names)
    else:
        resolved_actuator_names = list(synergy_wrapper.body_actuator_names)
        if actuator_names is not None and list(actuator_names) != resolved_actuator_names:
            raise ValueError(
                "supplied actuator names differ from early-synergy body action order"
            )
    if resolved_actuator_names is None:
        raise ValueError("DAgger collector could not resolve ordered policy actuator names")
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
    rollout_env = wrap_env(policy_env, rollout_cfg)

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
        raw_student_action = student_pi.sample(seed=student_rng)
        student_action = jnp.clip(raw_student_action, -1.0, 1.0)

        (teacher_pi, teacher_value), t_updates = teacher_agent_conf.network.apply(
            {"params": t_ts.params, "run_stats": t_ts.run_stats},
            cur_full_obs,
            mutable=["run_stats"],
        )
        teacher_mu = distribution_mean(teacher_pi)
        teacher_policy_action = jnp.clip(teacher_mu, -1.0, 1.0)
        teacher_policy_log_std = jnp.broadcast_to(
            distribution_log_std(teacher_pi), teacher_mu.shape
        )
        teacher_log_prob_teacher_mu = teacher_pi.log_prob(teacher_mu)
        teacher_log_prob_student_action = teacher_pi.log_prob(raw_student_action)
        use_teacher = jax.random.uniform(mix_rng, shape=(int(num_envs),)) < float(mix_teacher_action_prob)
        rollout_policy_action = jnp.where(
            use_teacher[:, None], teacher_policy_action, student_action
        )
        teacher_log_prob_rollout_action = teacher_pi.log_prob(
            rollout_policy_action
        )
        if synergy_wrapper is None:
            teacher_action = teacher_policy_action
            teacher_mu_body = teacher_mu
            student_action_body = student_action
            rollout_action = rollout_policy_action
            teacher_log_std = teacher_policy_log_std
            synergy_coefficients = jnp.zeros(
                (*teacher_action.shape[:-1], 0), dtype=teacher_action.dtype
            )
            residual_coefficients = jnp.zeros_like(synergy_coefficients)
        else:
            teacher_decoded = synergy_wrapper.decode_action(
                teacher_policy_action
            )
            teacher_mean_decoded = synergy_wrapper.decode_action(teacher_mu)
            student_decoded = synergy_wrapper.decode_action(student_action)
            rollout_decoded = synergy_wrapper.decode_action(
                rollout_policy_action
            )
            teacher_action = teacher_decoded.body_action
            teacher_mu_body = teacher_mean_decoded.body_action
            student_action_body = student_decoded.body_action
            rollout_action = rollout_decoded.body_action
            teacher_log_std = jnp.zeros_like(teacher_action)
            synergy_coefficients = teacher_decoded.synergy_coefficients
            residual_coefficients = teacher_decoded.residual_coefficients
        next_obs, reward, absorbing, done, info, next_env_state, _transition_state = rollout_env.step_with_transition(
            cur_env_state,
            rollout_policy_action,
        )
        t_ts = t_ts if freeze_run_stats else t_ts.replace(run_stats=t_updates["run_stats"])
        s_ts = s_ts if freeze_run_stats else s_ts.replace(run_stats=s_updates["run_stats"])
        return (
            t_ts,
            s_ts,
            next_obs,
            next_env_state,
            cur_rng,
            teacher_mu_body,
            teacher_action,
            teacher_value,
            teacher_log_std,
            teacher_log_prob_teacher_mu,
            teacher_log_prob_student_action,
            teacher_log_prob_rollout_action,
            student_action_body,
            rollout_action,
            teacher_mu,
            teacher_policy_action,
            teacher_policy_log_std,
            student_action,
            rollout_policy_action,
            synergy_coefficients,
            residual_coefficients,
            use_teacher,
            reward,
            absorbing,
            done,
            info,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    identity_tracker = None
    collection_uid = None
    if motion_identity_map is not None:
        collection_uid = stable_collection_uid(
            motion_identity_map.motion_paths,
            split=split,
            seed=int(seed),
            collector="dagger_student_rollout_teacher_relabel",
            run_tag=(
                (metadata or {}).get("dagger_iteration")
                if (metadata or {}).get("dagger_iteration") is not None
                else (metadata or {}).get("student_checkpoint_step", (metadata or {}).get("student_ckpt"))
            ),
        )
        identity_tracker = RolloutIdentityTracker(num_envs=int(num_envs), collection_uid=collection_uid)
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
        teacher_action_data = np.asarray(data["teacher_action"])
        rollout_action_data = np.asarray(data["rollout_action"])
        if (
            not np.isfinite(teacher_action_data).all()
            or not np.isfinite(rollout_action_data).all()
            or np.any(np.abs(teacher_action_data) > 1.0 + 1e-6)
            or np.any(np.abs(rollout_action_data) > 1.0 + 1e-6)
        ):
            raise ValueError("DAgger applied actions must be finite and normalized to [-1,1]")
        shard_metadata = {
            **(metadata or {}),
            "collector": "dagger_student_rollout_teacher_relabel",
            "teacher_action_semantics": "clipped_normalized_applied_mean",
            "teacher_mu_semantics": (
                "raw_unbounded_gaussian_mean"
                if synergy_wrapper is None
                else "decoded_body_action_at_raw_policy_mean"
            ),
            "teacher_log_std_semantics": (
                "raw_gaussian_log_standard_deviation"
                if synergy_wrapper is None
                else "unavailable_for_nonlinear_decoded_body_action"
            ),
            "rollout_action_semantics": "clipped_normalized_applied_action",
            "normalized_action_bounds": [-1.0, 1.0],
            "num_envs": int(num_envs),
            "requested_num_steps": budget.legacy_num_steps,
            "requested_num_transitions": budget.requested_transitions,
            "planned_vector_steps": budget.vector_steps,
            "planned_transitions_before_trim": budget.planned_transitions_before_trim,
            "student_obs_filter": filter_cfg,
            "student_obs_dim": int(spec.student_obs_dim),
            "action_dim": int(data["teacher_action"].shape[-1]),
            "mix_teacher_action_prob": float(mix_teacher_action_prob),
            "freeze_run_stats": bool(freeze_run_stats),
            "student_state_schema": state_schema,
            "student_state_schema_hash": state_schema["schema_hash"],
            "body_obs_schema": body_obs_schema,
            "body_obs_schema_hash": body_obs_schema["semantic_hash"],
        }
        if synergy_wrapper is not None:
            contract = synergy_wrapper.action_interface.body_synergy_contract
            frozen = synergy_wrapper.action_interface.frozen_decoder
            shard_metadata.update(
                {
                    "teacher_policy_action_semantics": (
                        "clipped_raw_c_rho_coordinates"
                    ),
                    "teacher_policy_action_dim": int(
                        synergy_wrapper.action_interface.policy_action_dim
                    ),
                    "teacher_policy_mu_semantics": (
                        "raw_unbounded_gaussian_c_rho_mean"
                    ),
                    "teacher_policy_log_std_semantics": (
                        "raw_gaussian_c_rho_log_standard_deviation"
                    ),
                    "student_policy_action_semantics": (
                        "clipped_sampled_raw_c_rho_coordinates"
                    ),
                    "rollout_policy_action_semantics": (
                        "clipped_mixed_raw_c_rho_coordinates"
                    ),
                    "body_synergy_contract": contract.to_manifest(),
                    "body_synergy_contract_fingerprint": (
                        contract.contract_fingerprint
                    ),
                    "body_synergy_portable_core_fingerprint": (
                        contract.portable_decoder_core_fingerprint
                    ),
                    "frozen_body_decoder_fingerprint": (
                        frozen.artifact_fingerprint
                    ),
                }
            )
        if resolved_actuator_names is not None:
            if len(resolved_actuator_names) != int(data["teacher_action"].shape[-1]):
                raise ValueError(
                    "resolved actuator name count does not match DAgger teacher action: "
                    f"names={len(resolved_actuator_names)} action_dim={data['teacher_action'].shape[-1]}"
                )
            shard_metadata["actuator_names"] = resolved_actuator_names
            shard_metadata["action_schema_hash"] = actuator_schema_hash(resolved_actuator_names)
            shard_metadata["actuator_ctrlrange"] = actuator_ctrlrange.tolist()
            shard_metadata["ctrlrange_schema_hash"] = ctrlrange_schema_hash
        if motion_identity_map is not None:
            shard_metadata["motion_identity"] = motion_identity_map.to_manifest()
            shard_metadata["collection_uid"] = int(collection_uid)
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
        (
            teacher_ts,
            student_ts,
            next_full_obs,
            env_state,
            rng,
            teacher_mu,
            teacher_action,
            teacher_value,
            teacher_log_std,
            teacher_log_prob_teacher_mu,
            teacher_log_prob_student_action,
            teacher_log_prob_rollout_action,
            student_action,
            rollout_action,
            teacher_policy_mu,
            teacher_policy_action,
            teacher_policy_log_std,
            student_policy_action,
            rollout_policy_action,
            synergy_coefficients,
            residual_coefficients,
            used_teacher_action,
            reward,
            absorbing,
            done,
            info,
        ) = policy_step(teacher_ts, student_ts, full_obs, env_state, rng)
        done_np = np.asarray(jax.device_get(done), dtype=bool)
        traj_no = _info_array(info, "traj_no", int(num_envs), np.int32)
        final_traj_no = (
            _info_array(info, "final_traj_no", int(num_envs), np.int32)
            if "final_traj_no" in info
            else None
        )
        traj_no = select_transition_traj_no(traj_no, done_np, final_traj_no=final_traj_no)
        identity_fields: dict[str, np.ndarray] = {}
        if motion_identity_map is not None and identity_tracker is not None:
            rollout_uid, rollout_step, env_index = identity_tracker.current()
            identity_fields = {
                "motion_uid": motion_identity_map.map_traj_no(traj_no),
                "rollout_uid": rollout_uid,
                "rollout_step": rollout_step,
                "env_index": env_index,
            }
        batch_data = build_dagger_shard_data(
                full_obs=full_obs,
                teacher_mu=teacher_mu,
                teacher_action=teacher_action,
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
                teacher_log_std=(
                    teacher_log_std if synergy_wrapper is None else None
                ),
                teacher_log_prob_teacher_mu=teacher_log_prob_teacher_mu,
                teacher_log_prob_student_action=teacher_log_prob_student_action,
                teacher_log_prob_rollout_action=teacher_log_prob_rollout_action,
                teacher_policy_mu=(
                    teacher_policy_mu if synergy_wrapper is not None else None
                ),
                teacher_policy_action=(
                    teacher_policy_action if synergy_wrapper is not None else None
                ),
                teacher_policy_log_std=(
                    teacher_policy_log_std if synergy_wrapper is not None else None
                ),
                student_policy_action=(
                    student_policy_action if synergy_wrapper is not None else None
                ),
                rollout_policy_action=(
                    rollout_policy_action if synergy_wrapper is not None else None
                ),
                teacher_synergy_coefficients=(
                    synergy_coefficients if synergy_wrapper is not None else None
                ),
                teacher_residual_coefficients=(
                    residual_coefficients if synergy_wrapper is not None else None
                ),
                save_full_obs=save_full_obs,
                save_reference_features=save_reference_features,
                include_reference_phase=include_reference_phase,
                traj_no_override=traj_no,
                **identity_fields,
            )
        append_buffer({name: value[:batch_keep] for name, value in batch_data.items()})
        if identity_tracker is not None:
            identity_tracker.advance(done_np)
        full_obs = next_full_obs
        collected += batch_keep
        flush(force=False)

    flush(force=True)
    if total_written != budget.requested_transitions:
        raise RuntimeError(
            f"DAgger collector wrote {total_written} samples; expected exactly {budget.requested_transitions}"
        )
    print(f"[distill_dagger] wrote {total_written} relabeled samples in {len(written)} shards to {output_path}")
    return written
