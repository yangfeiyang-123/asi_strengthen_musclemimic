"""Teacher rollout collection for student policy distillation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers, wrap_env
from musclemimic.badminton.data.event_lookup import (
    EVENT_LOOKUP_FIELDS,
    EventReferenceLookup,
    select_transition_coordinates,
)
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.collection_budget import resolve_collection_budget
from musclemimic.distill.dataset import write_distill_shard, write_split_shard
from musclemimic.distill.losses import distribution_log_std, distribution_mean
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    RolloutIdentityTracker,
    normalize_motion_path,
    select_transition_traj_no,
    stable_collection_uid,
)
from musclemimic.distill.obs_filter import (
    build_student_obs_indices,
    extract_reference_features,
    filter_student_obs,
    reference_feature_indices,
)
from musclemimic.distill.physical import (
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    physical_ctrl_to_effective_muscle_excitation,
    physical_signal_metadata,
    resolve_muscle_channel_contract,
    validate_muscle_channel_contract,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.physiology import (
    build_emg_observation_projection,
    load_emg_phase_reference_tube,
    load_json_mapping,
    resolve_emg_reference_reward_gate,
)
from musclemimic.runner.export_metadata import model_actuator_names
from musclemimic.synergy.multistage_contract import (
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    FULL_354_MODE,
    canonical_action_mode,
)

EMG_REFERENCE_CAPTURE_SCHEMA_VERSION = "emg_reference_capture_v1"
EMG_OBSERVATION_MAPPING_FILENAME = "emg_observation_mapping.json"

SIMULATOR_PRE_STATE_SCHEMA_VERSION = "mujoco_mjx_pre_transition_state_v1"
SIMULATOR_PRE_STATE_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "plugin_state",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


def _tree_get_info(info: dict[str, Any], key: str, shape, dtype):
    value = info.get(key)
    if value is None:
        return np.zeros(shape, dtype=dtype)
    return np.asarray(jax.device_get(value), dtype=dtype)


def _find_synergy_action_wrapper(env: Any):
    from musclemimic.core.wrappers.synergy_action import SynergyActionWrapper

    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, SynergyActionWrapper):
            return current
        visited.add(id(current))
        child = getattr(current, "env", None)
        if child is current:
            break
        current = child
    return None


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
    save_physical_muscle_state: bool = False,
    save_event_features: bool = False,
    event_reference_manifest: str | Path | None = None,
    emg_reference_cache: str | Path | None = None,
    save_emg_reference: bool = False,
    save_sim_anchor_activation: bool = False,
    physical_racket_site_name: str | None = None,
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
    if emg_reference_cache is not None and not save_emg_reference:
        raise ValueError("emg_reference_cache requires save_emg_reference=True")
    if save_emg_reference and emg_reference_cache is None:
        raise ValueError("save_emg_reference=True requires emg_reference_cache")
    if save_sim_anchor_activation and not save_physical_muscle_state:
        raise ValueError(
            "save_sim_anchor_activation=True requires save_physical_muscle_state=True; "
            "the diagnostic projection consumes the measured 354-D muscle activation"
        )
    if save_sim_anchor_activation and not save_emg_reference:
        raise ValueError(
            "save_sim_anchor_activation=True requires save_emg_reference=True; "
            "the projection is only defined against a bound electrode mapping"
        )

    exp_cfg = build_teacher_rollout_config(agent_conf.config.experiment, num_envs=num_envs)
    action_mode = canonical_action_mode(exp_cfg.get("action_representation", {}) or {})
    policy_env = apply_policy_interface_wrappers(env, exp_cfg, include_student=False)
    synergy_wrapper = _find_synergy_action_wrapper(policy_env)
    if action_mode in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}:
        if synergy_wrapper is None:
            raise ValueError("early-synergy teacher rollout did not construct SynergyActionWrapper")
    elif action_mode == FULL_354_MODE:
        if synergy_wrapper is not None:
            raise ValueError("full_354 teacher unexpectedly contains a synergy wrapper")
    else:  # pragma: no cover - canonical_action_mode owns the finite mode set.
        raise AssertionError(f"unhandled teacher action mode {action_mode!r}")

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
            raise ValueError("supplied actuator_names differ from early-synergy body actuator order")
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
    if event_reference_manifest is not None and not save_event_features:
        raise ValueError("event_reference_manifest requires save_event_features=True")
    event_lookup = (
        EventReferenceLookup.from_manifest(
            event_reference_manifest,
            motion_identity_map=motion_identity_map,
        )
        if event_reference_manifest is not None
        else None
    )
    if event_lookup is not None:
        if not hasattr(env, "dt"):
            raise ValueError("event-aware teacher collection requires an environment control dt")
        event_lookup.validate_control_dt(float(env.dt))
    physical_capture = (
        _build_physical_capture_spec(
            policy_env,
            resolved_actuator_names,
            actuator_ctrlrange,
            racket_site_name=physical_racket_site_name,
        )
        if save_physical_muscle_state
        else None
    )
    emg_reference = (
        _build_emg_reference_capture_spec(
            emg_reference_cache,
            resolved_actuator_names,
            include_sim_anchor_activation=save_sim_anchor_activation,
        )
        if save_emg_reference
        else None
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
        policy_action = jnp.clip(raw_action, -1.0, 1.0)
        if synergy_wrapper is None:
            action = policy_action
            decoded_mean_action = raw_mean_action
            synergy_coefficients = jnp.zeros((*policy_action.shape[:-1], 0), dtype=policy_action.dtype)
            residual_coefficients = jnp.zeros_like(synergy_coefficients)
        else:
            decoded = synergy_wrapper.decode_action(policy_action)
            decoded_mean = synergy_wrapper.decode_action(raw_mean_action)
            action = decoded.body_action
            decoded_mean_action = decoded_mean.body_action
            synergy_coefficients = decoded.synergy_coefficients
            residual_coefficients = decoded.residual_coefficients
        log_prob = pi.log_prob(raw_action)
        next_obs, reward, absorbing, done, info, next_env_state, transition_state = teacher_env.step_with_transition(
            cur_env_state,
            policy_action,
        )
        physical = (
            _capture_physical_transition(transition_state.data, physical_capture)
            if physical_capture is not None
            else {}
        )
        simulator_pre_state = _capture_simulator_pre_state(cur_env_state.data) if physical_capture is not None else {}
        next_ts = ts if freeze_run_stats else ts.replace(run_stats=updates["run_stats"])
        return (
            next_ts,
            next_obs,
            next_env_state,
            cur_rng,
            raw_mean_action,
            raw_action,
            teacher_log_std,
            action,
            decoded_mean_action,
            policy_action,
            synergy_coefficients,
            residual_coefficients,
            value,
            log_prob,
            reward,
            absorbing,
            done,
            info,
            physical,
            simulator_pre_state,
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
        if "racket_rotation_matrix" in data:
            data["racket_quaternion"] = _rotation_matrices_to_wxyz(data.pop("racket_rotation_matrix"))
        teacher_action = np.asarray(data["teacher_action"])
        if not np.isfinite(teacher_action).all() or np.any(np.abs(teacher_action) > 1.0 + 1e-6):
            raise ValueError("persisted teacher_action must be finite normalized applied action in [-1,1]")
        shard_metadata = {
            **(metadata or {}),
            "collector": "teacher_lookahead_rollout",
            "collector_obs_mode": "teacher_full_obs",
            "teacher_action_target": "mean" if deterministic_teacher else "sample",
            "teacher_action_semantics": "clipped_normalized_applied_action",
            "teacher_mu_semantics": (
                "raw_unbounded_gaussian_mean" if synergy_wrapper is None else "decoded_body_action_at_raw_policy_mean"
            ),
            "teacher_log_std_semantics": (
                "raw_gaussian_log_standard_deviation"
                if synergy_wrapper is None
                else "unavailable_for_nonlinear_decoded_body_action"
            ),
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
        if synergy_wrapper is not None:
            contract = synergy_wrapper.action_interface.body_synergy_contract
            frozen = synergy_wrapper.action_interface.frozen_decoder
            shard_metadata.update(
                {
                    "teacher_policy_action_semantics": ("clipped_raw_c_rho_coordinates"),
                    "teacher_policy_action_dim": int(synergy_wrapper.action_interface.policy_action_dim),
                    "teacher_policy_mu_semantics": ("raw_unbounded_gaussian_c_rho_mean"),
                    "teacher_policy_log_std_semantics": ("raw_gaussian_c_rho_log_standard_deviation"),
                    "body_synergy_contract": contract.to_manifest(),
                    "body_synergy_contract_fingerprint": (contract.contract_fingerprint),
                    "body_synergy_portable_core_fingerprint": (contract.portable_decoder_core_fingerprint),
                    "frozen_body_decoder_fingerprint": (frozen.artifact_fingerprint),
                }
            )
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
        if physical_capture is not None:
            _validate_physical_batch(
                data,
                actuator_ctrlrange=actuator_ctrlrange,
                channel_contract=physical_capture["channel_contract"],
            )
            shard_metadata["physical_signal_semantics"] = physical_signal_metadata()
            shard_metadata["physical_capture"] = physical_capture["metadata"]
            shard_metadata["simulator_pre_transition_state"] = {
                "schema_version": SIMULATOR_PRE_STATE_SCHEMA_VERSION,
                "source": "pre_step_rollout_carry.data",
                "backend": "mjx_data_numeric_state",
                "fields": [f"sim_pre_{name}" for name in SIMULATOR_PRE_STATE_FIELDS],
                "timing": "same_s_t_as_student_obs_before_teacher_action",
                "cpu_injection_policy": (
                    "inject integration fields, rebuild trajectory carry from exact coordinates, "
                    "then require live student_obs equality before causal use"
                ),
            }
        if emg_reference is not None:
            shard_metadata["emg_reference_semantics"] = emg_reference["metadata"]
        if save_event_features:
            shard_metadata["event_features_required"] = True
            shard_metadata["event_feature_source"] = (
                "host_exact_event_reference_bank" if event_lookup is not None else "environment_reward_info"
            )
            if event_lookup is not None:
                shard_metadata["event_reference_bank_manifest"] = str(event_lookup.manifest_path)
                shard_metadata["event_reference_bank_fingerprint"] = event_lookup.fingerprint
                shard_metadata["event_reference_control_dt"] = float(env.dt)
                ordered_event_entries = sorted(event_lookup.entries, key=lambda value: value.traj_no)
                shard_metadata["event_reference_bundle_fingerprints"] = [
                    entry.reference_bundle_content_fingerprint for entry in ordered_event_entries
                ]
                shard_metadata["event_reference_bank_motion_uids"] = [
                    int(entry.motion_uid) for entry in ordered_event_entries
                ]
                shard_metadata["event_reference_bank_motion_paths"] = [
                    entry.motion_path for entry in ordered_event_entries
                ]
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
            train_state,
            next_obs,
            env_state,
            rng,
            raw_mean_action,
            raw_action,
            teacher_log_std,
            action,
            decoded_mean_action,
            policy_action,
            synergy_coefficients,
            residual_coefficients,
            value,
            log_prob,
            reward,
            absorbing,
            done,
            info,
            physical,
            simulator_pre_state,
        ) = policy_step(
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
        append("teacher_mu", decoded_mean_action, batch_keep)
        if synergy_wrapper is not None:
            append("teacher_policy_action", policy_action, batch_keep)
            append("teacher_policy_mu", raw_mean_action, batch_keep)
            append("teacher_policy_log_std", teacher_log_std, batch_keep)
            append(
                "teacher_synergy_coefficients",
                synergy_coefficients,
                batch_keep,
            )
            append(
                "teacher_residual_coefficients",
                residual_coefficients,
                batch_keep,
            )
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
        if synergy_wrapper is None:
            append("teacher_log_std", teacher_log_std, batch_keep)
        append("teacher_value", value, batch_keep)
        append("teacher_log_prob", log_prob, batch_keep)
        append("reward", reward, batch_keep)
        append("done", done, batch_keep)
        append("absorbing", absorbing, batch_keep)
        done_np = np.asarray(jax.device_get(done), dtype=bool)
        current_traj_no = _tree_get_info(info, "traj_no", (int(num_envs),), np.int32)
        current_step_no = _tree_get_info(info, "subtraj_step_no", (int(num_envs),), np.int32)
        final_traj_no = (
            _tree_get_info(info, "final_traj_no", (int(num_envs),), np.int32) if "final_traj_no" in info else None
        )
        final_step_no = (
            _tree_get_info(info, "final_subtraj_step_no", (int(num_envs),), np.int32)
            if "final_subtraj_step_no" in info
            else None
        )
        traj_no = select_transition_traj_no(current_traj_no, done_np, final_traj_no=final_traj_no)
        step_no = (
            np.where(done_np, final_step_no, current_step_no).astype(np.int32)
            if final_step_no is not None
            else current_step_no
        )
        append("traj_no", traj_no, batch_keep)
        append("subtraj_step_no", step_no, batch_keep)
        motion_uid = None
        if motion_identity_map is not None and identity_tracker is not None:
            rollout_uid, rollout_step, env_index = identity_tracker.current()
            motion_uid = motion_identity_map.map_traj_no(traj_no)
            append("motion_uid", motion_uid, batch_keep)
            append("rollout_uid", rollout_uid, batch_keep)
            append("rollout_step", rollout_step, batch_keep)
            append("env_index", env_index, batch_keep)
        append("phase", phase, batch_keep)
        if save_event_features:
            required_event_fields = {
                "phase_global": np.float32,
                "phase_id": np.int32,
                "phase_local": np.float32,
                "time_to_impact_s": np.float32,
                "time_from_impact_s": np.float32,
                "impact_flag": np.bool_,
            }
            present = set(required_event_fields) & set(info)
            if present and present != set(required_event_fields):
                raise ValueError(
                    "environment exposes only a partial event contract; missing "
                    f"{sorted(set(required_event_fields) - set(info))}"
                )
            if event_lookup is not None:
                lookup_traj_no, lookup_step_no = select_transition_coordinates(
                    current_traj_no,
                    current_step_no,
                    done_np,
                    final_traj_no=final_traj_no,
                    final_subtraj_step_no=final_step_no,
                )
                if not np.array_equal(lookup_traj_no, traj_no) or not np.array_equal(lookup_step_no, step_no):
                    raise RuntimeError("event lookup transition coordinate invariant violated")
                if motion_uid is None:
                    raise ValueError("event reference bank requires a stable MotionIdentityMap")
                event_values = event_lookup.lookup_batch(
                    traj_no=lookup_traj_no,
                    subtraj_step_no=lookup_step_no,
                    motion_uid=motion_uid,
                )
                if present:
                    for field, dtype in required_event_fields.items():
                        observed = _tree_get_info(info, field, (int(num_envs),), dtype)
                        expected = np.asarray(event_values[field], dtype=dtype)
                        if not np.allclose(observed, expected, rtol=0.0, atol=1e-6):
                            raise ValueError(f"environment event info {field} differs from exact host cache lookup")
                for field in EVENT_LOOKUP_FIELDS:
                    append(field, event_values[field], batch_keep)
                append("event_reference_frame", event_values["event_reference_frame"], batch_keep)
            elif present:
                for field, dtype in required_event_fields.items():
                    append(
                        field,
                        _tree_get_info(info, field, (int(num_envs),), dtype),
                        batch_keep,
                    )
                for field in ("motion_quality_weight", "reference_confidence"):
                    if field in info:
                        append(
                            field,
                            _tree_get_info(info, field, (int(num_envs),), np.float32),
                            batch_keep,
                        )
            else:
                raise ValueError(
                    "event-aware collection requires a complete environment event contract or "
                    "event_reference_manifest; linear frame phase is forbidden"
                )
        for field, value in physical.items():
            append(field, value, batch_keep)
        if emg_reference is not None:
            # Reuse the already-captured normalized phase; recomputing it here
            # would risk drifting from the phase persisted alongside the row.
            phase_np = np.asarray(jax.device_get(phase), dtype=np.float64)
            emg_rows = _capture_emg_reference_rows(
                emg_reference,
                action_indices=_resolve_emg_action_indices(
                    emg_reference,
                    motion_uid=motion_uid,
                    motion_identity_map=motion_identity_map,
                    batch_size=phase_np.shape[0],
                ),
                phase=phase_np,
            )
            for field, value in emg_rows.items():
                append(field, value, batch_keep)
            if emg_reference["include_sim_anchor_activation"]:
                activation = np.asarray(
                    jax.device_get(physical["muscle_activation"]),
                    dtype=np.float64,
                )
                append(
                    "sim_anchor_activation",
                    activation @ emg_reference["projection"].T,
                    batch_keep,
                )
        for field, value in simulator_pre_state.items():
            append(field, value, batch_keep)
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
        raise RuntimeError(f"collector wrote {total_written} samples; expected exactly {budget.requested_transitions}")
    print(f"[distill_collect] wrote {total_written} samples in {len(written)} shards to {output_path}")
    return written


def _resolve_model(env: Any) -> mujoco.MjModel:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        model = getattr(current, "_model", getattr(current, "model", None))
        if model is not None:
            return model
        current = getattr(current, "env", None)
    raise ValueError("physical collector cannot resolve the MuJoCo model")


def _build_physical_capture_spec(
    env: Any,
    actuator_names: list[str],
    actuator_ctrlrange: np.ndarray,
    *,
    racket_site_name: str | None,
) -> dict[str, Any]:
    model = _resolve_model(env)
    channel_contract = resolve_muscle_channel_contract(model, actuator_names)
    actuator_ids = list(channel_contract.actuator_ids)
    act_addresses = list(channel_contract.actuator_actadr)
    activation_valid = [True] * len(actuator_names)

    candidates = [
        racket_site_name,
        "racket_stringbed_center_site",
        "overall_stringbed_center_site",
        "racket_head_site",
    ]
    racket_site_id = -1
    resolved_racket_site = None
    for candidate in candidates:
        if not candidate:
            continue
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(candidate))
        if site_id >= 0:
            racket_site_id = int(site_id)
            resolved_racket_site = str(candidate)
            break
    if racket_site_name is not None and resolved_racket_site != str(racket_site_name):
        raise ValueError(f"explicit physical racket site {racket_site_name!r} is missing from the model")
    racket_body_id = -1 if racket_site_id < 0 else int(model.site_bodyid[racket_site_id])
    racket_root_id = -1 if racket_body_id < 0 else int(model.body_rootid[racket_body_id])
    validate_unit_muscle_ctrlrange(
        actuator_names,
        actuator_ctrlrange,
    )
    return {
        "actuator_ids": jnp.asarray(actuator_ids, dtype=jnp.int32),
        "act_addresses": jnp.asarray(act_addresses, dtype=jnp.int32),
        "activation_valid": jnp.asarray(activation_valid, dtype=bool),
        "channel_contract": channel_contract,
        "racket_site_id": racket_site_id,
        "racket_body_id": racket_body_id,
        "racket_root_id": racket_root_id,
        "metadata": {
            "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
            "actuator_names": list(actuator_names),
            "model_nu": int(model.nu),
            "model_nv": int(model.nv),
            "model_na": int(model.na),
            "activation_valid_mask": activation_valid,
            "muscle_channel_contract": channel_contract.to_metadata(),
            "racket_site_name": resolved_racket_site,
        },
    }


def _build_emg_reference_capture_spec(
    emg_reference_cache: str | Path,
    actuator_names: list[str],
    *,
    include_sim_anchor_activation: bool,
) -> dict[str, Any]:
    """Load the human tube once and bind it to the ordered actuator vector.

    The observation operator ``P`` is rebuilt from the mapping the tube itself
    declares, so a shard can never pair one tube's statistics with another
    mapping's electrode order.  Tube arrays are cached as numpy on the host: the
    per-transition lookup is a plain gather, not a device round trip.
    """

    tube = load_emg_phase_reference_tube(emg_reference_cache)
    # Collection is the first production consumer of the privileged signal.
    # A provisional/diagnostics-only tube may be inspected, but it must never
    # be baked into immutable training shards.
    resolve_emg_reference_reward_gate(tube, enabled=True)
    mapping_path = Path(emg_reference_cache).expanduser()
    if mapping_path.is_file():
        mapping_path = mapping_path.parent
    mapping_path = mapping_path / EMG_OBSERVATION_MAPPING_FILENAME
    if not mapping_path.is_file():
        raise ValueError(f"EMG reference cache is missing its bound observation mapping: {mapping_path}")
    mapping = load_json_mapping(mapping_path)
    declared_mapping_id = str(tube.mapping_binding.get("mapping_id", "")).strip()
    supplied_mapping_id = str(mapping.get("mapping_id", "")).strip()
    if declared_mapping_id and supplied_mapping_id != declared_mapping_id:
        raise ValueError(
            "EMG reference cache mapping identity mismatch: tube declares "
            f"{declared_mapping_id!r} but the bundled mapping is {supplied_mapping_id!r}"
        )
    declared_mapping_sha256 = str(tube.mapping_binding.get("mapping_sha256", "")).strip()
    digest = hashlib.sha256()
    with mapping_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    bundled_mapping_sha256 = digest.hexdigest()
    if bundled_mapping_sha256 != declared_mapping_sha256:
        raise ValueError(
            "EMG reference cache mapping SHA-256 mismatch: tube declares "
            f"{declared_mapping_sha256!r} but the bundled mapping hashes to "
            f"{bundled_mapping_sha256!r}"
        )
    projection, channel_names = build_emg_observation_projection(mapping, actuator_names)
    if tuple(channel_names) != tube.channel_names:
        raise ValueError(
            "EMG observation mapping channel order diverges from the reference tube: "
            f"mapping={list(channel_names)} tube={list(tube.channel_names)}"
        )
    trial_qc_review = dict(tube.provenance["trial_qc_review"])

    return {
        "tube": tube,
        "projection": projection,
        "channel_names": list(channel_names),
        "anchor_mean": np.asarray(tube.anchor_mean, dtype=np.float32),
        "anchor_scale": np.asarray(tube.anchor_scale, dtype=np.float32),
        "anchor_confidence": np.asarray(tube.anchor_valid, dtype=np.float32),
        "synergy_mean": np.asarray(tube.synergy_mean, dtype=np.float32),
        "synergy_scale": np.asarray(tube.synergy_scale, dtype=np.float32),
        "synergy_valid": np.asarray(tube.synergy_valid, dtype=np.float32),
        "include_sim_anchor_activation": bool(include_sim_anchor_activation),
        "metadata": {
            "schema_version": EMG_REFERENCE_CAPTURE_SCHEMA_VERSION,
            "reference_id": tube.reference_id,
            "reference_fingerprint": tube.reference_fingerprint,
            "review_status": tube.review_status,
            "training_enabled": bool(tube.training_enabled),
            "mapping_id": supplied_mapping_id or declared_mapping_id,
            "mapping_sha256": bundled_mapping_sha256,
            "mapping_review_status": str(tube.mapping_binding.get("mapping_review_status", "")),
            "trial_qc_review_schema_version": str(trial_qc_review["schema_version"]),
            "trial_qc_review_sha256": str(trial_qc_review["review_sha256"]),
            "channel_names": list(channel_names),
            "action_ids": list(tube.action_ids),
            "phase_bin_count": int(tube.phase_bin_count),
            "channel_count": int(tube.channel_count),
            "synergy_count": int(tube.synergy_count),
            "sim_anchor_activation_included": bool(include_sim_anchor_activation),
        },
    }


def _resolve_emg_action_indices(
    spec: dict[str, Any],
    *,
    motion_uid: np.ndarray | None,
    motion_identity_map: MotionIdentityMap | None,
    batch_size: int,
) -> np.ndarray:
    """Resolve one tube action row per transition, failing closed on ambiguity."""

    tube = spec["tube"]
    if tube.action_count == 1:
        if motion_identity_map is None:
            raise ValueError(
                f"single-action EMG reference {tube.reference_id!r} still requires MotionIdentityMap action binding"
            )
        expected = _registry_action_slug(tube.action_ids[0])
        observed = {_registry_motion_action_slug(path) for path in motion_identity_map.motion_paths}
        if observed != {expected}:
            raise ValueError(
                f"EMG reference action {expected!r} does not match collected motion action(s) {sorted(observed)}"
            )
        return np.zeros((int(batch_size),), dtype=np.int64)
    if motion_identity_map is not None and motion_uid is not None:
        uid_to_action = spec.setdefault("_uid_to_action", {})
        if not uid_to_action:
            for uid, path in zip(
                motion_identity_map.motion_uids.tolist(),
                motion_identity_map.motion_paths,
                strict=True,
            ):
                uid_to_action[int(uid)] = _match_motion_path_to_action(path, tube.action_ids)
        return np.asarray(
            [uid_to_action[int(uid)] for uid in np.asarray(motion_uid).tolist()],
            dtype=np.int64,
        )
    raise ValueError(
        f"EMG reference {tube.reference_id!r} declares {tube.action_count} actions "
        f"({list(tube.action_ids)}) but the collector has no MotionIdentityMap to resolve "
        "which action each transition belongs to; supply motion_identity_map or a single-action tube"
    )


def _match_motion_path_to_action(motion_path: str, action_ids: tuple[str, ...]) -> int:
    """Map a motion path onto exactly one tube action id."""

    motion_action = _registry_motion_action_slug(motion_path)
    matches = [index for index, action_id in enumerate(action_ids) if _registry_action_slug(action_id) == motion_action]
    if len(matches) == 1:
        return int(matches[0])
    if not matches:
        raise ValueError(
            f"motion {motion_path!r} matches no EMG reference action in {list(action_ids)}; "
            "an unlabelled motion must not silently borrow another action's tube"
        )
    raise ValueError(
        f"motion {motion_path!r} matches several EMG reference actions "
        f"{[action_ids[index] for index in matches]}; action labels must be unambiguous"
    )


def _registry_action_slug(action: str) -> str:
    from musclemimic.badminton.action_registry import resolve

    try:
        return resolve(str(action)).slug
    except ValueError as exc:
        raise ValueError(
            f"EMG reference action {action!r} is not registered; it cannot be bound to a training dataset"
        ) from exc


def _registry_motion_action_slug(motion_path: str) -> str:
    normalized = normalize_motion_path(motion_path)
    dataset_action = normalized.split("/", 1)[0]
    from musclemimic.badminton.action_registry import resolve

    try:
        return resolve(dataset_action).slug
    except ValueError as exc:
        raise ValueError(
            f"motion {motion_path!r} does not start with a registered action id; "
            "its EMG reference cannot be inferred safely"
        ) from exc


def _capture_emg_reference_rows(
    spec: dict[str, Any],
    *,
    action_indices: np.ndarray,
    phase: np.ndarray,
) -> dict[str, np.ndarray]:
    """Gather the tube row for each ``(action, phase_bin)`` pair."""

    bins = int(spec["tube"].phase_bin_count)
    phase_values = np.asarray(phase, dtype=np.float64)
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("EMG reference lookup requires finite normalized phase")
    bin_indices = np.clip(np.floor(phase_values * bins).astype(np.int64), 0, bins - 1)
    rows = np.asarray(action_indices, dtype=np.int64)
    return {
        "emg_anchor_mean": spec["anchor_mean"][rows, bin_indices],
        "emg_anchor_scale": spec["anchor_scale"][rows, bin_indices],
        "emg_channel_confidence": spec["anchor_confidence"][rows, bin_indices],
        "emg_synergy_mean": spec["synergy_mean"][rows, bin_indices],
        "emg_synergy_scale": spec["synergy_scale"][rows, bin_indices],
        "emg_synergy_valid": spec["synergy_valid"][rows, bin_indices],
    }


def _capture_physical_transition(data: Any, spec: dict[str, Any]) -> dict[str, Any]:
    actuator_ids = spec["actuator_ids"]
    ctrl = data.ctrl[..., actuator_ids]
    # This clip is MuJoCo's muscle-control semantics, not a fallback for an
    # invalid signal.  ``teacher_ctrl_physical`` below retains the raw evidence.
    unit_excitation = jnp.clip(ctrl, 0.0, 1.0)
    activation = data.act[..., spec["act_addresses"]]
    velocity = data.actuator_velocity[..., actuator_ids]
    force = data.actuator_force[..., actuator_ids]
    result = {
        "teacher_ctrl_physical": ctrl,
        "muscle_excitation": unit_excitation,
        "muscle_activation": activation,
        "muscle_force": force,
        "muscle_tendon_length": data.actuator_length[..., actuator_ids],
        "muscle_tendon_velocity": velocity,
        "actuator_power": force * velocity,
        "qfrc_actuator": data.qfrc_actuator,
    }
    site_id = int(spec["racket_site_id"])
    if site_id >= 0:
        body_id = int(spec["racket_body_id"])
        root_id = int(spec["racket_root_id"])
        position = data.site_xpos[..., site_id, :]
        site_xmat = data.site_xmat
        if tuple(site_xmat.shape[-2:]) == (3, 3):
            matrix = site_xmat[..., site_id, :, :]
        elif int(site_xmat.shape[-1]) == 9:
            flattened = site_xmat[..., site_id, :]
            matrix = jnp.reshape(flattened, (*flattened.shape[:-1], 3, 3))
        else:
            raise ValueError(
                f"unsupported MuJoCo/MJX site_xmat layout {site_xmat.shape}; expected [...,site,9] or [...,site,3,3]"
            )
        cvel = data.cvel[..., body_id, :]
        offset = position - data.subtree_com[..., root_id, :]
        angular_velocity = cvel[..., :3]
        linear_velocity = cvel[..., 3:] + jnp.cross(angular_velocity, offset)
        result.update(
            {
                "racket_position": position,
                "racket_rotation_matrix": matrix,
                "racket_linear_velocity": linear_velocity,
                "racket_angular_velocity": angular_velocity,
                "stringbed_normal": matrix[..., :, 2],
            }
        )
    return result


def _capture_simulator_pre_state(data: Any) -> dict[str, Any]:
    """Capture the numeric MJX integration state aligned with pre-action ``s_t``.

    These fields are deliberately stored as ordinary NPZ arrays rather than an
    opaque pickle.  A CPU causal adapter may inject them only after rebuilding
    the matching trajectory carry and must still reproduce ``student_obs``;
    the state record alone is never treated as proof of cross-backend identity.
    """

    missing = [name for name in SIMULATOR_PRE_STATE_FIELDS if not hasattr(data, name)]
    if missing:
        raise ValueError(f"MJX rollout state lacks required causal snapshot fields: {missing}")
    return {f"sim_pre_{name}": jnp.asarray(getattr(data, name)) for name in SIMULATOR_PRE_STATE_FIELDS}


def _rotation_matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    value = np.asarray(matrices, dtype=np.float64)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"racket rotation matrices must end in [3,3], got {value.shape}")
    xyzw = Rotation.from_matrix(value.reshape(-1, 3, 3)).as_quat(canonical=True)
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)
    return wxyz.reshape(*value.shape[:-2], 4).astype(np.float32)


def _validate_physical_batch(
    data: dict[str, np.ndarray],
    *,
    actuator_ctrlrange: np.ndarray,
    channel_contract: Any,
) -> None:
    required = {
        "teacher_ctrl_physical",
        "muscle_excitation",
        "muscle_activation",
        "muscle_force",
        "muscle_tendon_length",
        "muscle_tendon_velocity",
        "actuator_power",
        "qfrc_actuator",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"physical transition batch is missing fields: {missing}")
    for field in required:
        if not np.all(np.isfinite(np.asarray(data[field]))):
            raise ValueError(f"physical transition field {field!r} contains non-finite values")
    contract = validate_muscle_channel_contract(channel_contract)
    validate_unit_muscle_ctrlrange(
        contract.actuator_names,
        actuator_ctrlrange,
    )
    expected = physical_ctrl_to_effective_muscle_excitation(
        data["teacher_ctrl_physical"],
        channel_contract=contract,
    )
    np.testing.assert_allclose(
        np.asarray(data["muscle_excitation"], dtype=np.float32),
        expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg="persisted muscle excitation differs from clip(raw data.ctrl,0,1)",
    )
    validate_unit_muscle_activation(data["muscle_activation"])


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
                    channel["xml_name_count"] = len(xml_names)
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
