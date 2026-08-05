"""Batched JAX/MJX incoming-shuttle hit environment for GPU-parallel PPO.

Batch-native: every function operates on arrays with a leading world/env
dimension. On the Warp backend (default, fast) contact pools are shared
across worlds and ``mjx.step`` is called directly — Warp kernels vmap over
worlds implicitly. On the classic JAX backend the Data pytree is fully
batched and ``mjx.step`` is vmapped explicitly (slow; kept for debugging).

Semantics mirror ``IncomingShuttleHitEnv`` (CPU):

- reset: ``overall_ready`` keyframe + a feeder launch state for the shuttle
- step: action -> muscle ctrl scaling -> ``control_substeps`` badminton
  physics substeps (aero + stringbed + event rebound) -> state machine
  INCOMING/HIT/FLIGHT/DONE -> reward terms -> auto-reset of done envs
- observation layout is identical to the CPU env; reset observations are
  precomputed with the CPU pipeline per feed, so a freshly reset env produces
  bit-equivalent features on both backends.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from musclemimic.distill.physical import resolve_muscle_channel_contract

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics import BadmintonPhysicsConfig
from environment.overall_environment.src.badminton_physics_mjx import (
    event_rebound_velocity,
    make_batched_substep_fn,
    make_ids,
    make_params,
)
from environment.overall_environment.src.incoming_shuttle_hit_env import (
    BODY_FALL_ROOT_HEIGHT_M,
    GROUND_REST_HEIGHT_M,
    IMPACT_RECOVERY_PROFILE,
    LEGACY_PROFILE,
    V2_OBSERVATION_SIZE,
    IncomingShuttleHitEnv,
    _validate_contact_guidance_contract,
    _validate_reward_weights,
    incoming_hit_policy_abi_hash,
)
from environment.overall_environment.src.shuttle_feeder import FeedSample
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    V2_REWARD_TERM_ORDER,
)
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    curriculum_complete as task_curriculum_complete,
)
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    runtime_values as task_runtime_values,
)
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    runtime_values_for_stage as task_runtime_values_for_stage,
)
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    stage_by_name as task_stage_by_name,
)

STATE_INCOMING = 0
STATE_HIT = 1
STATE_FLIGHT = 2
STATE_RECOVERY = 3

_COURT_HALF_LENGTH = 6.70
_NET_FRONT_DEPTH = 2.0
_BACK_DEPTH = 5.35


class EnvState(NamedTuple):
    data: Any  # mjx.Data (semi-batched on warp, fully batched on jax impl)
    obs: jnp.ndarray  # (N, obs_size)
    cooldown: jnp.ndarray  # (N,) int32
    step_index: jnp.ndarray  # (N,) int32
    phase_code: jnp.ndarray  # (N,) int32
    hit_rewarded: jnp.ndarray  # (N,) bool
    crossed_rewarded: jnp.ndarray  # (N,) bool
    invalid_net_crossed: jnp.ndarray  # (N,) bool
    hit_closing_speed: jnp.ndarray  # (N,) float32
    best_shuttle_proximity_potential: jnp.ndarray  # (N,) float32
    best_timed_intercept_potential: jnp.ndarray  # (N,) float32
    best_racket_direction_potential: jnp.ndarray  # (N,) float32
    closest_racket_distance_m: jnp.ndarray  # (N,) float32
    closest_racket_direction_score: jnp.ndarray  # (N,) float32
    landing_recorded: jnp.ndarray  # (N,) bool; v2 only
    landing_xy: jnp.ndarray  # (N, 2); v2 only
    apex_height: jnp.ndarray  # (N,) float32; v2 only
    recovery_step: jnp.ndarray  # (N,) int32; v2 only
    recovery_active: jnp.ndarray  # (N,) bool; starts at impact
    recovery_complete: jnp.ndarray  # (N,) bool; independent of flight
    flight_resolved: jnp.ndarray  # (N,) bool; landing or static-target mode
    feed_idx: jnp.ndarray  # (N,) int32
    lab_state: jnp.ndarray  # (N, lab_state_size), empty outside LAB mode
    lambda_lab: jnp.ndarray  # scalar curriculum value
    active_feed_count: jnp.ndarray  # scalar easy-to-hard feed count
    residual_authority_progress: jnp.ndarray  # scalar 0->1 authority ramp
    v2_stage_index: jnp.ndarray  # scalar Stage-3 v2 curriculum index
    v2_environment_mode: jnp.ndarray  # scalar; see stage3_task_curriculum_v2
    v2_reward_mask: jnp.ndarray  # (11,) ordered by V2_REWARD_TERM_ORDER
    key: jnp.ndarray  # (2,) PRNGKey shared


def _warp_unbatched_fields() -> frozenset[str]:
    from mujoco.mjx.warp import types as mjxw_types

    return frozenset(k for k, v in mjxw_types._BATCH_DIM["Data"].items() if not v)


def _attr_name(path) -> str:
    return "__".join(p.name for p in path if hasattr(p, "name")).removeprefix("_impl__")


@dataclass
class IncomingHitMjxEnv:
    """Batch-native env factory: reset(key) and step(state, actions)."""

    xml: str | Path
    feed_bank: list[FeedSample]
    physics_config: BadmintonPhysicsConfig | None = None
    control_substeps: int = 10
    max_episode_steps: int = 300
    reward_weights: dict[str, float] = field(default_factory=dict)
    player_half_sign: int = -1
    singles: bool = True
    impl: str = "warp"
    nconmax_per_env: int = 128
    # Stage-3 residual mode: frozen distilled base policy drives the body,
    # the PPO action becomes a residual on top of it
    base_policy_artifact: str | Path | None = None
    residual_scale: float = 0.3
    residual_scale_overrides: dict[str, float] | None = None
    residual_scale_schedule: dict[str, Any] | None = None
    swing_duration_s: float = 1.2
    contact_phase: float = 0.76
    swing_phase_advance_s: float = 0.0
    return_net_x_m: float = 0.0
    return_net_height_m: float = 1.55
    min_return_net_clearance_m: float | None = None
    desired_return_up_component: float = 0.40
    ballistic_return_score_softness_m: float = 0.35
    shuttle_proximity_softness_m: float = 0.35
    timed_intercept_softness_m: float = 0.30
    direction_distance_softness_m: float = 0.45
    contact_guidance_reward_mode: str = "dense_per_step"
    contact_guidance_discount: float = 1.0
    racket_velocity_direction_fraction: float = 0.30
    direction_reward_mode: str = "positive_projection"
    clearance_reward_mode: str = "positive_score"
    hit_event_mode: str = "any_stringbed_contact"
    racket_guidance_mode: str = "component_projection"
    inverse_target_speed_m_s: float = 12.0
    inverse_velocity_softness_m_s: float = 6.0
    base_skill: str | None = None
    lab_controller: Any | None = None
    lab_state_builder: Any | None = None
    curriculum: Any | None = None
    curriculum_feed_order: str = "difficulty_sorted"
    filter_finger_observation: bool | None = None
    feed_bank_manifest: dict[str, Any] | None = None
    task_profile: str = LEGACY_PROFILE
    impact_target_bank: Any | None = None
    recovery_horizon_steps: int = 60
    task_curriculum_max_stage: str | None = None
    racket_attachment_contract: str | Path | None = None

    def __post_init__(self) -> None:
        self.task_profile = str(self.task_profile)
        self.weights = _validate_reward_weights(dict(self.reward_weights), task_profile=self.task_profile)
        self.swing_phase_advance_s = float(self.swing_phase_advance_s)
        if not math.isfinite(self.swing_phase_advance_s) or self.swing_phase_advance_s < 0.0:
            raise ValueError("swing_phase_advance_s must be finite and non-negative")
        self.return_net_x_m = float(self.return_net_x_m)
        self.return_net_height_m = float(self.return_net_height_m)
        self.min_return_net_clearance_m = (
            None if self.min_return_net_clearance_m is None else float(self.min_return_net_clearance_m)
        )
        self.desired_return_up_component = float(self.desired_return_up_component)
        self.ballistic_return_score_softness_m = float(self.ballistic_return_score_softness_m)
        self.shuttle_proximity_softness_m = float(self.shuttle_proximity_softness_m)
        self.timed_intercept_softness_m = float(self.timed_intercept_softness_m)
        self.direction_distance_softness_m = float(self.direction_distance_softness_m)
        self.contact_guidance_reward_mode = _validate_contact_guidance_contract(
            self.contact_guidance_reward_mode,
            self.weights,
        )
        self.contact_guidance_discount = float(self.contact_guidance_discount)
        self.racket_velocity_direction_fraction = float(self.racket_velocity_direction_fraction)
        self.direction_reward_mode = str(self.direction_reward_mode)
        self.clearance_reward_mode = str(self.clearance_reward_mode)
        self.hit_event_mode = str(self.hit_event_mode)
        self.racket_guidance_mode = str(self.racket_guidance_mode)
        self.inverse_target_speed_m_s = float(self.inverse_target_speed_m_s)
        self.inverse_velocity_softness_m_s = float(self.inverse_velocity_softness_m_s)
        if not all(
            math.isfinite(value)
            for value in (
                self.return_net_x_m,
                self.return_net_height_m,
                self.desired_return_up_component,
                self.ballistic_return_score_softness_m,
                self.shuttle_proximity_softness_m,
                self.timed_intercept_softness_m,
                self.direction_distance_softness_m,
                self.contact_guidance_discount,
                self.racket_velocity_direction_fraction,
                self.inverse_target_speed_m_s,
                self.inverse_velocity_softness_m_s,
            )
        ):
            raise ValueError("return constraints must be finite")
        if self.return_net_height_m <= 0.0:
            raise ValueError("return_net_height_m must be positive")
        if self.min_return_net_clearance_m is not None and (
            not math.isfinite(self.min_return_net_clearance_m) or self.min_return_net_clearance_m < 0.0
        ):
            raise ValueError("min_return_net_clearance_m must be finite and non-negative")
        if self.desired_return_up_component <= 0.0:
            raise ValueError("desired_return_up_component must be positive")
        if self.ballistic_return_score_softness_m <= 0.0:
            raise ValueError("ballistic_return_score_softness_m must be positive")
        if self.shuttle_proximity_softness_m <= 0.0:
            raise ValueError("shuttle_proximity_softness_m must be positive")
        if self.timed_intercept_softness_m <= 0.0:
            raise ValueError("timed_intercept_softness_m must be positive")
        if self.direction_distance_softness_m <= 0.0:
            raise ValueError("direction_distance_softness_m must be positive")
        if not 0.0 < self.contact_guidance_discount <= 1.0:
            raise ValueError("contact_guidance_discount must lie in (0, 1]")
        if not 0.0 <= self.racket_velocity_direction_fraction <= 1.0:
            raise ValueError("racket_velocity_direction_fraction must lie in [0, 1]")
        if self.inverse_target_speed_m_s <= 0.0:
            raise ValueError("inverse_target_speed_m_s must be positive")
        if self.inverse_velocity_softness_m_s <= 0.0:
            raise ValueError("inverse_velocity_softness_m_s must be positive")
        if self.direction_reward_mode not in {
            "positive_projection",
            "signed_projection",
        }:
            raise ValueError("direction_reward_mode must be positive_projection or signed_projection")
        if self.clearance_reward_mode not in {
            "positive_score",
            "signed_centered",
        }:
            raise ValueError("clearance_reward_mode must be positive_score or signed_centered")
        if self.hit_event_mode not in {
            "any_stringbed_contact",
            "event_rebound",
        }:
            raise ValueError("hit_event_mode must be any_stringbed_contact or event_rebound")
        if self.racket_guidance_mode not in {
            "component_projection",
            "counterfactual_rebound",
            "counterfactual_clearance_priority",
            "inverse_impact_target",
            "inverse_impact_decomposed",
        }:
            raise ValueError(
                "racket_guidance_mode must be component_projection or "
                "counterfactual_rebound or counterfactual_clearance_priority "
                "or inverse_impact_target or inverse_impact_decomposed"
            )
        if self.contact_guidance_reward_mode in {
            "event_direction",
            "potential_event_direction",
            "closest_approach_event_direction",
        }:
            if self.hit_event_mode != "event_rebound":
                raise ValueError(
                    "event-based direction contact guidance requires "
                    "hit_event_mode=event_rebound"
                )
            if self.racket_guidance_mode != "inverse_impact_decomposed":
                raise ValueError(
                    "event-based direction contact guidance requires "
                    "racket_guidance_mode=inverse_impact_decomposed"
                )
        if self.task_curriculum_max_stage is not None:
            if self.task_profile != IMPACT_RECOVERY_PROFILE:
                raise ValueError("task_curriculum_max_stage is only valid for impact_recovery_v2")
            task_stage_by_name(self.task_curriculum_max_stage)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.racket_attachment_contract_path = (
            None
            if self.racket_attachment_contract is None
            else Path(self.racket_attachment_contract).expanduser().resolve()
        )
        if self.lab_controller is not None and self.base_policy_artifact is not None:
            raise ValueError("LAB and legacy full-action residual modes are mutually exclusive")
        if (self.lab_controller is None) != (self.lab_state_builder is None):
            raise ValueError("lab_controller and lab_state_builder must be provided together")
        self.filter_finger_observation = bool(
            self.lab_controller is not None
            if self.filter_finger_observation is None
            else self.filter_finger_observation
        )
        if self.curriculum_feed_order not in {"difficulty_sorted", "stored"}:
            raise ValueError("curriculum_feed_order must be difficulty_sorted or stored")
        if self.curriculum is not None and self.curriculum_feed_order == "difficulty_sorted":
            self.feed_bank = sorted(self.feed_bank, key=_feed_difficulty)
        if self.feed_bank_manifest is not None:
            from environment.overall_environment.src.shuttle_feeder import (
                feed_sample_fingerprint,
            )

            manifest = dict(self.feed_bank_manifest)
            stored_fingerprints = manifest.get("sample_fingerprints")
            current_fingerprints = [feed_sample_fingerprint(sample) for sample in self.feed_bank]
            if not isinstance(stored_fingerprints, list) or sorted(
                str(value) for value in stored_fingerprints
            ) != sorted(current_fingerprints):
                raise ValueError("training feed-bank manifest does not describe feed_bank")
            manifest["consumer_order"] = {
                "schema_version": "incoming_hit_curriculum_feed_order_v1",
                "mode": (self.curriculum_feed_order if self.curriculum is not None else "stored"),
                "sample_fingerprints": current_fingerprints,
            }
            self.feed_bank_manifest = manifest
        self._effective_ctrlrange_hash: str | None = None
        self._control_manifest_cache: dict[str, Any] | None = None
        if self.lab_controller is not None:
            if int(self.lab_controller.router.full_size) != int(self.model.nu):
                raise ValueError("Stage-3 LAB router does not cover every model actuator")
            if int(self.lab_state_builder.expected_state_dim) != int(self.lab_controller.lab_state_size):
                raise ValueError("LAB state builder and latent runtime dimensions differ")
            from environment.overall_environment.src.stage3_lab import (
                apply_teacher_body_ctrlrange,
            )

            self._effective_ctrlrange_hash = apply_teacher_body_ctrlrange(self.model, self.lab_controller)
        self.ids = make_ids(self.model, self.physics_config)
        self.params = make_params(self.model, self.physics_config)

        cpu_env = IncomingShuttleHitEnv(
            self.xml,
            feed_bank=self.feed_bank,
            control_substeps=self.control_substeps,
            max_episode_steps=self.max_episode_steps,
            reward_weights=self.weights,
            task_profile=self.task_profile,
            impact_target_bank=self.impact_target_bank,
            recovery_horizon_steps=self.recovery_horizon_steps,
            player_half_sign=self.player_half_sign,
            singles=self.singles,
            filter_finger_observation=self.filter_finger_observation,
            swing_duration_s=self.swing_duration_s,
            contact_phase=self.contact_phase,
            swing_phase_advance_s=self.swing_phase_advance_s,
            return_net_x_m=self.return_net_x_m,
            return_net_height_m=self.return_net_height_m,
            min_return_net_clearance_m=self.min_return_net_clearance_m,
            desired_return_up_component=self.desired_return_up_component,
            ballistic_return_score_softness_m=self.ballistic_return_score_softness_m,
            shuttle_proximity_softness_m=self.shuttle_proximity_softness_m,
            timed_intercept_softness_m=self.timed_intercept_softness_m,
            direction_distance_softness_m=self.direction_distance_softness_m,
            contact_guidance_reward_mode=self.contact_guidance_reward_mode,
            racket_velocity_direction_fraction=self.racket_velocity_direction_fraction,
            direction_reward_mode=self.direction_reward_mode,
            clearance_reward_mode=self.clearance_reward_mode,
            hit_event_mode=self.hit_event_mode,
            racket_guidance_mode=self.racket_guidance_mode,
            inverse_target_speed_m_s=self.inverse_target_speed_m_s,
            inverse_velocity_softness_m_s=(self.inverse_velocity_softness_m_s),
            racket_attachment_contract=self.racket_attachment_contract_path,
        )
        self.observation_size = cpu_env.observation_size
        self.full_action_size = int(self.model.nu)
        if self.lab_controller is None:
            self.action_size = self.full_action_size
            self.lab_state_size = 0
        else:
            if int(self.lab_controller.router.full_size) != self.full_action_size:
                raise ValueError("Stage-3 LAB router does not cover every model actuator")
            if int(self.lab_state_builder.expected_state_dim) != int(self.lab_controller.lab_state_size):
                raise ValueError("LAB state builder and latent runtime dimensions differ")
            self.action_size = int(self.lab_controller.task_action_size)
            self.lab_state_size = int(self.lab_controller.lab_state_size)
        self._qpos_obs_index = jnp.asarray(cpu_env._qpos_obs_index)
        self._qvel_obs_index = jnp.asarray(cpu_env._qvel_obs_index)
        self._root_qadr = cpu_env._root_qadr
        self._shuttle_qadr = cpu_env._shuttle_qadr
        self._shuttle_dadr = cpu_env._shuttle_dadr
        self._stringbed_site = cpu_env._stringbed_site
        self._cork_site = cpu_env._cork_site
        self._racket_body = cpu_env._racket_body
        self._racket_root = int(self.model.body_rootid[self._racket_body])
        self._root_dadr = cpu_env._root_dadr
        self._ready_qpos_index = jnp.asarray(cpu_env._ready_qpos_index)
        self._ready_qpos = jnp.asarray(cpu_env._ready_qpos, dtype=jnp.float32)
        self.impact_target_bank = cpu_env.impact_target_bank
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            arrays = cpu_env._target_arrays
            self.target_impact_position = jnp.asarray(arrays["impact_position_world"], dtype=jnp.float32)
            self.target_impact_time = jnp.asarray(arrays["impact_time_s"], dtype=jnp.float32)
            self.target_normal = jnp.asarray(arrays["stringbed_normal_world"], dtype=jnp.float32)
            self.target_linear_velocity = jnp.asarray(arrays["racket_linear_velocity_world"], dtype=jnp.float32)
            self.target_angular_velocity = jnp.asarray(arrays["racket_angular_velocity_world"], dtype=jnp.float32)
            self.target_landing_xy = jnp.asarray(arrays["landing_target_xy"], dtype=jnp.float32)
            self.target_apex_height = jnp.asarray(arrays["apex_height_m"], dtype=jnp.float32)
            self.target_recovery_horizon = jnp.asarray(arrays["recovery_horizon_steps"], dtype=jnp.int32)

        # per-feed reset templates + exact CPU-computed reset observations
        n_feeds = len(self.feed_bank)
        qpos_bank = np.zeros((n_feeds, self.model.nq))
        qvel_bank = np.zeros((n_feeds, self.model.nv))
        obs_bank = np.zeros((n_feeds, self.observation_size))
        for i in range(n_feeds):
            obs0, _info = cpu_env.reset(feed_index=i)
            qpos_bank[i] = cpu_env.data.qpos
            qvel_bank[i] = cpu_env.data.qvel
            obs_bank[i] = obs0
        self._key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
        self.qpos_bank = jnp.asarray(qpos_bank, dtype=jnp.float32)
        self.qvel_bank = jnp.asarray(qvel_bank, dtype=jnp.float32)
        self.obs_bank = jnp.asarray(obs_bank, dtype=jnp.float32)
        self.intercept_points = jnp.asarray(np.stack([f.intercept_point for f in self.feed_bank]), dtype=jnp.float32)
        self.intercept_times = jnp.asarray(np.array([f.intercept_time_s for f in self.feed_bank]), dtype=jnp.float32)

        # muscle ctrl scaling (normalized_action_to_model_ctrl)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        lower = np.asarray(self.model.actuator_ctrlrange[:, 0])
        upper = np.asarray(self.model.actuator_ctrlrange[:, 1])
        self._ctrl_limited = jnp.asarray(limited)
        self._ctrl_mean = jnp.asarray((upper + lower) / 2.0, dtype=jnp.float32)
        self._ctrl_delta = jnp.asarray((upper - lower) / 2.0, dtype=jnp.float32)
        self._ctrl_lower = jnp.asarray(lower, dtype=jnp.float32)
        self._ctrl_upper = jnp.asarray(upper, dtype=jnp.float32)

        self.timestep = float(self.model.opt.timestep)

        self._base = None
        self._base_control_binding = None
        if self.base_policy_artifact is None and (self.residual_scale_overrides or self.residual_scale_schedule):
            raise ValueError("residual scale overrides/schedule require base_policy_artifact")
        if self.base_policy_artifact is not None:
            self._init_base_policy()

    @property
    def expects_raw_latent(self) -> bool:
        return self.lab_controller is not None

    @property
    def control_manifest(self) -> dict[str, Any]:
        if self._control_manifest_cache is not None:
            return self._control_manifest_cache
        if self.lab_controller is None:
            payload: dict[str, Any] = {
                "schema_version": (
                    "incoming_hit_direct_action_v1"
                    if self.task_profile == LEGACY_PROFILE
                    else "incoming_hit_direct_action_impact_recovery_v2"
                )
            }
        else:
            payload = dict(self.lab_controller.control_manifest)
            payload["lab_state_schema_hash"] = self.lab_state_builder.schema_hash
        if self._base_control_binding is not None:
            payload["frozen_base_residual"] = self._base_control_binding
        from environment.overall_environment.src.stage3_lab import (
            stage3_attachment_report,
        )

        payload["racket_attachment"] = stage3_attachment_report(
            self.model,
            self.xml,
            contract_path=self.racket_attachment_contract_path,
        )
        payload["filter_finger_observation"] = self.filter_finger_observation
        environment_abi = {
            "schema_version": (
                "incoming_hit_environment_v1"
                if self.task_profile == LEGACY_PROFILE
                else "incoming_hit_environment_impact_recovery_v2"
            ),
            "scene_sha256": hashlib.sha256(Path(self.xml).read_bytes()).hexdigest(),
            "effective_ctrlrange_hash": self._effective_ctrlrange_hash,
            "full_action_size": self.full_action_size,
            "control_substeps": self.control_substeps,
            "max_episode_steps": self.max_episode_steps,
            "reward_weights": {
                name: value
                for name, value in self.weights.items()
                if not (
                    value == 0.0
                    and (
                        name == "return_clearance"
                        or (name == "invalid_net_crossing" and self.min_return_net_clearance_m is None)
                    )
                )
            },
            "player_half_sign": self.player_half_sign,
            "singles": self.singles,
            "terminate_on_body_fall": True,
            "swing_duration_s": self.swing_duration_s,
            "contact_phase": self.contact_phase,
        }
        if self.swing_phase_advance_s != 0.0:
            environment_abi["swing_phase_advance_s"] = self.swing_phase_advance_s
        if self.min_return_net_clearance_m is not None:
            environment_abi["return_constraints"] = {
                "net_x_m": self.return_net_x_m,
                "net_height_m": self.return_net_height_m,
                "min_clearance_m": self.min_return_net_clearance_m,
                "desired_up_component": self.desired_return_up_component,
            }
            if self.ballistic_return_score_softness_m != 0.35:
                environment_abi["return_constraints"]["ballistic_score_softness_m"] = (
                    self.ballistic_return_score_softness_m
                )
            if self.shuttle_proximity_softness_m != 0.35:
                environment_abi["return_constraints"]["shuttle_proximity_softness_m"] = (
                    self.shuttle_proximity_softness_m
                )
            if self.timed_intercept_softness_m != 0.30:
                environment_abi["return_constraints"]["timed_intercept_softness_m"] = (
                    self.timed_intercept_softness_m
                )
            if self.direction_distance_softness_m != 0.45:
                environment_abi["return_constraints"]["direction_distance_softness_m"] = (
                    self.direction_distance_softness_m
                )
            if self.contact_guidance_reward_mode != "dense_per_step":
                environment_abi["return_constraints"]["contact_guidance_reward_mode"] = (
                    self.contact_guidance_reward_mode
                )
            if self.contact_guidance_reward_mode == "potential_event_direction":
                environment_abi["return_constraints"]["contact_guidance_discount"] = (
                    self.contact_guidance_discount
                )
            if self.racket_velocity_direction_fraction != 0.30:
                environment_abi["return_constraints"]["racket_velocity_direction_fraction"] = (
                    self.racket_velocity_direction_fraction
                )
            if self.direction_reward_mode != "positive_projection":
                environment_abi["return_constraints"]["direction_reward_mode"] = self.direction_reward_mode
            if self.clearance_reward_mode != "positive_score":
                environment_abi["return_constraints"]["clearance_reward_mode"] = self.clearance_reward_mode
            if self.hit_event_mode != "any_stringbed_contact":
                environment_abi["return_constraints"]["hit_event_mode"] = self.hit_event_mode
            if self.racket_guidance_mode != "component_projection":
                environment_abi["return_constraints"]["racket_guidance_mode"] = self.racket_guidance_mode
            if self.racket_guidance_mode in {
                "inverse_impact_target",
                "inverse_impact_decomposed",
            }:
                environment_abi["return_constraints"].update(
                    {
                        "inverse_target_speed_m_s": self.inverse_target_speed_m_s,
                        "inverse_velocity_softness_m_s": (self.inverse_velocity_softness_m_s),
                    }
                )
        if self.weights.get("return_clearance", 0.0) != 0.0:
            if self.contact_guidance_reward_mode == "closest_approach_event_direction":
                environment_abi["reward_semantics"] = (
                    "incoming_hit_closest_approach_event_direction_v29"
                )
            elif self.contact_guidance_reward_mode == "potential_event_direction":
                environment_abi["reward_semantics"] = (
                    "incoming_hit_discounted_potential_event_direction_v27"
                )
            elif self.contact_guidance_reward_mode == "event_direction":
                environment_abi["reward_semantics"] = (
                    "incoming_hit_event_direction_quality_v26"
                )
            elif self.contact_guidance_reward_mode == "best_progress":
                environment_abi["reward_semantics"] = "incoming_hit_bounded_contact_progress_v23"
            elif (
                self.direction_reward_mode == "positive_projection"
                and self.clearance_reward_mode == "positive_score"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_decomposed"
                and self.weights.get("miss", 0.0) != 0.0
            ):
                environment_abi["reward_semantics"] = "incoming_hit_wrist_hierarchical_quality_v19"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_decomposed"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_inverse_impact_decomposed_quality_v16"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "inverse_impact_target"
            ):
                environment_abi["reward_semantics"] = (
                    "incoming_hit_quality_hierarchy_v15"
                    if self.clearance_reward_mode == "signed_centered" or self.weights.get("miss", 0.0) != 0.0
                    else "incoming_hit_inverse_impact_target_guidance_v12"
                )
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "counterfactual_clearance_priority"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_counterfactual_clearance_priority_v11"
            elif (
                self.direction_reward_mode == "signed_projection"
                and self.hit_event_mode == "event_rebound"
                and self.racket_guidance_mode == "counterfactual_rebound"
            ):
                environment_abi["reward_semantics"] = "incoming_hit_counterfactual_rebound_guidance_v10"
            elif self.direction_reward_mode == "signed_projection" and self.hit_event_mode == "event_rebound":
                environment_abi["reward_semantics"] = "incoming_hit_effective_rebound_direction_v9"
            elif self.direction_reward_mode == "signed_projection":
                environment_abi["reward_semantics"] = "incoming_hit_signed_task_direction_v8"
            else:
                environment_abi["reward_semantics"] = (
                    "incoming_hit_non_saturating_ballistic_direction_v6"
                    if (
                        self.ballistic_return_score_softness_m != 0.35
                        or self.racket_velocity_direction_fraction != 0.30
                    )
                    else "incoming_hit_ballistic_legal_return_v5"
                )
        elif any(
            self.weights.get(name, 0.0) != 0.0
            for name in (
                "shuttle_proximity",
                "timed_intercept",
                "racket_direction",
                "hit_speed",
                "return_direction",
            )
        ):
            environment_abi["reward_semantics"] = (
                "incoming_hit_legal_return_stability_v4"
                if self.min_return_net_clearance_m is not None
                else "incoming_hit_timed_cork_task_direction_v3"
            )
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            environment_abi.update(
                {
                    "task_profile": self.task_profile,
                    "target_bank_sha256": self.impact_target_bank.bank_sha256,
                    "v2_observation_size": V2_OBSERVATION_SIZE,
                    "recovery_horizon_steps": self.recovery_horizon_steps,
                }
            )
        payload["environment_abi"] = environment_abi
        payload["curriculum"] = None if self.curriculum is None else dict(vars(self.curriculum))
        if self.curriculum is not None and self.curriculum_feed_order != "difficulty_sorted":
            payload["curriculum_feed_order"] = self.curriculum_feed_order
        # Do not perturb legacy Stage-3 checkpoint control hashes.  This field
        # is required only by the opt-in impact/recovery v2 artifact contract.
        if self.task_profile == IMPACT_RECOVERY_PROFILE:
            payload["policy_abi_hash"] = incoming_hit_policy_abi_hash(payload)
        payload_without_hash = dict(payload)
        payload_without_hash.pop("control_hash", None)
        payload["control_hash"] = hashlib.sha256(
            json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._control_manifest_cache = payload
        return payload

    @property
    def control_hash(self) -> str | None:
        return self.control_manifest.get("control_hash")

    @property
    def policy_abi_hash(self) -> str | None:
        return self.control_manifest.get("policy_abi_hash")

    def curriculum_values(self, env_steps: int):
        if self.curriculum is None:
            from environment.overall_environment.src.stage3_lab import Stage3CurriculumValues

            return Stage3CurriculumValues(
                lambda_lab=float(self.lab_controller.lambda_lab if self.lab_controller is not None else 0.0),
                feed_fraction=1.0,
                active_feed_count=len(self.feed_bank),
            )
        return self.curriculum.values(env_steps=int(env_steps), feed_bank_size=len(self.feed_bank))

    def task_curriculum_values(self, env_steps: int, *, stage_index: int | None = None):
        if self.task_curriculum_max_stage is None:
            stage = task_stage_by_name("C7_recovery")
            return task_runtime_values(
                stage.start_steps,
                feed_bank_size=len(self.feed_bank),
                max_stage=stage,
            )
        if stage_index is not None:
            return task_runtime_values_for_stage(
                stage_index,
                feed_bank_size=len(self.feed_bank),
                max_stage=self.task_curriculum_max_stage,
            )
        return task_runtime_values(
            env_steps,
            feed_bank_size=len(self.feed_bank),
            max_stage=self.task_curriculum_max_stage,
        )

    def task_curriculum_phase(self, env_steps: int) -> str:
        return self.task_curriculum_values(env_steps).stage_name

    def task_curriculum_complete(self, env_steps: int) -> bool:
        if self.task_curriculum_max_stage is None:
            return True
        return task_curriculum_complete(env_steps, max_stage=self.task_curriculum_max_stage)

    def apply_curriculum(
        self,
        state: EnvState,
        *,
        env_steps: int = 0,
        lambda_lab: float,
        active_feed_count: int,
        v2_stage_index: int | None = None,
        v2_environment_mode: int | None = None,
        v2_reward_mask: tuple[float, ...] | None = None,
    ) -> EnvState:
        if not (1 <= int(active_feed_count) <= len(self.feed_bank)):
            raise ValueError("active_feed_count is outside the feed bank")
        updates: dict[str, Any] = {
            "lambda_lab": jnp.asarray(lambda_lab, dtype=jnp.float32),
            "active_feed_count": jnp.asarray(active_feed_count, dtype=jnp.int32),
            "residual_authority_progress": jnp.asarray(
                (
                    1.0
                    if self._base is None or int(self._base["residual_scale_ramp_steps"]) <= 0
                    else min(
                        1.0,
                        max(0.0, float(env_steps) / float(self._base["residual_scale_ramp_steps"])),
                    )
                ),
                dtype=jnp.float32,
            ),
        }
        if v2_stage_index is not None:
            if v2_environment_mode is None or v2_reward_mask is None:
                raise ValueError("incomplete Stage-3 v2 curriculum runtime values")
            if len(v2_reward_mask) != len(V2_REWARD_TERM_ORDER):
                raise ValueError("Stage-3 v2 reward mask has the wrong length")
            updates.update(
                {
                    "v2_stage_index": jnp.asarray(v2_stage_index, dtype=jnp.int32),
                    "v2_environment_mode": jnp.asarray(v2_environment_mode, dtype=jnp.int32),
                    "v2_reward_mask": jnp.asarray(v2_reward_mask, dtype=jnp.float32),
                }
            )
        return state._replace(**updates)

    def _init_base_policy(self) -> None:
        """Load the frozen base policy and precompute batched body-obs indices."""
        import mujoco as mj

        from environment.overall_environment.src.base_swing_bridge import (
            BaseSwingBridge,
            SwingPhaseConfig,
        )

        bridge = BaseSwingBridge(
            self.base_policy_artifact,
            self.model,
            residual_scale=self.residual_scale,
            residual_scale_overrides=self.residual_scale_overrides,
            residual_scale_schedule=self.residual_scale_schedule,
            phase_config=SwingPhaseConfig(
                swing_duration_s=self.swing_duration_s,
                contact_phase=self.contact_phase,
                phase_advance_s=self.swing_phase_advance_s,
            ),
            skill=self.base_skill,
        )
        self._base_control_binding = bridge.control_binding
        tensors = bridge.jax_arrays()
        schema = bridge.schema
        model = self.model

        def _joint(name: str) -> int:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"hitting scene missing base-policy joint {name!r}")
            return jid

        root_id = _joint(schema.root_joint_name)
        root_qadr = int(model.jnt_qposadr[root_id])
        root_dadr = int(model.jnt_dofadr[root_id])

        def _width(jid: int, kind: str) -> int:
            jt = int(model.jnt_type[jid])
            if jt == int(mj.mjtJoint.mjJNT_FREE):
                return 7 if kind == "qpos" else 6
            if jt == int(mj.mjtJoint.mjJNT_BALL):
                return 4 if kind == "qpos" else 3
            return 1

        qpos_idx: list[int] = []
        qvel_idx: list[int] = []
        for name in schema.joint_names:
            jid = _joint(name)
            qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            qpos_idx.extend(range(qadr, qadr + _width(jid, "qpos")))
            qvel_idx.extend(range(dadr, dadr + _width(jid, "qvel")))

        muscle_contract = resolve_muscle_channel_contract(
            model,
            schema.actuator_names,
        )
        actuator_ids = list(muscle_contract.actuator_ids)
        activation_addresses = list(muscle_contract.actuator_actadr)
        sensor_adr = []
        for name in schema.touch_sensor_names:
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"hitting scene missing base-policy sensor {name!r}")
            sensor_adr.append(int(model.sensor_adr[sid]))

        # residual/base action mapping: source (checkpoint) actuator -> full ctrl index
        target_ids = jnp.asarray(actuator_ids, dtype=jnp.int32)

        self._base = {
            "layers": [{k: jnp.asarray(v) for k, v in layer.items()} for layer in tensors["layers"]],
            "activation": tensors["activation"],
            "use_layernorm": tensors["use_layernorm"],
            "layernorm_eps": tensors["layernorm_eps"],
            "obs_mean": jnp.asarray(tensors["obs_mean"]),
            "obs_var": jnp.asarray(tensors["obs_var"]),
            "goal_size": int(tensors["goal_size"]),
            "root_qadr": root_qadr,
            "root_dadr": root_dadr,
            "qpos_idx": jnp.asarray(qpos_idx, dtype=jnp.int32),
            "qvel_idx": jnp.asarray(qvel_idx, dtype=jnp.int32),
            "actuator_ids": jnp.asarray(actuator_ids, dtype=jnp.int32),
            "activation_addresses": jnp.asarray(
                activation_addresses,
                dtype=jnp.int32,
            ),
            "sensor_adr": jnp.asarray(sensor_adr, dtype=jnp.int32),
            "target_ids": target_ids,
            "skill_onehot": jnp.asarray(tensors["skill_onehot"]),
            "residual_scale_vector": jnp.asarray(tensors["residual_scale_vector"]),
            "residual_scale_initial_vector": jnp.asarray(tensors["residual_scale_initial_vector"]),
            "residual_scale_ramp_steps": int(tensors["residual_scale_ramp_steps"]),
            "residual_override_indices": jnp.asarray(
                tensors["residual_override_indices"],
                dtype=jnp.int32,
            ),
        }

    # ---- base policy (batched jnp) -------------------------------------------

    def _base_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        name = self._base["activation"]
        if name in ("tanh",):
            return jnp.tanh(x)
        if name in ("relu",):
            return jax.nn.relu(x)
        if name in ("elu",):
            return jax.nn.elu(x)
        if name in ("swish", "silu"):
            return jax.nn.silu(x)
        raise ValueError(f"unsupported base activation {name!r}")

    def _base_forward(self, obs: jnp.ndarray) -> jnp.ndarray:
        b = self._base
        x = (obs - b["obs_mean"]) / jnp.sqrt(b["obs_var"] + 1e-8)
        layers = b["layers"]
        for layer in layers[:-1]:
            x = x @ layer["kernel"] + layer["bias"]
            if b["use_layernorm"]:
                mean = x.mean(axis=-1, keepdims=True)
                var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
                x = (x - mean) / jnp.sqrt(var + b["layernorm_eps"])
                x = x * layer["ln_scale"] + layer["ln_bias"]
            x = self._base_activation(x)
        out = layers[-1]
        return x @ out["kernel"] + out["bias"]

    def _base_body_obs(self, data: Any, phase: jnp.ndarray) -> jnp.ndarray:
        b = self._base
        n = data.qpos.shape[0]
        rq, rd = b["root_qadr"], b["root_dadr"]
        aid = b["actuator_ids"]
        activation_addresses = b["activation_addresses"]
        muscle = jnp.stack(
            [
                data.actuator_length[:, aid],
                data.actuator_velocity[:, aid],
                data.actuator_force[:, aid],
                data.ctrl[:, aid],
                data.act[:, activation_addresses],
            ],
            axis=2,
        ).reshape(n, -1)
        touch = data.sensordata[:, b["sensor_adr"]] if int(b["sensor_adr"].shape[0]) > 0 else jnp.zeros((n, 0))
        goal = jnp.zeros((n, b["goal_size"]))
        if b["goal_size"] >= 1:
            goal = goal.at[:, -1].set(jnp.clip(phase, 0.0, 1.0))
        parts = [
            data.qpos[:, rq + 2 : rq + 7],
            data.qpos[:, b["qpos_idx"]],
            data.qvel[:, rd : rd + 6],
            data.qvel[:, b["qvel_idx"]],
            muscle,
            touch,
            goal,
        ]
        if int(b["skill_onehot"].shape[0]) > 0:
            parts.append(jnp.broadcast_to(b["skill_onehot"], (n, b["skill_onehot"].shape[0])))
        return jnp.concatenate(parts, axis=-1)

    def _swing_phase(self, step_index: jnp.ndarray, intercept_time: jnp.ndarray) -> jnp.ndarray:
        elapsed = step_index.astype(jnp.float32) * self.control_substeps * self.timestep
        start = intercept_time - self.swing_phase_advance_s - self.contact_phase * self.swing_duration_s
        return jnp.clip((elapsed - start) / self.swing_duration_s, 0.0, 1.0)

    def _compose_action(self, data: Any, state: EnvState, action: jnp.ndarray):
        """Return the normalized full-body action ([-1,1]) for this policy step."""
        if self.lab_controller is not None:
            output = self.lab_controller.decode_task_jax(
                lab_state=state.lab_state,
                task_action=action,
                lambda_lab=state.lambda_lab,
            )
            return output.full_action, output
        if self._base is None:
            return action, None
        phase = self._swing_phase(state.step_index, self.intercept_times[state.feed_idx])
        body_obs = self._base_body_obs(data, phase)
        base_src = self._base_forward(body_obs)
        n = action.shape[0]
        base_full = jnp.zeros((n, self.full_action_size), dtype=action.dtype)
        base_full = base_full.at[:, self._base["target_ids"]].set(base_src)
        residual_scale_vector = self._base["residual_scale_initial_vector"] + state.residual_authority_progress * (
            self._base["residual_scale_vector"] - self._base["residual_scale_initial_vector"]
        )
        return jnp.clip(
            base_full + residual_scale_vector * action,
            -1.0,
            1.0,
        ), None

    # ---- backend objects ---------------------------------------------------

    def put_model(self, num_envs: int):
        if self.impl == "warp":
            return mjx.put_model(self.model, impl="warp")
        return mjx.put_model(self.model)

    def make_batched_template(self, num_envs: int):
        """Semi-batched (warp) or fully batched (jax) Data at the ready keyframe."""
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self._key_id)
        mujoco.mj_forward(self.model, data)
        if self.impl == "warp":
            budget = self.nconmax_per_env * num_envs
            dx = mjx.make_data(
                self.model,
                impl="warp",
                nconmax=budget,
                njmax=self.nconmax_per_env * 16,
                naconmax=budget,
            )
            dx = dx.replace(
                qpos=jnp.asarray(data.qpos, dtype=jnp.float32),
                qvel=jnp.asarray(data.qvel, dtype=jnp.float32),
                act=jnp.asarray(data.act, dtype=jnp.float32),
            )
            unbatched = _warp_unbatched_fields()

            def tile(path, x):
                if _attr_name(path) in unbatched or not hasattr(x, "ndim"):
                    return x
                return jnp.tile(x[None], (num_envs,) + (1,) * x.ndim)

            return jax.tree_util.tree_map_with_path(tile, dx)

        dx = mjx.put_data(self.model, data)
        return jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None], (num_envs,) + (1,) * x.ndim) if hasattr(x, "ndim") else x,
            dx,
        )

    # ---- jit-able bits -------------------------------------------------------

    def scale_action(self, action: jnp.ndarray) -> jnp.ndarray:
        scaled = action * self._ctrl_delta + self._ctrl_mean
        scaled = jnp.clip(scaled, self._ctrl_lower, self._ctrl_upper)
        return jnp.where(self._ctrl_limited, scaled, action)

    def _reset_arrays(self, data: Any, feed_idx: jnp.ndarray) -> Any:
        """Data with qpos/qvel/act/ctrl replaced by reset values (batched)."""
        return data.replace(
            qpos=self.qpos_bank[feed_idx],
            qvel=self.qvel_bank[feed_idx],
            act=jnp.zeros_like(data.act),
            ctrl=jnp.zeros_like(data.ctrl),
            qfrc_applied=jnp.zeros_like(data.qfrc_applied),
        )

    def _forward_all(self, mx):
        if self.impl == "warp":
            return lambda d: mjx.forward(mx, d)
        return jax.vmap(lambda d: mjx.forward(mx, d))

    def make_reset_fn(self, mx, num_envs: int):
        forward_all = self._forward_all(mx)

        def reset(key: jnp.ndarray, template: Any) -> EnvState:
            key, sub = jax.random.split(key)
            values = self.curriculum_values(0)
            task_values = self.task_curriculum_values(0)
            active_feed_count = jnp.asarray(
                min(values.active_feed_count, task_values.active_feed_count),
                dtype=jnp.int32,
            )
            feed_idx = jax.random.randint(sub, (num_envs,), 0, active_feed_count)
            data = self._reset_arrays(template, feed_idx)
            if self.task_profile == IMPACT_RECOVERY_PROFILE and task_values.environment_mode_code != 3:
                static_position = self.target_impact_position[feed_idx]
                if task_values.environment_mode_code in (0, 1):
                    static_position = static_position.at[:, 2].set(100.0)
                qpos = data.qpos.at[:, self._shuttle_qadr : self._shuttle_qadr + 3].set(static_position)
                qvel = data.qvel.at[:, self._shuttle_dadr : self._shuttle_dadr + 6].set(0.0)
                data = data.replace(qpos=qpos, qvel=qvel)
            data = forward_all(data)
            zeros_i = jnp.zeros((num_envs,), jnp.int32)
            zeros_b = jnp.zeros((num_envs,), bool)
            lab_state = (
                self.lab_state_builder.build_jax(
                    data=data,
                    phase=self._swing_phase(zeros_i, self.intercept_times[feed_idx]),
                )
                if self.lab_state_builder is not None
                else jnp.zeros((num_envs, 0), dtype=jnp.float32)
            )
            return EnvState(
                data=data,
                obs=(
                    self._observation(
                        data,
                        feed_idx,
                        zeros_i,
                        phase_code=zeros_i,
                        recovery_step=zeros_i,
                        apex_height=data.qpos[:, self._shuttle_qadr + 2],
                    )
                    if self.task_profile == IMPACT_RECOVERY_PROFILE and task_values.environment_mode_code != 3
                    else self.obs_bank[feed_idx]
                ),
                cooldown=zeros_i,
                step_index=zeros_i,
                phase_code=zeros_i,
                hit_rewarded=zeros_b,
                crossed_rewarded=zeros_b,
                invalid_net_crossed=zeros_b,
                hit_closing_speed=jnp.zeros((num_envs,), jnp.float32),
                best_shuttle_proximity_potential=jnp.zeros((num_envs,), jnp.float32),
                best_timed_intercept_potential=jnp.zeros((num_envs,), jnp.float32),
                best_racket_direction_potential=jnp.zeros((num_envs,), jnp.float32),
                closest_racket_distance_m=jnp.full(
                    (num_envs,), jnp.inf, jnp.float32
                ),
                closest_racket_direction_score=jnp.zeros(
                    (num_envs,), jnp.float32
                ),
                landing_recorded=zeros_b,
                landing_xy=jnp.zeros((num_envs, 2), jnp.float32),
                apex_height=data.qpos[:, self._shuttle_qadr + 2],
                recovery_step=zeros_i,
                recovery_active=zeros_b,
                recovery_complete=zeros_b,
                flight_resolved=jnp.full((num_envs,), task_values.environment_mode_code != 3, bool),
                feed_idx=feed_idx,
                lab_state=lab_state,
                lambda_lab=jnp.asarray(values.lambda_lab, dtype=jnp.float32),
                active_feed_count=active_feed_count,
                residual_authority_progress=jnp.asarray(
                    (0.0 if self._base is not None and int(self._base["residual_scale_ramp_steps"]) > 0 else 1.0),
                    dtype=jnp.float32,
                ),
                v2_stage_index=jnp.asarray(task_values.stage_index, dtype=jnp.int32),
                v2_environment_mode=jnp.asarray(task_values.environment_mode_code, dtype=jnp.int32),
                v2_reward_mask=jnp.asarray(task_values.reward_mask, dtype=jnp.float32),
                key=key,
            )

        return reset

    def _observation(
        self,
        data: Any,
        feed_idx: jnp.ndarray,
        step_index: jnp.ndarray,
        *,
        phase_code: jnp.ndarray | None = None,
        recovery_step: jnp.ndarray | None = None,
        recovery_active: jnp.ndarray | None = None,
        apex_height: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        qpos = data.qpos[:, self._qpos_obs_index]
        qvel = data.qvel[:, self._qvel_obs_index]

        root_pos = data.qpos[:, self._root_qadr : self._root_qadr + 3]
        shuttle_pos = data.qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3]
        shuttle_vel = data.qvel[:, self._shuttle_dadr : self._shuttle_dadr + 3]
        stringbed_pos = data.site_xpos[:, self._stringbed_site]
        stringbed_mat = data.site_xmat[:, self._stringbed_site].reshape(-1, 3, 3)
        face_normal = stringbed_mat[:, :, 2]

        cvel = data.cvel[:, self._racket_body]
        offset = stringbed_pos - data.subtree_com[:, self._racket_root]
        face_vel = cvel[:, 3:] + jnp.cross(cvel[:, :3], offset)

        shuttle_features = jnp.concatenate(
            [
                shuttle_pos - root_pos,
                shuttle_vel,
                shuttle_pos - stringbed_pos,
                shuttle_vel - face_vel,
            ],
            axis=-1,
        )
        racket_features = jnp.concatenate([stringbed_pos - root_pos, face_normal, face_vel], axis=-1)

        intercept = self.intercept_points[feed_idx]
        elapsed = step_index.astype(jnp.float32) * self.control_substeps * self.timestep
        time_to_intercept = jnp.maximum(0.0, self.intercept_times[feed_idx] - elapsed)
        phase = self._swing_phase(step_index, self.intercept_times[feed_idx]).astype(jnp.float32)
        task_features = jnp.concatenate(
            [
                intercept - stringbed_pos,
                intercept - root_pos,
                time_to_intercept[:, None],
                phase[:, None],
            ],
            axis=-1,
        )
        legacy = jnp.concatenate([qpos, qvel, shuttle_features, racket_features, task_features], axis=-1)
        if self.task_profile == LEGACY_PROFILE:
            return legacy
        if phase_code is None:
            phase_code = jnp.zeros_like(step_index)
        if recovery_step is None:
            recovery_step = jnp.zeros_like(step_index)
        if recovery_active is None:
            recovery_active = phase_code == STATE_RECOVERY
        if apex_height is None:
            apex_height = shuttle_pos[:, 2]
        target_position = self.target_impact_position[feed_idx]
        target_normal = self.target_normal[feed_idx]
        target_linear = self.target_linear_velocity[feed_idx]
        target_angular = self.target_angular_velocity[feed_idx]
        target_landing = self.target_landing_xy[feed_idx]
        target_apex = self.target_apex_height[feed_idx]
        target_time = self.target_impact_time[feed_idx]
        horizon = self.target_recovery_horizon[feed_idx]
        ready_delta = data.qpos[:, self._ready_qpos_index] - self._ready_qpos
        ready_error = jnp.sqrt(jnp.mean(jnp.square(ready_delta), axis=-1))
        recovery_progress = jnp.minimum(
            1.0,
            recovery_step.astype(jnp.float32) / jnp.maximum(horizon.astype(jnp.float32), 1.0),
        )
        v2 = jnp.concatenate(
            [
                target_position - stringbed_pos,
                target_normal,
                target_linear - face_vel,
                target_angular - cvel[:, :3],
                target_landing - shuttle_pos[:, :2],
                (target_apex - apex_height)[:, None],
                jnp.maximum(0.0, target_time - elapsed)[:, None],
                recovery_progress[:, None],
                ready_error[:, None],
                recovery_active.astype(jnp.float32)[:, None],
            ],
            axis=-1,
        )
        return jnp.concatenate([legacy, v2], axis=-1)

    def _landing_score(self, xy: jnp.ndarray) -> jnp.ndarray:
        half_width = 2.59 if self.singles else 3.05
        x, y = xy[:, 0], xy[:, 1]
        out = (jnp.abs(x) > _COURT_HALF_LENGTH) | (jnp.abs(y) > half_width)
        own = jnp.sign(x) == self.player_half_sign
        depth = jnp.abs(x)
        opponent = jnp.where(depth < _NET_FRONT_DEPTH, 0.2, jnp.where(depth >= _BACK_DEPTH, 1.0, 0.5))
        return jnp.where(out, -1.0, jnp.where(own, -0.5, opponent))

    def make_step_fn(self, mx, num_envs: int):
        substep = make_batched_substep_fn(mx, self.ids, self.params, self.model, vmap_mjx_step=self.impl != "warp")
        forward_all = self._forward_all(mx)
        w = self.weights
        v2_reward_index = {name: index for index, name in enumerate(V2_REWARD_TERM_ORDER)}
        if self.lab_controller is not None:
            normalizer = getattr(self.lab_controller.runtime, "normalizer", None)
            if normalizer is None:
                lab_norm_mean = jnp.full((self.lab_state_size,), jnp.nan, dtype=jnp.float32)
                lab_norm_std = jnp.ones((self.lab_state_size,), dtype=jnp.float32)
            else:
                lab_norm_mean = jnp.asarray(normalizer.mean, dtype=jnp.float32)
                lab_norm_std = jnp.asarray(normalizer.std, dtype=jnp.float32)
        else:
            lab_norm_mean = jnp.zeros((0,), dtype=jnp.float32)
            lab_norm_std = jnp.ones((0,), dtype=jnp.float32)

        def step(state: EnvState, action: jnp.ndarray):
            previous_shuttle_pos = state.data.qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3]
            composed, lab_output = self._compose_action(state.data, state, action)
            ctrl = self.scale_action(composed)
            data = state.data.replace(ctrl=ctrl)

            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                no_contact_static = state.v2_environment_mode <= 1
                fixed_static = state.v2_environment_mode == 2
                hold_static = no_contact_static | (
                    fixed_static & (~state.hit_rewarded) & (~state.recovery_active) & (~state.recovery_complete)
                )
                static_position = self.target_impact_position[state.feed_idx]
                static_position = static_position.at[:, 2].set(
                    jnp.where(no_contact_static, 100.0, static_position[:, 2])
                )
                shuttle_position = jnp.where(
                    hold_static,
                    static_position,
                    data.qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3],
                )
                shuttle_position = shuttle_position.at[:, 2].set(
                    jnp.where(
                        state.landing_recorded,
                        jnp.asarray(GROUND_REST_HEIGHT_M, data.qpos.dtype),
                        shuttle_position[:, 2],
                    )
                )
                qpos = data.qpos.at[:, self._shuttle_qadr : self._shuttle_qadr + 3].set(shuttle_position)
                shuttle_qvel = data.qvel[:, self._shuttle_dadr : self._shuttle_dadr + 6]
                shuttle_qvel = jnp.where(
                    (hold_static | state.landing_recorded)[:, None],
                    0.0,
                    shuttle_qvel,
                )
                qvel = data.qvel.at[:, self._shuttle_dadr : self._shuttle_dadr + 6].set(shuttle_qvel)
                data = data.replace(qpos=qpos, qvel=qvel)

            def sub_body(carry, _):
                (
                    d,
                    cd,
                    hit,
                    contact_seen,
                    rebound_seen,
                    closing,
                    event_normal,
                    event_shuttle_before,
                    event_shuttle_after,
                    event_racket_velocity,
                ) = carry
                d, cd, diag = substep(d, cd)
                contact_speed = jnp.abs(diag["relative_normal_velocity"]).astype(jnp.float32)
                sub_contact = diag["stringbed_active"] & (contact_speed > 0.05)
                sub_rebound = diag["event_rebound_used"]
                sub_hit = sub_rebound if self.hit_event_mode == "event_rebound" else (sub_rebound | sub_contact)
                closing = jnp.where(sub_hit, jnp.maximum(closing, contact_speed), closing)
                event_normal = jnp.where(
                    sub_rebound[:, None],
                    diag["event_stringbed_normal_world"],
                    event_normal,
                )
                event_shuttle_before = jnp.where(
                    sub_rebound[:, None],
                    diag["event_shuttle_velocity_before_world_m_s"],
                    event_shuttle_before,
                )
                event_shuttle_after = jnp.where(
                    sub_rebound[:, None],
                    diag["event_shuttle_velocity_after_world_m_s"],
                    event_shuttle_after,
                )
                event_racket_velocity = jnp.where(
                    sub_rebound[:, None],
                    diag["event_racket_surface_velocity_world_m_s"],
                    event_racket_velocity,
                )
                return (
                    d,
                    cd,
                    hit | sub_hit,
                    contact_seen | sub_contact,
                    rebound_seen | sub_rebound,
                    closing,
                    event_normal,
                    event_shuttle_before,
                    event_shuttle_after,
                    event_racket_velocity,
                ), None

            if self.task_profile == IMPACT_RECOVERY_PROFILE:

                def sub_body_v2(carry, _):
                    (
                        d,
                        cd,
                        hit,
                        contact_seen,
                        rebound_seen,
                        closing,
                        best_rho2,
                        best_normal,
                        best_position,
                        event_normal,
                        event_shuttle_before,
                        event_shuttle_after,
                        event_racket_velocity,
                    ) = carry
                    d, cd, diag = substep(d, cd)
                    contact_speed = jnp.abs(diag["relative_normal_velocity"]).astype(jnp.float32)
                    sub_contact = diag["stringbed_active"] & (contact_speed > 0.05)
                    sub_rebound = diag["event_rebound_used"]
                    sub_hit = sub_rebound if self.hit_event_mode == "event_rebound" else (sub_rebound | sub_contact)
                    better = sub_contact & (diag["stringbed_rho2"] < best_rho2)
                    best_rho2 = jnp.where(better, diag["stringbed_rho2"], best_rho2)
                    best_normal = jnp.where(better[:, None], diag["stringbed_normal_world"], best_normal)
                    best_position = jnp.where(
                        better[:, None],
                        d.site_xpos[:, self._stringbed_site],
                        best_position,
                    )
                    closing = jnp.where(sub_hit, jnp.maximum(closing, contact_speed), closing)
                    event_normal = jnp.where(
                        sub_rebound[:, None],
                        diag["event_stringbed_normal_world"],
                        event_normal,
                    )
                    event_shuttle_before = jnp.where(
                        sub_rebound[:, None],
                        diag["event_shuttle_velocity_before_world_m_s"],
                        event_shuttle_before,
                    )
                    event_shuttle_after = jnp.where(
                        sub_rebound[:, None],
                        diag["event_shuttle_velocity_after_world_m_s"],
                        event_shuttle_after,
                    )
                    event_racket_velocity = jnp.where(
                        sub_rebound[:, None],
                        diag["event_racket_surface_velocity_world_m_s"],
                        event_racket_velocity,
                    )
                    return (
                        d,
                        cd,
                        hit | sub_hit,
                        contact_seen | sub_contact,
                        rebound_seen | sub_rebound,
                        closing,
                        best_rho2,
                        best_normal,
                        best_position,
                        event_normal,
                        event_shuttle_before,
                        event_shuttle_after,
                        event_racket_velocity,
                    ), None

                (
                    (
                        data,
                        cooldown,
                        hit_this_step,
                        contact_this_step,
                        rebound_this_step,
                        closing_speed,
                        best_rho2,
                        best_contact_normal,
                        best_contact_position,
                        event_contact_normal,
                        event_shuttle_velocity_before,
                        event_shuttle_velocity_after,
                        event_racket_surface_velocity,
                    ),
                    _,
                ) = jax.lax.scan(
                    sub_body_v2,
                    (
                        data,
                        state.cooldown,
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), jnp.float32),
                        jnp.full((num_envs,), jnp.inf, jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                    ),
                    None,
                    length=self.control_substeps,
                )
            else:
                (
                    (
                        data,
                        cooldown,
                        hit_this_step,
                        contact_this_step,
                        rebound_this_step,
                        closing_speed,
                        event_contact_normal,
                        event_shuttle_velocity_before,
                        event_shuttle_velocity_after,
                        event_racket_surface_velocity,
                    ),
                    _,
                ) = jax.lax.scan(
                    sub_body,
                    (
                        data,
                        state.cooldown,
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                    ),
                    None,
                    length=self.control_substeps,
                )
                best_rho2 = jnp.ones((num_envs,), jnp.float32)
                best_contact_normal = jnp.zeros((num_envs, 3), jnp.float32)
                best_contact_position = jnp.zeros((num_envs, 3), jnp.float32)

            step_index = state.step_index + 1
            was_incoming = state.phase_code == STATE_INCOMING
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                previous_elapsed = state.step_index.astype(jnp.float32) * self.control_substeps * self.timestep
                current_elapsed = step_index.astype(jnp.float32) * self.control_substeps * self.timestep
                virtual_hit = (
                    (state.v2_environment_mode == 1)
                    & was_incoming
                    & (previous_elapsed < self.target_impact_time[state.feed_idx])
                    & (self.target_impact_time[state.feed_idx] <= current_elapsed)
                )
                virtual_position = data.site_xpos[:, self._stringbed_site]
                virtual_position_error = jnp.linalg.norm(
                    virtual_position - self.target_impact_position[state.feed_idx],
                    axis=-1,
                )
                hit_this_step = hit_this_step | virtual_hit
                best_rho2 = jnp.where(
                    virtual_hit,
                    jnp.square(virtual_position_error / 0.15),
                    best_rho2,
                )
                virtual_normal = data.site_xmat[:, self._stringbed_site].reshape(-1, 3, 3)[:, :, 2]
                best_contact_normal = jnp.where(virtual_hit[:, None], virtual_normal, best_contact_normal)
                best_contact_position = jnp.where(virtual_hit[:, None], virtual_position, best_contact_position)
            phase_code = jnp.where(was_incoming & hit_this_step, STATE_HIT, state.phase_code)
            hit_closing_speed = jnp.where(was_incoming & hit_this_step, closing_speed, state.hit_closing_speed)

            shuttle_pos = data.qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3]
            # Post-substep COM velocity includes any real stringbed rebound and
            # is therefore the correct signal for the one-shot return reward.
            shuttle_vel = data.qvel[:, self._shuttle_dadr : self._shuttle_dadr + 3]
            shuttle_contact_pos = data.site_xpos[:, self._cork_site]
            landed = shuttle_pos[:, 2] <= GROUND_REST_HEIGHT_M
            has_hit = state.hit_rewarded | hit_this_step
            previous_x = previous_shuttle_pos[:, 0] - self.return_net_x_m
            current_x = shuttle_pos[:, 0] - self.return_net_x_m
            opponent_side = (jnp.sign(current_x) == -self.player_half_sign) & (jnp.abs(current_x) > 1.0e-9)
            if self.min_return_net_clearance_m is None:
                return_plane_crossed = opponent_side
                net_crossing_height = shuttle_pos[:, 2]
                valid_return_cross = has_hit & return_plane_crossed
            else:
                travelled_to_opponent = -float(self.player_half_sign) * (current_x - previous_x) > 0.0
                return_plane_crossed = (
                    (float(self.player_half_sign) * previous_x >= 0.0)
                    & (float(self.player_half_sign) * current_x < 0.0)
                    & travelled_to_opponent
                )
                denominator = current_x - previous_x
                alpha = jnp.clip(
                    -previous_x / jnp.where(jnp.abs(denominator) > 1.0e-9, denominator, 1.0),
                    0.0,
                    1.0,
                )
                net_crossing_height = previous_shuttle_pos[:, 2] + alpha * (
                    shuttle_pos[:, 2] - previous_shuttle_pos[:, 2]
                )
                required_height = self.return_net_height_m + self.min_return_net_clearance_m
                valid_return_cross = has_hit & return_plane_crossed & (net_crossing_height >= required_height)
            invalid_return_cross = has_hit & return_plane_crossed & (~valid_return_cross)
            phase_code = jnp.where(
                (phase_code == STATE_HIT) & valid_return_cross,
                STATE_FLIGHT,
                phase_code,
            )
            invalid_net_crossed = state.invalid_net_crossed | invalid_return_cross

            root_z = data.qpos[:, self._root_qadr + 2]
            body_fall = root_z < BODY_FALL_ROOT_HEIGHT_M
            first_hit = hit_this_step & (~state.hit_rewarded)
            apex_height = jnp.where(
                has_hit,
                jnp.maximum(state.apex_height, shuttle_pos[:, 2]),
                state.apex_height,
            )
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                miss = landed & (phase_code == STATE_INCOMING)
                landing_fire = landed & has_hit & (~state.landing_recorded)
                landing_recorded = state.landing_recorded | landing_fire
                landing_xy = jnp.where(landing_fire[:, None], shuttle_pos[:, :2], state.landing_xy)
                static_target = state.v2_environment_mode != 3
                flight_resolved = state.flight_resolved | landing_fire | (first_hit & static_target)
                recovery_step = jnp.where(
                    first_hit,
                    jnp.zeros_like(state.recovery_step),
                    jnp.where(
                        state.recovery_active,
                        state.recovery_step + 1,
                        state.recovery_step,
                    ),
                )
                recovery_complete = state.recovery_complete | (
                    state.recovery_active & (recovery_step >= self.target_recovery_horizon[state.feed_idx])
                )
                recovery_active = (state.recovery_active | first_hit) & (~recovery_complete)
                phase_code = jnp.where(
                    (landing_recorded | static_target) & recovery_active,
                    STATE_RECOVERY,
                    phase_code,
                )
                recovery_done = has_hit & recovery_complete & flight_resolved
                terminated = miss | body_fall | recovery_done
            else:
                miss = landed & (phase_code == STATE_INCOMING)
                landing_fire = landed & (~miss) & (state.hit_rewarded | hit_this_step)
                landing_recorded = state.landing_recorded
                landing_xy = state.landing_xy
                recovery_step = state.recovery_step
                recovery_active = state.recovery_active
                recovery_complete = state.recovery_complete
                flight_resolved = state.flight_resolved
                terminated = landed | body_fall
            truncated = (~terminated) & (step_index >= self.max_episode_steps)
            done = terminated | truncated

            stringbed_pos = data.site_xpos[:, self._stringbed_site]
            intercept = self.intercept_points[state.feed_idx]
            dynamic_task = True if self.task_profile == LEGACY_PROFILE else state.v2_environment_mode == 3
            elapsed = step_index.astype(jnp.float32) * self.control_substeps * self.timestep
            time_to_intercept = self.intercept_times[state.feed_idx] - elapsed
            intercept_distance = jnp.linalg.norm(intercept - stringbed_pos, axis=-1)
            ball_racket_distance = jnp.linalg.norm(shuttle_contact_pos - stringbed_pos, axis=-1)
            approach = jnp.where(
                (phase_code == STATE_INCOMING) & dynamic_task,
                w["approach"] * jnp.exp(-2.0 * intercept_distance),
                0.0,
            )
            guidance_lower_bound = (
                -0.25
                if self.contact_guidance_reward_mode
                == "closest_approach_event_direction"
                else -0.08
            )
            guidance_active = (
                (phase_code == STATE_INCOMING)
                & dynamic_task
                & (time_to_intercept >= guidance_lower_bound)
                & (time_to_intercept <= 0.70)
            )
            time_gate = jnp.exp(-0.5 * jnp.square(time_to_intercept / 0.28))
            shuttle_proximity_potential = jnp.where(
                guidance_active,
                time_gate
                * jnp.exp(-0.5 * jnp.square(ball_racket_distance / self.shuttle_proximity_softness_m)),
                0.0,
            )
            timed_intercept_potential = jnp.where(
                guidance_active,
                time_gate
                * jnp.exp(-0.5 * jnp.square(intercept_distance / self.timed_intercept_softness_m)),
                0.0,
            )
            raw_face_normal = data.site_xmat[:, self._stringbed_site].reshape(-1, 3, 3)[:, :, 2]
            ball_side = jnp.sum(
                (shuttle_contact_pos - stringbed_pos) * raw_face_normal,
                axis=-1,
            )
            signed_face_normal = raw_face_normal * jnp.where(ball_side >= 0.0, 1.0, -1.0)[:, None]
            desired_return = jnp.stack(
                [
                    jnp.full_like(shuttle_pos[:, 0], -float(self.player_half_sign)),
                    -0.15 * shuttle_contact_pos[:, 1],
                    jnp.full_like(
                        shuttle_contact_pos[:, 2],
                        self.desired_return_up_component,
                    ),
                ],
                axis=-1,
            )
            desired_return = desired_return / jnp.maximum(
                jnp.linalg.norm(desired_return, axis=-1, keepdims=True), 1.0e-9
            )
            racket_cvel = data.cvel[:, self._racket_body]
            racket_offset = stringbed_pos - data.subtree_com[:, self._racket_root]
            racket_head_velocity = racket_cvel[:, 3:] + jnp.cross(racket_cvel[:, :3], racket_offset)
            normal_direction_signed_score = jnp.clip(jnp.sum(signed_face_normal * desired_return, axis=-1), -1.0, 1.0)
            velocity_direction_signed_score = jnp.clip(
                jnp.sum(racket_head_velocity * desired_return, axis=-1) / 8.0,
                -1.0,
                1.0,
            )
            normal_direction_score = jnp.maximum(
                normal_direction_signed_score,
                0.0,
            )
            velocity_direction_score = jnp.maximum(
                velocity_direction_signed_score,
                0.0,
            )
            if self.direction_reward_mode == "signed_projection":
                normal_direction_reward_score = normal_direction_signed_score
                velocity_direction_reward_score = velocity_direction_signed_score
            else:
                normal_direction_reward_score = normal_direction_score
                velocity_direction_reward_score = velocity_direction_score
            if self.racket_guidance_mode in {
                "counterfactual_rebound",
                "counterfactual_clearance_priority",
            }:
                counterfactual_closing_speed = -jnp.sum(
                    (shuttle_vel - racket_head_velocity) * signed_face_normal,
                    axis=-1,
                )
                counterfactual_closing_gate = jax.nn.sigmoid(
                    (counterfactual_closing_speed - float(self.params.min_speed_for_event)) / 1.0
                )
                counterfactual_velocity = jax.vmap(
                    lambda shuttle_v, racket_v, normal: event_rebound_velocity(
                        self.params,
                        shuttle_velocity=shuttle_v,
                        racket_surface_velocity=racket_v,
                        normal_world=normal,
                    )
                )(shuttle_vel, racket_head_velocity, signed_face_normal)
                counterfactual_speed = jnp.linalg.norm(
                    counterfactual_velocity,
                    axis=-1,
                )
                counterfactual_unit = counterfactual_velocity / jnp.maximum(
                    counterfactual_speed[:, None],
                    1.0e-9,
                )
                counterfactual_direction_signed_score = jnp.clip(
                    jnp.sum(counterfactual_unit * desired_return, axis=-1),
                    -1.0,
                    1.0,
                )
                if self.racket_guidance_mode == "counterfactual_clearance_priority":
                    counterfactual_direction_score = jnp.maximum(
                        counterfactual_direction_signed_score,
                        0.0,
                    )
                    counterfactual_direction_fraction = 0.30
                else:
                    counterfactual_direction_score = 0.5 * (counterfactual_direction_signed_score + 1.0)
                    counterfactual_direction_fraction = 0.65
                counterfactual_distance_to_net = jnp.maximum(
                    0.0,
                    float(self.player_half_sign) * (shuttle_pos[:, 0] - self.return_net_x_m),
                )
                counterfactual_forward_speed = -float(self.player_half_sign) * counterfactual_velocity[:, 0]
                counterfactual_forward_gate = jnp.clip(
                    counterfactual_forward_speed / 4.0,
                    0.0,
                    1.0,
                )
                counterfactual_time_to_net = jnp.clip(
                    counterfactual_distance_to_net / jnp.maximum(counterfactual_forward_speed, 0.25),
                    0.0,
                    1.5,
                )
                counterfactual_net_height = (
                    shuttle_pos[:, 2]
                    + counterfactual_velocity[:, 2] * counterfactual_time_to_net
                    - 0.5 * 9.81 * jnp.square(counterfactual_time_to_net)
                )
                counterfactual_predicted_clearance = counterfactual_net_height - self.return_net_height_m
                counterfactual_required_clearance = (
                    0.0 if self.min_return_net_clearance_m is None else self.min_return_net_clearance_m
                )
                counterfactual_clearance_score = counterfactual_forward_gate * jax.nn.sigmoid(
                    (counterfactual_predicted_clearance - counterfactual_required_clearance)
                    / self.ballistic_return_score_softness_m
                )
                counterfactual_rebound_score = counterfactual_closing_gate * (
                    counterfactual_direction_fraction * counterfactual_direction_score
                    + (1.0 - counterfactual_direction_fraction) * counterfactual_clearance_score
                )
            else:
                counterfactual_closing_speed = jnp.zeros_like(normal_direction_score)
                counterfactual_closing_gate = jnp.zeros_like(normal_direction_score)
                counterfactual_velocity = jnp.zeros_like(shuttle_vel)
                counterfactual_direction_signed_score = jnp.zeros_like(normal_direction_score)
                counterfactual_clearance_score = jnp.zeros_like(normal_direction_score)
                counterfactual_predicted_clearance = jnp.zeros_like(normal_direction_score)
                counterfactual_rebound_score = jnp.zeros_like(normal_direction_score)
            if self.racket_guidance_mode in {
                "inverse_impact_target",
                "inverse_impact_decomposed",
            }:
                inverse_target_outgoing_velocity = self.inverse_target_speed_m_s * desired_return
                inverse_required_delta = inverse_target_outgoing_velocity - shuttle_vel
                inverse_required_delta_norm = jnp.linalg.norm(
                    inverse_required_delta,
                    axis=-1,
                )
                inverse_target_normal = inverse_required_delta / jnp.maximum(
                    inverse_required_delta_norm[:, None],
                    1.0e-9,
                )
                inverse_target_closing_speed = inverse_required_delta_norm / (
                    1.0 + float(self.params.event_restitution_normal)
                )
                inverse_target_racket_velocity = (
                    shuttle_vel + inverse_target_closing_speed[:, None] * inverse_target_normal
                )
                inverse_signed_normal_alignment = jnp.clip(
                    jnp.sum(
                        signed_face_normal * inverse_target_normal,
                        axis=-1,
                    ),
                    -1.0,
                    1.0,
                )
                inverse_normal_alignment = jnp.maximum(
                    inverse_signed_normal_alignment,
                    0.0,
                )
                inverse_shifted_normal_score = 0.5 * (inverse_signed_normal_alignment + 1.0)
                inverse_racket_velocity_error = jnp.linalg.norm(
                    racket_head_velocity - inverse_target_racket_velocity,
                    axis=-1,
                )
                inverse_racket_velocity_score = 1.0 / (
                    1.0 + jnp.square(inverse_racket_velocity_error / self.inverse_velocity_softness_m_s)
                )
                inverse_impact_score = inverse_normal_alignment * inverse_racket_velocity_score
                inverse_decomposed_score = (
                    (1.0 - self.racket_velocity_direction_fraction) * inverse_shifted_normal_score
                    + self.racket_velocity_direction_fraction * inverse_racket_velocity_score
                )
            else:
                inverse_target_racket_velocity = jnp.zeros_like(shuttle_vel)
                inverse_target_closing_speed = jnp.zeros_like(normal_direction_score)
                inverse_signed_normal_alignment = jnp.zeros_like(normal_direction_score)
                inverse_shifted_normal_score = jnp.zeros_like(normal_direction_score)
                inverse_normal_alignment = jnp.zeros_like(normal_direction_score)
                inverse_racket_velocity_error = jnp.zeros_like(normal_direction_score)
                inverse_racket_velocity_score = jnp.zeros_like(normal_direction_score)
                inverse_impact_score = jnp.zeros_like(normal_direction_score)
                inverse_decomposed_score = jnp.zeros_like(normal_direction_score)

            # A fast event rebound can move the cork through the zero-thickness
            # stringbed during this control step.  Re-orienting the normal from
            # the post-step cork side then flips it and made the old hit metrics
            # report a correct contact normal as approximately -1.  Preserve
            # and score the exact pre-rebound state emitted by the physics
            # substep; fall back to the historical post-step values only for
            # non-event contact modes.
            event_metric_normal = jnp.where(
                rebound_this_step[:, None],
                event_contact_normal,
                signed_face_normal,
            )
            event_metric_shuttle_before = jnp.where(
                rebound_this_step[:, None],
                event_shuttle_velocity_before,
                shuttle_vel,
            )
            event_metric_shuttle_after = jnp.where(
                rebound_this_step[:, None],
                event_shuttle_velocity_after,
                shuttle_vel,
            )
            event_metric_racket_velocity = jnp.where(
                rebound_this_step[:, None],
                event_racket_surface_velocity,
                racket_head_velocity,
            )
            event_normal_direction_signed_score = jnp.clip(
                jnp.sum(event_metric_normal * desired_return, axis=-1),
                -1.0,
                1.0,
            )
            event_velocity_direction_signed_score = jnp.clip(
                jnp.sum(event_metric_racket_velocity * desired_return, axis=-1) / 8.0,
                -1.0,
                1.0,
            )
            if self.racket_guidance_mode in {
                "inverse_impact_target",
                "inverse_impact_decomposed",
            }:
                event_inverse_required_delta = (
                    self.inverse_target_speed_m_s * desired_return - event_metric_shuttle_before
                )
                event_inverse_required_delta_norm = jnp.linalg.norm(
                    event_inverse_required_delta,
                    axis=-1,
                )
                event_inverse_target_normal = event_inverse_required_delta / jnp.maximum(
                    event_inverse_required_delta_norm[:, None],
                    1.0e-9,
                )
                event_inverse_target_closing_speed = event_inverse_required_delta_norm / (
                    1.0 + float(self.params.event_restitution_normal)
                )
                event_inverse_target_racket_velocity = (
                    event_metric_shuttle_before
                    + event_inverse_target_closing_speed[:, None] * event_inverse_target_normal
                )
                event_inverse_signed_normal_alignment = jnp.clip(
                    jnp.sum(event_metric_normal * event_inverse_target_normal, axis=-1),
                    -1.0,
                    1.0,
                )
                event_inverse_normal_alignment = jnp.maximum(
                    event_inverse_signed_normal_alignment,
                    0.0,
                )
                event_inverse_shifted_normal_score = 0.5 * (
                    event_inverse_signed_normal_alignment + 1.0
                )
                event_inverse_racket_velocity_error = jnp.linalg.norm(
                    event_metric_racket_velocity - event_inverse_target_racket_velocity,
                    axis=-1,
                )
                event_inverse_racket_velocity_score = 1.0 / (
                    1.0
                    + jnp.square(
                        event_inverse_racket_velocity_error / self.inverse_velocity_softness_m_s
                    )
                )
                event_inverse_impact_score = (
                    event_inverse_normal_alignment * event_inverse_racket_velocity_score
                )
                event_inverse_decomposed_score = (
                    (1.0 - self.racket_velocity_direction_fraction)
                    * event_inverse_shifted_normal_score
                    + self.racket_velocity_direction_fraction
                    * event_inverse_racket_velocity_score
                )
            else:
                event_inverse_signed_normal_alignment = jnp.zeros_like(normal_direction_score)
                event_inverse_normal_alignment = jnp.zeros_like(normal_direction_score)
                event_inverse_racket_velocity_error = jnp.zeros_like(normal_direction_score)
                event_inverse_impact_score = jnp.zeros_like(normal_direction_score)
                event_inverse_decomposed_score = jnp.zeros_like(normal_direction_score)
            event_direction_reward_score = jnp.clip(
                2.0 * event_inverse_decomposed_score - 1.0,
                -1.0,
                1.0,
            )
            direction_guidance_active = guidance_active & (time_to_intercept <= 0.30)
            direction_gate = jnp.exp(-0.5 * jnp.square(time_to_intercept / 0.15)) * jnp.exp(
                -0.5 * jnp.square(ball_racket_distance / self.direction_distance_softness_m)
            )
            racket_direction_potential = jnp.where(
                direction_guidance_active,
                direction_gate
                * (
                    (
                        inverse_decomposed_score
                        if self.racket_guidance_mode == "inverse_impact_decomposed"
                        else inverse_impact_score
                    )
                    if self.racket_guidance_mode
                    in {
                        "inverse_impact_target",
                        "inverse_impact_decomposed",
                    }
                    else (
                        counterfactual_rebound_score
                        if self.racket_guidance_mode
                        in {
                            "counterfactual_rebound",
                            "counterfactual_clearance_priority",
                        }
                        else (
                            (1.0 - self.racket_velocity_direction_fraction) * normal_direction_reward_score
                            + self.racket_velocity_direction_fraction * velocity_direction_reward_score
                        )
                    )
                ),
                0.0,
            )
            if self.contact_guidance_reward_mode == "potential_event_direction":
                racket_direction_potential = jnp.where(
                    direction_guidance_active,
                    direction_gate * (2.0 * inverse_decomposed_score - 1.0),
                    0.0,
                )
            if (
                self.contact_guidance_reward_mode
                == "closest_approach_event_direction"
            ):
                closest_candidate = direction_guidance_active & (
                    ball_racket_distance < state.closest_racket_distance_m
                )
                closest_candidate_score = jnp.clip(
                    jnp.exp(
                        -0.5
                        * jnp.square(
                            ball_racket_distance
                            / self.direction_distance_softness_m
                        )
                    )
                    * (2.0 * inverse_decomposed_score - 1.0),
                    -1.0,
                    1.0,
                )
                closest_racket_distance_m = jnp.where(
                    closest_candidate,
                    ball_racket_distance,
                    state.closest_racket_distance_m,
                )
                closest_racket_direction_score = jnp.where(
                    closest_candidate,
                    closest_candidate_score,
                    state.closest_racket_direction_score,
                )
            else:
                closest_racket_distance_m = state.closest_racket_distance_m
                closest_racket_direction_score = (
                    state.closest_racket_direction_score
                )
            hit_bonus_fire = hit_this_step & (~state.hit_rewarded)
            closest_approach_terminal_fire = jnp.zeros_like(done)
            closest_approach_terminal_score = jnp.zeros_like(
                ball_racket_distance
            )
            if self.contact_guidance_reward_mode == "best_progress":
                best_shuttle_proximity_potential = jnp.maximum(
                    state.best_shuttle_proximity_potential,
                    shuttle_proximity_potential,
                )
                best_timed_intercept_potential = jnp.maximum(
                    state.best_timed_intercept_potential,
                    timed_intercept_potential,
                )
                best_racket_direction_potential = jnp.maximum(
                    state.best_racket_direction_potential,
                    racket_direction_potential,
                )
                shuttle_proximity = w["shuttle_proximity"] * (
                    best_shuttle_proximity_potential - state.best_shuttle_proximity_potential
                )
                timed_intercept = w["timed_intercept"] * (
                    best_timed_intercept_potential - state.best_timed_intercept_potential
                )
                racket_direction = w["racket_direction"] * (
                    best_racket_direction_potential - state.best_racket_direction_potential
                )
            elif self.contact_guidance_reward_mode == "event_direction":
                best_shuttle_proximity_potential = jnp.maximum(
                    state.best_shuttle_proximity_potential,
                    shuttle_proximity_potential,
                )
                best_timed_intercept_potential = jnp.maximum(
                    state.best_timed_intercept_potential,
                    timed_intercept_potential,
                )
                best_racket_direction_potential = state.best_racket_direction_potential
                shuttle_proximity = w["shuttle_proximity"] * (
                    best_shuttle_proximity_potential - state.best_shuttle_proximity_potential
                )
                timed_intercept = w["timed_intercept"] * (
                    best_timed_intercept_potential - state.best_timed_intercept_potential
                )
                racket_direction = jnp.where(
                    hit_bonus_fire & dynamic_task,
                    w["racket_direction"] * event_direction_reward_score,
                    0.0,
                )
            elif (
                self.contact_guidance_reward_mode
                == "closest_approach_event_direction"
            ):
                best_shuttle_proximity_potential = jnp.maximum(
                    state.best_shuttle_proximity_potential,
                    shuttle_proximity_potential,
                )
                best_timed_intercept_potential = jnp.maximum(
                    state.best_timed_intercept_potential,
                    timed_intercept_potential,
                )
                best_racket_direction_potential = (
                    state.best_racket_direction_potential
                )
                shuttle_proximity = w["shuttle_proximity"] * (
                    best_shuttle_proximity_potential
                    - state.best_shuttle_proximity_potential
                )
                timed_intercept = w["timed_intercept"] * (
                    best_timed_intercept_potential
                    - state.best_timed_intercept_potential
                )
                closest_event_fire = hit_bonus_fire & dynamic_task
                closest_approach_terminal_fire = (
                    done
                    & dynamic_task
                    & (~state.hit_rewarded)
                    & (~closest_event_fire)
                )
                closest_approach_terminal_score = jnp.where(
                    closest_approach_terminal_fire,
                    closest_racket_direction_score,
                    0.0,
                )
                racket_direction = jnp.where(
                    closest_event_fire,
                    w["racket_direction"] * event_direction_reward_score,
                    jnp.where(
                        closest_approach_terminal_fire,
                        w["racket_direction"]
                        * closest_approach_terminal_score,
                        0.0,
                    ),
                )
            elif self.contact_guidance_reward_mode == "potential_event_direction":
                best_shuttle_proximity_potential = jnp.maximum(
                    state.best_shuttle_proximity_potential,
                    shuttle_proximity_potential,
                )
                best_timed_intercept_potential = jnp.maximum(
                    state.best_timed_intercept_potential,
                    timed_intercept_potential,
                )
                event_direction_fire = hit_bonus_fire & dynamic_task
                terminal_without_event = (
                    done
                    & (~state.hit_rewarded)
                    & (~event_direction_fire)
                )
                best_racket_direction_potential = jnp.where(
                    event_direction_fire | terminal_without_event,
                    0.0,
                    racket_direction_potential,
                )
                shuttle_proximity = w["shuttle_proximity"] * (
                    best_shuttle_proximity_potential - state.best_shuttle_proximity_potential
                )
                timed_intercept = w["timed_intercept"] * (
                    best_timed_intercept_potential - state.best_timed_intercept_potential
                )
                direction_increment = jnp.where(
                    event_direction_fire,
                    event_direction_reward_score - state.best_racket_direction_potential,
                    jnp.where(
                        terminal_without_event,
                        -state.best_racket_direction_potential,
                        self.contact_guidance_discount * racket_direction_potential
                        - state.best_racket_direction_potential,
                    ),
                )
                racket_direction = w["racket_direction"] * direction_increment
            else:
                best_shuttle_proximity_potential = state.best_shuttle_proximity_potential
                best_timed_intercept_potential = state.best_timed_intercept_potential
                best_racket_direction_potential = state.best_racket_direction_potential
                shuttle_proximity = w["shuttle_proximity"] * shuttle_proximity_potential
                timed_intercept = w["timed_intercept"] * timed_intercept_potential
                racket_direction = w["racket_direction"] * racket_direction_potential
            hit_bonus = jnp.where(
                hit_bonus_fire & dynamic_task,
                w["hit_bonus"],
                0.0,
            )
            hit_speed = jnp.where(
                hit_bonus_fire & dynamic_task,
                w["hit_speed"] * jnp.minimum(1.0, hit_closing_speed / 8.0),
                0.0,
            )
            outgoing_speed = jnp.linalg.norm(shuttle_vel, axis=-1)
            outgoing_unit = shuttle_vel / jnp.maximum(outgoing_speed[:, None], 1.0e-9)
            return_direction_signed_score = jnp.clip(jnp.sum(outgoing_unit * desired_return, axis=-1), -1.0, 1.0)
            return_direction_score = jnp.maximum(
                return_direction_signed_score,
                0.0,
            )
            return_direction_reward_score = (
                return_direction_signed_score
                if self.direction_reward_mode == "signed_projection"
                else return_direction_score
            )
            return_direction = jnp.where(
                hit_bonus_fire & dynamic_task,
                w["return_direction"] * return_direction_reward_score * jnp.minimum(1.0, outgoing_speed / 10.0),
                0.0,
            )
            distance_to_net = jnp.maximum(
                0.0,
                float(self.player_half_sign) * (shuttle_pos[:, 0] - self.return_net_x_m),
            )
            forward_speed = -float(self.player_half_sign) * shuttle_vel[:, 0]
            forward_gate = jnp.clip(forward_speed / 4.0, 0.0, 1.0)
            predicted_time_to_net = jnp.clip(
                distance_to_net / jnp.maximum(forward_speed, 0.25),
                0.0,
                1.5,
            )
            predicted_net_height = (
                shuttle_pos[:, 2]
                + shuttle_vel[:, 2] * predicted_time_to_net
                - 0.5 * 9.81 * jnp.square(predicted_time_to_net)
            )
            predicted_net_clearance = predicted_net_height - self.return_net_height_m
            required_clearance = 0.0 if self.min_return_net_clearance_m is None else self.min_return_net_clearance_m
            return_clearance_score = forward_gate * jax.nn.sigmoid(
                (predicted_net_clearance - required_clearance) / self.ballistic_return_score_softness_m
            )
            clearance_reward_score = (
                2.0 * return_clearance_score - 1.0
                if self.clearance_reward_mode == "signed_centered"
                else return_clearance_score
            )
            return_clearance = jnp.where(
                hit_bonus_fire & dynamic_task,
                w["return_clearance"] * clearance_reward_score,
                0.0,
            )
            hit_rewarded = state.hit_rewarded | hit_this_step
            crossed_fire = (
                (phase_code >= STATE_FLIGHT)
                & hit_rewarded
                & valid_return_cross
                & (~state.crossed_rewarded)
                & dynamic_task
            )
            crossed_bonus = jnp.where(crossed_fire, w["crossed_net"], 0.0)
            crossed_rewarded = state.crossed_rewarded | crossed_fire
            invalid_cross_penalty = jnp.where(
                invalid_return_cross & (~state.invalid_net_crossed) & dynamic_task,
                -w["invalid_net_crossing"],
                0.0,
            )

            raw_landing_score = self._landing_score(shuttle_pos[:, :2])
            landing_score = jnp.where(
                crossed_rewarded,
                raw_landing_score,
                -jnp.ones_like(raw_landing_score),
            )
            landing_term = jnp.where(
                landing_fire & dynamic_task,
                w["landing_region"] * landing_score,
                0.0,
            )
            effort = -w["effort"] * jnp.mean(jnp.square(composed), axis=-1)
            residual_term = (
                -w.get("residual", 0.0) * jnp.mean(jnp.square(action), axis=-1)
                if self._base is not None and w.get("residual", 0.0) != 0.0
                else 0.0
            )
            posture = -w["posture"] * jnp.maximum(0.0, 0.85 - root_z)
            miss_term = jnp.where(miss & dynamic_task, -w["miss"], 0.0)
            fall_term = jnp.where(body_fall, -w["body_fall"], 0.0)

            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                target_time = self.target_impact_time[state.feed_idx]
                impact_position_error = jnp.linalg.norm(
                    best_contact_position - self.target_impact_position[state.feed_idx],
                    axis=-1,
                )
                impact_timing_error = jnp.abs(
                    step_index.astype(jnp.float32) * self.control_substeps * self.timestep - target_time
                )
                normal_dot = jnp.clip(
                    jnp.sum(
                        best_contact_normal * self.target_normal[state.feed_idx],
                        axis=-1,
                    ),
                    -1.0,
                    1.0,
                )
                normal_error = jnp.arccos(normal_dot)
                linear_error = jnp.sqrt(
                    jnp.mean(
                        jnp.square(racket_head_velocity - self.target_linear_velocity[state.feed_idx]),
                        axis=-1,
                    )
                )
                angular_error = jnp.sqrt(
                    jnp.mean(
                        jnp.square(racket_cvel[:, :3] - self.target_angular_velocity[state.feed_idx]),
                        axis=-1,
                    )
                )
                impact_center_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_center"]
                    * state.v2_reward_mask[v2_reward_index["impact_center"]]
                    * jnp.exp(-2.0 * best_rho2),
                    0.0,
                )
                impact_position_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_position"]
                    * state.v2_reward_mask[v2_reward_index["impact_position"]]
                    * jnp.exp(-0.5 * jnp.square(impact_position_error / 0.12)),
                    0.0,
                )
                impact_time_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_time"]
                    * state.v2_reward_mask[v2_reward_index["impact_time"]]
                    * jnp.exp(-0.5 * jnp.square(impact_timing_error / 0.08)),
                    0.0,
                )
                impact_normal_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_normal"]
                    * state.v2_reward_mask[v2_reward_index["impact_normal"]]
                    * jnp.exp(-2.0 * jnp.square(normal_error)),
                    0.0,
                )
                impact_linear_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_linear_velocity"]
                    * state.v2_reward_mask[v2_reward_index["impact_linear_velocity"]]
                    * jnp.exp(-0.25 * jnp.square(linear_error)),
                    0.0,
                )
                impact_angular_term = jnp.where(
                    hit_bonus_fire,
                    w["impact_angular_velocity"]
                    * state.v2_reward_mask[v2_reward_index["impact_angular_velocity"]]
                    * jnp.exp(-0.02 * jnp.square(angular_error)),
                    0.0,
                )
                landing_error = jnp.linalg.norm(
                    shuttle_pos[:, :2] - self.target_landing_xy[state.feed_idx],
                    axis=-1,
                )
                apex_error = jnp.abs(apex_height - self.target_apex_height[state.feed_idx])
                precise_landing_term = jnp.where(
                    landing_fire,
                    w["precise_landing"]
                    * state.v2_reward_mask[v2_reward_index["precise_landing"]]
                    * jnp.exp(-0.5 * jnp.square(landing_error / 0.75)),
                    0.0,
                )
                apex_term = jnp.where(
                    landing_fire,
                    w["apex"]
                    * state.v2_reward_mask[v2_reward_index["apex"]]
                    * jnp.exp(-0.5 * jnp.square(apex_error / 0.35)),
                    0.0,
                )
                ready_error = jnp.sqrt(
                    jnp.mean(
                        jnp.square(data.qpos[:, self._ready_qpos_index] - self._ready_qpos),
                        axis=-1,
                    )
                )
                root_speed = jnp.linalg.norm(data.qvel[:, self._root_dadr : self._root_dadr + 6], axis=-1)
                recovery_racket_speed = jnp.linalg.norm(racket_head_velocity, axis=-1) + 0.15 * jnp.linalg.norm(
                    racket_cvel[:, :3], axis=-1
                )
                recovery_reward_active = recovery_active | (state.v2_environment_mode == 0)
                recovery_ready_term = jnp.where(
                    recovery_reward_active,
                    w["recovery_ready"]
                    * state.v2_reward_mask[v2_reward_index["recovery_ready"]]
                    * jnp.exp(-8.0 * jnp.square(ready_error)),
                    0.0,
                )
                recovery_balance_term = jnp.where(
                    recovery_reward_active,
                    w["recovery_balance"]
                    * state.v2_reward_mask[v2_reward_index["recovery_balance"]]
                    * jnp.exp(-2.0 * jnp.square(root_speed)),
                    0.0,
                )
                recovery_deceleration_term = jnp.where(
                    recovery_reward_active,
                    w["recovery_deceleration"]
                    * state.v2_reward_mask[v2_reward_index["recovery_deceleration"]]
                    * jnp.exp(-0.5 * jnp.square(recovery_racket_speed)),
                    0.0,
                )
                v2_reward = (
                    impact_position_term
                    + impact_center_term
                    + impact_time_term
                    + impact_normal_term
                    + impact_linear_term
                    + impact_angular_term
                    + precise_landing_term
                    + apex_term
                    + recovery_ready_term
                    + recovery_balance_term
                    + recovery_deceleration_term
                )
            else:
                v2_reward = 0.0

            reward = (
                approach
                + shuttle_proximity
                + timed_intercept
                + racket_direction
                + hit_bonus
                + hit_speed
                + return_direction
                + return_clearance
                + crossed_bonus
                + invalid_cross_penalty
                + landing_term
                + effort
                + residual_term
                + posture
                + miss_term
                + fall_term
                + v2_reward
            )

            obs = self._observation(
                data,
                state.feed_idx,
                step_index,
                phase_code=phase_code,
                recovery_step=recovery_step,
                recovery_active=recovery_active,
                apex_height=apex_height,
            )
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            racket_head_speed = jnp.linalg.norm(racket_head_velocity, axis=-1)
            muscle_power_abs_mean = jnp.mean(jnp.abs(data.actuator_force * data.actuator_velocity), axis=-1)
            normalized_control_energy = jnp.mean(jnp.square(composed), axis=-1)

            # auto-reset done envs
            key, sub = jax.random.split(state.key)
            new_feed = jax.random.randint(sub, (num_envs,), 0, state.active_feed_count)
            done_col = done[:, None]
            qpos = jnp.where(done_col, self.qpos_bank[new_feed], data.qpos)
            qvel = jnp.where(done_col, self.qvel_bank[new_feed], data.qvel)
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                reset_static = state.v2_environment_mode != 3
                reset_position = self.target_impact_position[new_feed]
                reset_position = reset_position.at[:, 2].set(
                    jnp.where(
                        state.v2_environment_mode <= 1,
                        100.0,
                        reset_position[:, 2],
                    )
                )
                reset_shuttle_position = jnp.where(
                    reset_static, reset_position, self.qpos_bank[new_feed, self._shuttle_qadr : self._shuttle_qadr + 3]
                )
                qpos = qpos.at[:, self._shuttle_qadr : self._shuttle_qadr + 3].set(
                    jnp.where(done_col, reset_shuttle_position, qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3])
                )
                reset_shuttle_velocity = jnp.where(
                    reset_static,
                    jnp.zeros((num_envs, 6), dtype=qvel.dtype),
                    self.qvel_bank[new_feed, self._shuttle_dadr : self._shuttle_dadr + 6],
                )
                qvel = qvel.at[:, self._shuttle_dadr : self._shuttle_dadr + 6].set(
                    jnp.where(done_col, reset_shuttle_velocity, qvel[:, self._shuttle_dadr : self._shuttle_dadr + 6])
                )
            act = jnp.where(done_col, 0.0, data.act)
            ctrl_next = jnp.where(done_col, 0.0, data.ctrl)
            data = data.replace(qpos=qpos, qvel=qvel, act=act, ctrl=ctrl_next)
            # refresh derived fields (sites/cvel) so the first substep of a
            # freshly reset env does not see stale kinematics
            data = forward_all(data)

            zeros_i = jnp.zeros((num_envs,), jnp.int32)
            next_feed_idx = jnp.where(done, new_feed, state.feed_idx)
            next_step_index = jnp.where(done, zeros_i, step_index)
            next_lab_state = (
                self.lab_state_builder.build_jax(
                    data=data,
                    phase=self._swing_phase(
                        next_step_index,
                        self.intercept_times[next_feed_idx],
                    ),
                )
                if self.lab_state_builder is not None
                else state.lab_state
            )
            reset_observation = (
                self._observation(
                    data,
                    new_feed,
                    zeros_i,
                    phase_code=zeros_i,
                    recovery_step=zeros_i,
                    recovery_active=jnp.zeros((num_envs,), bool),
                    apex_height=data.qpos[:, self._shuttle_qadr + 2],
                )
                if self.task_profile == IMPACT_RECOVERY_PROFILE
                else self.obs_bank[new_feed]
            )
            next_state = EnvState(
                data=data,
                obs=jnp.where(done_col, reset_observation, obs),
                cooldown=jnp.where(done, zeros_i, cooldown),
                step_index=next_step_index,
                phase_code=jnp.where(done, zeros_i, phase_code),
                hit_rewarded=jnp.where(done, False, hit_rewarded),
                crossed_rewarded=jnp.where(done, False, crossed_rewarded),
                invalid_net_crossed=jnp.where(done, False, invalid_net_crossed),
                hit_closing_speed=jnp.where(done, 0.0, hit_closing_speed),
                best_shuttle_proximity_potential=jnp.where(
                    done, 0.0, best_shuttle_proximity_potential
                ),
                best_timed_intercept_potential=jnp.where(
                    done, 0.0, best_timed_intercept_potential
                ),
                best_racket_direction_potential=jnp.where(
                    done, 0.0, best_racket_direction_potential
                ),
                closest_racket_distance_m=jnp.where(
                    done, jnp.inf, closest_racket_distance_m
                ),
                closest_racket_direction_score=jnp.where(
                    done, 0.0, closest_racket_direction_score
                ),
                landing_recorded=jnp.where(done, False, landing_recorded),
                landing_xy=jnp.where(done_col, jnp.zeros_like(landing_xy), landing_xy),
                apex_height=jnp.where(
                    done,
                    self.qpos_bank[new_feed, self._shuttle_qadr + 2],
                    apex_height,
                ),
                recovery_step=jnp.where(done, zeros_i, recovery_step),
                recovery_active=jnp.where(done, False, recovery_active),
                recovery_complete=jnp.where(done, False, recovery_complete),
                flight_resolved=jnp.where(
                    done,
                    state.v2_environment_mode != 3,
                    flight_resolved,
                ),
                feed_idx=next_feed_idx,
                lab_state=next_lab_state,
                lambda_lab=state.lambda_lab,
                active_feed_count=state.active_feed_count,
                residual_authority_progress=state.residual_authority_progress,
                v2_stage_index=state.v2_stage_index,
                v2_environment_mode=state.v2_environment_mode,
                v2_reward_mask=state.v2_reward_mask,
                key=key,
            )
            transition = {
                "obs": state.obs,
                "next_obs": obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "hit": hit_rewarded,
                "crossed_net": crossed_rewarded,
                "invalid_net_crossed": invalid_net_crossed,
                "landing_score": jnp.where(landing_fire, landing_score, 0.0),
                "miss": miss,
                "body_fall": body_fall,
                "hit_event": hit_bonus_fire,
                "stringbed_contact_event": contact_this_step,
                "event_rebound_event": rebound_this_step,
                "rewarded_hit_was_event_rebound": hit_bonus_fire & rebound_this_step,
                "landing_event": landing_fire,
            }
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                transition.update(
                    {
                        "impact_position_error_m": jnp.where(hit_bonus_fire, impact_position_error, 0.0),
                        "impact_rho2": jnp.where(hit_bonus_fire, best_rho2, 0.0),
                        "impact_timing_error_s": jnp.where(hit_bonus_fire, impact_timing_error, 0.0),
                        "stringbed_normal_error_rad": jnp.where(hit_bonus_fire, normal_error, 0.0),
                        "racket_linear_velocity_error_m_s": jnp.where(hit_bonus_fire, linear_error, 0.0),
                        "racket_angular_velocity_error_rad_s": jnp.where(hit_bonus_fire, angular_error, 0.0),
                        "landing_error_m": jnp.where(landing_fire, landing_error, 0.0),
                        "apex_error_m": jnp.where(landing_fire, apex_error, 0.0),
                        "ready_pose_error": jnp.where(
                            recovery_reward_active | recovery_done,
                            ready_error,
                            0.0,
                        ),
                        "recovery_metric_event": recovery_reward_active | recovery_done,
                        "recovery_progress": recovery_step.astype(jnp.float32)
                        / jnp.maximum(
                            self.target_recovery_horizon[state.feed_idx].astype(jnp.float32),
                            1.0,
                        ),
                        "recovery_complete": recovery_done,
                        "flight_resolved": flight_resolved,
                        "task_curriculum_stage_index": jnp.broadcast_to(state.v2_stage_index, (num_envs,)),
                    }
                )
            # Metrics shared by the pure 354-D and LAB policies use the same
            # normalized full-muscle action.  Latent/OOD diagnostics remain
            # LAB-only below; the direct branch must never synthesize them.
            body_action = composed if lab_output is None else lab_output.body_action
            transition.update(
                {
                    "control_finite": jnp.all(jnp.isfinite(composed), axis=-1).astype(jnp.float32),
                    "body_action_rms": jnp.sqrt(jnp.mean(jnp.square(body_action), axis=-1)),
                    "racket_head_speed_m_s": racket_head_speed,
                    "ball_racket_distance_m": ball_racket_distance,
                    "intercept_position_error_m": intercept_distance,
                    "time_to_intercept_s": time_to_intercept,
                    "shuttle_proximity_reward": shuttle_proximity,
                    "timed_intercept_reward": timed_intercept,
                    "racket_direction_reward": racket_direction,
                    "shuttle_proximity_potential": shuttle_proximity_potential,
                    "timed_intercept_potential": timed_intercept_potential,
                    "racket_direction_potential": racket_direction_potential,
                    "closest_approach_distance_m": jnp.where(
                        jnp.isfinite(closest_racket_distance_m),
                        closest_racket_distance_m,
                        0.0,
                    ),
                    "closest_approach_direction_score": (
                        closest_racket_direction_score
                    ),
                    "closest_approach_terminal_direction_score": (
                        closest_approach_terminal_score
                    ),
                    "closest_approach_terminal_event": (
                        closest_approach_terminal_fire
                    ),
                    "racket_direction_score": normal_direction_score,
                    "racket_direction_signed_score": normal_direction_signed_score,
                    "racket_velocity_direction_score": velocity_direction_score,
                    "racket_velocity_direction_signed_score": (velocity_direction_signed_score),
                    "counterfactual_rebound_score": counterfactual_rebound_score,
                    "counterfactual_rebound_closing_speed_m_s": (counterfactual_closing_speed),
                    "counterfactual_rebound_closing_gate": (counterfactual_closing_gate),
                    "counterfactual_rebound_direction_signed_score": (counterfactual_direction_signed_score),
                    "counterfactual_rebound_clearance_score": (counterfactual_clearance_score),
                    "counterfactual_rebound_predicted_clearance_m": (counterfactual_predicted_clearance),
                    "counterfactual_rebound_velocity_x_m_s": (counterfactual_velocity[:, 0]),
                    "counterfactual_rebound_velocity_y_m_s": (counterfactual_velocity[:, 1]),
                    "counterfactual_rebound_velocity_z_m_s": (counterfactual_velocity[:, 2]),
                    "inverse_impact_score": inverse_impact_score,
                    "inverse_impact_decomposed_score": inverse_decomposed_score,
                    "inverse_impact_signed_normal_alignment": (inverse_signed_normal_alignment),
                    "inverse_impact_shifted_normal_score": (inverse_shifted_normal_score),
                    "inverse_impact_normal_alignment": (inverse_normal_alignment),
                    "inverse_impact_racket_velocity_score": (inverse_racket_velocity_score),
                    "inverse_impact_racket_velocity_error_m_s": (inverse_racket_velocity_error),
                    "inverse_impact_target_closing_speed_m_s": (inverse_target_closing_speed),
                    "inverse_impact_target_racket_velocity_x_m_s": (inverse_target_racket_velocity[:, 0]),
                    "inverse_impact_target_racket_velocity_y_m_s": (inverse_target_racket_velocity[:, 1]),
                    "inverse_impact_target_racket_velocity_z_m_s": (inverse_target_racket_velocity[:, 2]),
                    "return_direction_reward": return_direction,
                    "return_direction_score": jnp.where(
                        hit_bonus_fire,
                        return_direction_score,
                        0.0,
                    ),
                    "return_direction_signed_score": jnp.where(
                        hit_bonus_fire,
                        return_direction_signed_score,
                        0.0,
                    ),
                    "return_clearance_reward": return_clearance,
                    "miss_penalty_reward": miss_term,
                    "return_clearance_score": jnp.where(
                        hit_bonus_fire,
                        return_clearance_score,
                        0.0,
                    ),
                    "predicted_net_clearance_m": jnp.where(
                        hit_bonus_fire,
                        predicted_net_clearance,
                        0.0,
                    ),
                    "hit_contact_speed_m_s": jnp.where(
                        hit_bonus_fire,
                        hit_closing_speed,
                        0.0,
                    ),
                    "hit_racket_head_speed_m_s": jnp.where(
                        hit_bonus_fire,
                        jnp.linalg.norm(event_metric_racket_velocity, axis=-1),
                        0.0,
                    ),
                    "hit_racket_direction_signed_score": jnp.where(
                        hit_bonus_fire,
                        event_normal_direction_signed_score,
                        0.0,
                    ),
                    "hit_racket_velocity_direction_signed_score": jnp.where(
                        hit_bonus_fire,
                        event_velocity_direction_signed_score,
                        0.0,
                    ),
                    "hit_event_direction_reward_score": jnp.where(
                        hit_bonus_fire,
                        event_direction_reward_score,
                        0.0,
                    ),
                    "hit_inverse_impact_score": jnp.where(
                        hit_bonus_fire,
                        event_inverse_impact_score,
                        0.0,
                    ),
                    "hit_inverse_impact_decomposed_score": jnp.where(
                        hit_bonus_fire,
                        event_inverse_decomposed_score,
                        0.0,
                    ),
                    "hit_inverse_impact_signed_normal_alignment": jnp.where(
                        hit_bonus_fire,
                        event_inverse_signed_normal_alignment,
                        0.0,
                    ),
                    "hit_inverse_impact_normal_alignment": jnp.where(
                        hit_bonus_fire,
                        event_inverse_normal_alignment,
                        0.0,
                    ),
                    "hit_inverse_impact_racket_velocity_error_m_s": jnp.where(
                        hit_bonus_fire,
                        event_inverse_racket_velocity_error,
                        0.0,
                    ),
                    "hit_outgoing_velocity_x_m_s": jnp.where(
                        hit_bonus_fire,
                        event_metric_shuttle_after[:, 0],
                        0.0,
                    ),
                    "hit_outgoing_velocity_y_m_s": jnp.where(
                        hit_bonus_fire,
                        event_metric_shuttle_after[:, 1],
                        0.0,
                    ),
                    "hit_outgoing_velocity_z_m_s": jnp.where(
                        hit_bonus_fire,
                        event_metric_shuttle_after[:, 2],
                        0.0,
                    ),
                    "muscle_power_abs_mean": muscle_power_abs_mean,
                    "normalized_control_energy": normalized_control_energy,
                    "body_action_saturation_fraction": jnp.mean(
                        (jnp.abs(body_action) > 0.98).astype(jnp.float32),
                        axis=-1,
                    ),
                    "full_action_saturation_fraction": jnp.mean(
                        (jnp.abs(composed) > 0.98).astype(jnp.float32),
                        axis=-1,
                    ),
                    "net_clearance_m": jnp.where(
                        crossed_fire,
                        net_crossing_height - self.return_net_height_m,
                        0.0,
                    ),
                    "valid_net_cross_event": crossed_fire,
                    "invalid_net_cross_event": (invalid_return_cross & (~state.invalid_net_crossed)),
                    "opponent_back_landing": landing_fire & (landing_score == 1.0),
                }
            )
            if self._base is not None and int(self._base["residual_override_indices"].shape[0]) > 0:
                override_ids = self._base["residual_override_indices"]
                transition.update(
                    {
                        "residual_override_action_rms": jnp.sqrt(
                            jnp.mean(jnp.square(action[:, override_ids]), axis=-1)
                        ),
                        "residual_override_composed_saturation_fraction": jnp.mean(
                            (jnp.abs(composed[:, override_ids]) > 0.98).astype(jnp.float32),
                            axis=-1,
                        ),
                        "residual_authority_progress": jnp.broadcast_to(
                            state.residual_authority_progress,
                            (num_envs,),
                        ),
                    }
                )
            if lab_output is not None:
                unclipped_state_z = (state.lab_state - lab_norm_mean) / lab_norm_std
                right_grip_action_rms = (
                    jnp.zeros(lab_output.body_action.shape[:-1], dtype=lab_output.body_action.dtype)
                    if lab_output.right_grip_action.shape[-1] == 0
                    else jnp.sqrt(jnp.mean(jnp.square(lab_output.right_grip_action), axis=-1))
                )
                transition.update(
                    {
                        "raw_latent_rms": jnp.sqrt(jnp.mean(jnp.square(lab_output.raw_latent), axis=-1)),
                        "raw_latent_saturation": jnp.mean(
                            (jnp.abs(lab_output.raw_latent) > 2.0).astype(jnp.float32),
                            axis=-1,
                        ),
                        "latent_norm": jnp.linalg.norm(lab_output.latent, axis=-1),
                        "prior_sigma_mean": jnp.mean(lab_output.prior_sigma, axis=-1),
                        "lab_state_unclipped_z_rms": jnp.sqrt(jnp.mean(jnp.square(unclipped_state_z), axis=-1)),
                        "lab_state_ood_fraction": jnp.mean(
                            (jnp.abs(unclipped_state_z) > 5.0).astype(jnp.float32),
                            axis=-1,
                        ),
                        "right_grip_action_rms": right_grip_action_rms,
                        "lambda_lab": jnp.broadcast_to(state.lambda_lab, (num_envs,)),
                        "active_feed_count": jnp.broadcast_to(state.active_feed_count.astype(jnp.float32), (num_envs,)),
                    }
                )
                if lab_output.raw_bounded_residual is not None:
                    transition["bounded_residual_rms"] = jnp.sqrt(
                        jnp.mean(jnp.square(lab_output.raw_bounded_residual), axis=-1)
                    )
            return next_state, transition

        return step


def _feed_difficulty(feed: FeedSample) -> float:
    """Stable easy-to-hard order for Stage-3 curriculum prefixes."""
    point = np.asarray(feed.intercept_point, dtype=float)
    center_penalty = abs(float(point[1])) + 0.6 * abs(float(point[2]) - 1.8)
    timing_penalty = 0.25 * abs(float(feed.intercept_time_s) - 0.75)
    speed_penalty = 0.02 * float(np.linalg.norm(feed.intercept_velocity))
    return center_penalty + timing_penalty + speed_penalty
