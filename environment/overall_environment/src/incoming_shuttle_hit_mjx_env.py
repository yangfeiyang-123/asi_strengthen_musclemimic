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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics import BadmintonPhysicsConfig
from environment.overall_environment.src.badminton_physics_mjx import (
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
    hit_closing_speed: jnp.ndarray  # (N,) float32
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
    swing_duration_s: float = 1.2
    contact_phase: float = 0.55
    base_skill: str | None = None
    lab_controller: Any | None = None
    lab_state_builder: Any | None = None
    curriculum: Any | None = None
    filter_finger_observation: bool | None = None
    feed_bank_manifest: dict[str, Any] | None = None
    task_profile: str = LEGACY_PROFILE
    impact_target_bank: Any | None = None
    recovery_horizon_steps: int = 60
    task_curriculum_max_stage: str | None = None

    def __post_init__(self) -> None:
        self.task_profile = str(self.task_profile)
        self.weights = _validate_reward_weights(dict(self.reward_weights), task_profile=self.task_profile)
        if self.task_curriculum_max_stage is not None:
            if self.task_profile != IMPACT_RECOVERY_PROFILE:
                raise ValueError("task_curriculum_max_stage is only valid for impact_recovery_v2")
            task_stage_by_name(self.task_curriculum_max_stage)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        if self.lab_controller is not None and self.base_policy_artifact is not None:
            raise ValueError("LAB and legacy full-action residual modes are mutually exclusive")
        if (self.lab_controller is None) != (self.lab_state_builder is None):
            raise ValueError("lab_controller and lab_state_builder must be provided together")
        self.filter_finger_observation = bool(
            self.lab_controller is not None
            if self.filter_finger_observation is None
            else self.filter_finger_observation
        )
        if self.curriculum is not None:
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
                "mode": "difficulty_sorted" if self.curriculum is not None else "stored",
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
        if self.base_policy_artifact is not None:
            self._init_base_policy()

    @property
    def expects_raw_latent(self) -> bool:
        return self.lab_controller is not None

    @property
    def control_manifest(self) -> dict[str, Any]:
        if self.lab_controller is None and self.task_profile == LEGACY_PROFILE:
            return {"schema_version": "incoming_hit_direct_action_v1"}
        if self._control_manifest_cache is not None:
            return self._control_manifest_cache
        if self.lab_controller is None:
            payload: dict[str, Any] = {"schema_version": "incoming_hit_direct_action_impact_recovery_v2"}
        else:
            from environment.overall_environment.src.stage3_lab import (
                stage3_attachment_report,
            )

            payload = dict(self.lab_controller.control_manifest)
            payload["lab_state_schema_hash"] = self.lab_state_builder.schema_hash
            payload["racket_attachment"] = stage3_attachment_report(self.model, self.xml)
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
            "reward_weights": self.weights,
            "player_half_sign": self.player_half_sign,
            "singles": self.singles,
            "terminate_on_body_fall": True,
            "swing_duration_s": self.swing_duration_s,
            "contact_phase": self.contact_phase,
        }
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

        from environment.overall_environment.src.base_swing_bridge import BaseSwingBridge

        bridge = BaseSwingBridge(
            self.base_policy_artifact,
            self.model,
            residual_scale=self.residual_scale,
            skill=self.base_skill,
        )
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

        actuator_ids = []
        for name in schema.actuator_names:
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"hitting scene missing base-policy actuator {name!r}")
            actuator_ids.append(aid)
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
            "sensor_adr": jnp.asarray(sensor_adr, dtype=jnp.int32),
            "target_ids": target_ids,
            "skill_onehot": jnp.asarray(tensors["skill_onehot"]),
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
        muscle = jnp.stack(
            [
                data.actuator_length[:, aid],
                data.actuator_velocity[:, aid],
                data.actuator_force[:, aid],
                data.ctrl[:, aid],
                data.act[:, aid],
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
        start = intercept_time - self.contact_phase * self.swing_duration_s
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
        return jnp.clip(base_full + self.residual_scale * action, -1.0, 1.0), None

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
                hit_closing_speed=jnp.zeros((num_envs,), jnp.float32),
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
                d, cd, hit, closing = carry
                d, cd, diag = substep(d, cd)
                contact_closing = jnp.maximum(0.0, -diag["relative_normal_velocity"]).astype(jnp.float32)
                sub_hit = diag["event_rebound_used"] | (diag["stringbed_active"] & (contact_closing > 0.0))
                closing = jnp.where(sub_hit, jnp.maximum(closing, contact_closing), closing)
                return (d, cd, hit | sub_hit, closing), None

            if self.task_profile == IMPACT_RECOVERY_PROFILE:

                def sub_body_v2(carry, _):
                    d, cd, hit, closing, best_rho2, best_normal, best_position = carry
                    d, cd, diag = substep(d, cd)
                    contact_closing = jnp.maximum(0.0, -diag["relative_normal_velocity"]).astype(jnp.float32)
                    sub_hit = diag["event_rebound_used"] | (diag["stringbed_active"] & (contact_closing > 0.0))
                    better = sub_hit & (diag["stringbed_rho2"] < best_rho2)
                    best_rho2 = jnp.where(better, diag["stringbed_rho2"], best_rho2)
                    best_normal = jnp.where(better[:, None], diag["stringbed_normal_world"], best_normal)
                    best_position = jnp.where(
                        better[:, None],
                        d.site_xpos[:, self._stringbed_site],
                        best_position,
                    )
                    closing = jnp.where(sub_hit, jnp.maximum(closing, contact_closing), closing)
                    return (
                        d,
                        cd,
                        hit | sub_hit,
                        closing,
                        best_rho2,
                        best_normal,
                        best_position,
                    ), None

                (
                    (
                        data,
                        cooldown,
                        hit_this_step,
                        closing_speed,
                        best_rho2,
                        best_contact_normal,
                        best_contact_position,
                    ),
                    _,
                ) = jax.lax.scan(
                    sub_body_v2,
                    (
                        data,
                        state.cooldown,
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), jnp.float32),
                        jnp.full((num_envs,), jnp.inf, jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                        jnp.zeros((num_envs, 3), jnp.float32),
                    ),
                    None,
                    length=self.control_substeps,
                )
            else:
                (data, cooldown, hit_this_step, closing_speed), _ = jax.lax.scan(
                    sub_body,
                    (
                        data,
                        state.cooldown,
                        jnp.zeros((num_envs,), bool),
                        jnp.zeros((num_envs,), jnp.float32),
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
            landed = shuttle_pos[:, 2] <= GROUND_REST_HEIGHT_M
            crossed_net = (jnp.sign(shuttle_pos[:, 0]) == -self.player_half_sign) & (jnp.abs(shuttle_pos[:, 0]) > 1e-9)
            phase_code = jnp.where((phase_code == STATE_HIT) & crossed_net, STATE_FLIGHT, phase_code)

            root_z = data.qpos[:, self._root_qadr + 2]
            body_fall = root_z < BODY_FALL_ROOT_HEIGHT_M
            if self.task_profile == IMPACT_RECOVERY_PROFILE:
                has_hit = state.hit_rewarded | hit_this_step
                first_hit = hit_this_step & (~state.hit_rewarded)
                apex_height = jnp.where(
                    has_hit,
                    jnp.maximum(state.apex_height, shuttle_pos[:, 2]),
                    state.apex_height,
                )
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
                apex_height = state.apex_height
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
            approach = jnp.where(
                (phase_code == STATE_INCOMING) & dynamic_task,
                w["approach"] * jnp.exp(-2.0 * jnp.linalg.norm(intercept - stringbed_pos, axis=-1)),
                0.0,
            )
            hit_bonus_fire = hit_this_step & (~state.hit_rewarded)
            hit_bonus = jnp.where(
                hit_bonus_fire & dynamic_task,
                w["hit_bonus"] * jnp.minimum(1.0, hit_closing_speed / 8.0),
                0.0,
            )
            hit_rewarded = state.hit_rewarded | hit_this_step
            crossed_fire = (
                (phase_code >= STATE_FLIGHT) & hit_rewarded & crossed_net & (~state.crossed_rewarded) & dynamic_task
            )
            crossed_bonus = jnp.where(crossed_fire, w["crossed_net"], 0.0)
            crossed_rewarded = state.crossed_rewarded | crossed_fire

            landing_score = self._landing_score(shuttle_pos[:, :2])
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
            fall_term = jnp.where(body_fall, -w["body_fall"], 0.0)

            racket_cvel = data.cvel[:, self._racket_body]
            racket_offset = stringbed_pos - data.subtree_com[:, self._racket_root]
            racket_head_velocity = racket_cvel[:, 3:] + jnp.cross(racket_cvel[:, :3], racket_offset)

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
                + hit_bonus
                + crossed_bonus
                + landing_term
                + effort
                + residual_term
                + posture
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
                hit_closing_speed=jnp.where(done, 0.0, hit_closing_speed),
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
                "landing_score": jnp.where(landing_fire, landing_score, 0.0),
                "miss": miss,
                "body_fall": body_fall,
                "hit_event": hit_bonus_fire,
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
            if lab_output is not None:
                unclipped_state_z = (state.lab_state - lab_norm_mean) / lab_norm_std
                transition.update(
                    {
                        "control_finite": jnp.all(jnp.isfinite(lab_output.full_action), axis=-1).astype(jnp.float32),
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
                        "body_action_rms": jnp.sqrt(jnp.mean(jnp.square(lab_output.body_action), axis=-1)),
                        "right_grip_action_rms": jnp.sqrt(jnp.mean(jnp.square(lab_output.right_grip_action), axis=-1)),
                        "lambda_lab": jnp.broadcast_to(state.lambda_lab, (num_envs,)),
                        "active_feed_count": jnp.broadcast_to(state.active_feed_count.astype(jnp.float32), (num_envs,)),
                        "racket_head_speed_m_s": racket_head_speed,
                        "muscle_power_abs_mean": muscle_power_abs_mean,
                        "normalized_control_energy": normalized_control_energy,
                        "body_action_saturation_fraction": jnp.mean(
                            (jnp.abs(lab_output.body_action) > 0.98).astype(jnp.float32),
                            axis=-1,
                        ),
                        "full_action_saturation_fraction": jnp.mean(
                            (jnp.abs(lab_output.full_action) > 0.98).astype(jnp.float32),
                            axis=-1,
                        ),
                        "net_clearance_m": jnp.where(crossed_fire, shuttle_pos[:, 2] - 1.55, 0.0),
                        "opponent_back_landing": landing_fire & (landing_score == 1.0),
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
