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
    IncomingShuttleHitEnv,
    _validate_reward_weights,
)
from environment.overall_environment.src.shuttle_feeder import FeedSample

STATE_INCOMING = 0
STATE_HIT = 1
STATE_FLIGHT = 2

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
    feed_idx: jnp.ndarray  # (N,) int32
    lab_state: jnp.ndarray  # (N, lab_state_size), empty outside LAB mode
    lambda_lab: jnp.ndarray  # scalar curriculum value
    active_feed_count: jnp.ndarray  # scalar easy-to-hard feed count
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

    def __post_init__(self) -> None:
        self.weights = _validate_reward_weights(dict(self.reward_weights))
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
            current_fingerprints = [
                feed_sample_fingerprint(sample) for sample in self.feed_bank
            ]
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
            if int(self.lab_state_builder.expected_state_dim) != int(
                self.lab_controller.lab_state_size
            ):
                raise ValueError("LAB state builder and latent runtime dimensions differ")
            from environment.overall_environment.src.stage3_lab import (
                apply_teacher_body_ctrlrange,
            )

            self._effective_ctrlrange_hash = apply_teacher_body_ctrlrange(
                self.model, self.lab_controller
            )
        self.ids = make_ids(self.model, self.physics_config)
        self.params = make_params(self.model, self.physics_config)

        cpu_env = IncomingShuttleHitEnv(
            self.xml,
            feed_bank=self.feed_bank,
            control_substeps=self.control_substeps,
            max_episode_steps=self.max_episode_steps,
            reward_weights=self.weights,
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
            if int(self.lab_state_builder.expected_state_dim) != int(
                self.lab_controller.lab_state_size
            ):
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
        self.intercept_points = jnp.asarray(
            np.stack([f.intercept_point for f in self.feed_bank]), dtype=jnp.float32
        )
        self.intercept_times = jnp.asarray(
            np.array([f.intercept_time_s for f in self.feed_bank]), dtype=jnp.float32
        )

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
        if self.lab_controller is None:
            return {"schema_version": "incoming_hit_direct_action_v1"}
        if self._control_manifest_cache is not None:
            return self._control_manifest_cache
        from environment.overall_environment.src.stage3_lab import (
            stage3_attachment_report,
        )

        payload = dict(self.lab_controller.control_manifest)
        payload["lab_state_schema_hash"] = self.lab_state_builder.schema_hash
        payload["filter_finger_observation"] = self.filter_finger_observation
        payload["racket_attachment"] = stage3_attachment_report(
            self.model, self.xml
        )
        payload["environment_abi"] = {
            "schema_version": "incoming_hit_environment_v1",
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
        payload["curriculum"] = (
            None
            if self.curriculum is None
            else dict(vars(self.curriculum))
        )
        payload_without_hash = dict(payload)
        payload_without_hash.pop("control_hash", None)
        payload["control_hash"] = hashlib.sha256(
            json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        self._control_manifest_cache = payload
        return payload

    @property
    def control_hash(self) -> str | None:
        return self.control_manifest.get("control_hash")

    def curriculum_values(self, env_steps: int):
        if self.curriculum is None:
            from environment.overall_environment.src.stage3_lab import Stage3CurriculumValues

            return Stage3CurriculumValues(
                lambda_lab=float(
                    self.lab_controller.lambda_lab if self.lab_controller is not None else 0.0
                ),
                feed_fraction=1.0,
                active_feed_count=len(self.feed_bank),
            )
        return self.curriculum.values(
            env_steps=int(env_steps), feed_bank_size=len(self.feed_bank)
        )

    def apply_curriculum(
        self,
        state: EnvState,
        *,
        lambda_lab: float,
        active_feed_count: int,
    ) -> EnvState:
        if not (1 <= int(active_feed_count) <= len(self.feed_bank)):
            raise ValueError("active_feed_count is outside the feed bank")
        return state._replace(
            lambda_lab=jnp.asarray(lambda_lab, dtype=jnp.float32),
            active_feed_count=jnp.asarray(active_feed_count, dtype=jnp.int32),
        )

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
            "layers": [
                {k: jnp.asarray(v) for k, v in layer.items()} for layer in tensors["layers"]
            ],
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
        touch = (
            data.sensordata[:, b["sensor_adr"]]
            if int(b["sensor_adr"].shape[0]) > 0
            else jnp.zeros((n, 0))
        )
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

    def _compose_action(self, data: Any, state: "EnvState", action: jnp.ndarray):
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
            active_feed_count = jnp.asarray(values.active_feed_count, dtype=jnp.int32)
            feed_idx = jax.random.randint(sub, (num_envs,), 0, active_feed_count)
            data = self._reset_arrays(template, feed_idx)
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
                obs=self.obs_bank[feed_idx],
                cooldown=zeros_i,
                step_index=zeros_i,
                phase_code=zeros_i,
                hit_rewarded=zeros_b,
                crossed_rewarded=zeros_b,
                hit_closing_speed=jnp.zeros((num_envs,), jnp.float32),
                feed_idx=feed_idx,
                lab_state=lab_state,
                lambda_lab=jnp.asarray(values.lambda_lab, dtype=jnp.float32),
                active_feed_count=active_feed_count,
                key=key,
            )

        return reset

    def _observation(self, data: Any, feed_idx: jnp.ndarray, step_index: jnp.ndarray) -> jnp.ndarray:
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
        racket_features = jnp.concatenate(
            [stringbed_pos - root_pos, face_normal, face_vel], axis=-1
        )

        intercept = self.intercept_points[feed_idx]
        elapsed = step_index.astype(jnp.float32) * self.control_substeps * self.timestep
        time_to_intercept = jnp.maximum(0.0, self.intercept_times[feed_idx] - elapsed)
        phase = self._swing_phase(step_index, self.intercept_times[feed_idx]).astype(
            jnp.float32
        )
        task_features = jnp.concatenate(
            [
                intercept - stringbed_pos,
                intercept - root_pos,
                time_to_intercept[:, None],
                phase[:, None],
            ],
            axis=-1,
        )
        return jnp.concatenate(
            [qpos, qvel, shuttle_features, racket_features, task_features], axis=-1
        )

    def _landing_score(self, xy: jnp.ndarray) -> jnp.ndarray:
        half_width = 2.59 if self.singles else 3.05
        x, y = xy[:, 0], xy[:, 1]
        out = (jnp.abs(x) > _COURT_HALF_LENGTH) | (jnp.abs(y) > half_width)
        own = jnp.sign(x) == self.player_half_sign
        depth = jnp.abs(x)
        opponent = jnp.where(depth < _NET_FRONT_DEPTH, 0.2, jnp.where(depth >= _BACK_DEPTH, 1.0, 0.5))
        return jnp.where(out, -1.0, jnp.where(own, -0.5, opponent))

    def make_step_fn(self, mx, num_envs: int):
        substep = make_batched_substep_fn(
            mx, self.ids, self.params, self.model, vmap_mjx_step=self.impl != "warp"
        )
        forward_all = self._forward_all(mx)
        w = self.weights
        if self.lab_controller is not None:
            normalizer = getattr(self.lab_controller.runtime, "normalizer", None)
            if normalizer is None:
                lab_norm_mean = jnp.full(
                    (self.lab_state_size,), jnp.nan, dtype=jnp.float32
                )
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

            def sub_body(carry, _):
                d, cd, hit, closing = carry
                d, cd, diag = substep(d, cd)
                contact_closing = jnp.maximum(0.0, -diag["relative_normal_velocity"]).astype(
                    jnp.float32
                )
                sub_hit = diag["event_rebound_used"] | (
                    diag["stringbed_active"] & (contact_closing > 0.0)
                )
                closing = jnp.where(sub_hit, jnp.maximum(closing, contact_closing), closing)
                return (d, cd, hit | sub_hit, closing), None

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

            step_index = state.step_index + 1
            was_incoming = state.phase_code == STATE_INCOMING
            phase_code = jnp.where(was_incoming & hit_this_step, STATE_HIT, state.phase_code)
            hit_closing_speed = jnp.where(
                was_incoming & hit_this_step, closing_speed, state.hit_closing_speed
            )

            shuttle_pos = data.qpos[:, self._shuttle_qadr : self._shuttle_qadr + 3]
            landed = shuttle_pos[:, 2] <= GROUND_REST_HEIGHT_M
            crossed_net = (jnp.sign(shuttle_pos[:, 0]) == -self.player_half_sign) & (
                jnp.abs(shuttle_pos[:, 0]) > 1e-9
            )
            phase_code = jnp.where((phase_code == STATE_HIT) & crossed_net, STATE_FLIGHT, phase_code)

            root_z = data.qpos[:, self._root_qadr + 2]
            body_fall = root_z < BODY_FALL_ROOT_HEIGHT_M
            miss = landed & (phase_code == STATE_INCOMING)
            terminated = landed | body_fall
            truncated = (~terminated) & (step_index >= self.max_episode_steps)
            done = terminated | truncated

            stringbed_pos = data.site_xpos[:, self._stringbed_site]
            intercept = self.intercept_points[state.feed_idx]
            approach = jnp.where(
                phase_code == STATE_INCOMING,
                w["approach"]
                * jnp.exp(-2.0 * jnp.linalg.norm(intercept - stringbed_pos, axis=-1)),
                0.0,
            )
            hit_bonus_fire = hit_this_step & (~state.hit_rewarded)
            hit_bonus = jnp.where(
                hit_bonus_fire,
                w["hit_bonus"] * jnp.minimum(1.0, hit_closing_speed / 8.0),
                0.0,
            )
            hit_rewarded = state.hit_rewarded | hit_this_step
            crossed_fire = (
                (phase_code >= STATE_FLIGHT)
                & hit_rewarded
                & crossed_net
                & (~state.crossed_rewarded)
            )
            crossed_bonus = jnp.where(crossed_fire, w["crossed_net"], 0.0)
            crossed_rewarded = state.crossed_rewarded | crossed_fire

            landing_fire = landed & (~miss) & hit_rewarded
            landing_score = self._landing_score(shuttle_pos[:, :2])
            landing_term = jnp.where(landing_fire, w["landing_region"] * landing_score, 0.0)
            effort = -w["effort"] * jnp.mean(jnp.square(composed), axis=-1)
            residual_term = (
                -w.get("residual", 0.0) * jnp.mean(jnp.square(action), axis=-1)
                if self._base is not None and w.get("residual", 0.0) != 0.0
                else 0.0
            )
            posture = -w["posture"] * jnp.maximum(0.0, 0.85 - root_z)
            fall_term = jnp.where(body_fall, -w["body_fall"], 0.0)

            reward = (
                approach
                + hit_bonus
                + crossed_bonus
                + landing_term
                + effort
                + residual_term
                + posture
                + fall_term
            )

            obs = self._observation(data, state.feed_idx, step_index)
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            racket_cvel = data.cvel[:, self._racket_body]
            racket_offset = stringbed_pos - data.subtree_com[:, self._racket_root]
            racket_head_velocity = racket_cvel[:, 3:] + jnp.cross(
                racket_cvel[:, :3], racket_offset
            )
            racket_head_speed = jnp.linalg.norm(racket_head_velocity, axis=-1)
            muscle_power_abs_mean = jnp.mean(
                jnp.abs(data.actuator_force * data.actuator_velocity), axis=-1
            )
            normalized_control_energy = jnp.mean(jnp.square(composed), axis=-1)

            # auto-reset done envs
            key, sub = jax.random.split(state.key)
            new_feed = jax.random.randint(sub, (num_envs,), 0, state.active_feed_count)
            done_col = done[:, None]
            qpos = jnp.where(done_col, self.qpos_bank[new_feed], data.qpos)
            qvel = jnp.where(done_col, self.qvel_bank[new_feed], data.qvel)
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
            next_state = EnvState(
                data=data,
                obs=jnp.where(done_col, self.obs_bank[new_feed], obs),
                cooldown=jnp.where(done, zeros_i, cooldown),
                step_index=next_step_index,
                phase_code=jnp.where(done, zeros_i, phase_code),
                hit_rewarded=jnp.where(done, False, hit_rewarded),
                crossed_rewarded=jnp.where(done, False, crossed_rewarded),
                hit_closing_speed=jnp.where(done, 0.0, hit_closing_speed),
                feed_idx=next_feed_idx,
                lab_state=next_lab_state,
                lambda_lab=state.lambda_lab,
                active_feed_count=state.active_feed_count,
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
            }
            if lab_output is not None:
                unclipped_state_z = (
                    state.lab_state - lab_norm_mean
                ) / lab_norm_std
                transition.update(
                    {
                        "control_finite": jnp.all(
                            jnp.isfinite(lab_output.full_action), axis=-1
                        ).astype(jnp.float32),
                        "raw_latent_rms": jnp.sqrt(
                            jnp.mean(jnp.square(lab_output.raw_latent), axis=-1)
                        ),
                        "raw_latent_saturation": jnp.mean(
                            (jnp.abs(lab_output.raw_latent) > 2.0).astype(jnp.float32),
                            axis=-1,
                        ),
                        "latent_norm": jnp.linalg.norm(lab_output.latent, axis=-1),
                        "prior_sigma_mean": jnp.mean(lab_output.prior_sigma, axis=-1),
                        "lab_state_unclipped_z_rms": jnp.sqrt(
                            jnp.mean(jnp.square(unclipped_state_z), axis=-1)
                        ),
                        "lab_state_ood_fraction": jnp.mean(
                            (jnp.abs(unclipped_state_z) > 5.0).astype(jnp.float32),
                            axis=-1,
                        ),
                        "body_action_rms": jnp.sqrt(
                            jnp.mean(jnp.square(lab_output.body_action), axis=-1)
                        ),
                        "right_grip_action_rms": jnp.sqrt(
                            jnp.mean(jnp.square(lab_output.right_grip_action), axis=-1)
                        ),
                        "lambda_lab": jnp.broadcast_to(state.lambda_lab, (num_envs,)),
                        "active_feed_count": jnp.broadcast_to(
                            state.active_feed_count.astype(jnp.float32), (num_envs,)
                        ),
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
                        "net_clearance_m": jnp.where(
                            crossed_fire, shuttle_pos[:, 2] - 1.55, 0.0
                        ),
                        "opponent_back_landing": landing_fire & (landing_score == 1.0),
                    }
                )
                if lab_output.raw_bounded_residual is not None:
                    transition["bounded_residual_rms"] = jnp.sqrt(
                        jnp.mean(
                            jnp.square(lab_output.raw_bounded_residual), axis=-1
                        )
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
