"""Post-Stage3 fixed-state latent interventions with real task outcomes.

This module is the final task-causal layer.  It intentionally does not upgrade
the Stage-2 diagnostic adapter into a task claim: only a selected, completed
Stage-3 C7 CPU evaluator can produce ``latent_task_causal_v2`` evidence.

One configuration-driven run performs four operations:

1. bind the selected ``best_synergy`` branch directly from its sealed Stage-3
   evaluation and selection manifest (the full354 branch has no latent and is
   explicitly N/A);
2. materialize checkpoint-bound Stage-3 sample states and latent directions;
3. use the generic causal driver/artifact builder for exact simulator-state
   restore and common-random-number (CRN) paired rollouts; and
4. publish a mask-aware task report.  Missing impacts and return landings have
   explicit presence fields.  Their zero storage sentinels are never treated as
   continuous measurements or included in error deltas.

The real adapter is built in.  Users provide artifacts and JSON; they do not
write an evaluator plugin.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
from musclemimic.latent_muscle.analysis_export import ANALYSIS_INPUT_SCHEMA_VERSION
from musclemimic.latent_muscle.causal_rollout_artifact import (
    REQUIRED_OUTCOMES,
    build_causal_rollout_artifact,
    validate_causal_rollout_artifact,
)
from musclemimic.latent_muscle.causal_rollout_driver import (
    ADAPTER_SCHEMA_VERSION,
    RolloutRequest,
    produce_paired_rollouts,
)

CONFIG_SCHEMA_VERSION = "latent_task_causal_config_v2"
SAMPLE_PLAN_SCHEMA_VERSION = "latent_task_causal_sample_plan_v2"
TASK_CAUSAL_SCHEMA_VERSION = "latent_task_causal_v2"
TASK_CAUSAL_BRANCH_SCHEMA_VERSION = "latent_task_causal_branch_v2"
TASK_EFFECTS_SCHEMA_VERSION = "latent_task_causal_masked_effects_v1"
ROLLOUT_ENGINE = "stage3_cpu_final_c7_latent_task_causal_v2"
PROMOTION_FILENAME = "promotion_metrics.json"
MASKED_EFFECTS_FILENAME = "task_effects.npz"
MASKED_EFFECTS_MANIFEST_FILENAME = "task_effects.json"
_FORMAL_TASK_CAUSAL_FAMILIES = frozenset(("best_synergy",))
_HEX = frozenset("0123456789abcdef")
_IMPACT_SENTINEL_CONTRACTS = (
    ("impact_position_error_present", "impact_position_error_m"),
    ("impact_timing_error_present", "impact_timing_signed_error_s"),
    ("stringbed_normal_error_present", "stringbed_normal_error_rad"),
    ("racket_linear_velocity_error_present", "racket_linear_velocity_error_m_s"),
    ("racket_angular_velocity_error_present", "racket_angular_velocity_error_rad_s"),
)
_LANDING_SENTINEL_CONTRACTS = (
    ("landing_error_present", "landing_error_m"),
    ("landing_xy_present", "landing_x_m"),
    ("landing_xy_present", "landing_y_m"),
    ("ground_contact_present", "ground_contact_x_m"),
    ("ground_contact_present", "ground_contact_y_m"),
)


@dataclass(frozen=True)
class TaskCausalSample:
    sample_uid: str
    feed_index: int
    step_index: int
    feed_fingerprint: str
    baseline_latent: np.ndarray


@dataclass(frozen=True)
class Stage3BranchContext:
    family: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    selected_synergy_source_fingerprint: str
    evaluation_report_path: Path
    evaluation_report_sha256: str
    spec_path: Path
    stage3_checkpoint_path: Path
    stage3_checkpoint_payload_sha256: str
    latent_checkpoint_fingerprint: str
    formal_synergy_basis_fingerprint: str
    policy_abi_hash: str
    evaluation_seed: int
    evaluation_feed_fingerprints: tuple[str, ...]


@dataclass
class Stage3RuntimeBundle:
    env: Any
    policy_action: Callable[[np.ndarray], np.ndarray]
    context: Stage3BranchContext
    body_actuator_names: tuple[str, ...]
    signal_layout: Any
    environment_fingerprint: str


def _outcome_schema(
    feature_names: Sequence[str],
    units: Sequence[str],
    *,
    coordinate_frame: str,
    semantics: str,
    **extra: Any,
) -> dict[str, Any]:
    names = [str(value) for value in feature_names]
    unit_values = [str(value) for value in units]
    if not names or len(names) != len(unit_values) or len(set(names)) != len(names):
        raise ValueError("task-causal outcome names/units must be aligned and unique")
    payload = {
        "feature_names": names,
        "units": unit_values,
        "coordinate_frame": str(coordinate_frame),
        "semantics": str(semantics),
        "available": True,
    }
    payload.update(extra)
    return payload


def task_outcome_schemas(
    *,
    muscle_names: Sequence[str],
    joint_names: Sequence[str],
    trunk_body_names: Sequence[str] = ("Full Body", "torso"),
) -> dict[str, dict[str, Any]]:
    """Return the ordered real-rollout outcome and missing-event contract."""

    muscles = _unique_nonempty_strings(muscle_names, "muscle_names")
    joints = _unique_nonempty_strings(joint_names, "joint_names")
    trunk = _unique_nonempty_strings(trunk_body_names, "trunk_body_names")
    muscle_features = [f"time_mean:{name}" for name in muscles]
    joint_features = [f"time_mean:{name}" for name in joints]
    trunk_features = [
        feature
        for body in trunk
        for feature in (
            f"time_mean_angular_velocity_x:{body}",
            f"time_mean_angular_velocity_y:{body}",
            f"time_mean_angular_velocity_z:{body}",
            f"peak_angular_speed:{body}",
        )
    ]
    racket_features = (
        "position_at_peak_linear_speed_x",
        "position_at_peak_linear_speed_y",
        "position_at_peak_linear_speed_z",
        "linear_velocity_at_peak_x",
        "linear_velocity_at_peak_y",
        "linear_velocity_at_peak_z",
        "angular_velocity_at_peak_x",
        "angular_velocity_at_peak_y",
        "angular_velocity_at_peak_z",
        "stringbed_normal_at_peak_x",
        "stringbed_normal_at_peak_y",
        "stringbed_normal_at_peak_z",
    )
    impact_features = (
        "hit_present",
        "impact_measurement_present",
        "impact_position_error_present",
        "impact_position_error_m",
        "impact_timing_error_present",
        "impact_timing_signed_error_s",
        "stringbed_normal_error_present",
        "stringbed_normal_error_rad",
        "racket_linear_velocity_error_present",
        "racket_linear_velocity_error_m_s",
        "racket_angular_velocity_error_present",
        "racket_angular_velocity_error_rad_s",
        "closest_shuttle_stringbed_distance_m",
        "miss_present",
        "body_fall_present",
    )
    landing_features = (
        "return_landing_present",
        "landing_error_present",
        "landing_error_m",
        "landing_xy_present",
        "landing_x_m",
        "landing_y_m",
        "flight_resolved_present",
        "ground_contact_present",
        "ground_contact_x_m",
        "ground_contact_y_m",
        "apex_measurement_present",
        "apex_height_m",
        "unresolved_flight_present",
        "recovery_complete_present",
        "body_fall_present",
    )
    sentinel_policy = {
        "schema_version": "event_presence_masked_zero_sentinel_v1",
        "storage_sentinel": 0.0,
        "effect_policy": (
            "continuous error deltas are computed only when baseline and perturbed "
            "presence fields both equal one; stored zero sentinels are never measurements"
        ),
    }
    return {
        "muscle_excitation": _outcome_schema(
            muscle_features,
            ["unit_interval"] * len(muscle_features),
            coordinate_frame="ordered_stage3_body_actuator_abi",
            semantics="unit_interval_excitation",
            temporal_reduction="arithmetic_mean_over_real_post_snapshot_transitions",
        ),
        "muscle_activation": _outcome_schema(
            muscle_features,
            ["unit_interval"] * len(muscle_features),
            coordinate_frame="ordered_stage3_body_actuator_abi",
            semantics="mujoco_unit_interval_activation_state",
            temporal_reduction="arithmetic_mean_over_real_post_snapshot_transitions",
        ),
        "joint_position": _outcome_schema(
            joint_features,
            ["rad"] * len(joint_features),
            coordinate_frame="ordered_scalar_hinge_joint_qpos",
            semantics="ordered_joint_qpos",
            temporal_reduction="arithmetic_mean_over_real_post_snapshot_transitions",
        ),
        "joint_velocity": _outcome_schema(
            joint_features,
            ["rad_s-1"] * len(joint_features),
            coordinate_frame="ordered_scalar_hinge_joint_dof",
            semantics="ordered_joint_qvel",
            temporal_reduction="arithmetic_mean_over_real_post_snapshot_transitions",
        ),
        "trunk_state": _outcome_schema(
            trunk_features,
            [unit for _body in trunk for unit in ("rad_s-1", "rad_s-1", "rad_s-1", "rad_s-1")],
            coordinate_frame="world_frame_mujoco_object_velocity",
            semantics="ordered_trunk_state",
        ),
        "racket_state": _outcome_schema(
            racket_features,
            (
                "m",
                "m",
                "m",
                "m_s-1",
                "m_s-1",
                "m_s-1",
                "rad_s-1",
                "rad_s-1",
                "rad_s-1",
                "unitless",
                "unitless",
                "unitless",
            ),
            coordinate_frame="world_frame_stringbed_at_peak_linear_speed",
            semantics="ordered_racket_state",
        ),
        "impact_outcome": _outcome_schema(
            impact_features,
            (
                "binary",
                "binary",
                "binary",
                "m",
                "binary",
                "s",
                "binary",
                "rad",
                "binary",
                "m_s-1",
                "binary",
                "rad_s-1",
                "m",
                "binary",
                "binary",
            ),
            coordinate_frame="stage3_target_and_world_frame",
            semantics="ordered_impact_outcome",
            missing_event_contract=sentinel_policy,
            masked_value_contracts=[
                {
                    "presence_feature": presence,
                    "value_feature": value,
                    "missing_sentinel": 0.0,
                }
                for presence, value in _IMPACT_SENTINEL_CONTRACTS
            ],
        ),
        "landing_outcome": _outcome_schema(
            landing_features,
            (
                "binary",
                "binary",
                "m",
                "binary",
                "m",
                "m",
                "binary",
                "binary",
                "m",
                "m",
                "binary",
                "m",
                "binary",
                "binary",
                "binary",
            ),
            coordinate_frame="world_court_frame",
            semantics="ordered_landing_outcome",
            missing_event_contract=sentinel_policy,
            masked_value_contracts=[
                {
                    "presence_feature": presence,
                    "value_feature": value,
                    "missing_sentinel": 0.0,
                }
                for presence, value in _LANDING_SENTINEL_CONTRACTS
            ],
        ),
    }


def validate_task_event_schema(schemas: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless impact/landing missing values are explicitly masked."""

    if not isinstance(schemas, Mapping) or set(schemas) != set(REQUIRED_OUTCOMES):
        raise ValueError("task-causal schemas must contain exactly the generic outcomes")
    result = copy.deepcopy(dict(schemas))
    for outcome, contracts in (
        ("impact_outcome", _IMPACT_SENTINEL_CONTRACTS),
        ("landing_outcome", _LANDING_SENTINEL_CONTRACTS),
    ):
        schema = result.get(outcome)
        if not isinstance(schema, dict):
            raise ValueError(f"{outcome} schema is missing")
        names = schema.get("feature_names")
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError(f"{outcome} feature names are invalid")
        sentinel = schema.get("missing_event_contract")
        if not isinstance(sentinel, dict) or sentinel.get("schema_version") != (
            "event_presence_masked_zero_sentinel_v1"
        ):
            raise ValueError(f"{outcome} lacks the event-presence sentinel contract")
        if sentinel.get("storage_sentinel") != 0.0 or "never measurements" not in str(
            sentinel.get("effect_policy", "")
        ):
            raise ValueError(f"{outcome} sentinel/effect policy is unsafe")
        supplied = schema.get("masked_value_contracts")
        expected = [
            {
                "presence_feature": presence,
                "value_feature": value,
                "missing_sentinel": 0.0,
            }
            for presence, value in contracts
        ]
        if supplied != expected:
            raise ValueError(f"{outcome} masked value contracts changed")
        for presence, value in contracts:
            if presence not in names or value not in names:
                raise ValueError(f"{outcome} is missing presence/value feature {presence}/{value}")
    return result


class Stage3TaskCausalAdapter:
    """Concrete C7 CPU adapter implementing the generic causal protocol."""

    def __init__(
        self,
        *,
        runtime: Stage3RuntimeBundle,
        samples: Sequence[TaskCausalSample],
        rollout_horizon_steps: int,
        intervention_duration_steps: int = 1,
        latent_match_atol: float = 1e-5,
        trunk_body_names: Sequence[str] = ("Full Body", "torso"),
    ) -> None:
        self.runtime = runtime
        self.env = runtime.env
        self.policy_action = runtime.policy_action
        self.samples = {sample.sample_uid: sample for sample in samples}
        if not self.samples or len(self.samples) != len(samples):
            raise ValueError("task-causal samples must be non-empty and UID-unique")
        self.rollout_horizon_steps = int(rollout_horizon_steps)
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.latent_match_atol = float(latent_match_atol)
        if self.rollout_horizon_steps <= 0:
            raise ValueError("rollout_horizon_steps must be positive")
        if not 1 <= self.intervention_duration_steps <= self.rollout_horizon_steps:
            raise ValueError("intervention_duration_steps must lie within the rollout horizon")
        required_horizon = max(int(self.env.max_episode_steps) - sample.step_index for sample in samples)
        if self.rollout_horizon_steps < required_horizon:
            raise ValueError(
                "rollout_horizon_steps cannot cover impact/landing/recovery through the "
                f"episode limit; required at least {required_horizon}"
            )
        if not np.isfinite(self.latent_match_atol) or self.latent_match_atol < 0.0:
            raise ValueError("latent_match_atol must be finite and non-negative")
        self.trunk_body_names = tuple(_unique_nonempty_strings(trunk_body_names, "trunk_body_names"))
        self._joint_qpos_addresses = self._resolve_joint_qpos_addresses()
        self._trunk_body_ids = self._resolve_trunk_body_ids()
        layout = self.runtime.signal_layout
        self._schemas = validate_task_event_schema(
            task_outcome_schemas(
                muscle_names=layout.actuator_names,
                joint_names=layout.joint_names,
                trunk_body_names=self.trunk_body_names,
            )
        )
        self._active_sample: TaskCausalSample | None = None
        self._active_snapshot: dict[str, Any] | None = None
        self._last_observation: np.ndarray | None = None
        self._rollout_seed: int | None = None
        self._rng_fingerprint: str | None = None

    def descriptor(self) -> Mapping[str, Any]:
        context = self.runtime.context
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "checkpoint_fingerprint": context.latent_checkpoint_fingerprint,
            "synergy_basis_fingerprint": context.formal_synergy_basis_fingerprint,
            "environment_fingerprint": self.runtime.environment_fingerprint,
            "policy_abi_hash": context.policy_abi_hash,
            "rollout_engine": ROLLOUT_ENGINE,
            "physical_signal_semantics": physical_signal_metadata(),
            "activation_valid_mask": np.asarray(
                self.runtime.signal_layout.activation_valid_mask,
                dtype=bool,
            ).tolist(),
            "outcome_schemas": copy.deepcopy(self._schemas),
            "outcome_availability": dict.fromkeys(REQUIRED_OUTCOMES, True),
            "stage2_diagnostic_outcomes_complete": True,
            "task_outcomes_complete": True,
        }

    def prepare_analysis_sample(self, sample_uid: str) -> dict[str, Any]:
        try:
            sample = self.samples[str(sample_uid)]
        except KeyError as exc:
            raise ValueError(f"Stage-3 sample plan has no UID {sample_uid!r}") from exc
        obs, _info = self.env.reset(feed_index=sample.feed_index)
        terminated = truncated = False
        for _ in range(sample.step_index):
            action = self._policy(obs)
            obs, _reward, terminated, truncated, _info = self.env.step(action)
            if terminated or truncated:
                raise ValueError(
                    "Stage-3 causal sample lies after episode termination: "
                    f"uid={sample.sample_uid} feed={sample.feed_index} step={sample.step_index}"
                )
        if int(self.env.step_index) != sample.step_index:
            raise ValueError("Stage-3 evaluator did not reach the exact requested sample step")
        self._assert_pre_hit_snapshot(sample)
        current_latent = self._effective_latent(obs)
        if not np.allclose(
            current_latent,
            sample.baseline_latent,
            atol=self.latent_match_atol,
            rtol=0.0,
        ):
            raise ValueError("sample-plan baseline latent differs from the live Stage-3 policy")
        self._active_sample = sample
        self._last_observation = np.asarray(obs, dtype=np.float64).copy()
        snapshot = self._capture_snapshot()
        self._active_snapshot = snapshot
        self._rollout_seed = None
        self._rng_fingerprint = None
        return snapshot

    def snapshot_to_bytes(self, snapshot: Any) -> bytes:
        if not isinstance(snapshot, Mapping):
            raise ValueError("Stage-3 simulator snapshot must be a mapping")
        payload = _jsonable(snapshot)
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def restore_snapshot(self, snapshot: Any) -> None:
        if not isinstance(snapshot, Mapping) or self._active_sample is None:
            raise RuntimeError("prepare_analysis_sample must precede Stage-3 restore")
        self._restore_snapshot(dict(snapshot))
        self._active_snapshot = copy.deepcopy(dict(snapshot))
        self._rollout_seed = None
        self._rng_fingerprint = None

    def capture_snapshot(self) -> dict[str, Any]:
        if self._active_sample is None:
            raise RuntimeError("Stage-3 causal adapter has no active sample")
        return self._capture_snapshot()

    def set_common_random_seed(self, seed: int) -> None:
        value = int(seed)
        if value < 0:
            raise ValueError("Stage-3 rollout seed must be non-negative")
        self.env.rng = np.random.default_rng(value)
        self._rollout_seed = value
        state = {
            "schema_version": "stage3_task_causal_rng_state_v1",
            "rollout_seed": value,
            "environment_rng": _jsonable(self.env.rng.bit_generator.state),
            "high_level_policy_rng": "deterministic_mean_action_no_rng",
            "low_level_decoder_rng": "deterministic_frozen_decoder_no_rng",
            "physics_rng": "deterministic_no_rng",
        }
        self._rng_fingerprint = canonical_json_sha256(state)

    def random_state_fingerprint(self) -> str:
        if self._rollout_seed is None or self._rng_fingerprint is None:
            raise RuntimeError("set_common_random_seed must precede RNG fingerprinting")
        return self._rng_fingerprint

    def evaluate_rollout(self, request: RolloutRequest) -> Mapping[str, Any]:
        if self._active_sample is None or self._rollout_seed != int(request.rollout_seed):
            raise RuntimeError("Stage-3 rollout was not restored and CRN-seeded")
        if request.sample_uid != self._active_sample.sample_uid:
            raise ValueError("Stage-3 rollout request UID differs from the active snapshot")
        obs = np.asarray(self.env._observation(), dtype=np.float64)
        baseline_latent = self._effective_latent(obs)
        if not np.allclose(
            baseline_latent,
            np.asarray(request.baseline_latent, dtype=np.float64),
            atol=self.latent_match_atol,
            rtol=0.0,
        ):
            raise ValueError("rollout request baseline latent differs from the restored Stage-3 policy")

        muscle_excitation: list[np.ndarray] = []
        muscle_activation: list[np.ndarray] = []
        joint_position: list[np.ndarray] = []
        joint_velocity: list[np.ndarray] = []
        trunk_velocity: list[np.ndarray] = []
        racket_trace: list[np.ndarray] = []
        closest_distance = float("inf")
        ground_contact_xy: np.ndarray | None = None
        hit_observed = False
        body_fall = False
        terminated = truncated = False

        for rollout_step in range(self.rollout_horizon_steps):
            action = self._policy(obs)
            override = None
            if not request.is_baseline and rollout_step < self.intervention_duration_steps:
                current = self._effective_latent(obs, task_action=action)
                override = current + float(request.intervention_epsilon) * np.asarray(
                    request.intervention_direction,
                    dtype=np.float64,
                )
            obs, _reward, terminated, truncated, info = self.env.step(
                action,
                effective_latent_override=override,
            )
            transition = self.runtime.signal_layout.capture_transition(self.env, info)
            muscle_excitation.append(np.asarray(transition["muscle_excitation"], dtype=np.float64))
            muscle_activation.append(np.asarray(transition["muscle_activation"], dtype=np.float64))
            joint_position.append(np.asarray(self.env.data.qpos, dtype=np.float64)[self._joint_qpos_addresses].copy())
            joint_velocity.append(np.asarray(transition["joint_angular_velocity"], dtype=np.float64))
            trunk_velocity.append(self._trunk_angular_velocity())
            racket_trace.append(self._racket_state())
            shuttle = np.asarray(
                self.env.data.qpos[self.env._shuttle_qadr : self.env._shuttle_qadr + 3],
                dtype=np.float64,
            )
            stringbed = np.asarray(
                self.env.data.site_xpos[self.env._stringbed_site],
                dtype=np.float64,
            )
            closest_distance = min(closest_distance, float(np.linalg.norm(shuttle - stringbed)))
            hit_observed = hit_observed or bool(info.get("hit_this_step", False))
            body_fall = body_fall or bool(info.get("body_fall", False))
            flight = info.get("flight")
            if isinstance(flight, Mapping) and bool(flight.get("landed", False)):
                xyz = np.asarray(flight.get("shuttle_xyz"), dtype=np.float64)
                if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                    ground_contact_xy = xyz[:2].copy()
            if terminated or truncated:
                break
        if not muscle_excitation:
            raise ValueError("Stage-3 task-causal rollout produced no real transition")
        if not (terminated or truncated):
            raise ValueError(
                "Stage-3 causal horizon ended before impact/landing/recovery resolution or "
                "the environment episode limit"
            )
        return self._summarize_outcomes(
            muscle_excitation=muscle_excitation,
            muscle_activation=muscle_activation,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            trunk_velocity=trunk_velocity,
            racket_trace=racket_trace,
            closest_distance=closest_distance,
            ground_contact_xy=ground_contact_xy,
            hit_observed=hit_observed,
            body_fall=body_fall,
            terminated=terminated,
            truncated=truncated,
        )

    def _policy(self, observation: np.ndarray) -> np.ndarray:
        action = np.asarray(self.policy_action(np.asarray(observation, dtype=np.float64)), dtype=np.float64)
        if action.shape != (int(self.env.action_size),) or not np.all(np.isfinite(action)):
            raise ValueError("Stage-3 high-level policy returned an invalid deterministic action")
        return action

    def _assert_pre_hit_snapshot(self, sample: TaskCausalSample) -> None:
        state_name = str(getattr(self.env.state, "value", self.env.state))
        if (
            state_name != "INCOMING"
            or bool(self.env._hit_rewarded)
            or self.env._impact_diag is not None
            or bool(self.env._flight_resolved)
        ):
            raise ValueError(
                "task-causal snapshots must be strictly before the first real hit and "
                f"before task resolution: uid={sample.sample_uid} state={state_name}"
            )

    def _effective_latent(
        self,
        observation: np.ndarray,
        *,
        task_action: np.ndarray | None = None,
    ) -> np.ndarray:
        action = self._policy(observation) if task_action is None else np.asarray(task_action)
        state = self.env.lab_state_builder.build_numpy(
            model=self.env.model,
            data=self.env.data,
            phase=self.env._swing_phase(),
        )
        output = self.env.lab_controller.decode_task_numpy(
            lab_state=state,
            task_action=action,
        )
        latent = np.asarray(output.latent, dtype=np.float64)
        if latent.shape != (int(self.env.lab_controller.latent_action_size),):
            raise ValueError("Stage-3 effective latent has the wrong dimension")
        return latent

    def _resolve_joint_qpos_addresses(self) -> np.ndarray:
        import mujoco

        addresses: list[int] = []
        for name in self.runtime.signal_layout.joint_names:
            joint_id = int(mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0 or int(self.env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"Stage-3 task-causal joint {name!r} is not a scalar hinge")
            addresses.append(int(self.env.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=np.int32)

    def _resolve_trunk_body_ids(self) -> np.ndarray:
        import mujoco

        values = []
        for name in self.trunk_body_names:
            body_id = int(mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name))
            if body_id < 0:
                raise ValueError(f"Stage-3 task-causal trunk body {name!r} is absent")
            values.append(body_id)
        return np.asarray(values, dtype=np.int32)

    def _trunk_angular_velocity(self) -> np.ndarray:
        import mujoco

        values = []
        for body_id in self._trunk_body_ids.tolist():
            velocity = np.zeros((6,), dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.env.model,
                self.env.data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            values.append(velocity[:3].copy())
        return np.stack(values, axis=0)

    def _racket_state(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self.env.data.site_xpos[self.env._stringbed_site], dtype=np.float64),
                np.asarray(self.env._stringbed_velocity(), dtype=np.float64),
                np.asarray(self.env._stringbed_angular_velocity(), dtype=np.float64),
                np.asarray(self.env._stringbed_normal(), dtype=np.float64),
            )
        )

    def _summarize_outcomes(
        self,
        *,
        muscle_excitation: Sequence[np.ndarray],
        muscle_activation: Sequence[np.ndarray],
        joint_position: Sequence[np.ndarray],
        joint_velocity: Sequence[np.ndarray],
        trunk_velocity: Sequence[np.ndarray],
        racket_trace: Sequence[np.ndarray],
        closest_distance: float,
        ground_contact_xy: np.ndarray | None,
        hit_observed: bool,
        body_fall: bool,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, np.ndarray]:
        excitation = np.mean(np.stack(muscle_excitation, axis=0), axis=0)
        activation = np.mean(np.stack(muscle_activation, axis=0), axis=0)
        qpos = np.mean(np.stack(joint_position, axis=0), axis=0)
        qvel = np.mean(np.stack(joint_velocity, axis=0), axis=0)
        trunk_trace = np.stack(trunk_velocity, axis=0)
        trunk_values = []
        for body_index in range(trunk_trace.shape[1]):
            body = trunk_trace[:, body_index, :]
            trunk_values.extend(np.mean(body, axis=0).tolist())
            trunk_values.append(float(np.max(np.linalg.norm(body, axis=1))))
        racket = np.stack(racket_trace, axis=0)
        peak_index = int(np.argmax(np.linalg.norm(racket[:, 3:6], axis=1)))
        racket_value = racket[peak_index]

        impact = np.zeros((15,), dtype=np.float64)
        impact[0] = float(hit_observed)
        diag = self.env._impact_diag
        impact_present = bool(hit_observed and isinstance(diag, Mapping))
        impact[1] = float(impact_present)
        if impact_present:
            target_position = np.asarray(
                self.env._active_target_value("impact_position_world"),
                dtype=np.float64,
            )
            target_normal = np.asarray(
                self.env._active_target_value("stringbed_normal_world"),
                dtype=np.float64,
            )
            target_linear = np.asarray(
                self.env._active_target_value("racket_linear_velocity_world"),
                dtype=np.float64,
            )
            target_angular = np.asarray(
                self.env._active_target_value("racket_angular_velocity_world"),
                dtype=np.float64,
            )
            target_time = float(self.env._active_target_value("impact_time_s"))
            measured_normal = np.asarray(diag["normal_world"], dtype=np.float64)
            impact[2] = impact[4] = impact[6] = impact[8] = impact[10] = 1.0
            impact[3] = float(np.linalg.norm(np.asarray(diag["position_world"], dtype=np.float64) - target_position))
            impact[5] = float(diag["impact_time_s"]) - target_time
            impact[7] = float(np.arccos(np.clip(np.dot(measured_normal, target_normal), -1.0, 1.0)))
            impact[9] = float(np.linalg.norm(np.asarray(diag["linear_velocity_world"]) - target_linear))
            impact[11] = float(np.linalg.norm(np.asarray(diag["angular_velocity_world"]) - target_angular))
        impact[12] = float(closest_distance)
        impact[13] = float(not hit_observed)
        impact[14] = float(body_fall)

        landing = np.zeros((15,), dtype=np.float64)
        landing_xy = self.env._landing_xy
        landing_present = bool(hit_observed and landing_xy is not None)
        landing[0] = float(landing_present)
        if landing_present:
            measured_xy = np.asarray(landing_xy, dtype=np.float64)
            target_xy = np.asarray(
                self.env._active_target_value("landing_target_xy"),
                dtype=np.float64,
            )
            landing[1] = landing[3] = 1.0
            landing[2] = float(np.linalg.norm(measured_xy - target_xy))
            landing[4:6] = measured_xy
        landing[6] = float(bool(self.env._flight_resolved))
        if ground_contact_xy is not None:
            landing[7] = 1.0
            landing[8:10] = np.asarray(ground_contact_xy, dtype=np.float64)
        landing[10] = 1.0
        landing[11] = float(self.env._apex_height_m)
        landing[12] = float(not bool(self.env._flight_resolved))
        landing[13] = float(bool(self.env._recovery_complete))
        landing[14] = float(body_fall)
        _validate_event_vector(impact, self._schemas["impact_outcome"])
        _validate_event_vector(landing, self._schemas["landing_outcome"])
        return {
            "muscle_excitation": np.asarray(excitation, dtype=np.float32),
            "muscle_activation": np.asarray(activation, dtype=np.float32),
            "joint_position": np.asarray(qpos, dtype=np.float32),
            "joint_velocity": np.asarray(qvel, dtype=np.float32),
            "trunk_state": np.asarray(trunk_values, dtype=np.float32),
            "racket_state": np.asarray(racket_value, dtype=np.float32),
            "impact_outcome": np.asarray(impact, dtype=np.float32),
            "landing_outcome": np.asarray(landing, dtype=np.float32),
        }

    def _capture_snapshot(self) -> dict[str, Any]:
        import mujoco

        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(
            (int(mujoco.mj_stateSize(self.env.model, state_spec)),),
            dtype=np.float64,
        )
        mujoco.mj_getState(self.env.model, self.env.data, integration, state_spec)
        payload = {
            "schema_version": "stage3_complete_simulator_snapshot_v1",
            "mujoco_state_spec": int(state_spec),
            "mujoco_integration_state": integration.tolist(),
            "environment_rng_state": _jsonable(self.env.rng.bit_generator.state),
            "python_state": self._python_state(),
            "observation": np.asarray(self.env._observation(), dtype=np.float64).tolist(),
        }
        return payload

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        import mujoco

        if snapshot.get("schema_version") != "stage3_complete_simulator_snapshot_v1":
            raise ValueError("unsupported Stage-3 complete simulator snapshot")
        state_spec = mujoco.mjtState(int(snapshot["mujoco_state_spec"]))
        if state_spec != mujoco.mjtState.mjSTATE_INTEGRATION:
            raise ValueError("Stage-3 snapshot does not contain mjSTATE_INTEGRATION")
        integration = np.asarray(snapshot["mujoco_integration_state"], dtype=np.float64)
        expected = int(mujoco.mj_stateSize(self.env.model, state_spec))
        if integration.shape != (expected,) or not np.all(np.isfinite(integration)):
            raise ValueError("Stage-3 MuJoCo integration snapshot is malformed")
        mujoco.mj_setState(self.env.model, self.env.data, integration, state_spec)
        mujoco.mj_forward(self.env.model, self.env.data)
        # ``mj_forward`` refreshes all derived quantities.  Re-applying the
        # integration vector keeps warm-start/control state byte-identical.
        mujoco.mj_setState(self.env.model, self.env.data, integration, state_spec)
        self.env.rng.bit_generator.state = copy.deepcopy(snapshot["environment_rng_state"])
        self._restore_python_state(snapshot["python_state"])
        observed = np.asarray(self.env._observation(), dtype=np.float64)
        expected_obs = np.asarray(snapshot["observation"], dtype=np.float64)
        if not np.array_equal(observed, expected_obs):
            raise ValueError("Stage-3 snapshot restore changed the exact policy observation")

    def _python_state(self) -> dict[str, Any]:
        impact = None
        if isinstance(self.env._impact_diag, Mapping):
            impact = {key: _jsonable(value) for key, value in self.env._impact_diag.items()}
        return {
            "state": str(self.env.state.value),
            "step_index": int(self.env.step_index),
            "termination_reason": self.env.termination_reason,
            "feed_index": int(self.env._feed_index),
            "active_target_index": int(self.env._active_target_index),
            "hit_closing_speed": float(self.env._hit_closing_speed),
            "hit_rewarded": bool(self.env._hit_rewarded),
            "crossed_net_rewarded": bool(self.env._crossed_net_rewarded),
            "landing_region": self.env._landing_region,
            "landing_rewarded": bool(self.env._landing_rewarded),
            "impact_diag": impact,
            "landing_xy": _optional_array_list(self.env._landing_xy),
            "apex_height_m": float(self.env._apex_height_m),
            "recovery_step": int(self.env._recovery_step),
            "recovery_active": bool(self.env._recovery_active),
            "recovery_complete": bool(self.env._recovery_complete),
            "flight_resolved": bool(self.env._flight_resolved),
            "last_v2_metrics": _jsonable(self.env._last_v2_metrics),
            "lab_state": _optional_array_list(self.env.lab_state),
            "last_lab_input_state": _optional_array_list(self.env._last_lab_input_state),
            "previous_raw_latent": _optional_array_list(self.env._previous_raw_latent),
            "physics_rebound_cooldown": int(self.env.physics._cooldown),
        }

    def _restore_python_state(self, payload: Mapping[str, Any]) -> None:
        from environment.overall_environment.src.incoming_shuttle_hit_env import (
            IncomingHitState,
        )

        self.env.state = IncomingHitState(str(payload["state"]))
        self.env.step_index = int(payload["step_index"])
        self.env.termination_reason = payload["termination_reason"]
        self.env._feed_index = int(payload["feed_index"])
        self.env._active_target_index = int(payload["active_target_index"])
        self.env.feed = self.env.feed_bank[self.env._feed_index]
        self.env._hit_closing_speed = float(payload["hit_closing_speed"])
        self.env._hit_rewarded = bool(payload["hit_rewarded"])
        self.env._crossed_net_rewarded = bool(payload["crossed_net_rewarded"])
        self.env._landing_region = payload["landing_region"]
        self.env._landing_rewarded = bool(payload["landing_rewarded"])
        raw_impact = payload["impact_diag"]
        self.env._impact_diag = (
            None
            if raw_impact is None
            else {
                key: (np.asarray(value, dtype=np.float64) if isinstance(value, list) else value)
                for key, value in raw_impact.items()
            }
        )
        self.env._landing_xy = _optional_array(payload["landing_xy"])
        self.env._apex_height_m = float(payload["apex_height_m"])
        self.env._recovery_step = int(payload["recovery_step"])
        self.env._recovery_active = bool(payload["recovery_active"])
        self.env._recovery_complete = bool(payload["recovery_complete"])
        self.env._flight_resolved = bool(payload["flight_resolved"])
        self.env._last_v2_metrics = {str(key): float(value) for key, value in payload["last_v2_metrics"].items()}
        self.env.lab_state = _optional_array(payload["lab_state"])
        self.env._last_lab_input_state = _optional_array(payload["last_lab_input_state"])
        self.env._previous_raw_latent = _optional_array(payload["previous_raw_latent"])
        self.env._last_lab_output = None
        self.env.physics._cooldown = int(payload["physics_rebound_cooldown"])


def build_mask_aware_task_effects(
    *,
    baseline_records: str | Path,
    perturbed_records: str | Path,
    outcome_schemas: Mapping[str, Any],
    output_npz: str | Path,
    output_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Publish event error deltas only for both-present baseline/pairs."""

    schemas = validate_task_event_schema(outcome_schemas)
    baseline_path = Path(baseline_records).expanduser().resolve(strict=True)
    perturbed_path = Path(perturbed_records).expanduser().resolve(strict=True)
    with np.load(baseline_path, allow_pickle=False) as values:
        baseline = {name: np.asarray(values[name]) for name in values.files}
    with np.load(perturbed_path, allow_pickle=False) as values:
        perturbed = {name: np.asarray(values[name]) for name in values.files}
    for outcome in ("impact_outcome", "landing_outcome"):
        if outcome not in baseline or outcome not in perturbed:
            raise ValueError(f"task records are missing {outcome}")
        if perturbed[outcome].shape[3:] != baseline[outcome].shape[1:]:
            raise ValueError(f"task records have inconsistent {outcome} widths")
        _validate_event_matrix(baseline[outcome], schemas[outcome])
        _validate_event_matrix(perturbed[outcome], schemas[outcome])

    arrays: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "schema_version": TASK_EFFECTS_SCHEMA_VERSION,
        "mask_policy": "both_baseline_and_perturbed_presence_required_v1",
        "zero_sentinel_used_as_measurement": False,
        "event_pair_counts": {},
    }
    for outcome, contracts in (
        ("impact_outcome", _IMPACT_SENTINEL_CONTRACTS),
        ("landing_outcome", _LANDING_SENTINEL_CONTRACTS),
    ):
        schema = schemas[outcome]
        names = list(schema["feature_names"])
        base = np.asarray(baseline[outcome], dtype=np.float64)
        changed = np.asarray(perturbed[outcome], dtype=np.float64)
        event_name = "hit_present" if outcome == "impact_outcome" else "return_landing_present"
        event_index = names.index(event_name)
        base_event = base[:, event_index] > 0.5
        changed_event = changed[..., event_index] > 0.5
        pair_category = np.full(changed_event.shape, 3, dtype=np.int8)
        pair_category[base_event[:, None, None] & changed_event] = 0
        pair_category[base_event[:, None, None] & ~changed_event] = 1
        pair_category[~base_event[:, None, None] & changed_event] = 2
        arrays[f"{outcome}__event_pair_category"] = pair_category
        report["event_pair_counts"][outcome] = {
            "both_present": int(np.sum(pair_category == 0)),
            "lost_event": int(np.sum(pair_category == 1)),
            "gained_event": int(np.sum(pair_category == 2)),
            "neither_present": int(np.sum(pair_category == 3)),
        }
        for presence_name, value_name in contracts:
            presence_index = names.index(presence_name)
            value_index = names.index(value_name)
            base_presence = base[:, presence_index] > 0.5
            changed_presence = changed[..., presence_index] > 0.5
            valid = base_presence[:, None, None] & changed_presence
            delta = np.full(valid.shape, np.nan, dtype=np.float32)
            raw_delta = changed[..., value_index] - base[:, None, None, value_index]
            delta[valid] = raw_delta[valid].astype(np.float32)
            prefix = f"{outcome}__{value_name}"
            arrays[f"{prefix}__both_present_mask"] = valid
            arrays[f"{prefix}__delta"] = delta
    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest_path = (
        Path(output_manifest) if output_manifest is not None else output.with_name(MASKED_EFFECTS_MANIFEST_FILENAME)
    )
    report.update(
        {
            "npz_path": str(output.resolve()),
            "npz_sha256": file_sha256(output),
            "baseline_records_sha256": file_sha256(baseline_path),
            "perturbed_records_sha256": file_sha256(perturbed_path),
            "outcome_schemas_fingerprint": canonical_json_sha256(schemas),
        }
    )
    report["manifest_fingerprint"] = canonical_json_sha256(report)
    _write_json_atomic(manifest_path, report)
    return report


def build_task_causal_promotion(
    *,
    context: Stage3BranchContext,
    paired_rollout_manifest: str | Path,
    causal_artifact_npz: str | Path,
    causal_artifact_manifest: str | Path,
    task_effects_manifest: str | Path,
    sample_plan: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate the complete task layer and emit the final-claim gate."""

    rollout_path = Path(paired_rollout_manifest).expanduser().resolve(strict=True)
    rollout = _load_self_fingerprinted_json(rollout_path)
    causal = validate_causal_rollout_artifact(
        causal_artifact_npz,
        causal_artifact_manifest,
    )
    effects_path = Path(task_effects_manifest).expanduser().resolve(strict=True)
    effects = _load_self_fingerprinted_json(effects_path)
    effects_npz = Path(str(effects.get("npz_path", ""))).expanduser().resolve(strict=True)
    if effects.get("npz_sha256") != file_sha256(effects_npz):
        raise ValueError("mask-aware task effects NPZ changed")
    if effects.get("baseline_records_sha256") != rollout.get("baseline_records_sha256") or effects.get(
        "perturbed_records_sha256"
    ) != rollout.get("perturbed_records_sha256"):
        raise ValueError("mask-aware task effects use different paired rollout records")
    plan_path = Path(sample_plan).expanduser().resolve(strict=True)
    plan = _load_self_fingerprinted_json(plan_path)
    schemas = validate_task_event_schema(rollout.get("outcome_schemas"))
    all_available = rollout.get("outcome_availability") == dict.fromkeys(
        REQUIRED_OUTCOMES,
        True,
    )
    matrix = (
        int(rollout.get("num_samples", 0)) > 0
        and int(rollout.get("num_directions", 0)) > 0
        and int(rollout.get("num_epsilons", 0)) > 0
    )
    masked_exclusion_verified = True
    for outcome in ("impact_outcome", "landing_outcome"):
        expected_excluded = [contract["value_feature"] for contract in schemas[outcome]["masked_value_contracts"]]
        layout = (causal.get("outcome_layout") or {}).get(outcome)
        if not isinstance(layout, Mapping) or layout.get("excluded_masked_value_features") != expected_excluded:
            masked_exclusion_verified = False
            break
        flattened = set(layout.get("effect_names", ()))
        if any(f"{outcome}:{value}" in flattened for value in expected_excluded):
            masked_exclusion_verified = False
            break
    gates = {
        "selected_synergy_source_binding_verified": (
            plan.get("selected_synergy_source_fingerprint") == context.selected_synergy_source_fingerprint
        ),
        "stage3_c7_checkpoint_verified": (
            plan.get("stage3_checkpoint_payload_sha256") == context.stage3_checkpoint_payload_sha256
            and rollout.get("policy_abi_hash") == context.policy_abi_hash
            and rollout.get("checkpoint_fingerprint") == context.latent_checkpoint_fingerprint
        ),
        "exact_snapshot_restore": (
            rollout.get("fixed_state_initialization") == "exact_snapshot_restore"
            and bool(rollout.get("snapshot_fingerprints"))
        ),
        "common_random_numbers": (
            rollout.get("common_random_numbers") is True and bool(rollout.get("random_state_fingerprints"))
        ),
        "full_intervention_matrix_complete": matrix,
        "all_task_outcomes_available": all_available,
        "task_outcomes_complete": (
            rollout.get("task_outcomes_complete") is True and causal.get("task_outcomes_complete") is True
        ),
        "masked_impact_schema_verified": bool(schemas.get("impact_outcome")),
        "masked_landing_schema_verified": bool(schemas.get("landing_outcome")),
        "missing_event_sentinel_contract_verified": (
            effects.get("zero_sentinel_used_as_measurement") is False
            and effects.get("mask_policy") == "both_baseline_and_perturbed_presence_required_v1"
        ),
        "masked_event_effects_verified": (
            effects.get("schema_version") == TASK_EFFECTS_SCHEMA_VERSION
            and isinstance(effects.get("event_pair_counts"), dict)
            and effects.get("outcome_schemas_fingerprint") == canonical_json_sha256(schemas)
        ),
        "masked_task_values_excluded_from_generic_effects": masked_exclusion_verified,
        "pre_hit_snapshot_verified": (
            plan.get("pre_hit_snapshot_verified") is True
            and plan.get("snapshot_timing_contract") == "strictly_before_first_real_hit_v1"
        ),
        "complete_task_horizon_verified": (
            plan.get("horizon_covers_impact_landing_recovery") is True
            and int(plan.get("rollout_horizon_steps", 0))
            >= max(int(value) for value in plan.get("remaining_episode_steps", ()))
        ),
    }
    payload = {
        "schema_version": TASK_CAUSAL_BRANCH_SCHEMA_VERSION,
        "claim_scope": "post_stage3_single_branch_task_causal_evidence",
        "family": context.family,
        "passed": all(gates.values()),
        "task_causal_complete": all(gates.values()),
        **gates,
        "gates": gates,
        "num_samples": int(rollout["num_samples"]),
        "num_directions": int(rollout["num_directions"]),
        "num_epsilons": int(rollout["num_epsilons"]),
        "hit_pair_counts": effects["event_pair_counts"]["impact_outcome"],
        "landing_pair_counts": effects["event_pair_counts"]["landing_outcome"],
        "bindings": {
            "selection_manifest_path": str(context.selection_manifest_path),
            "selection_manifest_sha256": context.selection_manifest_sha256,
            "selected_synergy_source_fingerprint": context.selected_synergy_source_fingerprint,
            "stage3_checkpoint_payload_sha256": context.stage3_checkpoint_payload_sha256,
            "latent_checkpoint_fingerprint": context.latent_checkpoint_fingerprint,
            "formal_synergy_basis_fingerprint": context.formal_synergy_basis_fingerprint,
            "paired_rollout_manifest_path": str(rollout_path),
            "paired_rollout_manifest_fingerprint": rollout["manifest_fingerprint"],
            "causal_artifact_path": str(Path(causal_artifact_npz).resolve()),
            "causal_artifact_sha256": file_sha256(Path(causal_artifact_npz)),
            "causal_artifact_manifest_fingerprint": causal["manifest_fingerprint"],
            "task_effects_path": str(effects_npz),
            "task_effects_sha256": effects["npz_sha256"],
            "task_effects_manifest_fingerprint": effects["manifest_fingerprint"],
            "sample_plan_path": str(plan_path),
            "sample_plan_fingerprint": plan["manifest_fingerprint"],
        },
        "interpretation": (
            "Only this passed latent_task_causal_v2 gate supports final task-causal claims. "
            "Stage-2 causal evidence remains a pre-Stage3 diagnostic. Event-error effects "
            "are mask-aware; missing-event zero sentinels are excluded from continuous deltas."
        ),
    }
    payload["promotion_fingerprint"] = canonical_json_sha256(payload)
    _write_json_atomic(Path(output), payload)
    return payload


def load_branch_context(
    synergy_evaluation: str | Path,
    *,
    synergy_selection: str | Path,
    family: str,
) -> Stage3BranchContext:
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        validate_selected_artifact,
    )
    from musclemimic.badminton.stage3_paired_comparison import (
        _validate_evaluation_binding,
    )

    branch = str(family)
    if branch not in _FORMAL_TASK_CAUSAL_FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FORMAL_TASK_CAUSAL_FAMILIES)}")
    evaluation_path = resolve_task_causal_cli_path(synergy_evaluation, strict=True)
    selection_path = resolve_task_causal_cli_path(synergy_selection, strict=True)
    selection = validate_selected_artifact(selection_path)
    selected = (selection.get("checkpoints") or {}).get(branch)
    if not isinstance(selected, Mapping):
        raise ValueError("task causal evaluation requires selected best_synergy evidence")
    latent_fingerprint = _require_sha256(
        selected.get("checkpoint_fingerprint"),
        "selected best_synergy checkpoint",
    )
    evaluation = load_json_strict(evaluation_path)
    if not isinstance(evaluation, dict):
        raise ValueError("Stage-3 evaluation report must be a JSON object")
    validated = _validate_evaluation_binding(
        evaluation,
        report_path=evaluation_path,
        family=branch,
        expected_action_family="fixed_synergy",
        expected_latent_fingerprint=latent_fingerprint,
    )
    binding = validated["binding"]
    spec_path = Path(binding["spec_path"]).expanduser().resolve(strict=True)
    checkpoint_path = Path(binding["checkpoint_payload_path"]).expanduser().resolve(strict=True)
    feed_manifest = evaluation.get("evaluation_feed_manifest")
    feed_values = feed_manifest.get("sample_fingerprints") if isinstance(feed_manifest, Mapping) else None
    if not isinstance(feed_values, list):
        raise ValueError("selected synergy evaluation has no held-out feed fingerprints")
    feed_fingerprints = tuple(str(value) for value in feed_values[: len(evaluation["episodes"])])
    if not feed_fingerprints or len(set(feed_fingerprints)) != len(feed_fingerprints):
        raise ValueError("selected synergy evaluation feed fingerprints are invalid")
    source_fingerprint = canonical_json_sha256(
        {
            "schema_version": "stage3_selected_synergy_task_causal_source_v1",
            "evaluation_binding_sha256": binding["binding_sha256"],
            "selection_manifest_fingerprint": selection["selection_manifest_fingerprint"],
            "latent_checkpoint_fingerprint": latent_fingerprint,
        }
    )
    return Stage3BranchContext(
        family=branch,
        selection_manifest_path=selection_path,
        selection_manifest_sha256=file_sha256(selection_path),
        selected_synergy_source_fingerprint=source_fingerprint,
        evaluation_report_path=evaluation_path,
        evaluation_report_sha256=file_sha256(evaluation_path),
        spec_path=spec_path,
        stage3_checkpoint_path=checkpoint_path,
        stage3_checkpoint_payload_sha256=_require_sha256(
            binding["checkpoint_payload_sha256"],
            "stage3_checkpoint_payload_sha256",
        ),
        latent_checkpoint_fingerprint=latent_fingerprint,
        formal_synergy_basis_fingerprint=_require_sha256(
            selected["formal_synergy_basis_fingerprint"],
            "formal_synergy_basis_fingerprint",
        ),
        policy_abi_hash=_require_sha256(binding["policy_abi_hash"], "policy_abi_hash"),
        evaluation_seed=int(evaluation["evaluation_seed"]),
        evaluation_feed_fingerprints=feed_fingerprints,
    )


def build_stage3_runtime(context: Stage3BranchContext) -> Stage3RuntimeBundle:
    """Construct the exact final CPU evaluator selected by the paired report."""

    from environment.overall_environment.src.incoming_shuttle_hit_env import (
        IMPACT_RECOVERY_PROFILE,
        IncomingShuttleHitEnv,
    )
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        load_training_checkpoint_metadata,
    )
    from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
        _build_stage3_lab_components,
        _ensure_feed_bank_artifact,
        load_incoming_hit_spec,
    )
    from musclemimic.evaluation.stage3_signal_export import Stage3SignalLayout

    if file_sha256(context.stage3_checkpoint_path) != context.stage3_checkpoint_payload_sha256:
        raise ValueError("selected Stage-3 checkpoint payload changed")
    paths = load_incoming_hit_spec(context.spec_path)
    if paths.task_profile != IMPACT_RECOVERY_PROFILE:
        raise ValueError("task causal evaluation requires impact_recovery_v2")
    metadata = load_training_checkpoint_metadata(context.stage3_checkpoint_path)
    if metadata.get("checkpoint_version") != "incoming_hit_training_v3":
        raise ValueError("task causal evaluation requires the final Stage-3 v3 checkpoint")
    task_state = metadata.get("task_curriculum_state")
    if (
        not isinstance(task_state, Mapping)
        or task_state.get("stage") != "C7_recovery"
        or task_state.get("complete") is not True
    ):
        raise ValueError("task causal evaluation requires a completed C7_recovery checkpoint")
    control = metadata.get("control_manifest")
    if not isinstance(control, Mapping) or control.get("schema_version") != "stage3_lab_control_v1":
        raise ValueError("task causal evaluation requires a Stage-3 LAB checkpoint")
    if control.get("latent_checkpoint_fingerprint") != context.latent_checkpoint_fingerprint:
        raise ValueError("selected Stage-3 checkpoint uses a different low-level latent policy")
    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=control.get("latent_checkpoint_dir"),
        lambda_lab=(metadata.get("curriculum_state") or {}).get("lambda_lab"),
    )
    if lab is None:
        raise ValueError("selected Stage-3 checkpoint could not construct its LAB runtime")
    feed_artifact = _ensure_feed_bank_artifact(paths, evaluation=True)
    if tuple(feed_artifact.manifest.get("sample_fingerprints", ()))[: len(context.evaluation_feed_fingerprints)] != (
        context.evaluation_feed_fingerprints
    ):
        raise ValueError("live held-out feed bank differs from the selected-synergy evaluation")
    env = IncomingShuttleHitEnv(
        paths.scene_xml,
        feed_bank=feed_artifact.bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        task_profile=paths.task_profile,
        impact_target_bank=(
            paths.eval_target_bank_path if paths.eval_target_bank_path is not None else paths.target_bank_path
        ),
        recovery_horizon_steps=paths.recovery_horizon_steps,
        task_curriculum_stage="C7_recovery",
        terminate_on_body_fall=True,
        lab_controller=lab.controller,
        lab_state_builder=lab.state_builder,
        curriculum=lab.curriculum,
        filter_finger_observation=True,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.76)),
        seed=context.evaluation_seed,
    )
    if env.policy_abi_hash != context.policy_abi_hash:
        raise ValueError("live Stage-3 CPU evaluator policy ABI differs from the selected-synergy evaluation")
    if int(metadata["obs_size"]) != env.observation_size or int(metadata["action_size"]) != env.action_size:
        raise ValueError("live Stage-3 CPU observation/action ABI differs from its checkpoint")
    policy = _load_stage3_policy_mean(context.stage3_checkpoint_path, metadata)
    layout = Stage3SignalLayout.from_environment(
        env,
        body_actuator_names=lab.controller.router.body_actuator_names,
    )
    environment_fingerprint = canonical_json_sha256(
        {
            "schema_version": "stage3_task_causal_environment_binding_v1",
            "control_manifest": env.control_manifest,
            "evaluation_feed_manifest": feed_artifact.manifest,
            "evaluation_target_bank_sha256": env.impact_target_bank.bank_sha256,
            "stage3_checkpoint_payload_sha256": context.stage3_checkpoint_payload_sha256,
            "evaluation_seed": context.evaluation_seed,
            "rollout_engine": ROLLOUT_ENGINE,
        }
    )
    return Stage3RuntimeBundle(
        env=env,
        policy_action=policy,
        context=context,
        body_actuator_names=tuple(lab.controller.router.body_actuator_names),
        signal_layout=layout,
        environment_fingerprint=environment_fingerprint,
    )


def create_adapter(config: Mapping[str, Any]) -> Stage3TaskCausalAdapter:
    """Generic-driver factory for an already materialized Stage-3 sample plan."""

    payload = dict(config)
    context = load_branch_context(
        payload.get("synergy_evaluation", ""),
        synergy_selection=payload.get("synergy_selection", ""),
        family=str(payload.get("family", "")),
    )
    plan_path = resolve_task_causal_cli_path(
        payload.get("sample_plan", ""),
        strict=True,
    )
    plan = _load_self_fingerprinted_json(plan_path)
    if plan.get("schema_version") != SAMPLE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-3 task-causal sample plan")
    if plan.get("selected_synergy_source_fingerprint") != context.selected_synergy_source_fingerprint:
        raise ValueError("sample plan differs from the selected-synergy source")
    samples = _samples_from_plan(plan)
    runtime = build_stage3_runtime(context)
    if "rollout_horizon_steps" not in payload:
        raise ValueError("Stage-3 adapter config requires rollout_horizon_steps")
    return Stage3TaskCausalAdapter(
        runtime=runtime,
        samples=samples,
        rollout_horizon_steps=int(payload["rollout_horizon_steps"]),
        intervention_duration_steps=int(payload.get("intervention_duration_steps", 1)),
        latent_match_atol=float(payload.get("latent_match_atol", 1e-5)),
        trunk_body_names=payload.get("trunk_body_names", ("Full Body", "torso")),
    )


def validate_task_causal_branch_registry(value: Any) -> dict[str, dict[str, Any]]:
    """Require only the selected synergy branch for formal latent intervention."""

    if not isinstance(value, Mapping) or set(value) != _FORMAL_TASK_CAUSAL_FAMILIES:
        raise ValueError("formal task-causal branches must contain exactly best_synergy")
    result: dict[str, dict[str, Any]] = {}
    for family in sorted(_FORMAL_TASK_CAUSAL_FAMILIES):
        entry = value[family]
        if not isinstance(entry, Mapping):
            raise ValueError(f"task-causal branch {family} must be an object")
        direction_source = entry.get("direction_source")
        if not isinstance(direction_source, Mapping):
            raise ValueError(f"task-causal branch {family} lacks direction_source")
        result[family] = dict(entry)
    return result


def validate_symmetric_intervention_epsilons(value: Any) -> np.ndarray:
    """Require a unique non-zero matrix containing every ``+/- epsilon`` pair."""

    epsilons = np.asarray(value, dtype=np.float32)
    if (
        epsilons.ndim != 1
        or epsilons.size < 2
        or np.any(epsilons == 0.0)
        or not np.all(np.isfinite(epsilons))
        or np.unique(epsilons).size != epsilons.size
    ):
        raise ValueError("intervention_epsilons must be finite, unique, non-zero and contain at least one +/- pair")
    values = set(epsilons.astype(float).tolist())
    if not any(value > 0.0 for value in values) or any(-value not in values for value in values):
        raise ValueError("every intervention epsilon must have an exact symmetric +/- partner")
    return epsilons


def run_task_causal(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    source = resolve_task_causal_cli_path(config_path, strict=True)
    config = load_json_strict(source)
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"task-causal config schema must be {CONFIG_SCHEMA_VERSION}")
    registry = validate_task_causal_branch_registry(config.get("branches"))
    if "rollout_horizon_steps" not in config:
        raise ValueError("task-causal config requires rollout_horizon_steps")
    rollout_horizon_steps = int(config["rollout_horizon_steps"])
    if rollout_horizon_steps <= 0:
        raise ValueError("rollout_horizon_steps must be positive")
    contexts = {
        family: load_branch_context(
            config.get("synergy_evaluation", ""),
            synergy_selection=config.get("synergy_selection", ""),
            family=family,
        )
        for family in sorted(_FORMAL_TASK_CAUSAL_FAMILIES)
    }
    reference = contexts["best_synergy"]
    points = _validate_sample_points(config.get("sample_points"), context=reference)
    epsilons = validate_symmetric_intervention_epsilons(config.get("intervention_epsilons", (-0.5, 0.5)))
    branch_inputs: dict[str, dict[str, Any]] = {}
    for family, context in contexts.items():
        entry = registry[family]
        directions, names, binding = _load_direction_source(
            entry["direction_source"],
            context=context,
            max_directions=int(entry.get("max_directions", config.get("max_directions", 8))),
        )
        branch_inputs[family] = {
            "directions": directions,
            "direction_names": names,
            "direction_source_binding": binding,
        }
    output_dir = resolve_task_causal_cli_path(config.get("output_dir", ""))
    if output_dir.exists():
        raise FileExistsError(f"task-causal output directory already exists: {output_dir}")
    protocol = {
        "selected_synergy_source_fingerprint": reference.selected_synergy_source_fingerprint,
        "sample_points": [{"feed_index": feed_index, "step_index": step_index} for feed_index, step_index in points],
        "intervention_epsilons": epsilons.astype(float).tolist(),
        "intervention_duration_steps": int(config.get("intervention_duration_steps", 1)),
        "rollout_horizon_steps": rollout_horizon_steps,
        "base_seed": int(config.get("base_seed", 20_260_713)),
    }
    if dry_run:
        return {
            "schema_version": "latent_task_causal_dry_run_v2",
            "passed": True,
            "rollouts_executed": False,
            "output_published": False,
            "fixed_synergy_branch_registered": True,
            "full354_latent_intervention_applicable": False,
            "branches": {
                family: {
                    "num_directions": int(branch_inputs[family]["directions"].shape[0]),
                    "stage3_checkpoint_payload_sha256": context.stage3_checkpoint_payload_sha256,
                    "direction_source_binding": branch_inputs[family]["direction_source_binding"],
                }
                for family, context in contexts.items()
            },
            "num_samples": len(points),
            "num_epsilons": int(epsilons.shape[0]),
            "paired_protocol": protocol,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    branch_results: dict[str, dict[str, Any]] = {}
    for family in ("best_synergy",):
        branch_results[family] = _run_task_causal_branch(
            config=config,
            context=contexts[family],
            points=points,
            directions=branch_inputs[family]["directions"],
            direction_names=branch_inputs[family]["direction_names"],
            source_binding=branch_inputs[family]["direction_source_binding"],
            epsilons=epsilons,
            output_dir=output_dir / family,
        )
    promotion = _build_synergy_only_promotion(
        output_dir=output_dir,
        contexts=contexts,
        branch_results=branch_results,
        protocol=protocol,
    )
    return {
        "schema_version": TASK_CAUSAL_SCHEMA_VERSION,
        "passed": bool(promotion["passed"]),
        "task_causal_complete": bool(promotion["task_causal_complete"]),
        "fixed_synergy_branch_complete": bool(promotion["fixed_synergy_branch_complete"]),
        "full354_latent_intervention_applicable": False,
        "output_dir": str(output_dir),
        "promotion_metrics": str((output_dir / PROMOTION_FILENAME).resolve()),
        "branches": branch_results,
    }


def _run_task_causal_branch(
    *,
    config: Mapping[str, Any],
    context: Stage3BranchContext,
    points: Sequence[tuple[int, int]],
    directions: np.ndarray,
    direction_names: Sequence[str],
    source_binding: Mapping[str, Any],
    epsilons: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    runtime = build_stage3_runtime(context)
    samples = _scout_samples(
        runtime,
        points=points,
        latent_match_atol=float(config.get("latent_match_atol", 1e-5)),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    analysis_path = output_dir / "analysis_inputs.npz"
    np.savez_compressed(
        analysis_path,
        sample_uids=np.asarray([sample.sample_uid for sample in samples], dtype=np.str_),
        latents=np.stack([sample.baseline_latent for sample in samples], axis=0).astype(np.float32),
        intervention_directions=np.asarray(directions, dtype=np.float32),
        intervention_epsilons=np.asarray(epsilons, dtype=np.float32),
        intervention_direction_names=np.asarray(direction_names, dtype=np.str_),
    )
    analysis_manifest = {
        "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
        "npz_path": str(analysis_path.resolve()),
        "npz_sha256": file_sha256(analysis_path),
        "checkpoint_fingerprint": context.latent_checkpoint_fingerprint,
        "formal_synergy_basis_fingerprint": context.formal_synergy_basis_fingerprint,
        "analysis_scope": "post_stage3_task_causal",
        "selected_synergy_source_fingerprint": context.selected_synergy_source_fingerprint,
        "stage3_checkpoint_payload_sha256": context.stage3_checkpoint_payload_sha256,
        "direction_source_binding": source_binding,
        "num_samples": len(samples),
        "latent_dim": int(directions.shape[1]),
    }
    analysis_manifest["manifest_fingerprint"] = canonical_json_sha256(analysis_manifest)
    analysis_manifest_path = output_dir / "analysis_inputs.json"
    _write_json_atomic(analysis_manifest_path, analysis_manifest)
    sample_plan = _sample_plan_payload(
        context=context,
        samples=samples,
        analysis_inputs_sha256=file_sha256(analysis_path),
        max_episode_steps=int(runtime.env.max_episode_steps),
        rollout_horizon_steps=int(config["rollout_horizon_steps"]),
    )
    sample_plan_path = output_dir / "sample_plan.json"
    _write_json_atomic(sample_plan_path, sample_plan)

    adapter = Stage3TaskCausalAdapter(
        runtime=runtime,
        samples=samples,
        rollout_horizon_steps=int(config["rollout_horizon_steps"]),
        intervention_duration_steps=int(config.get("intervention_duration_steps", 1)),
        latent_match_atol=float(config.get("latent_match_atol", 1e-5)),
        trunk_body_names=config.get("trunk_body_names", ("Full Body", "torso")),
    )
    adapter_config = {
        "synergy_evaluation": str(context.evaluation_report_path),
        "synergy_selection": str(context.selection_manifest_path),
        "family": context.family,
        "sample_plan": str(sample_plan_path.resolve()),
        "rollout_horizon_steps": adapter.rollout_horizon_steps,
        "intervention_duration_steps": adapter.intervention_duration_steps,
        "latent_match_atol": adapter.latent_match_atol,
        "trunk_body_names": list(adapter.trunk_body_names),
    }
    paired_dir = output_dir / "paired_rollouts"
    paired_manifest = produce_paired_rollouts(
        analysis_inputs=analysis_path,
        analysis_manifest=analysis_manifest_path,
        adapter=adapter,
        output_dir=paired_dir,
        base_seed=int(config.get("base_seed", 20_260_713)),
        adapter_import="musclemimic.badminton.stage3_task_causal:create_adapter",
        adapter_config=adapter_config,
    )
    causal_npz = output_dir / "causal_interventions.npz"
    causal_json = output_dir / "causal_interventions.json"
    causal_manifest = build_causal_rollout_artifact(
        analysis_inputs=analysis_path,
        analysis_manifest=analysis_manifest_path,
        baseline_records=paired_dir / "baseline_records.npz",
        perturbed_records=paired_dir / "perturbed_records.npz",
        rollout_manifest=paired_dir / "paired_rollout_manifest.json",
        output_npz=causal_npz,
        output_manifest=causal_json,
    )
    effects_manifest = build_mask_aware_task_effects(
        baseline_records=paired_dir / "baseline_records.npz",
        perturbed_records=paired_dir / "perturbed_records.npz",
        outcome_schemas=paired_manifest["outcome_schemas"],
        output_npz=output_dir / MASKED_EFFECTS_FILENAME,
        output_manifest=output_dir / MASKED_EFFECTS_MANIFEST_FILENAME,
    )
    promotion = build_task_causal_promotion(
        context=context,
        paired_rollout_manifest=paired_dir / "paired_rollout_manifest.json",
        causal_artifact_npz=causal_npz,
        causal_artifact_manifest=causal_json,
        task_effects_manifest=output_dir / MASKED_EFFECTS_MANIFEST_FILENAME,
        sample_plan=sample_plan_path,
        output=output_dir / PROMOTION_FILENAME,
    )
    return {
        "schema_version": TASK_CAUSAL_BRANCH_SCHEMA_VERSION,
        "family": context.family,
        "passed": bool(promotion["passed"]),
        "output_dir": str(output_dir),
        "promotion_metrics": str((output_dir / PROMOTION_FILENAME).resolve()),
        "causal_artifact": str(causal_npz.resolve()),
        "task_effects": str((output_dir / MASKED_EFFECTS_FILENAME).resolve()),
        "paired_rollout_manifest_fingerprint": paired_manifest["manifest_fingerprint"],
        "causal_artifact_manifest_fingerprint": causal_manifest["manifest_fingerprint"],
        "task_effects_manifest_fingerprint": effects_manifest["manifest_fingerprint"],
    }


def _build_synergy_only_promotion(
    *,
    output_dir: Path,
    contexts: Mapping[str, Stage3BranchContext],
    branch_results: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish formal latent-causal evidence for the selected synergy policy."""

    if set(contexts) != _FORMAL_TASK_CAUSAL_FAMILIES or set(branch_results) != (_FORMAL_TASK_CAUSAL_FAMILIES):
        raise ValueError("formal task-causal promotion requires exactly best_synergy")
    family = "best_synergy"
    result = branch_results[family]
    branch = _load_branch_promotion(output_dir / family / PROMOTION_FILENAME)
    gates = {
        "fixed_synergy_branch_complete": result.get("passed") is True,
        "selected_fixed_synergy_checkpoint_verified": (
            branch.get("family") == family
            and (branch.get("bindings") or {}).get("stage3_checkpoint_payload_sha256")
            == contexts[family].stage3_checkpoint_payload_sha256
        ),
        "full354_latent_intervention_not_applicable": True,
    }
    payload = {
        "schema_version": TASK_CAUSAL_SCHEMA_VERSION,
        "claim_scope": "post_stage3_selected_fixed_synergy_task_causal",
        "passed": all(gates.values()),
        "task_causal_complete": all(gates.values()),
        "fixed_synergy_branch_complete": gates["fixed_synergy_branch_complete"],
        "full354_latent_intervention_applicable": False,
        "full354_latent_intervention_not_applicable": True,
        "gates": gates,
        "paired_protocol": dict(protocol),
        "branches": {
            family: {
                "output_dir": result["output_dir"],
                "promotion_metrics": result["promotion_metrics"],
                "stage3_checkpoint_payload_sha256": contexts[family].stage3_checkpoint_payload_sha256,
                "latent_checkpoint_fingerprint": contexts[family].latent_checkpoint_fingerprint,
                "paired_rollout_manifest_fingerprint": result["paired_rollout_manifest_fingerprint"],
                "causal_artifact_manifest_fingerprint": result["causal_artifact_manifest_fingerprint"],
                "task_effects_manifest_fingerprint": result["task_effects_manifest_fingerprint"],
                "branch_promotion_fingerprint": branch["promotion_fingerprint"],
            }
        },
        "interpretation": (
            "Latent intervention is defined only for the selected fixed-synergy controller. "
            "The independent full354 policy has no latent coordinate and is explicitly not applicable."
        ),
    }
    payload["promotion_fingerprint"] = canonical_json_sha256(payload)
    _write_json_atomic(output_dir / PROMOTION_FILENAME, payload)
    validate_task_causal_promotion(output_dir / PROMOTION_FILENAME)
    return payload


def _load_branch_promotion(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != (TASK_CAUSAL_BRANCH_SCHEMA_VERSION):
        raise ValueError("unsupported task-causal branch promotion schema")
    supplied = payload.get("promotion_fingerprint")
    unbound = {key: value for key, value in payload.items() if key != "promotion_fingerprint"}
    gates = payload.get("gates")
    if (
        supplied != canonical_json_sha256(unbound)
        or not isinstance(gates, dict)
        or not gates
        or any(value is not True for value in gates.values())
        or payload.get("passed") is not True
        or payload.get("task_causal_complete") is not True
    ):
        raise ValueError("task-causal branch promotion is incomplete or changed")
    return payload


def validate_task_causal_promotion(path: str | Path) -> dict[str, Any]:
    """Revalidate selected-synergy latent-causal evidence and its branch."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = load_json_strict(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != TASK_CAUSAL_SCHEMA_VERSION:
        raise ValueError("unsupported final task-causal promotion schema")
    supplied = payload.get("promotion_fingerprint")
    unbound = {key: value for key, value in payload.items() if key != "promotion_fingerprint"}
    if supplied != canonical_json_sha256(unbound):
        raise ValueError("final task-causal promotion fingerprint mismatch")
    gates = payload.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(type(value) is not bool for value in gates.values())
        or payload.get("passed") is not all(gates.values())
        or payload.get("task_causal_complete") is not all(gates.values())
    ):
        raise ValueError("final task-causal promotion gates are incomplete")
    branches = payload.get("branches")
    if not isinstance(branches, dict) or set(branches) != _FORMAL_TASK_CAUSAL_FAMILIES:
        raise ValueError("final task-causal promotion lacks the selected synergy branch")
    for family, binding in branches.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"final task-causal branch binding {family} is malformed")
        branch = _load_branch_promotion(Path(str(binding.get("promotion_metrics", ""))))
        if branch.get("family") != family or branch.get("promotion_fingerprint") != binding.get(
            "branch_promotion_fingerprint"
        ):
            raise ValueError(f"final task-causal branch promotion changed: {family}")
    return payload


def _scout_samples(
    runtime: Stage3RuntimeBundle,
    *,
    points: Sequence[tuple[int, int]],
    latent_match_atol: float,
) -> list[TaskCausalSample]:
    placeholder = []
    latent_dim = int(runtime.env.lab_controller.latent_action_size)
    for feed_index, step_index in points:
        placeholder.append(
            TaskCausalSample(
                sample_uid="pending",
                feed_index=feed_index,
                step_index=step_index,
                feed_fingerprint=runtime.context.evaluation_feed_fingerprints[feed_index],
                baseline_latent=np.zeros((latent_dim,), dtype=np.float32),
            )
        )
    # Reuse the adapter's policy/latent semantics without resolving signal IDs
    # twice.  Temporary UID values are never published.
    scout = Stage3TaskCausalAdapter(
        runtime=runtime,
        samples=[
            TaskCausalSample(
                sample_uid=f"scout-{index}",
                feed_index=value.feed_index,
                step_index=value.step_index,
                feed_fingerprint=value.feed_fingerprint,
                baseline_latent=value.baseline_latent,
            )
            for index, value in enumerate(placeholder)
        ],
        rollout_horizon_steps=int(runtime.env.max_episode_steps),
        intervention_duration_steps=1,
        latent_match_atol=latent_match_atol,
    )
    result = []
    for feed_index, step_index in points:
        obs, _info = runtime.env.reset(feed_index=feed_index)
        terminated = truncated = False
        for _ in range(step_index):
            action = scout._policy(obs)
            obs, _reward, terminated, truncated, _info = runtime.env.step(action)
            if terminated or truncated:
                raise ValueError(f"sample point feed={feed_index} step={step_index} occurs after termination")
        scout._assert_pre_hit_snapshot(
            TaskCausalSample(
                sample_uid="scout-pre-hit-check",
                feed_index=feed_index,
                step_index=step_index,
                feed_fingerprint=runtime.context.evaluation_feed_fingerprints[feed_index],
                baseline_latent=np.zeros((latent_dim,), dtype=np.float32),
            )
        )
        latent = scout._effective_latent(obs).astype(np.float32)
        uid_payload = {
            "schema_version": "stage3_task_causal_sample_uid_v1",
            "feed_fingerprint": runtime.context.evaluation_feed_fingerprints[feed_index],
            "step_index": step_index,
            "selected_synergy_source_fingerprint": (runtime.context.selected_synergy_source_fingerprint),
        }
        result.append(
            TaskCausalSample(
                sample_uid=canonical_json_sha256(uid_payload),
                feed_index=feed_index,
                step_index=step_index,
                feed_fingerprint=runtime.context.evaluation_feed_fingerprints[feed_index],
                baseline_latent=latent,
            )
        )
    return result


def _sample_plan_payload(
    *,
    context: Stage3BranchContext,
    samples: Sequence[TaskCausalSample],
    analysis_inputs_sha256: str,
    max_episode_steps: int,
    rollout_horizon_steps: int,
) -> dict[str, Any]:
    remaining = [int(max_episode_steps) - sample.step_index for sample in samples]
    if not remaining or min(remaining) <= 0 or int(rollout_horizon_steps) < max(remaining):
        raise ValueError("sample plan horizon cannot cover the complete remaining Stage-3 episode")
    payload = {
        "schema_version": SAMPLE_PLAN_SCHEMA_VERSION,
        "family": context.family,
        "selected_synergy_source_fingerprint": context.selected_synergy_source_fingerprint,
        "stage3_checkpoint_payload_sha256": context.stage3_checkpoint_payload_sha256,
        "latent_checkpoint_fingerprint": context.latent_checkpoint_fingerprint,
        "formal_synergy_basis_fingerprint": context.formal_synergy_basis_fingerprint,
        "analysis_inputs_sha256": analysis_inputs_sha256,
        "snapshot_timing_contract": "strictly_before_first_real_hit_v1",
        "pre_hit_snapshot_verified": True,
        "max_episode_steps": int(max_episode_steps),
        "rollout_horizon_steps": int(rollout_horizon_steps),
        "remaining_episode_steps": remaining,
        "horizon_covers_impact_landing_recovery": True,
        "samples": [
            {
                "sample_uid": sample.sample_uid,
                "feed_index": sample.feed_index,
                "step_index": sample.step_index,
                "feed_fingerprint": sample.feed_fingerprint,
                "baseline_latent": np.asarray(sample.baseline_latent, dtype=float).tolist(),
            }
            for sample in samples
        ],
    }
    payload["manifest_fingerprint"] = canonical_json_sha256(payload)
    return payload


def _samples_from_plan(plan: Mapping[str, Any]) -> list[TaskCausalSample]:
    raw = plan.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Stage-3 sample plan has no samples")
    result = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Stage-3 sample plan entry must be an object")
        latent = np.asarray(entry.get("baseline_latent"), dtype=np.float32)
        if latent.ndim != 1 or latent.size == 0 or not np.all(np.isfinite(latent)):
            raise ValueError("Stage-3 sample-plan latent is malformed")
        result.append(
            TaskCausalSample(
                sample_uid=_require_sha256(entry.get("sample_uid"), "sample_uid"),
                feed_index=int(entry.get("feed_index")),
                step_index=int(entry.get("step_index")),
                feed_fingerprint=_require_sha256(
                    entry.get("feed_fingerprint"),
                    "feed_fingerprint",
                ),
                baseline_latent=latent,
            )
        )
    return result


def _validate_sample_points(
    value: Any,
    *,
    context: Stage3BranchContext,
) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sample_points must be a non-empty list")
    result = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("each sample point must contain feed_index and step_index")
        feed = int(entry.get("feed_index", -1))
        step = int(entry.get("step_index", -1))
        if feed < 0 or feed >= len(context.evaluation_feed_fingerprints) or step < 0:
            raise ValueError("sample point feed/step lies outside the held-out evaluator")
        result.append((feed, step))
    if len(set(result)) != len(result):
        raise ValueError("sample_points must be unique")
    return result


def _load_direction_source(
    value: Any,
    *,
    context: Stage3BranchContext,
    max_directions: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("direction_source must bind an existing latent analysis artifact")
    npz_path = resolve_task_causal_cli_path(value.get("analysis_inputs", ""), strict=True)
    manifest_path = resolve_task_causal_cli_path(
        value.get("analysis_manifest", ""),
        strict=True,
    )
    manifest = _load_self_fingerprinted_json(manifest_path)
    if manifest.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        raise ValueError("direction source must be latent_synergy_analysis_inputs_v2")
    if manifest.get("npz_sha256") != file_sha256(npz_path):
        raise ValueError("direction-source NPZ differs from its manifest")
    if manifest.get("checkpoint_fingerprint") != context.latent_checkpoint_fingerprint:
        raise ValueError("direction source uses a different low-level latent checkpoint")
    if manifest.get("formal_synergy_basis_fingerprint") != context.formal_synergy_basis_fingerprint:
        raise ValueError("direction source uses a different formal synergy basis")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if "intervention_directions" not in arrays.files:
            raise ValueError("direction source has no intervention_directions")
        directions = np.asarray(arrays["intervention_directions"], dtype=np.float32)
        names = (
            np.asarray(arrays["intervention_direction_names"]).astype(str).tolist()
            if "intervention_direction_names" in arrays.files
            else [f"latent_direction_{index}" for index in range(directions.shape[0])]
        )
    if (
        directions.ndim != 2
        or directions.shape[0] == 0
        or not np.all(np.isfinite(directions))
        or np.any(np.linalg.norm(directions, axis=1) <= 1e-12)
    ):
        raise ValueError("direction source contains invalid latent directions")
    count = min(int(max_directions), directions.shape[0])
    if count <= 0:
        raise ValueError("max_directions must be positive")
    directions = directions[:count]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    names = [str(value) for value in names[:count]]
    if len(names) != count or len(set(names)) != count:
        raise ValueError("direction names must be complete and unique")
    return (
        directions,
        names,
        {
            "analysis_inputs_path": str(npz_path),
            "analysis_inputs_sha256": file_sha256(npz_path),
            "analysis_manifest_path": str(manifest_path),
            "analysis_manifest_fingerprint": manifest["manifest_fingerprint"],
        },
    )


def _load_stage3_policy_mean(
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
) -> Callable[[np.ndarray], np.ndarray]:
    hidden = tuple(int(value) for value in metadata.get("hidden", metadata["config"]["hidden"]))
    import jax
    import jax.numpy as jnp
    import optax

    from environment.overall_environment.src.train_incoming_hit_mjx import (
        _mlp,
        init_agent,
        load_training_checkpoint,
    )

    config = dict(metadata["config"])
    template = init_agent(
        jax.random.PRNGKey(0),
        obs_size=int(metadata["obs_size"]),
        action_size=int(metadata["action_size"]),
        hidden=hidden,
        action_std_init=float(config.get("action_std_init", 0.35)),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.get("max_grad_norm", 0.5))),
        optax.adam(float(config.get("learning_rate", 3e-4))),
    )
    restored = load_training_checkpoint(
        checkpoint_path,
        agent_template=template,
        optimizer_state_template=optimizer.init(template),
    )
    mean = np.asarray(restored.obs_rms.mean, dtype=np.float64)
    variance = np.asarray(restored.obs_rms.var, dtype=np.float64)

    def policy(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        normalized = np.clip((obs - mean) / np.sqrt(variance + 1e-8), -10.0, 10.0)
        return np.asarray(
            jax.device_get(_mlp(restored.agent["policy"], jnp.asarray(normalized))),
            dtype=np.float64,
        )

    return policy


def _validate_event_vector(values: np.ndarray, schema: Mapping[str, Any]) -> None:
    vector = np.asarray(values, dtype=np.float64)
    names = list(schema["feature_names"])
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise ValueError("task event vector is malformed")
    for index, name in enumerate(names):
        if name.endswith("_present") and float(vector[index]) not in (0.0, 1.0):
            raise ValueError("all task event presence fields must be exact binary values")
    for contract in schema["masked_value_contracts"]:
        presence = float(vector[names.index(contract["presence_feature"])])
        measured = float(vector[names.index(contract["value_feature"])])
        if presence not in (0.0, 1.0):
            raise ValueError("event presence fields must be exact binary values")
        if presence == 0.0 and measured != float(contract["missing_sentinel"]):
            raise ValueError("missing event value differs from its explicit storage sentinel")


def _validate_event_matrix(values: np.ndarray, schema: Mapping[str, Any]) -> None:
    array = np.asarray(values, dtype=np.float64)
    width = len(schema["feature_names"])
    if array.ndim < 2 or array.shape[-1] != width or not np.all(np.isfinite(array)):
        raise ValueError("task event record matrix is malformed")
    for row in array.reshape((-1, width)):
        _validate_event_vector(row, schema)


def _load_self_fingerprinted_json(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    supplied = payload.get("manifest_fingerprint")
    unbound = {key: value for key, value in payload.items() if key != "manifest_fingerprint"}
    if supplied != canonical_json_sha256(unbound):
        raise ValueError(f"JSON fingerprint mismatch: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _unique_nonempty_strings(values: Sequence[str], label: str) -> list[str]:
    result = [str(value).strip() for value in values]
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _optional_array_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("snapshot arrays must be finite")
    return array.tolist()


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("snapshot arrays must be finite")
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("snapshot floats must be finite")
        return result
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str | int | float | bool) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("snapshot floats must be finite")
        return value
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"snapshot contains unsupported value type: {type(value).__name__}")


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return result


def resolve_task_causal_cli_path(value: Any, *, strict: bool = False) -> Path:
    """Resolve every config path against the CLI working directory."""

    text = str(value).strip()
    if not text:
        raise ValueError("task-causal config paths must be non-empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=bool(strict))


def config_template() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "synergy_evaluation": "artifacts/stage3_synergy_evaluate_report.json",
        "synergy_selection": "artifacts/latent_synergy_selection_manifest.json",
        "branches": {
            "best_synergy": {
                "direction_source": {
                    "analysis_inputs": "artifacts/best_synergy/analysis_inputs.npz",
                    "analysis_manifest": "artifacts/best_synergy/analysis_inputs.json",
                },
                "max_directions": 8,
            },
        },
        "sample_points": [
            {"feed_index": 0, "step_index": 50},
            {"feed_index": 1, "step_index": 50},
        ],
        "intervention_epsilons": [-0.5, 0.5],
        "intervention_duration_steps": 1,
        "rollout_horizon_steps": 420,
        "latent_match_atol": 1e-5,
        "trunk_body_names": ["Full Body", "torso"],
        "base_seed": 20260713,
        "output_dir": "outputs/synergy_v3/stage3_task_causal",
        "claim_gate": {
            "pre_stage3": "stage2_diagnostic_only_no_task_claim",
            "final": TASK_CAUSAL_SCHEMA_VERSION,
            "full354_latent_intervention": "not_applicable_no_latent_coordinate",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-template", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_template is not None:
        if args.config is not None or args.dry_run:
            raise ValueError("--write-template cannot be combined with evaluation arguments")
        if args.write_template.exists():
            raise FileExistsError(f"template target already exists: {args.write_template}")
        _write_json_atomic(args.write_template, config_template())
        print(args.write_template)
        return 0
    if args.config is None:
        raise ValueError("--config is required unless --write-template is used")
    report = run_task_causal(args.config, dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
