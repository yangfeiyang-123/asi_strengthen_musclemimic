"""Standalone GPU PPO trainer for the incoming-shuttle hit task.

PureJaxRL-style: the whole train iteration (rollout via vmapped env step +
GAE + minibatched PPO updates) is one jitted function. Independent from the
musclemimic trajectory-tracking pipeline.

Policy/value: shared-input MLPs; tanh-squashed Gaussian policy with a
state-independent learned log-std (same semantics as the CPU
``PolicyValueNet``). Observation normalization uses a running mean/std
updated from each rollout batch.

Run from the repository root (GPU env: source configs/env.sh):

    .venv/bin/python -m environment.overall_environment.src.train_incoming_hit_mjx \
        --spec experiments/posttrain/incoming_shuttle_hit_v1.yaml \
        --num-envs 512 --total-env-steps 2000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    EnvState,
    IncomingHitMjxEnv,
)
from environment.overall_environment.src.base_swing_bridge import (  # noqa: E402
    compose_selected_physical_correction,
    interpolate_correction_prior,
    selected_correction_window,
)

PHYSICAL_CORRECTION_POLICY_MODES = frozenset(
    {"selected_physical_correction", "graded_full_body_correction"}
)


_TEACHER_VERIFICATION_CONTEXT_BY_BACKEND = {
    "warp": "same_candidate_relocated_across_deterministic_warp_batch_lanes",
    "jax": (
        "same_candidate_relocated_across_deterministic_"
        "standard_mjx_jax_batch_lanes"
    ),
}
_TEACHER_VERIFICATION_SOURCE_BY_BACKEND = {
    "warp": "warp_training_backend_plus_independent_cpu_mujoco_quality_replay",
    "jax": (
        "standard_mjx_jax_training_backend_plus_independent_"
        "cpu_mujoco_quality_replay"
    ),
}
_OUTGOING_VELOCITY_SEMANTICS = (
    "post_control_step_after_all_physics_substeps"
)
_EVENT_REBOUND_CONTACT_SEMANTICS = (
    "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
)


class _CompletedEpisodeGateWindow:
    """Episode-weighted host-side metrics for Stage-3 curriculum gates.

    PPO rollouts are not aligned to episode boundaries, so the number of
    completed episodes in one rollout can vary from zero to hundreds.  A
    curriculum decision based on one rollout can therefore be dominated by a
    handful of lucky episodes.  This window keeps exact event counts, not an
    unweighted mean of per-rollout rates, and fails closed until enough
    completed episodes have been observed.
    """

    schema_version = "stage3_completed_episode_gate_window_v2"

    def __init__(
        self,
        *,
        min_completed_episodes: int = 512,
        max_iterations: int = 16,
        rows: list[dict[str, float]] | None = None,
    ) -> None:
        if int(min_completed_episodes) <= 0:
            raise ValueError("min_completed_episodes must be positive")
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        self.min_completed_episodes = int(min_completed_episodes)
        self.max_iterations = int(max_iterations)
        self.rows: list[dict[str, float]] = []
        for row in rows or []:
            self._append_row(row)

    @staticmethod
    def _validated_row(row: dict[str, float]) -> dict[str, float]:
        episodes = float(row.get("episodes", 0.0))
        hits = float(row.get("hits", 0.0))
        crossed = float(row.get("crossed", 0.0))
        falls = float(row.get("falls", 0.0))
        outgoing_z_hit_events = float(row.get("outgoing_z_hit_events", hits))
        positive_outgoing_z_hits = float(row.get("positive_outgoing_z_hits", 0.0))
        values = (
            episodes,
            hits,
            crossed,
            falls,
            outgoing_z_hit_events,
            positive_outgoing_z_hits,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("curriculum gate window contains non-finite counts")
        if episodes < 0.0 or any(
            value < 0.0 or value > episodes
            for value in (hits, crossed, falls)
        ):
            raise ValueError("curriculum gate window contains invalid event counts")
        if (
            outgoing_z_hit_events < 0.0
            or positive_outgoing_z_hits < 0.0
            or positive_outgoing_z_hits > outgoing_z_hit_events
        ):
            raise ValueError(
                "positive outgoing-z event count exceeds the hit-event count"
            )
        return {
            "episodes": episodes,
            "hits": hits,
            "crossed": crossed,
            "falls": falls,
            "outgoing_z_hit_events": outgoing_z_hit_events,
            "positive_outgoing_z_hits": positive_outgoing_z_hits,
        }

    def _append_row(self, row: dict[str, float]) -> None:
        self.rows.append(self._validated_row(row))
        if len(self.rows) > self.max_iterations:
            self.rows = self.rows[-self.max_iterations :]

    def update(self, metrics: dict[str, float]) -> None:
        episodes = max(0.0, float(metrics.get("episodes_finished", 0.0)))

        def count(rate_name: str) -> float:
            rate = float(np.clip(float(metrics.get(rate_name, 0.0)), 0.0, 1.0))
            return episodes * rate

        # Hit quality is observed at the contact transition, while the same
        # episode can finish in a later rollout.  Keep those event counts
        # independent from completed-episode counts so rollout boundaries do
        # not create the alternating high/zero quality gate seen in v37.
        completed_hits = count("hit_rate")
        outgoing_z_hit_events = max(
            0.0,
            float(metrics.get("hit_events", completed_hits)),
        )
        positive_outgoing_z_hits = max(
            0.0,
            float(
                metrics.get(
                    "positive_outgoing_z_hit_events",
                    outgoing_z_hit_events
                    * float(
                        np.clip(
                            float(
                                metrics.get(
                                    "positive_outgoing_z_rate_on_hit", 0.0
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    ),
                )
            ),
        )
        self._append_row(
            {
                "episodes": episodes,
                "hits": completed_hits,
                "crossed": count("crossed_net_rate"),
                "falls": count("fall_rate"),
                "outgoing_z_hit_events": outgoing_z_hit_events,
                "positive_outgoing_z_hits": positive_outgoing_z_hits,
            }
        )

    def summary(self) -> dict[str, float | bool]:
        episodes = float(sum(row["episodes"] for row in self.rows))

        def rate(event: str) -> float:
            if episodes <= 0.0:
                return 0.0
            return float(sum(row[event] for row in self.rows) / episodes)

        return {
            "episodes_finished": episodes,
            "hit_rate": rate("hits"),
            "crossed_net_rate": rate("crossed"),
            "fall_rate": rate("falls"),
            "positive_outgoing_z_rate_on_hit": (
                0.0
                if sum(row["outgoing_z_hit_events"] for row in self.rows)
                <= 0.0
                else float(
                    sum(row["positive_outgoing_z_hits"] for row in self.rows)
                    / sum(row["outgoing_z_hit_events"] for row in self.rows)
                )
            ),
            "ready": episodes >= float(self.min_completed_episodes),
        }

    def metrics_for_gate(self) -> dict[str, float]:
        summary = self.summary()
        # Stage3Curriculum already fails closed when episodes_finished == 0.
        # Preserve the measured rates for diagnostics while using that existing
        # contract to block decisions made with too few completed episodes.
        return {
            "episodes_finished": (float(summary["episodes_finished"]) if bool(summary["ready"]) else 0.0),
            "hit_rate": float(summary["hit_rate"]),
            "crossed_net_rate": float(summary["crossed_net_rate"]),
            "fall_rate": float(summary["fall_rate"]),
            "positive_outgoing_z_rate_on_hit": float(
                summary["positive_outgoing_z_rate_on_hit"]
            ),
        }

    def clear(self) -> None:
        self.rows.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "min_completed_episodes": self.min_completed_episodes,
            "max_iterations": self.max_iterations,
            "rows": [dict(row) for row in self.rows],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> _CompletedEpisodeGateWindow:
        schema_version = state.get("schema_version")
        if schema_version == "stage3_completed_episode_gate_window_v1":
            # v1 multiplied contact quality from one rollout by episode
            # completions from another.  Those rows cannot be repaired from
            # aggregate data, so resume safely with an empty evidence window.
            return cls(
                min_completed_episodes=int(state["min_completed_episodes"]),
                max_iterations=int(state["max_iterations"]),
            )
        if schema_version != cls.schema_version:
            raise ValueError("unsupported Stage-3 curriculum gate window schema")
        return cls(
            min_completed_episodes=int(state["min_completed_episodes"]),
            max_iterations=int(state["max_iterations"]),
            rows=[dict(row) for row in state.get("rows", [])],
        )


# ---------------------------------------------------------------------------
# networks (plain pytree params)
# ---------------------------------------------------------------------------


def _init_mlp(key, sizes):
    params = []
    for k_in, k_out in pairwise(sizes):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / k_in)
        params.append(
            {
                "w": jax.random.normal(sub, (k_in, k_out)) * scale,
                "b": jnp.zeros(k_out),
            }
        )
    return params


def _mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]


def init_agent(
    key,
    obs_size,
    action_size,
    hidden,
    action_std_init,
    *,
    policy_delta_hidden=(),
    policy_refinement_delta_hidden=(),
    policy_correction_hidden=(),
    correction_action_size=0,
    correction_std_init=(),
):
    k1, k2 = jax.random.split(key)
    policy = _init_mlp(k1, (obs_size, *hidden, action_size))
    value = _init_mlp(k2, (obs_size, *hidden, 1))
    policy[-1]["w"] = policy[-1]["w"] * 0.01
    agent = {
        "policy": policy,
        "value": value,
        "log_std": jnp.full((action_size,), jnp.log(action_std_init)),
    }
    delta_hidden = tuple(int(size) for size in policy_delta_hidden)
    if delta_hidden:
        if any(size <= 0 for size in delta_hidden):
            raise ValueError("policy_delta_hidden sizes must be positive")
        # fold_in deliberately leaves the historical policy/value RNG split
        # untouched, so enabling an adapter cannot silently change the base
        # actor that is about to be initialized from a checkpoint.
        delta_key = jax.random.fold_in(key, 0xD371A)
        policy_delta = _init_mlp(delta_key, (obs_size, *delta_hidden, action_size))
        # Exact zero initialization is the identity map: before its first PPO
        # update, the composed policy mean is bit-for-bit the inherited actor.
        policy_delta[-1]["w"] = jnp.zeros_like(policy_delta[-1]["w"])
        policy_delta[-1]["b"] = jnp.zeros_like(policy_delta[-1]["b"])
        agent["policy_delta"] = policy_delta
    refinement_hidden = tuple(int(size) for size in policy_refinement_delta_hidden)
    if refinement_hidden:
        if any(size <= 0 for size in refinement_hidden):
            raise ValueError("policy_refinement_delta_hidden sizes must be positive")
        refinement_key = jax.random.fold_in(key, 0xD371B)
        refinement_delta = _init_mlp(
            refinement_key,
            (obs_size, *refinement_hidden, action_size),
        )
        # Phase-B wrist repair must start as the exact Phase-A actor.  Only the
        # new refinement branch is initialized to zero; the learned Phase-A
        # policy_delta is imported and then frozen by its update contract.
        refinement_delta[-1]["w"] = jnp.zeros_like(refinement_delta[-1]["w"])
        refinement_delta[-1]["b"] = jnp.zeros_like(refinement_delta[-1]["b"])
        agent["policy_refinement_delta"] = refinement_delta
    correction_hidden = tuple(int(size) for size in policy_correction_hidden)
    if correction_hidden:
        if any(size <= 0 for size in correction_hidden):
            raise ValueError("policy_correction_hidden sizes must be positive")
        selected_size = int(correction_action_size)
        if selected_size <= 0:
            raise ValueError("policy_correction_hidden requires correction_action_size")
        correction_key = jax.random.fold_in(key, 0xD371C)
        correction = _init_mlp(
            correction_key,
            (obs_size, *correction_hidden, selected_size),
        )
        correction[-1]["w"] = jnp.zeros_like(correction[-1]["w"])
        correction[-1]["b"] = jnp.zeros_like(correction[-1]["b"])
        std_init = jnp.asarray(correction_std_init, dtype=jnp.float32)
        if std_init.shape == ():
            std_init = jnp.full((selected_size,), std_init)
        if std_init.shape != (selected_size,):
            raise ValueError("correction_std_init must contain one value per selected action")
        agent["policy_correction"] = correction
        agent["correction_log_std"] = jnp.log(std_init)
    return agent


def _dist(params, obs):
    mean = _mlp(params["policy"], obs)
    if "policy_delta" in params:
        mean = mean + _mlp(params["policy_delta"], obs)
    if "policy_refinement_delta" in params:
        mean = mean + _mlp(params["policy_refinement_delta"], obs)
    # Distal-repair policies can intentionally make every frozen body action
    # effectively deterministic while retaining Gaussian exploration on the
    # selected wrist outputs.  The historical -5 floor (std ~= 6.7e-3) still
    # injected visible noise into all 345 frozen body channels.
    std = jnp.exp(jnp.clip(params["log_std"], -12.0, 1.0))
    return mean, std


def _inherited_policy_mean(params, obs):
    """Full inherited Stage-3 logits, excluding the independent correction."""

    mean = _mlp(params["policy"], obs)
    if "policy_delta" in params:
        mean = mean + _mlp(params["policy_delta"], obs)
    if "policy_refinement_delta" in params:
        mean = mean + _mlp(params["policy_refinement_delta"], obs)
    return mean


def _selected_correction_dist(params, obs, *, std_min, std_max):
    if "policy_correction" not in params or "correction_log_std" not in params:
        raise ValueError("selected physical correction parameters are missing")
    mean = _mlp(params["policy_correction"], obs)
    std = jnp.exp(params["correction_log_std"])
    std = jnp.clip(std, jnp.asarray(std_min), jnp.asarray(std_max))
    return mean, std


def _normal_logprob(mean, std, raw):
    base = -0.5 * (((raw - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    return jnp.sum(base, axis=-1)


def sample_action(params, obs, key, *, squash_action: bool = True):
    """Sample an action and return the underlying Gaussian sample.

    Full-muscle policies retain the historical tanh-squashed distribution.
    Stage-3 policies pass ``squash_action=False``: their output is the raw
    Gaussian latent ``u`` and LAB performs the one and only tanh in
    ``z = mu + lambda * sigma * tanh(u)``.
    """
    mean, std = _dist(params, obs)
    raw = mean + std * jax.random.normal(key, mean.shape)
    if squash_action:
        action = jnp.tanh(raw)
        logp = _tanh_normal_logprob(mean, std, raw, action)
    else:
        action = raw
        logp = _normal_logprob(mean, std, raw)
    return action, raw, logp


def _tanh_normal_logprob(mean, std, raw, squashed):
    base = -0.5 * (((raw - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    correction = jnp.log(1.0 - squashed**2 + 1e-6)
    return jnp.sum(base - correction, axis=-1)


def evaluate_actions(
    params,
    obs,
    raw_actions,
    *,
    squash_action: bool = True,
    entropy_action_mask: jax.Array | None = None,
):
    mean, std = _dist(params, obs)
    if squash_action:
        squashed = jnp.tanh(raw_actions)
        logp = _tanh_normal_logprob(mean, std, raw_actions, squashed)
    else:
        logp = _normal_logprob(mean, std, raw_actions)
    entropy_per_action = 0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(std)
    if entropy_action_mask is not None:
        entropy_per_action = entropy_per_action * jnp.asarray(entropy_action_mask)[None, :]
    entropy = jnp.sum(entropy_per_action, axis=-1)
    value = _mlp(params["value"], obs)[..., 0]
    return logp, entropy, value


def bounded_ppo_ratio(
    log_prob: jax.Array,
    log_prob_old: jax.Array,
    *,
    max_abs_log_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exponentiate a bounded PPO log-ratio and expose guard diagnostics.

    A Stage-3 correction policy can contain hundreds of low-variance action
    dimensions.  Summing their Gaussian log probabilities makes an otherwise
    finite policy update large enough to overflow ``exp`` before PPO's clipped
    surrogate can discard it.  Bounding in log space preserves the ordinary
    PPO computation around ratio one while keeping extreme out-of-trust-region
    minibatches finite and explicitly measurable.
    """

    limit = jnp.asarray(max_abs_log_ratio, dtype=log_prob.dtype)
    log_ratio = log_prob - log_prob_old
    guarded = jnp.clip(log_ratio, -limit, limit)
    ratio = jnp.exp(guarded)
    guard_applied = (jnp.abs(log_ratio) > limit).astype(jnp.float32)
    return ratio, log_ratio, guard_applied


def post_update_logprob_audit(
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
    *,
    max_abs_log_ratio: float,
) -> dict[str, jax.Array]:
    """Measure policy drift after the optimizer has changed the actor."""

    log_ratio = jnp.asarray(new_log_prob) - jnp.asarray(old_log_prob)
    return {
        "ppo_post_update_log_ratio_abs_max": jnp.max(jnp.abs(log_ratio)),
        "ppo_post_update_log_ratio_abs_mean": jnp.mean(jnp.abs(log_ratio)),
        "ppo_post_update_kl_estimate": jnp.mean(-log_ratio),
        "ppo_post_update_ratio_guard_fraction": jnp.mean(
            (jnp.abs(log_ratio) > float(max_abs_log_ratio)).astype(jnp.float32)
        ),
    }


# ---------------------------------------------------------------------------
# running obs normalization
# ---------------------------------------------------------------------------


class ObsRms(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @staticmethod
    def create(obs_size):
        return ObsRms(jnp.zeros(obs_size), jnp.ones(obs_size), jnp.asarray(1e-4))

    def update(self, batch: jnp.ndarray) -> ObsRms:
        batch = batch.reshape(-1, batch.shape[-1])
        b_mean = batch.mean(0)
        b_var = batch.var(0)
        b_count = batch.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta**2 * self.count * b_count / tot
        return ObsRms(mean, m2 / tot, tot)

    def normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip((obs - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class TrainConfig(NamedTuple):
    num_envs: int = 512
    rollout_steps: int = 64
    total_env_steps: int = 2_000_000
    update_epochs: int = 4
    num_minibatches: int = 8
    minibatch_size: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    max_grad_norm: float = 0.5
    max_abs_log_ratio: float = 10.0
    max_post_update_ratio_guard_fraction: float = 1.0
    max_post_update_kl_estimate: float = 1.0e9
    hidden: tuple = (256, 256)
    action_std_init: float = 0.35
    policy_update_mode: str = "full_network"
    policy_trainable_action_indices: tuple = ()
    policy_delta_hidden: tuple = ()
    policy_refinement_delta_hidden: tuple = ()
    policy_correction_hidden: tuple = ()
    correction_physical_scales: tuple = ()
    correction_std_init: tuple = ()
    correction_std_min: tuple = ()
    correction_std_max: tuple = ()
    reset_correction_std_on_actor_initialization: bool = False
    correction_window_open_s: float = 0.70
    correction_window_close_s: float = -0.10
    correction_window_smoothing_s: float = 0.05
    teacher_action_prior_mode: str = "none"
    teacher_prior_time_to_intercept_s: tuple = ()
    teacher_prior_correction_raw: tuple = ()
    quality_success_min_outgoing_z_m_s: float = 0.5
    quality_success_min_forward_m_s: float = 2.0
    quality_success_min_predicted_net_clearance_m: float = -1.0e9
    quality_success_min_return_direction_signed_score: float = -1.0
    quality_success_min_racket_face_forward_alignment: float = -1.0
    quality_success_require_episode_no_fall: bool = False
    quality_imitation_mode: str = "strict_success"
    quality_imitation_min_weight: float = 0.0
    quality_imitation_forward_softness_m_s: float = 1.0
    quality_imitation_vertical_softness_m_s: float = 0.75
    quality_imitation_clearance_softness_m: float = 0.75
    quality_imitation_direction_softness: float = 0.10
    quality_imitation_require_episode_no_fall: bool = False
    teacher_bc_pretrain_steps: int = 0
    teacher_bc_batch_size: int = 256
    teacher_bc_learning_rate: float = 3.0e-4
    teacher_bc_initial_coef: float = 0.0
    teacher_bc_final_coef: float = 0.0
    teacher_bc_decay_steps: int = 0
    freeze_observation_normalizer: bool = False
    frozen_action_std: float | None = None
    freeze_trainable_action_std: bool = False
    successful_action_imitation_coef: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class QualityTeacherDataset:
    observation_normalized: np.ndarray
    correction_raw: np.ndarray
    sample_weight: np.ndarray
    time_to_intercept_s: np.ndarray
    binding: dict[str, Any]


def load_quality_teacher_dataset(
    path: str | Path,
    *,
    selected_action_indices: tuple[int, ...],
    correction_physical_scales: tuple[float, ...],
    source_checkpoint_sha256: str | None = None,
    source_base_phase_advance_s: float | None = None,
    allow_cpu_certified_exploration_prior: bool = False,
) -> QualityTeacherDataset:
    """Load a robust teacher, or an explicitly requested exploration prior.

    The exploration path is deliberately opt-in and retains a distinct
    binding schema.  It accepts a CPU-quality trajectory only when the same
    candidate has at least one real upward/forward Warp replay, while keeping
    ``training_backend_quality_verified`` false.  This gives PPO a bounded
    contact seed without ever promoting the artifact as a quality teacher.
    """

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"teacher trajectory is missing: {dataset_path}")
    report_path = dataset_path.parent / "cem_report.json"
    if not report_path.is_file():
        raise ValueError("teacher trajectory has no sibling cem_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    native_report_schema = "stage3_single_feed_mjx_cem_report_v3"
    cross_backend_report_schema = "stage3_cross_backend_quality_teacher_report_v3"
    outgoing_velocity_semantics = _OUTGOING_VELOCITY_SEMANTICS
    event_rebound_contact_semantics = _EVENT_REBOUND_CONTACT_SEMANTICS
    supported_report_schemas = {native_report_schema, cross_backend_report_schema}
    if (
        not isinstance(report, dict)
        or report.get("schema_version") not in supported_report_schemas
    ):
        raise ValueError("teacher CEM report schema is incompatible")
    exploration_prior = bool(allow_cpu_certified_exploration_prior)
    if exploration_prior and report.get("schema_version") != native_report_schema:
        raise ValueError("exploration prior requires a native training-backend CEM report")
    verified = dict(report.get("verified_metrics", {}) or {})
    report_contract = dict(report.get("contract", {}) or {})
    if report_contract.get("outgoing_velocity_semantics") != (
        outgoing_velocity_semantics
    ):
        raise ValueError(
            "teacher quality was not measured from the settled post-control outgoing velocity"
        )
    if report_contract.get("event_rebound_contact_semantics") != (
        event_rebound_contact_semantics
    ):
        raise ValueError("teacher permits a double-applied stringbed impact")
    contract_feed_fingerprint = str(report_contract.get("feed_fingerprint", ""))
    contract_swing_phase_advance_s = float(
        report_contract.get("swing_phase_advance_s", math.nan)
    )
    if (
        not contract_feed_fingerprint
        or not math.isfinite(contract_swing_phase_advance_s)
        or contract_swing_phase_advance_s < 0.0
        or report_contract.get("swing_phase_timing_semantics")
        != (
            "frozen_base_swing_phase_advance_applied_identically_to_search_"
            "backend_and_cpu_replays"
        )
    ):
        raise ValueError("teacher CEM report has an incompatible feed/timing contract")
    if exploration_prior:
        if (
            report.get("passed") is not False
            or report.get("mjx_teacher_passed") is not False
            or report.get("cpu_replay_passed") is not True
            or report_contract.get("mjx_impl") != "warp"
            or float(verified.get("teacher_success_rate", 0.0)) <= 0.0
            or float(verified.get("return_quality_rate", 0.0)) <= 0.0
            or float(verified.get("positive_outgoing_z_rate", 0.0)) <= 0.0
            or float(verified.get("positive_outgoing_forward_rate", 0.0)) <= 0.0
            or float(verified.get("high_region_contact_rate", 0.0)) <= 0.0
            or float(verified.get("no_fall_rate", 0.0)) < 1.0
        ):
            raise ValueError(
                "exploration prior lacks CPU quality plus a real non-robust "
                "upward/forward Warp replay"
            )
    elif report.get("passed") is not True or verified.get("teacher_success") is not True:
        raise ValueError("teacher CEM report did not pass robust return-success gates")
    report_schema = str(report["schema_version"])
    training_backend_evidence: dict[str, Any] | None = None
    if report_schema == cross_backend_report_schema:
        training_backend_evidence = dict(report.get("cross_backend_evidence", {}) or {})
        training_metrics = dict(
            training_backend_evidence.get("independent_mjx_replay_metrics", {}) or {}
        )
        deployment_gate = dict(
            training_backend_evidence.get("deployment_quality_gate", {}) or {}
        )
        search_margin = dict(
            training_backend_evidence.get("search_quality_margin", {}) or {}
        )
        verification_context_semantics = training_backend_evidence.get(
            "verification_context_semantics"
        )
        verification_group_indices = training_backend_evidence.get(
            "verification_group_indices"
        )
        training_backend = str(
            training_backend_evidence.get("training_backend", "")
        )
        expected_verification_context = (
            _TEACHER_VERIFICATION_CONTEXT_BY_BACKEND.get(training_backend)
        )
        expected_verification_source = (
            _TEACHER_VERIFICATION_SOURCE_BY_BACKEND.get(training_backend)
        )
        required_rate = float(deployment_gate.get("min_replica_rate", math.nan))
        min_outgoing_z = float(
            deployment_gate.get("min_outgoing_z_m_s", math.nan)
        )
        min_forward = float(deployment_gate.get("min_forward_m_s", math.nan))
        search_min_outgoing_z = float(
            search_margin.get("min_outgoing_z_m_s", math.nan)
        )
        search_min_forward = float(
            search_margin.get("min_forward_m_s", math.nan)
        )
        if (
            report.get("mjx_teacher_passed") is not True
            or expected_verification_context is None
            or expected_verification_source is None
            or report_contract.get("mjx_impl") != training_backend
            or report.get("verification_source") != expected_verification_source
            or training_backend_evidence.get("training_backend_quality_verified") is not True
            or training_backend_evidence.get("outgoing_velocity_semantics")
            != outgoing_velocity_semantics
            or training_backend_evidence.get("event_rebound_contact_semantics")
            != event_rebound_contact_semantics
            or training_backend_evidence.get("feed_fingerprint")
            != contract_feed_fingerprint
            or not math.isclose(
                float(
                    training_backend_evidence.get(
                        "training_backend_swing_phase_advance_s",
                        math.nan,
                    )
                ),
                contract_swing_phase_advance_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            or not math.isclose(
                float(
                    training_backend_evidence.get(
                        "cpu_replay_swing_phase_advance_s",
                        math.nan,
                    )
                ),
                contract_swing_phase_advance_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            or training_metrics.get("teacher_success") is not True
            or training_metrics.get("return_quality") is not True
            or training_metrics.get("event_rebound") is not True
            or training_metrics.get("high_region_contact") is not True
            or training_metrics.get("no_fall") is not True
            or not math.isfinite(required_rate)
            or not 0.0 < required_rate <= 1.0
            or not math.isfinite(min_outgoing_z)
            or min_outgoing_z < 0.5
            or not math.isfinite(min_forward)
            or min_forward < 2.0
            or search_margin.get("semantics")
            != "same_replica_training_backend_margin_gate"
            or verification_context_semantics != expected_verification_context
            or not isinstance(verification_group_indices, list)
            or len(verification_group_indices) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in verification_group_indices
            )
            or len(set(verification_group_indices))
            != len(verification_group_indices)
            or not math.isfinite(search_min_outgoing_z)
            or search_min_outgoing_z < min_outgoing_z
            or not math.isfinite(search_min_forward)
            or search_min_forward < min_forward
            or float(training_metrics.get("teacher_success_rate", 0.0)) + 1.0e-9
            < required_rate
            or float(training_metrics.get("return_quality_rate", 0.0)) + 1.0e-9
            < required_rate
            or float(training_metrics.get("positive_outgoing_z_rate", 0.0)) + 1.0e-9
            < required_rate
            or float(training_metrics.get("positive_outgoing_forward_rate", 0.0)) + 1.0e-9
            < required_rate
        ):
            raise ValueError(
                "cross-backend teacher lacks robust upward-forward quality on the "
                "actual training backend"
            )
    cpu_audit = dict(report.get("cpu_replay_audit", {}) or {})
    if (
        report.get("cpu_replay_event_equivalent") is not True
        or cpu_audit.get("hit") is not True
        or cpu_audit.get("event_rebound") is not True
        or cpu_audit.get("body_fall") is not False
        or cpu_audit.get("feed_fingerprint") != contract_feed_fingerprint
        or not math.isclose(
            float(cpu_audit.get("swing_phase_advance_s", math.nan)),
            contract_swing_phase_advance_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise ValueError("teacher CEM report did not pass the independent CPU replay gate")
    cpu_quality_audit = dict(report.get("cpu_gated_best_audit", {}) or {})
    if exploration_prior and (
        cpu_quality_audit.get("cpu_quality_passed") is not True
        or cpu_quality_audit.get("hit") is not True
        or cpu_quality_audit.get("event_rebound") is not True
        or cpu_quality_audit.get("high_region_contact") is not True
        or cpu_quality_audit.get("body_fall") is not False
        or cpu_quality_audit.get("feed_fingerprint") != contract_feed_fingerprint
        or float(cpu_quality_audit.get("outgoing_z_m_s", -math.inf)) < 0.5
        or float(cpu_quality_audit.get("outgoing_forward_m_s", -math.inf)) < 2.0
        or not math.isclose(
            float(cpu_quality_audit.get("swing_phase_advance_s", math.nan)),
            contract_swing_phase_advance_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise ValueError("exploration prior did not pass the full CPU quality gate")
    timing_evidence = dict(report.get("base_timing_transfer", {}) or {})
    if exploration_prior and not timing_evidence:
        # The CEM report's configured phase belongs to the search spec and can
        # differ from the actor checkpoint's actual frozen-base timing.  The
        # production trainer has the inherited checkpoint open already, so it
        # supplies that sealed control-manifest value here.  Standalone audit
        # callers retain the report value as a compatibility fallback; actor
        # initialization will independently compare the evidence to the real
        # checkpoint and fail closed if it is not exact.
        source_phase = float(
            report_contract.get("configured_swing_phase_advance_s", math.nan)
            if source_base_phase_advance_s is None
            else source_base_phase_advance_s
        )
        runtime_phase = contract_swing_phase_advance_s
        if not math.isfinite(source_phase):
            raise ValueError("exploration prior lacks its source base timing")
        timing_evidence = {
            "schema_version": "stage3_teacher_verified_base_timing_transfer_v1",
            "verification_source": "independent_cpu_mujoco_quality_replay_at_runtime_timing",
            "cpu_quality_verified": True,
            "source_phase_advance_s": source_phase,
            "runtime_phase_advance_s": runtime_phase,
        }
        timing_evidence["evidence_sha256"] = _stable_json_hash(timing_evidence)
    if timing_evidence:
        recorded_evidence_sha256 = timing_evidence.get("evidence_sha256")
        unhashed_timing_evidence = dict(timing_evidence)
        unhashed_timing_evidence.pop("evidence_sha256", None)
        if (
            timing_evidence.get("schema_version")
            != "stage3_teacher_verified_base_timing_transfer_v1"
            or timing_evidence.get("verification_source")
            != "independent_cpu_mujoco_quality_replay_at_runtime_timing"
            or timing_evidence.get("cpu_quality_verified") is not True
            or recorded_evidence_sha256 != _stable_json_hash(unhashed_timing_evidence)
        ):
            raise ValueError("teacher base-timing transfer evidence is incompatible")
        for name in ("source_phase_advance_s", "runtime_phase_advance_s"):
            if not math.isfinite(float(timing_evidence.get(name, math.nan))):
                raise ValueError("teacher base-timing transfer phase is invalid")
    high_region_contract = dict(report_contract.get("high_region_contact", {}) or {})
    if (
        high_region_contract.get("semantics")
        != "soft_window_teacher_gate_not_exact_apex"
        or (
            not exploration_prior
            and verified.get("high_region_contact") is not True
        )
        or (
            exploration_prior
            and dict(report.get("best_search_metrics", {}) or {}).get(
                "high_region_contact"
            )
            is not True
        )
    ):
        raise ValueError("teacher CEM report did not pass the high-region contact window")
    trace_report = dict(report.get("teacher_trace", {}) or {})
    recorded_path = Path(str(trace_report.get("trace_path", ""))).expanduser().resolve()
    if recorded_path != dataset_path:
        raise ValueError("teacher trajectory path differs from its CEM report")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if trace_report.get("trace_sha256") != dataset_sha256:
        raise ValueError("teacher trajectory content hash differs from its CEM report")
    selected_replica_metrics = dict(trace_report.get("selected_replica_metrics", {}) or {})
    if selected_replica_metrics.get("teacher_success") is not True:
        raise ValueError("saved teacher trace is not itself a return-success replica")
    if selected_replica_metrics.get("high_region_contact") is not True:
        raise ValueError("saved teacher trace contacted outside the allowed high region")

    required = {
        "observation_normalized", "correction_raw", "correction_window",
        "time_to_intercept_s",
        "event_rebound", "outgoing_shuttle_velocity_xyz_m_s",
        "selected_action_indices", "feed_fingerprint",
        "swing_phase_advance_s",
        "physical_scales",
        "source_checkpoint_sha256", "search_contract_sha256",
        "outgoing_velocity_semantics",
        "event_rebound_contact_semantics",
    }
    with np.load(dataset_path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError("teacher trajectory is missing fields: " + ", ".join(missing))
        observations = np.asarray(payload["observation_normalized"], dtype=np.float32)
        corrections = np.asarray(payload["correction_raw"], dtype=np.float32)
        window = np.asarray(payload["correction_window"], dtype=np.float32)
        time_to_intercept_s = np.asarray(
            payload["time_to_intercept_s"], dtype=np.float32
        )
        rebound = np.asarray(payload["event_rebound"], dtype=bool)
        outgoing = np.asarray(payload["outgoing_shuttle_velocity_xyz_m_s"], dtype=np.float32)
        recorded_indices = tuple(
            int(value) for value in np.asarray(payload["selected_action_indices"]).tolist()
        )
        recorded_physical_scales = np.asarray(
            payload["physical_scales"], dtype=np.float32
        )
        feed_fingerprint = str(np.asarray(payload["feed_fingerprint"]).item())
        swing_phase_advance_s = float(
            np.asarray(payload["swing_phase_advance_s"]).item()
        )
        recorded_source_sha256 = str(np.asarray(payload["source_checkpoint_sha256"]).item())
        search_contract_sha256 = str(np.asarray(payload["search_contract_sha256"]).item())
        dataset_outgoing_velocity_semantics = str(
            np.asarray(payload["outgoing_velocity_semantics"]).item()
        )
        dataset_event_rebound_contact_semantics = str(
            np.asarray(payload["event_rebound_contact_semantics"]).item()
        )
        trace_schema_version = (
            None
            if "trace_schema_version" not in payload.files
            else str(np.asarray(payload["trace_schema_version"]).item())
        )
        dataset_source_phase = (
            None
            if "source_base_phase_advance_s" not in payload.files
            else float(np.asarray(payload["source_base_phase_advance_s"]).item())
        )
        dataset_runtime_phase = (
            None
            if "runtime_base_phase_advance_s" not in payload.files
            else float(np.asarray(payload["runtime_base_phase_advance_s"]).item())
        )

    if report_schema == cross_backend_report_schema and trace_schema_version != (
        "stage3_cross_backend_training_backend_and_cpu_quality_teacher_v3"
    ):
        raise ValueError("cross-backend teacher trajectory schema is incompatible")
    if dataset_outgoing_velocity_semantics != outgoing_velocity_semantics:
        raise ValueError("teacher trajectory has incompatible outgoing-velocity semantics")
    if dataset_event_rebound_contact_semantics != event_rebound_contact_semantics:
        raise ValueError("teacher trajectory permits a double-applied stringbed impact")
    if feed_fingerprint != contract_feed_fingerprint:
        raise ValueError("teacher trajectory feed differs from its CEM contract")
    if (
        not math.isfinite(swing_phase_advance_s)
        or not math.isclose(
            swing_phase_advance_s,
            contract_swing_phase_advance_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise ValueError("teacher trajectory timing differs from its CEM contract")

    if timing_evidence:
        if (
            not exploration_prior
            and dataset_source_phase != float(timing_evidence["source_phase_advance_s"])
        ):
            raise ValueError("teacher source phase differs from its timing evidence")
        if (
            not exploration_prior
            and dataset_runtime_phase != float(timing_evidence["runtime_phase_advance_s"])
        ):
            raise ValueError("teacher runtime phase differs from its timing evidence")
    elif dataset_source_phase is not None or dataset_runtime_phase is not None:
        raise ValueError("teacher trajectory has unbound base-timing fields")

    if recorded_indices != tuple(int(value) for value in selected_action_indices):
        raise ValueError("teacher selected-action mapping differs from the policy contract")
    expected_physical_scales = np.asarray(
        correction_physical_scales, dtype=np.float32
    )
    expected_shape = (len(recorded_indices),)
    if (
        recorded_physical_scales.shape != expected_shape
        or expected_physical_scales.shape != expected_shape
        or not np.isfinite(recorded_physical_scales).all()
        or not np.isfinite(expected_physical_scales).all()
        or np.any(recorded_physical_scales <= 0.0)
        or np.any(expected_physical_scales <= 0.0)
    ):
        raise ValueError("teacher/runtime correction physical scales are invalid")
    if not np.allclose(
        recorded_physical_scales,
        expected_physical_scales,
        rtol=1.0e-6,
        atol=1.0e-8,
    ):
        raise ValueError(
            "teacher correction physical scales differ from the policy contract"
        )
    if source_checkpoint_sha256 is not None and recorded_source_sha256 != source_checkpoint_sha256:
        raise ValueError("teacher was generated from a different inherited checkpoint")
    if observations.ndim != 2 or corrections.shape != (observations.shape[0], len(recorded_indices)):
        raise ValueError("teacher observation/correction arrays have incompatible shapes")
    if (
        window.shape != (observations.shape[0],)
        or rebound.shape != window.shape
        or time_to_intercept_s.shape != window.shape
    ):
        raise ValueError("teacher window/event arrays have incompatible shapes")
    if outgoing.shape != (observations.shape[0], 3):
        raise ValueError("teacher outgoing-velocity evidence has an incompatible shape")
    if not all(
        np.isfinite(value).all()
        for value in (
            observations,
            corrections,
            window,
            time_to_intercept_s,
            outgoing,
        )
    ):
        raise ValueError("teacher trajectory contains non-finite values")
    hit_steps = np.flatnonzero(rebound)
    if hit_steps.size == 0 or float(np.max(outgoing[hit_steps, 2])) < 0.5:
        raise ValueError("teacher trace has no positive vertical real rebound")
    weights = np.square(np.clip(window, 0.0, 1.0)).astype(np.float32)
    keep = weights > 1.0e-4
    if int(keep.sum()) < 16:
        raise ValueError("teacher trace has too few correction-window samples")
    observations = observations[keep]
    corrections = corrections[keep]
    weights = weights[keep]
    time_to_intercept_s = time_to_intercept_s[keep]
    order = np.argsort(time_to_intercept_s, kind="stable")
    observations = observations[order]
    corrections = corrections[order]
    weights = weights[order]
    time_to_intercept_s = time_to_intercept_s[order]
    if np.any(np.diff(time_to_intercept_s) <= 0.0):
        raise ValueError("teacher time-to-intercept knots must be strictly monotonic")
    binding: dict[str, Any] = {
        "schema_version": (
            "stage3_cpu_certified_exploration_prior_binding_v1"
            if exploration_prior
            else "stage3_quality_teacher_dataset_binding_v1"
        ),
        "trajectory_path": str(dataset_path),
        "trajectory_sha256": dataset_sha256,
        "cem_report_path": str(report_path.resolve()),
        "cem_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "search_contract_sha256": search_contract_sha256,
        "source_checkpoint_sha256": recorded_source_sha256,
        "feed_fingerprint": feed_fingerprint,
        "swing_phase_advance_s": swing_phase_advance_s,
        "selected_action_indices": list(recorded_indices),
        "correction_physical_scales": recorded_physical_scales.tolist(),
        "sample_count": int(observations.shape[0]),
        "robust_teacher_success_rate": float(verified.get("teacher_success_rate", 1.0)),
        "cpu_quality_verified": True,
        "verification_source": (
            "cpu_quality_plus_nonrobust_warp_exploration_replay"
            if exploration_prior
            else str(report.get("verification_source", "mjx_plus_cpu_replay"))
        ),
        "verified_outgoing_z_m_s": float(
            cpu_quality_audit["outgoing_z_m_s"]
            if exploration_prior
            else verified["outgoing_z_m_s"]
        ),
        "verified_outgoing_forward_m_s": float(
            cpu_quality_audit["outgoing_forward_m_s"]
            if exploration_prior
            else verified["outgoing_forward_m_s"]
        ),
        "outgoing_velocity_semantics": outgoing_velocity_semantics,
        "event_rebound_contact_semantics": event_rebound_contact_semantics,
    }
    if exploration_prior:
        binding.update(
            {
                "prior_role": "bounded_exploration_prior_not_quality_teacher",
                "quality_teacher": False,
                "cpu_quality_verified": True,
                "training_backend": "warp",
                "training_backend_quality_verified": False,
                "training_backend_observed_teacher_success_rate": float(
                    verified["teacher_success_rate"]
                ),
                "training_backend_observed_return_quality_rate": float(
                    verified["return_quality_rate"]
                ),
                "training_backend_observed_no_fall_rate": float(
                    verified["no_fall_rate"]
                ),
            }
        )
    if training_backend_evidence is not None:
        training_metrics = dict(
            training_backend_evidence["independent_mjx_replay_metrics"]
        )
        binding.update(
            {
                "training_backend_quality_verified": True,
                "training_backend": str(
                    training_backend_evidence.get("training_backend", "unknown")
                ),
                "training_backend_outgoing_z_m_s": float(
                    training_metrics["outgoing_z_m_s"]
                ),
                "training_backend_outgoing_forward_m_s": float(
                    training_metrics["outgoing_forward_m_s"]
                ),
                "training_backend_teacher_success_rate": float(
                    training_metrics["teacher_success_rate"]
                ),
                "training_backend_verification_context_semantics": str(
                    training_backend_evidence["verification_context_semantics"]
                ),
                "training_backend_verification_group_indices": list(
                    training_backend_evidence["verification_group_indices"]
                ),
            }
        )
    if timing_evidence:
        binding["base_timing_transfer"] = {
            **timing_evidence,
            "teacher_cem_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    binding["binding_sha256"] = _stable_json_hash(binding)
    return QualityTeacherDataset(
        observations,
        corrections,
        weights,
        time_to_intercept_s,
        binding,
    )


def pretrain_selected_correction_bc(
    agent: Any,
    dataset: QualityTeacherDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit only the independent correction network to a quality teacher."""

    if "policy_correction" not in agent:
        raise ValueError("teacher BC requires policy_correction parameters")
    if int(steps) < 0 or int(batch_size) <= 0:
        raise ValueError("teacher BC steps/batch size are invalid")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("teacher BC learning rate must be finite and positive")
    obs = jnp.asarray(dataset.observation_normalized)
    target = jnp.asarray(dataset.correction_raw)
    weight = jnp.asarray(dataset.sample_weight)

    def weighted_loss(params, x, y, w):
        prediction = _mlp(params, x)
        per_sample = jnp.mean(jnp.square(prediction - y), axis=-1)
        return jnp.sum(per_sample * w) / jnp.maximum(jnp.sum(w), 1.0e-8)

    evaluate = jax.jit(lambda params: weighted_loss(params, obs, target, weight))
    params = agent["policy_correction"]
    initial_loss = float(evaluate(params))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(learning_rate)))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, indices):
        loss, grads = jax.value_and_grad(weighted_loss)(
            params,
            obs[indices],
            target[indices],
            weight[indices],
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(int(seed) ^ 0xBC31)
    count = int(obs.shape[0])
    last_loss = initial_loss
    for _step in range(int(steps)):
        effective_batch_size = min(int(batch_size), count)
        if effective_batch_size == count:
            # The sealed single-feed teacher currently has only tens of
            # weighted contact-window samples.  Sampling ``count`` elements
            # *with replacement* omitted roughly 37% of those samples on each
            # update and left a noisy ~1e-6 error floor exactly where contact
            # is most sensitive.  A requested full batch must really be the
            # complete deterministic dataset.
            indices = np.arange(count, dtype=np.int32)
        else:
            indices = rng.choice(
                count,
                size=effective_batch_size,
                replace=False,
            ).astype(np.int32, copy=False)
        params, opt_state, loss = update(params, opt_state, jnp.asarray(indices))
        last_loss = float(loss)
    final_loss = float(evaluate(params))
    updated_agent = {**agent, "policy_correction": params}
    report = {
        "schema_version": "stage3_selected_correction_bc_pretrain_v1",
        "teacher_binding": dataset.binding,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "initial_weighted_mse": initial_loss,
        "last_minibatch_mse": last_loss,
        "final_weighted_mse": final_loss,
        "improvement_fraction": (
            0.0 if initial_loss <= 0.0 else float((initial_loss - final_loss) / initial_loss)
        ),
        "passed": bool(np.isfinite(final_loss) and (int(steps) == 0 or final_loss < initial_loss)),
    }
    report["report_sha256"] = _stable_json_hash(report)
    if not report["passed"]:
        raise ValueError("teacher BC pretraining did not reduce the weighted correction loss")
    return updated_agent, report


def build_ppo_optimizer(
    agent: Any,
    *,
    max_grad_norm: float,
    learning_rate: float,
    actor_learning_rate: float | None = None,
    critic_learning_rate: float | None = None,
) -> optax.GradientTransformation:
    """Build the checkpoint-stable PPO optimizer, optionally split by role."""

    actor_lr = float(learning_rate) if actor_learning_rate is None else float(actor_learning_rate)
    critic_lr = float(learning_rate) if critic_learning_rate is None else float(critic_learning_rate)
    if not all(math.isfinite(value) and value > 0.0 for value in (actor_lr, critic_lr)):
        raise ValueError("actor and critic learning rates must be finite and positive")
    if actor_learning_rate is None and critic_learning_rate is None:
        parameter_update = optax.adam(float(learning_rate))
    else:
        labels = {
            name: jax.tree_util.tree_map(
                lambda _value, label=("critic" if name == "value" else "actor"): label,
                subtree,
            )
            for name, subtree in agent.items()
        }
        parameter_update = optax.multi_transform(
            {
                "actor": optax.adam(actor_lr),
                "critic": optax.adam(critic_lr),
            },
            labels,
        )
    return optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        parameter_update,
    )


def compute_rollout_gae(
    records: dict[str, jax.Array],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE from transition-local, pre-reset next-state values.

    ``next_value`` must be evaluated from the transition's terminal observation
    before an auto-reset swaps in a new episode.  True terminations suppress
    value bootstrapping; time-limit truncations bootstrap the terminal state but
    still stop the recursive advantage chain at the episode boundary.
    """

    def body(carry, step):
        advantage = carry
        delta = step["reward"] + float(gamma) * step["next_value"] * (1.0 - step["terminated"]) - step["value"]
        advantage = delta + float(gamma) * float(gae_lambda) * (1.0 - step["done"]) * advantage
        return advantage, advantage

    steps = {
        "reward": records["reward"],
        "value": records["value"],
        "next_value": records["next_value"],
        "done": records["done"].astype(jnp.float32),
        "terminated": records["terminated"].astype(jnp.float32),
    }
    _, advantages = jax.lax.scan(
        body,
        jnp.zeros_like(records["value"][-1]),
        jax.tree_util.tree_map(lambda value: value[::-1], steps),
    )
    advantages = advantages[::-1]
    return advantages, advantages + records["value"]


def backfill_pre_hit_success_mask(
    hit_event: jax.Array,
    done: jax.Array,
) -> jax.Array:
    """Label only transitions at or before a future hit in the same episode.

    The reverse scan is deliberately bounded by the current rollout.  It
    therefore captures the final arm/wrist decisions that led to a genuine
    contact without treating post-impact recovery actions or the next
    auto-reset episode as successful demonstrations.
    """

    hit_event = jnp.asarray(hit_event, dtype=jnp.bool_)
    done = jnp.asarray(done, dtype=jnp.bool_)
    if hit_event.shape != done.shape:
        raise ValueError("hit_event and done must have identical shapes")
    if hit_event.ndim < 1:
        raise ValueError("hit_event and done must include a rollout-time axis")

    def body(future_success, values):
        current_hit, current_done = values
        current_success = current_hit | ((~current_done) & future_success)
        return current_success, current_success

    initial = jnp.zeros_like(hit_event[-1], dtype=jnp.bool_)
    _, reverse_mask = jax.lax.scan(
        body,
        initial,
        (hit_event[::-1], done[::-1]),
    )
    return reverse_mask[::-1]


def successful_action_imitation_loss(
    mean: jax.Array,
    sampled_raw_action: jax.Array,
    success_weight: jax.Array,
    action_mask: jax.Array,
) -> jax.Array:
    """Regress selected policy means toward actions from genuine hit traces."""

    mean = jnp.asarray(mean)
    sampled_raw_action = jnp.asarray(sampled_raw_action)
    success_weight = jnp.asarray(success_weight, dtype=mean.dtype)
    action_mask = jnp.asarray(action_mask, dtype=mean.dtype)
    if mean.shape != sampled_raw_action.shape or mean.ndim != 2:
        raise ValueError("mean and sampled_raw_action must be equal rank-two batches")
    if success_weight.shape != mean.shape[:1]:
        raise ValueError("success_weight must contain one value per action sample")
    if action_mask.shape != mean.shape[1:]:
        raise ValueError("action_mask must match the action dimension")

    selected_count = jnp.maximum(action_mask.sum(), 1.0)
    squared_error = jnp.square(mean - jax.lax.stop_gradient(sampled_raw_action))
    per_sample = (squared_error * action_mask[None, :]).sum(axis=-1) / selected_count
    successful_count = jnp.maximum(success_weight.sum(), 1.0)
    return (per_sample * success_weight).sum() / successful_count


def window_successful_action_imitation_weight(
    success_mask: jax.Array,
    correction_window: jax.Array,
) -> jax.Array:
    """Keep self-imitation on physically active pre-impact corrections only."""

    success_mask = jnp.asarray(success_mask, dtype=jnp.bool_)
    return window_action_imitation_weight(
        success_mask.astype(jnp.float32),
        correction_window,
    )


def window_action_imitation_weight(
    event_weight: jax.Array,
    correction_window: jax.Array,
) -> jax.Array:
    """Restrict arbitrary non-negative imitation weights to active control."""

    event_weight = jnp.asarray(event_weight, dtype=jnp.float32)
    correction_window = jnp.asarray(correction_window, dtype=jnp.float32)
    if event_weight.shape != correction_window.shape:
        raise ValueError("imitation weights and correction window must have identical shapes")
    if event_weight.ndim < 1:
        raise ValueError("success imitation weights require a rollout-time axis")
    bounded_window = jnp.clip(correction_window, 0.0, 1.0)
    return jnp.maximum(event_weight, 0.0) * jnp.square(bounded_window)


def backfill_pre_hit_event_weight(
    event_weight: jax.Array,
    done: jax.Array,
) -> jax.Array:
    """Carry a hit-quality weight backward, without crossing episode ends."""

    event_weight = jnp.asarray(event_weight, dtype=jnp.float32)
    done = jnp.asarray(done, dtype=jnp.bool_)
    if event_weight.shape != done.shape:
        raise ValueError("event weights and done must have identical shapes")
    if event_weight.ndim < 1:
        raise ValueError("event weights require a rollout-time axis")

    def body(future_weight, values):
        current_weight, current_done = values
        propagated = jnp.where(current_done, 0.0, future_weight)
        value = jnp.maximum(current_weight, propagated)
        return value, value

    initial = jnp.zeros_like(event_weight[-1], dtype=jnp.float32)
    _, reverse = jax.lax.scan(
        body,
        initial,
        (event_weight[::-1], done[::-1]),
    )
    return reverse[::-1]


def future_episode_outcome(
    done: jax.Array,
    body_fall: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Report whether the current episode completes/falls later in a rollout.

    A hit transition cannot itself prove the advertised no-fall property: the
    body may fall during follow-through many control steps later.  This reverse
    scan supplies exact within-rollout episode outcomes.  Episodes whose
    terminal transition lies beyond the rollout remain explicitly unknown and
    therefore cannot satisfy a strict no-fall imitation contract.
    """

    done = jnp.asarray(done, dtype=jnp.bool_)
    body_fall = jnp.asarray(body_fall, dtype=jnp.bool_)
    if done.shape != body_fall.shape:
        raise ValueError("done and body_fall must have identical shapes")
    if done.ndim < 1:
        raise ValueError("episode outcomes require a rollout-time axis")

    def body(carry, values):
        future_completed, future_fall = carry
        current_done, current_fall = values
        completed = current_done | ((~current_done) & future_completed)
        fell = current_fall | ((~current_done) & future_fall)
        return (completed, fell), (completed, fell)

    initial = (
        jnp.zeros_like(done[-1], dtype=jnp.bool_),
        jnp.zeros_like(done[-1], dtype=jnp.bool_),
    )
    _, reverse = jax.lax.scan(
        body,
        initial,
        (done[::-1], body_fall[::-1]),
    )
    return reverse[0][::-1], reverse[1][::-1]


def quality_success_event_mask(
    *,
    hit_event: jax.Array,
    rewarded_hit_was_event_rebound: jax.Array,
    outgoing_z_m_s: jax.Array,
    outgoing_forward_m_s: jax.Array,
    predicted_net_clearance_m: jax.Array,
    return_direction_signed_score: jax.Array,
    body_fall: jax.Array,
    episode_completed_in_rollout: jax.Array,
    episode_fell_in_rollout: jax.Array,
    min_outgoing_z_m_s: float,
    min_forward_m_s: float,
    min_predicted_net_clearance_m: float,
    min_return_direction_signed_score: float,
    require_episode_no_fall: bool,
    racket_face_forward_alignment: jax.Array | None = None,
    min_racket_face_forward_alignment: float = -1.0,
) -> jax.Array:
    """Select real rebounds that also have usable direction and ballistics."""

    if racket_face_forward_alignment is None:
        # Historical callers predate the optional racket-face contract.  New
        # Stage-3 specs opt in explicitly and pass the measured event normal.
        racket_face_forward_alignment = jnp.ones_like(
            jnp.asarray(outgoing_z_m_s),
            dtype=jnp.float32,
        )

    arrays = tuple(
        jnp.asarray(value)
        for value in (
            hit_event,
            rewarded_hit_was_event_rebound,
            outgoing_z_m_s,
            outgoing_forward_m_s,
            predicted_net_clearance_m,
            return_direction_signed_score,
            racket_face_forward_alignment,
            body_fall,
            episode_completed_in_rollout,
            episode_fell_in_rollout,
        )
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("quality-success event arrays must have identical shapes")
    success = (
        arrays[0].astype(jnp.bool_)
        & arrays[1].astype(jnp.bool_)
        & (arrays[2] >= float(min_outgoing_z_m_s))
        & (arrays[3] >= float(min_forward_m_s))
        & (arrays[4] >= float(min_predicted_net_clearance_m))
        & (arrays[5] >= float(min_return_direction_signed_score))
        & (arrays[6] >= float(min_racket_face_forward_alignment))
    )
    if bool(require_episode_no_fall):
        return success & arrays[8].astype(jnp.bool_) & (~arrays[9].astype(jnp.bool_))
    return success & (~arrays[7].astype(jnp.bool_))


def progressive_quality_imitation_event_weight(
    *,
    hit_event: jax.Array,
    rewarded_hit_was_event_rebound: jax.Array,
    outgoing_z_m_s: jax.Array,
    outgoing_forward_m_s: jax.Array,
    predicted_net_clearance_m: jax.Array,
    return_direction_signed_score: jax.Array,
    body_fall: jax.Array,
    episode_completed_in_rollout: jax.Array,
    episode_fell_in_rollout: jax.Array,
    target_outgoing_z_m_s: float,
    target_forward_m_s: float,
    target_predicted_net_clearance_m: float,
    target_return_direction_signed_score: float,
    forward_softness_m_s: float,
    vertical_softness_m_s: float,
    clearance_softness_m: float,
    direction_softness: float,
    min_weight: float,
    require_episode_no_fall: bool,
) -> jax.Array:
    """Weight rare near-feasible rebounds without redefining strict success.

    The multiplicative score makes a typical lateral, deeply sub-net contact
    effectively zero, while a sample close to all four physical targets can
    dominate self-imitation before the first absolute success is observed.
    Strict curriculum/promotion metrics continue to use
    :func:`quality_success_event_mask` and are not relaxed by this bootstrap.
    """

    for label, value in (
        ("forward_softness_m_s", forward_softness_m_s),
        ("vertical_softness_m_s", vertical_softness_m_s),
        ("clearance_softness_m", clearance_softness_m),
        ("direction_softness", direction_softness),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if not math.isfinite(float(min_weight)) or not 0.0 <= float(min_weight) <= 1.0:
        raise ValueError("progressive imitation min_weight must lie in [0, 1]")

    hit_event = jnp.asarray(hit_event, dtype=jnp.bool_)
    rebound = jnp.asarray(rewarded_hit_was_event_rebound, dtype=jnp.bool_)
    outgoing_z_m_s = jnp.asarray(outgoing_z_m_s, dtype=jnp.float32)
    outgoing_forward_m_s = jnp.asarray(outgoing_forward_m_s, dtype=jnp.float32)
    predicted_net_clearance_m = jnp.asarray(
        predicted_net_clearance_m, dtype=jnp.float32
    )
    return_direction_signed_score = jnp.asarray(
        return_direction_signed_score, dtype=jnp.float32
    )
    body_fall = jnp.asarray(body_fall, dtype=jnp.bool_)
    completed = jnp.asarray(episode_completed_in_rollout, dtype=jnp.bool_)
    fell = jnp.asarray(episode_fell_in_rollout, dtype=jnp.bool_)
    arrays = (
        rebound,
        outgoing_z_m_s,
        outgoing_forward_m_s,
        predicted_net_clearance_m,
        return_direction_signed_score,
        body_fall,
        completed,
        fell,
    )
    if any(value.shape != hit_event.shape for value in arrays):
        raise ValueError("progressive imitation event arrays must have identical shapes")

    quality = (
        jax.nn.sigmoid(
            (outgoing_forward_m_s - float(target_forward_m_s))
            / float(forward_softness_m_s)
        )
        * jax.nn.sigmoid(
            (outgoing_z_m_s - float(target_outgoing_z_m_s))
            / float(vertical_softness_m_s)
        )
        * jax.nn.sigmoid(
            (
                predicted_net_clearance_m
                - float(target_predicted_net_clearance_m)
            )
            / float(clearance_softness_m)
        )
        * jax.nn.sigmoid(
            (
                return_direction_signed_score
                - float(target_return_direction_signed_score)
            )
            / float(direction_softness)
        )
    )
    # Always reject a fall that is observable later in this rollout.  The
    # relaxed bootstrap merely permits an episode whose terminal transition
    # lies beyond the current rollout; it never treats a known future fall as
    # useful imitation evidence.
    safe = completed & (~fell) if bool(require_episode_no_fall) else ~fell
    weight = jnp.where(hit_event & rebound & safe, quality, 0.0)
    return jnp.where(weight >= float(min_weight), weight, 0.0)


def mask_policy_update_gradients(grads: Any, action_mask: jax.Array) -> Any:
    """Freeze the policy trunk and all unmasked output/log-std parameters."""

    action_mask = jnp.asarray(action_mask)
    if action_mask.ndim != 1:
        raise ValueError("policy action mask must be one-dimensional")
    policy_grads = grads["policy"]
    output_layer = policy_grads[-1]
    if output_layer["b"].shape != action_mask.shape:
        raise ValueError("policy action mask shape does not match the policy output")
    masked_policy = [jax.tree_util.tree_map(jnp.zeros_like, layer) for layer in policy_grads[:-1]]
    masked_policy.append(
        {
            "w": output_layer["w"] * action_mask[None, :],
            "b": output_layer["b"] * action_mask,
        }
    )
    return {
        **grads,
        "policy": masked_policy,
        "log_std": grads["log_std"] * action_mask,
    }


def mask_selected_delta_adapter_gradients(grads: Any, action_mask: jax.Array) -> Any:
    """Freeze the inherited actor and constrain a nonlinear delta to selected outputs.

    The adapter trunk remains trainable, while its final layer and Gaussian
    exploration can affect only the explicitly selected actuator dimensions.
    Because every unselected final weight/bias starts at exact zero and always
    receives an exact zero gradient, all unselected actor means remain exactly
    equal to the inherited policy throughout training.
    """

    action_mask = jnp.asarray(action_mask)
    if action_mask.ndim != 1:
        raise ValueError("policy action mask must be one-dimensional")
    if "policy_delta" not in grads:
        raise ValueError("selected_delta_adapter requires policy_delta parameters")
    delta_grads = grads["policy_delta"]
    output_layer = delta_grads[-1]
    if output_layer["b"].shape != action_mask.shape:
        raise ValueError("policy action mask shape does not match the adapter output")
    masked_delta = list(delta_grads[:-1])
    masked_delta.append(
        {
            "w": output_layer["w"] * action_mask[None, :],
            "b": output_layer["b"] * action_mask,
        }
    )
    return {
        **grads,
        "policy": jax.tree_util.tree_map(jnp.zeros_like, grads["policy"]),
        "policy_delta": masked_delta,
        "log_std": grads["log_std"] * action_mask,
    }


def mask_selected_refinement_delta_adapter_gradients(grads: Any, action_mask: jax.Array) -> Any:
    """Freeze the inherited actor/Phase-A adapter and train a selected refinement.

    Unlike reusing ``selected_delta_adapter`` with a smaller output mask, this
    keeps the already learned adapter trunk bit-for-bit fixed.  A separate
    zero-output nonlinear branch may affect only the selected actuator rows, so
    wrist fine-tuning cannot indirectly alter learned shoulder/elbow outputs.
    """

    action_mask = jnp.asarray(action_mask)
    if action_mask.ndim != 1:
        raise ValueError("policy action mask must be one-dimensional")
    if "policy_delta" not in grads:
        raise ValueError("selected_refinement_delta_adapter requires a learned policy_delta")
    if "policy_refinement_delta" not in grads:
        raise ValueError("selected_refinement_delta_adapter requires policy_refinement_delta parameters")
    refinement_grads = grads["policy_refinement_delta"]
    output_layer = refinement_grads[-1]
    if output_layer["b"].shape != action_mask.shape:
        raise ValueError("policy action mask shape does not match the refinement adapter output")
    masked_refinement = list(refinement_grads[:-1])
    masked_refinement.append(
        {
            "w": output_layer["w"] * action_mask[None, :],
            "b": output_layer["b"] * action_mask,
        }
    )
    return {
        **grads,
        "policy": jax.tree_util.tree_map(jnp.zeros_like, grads["policy"]),
        "policy_delta": jax.tree_util.tree_map(jnp.zeros_like, grads["policy_delta"]),
        "policy_refinement_delta": masked_refinement,
        "log_std": grads["log_std"] * action_mask,
    }


def mask_selected_physical_correction_gradients(grads: Any) -> Any:
    """Freeze the inherited actor and train only the bounded correction/value."""

    if "policy_correction" not in grads or "correction_log_std" not in grads:
        raise ValueError("selected_physical_correction parameters are missing")
    result = {
        **grads,
        "policy": jax.tree_util.tree_map(jnp.zeros_like, grads["policy"]),
        "log_std": jnp.zeros_like(grads["log_std"]),
    }
    if "policy_delta" in grads:
        result["policy_delta"] = jax.tree_util.tree_map(
            jnp.zeros_like,
            grads["policy_delta"],
        )
    if "policy_refinement_delta" in grads:
        result["policy_refinement_delta"] = jax.tree_util.tree_map(
            jnp.zeros_like,
            grads["policy_refinement_delta"],
        )
    return result


def freeze_action_std_gradients(grads: Any) -> Any:
    """Keep every Gaussian standard deviation bitwise fixed during PPO."""

    return {**grads, "log_std": jnp.zeros_like(grads["log_std"])}


def apply_policy_exploration_contract(
    agent: Any,
    *,
    action_size: int,
    trainable_action_indices: tuple[int, ...],
    frozen_action_std: float | None,
) -> Any:
    """Suppress exploration on frozen outputs without changing actor means."""

    if frozen_action_std is None:
        return agent
    frozen_std = float(frozen_action_std)
    if not math.isfinite(frozen_std) or not 0.0 < frozen_std <= 1.0:
        raise ValueError("frozen_action_std must be finite and lie in (0, 1]")
    indices = tuple(int(index) for index in trainable_action_indices)
    if not indices:
        raise ValueError("frozen_action_std requires trainable action indices")
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= int(action_size):
        raise ValueError("invalid trainable action indices for frozen exploration")
    action_mask = (
        jnp.zeros((int(action_size),), dtype=jnp.bool_)
        .at[jnp.asarray(indices, dtype=jnp.int32)]
        .set(True)
    )
    log_std = jnp.asarray(agent["log_std"])
    if log_std.shape != (int(action_size),):
        raise ValueError("agent log_std shape does not match the action space")
    return {
        **agent,
        "log_std": jnp.where(action_mask, log_std, jnp.log(jnp.asarray(frozen_std, log_std.dtype))),
    }


def make_train_iteration(
    env: IncomingHitMjxEnv,
    mx,
    cfg: TrainConfig,
    optimizer,
    *,
    teacher_dataset: QualityTeacherDataset | None = None,
):
    step_env = env.make_step_fn(mx, cfg.num_envs)
    squash_action = not bool(getattr(env, "expects_raw_latent", False))
    num_samples = cfg.rollout_steps * cfg.num_envs
    audit_sample_count = min(int(num_samples), 2048)
    if not (
        math.isfinite(float(cfg.max_post_update_ratio_guard_fraction))
        and 0.0 <= float(cfg.max_post_update_ratio_guard_fraction) <= 1.0
    ):
        raise ValueError(
            "max_post_update_ratio_guard_fraction must be finite and lie in [0, 1]"
        )
    if not (
        math.isfinite(float(cfg.max_post_update_kl_estimate))
        and float(cfg.max_post_update_kl_estimate) > 0.0
    ):
        raise ValueError(
            "max_post_update_kl_estimate must be finite and positive"
        )
    if int(cfg.minibatch_size) > 0:
        mb_size = int(cfg.minibatch_size)
        if num_samples % mb_size:
            raise ValueError(f"rollout samples {num_samples} must be divisible by minibatch_size {mb_size}")
        num_minibatches = num_samples // mb_size
    else:
        num_minibatches = int(cfg.num_minibatches)
        if num_minibatches <= 0 or num_samples % num_minibatches:
            raise ValueError("num_minibatches must be positive and divide rollout_steps * num_envs")
        mb_size = num_samples // num_minibatches
    policy_update_mode = str(cfg.policy_update_mode)
    configured_indices = tuple(int(index) for index in cfg.policy_trainable_action_indices)
    delta_hidden = tuple(int(size) for size in cfg.policy_delta_hidden)
    refinement_hidden = tuple(int(size) for size in cfg.policy_refinement_delta_hidden)
    correction_hidden = tuple(int(size) for size in cfg.policy_correction_hidden)
    teacher_action_prior_mode = str(cfg.teacher_action_prior_mode)
    teacher_prior_enabled = (
        teacher_action_prior_mode == "time_interpolated_frozen_plus_delta"
    )
    if teacher_action_prior_mode not in {
        "none",
        "time_interpolated_frozen_plus_delta",
    }:
        raise ValueError("unsupported teacher action prior mode")
    teacher_prior_time_np = np.asarray(
        cfg.teacher_prior_time_to_intercept_s, dtype=np.float32
    )
    teacher_prior_raw_np = np.asarray(
        cfg.teacher_prior_correction_raw, dtype=np.float32
    )
    if teacher_prior_enabled:
        if (
            teacher_prior_time_np.ndim != 1
            or teacher_prior_time_np.size < 2
            or teacher_prior_raw_np.shape
            != (teacher_prior_time_np.size, len(configured_indices))
            or not np.isfinite(teacher_prior_time_np).all()
            or not np.isfinite(teacher_prior_raw_np).all()
            or np.any(np.diff(teacher_prior_time_np) <= 0.0)
        ):
            raise ValueError("frozen teacher action prior arrays are invalid")
        if teacher_dataset is None:
            raise ValueError("frozen teacher action prior requires teacher BC binding")
    elif teacher_prior_time_np.size or teacher_prior_raw_np.size:
        raise ValueError("teacher prior arrays require an enabled prior mode")
    teacher_prior_time = jnp.asarray(teacher_prior_time_np)
    teacher_prior_raw = jnp.asarray(teacher_prior_raw_np)

    def selected_correction_dist(params, obs, time_to_intercept):
        correction_delta, correction_std = _selected_correction_dist(
            params,
            obs,
            std_min=correction_std_min,
            std_max=correction_std_max,
        )
        if teacher_prior_enabled:
            correction_delta = correction_delta + interpolate_correction_prior(
                time_to_intercept,
                knot_time_to_intercept_s=teacher_prior_time,
                knot_correction_raw=teacher_prior_raw,
                array_module=jnp,
            )
        return correction_delta, correction_std
    successful_action_imitation_coef = float(cfg.successful_action_imitation_coef)
    if not math.isfinite(successful_action_imitation_coef) or successful_action_imitation_coef < 0.0:
        raise ValueError("successful_action_imitation_coef must be finite and non-negative")
    quality_thresholds = (
        float(cfg.quality_success_min_outgoing_z_m_s),
        float(cfg.quality_success_min_forward_m_s),
        float(cfg.quality_success_min_predicted_net_clearance_m),
        float(cfg.quality_success_min_return_direction_signed_score),
    )
    if not all(math.isfinite(value) for value in quality_thresholds):
        raise ValueError("quality-success thresholds must be finite")
    if quality_thresholds[0] < 0.0 or quality_thresholds[1] < 0.0:
        raise ValueError("quality-success velocity thresholds must be non-negative")
    if not -1.0 <= quality_thresholds[3] <= 1.0:
        raise ValueError("quality-success direction threshold must lie in [-1, 1]")
    quality_imitation_mode = str(cfg.quality_imitation_mode)
    if quality_imitation_mode not in {"strict_success", "progressive_ballistic"}:
        raise ValueError(
            "quality_imitation_mode must be strict_success or progressive_ballistic"
        )
    if (
        quality_imitation_mode == "progressive_ballistic"
        and policy_update_mode not in PHYSICAL_CORRECTION_POLICY_MODES
    ):
        raise ValueError(
            "progressive ballistic imitation requires selected_physical_correction"
        )
    progressive_values = (
        float(cfg.quality_imitation_forward_softness_m_s),
        float(cfg.quality_imitation_vertical_softness_m_s),
        float(cfg.quality_imitation_clearance_softness_m),
        float(cfg.quality_imitation_direction_softness),
    )
    if not all(math.isfinite(value) and value > 0.0 for value in progressive_values):
        raise ValueError("quality-imitation softness values must be finite and positive")
    if not (
        math.isfinite(float(cfg.quality_imitation_min_weight))
        and 0.0 <= float(cfg.quality_imitation_min_weight) <= 1.0
    ):
        raise ValueError("quality-imitation min_weight must lie in [0, 1]")
    if successful_action_imitation_coef > 0.0 and policy_update_mode not in {
        "selected_delta_adapter",
        "selected_refinement_delta_adapter",
        "selected_physical_correction",
        "graded_full_body_correction",
    }:
        raise ValueError(
            "successful action imitation is restricted to selected adapter update modes"
        )
    bc_values = (
        float(cfg.teacher_bc_initial_coef),
        float(cfg.teacher_bc_final_coef),
        float(cfg.teacher_bc_learning_rate),
    )
    if not all(math.isfinite(value) for value in bc_values):
        raise ValueError("teacher BC coefficients/learning rate must be finite")
    if cfg.teacher_bc_initial_coef < 0.0 or cfg.teacher_bc_final_coef < 0.0:
        raise ValueError("teacher BC coefficients must be non-negative")
    if int(cfg.teacher_bc_pretrain_steps) < 0 or int(cfg.teacher_bc_batch_size) <= 0:
        raise ValueError("teacher BC pretrain steps/batch size are invalid")
    if int(cfg.teacher_bc_decay_steps) < 0:
        raise ValueError("teacher BC decay steps must be non-negative")
    teacher_bc_enabled = teacher_dataset is not None
    teacher_requested = bool(
        int(cfg.teacher_bc_pretrain_steps) > 0
        or float(cfg.teacher_bc_initial_coef) > 0.0
        or float(cfg.teacher_bc_final_coef) > 0.0
    )
    if teacher_requested and not teacher_bc_enabled:
        raise ValueError("teacher BC is configured but no quality teacher dataset was provided")
    if teacher_bc_enabled and policy_update_mode not in PHYSICAL_CORRECTION_POLICY_MODES:
        raise ValueError("quality teacher BC is restricted to physical-correction modes")
    if teacher_bc_enabled:
        teacher_obs = jnp.asarray(teacher_dataset.observation_normalized)
        teacher_target = jnp.asarray(teacher_dataset.correction_raw)
        teacher_weight = jnp.asarray(teacher_dataset.sample_weight)
        if teacher_obs.shape[1:] != (int(env.observation_size),):
            raise ValueError("teacher observation width differs from the environment")
        if teacher_target.shape[1:] != (len(configured_indices),):
            raise ValueError("teacher correction width differs from the policy contract")
    else:
        teacher_obs = jnp.zeros((1, int(env.observation_size)), dtype=jnp.float32)
        teacher_target = jnp.zeros((1, max(len(configured_indices), 1)), dtype=jnp.float32)
        teacher_weight = jnp.zeros((1,), dtype=jnp.float32)
    if policy_update_mode == "full_network":
        if configured_indices:
            raise ValueError("full_network policy updates must not specify trainable action indices")
        if delta_hidden:
            raise ValueError("full_network policy updates must not configure policy_delta_hidden")
        if refinement_hidden:
            raise ValueError("full_network policy updates must not configure policy_refinement_delta_hidden")
        if correction_hidden:
            raise ValueError("full_network policy updates must not configure policy_correction_hidden")
        if bool(cfg.freeze_observation_normalizer):
            raise ValueError("full_network policy updates must not freeze the observation normalizer")
        if cfg.frozen_action_std is not None:
            raise ValueError("full_network policy updates must not configure frozen_action_std")
        if bool(cfg.freeze_trainable_action_std):
            raise ValueError("full_network policy updates must not freeze trainable action std")
        policy_action_mask = None
        frozen_log_std = None
    elif policy_update_mode == "distal_output_head_only":
        if delta_hidden:
            raise ValueError("distal_output_head_only must not configure policy_delta_hidden")
        if refinement_hidden:
            raise ValueError("distal_output_head_only must not configure policy_refinement_delta_hidden")
        if not configured_indices:
            raise ValueError("distal_output_head_only requires trainable action indices")
        if len(set(configured_indices)) != len(configured_indices):
            raise ValueError("policy trainable action indices must be unique")
        if min(configured_indices) < 0 or max(configured_indices) >= int(env.action_size):
            raise ValueError("policy trainable action index is outside the action space")
        if not bool(cfg.freeze_observation_normalizer):
            raise ValueError("distal_output_head_only requires a frozen observation normalizer")
        policy_action_mask = (
            jnp.zeros((int(env.action_size),), dtype=jnp.float32)
            .at[jnp.asarray(configured_indices, dtype=jnp.int32)]
            .set(1.0)
        )
        if cfg.frozen_action_std is None:
            frozen_log_std = None
        else:
            frozen_std = float(cfg.frozen_action_std)
            if not math.isfinite(frozen_std) or not 0.0 < frozen_std <= 1.0:
                raise ValueError("frozen_action_std must be finite and lie in (0, 1]")
            frozen_log_std = jnp.log(jnp.asarray(frozen_std, dtype=jnp.float32))
    elif policy_update_mode == "selected_delta_adapter":
        if not configured_indices:
            raise ValueError("selected_delta_adapter requires trainable action indices")
        if len(set(configured_indices)) != len(configured_indices):
            raise ValueError("policy trainable action indices must be unique")
        if min(configured_indices) < 0 or max(configured_indices) >= int(env.action_size):
            raise ValueError("policy trainable action index is outside the action space")
        if not delta_hidden or any(size <= 0 for size in delta_hidden):
            raise ValueError("selected_delta_adapter requires positive policy_delta_hidden sizes")
        if refinement_hidden:
            raise ValueError("selected_delta_adapter must not configure policy_refinement_delta_hidden")
        if not bool(cfg.freeze_observation_normalizer):
            raise ValueError("selected_delta_adapter requires a frozen observation normalizer")
        if cfg.frozen_action_std is None:
            raise ValueError("selected_delta_adapter requires frozen_action_std")
        policy_action_mask = (
            jnp.zeros((int(env.action_size),), dtype=jnp.float32)
            .at[jnp.asarray(configured_indices, dtype=jnp.int32)]
            .set(1.0)
        )
        frozen_std = float(cfg.frozen_action_std)
        if not math.isfinite(frozen_std) or not 0.0 < frozen_std <= 1.0:
            raise ValueError("frozen_action_std must be finite and lie in (0, 1]")
        frozen_log_std = jnp.log(jnp.asarray(frozen_std, dtype=jnp.float32))
    elif policy_update_mode == "selected_refinement_delta_adapter":
        if not configured_indices:
            raise ValueError("selected_refinement_delta_adapter requires trainable action indices")
        if len(set(configured_indices)) != len(configured_indices):
            raise ValueError("policy trainable action indices must be unique")
        if min(configured_indices) < 0 or max(configured_indices) >= int(env.action_size):
            raise ValueError("policy trainable action index is outside the action space")
        if not delta_hidden or any(size <= 0 for size in delta_hidden):
            raise ValueError("selected_refinement_delta_adapter requires the Phase-A policy_delta architecture")
        if not refinement_hidden or any(size <= 0 for size in refinement_hidden):
            raise ValueError("selected_refinement_delta_adapter requires positive refinement hidden sizes")
        if not bool(cfg.freeze_observation_normalizer):
            raise ValueError("selected_refinement_delta_adapter requires a frozen observation normalizer")
        if cfg.frozen_action_std is None:
            raise ValueError("selected_refinement_delta_adapter requires frozen_action_std")
        policy_action_mask = (
            jnp.zeros((int(env.action_size),), dtype=jnp.float32)
            .at[jnp.asarray(configured_indices, dtype=jnp.int32)]
            .set(1.0)
        )
        frozen_std = float(cfg.frozen_action_std)
        if not math.isfinite(frozen_std) or not 0.0 < frozen_std <= 1.0:
            raise ValueError("frozen_action_std must be finite and lie in (0, 1]")
        frozen_log_std = jnp.log(jnp.asarray(frozen_std, dtype=jnp.float32))
    elif policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
        if not configured_indices or len(set(configured_indices)) != len(configured_indices):
            raise ValueError("selected_physical_correction requires unique selected action indices")
        if min(configured_indices) < 0 or max(configured_indices) >= int(env.action_size):
            raise ValueError("selected physical correction index is outside the action space")
        if policy_update_mode == "selected_physical_correction" and (
            not delta_hidden or any(size <= 0 for size in delta_hidden)
        ):
            raise ValueError("selected_physical_correction requires the inherited policy_delta architecture")
        if policy_update_mode == "graded_full_body_correction" and delta_hidden:
            raise ValueError("graded_full_body_correction requires an exact-zero inherited residual")
        if refinement_hidden:
            raise ValueError("selected_physical_correction cannot use a coupled refinement adapter")
        if not correction_hidden or any(size <= 0 for size in correction_hidden):
            raise ValueError("selected_physical_correction requires positive correction hidden sizes")
        if policy_update_mode != "graded_full_body_correction" and not bool(
            cfg.freeze_observation_normalizer
        ):
            raise ValueError("selected_physical_correction requires a frozen observation normalizer")
        if policy_update_mode == "graded_full_body_correction" and bool(
            cfg.freeze_observation_normalizer
        ):
            raise ValueError(
                "graded_full_body_correction must learn correction-head observation statistics"
            )
        if cfg.frozen_action_std is not None or bool(cfg.freeze_trainable_action_std):
            raise ValueError("selected_physical_correction owns its selected-only exploration distribution")
        correction_size = len(configured_indices)
        for label, values in (
            ("correction_physical_scales", cfg.correction_physical_scales),
            ("correction_std_init", cfg.correction_std_init),
            ("correction_std_min", cfg.correction_std_min),
            ("correction_std_max", cfg.correction_std_max),
        ):
            if len(tuple(values)) != correction_size:
                raise ValueError(f"{label} must contain one value per selected action")
        correction_physical_scales = jnp.asarray(cfg.correction_physical_scales, dtype=jnp.float32)
        correction_std_min = jnp.asarray(cfg.correction_std_min, dtype=jnp.float32)
        correction_std_max = jnp.asarray(cfg.correction_std_max, dtype=jnp.float32)
        if not np.all(np.isfinite(np.asarray(cfg.correction_physical_scales, dtype=float))) or np.any(
            np.asarray(cfg.correction_physical_scales, dtype=float) <= 0.0
        ):
            raise ValueError("correction physical scales must be finite and positive")
        std_min_np = np.asarray(cfg.correction_std_min, dtype=float)
        std_init_np = np.asarray(cfg.correction_std_init, dtype=float)
        std_max_np = np.asarray(cfg.correction_std_max, dtype=float)
        if not (
            np.all(np.isfinite(std_min_np))
            and np.all(0.0 < std_min_np)
            and np.all(std_min_np <= std_init_np)
            and np.all(std_init_np <= std_max_np)
            and np.all(std_max_np <= 1.0)
        ):
            raise ValueError("correction standard deviation bounds are invalid")
        if env._base is None:
            raise ValueError("selected_physical_correction requires a frozen base policy")
        inherited_scale_vector = env._base["residual_scale_vector"]
        if int(env._base["residual_scale_ramp_steps"]) != 0:
            raise ValueError("selected_physical_correction requires constant inherited residual authority")
        policy_action_mask = jnp.ones((correction_size,), dtype=jnp.float32)
        frozen_log_std = None
    else:
        raise ValueError(
            "policy_update_mode must be full_network, distal_output_head_only, "
            "selected_delta_adapter, selected_refinement_delta_adapter, "
            "selected_physical_correction, or graded_full_body_correction"
        )

    def rollout(agent, obs_rms, env_states, key):
        def body(carry, _):
            env_states, key = carry
            key, sub = jax.random.split(key)
            obs_norm = obs_rms.normalize(env_states.obs)
            if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                elapsed = (
                    env_states.step_index.astype(jnp.float32)
                    * env.control_substeps
                    * env.timestep
                )
                time_to_intercept = env.intercept_times[env_states.feed_idx] - elapsed
                correction_mean, correction_std = selected_correction_dist(
                    agent,
                    obs_norm,
                    time_to_intercept,
                )
                raw = correction_mean + correction_std * jax.random.normal(
                    sub,
                    correction_mean.shape,
                )
                squashed_correction = jnp.tanh(raw)
                logp = _tanh_normal_logprob(
                    correction_mean,
                    correction_std,
                    raw,
                    squashed_correction,
                )
                inherited_residual = (
                    jnp.zeros_like(_mlp(agent["policy"], obs_norm))
                    if policy_update_mode == "graded_full_body_correction"
                    else jnp.tanh(_inherited_policy_mean(agent, obs_norm))
                )
                correction_window = selected_correction_window(
                    time_to_intercept,
                    open_s=cfg.correction_window_open_s,
                    close_s=cfg.correction_window_close_s,
                    smoothing_s=cfg.correction_window_smoothing_s,
                    array_module=jnp,
                )
                action = compose_selected_physical_correction(
                    inherited_residual,
                    raw,
                    selected_indices=configured_indices,
                    physical_scales=correction_physical_scales,
                    inherited_residual_scale=inherited_scale_vector,
                    window=correction_window,
                    array_module=jnp,
                )
            else:
                action, raw, logp = sample_action(
                    agent,
                    obs_norm,
                    sub,
                    squash_action=squash_action,
                )
            value = _mlp(agent["value"], obs_norm)[..., 0]
            next_states, tr = step_env(env_states, action)
            next_obs_norm = obs_rms.normalize(tr["next_obs"])
            next_value = _mlp(agent["value"], next_obs_norm)[..., 0]
            record = {
                "obs_norm": obs_norm,
                "raw_action": raw,
                "logp": logp,
                "value": value,
                "next_value": next_value,
                "reward": tr["reward"],
                "done": tr["done"],
                "terminated": tr["terminated"],
                "hit": tr["hit"],
                "crossed_net": tr["crossed_net"],
                "invalid_net_crossed": tr["invalid_net_crossed"],
                "landing_score": tr["landing_score"],
                "miss": tr["miss"],
                "body_fall": tr["body_fall"],
                "hit_event": tr["hit_event"],
                "stringbed_contact_event": tr["stringbed_contact_event"],
                "event_rebound_event": tr["event_rebound_event"],
                "rewarded_hit_was_event_rebound": tr["rewarded_hit_was_event_rebound"],
                "valid_net_cross_event": tr["valid_net_cross_event"],
                "invalid_net_cross_event": tr["invalid_net_cross_event"],
                "landing_event": tr["landing_event"],
                "feed_index": tr["feed_index"],
                "positive_outgoing_z_event": tr["positive_outgoing_z_event"],
                "obs_raw": env_states.obs,
            }
            if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                record.update(
                    {
                        "correction_window": correction_window,
                        "selected_correction_rms": jnp.sqrt(
                            jnp.mean(jnp.square(squashed_correction), axis=-1)
                        ),
                        "teacher_prior_time_to_intercept_s": time_to_intercept,
                    }
                )
            for name in (
                "control_finite",
                "raw_latent_rms",
                "raw_latent_saturation",
                "latent_norm",
                "prior_sigma_mean",
                "lab_state_unclipped_z_rms",
                "lab_state_ood_fraction",
                "body_action_rms",
                "right_grip_action_rms",
                "lambda_lab",
                "active_feed_count",
                "racket_head_speed_m_s",
                "ball_racket_distance_m",
                "intercept_position_error_m",
                "time_to_intercept_s",
                "shuttle_proximity_reward",
                "timed_intercept_reward",
                "racket_direction_reward",
                "closest_approach_distance_m",
                "closest_approach_direction_score",
                "closest_approach_terminal_direction_score",
                "closest_approach_terminal_event",
                "racket_direction_score",
                "racket_direction_signed_score",
                "racket_velocity_direction_score",
                "racket_velocity_direction_signed_score",
                "counterfactual_rebound_score",
                "counterfactual_rebound_closing_speed_m_s",
                "counterfactual_rebound_closing_gate",
                "counterfactual_rebound_direction_signed_score",
                "counterfactual_rebound_clearance_score",
                "counterfactual_rebound_predicted_clearance_m",
                "counterfactual_rebound_velocity_x_m_s",
                "counterfactual_rebound_velocity_y_m_s",
                "counterfactual_rebound_velocity_z_m_s",
                "inverse_impact_score",
                "inverse_impact_decomposed_score",
                "inverse_impact_signed_normal_alignment",
                "inverse_impact_shifted_normal_score",
                "inverse_impact_normal_alignment",
                "inverse_impact_racket_velocity_score",
                "inverse_impact_racket_velocity_error_m_s",
                "inverse_impact_target_closing_speed_m_s",
                "inverse_impact_target_racket_velocity_x_m_s",
                "inverse_impact_target_racket_velocity_y_m_s",
                "inverse_impact_target_racket_velocity_z_m_s",
                "return_direction_reward",
                "outgoing_vertical_reward",
                "outgoing_forward_reward",
                "return_direction_score",
                "return_direction_signed_score",
                "return_clearance_reward",
                "miss_penalty_reward",
                "return_clearance_score",
                "predicted_net_clearance_m",
                "hit_contact_speed_m_s",
                "hit_racket_head_speed_m_s",
                "hit_racket_direction_signed_score",
                "hit_racket_velocity_direction_signed_score",
                "hit_event_direction_reward_score",
                "hit_inverse_impact_score",
                "hit_inverse_impact_decomposed_score",
                "hit_inverse_impact_signed_normal_alignment",
                "hit_inverse_impact_normal_alignment",
                "hit_inverse_impact_racket_velocity_error_m_s",
                "hit_outgoing_velocity_x_m_s",
                "hit_outgoing_velocity_y_m_s",
                "hit_outgoing_velocity_z_m_s",
                "hit_outgoing_forward_velocity_m_s",
                "hit_racket_face_forward_alignment",
                "muscle_power_abs_mean",
                "normalized_control_energy",
                "body_action_saturation_fraction",
                "full_action_saturation_fraction",
                "residual_override_action_rms",
                "residual_override_composed_saturation_fraction",
                "residual_authority_progress",
                "bounded_residual_rms",
                "net_clearance_m",
                "opponent_back_landing",
                "impact_position_error_m",
                "impact_rho2",
                "impact_timing_error_s",
                "stringbed_normal_error_rad",
                "racket_linear_velocity_error_m_s",
                "racket_angular_velocity_error_rad_s",
                "landing_error_m",
                "apex_error_m",
                "ready_pose_error",
                "recovery_progress",
                "recovery_complete",
                "recovery_metric_event",
                "flight_resolved",
                "task_curriculum_stage_index",
            ):
                if name in tr:
                    record[name] = tr[name]
            return (next_states, key), record

        (env_states, key), records = jax.lax.scan(body, (env_states, key), None, length=cfg.rollout_steps)
        return env_states, key, records

    def ppo_update(agent, opt_state, batch, key, teacher_bc_coef):
        def epoch(carry, _):
            agent, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, num_samples)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)

            def minibatch(carry, mb):
                agent, opt_state = carry

                def loss_fn(params):
                    if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                        mean, std = selected_correction_dist(
                            params,
                            mb["obs_norm"],
                            mb["teacher_prior_time_to_intercept_s"],
                        )
                        squashed = jnp.tanh(mb["raw_action"])
                        logp = _tanh_normal_logprob(mean, std, mb["raw_action"], squashed)
                        entropy = jnp.sum(
                            0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(std),
                            axis=-1,
                        )
                        value = _mlp(params["value"], mb["obs_norm"])[..., 0]
                    else:
                        logp, entropy, value = evaluate_actions(
                            params,
                            mb["obs_norm"],
                            mb["raw_action"],
                            squash_action=squash_action,
                            entropy_action_mask=policy_action_mask,
                        )
                    ratio, log_ratio, ratio_guard_applied = bounded_ppo_ratio(
                        logp,
                        mb["logp"],
                        max_abs_log_ratio=cfg.max_abs_log_ratio,
                    )
                    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                    pg1 = -adv * ratio
                    pg2 = -adv * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    policy_loss = jnp.maximum(pg1, pg2).mean()
                    value_loss = 0.5 * jnp.square(value - mb["returns"]).mean()
                    entropy_loss = -entropy.mean()
                    if successful_action_imitation_coef > 0.0:
                        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                            mean, _ = selected_correction_dist(
                                params,
                                mb["obs_norm"],
                                mb["teacher_prior_time_to_intercept_s"],
                            )
                        else:
                            mean, _ = _dist(params, mb["obs_norm"])
                        imitation_loss = successful_action_imitation_loss(
                            mean,
                            mb["raw_action"],
                            mb["success_action_weight"],
                            policy_action_mask,
                        )
                    else:
                        imitation_loss = jnp.asarray(0.0, dtype=policy_loss.dtype)
                    if teacher_bc_enabled:
                        teacher_prediction = _mlp(params["policy_correction"], teacher_obs)
                        teacher_per_sample = jnp.mean(
                            jnp.square(teacher_prediction - teacher_target),
                            axis=-1,
                        )
                        teacher_bc_loss = (
                            jnp.sum(teacher_per_sample * teacher_weight)
                            / jnp.maximum(jnp.sum(teacher_weight), 1.0e-8)
                        )
                    else:
                        teacher_bc_loss = jnp.asarray(0.0, dtype=policy_loss.dtype)
                    total = (
                        policy_loss
                        + cfg.value_coef * value_loss
                        + cfg.entropy_coef * entropy_loss
                        + successful_action_imitation_coef * imitation_loss
                        + teacher_bc_coef * teacher_bc_loss
                    )
                    return total, (
                        policy_loss,
                        value_loss,
                        entropy_loss,
                        imitation_loss,
                        teacher_bc_loss,
                        jnp.max(jnp.abs(log_ratio)),
                        ratio_guard_applied.mean(),
                    )

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent)
                if policy_update_mode == "distal_output_head_only":
                    grads = mask_policy_update_gradients(grads, policy_action_mask)
                elif policy_update_mode == "selected_delta_adapter":
                    grads = mask_selected_delta_adapter_gradients(grads, policy_action_mask)
                elif policy_update_mode == "selected_refinement_delta_adapter":
                    grads = mask_selected_refinement_delta_adapter_gradients(grads, policy_action_mask)
                elif policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                    grads = mask_selected_physical_correction_gradients(grads)
                if bool(cfg.freeze_trainable_action_std):
                    grads = freeze_action_std_gradients(grads)
                gradient_l2_norm = optax.global_norm(grads)
                gradients_all_finite = jnp.all(
                    jnp.stack(
                        [
                            jnp.all(jnp.isfinite(gradient))
                            for gradient in jax.tree_util.tree_leaves(grads)
                        ]
                    )
                )
                updates, opt_state = optimizer.update(grads, opt_state, agent)
                agent = optax.apply_updates(agent, updates)
                updated_log_std = jnp.clip(agent["log_std"], -12.0, 1.0)
                if frozen_log_std is not None:
                    updated_log_std = jnp.where(
                        policy_action_mask > 0.0,
                        updated_log_std,
                        frozen_log_std,
                    )
                agent = {**agent, "log_std": updated_log_std}
                if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
                    agent = {
                        **agent,
                        "correction_log_std": jnp.clip(
                            agent["correction_log_std"],
                            jnp.log(correction_std_min),
                            jnp.log(correction_std_max),
                        ),
                    }
                parameters_all_finite = jnp.all(
                    jnp.stack(
                        [
                            jnp.all(jnp.isfinite(parameter))
                            for parameter in jax.tree_util.tree_leaves(agent)
                        ]
                    )
                )
                return (agent, opt_state), (
                    loss,
                    *aux,
                    gradient_l2_norm,
                    gradients_all_finite.astype(jnp.float32),
                    parameters_all_finite.astype(jnp.float32),
                )

            mbs = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, mb_size, *x.shape[1:]),
                shuffled,
            )
            (agent, opt_state), losses = jax.lax.scan(minibatch, (agent, opt_state), mbs)
            return (agent, opt_state, key), losses

        (agent, opt_state, key), losses = jax.lax.scan(epoch, (agent, opt_state, key), None, length=cfg.update_epochs)
        return agent, opt_state, key, losses

    @jax.jit
    def compiled_train_iteration(agent, opt_state, obs_rms, env_states, key, teacher_bc_coef):
        env_states, key, records = rollout(agent, obs_rms, env_states, key)
        advantages, returns = compute_rollout_gae(
            records,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        # Preserve the rollout-time normalization for every value target.  The
        # updated statistics apply only to the following iteration.
        if not bool(cfg.freeze_observation_normalizer):
            obs_rms = obs_rms.update(records["obs_raw"])

        def flat(x):
            return x.reshape(-1, *x.shape[2:])

        episode_completed_in_rollout, episode_fell_in_rollout = future_episode_outcome(
            records["done"],
            records["body_fall"],
        )
        imitation_success_event = records["hit_event"]
        imitation_event_weight = imitation_success_event.astype(jnp.float32)
        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
            imitation_success_event = quality_success_event_mask(
                hit_event=records["hit_event"],
                rewarded_hit_was_event_rebound=records[
                    "rewarded_hit_was_event_rebound"
                ],
                outgoing_z_m_s=records["hit_outgoing_velocity_z_m_s"],
                outgoing_forward_m_s=records[
                    "hit_outgoing_forward_velocity_m_s"
                ],
                predicted_net_clearance_m=records["predicted_net_clearance_m"],
                return_direction_signed_score=records[
                    "return_direction_signed_score"
                ],
                racket_face_forward_alignment=records[
                    "hit_racket_face_forward_alignment"
                ],
                body_fall=records["body_fall"],
                episode_completed_in_rollout=episode_completed_in_rollout,
                episode_fell_in_rollout=episode_fell_in_rollout,
                min_outgoing_z_m_s=cfg.quality_success_min_outgoing_z_m_s,
                min_forward_m_s=cfg.quality_success_min_forward_m_s,
                min_predicted_net_clearance_m=(
                    cfg.quality_success_min_predicted_net_clearance_m
                ),
                min_return_direction_signed_score=(
                    cfg.quality_success_min_return_direction_signed_score
                ),
                min_racket_face_forward_alignment=(
                    cfg.quality_success_min_racket_face_forward_alignment
                ),
                require_episode_no_fall=cfg.quality_success_require_episode_no_fall,
            )
            if quality_imitation_mode == "progressive_ballistic":
                imitation_event_weight = progressive_quality_imitation_event_weight(
                    hit_event=records["hit_event"],
                    rewarded_hit_was_event_rebound=records[
                        "rewarded_hit_was_event_rebound"
                    ],
                    outgoing_z_m_s=records["hit_outgoing_velocity_z_m_s"],
                    outgoing_forward_m_s=records[
                        "hit_outgoing_forward_velocity_m_s"
                    ],
                    predicted_net_clearance_m=records[
                        "predicted_net_clearance_m"
                    ],
                    return_direction_signed_score=records[
                        "return_direction_signed_score"
                    ],
                    body_fall=records["body_fall"],
                    episode_completed_in_rollout=episode_completed_in_rollout,
                    episode_fell_in_rollout=episode_fell_in_rollout,
                    target_outgoing_z_m_s=cfg.quality_success_min_outgoing_z_m_s,
                    target_forward_m_s=cfg.quality_success_min_forward_m_s,
                    target_predicted_net_clearance_m=(
                        cfg.quality_success_min_predicted_net_clearance_m
                    ),
                    target_return_direction_signed_score=(
                        cfg.quality_success_min_return_direction_signed_score
                    ),
                    forward_softness_m_s=(
                        cfg.quality_imitation_forward_softness_m_s
                    ),
                    vertical_softness_m_s=(
                        cfg.quality_imitation_vertical_softness_m_s
                    ),
                    clearance_softness_m=(
                        cfg.quality_imitation_clearance_softness_m
                    ),
                    direction_softness=cfg.quality_imitation_direction_softness,
                    min_weight=cfg.quality_imitation_min_weight,
                    require_episode_no_fall=(
                        cfg.quality_imitation_require_episode_no_fall
                    ),
                )
            else:
                imitation_event_weight = imitation_success_event.astype(
                    jnp.float32
                )
        success_action_weight = backfill_pre_hit_event_weight(
            imitation_event_weight,
            records["done"],
        )
        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
            success_action_weight = window_action_imitation_weight(
                success_action_weight,
                records["correction_window"],
            )
        batch = {
            "obs_norm": flat(records["obs_norm"]),
            "raw_action": flat(records["raw_action"]),
            "logp": flat(records["logp"]),
            "adv": flat(advantages),
            "returns": flat(returns),
            "success_action_weight": flat(success_action_weight),
        }
        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
            batch["teacher_prior_time_to_intercept_s"] = flat(
                records["teacher_prior_time_to_intercept_s"]
            )
        agent, opt_state, key, losses = ppo_update(
            agent,
            opt_state,
            batch,
            key,
            teacher_bc_coef,
        )
        audit_obs = batch["obs_norm"][:audit_sample_count]
        audit_raw_action = batch["raw_action"][:audit_sample_count]
        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
            audit_mean, audit_std = selected_correction_dist(
                agent,
                audit_obs,
                batch["teacher_prior_time_to_intercept_s"][:audit_sample_count],
            )
            audit_squashed = jnp.tanh(audit_raw_action)
            audit_new_logp = _tanh_normal_logprob(
                audit_mean,
                audit_std,
                audit_raw_action,
                audit_squashed,
            )
        else:
            audit_new_logp, _audit_entropy, _audit_value = evaluate_actions(
                agent,
                audit_obs,
                audit_raw_action,
                squash_action=squash_action,
                entropy_action_mask=policy_action_mask,
            )
        post_update_audit = post_update_logprob_audit(
            audit_new_logp,
            batch["logp"][:audit_sample_count],
            max_abs_log_ratio=cfg.max_abs_log_ratio,
        )

        done = records["done"]
        n_done = jnp.maximum(done.sum(), 1)
        metrics = {
            "mean_reward": records["reward"].mean(),
            "episodes_finished": done.sum(),
            "hit_rate": jnp.where(done, records["hit"], 0).sum() / n_done,
            "crossed_net_rate": jnp.where(done, records["crossed_net"], 0).sum() / n_done,
            "invalid_net_cross_rate": (jnp.where(done, records["invalid_net_crossed"], 0).sum() / n_done),
            "mean_landing_score": records["landing_score"].sum() / n_done,
            "miss_rate": jnp.where(done, records["miss"], 0).sum() / n_done,
            "fall_rate": jnp.where(done, records["body_fall"], 0).sum() / n_done,
            "loss": losses[0].mean(),
            "policy_loss": losses[1].mean(),
            "value_loss": losses[2].mean(),
            "entropy_loss": losses[3].mean(),
            "successful_action_imitation_loss": losses[4].mean(),
            "teacher_bc_loss": losses[5].mean(),
            "ppo_log_ratio_abs_max": losses[6].max(),
            "ppo_ratio_guard_fraction": losses[7].mean(),
            "ppo_gradient_l2_norm_max": losses[8].max(),
            "ppo_gradients_all_finite": losses[9].min(),
            "ppo_parameters_all_finite": losses[10].min(),
            **post_update_audit,
            "teacher_bc_coef": teacher_bc_coef,
            "successful_action_imitation_fraction": (
                success_action_weight > 0.0
            ).astype(jnp.float32).mean(),
            "successful_action_imitation_weight_mean": success_action_weight.mean(),
            "imitation_candidate_events": (
                imitation_event_weight > 0.0
            ).astype(jnp.float32).sum(),
            "imitation_event_weight_sum": imitation_event_weight.sum(),
            "strict_quality_success_events": imitation_success_event.astype(
                jnp.float32
            ).sum(),
        }
        if policy_update_mode in PHYSICAL_CORRECTION_POLICY_MODES:
            # Contact is millisecond-sensitive, so a run can look healthy in
            # reward space while its learned exploration scale quietly drifts
            # far enough to destroy the teacher contact.  Persist the actual
            # clipped distribution scale in metrics/W&B instead of requiring
            # post-hoc checkpoint tree inspection.
            correction_std = jnp.clip(
                jnp.exp(agent["correction_log_std"]),
                correction_std_min,
                correction_std_max,
            )
            metrics.update(
                {
                    "correction_std_min": jnp.min(correction_std),
                    "correction_std_mean": jnp.mean(correction_std),
                    "correction_std_max": jnp.max(correction_std),
                }
            )
        for name in (
            "control_finite",
            "raw_latent_rms",
            "raw_latent_saturation",
            "latent_norm",
            "prior_sigma_mean",
            "lab_state_unclipped_z_rms",
            "lab_state_ood_fraction",
            "body_action_rms",
            "right_grip_action_rms",
            "lambda_lab",
            "active_feed_count",
            "racket_head_speed_m_s",
            "ball_racket_distance_m",
            "intercept_position_error_m",
            "time_to_intercept_s",
            "shuttle_proximity_reward",
            "timed_intercept_reward",
            "racket_direction_reward",
            "racket_direction_score",
            "racket_direction_signed_score",
            "racket_velocity_direction_score",
            "racket_velocity_direction_signed_score",
            "counterfactual_rebound_score",
            "counterfactual_rebound_closing_speed_m_s",
            "counterfactual_rebound_closing_gate",
            "counterfactual_rebound_direction_signed_score",
            "counterfactual_rebound_clearance_score",
            "counterfactual_rebound_predicted_clearance_m",
            "counterfactual_rebound_velocity_x_m_s",
            "counterfactual_rebound_velocity_y_m_s",
            "counterfactual_rebound_velocity_z_m_s",
            "inverse_impact_score",
            "inverse_impact_decomposed_score",
            "inverse_impact_signed_normal_alignment",
            "inverse_impact_shifted_normal_score",
            "inverse_impact_normal_alignment",
            "inverse_impact_racket_velocity_score",
            "inverse_impact_racket_velocity_error_m_s",
            "inverse_impact_target_closing_speed_m_s",
            "inverse_impact_target_racket_velocity_x_m_s",
            "inverse_impact_target_racket_velocity_y_m_s",
            "inverse_impact_target_racket_velocity_z_m_s",
            "return_direction_reward",
            "outgoing_vertical_reward",
            "outgoing_forward_reward",
            "return_direction_score",
            "return_direction_signed_score",
            "return_clearance_reward",
            "miss_penalty_reward",
            "muscle_power_abs_mean",
            "normalized_control_energy",
            "body_action_saturation_fraction",
            "full_action_saturation_fraction",
            "residual_override_action_rms",
            "residual_override_composed_saturation_fraction",
            "residual_authority_progress",
            "bounded_residual_rms",
            "correction_window",
            "selected_correction_rms",
        ):
            if name in records:
                metrics[name] = records[name].mean()
        if "ball_racket_distance_m" in records:
            metrics["min_ball_racket_distance_m"] = records["ball_racket_distance_m"].min()
        if "hit_event" in records:
            hit_event = records["hit_event"]
            hit_count = hit_event.sum()
            safe_hit_count = jnp.maximum(hit_count, 1)
            positive_outgoing_z_hit_count = records[
                "positive_outgoing_z_event"
            ].sum()
            metrics["hit_events"] = hit_count
            metrics["positive_outgoing_z_hit_events"] = (
                positive_outgoing_z_hit_count
            )
            metrics["hit_contact_speed_m_s"] = jnp.where(
                hit_count > 0,
                jnp.where(hit_event, records["hit_contact_speed_m_s"], 0.0).sum() / safe_hit_count,
                0.0,
            )
            metrics["hit_racket_head_speed_m_s"] = jnp.where(
                hit_count > 0,
                jnp.where(
                    hit_event,
                    records["hit_racket_head_speed_m_s"],
                    0.0,
                ).sum()
                / safe_hit_count,
                0.0,
            )
            metrics["return_direction_score"] = jnp.where(
                hit_count > 0,
                jnp.where(hit_event, records["return_direction_score"], 0.0).sum() / safe_hit_count,
                0.0,
            )
            metrics["return_direction_signed_score"] = jnp.where(
                hit_count > 0,
                jnp.where(
                    hit_event,
                    records["return_direction_signed_score"],
                    0.0,
                ).sum()
                / safe_hit_count,
                0.0,
            )
            metrics["return_clearance_score"] = jnp.where(
                hit_count > 0,
                jnp.where(hit_event, records["return_clearance_score"], 0.0).sum() / safe_hit_count,
                0.0,
            )
            metrics["predicted_net_clearance_m"] = jnp.where(
                hit_count > 0,
                jnp.where(
                    hit_event,
                    records["predicted_net_clearance_m"],
                    0.0,
                ).sum()
                / safe_hit_count,
                0.0,
            )
            metrics["rewarded_hit_event_rebound_fraction"] = jnp.where(
                hit_count > 0,
                records["rewarded_hit_was_event_rebound"].sum() / safe_hit_count,
                0.0,
            )
            for name in (
                "hit_racket_direction_signed_score",
                "hit_racket_velocity_direction_signed_score",
                "hit_event_direction_reward_score",
                "hit_inverse_impact_score",
                "hit_inverse_impact_decomposed_score",
                "hit_inverse_impact_signed_normal_alignment",
                "hit_inverse_impact_normal_alignment",
                "hit_inverse_impact_racket_velocity_error_m_s",
                "hit_outgoing_velocity_x_m_s",
                "hit_outgoing_velocity_y_m_s",
                "hit_outgoing_velocity_z_m_s",
                "hit_outgoing_forward_velocity_m_s",
                "hit_racket_face_forward_alignment",
            ):
                metrics[name] = jnp.where(
                    hit_count > 0,
                    jnp.where(hit_event, records[name], 0.0).sum() / safe_hit_count,
                    0.0,
                )
            metrics["positive_outgoing_z_rate_on_hit"] = jnp.where(
                hit_count > 0,
                positive_outgoing_z_hit_count / safe_hit_count,
                0.0,
            )
            quality_hit_event = quality_success_event_mask(
                hit_event=hit_event,
                rewarded_hit_was_event_rebound=records[
                    "rewarded_hit_was_event_rebound"
                ],
                outgoing_z_m_s=records["hit_outgoing_velocity_z_m_s"],
                outgoing_forward_m_s=records[
                    "hit_outgoing_forward_velocity_m_s"
                ],
                predicted_net_clearance_m=records["predicted_net_clearance_m"],
                return_direction_signed_score=records[
                    "return_direction_signed_score"
                ],
                racket_face_forward_alignment=records[
                    "hit_racket_face_forward_alignment"
                ],
                body_fall=records["body_fall"],
                episode_completed_in_rollout=episode_completed_in_rollout,
                episode_fell_in_rollout=episode_fell_in_rollout,
                min_outgoing_z_m_s=cfg.quality_success_min_outgoing_z_m_s,
                min_forward_m_s=cfg.quality_success_min_forward_m_s,
                min_predicted_net_clearance_m=(
                    cfg.quality_success_min_predicted_net_clearance_m
                ),
                min_return_direction_signed_score=(
                    cfg.quality_success_min_return_direction_signed_score
                ),
                min_racket_face_forward_alignment=(
                    cfg.quality_success_min_racket_face_forward_alignment
                ),
                require_episode_no_fall=cfg.quality_success_require_episode_no_fall,
            )
            metrics["quality_return_events_per_1k_steps"] = (
                quality_hit_event.mean() * 1000.0
            )
            metrics["quality_return_events"] = quality_hit_event.sum()
            metrics["quality_return_rate"] = jnp.where(
                hit_count > 0,
                quality_hit_event.sum() / safe_hit_count,
                0.0,
            )
            hit_vertical_values = jnp.where(
                hit_event,
                records["hit_outgoing_velocity_z_m_s"],
                jnp.inf,
            ).reshape(-1)
            sorted_hit_vertical = jnp.sort(hit_vertical_values)
            p10_index = jnp.floor(
                0.10 * jnp.maximum(hit_count.astype(jnp.float32) - 1.0, 0.0)
            ).astype(jnp.int32)
            metrics["hit_outgoing_velocity_z_p10_m_s"] = jnp.where(
                hit_count > 0,
                sorted_hit_vertical[p10_index],
                0.0,
            )

        # The physical feed identity is retained on the transition before the
        # auto-reset.  Emit exact per-feed results so a bank average cannot
        # hide one easy contact feed or one systematically broken launch.
        feed_index = records["feed_index"]
        for feed_id in range(len(env.feed_bank)):
            on_feed = feed_index == feed_id
            feed_done = records["done"] & on_feed
            feed_episodes = feed_done.sum()
            safe_feed_episodes = jnp.maximum(feed_episodes, 1)
            feed_hit_event = records["hit_event"] & on_feed
            feed_hit_count = feed_hit_event.sum()
            safe_feed_hit_count = jnp.maximum(feed_hit_count, 1)
            prefix = f"feed_{feed_id:03d}"
            metrics[f"{prefix}/episodes_finished"] = feed_episodes
            metrics[f"{prefix}/hit_rate"] = (
                jnp.where(feed_done, records["hit"], False).sum() / safe_feed_episodes
            )
            metrics[f"{prefix}/positive_outgoing_z_rate_on_hit"] = jnp.where(
                feed_hit_count > 0,
                (records["positive_outgoing_z_event"] & on_feed).sum() / safe_feed_hit_count,
                0.0,
            )
            metrics[f"{prefix}/hit_outgoing_velocity_z_m_s"] = jnp.where(
                feed_hit_count > 0,
                jnp.where(
                    feed_hit_event,
                    records["hit_outgoing_velocity_z_m_s"],
                    0.0,
                ).sum()
                / safe_feed_hit_count,
                0.0,
            )
            metrics[f"{prefix}/hit_outgoing_forward_velocity_m_s"] = jnp.where(
                feed_hit_count > 0,
                jnp.where(
                    feed_hit_event,
                    records["hit_outgoing_forward_velocity_m_s"],
                    0.0,
                ).sum()
                / safe_feed_hit_count,
                0.0,
            )
            metrics[f"{prefix}/crossed_net_rate"] = (
                jnp.where(feed_done, records["crossed_net"], False).sum() / safe_feed_episodes
            )
            metrics[f"{prefix}/no_fall_rate"] = 1.0 - (
                jnp.where(feed_done, records["body_fall"], False).sum() / safe_feed_episodes
            )
        if "closest_approach_terminal_event" in records:
            closest_terminal_event = records[
                "closest_approach_terminal_event"
            ]
            closest_terminal_count = closest_terminal_event.sum()
            safe_closest_terminal_count = jnp.maximum(
                closest_terminal_count, 1
            )
            metrics["closest_approach_terminal_event_count"] = (
                closest_terminal_count
            )
            metrics["closest_approach_terminal_event_rate"] = (
                closest_terminal_count / n_done
            )
            metrics["closest_approach_terminal_direction_score"] = jnp.where(
                closest_terminal_count > 0,
                jnp.where(
                    closest_terminal_event,
                    records["closest_approach_terminal_direction_score"],
                    0.0,
                ).sum()
                / safe_closest_terminal_count,
                0.0,
            )
            metrics["closest_approach_terminal_distance_m"] = jnp.where(
                closest_terminal_count > 0,
                jnp.where(
                    closest_terminal_event,
                    records["closest_approach_distance_m"],
                    0.0,
                ).sum()
                / safe_closest_terminal_count,
                0.0,
            )
        metrics["stringbed_contact_events_per_1k_steps"] = records["stringbed_contact_event"].mean() * 1000.0
        metrics["event_rebound_events_per_1k_steps"] = records["event_rebound_event"].mean() * 1000.0
        if "valid_net_cross_event" in records:
            valid_cross_event = records["valid_net_cross_event"]
            valid_cross_count = valid_cross_event.sum()
            metrics["net_clearance_m"] = jnp.where(
                valid_cross_count > 0,
                jnp.where(
                    valid_cross_event,
                    records["net_clearance_m"],
                    0.0,
                ).sum()
                / jnp.maximum(valid_cross_count, 1),
                0.0,
            )
        if records["raw_action"].shape[0] > 1:
            action_delta = records["raw_action"][1:] - records["raw_action"][:-1]
            metrics["raw_action_rate_rms"] = jnp.sqrt(jnp.mean(jnp.square(action_delta)))
        if "opponent_back_landing" in records:
            metrics["opponent_back_landing_rate"] = jnp.where(done, records["opponent_back_landing"], 0).sum() / n_done
        if "impact_position_error_m" in records:
            hit_event = records["hit_event"]
            hit_count = hit_event.sum()
            safe_hit_count = jnp.maximum(hit_count, 1)
            missing_error = jnp.asarray(1.0e9, dtype=jnp.float32)

            def hit_mean(name):
                value = jnp.where(hit_event, records[name], 0.0).sum() / safe_hit_count
                return jnp.where(hit_count > 0, value, missing_error)

            def hit_rmse(name):
                value = jnp.sqrt(jnp.where(hit_event, jnp.square(records[name]), 0.0).sum() / safe_hit_count)
                return jnp.where(hit_count > 0, value, missing_error)

            metrics.update(
                {
                    "impact_position_error_m": hit_mean("impact_position_error_m"),
                    "center_hit_rate": jnp.where(
                        hit_count > 0,
                        jnp.where(
                            hit_event,
                            records["impact_rho2"] <= 0.25,
                            False,
                        ).sum()
                        / safe_hit_count,
                        0.0,
                    ),
                    "impact_timing_mae_s": hit_mean("impact_timing_error_s"),
                    "stringbed_normal_error_rad": hit_mean("stringbed_normal_error_rad"),
                    "racket_linear_velocity_rmse_m_s": hit_rmse("racket_linear_velocity_error_m_s"),
                    "racket_angular_velocity_rmse_rad_s": hit_rmse("racket_angular_velocity_error_rad_s"),
                }
            )
            landing_event = records["landing_event"]
            landing_count = landing_event.sum()
            safe_landing_count = jnp.maximum(landing_count, 1)
            landing_rmse = jnp.sqrt(
                jnp.where(
                    landing_event,
                    jnp.square(records["landing_error_m"]),
                    0.0,
                ).sum()
                / safe_landing_count
            )
            apex_mae = jnp.where(landing_event, records["apex_error_m"], 0.0).sum() / safe_landing_count
            metrics["landing_rmse_m"] = jnp.where(landing_count > 0, landing_rmse, missing_error)
            metrics["apex_mae_m"] = jnp.where(landing_count > 0, apex_mae, missing_error)
            recovery_event = records["recovery_metric_event"]
            recovery_count = recovery_event.sum()
            safe_recovery_count = jnp.maximum(recovery_count, 1)
            metrics["ready_pose_error"] = jnp.where(
                recovery_count > 0,
                jnp.where(recovery_event, records["ready_pose_error"], 0.0).sum() / safe_recovery_count,
                missing_error,
            )
            recovery_done = records["recovery_complete"]
            done_count = recovery_done.sum()
            metrics["recovery_ready_rate"] = jnp.where(
                done_count > 0,
                jnp.where(
                    recovery_done,
                    records["ready_pose_error"] <= 0.15,
                    False,
                ).sum()
                / jnp.maximum(done_count, 1),
                0.0,
            )
            metrics["no_fall_rate"] = 1.0 - metrics["fall_rate"]
        return agent, opt_state, obs_rms, env_states, key, metrics

    def train_iteration(agent, opt_state, obs_rms, env_states, key, teacher_bc_coef=None):
        coef = (
            float(cfg.teacher_bc_initial_coef)
            if teacher_bc_coef is None
            else teacher_bc_coef
        )
        return compiled_train_iteration(
            agent,
            opt_state,
            obs_rms,
            env_states,
            key,
            jnp.asarray(coef, dtype=jnp.float32),
        )

    return train_iteration


def _full_batch_budget(*, total_env_steps: int, steps_per_iteration: int) -> tuple[int, int, int]:
    """Return full JIT iterations, executed steps, and unused hard-cap budget.

    The vectorized rollout has a static shape, so a final partial iteration
    would require a separately compiled training function.  Production treats
    ``total_env_steps`` as a strict upper bound and therefore runs only full
    iterations that fit below it.  Reporting the small remainder explicitly is
    preferable to silently executing past the requested cap.
    """
    cap = int(total_env_steps)
    batch = int(steps_per_iteration)
    if cap <= 0:
        raise ValueError("total_env_steps must be positive")
    if batch <= 0:
        raise ValueError("steps_per_iteration must be positive")
    iterations = cap // batch
    if iterations <= 0:
        raise ValueError(
            "total_env_steps is smaller than one static JIT rollout: "
            f"cap={cap}, required_at_least={batch}; reduce num_envs or rollout_steps"
        )
    executed = iterations * batch
    return iterations, executed, cap - executed


def _training_iteration_budget(
    *,
    total_env_steps: int,
    steps_per_iteration: int,
    fresh_quality_teacher_bc: bool,
) -> tuple[int, int, int]:
    """Permit zero PPO steps only for a fresh sealed quality-teacher BC run."""

    if int(total_env_steps) == 0:
        if not fresh_quality_teacher_bc:
            raise ValueError(
                "total_env_steps=0 is reserved for a fresh sealed quality-teacher BC run"
            )
        return 0, 0, 0
    return _full_batch_budget(
        total_env_steps=total_env_steps,
        steps_per_iteration=steps_per_iteration,
    )


def reconcile_metrics_history(metrics_path: Path, *, checkpoint_iteration: int) -> dict[str, int]:
    """Atomically trim JSONL metrics to the exact resume boundary.

    A job may resume from an older immutable checkpoint after a later process
    already appended metrics.  Those future rows are not ancestors of the
    live policy, so retain one canonical row per completed iteration only.
    """

    boundary = int(checkpoint_iteration)
    if boundary < 0:
        raise ValueError("checkpoint_iteration must be non-negative")
    if not metrics_path.exists():
        return {"input_rows": 0, "retained_rows": 0, "removed_rows": 0}

    rows_by_iteration: dict[int, dict[str, Any]] = {}
    input_rows = 0
    for line_number, raw_line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        input_rows += 1
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Stage-3 metrics JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or isinstance(row.get("iteration"), bool):
            raise ValueError(f"Stage-3 metrics line {line_number} lacks an integer iteration")
        try:
            iteration = int(row["iteration"])
            exact_iteration = float(row["iteration"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Stage-3 metrics line {line_number} lacks an integer iteration") from exc
        if iteration < 1 or exact_iteration != float(iteration):
            raise ValueError(f"Stage-3 metrics line {line_number} has invalid iteration {row['iteration']!r}")
        if iteration <= boundary:
            rows_by_iteration[iteration] = row

    retained = [rows_by_iteration[index] for index in sorted(rows_by_iteration)]
    encoded = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in retained)
    tmp_path = metrics_path.with_name(f".{metrics_path.name}.resume-tmp")
    tmp_path.write_text(encoded, encoding="utf-8")
    os.replace(tmp_path, metrics_path)
    return {
        "input_rows": input_rows,
        "retained_rows": len(retained),
        "removed_rows": input_rows - len(retained),
    }


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestoredTrainingCheckpoint:
    agent: Any
    optimizer_state: Any
    obs_rms: ObsRms
    rng_key: jax.Array
    env_rng_key: jax.Array | None
    metadata: dict[str, Any]


VERSIONED_CHECKPOINT_SCHEMA = "incoming_hit_versioned_checkpoint_v1"
LATEST_POINTER_SCHEMA = "incoming_hit_checkpoint_pointer_v1"


def validate_training_feed_manifest(
    runtime_manifest: Any,
    *,
    checkpoint_manifest: Any | None = None,
    required: bool,
) -> None:
    """Fail closed on missing or changed Stage-3 training feed identity."""
    if required and not isinstance(runtime_manifest, dict):
        raise ValueError("Stage-3 training requires a verified feed-bank manifest")
    if runtime_manifest is not None and not isinstance(runtime_manifest, dict):
        raise ValueError("runtime training feed-bank manifest must be a mapping")
    if checkpoint_manifest is None:
        if required:
            raise ValueError("resume checkpoint is missing its training feed-bank manifest")
        return
    if not isinstance(checkpoint_manifest, dict):
        raise ValueError("resume training feed-bank manifest must be a mapping")
    if checkpoint_manifest != runtime_manifest:
        raise ValueError("resume Stage-3 training feed-bank contract changed")


def save_training_checkpoint(
    path: Path,
    *,
    agent: Any,
    optimizer_state: Any,
    obs_rms: ObsRms,
    rng_key: jax.Array,
    metadata: dict[str, Any],
    env_rng_key: jax.Array | None = None,
) -> None:
    """Save every state needed for deterministic Stage-3 continuation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    agent_flat, _ = jax.tree_util.tree_flatten(agent)
    optimizer_flat, _ = jax.tree_util.tree_flatten(optimizer_state)
    payload = {f"agent_{i}": np.asarray(value) for i, value in enumerate(agent_flat)}
    payload.update({f"optimizer_{i}": np.asarray(value) for i, value in enumerate(optimizer_flat)})
    payload["obs_mean"] = np.asarray(obs_rms.mean)
    payload["obs_var"] = np.asarray(obs_rms.var)
    payload["obs_count"] = np.asarray(obs_rms.count)
    payload["rng_key"] = np.asarray(rng_key)
    if env_rng_key is not None:
        payload["env_rng_key"] = np.asarray(env_rng_key)
    tmp_payload_path = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp_payload_path, **payload)
    os.replace(tmp_payload_path, path)
    metadata = dict(metadata)
    metadata["training_payload_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = path.with_suffix(".json")
    tmp_metadata_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    tmp_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp_metadata_path, metadata_path)


def save_versioned_training_checkpoint(
    out_dir: Path,
    *,
    agent: Any,
    optimizer_state: Any,
    obs_rms: ObsRms,
    rng_key: jax.Array,
    metadata: dict[str, Any],
    env_rng_key: jax.Array | None = None,
) -> Path:
    """Commit an immutable checkpoint directory, then atomically move latest.

    A crash can leave only a hidden temporary directory; readers never observe
    a new pointer until payload, metadata and the completion marker are all
    durable in the final version directory.
    """

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    try:
        env_steps = int(metadata["env_steps"])
        iteration = int(metadata["iteration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("versioned checkpoint requires iteration and env_steps") from exc
    if env_steps < 0 or iteration < 0:
        raise ValueError("checkpoint iteration/env_steps must be non-negative")
    version_name = f"checkpoint_{env_steps:012d}"
    final_dir = checkpoints / version_name
    temp_dir = checkpoints / f".{version_name}.{os.getpid()}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    payload_path = temp_dir / "policy.npz"
    try:
        save_training_checkpoint(
            payload_path,
            agent=agent,
            optimizer_state=optimizer_state,
            obs_rms=obs_rms,
            rng_key=rng_key,
            env_rng_key=env_rng_key,
            metadata={
                **metadata,
                "versioned_checkpoint_schema": VERSIONED_CHECKPOINT_SCHEMA,
                "version_name": version_name,
            },
        )
        metadata_path = payload_path.with_suffix(".json")
        completion = {
            "schema_version": VERSIONED_CHECKPOINT_SCHEMA,
            "version_name": version_name,
            "iteration": iteration,
            "env_steps": env_steps,
            "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
        completion["binding_sha256"] = _stable_json_hash(completion)
        (temp_dir / "_COMPLETE.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if final_dir.exists():
            # Never mutate an already published version.  An exact retry is
            # accepted semantically (NPZ ZIP timestamps may change its byte
            # hash); a different state at the same env-step is corruption.
            existing = _read_version_completion(final_dir)
            if existing != completion and not _training_checkpoint_semantically_equal(
                final_dir,
                temp_dir,
            ):
                raise ValueError(f"checkpoint version collision at env_steps={env_steps}")
            shutil.rmtree(temp_dir)
        else:
            os.replace(temp_dir, final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    final_payload = final_dir / "policy.npz"
    final_metadata = final_dir / "policy.json"
    completion = _read_version_completion(final_dir)
    pointer = {
        "schema_version": LATEST_POINTER_SCHEMA,
        "version_name": version_name,
        "checkpoint_dir": str(final_dir.resolve()),
        "payload_path": str(final_payload.resolve()),
        "payload_sha256": completion["payload_sha256"],
        "metadata_sha256": completion["metadata_sha256"],
        "iteration": iteration,
        "env_steps": env_steps,
    }
    pointer["binding_sha256"] = _stable_json_hash(pointer)
    pointer_path = root / "policy_latest.json"
    pointer_tmp = root / f".policy_latest.{os.getpid()}.tmp.json"
    pointer_tmp.write_text(
        json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(pointer_tmp, pointer_path)

    latest_payload = root / "policy_latest.npz"
    link_tmp = root / f".policy_latest.{os.getpid()}.tmp.npz"
    if link_tmp.exists() or link_tmp.is_symlink():
        link_tmp.unlink()
    relative_target = os.path.relpath(final_payload, root)
    os.symlink(relative_target, link_tmp)
    os.replace(link_tmp, latest_payload)
    # Re-read through the public pointer before returning success.
    resolved_payload, resolved_metadata = resolve_training_checkpoint(latest_payload)
    if resolved_payload != final_payload.resolve() or resolved_metadata != final_metadata.resolve():
        raise RuntimeError("published Stage-3 latest pointer did not resolve to committed version")
    return latest_payload


def _training_checkpoint_semantically_equal(left_dir: Path, right_dir: Path) -> bool:
    """Compare retry payloads without depending on ZIP container timestamps."""

    try:
        left_meta = json.loads((left_dir / "policy.json").read_text(encoding="utf-8"))
        right_meta = json.loads((right_dir / "policy.json").read_text(encoding="utf-8"))
        if not isinstance(left_meta, dict) or not isinstance(right_meta, dict):
            return False
        left_meta.pop("training_payload_sha256", None)
        right_meta.pop("training_payload_sha256", None)
        if left_meta != right_meta:
            return False
        with (
            np.load(left_dir / "policy.npz", allow_pickle=False) as left,
            np.load(right_dir / "policy.npz", allow_pickle=False) as right,
        ):
            if set(left.files) != set(right.files):
                return False
            return all(np.array_equal(left[name], right[name], equal_nan=True) for name in left.files)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def resolve_training_checkpoint(path: Path) -> tuple[Path, Path]:
    """Resolve latest pointers, version directories and legacy NPZ pairs."""

    candidate = Path(path).expanduser()
    if candidate.is_dir():
        if (candidate / "_COMPLETE.json").is_file():
            payload = candidate / "policy.npz"
        elif (candidate / "policy_latest.json").is_file():
            return resolve_training_checkpoint(candidate / "policy_latest.json")
        else:
            raise FileNotFoundError(f"checkpoint directory is incomplete: {candidate}")
    elif candidate.suffix == ".json":
        pointer = json.loads(candidate.read_text(encoding="utf-8"))
        if pointer.get("schema_version") != LATEST_POINTER_SCHEMA:
            raise ValueError(f"unsupported checkpoint pointer: {candidate}")
        recorded_hash = pointer.get("binding_sha256")
        unbound = dict(pointer)
        unbound.pop("binding_sha256", None)
        if recorded_hash != _stable_json_hash(unbound):
            raise ValueError("Stage-3 latest pointer binding hash mismatch")
        payload = Path(str(pointer.get("payload_path", ""))).expanduser()
    else:
        payload = candidate.resolve(strict=True) if candidate.is_symlink() else candidate
    payload = payload.resolve(strict=True)
    metadata_path = payload.with_suffix(".json")
    if not payload.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"training checkpoint is incomplete: {payload}")
    if (payload.parent / "_COMPLETE.json").is_file():
        completion = _read_version_completion(payload.parent)
        if hashlib.sha256(payload.read_bytes()).hexdigest() != completion["payload_sha256"]:
            raise ValueError("versioned checkpoint payload fingerprint mismatch")
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != completion["metadata_sha256"]:
            raise ValueError("versioned checkpoint metadata fingerprint mismatch")
    return payload, metadata_path


def load_training_checkpoint_metadata(path: Path) -> dict[str, Any]:
    _, metadata_path = resolve_training_checkpoint(path)
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training checkpoint metadata must be a JSON object")
    return value


def load_training_checkpoint(
    path: Path,
    *,
    agent_template: Any,
    optimizer_state_template: Any,
) -> RestoredTrainingCheckpoint:
    path, metadata_path = resolve_training_checkpoint(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_payload_hash = metadata.get("training_payload_sha256")
    if expected_payload_hash is not None:
        actual_payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(expected_payload_hash) != actual_payload_hash:
            raise ValueError(
                "Stage-3 training checkpoint content fingerprint mismatch: "
                f"stored={expected_payload_hash} computed={actual_payload_hash}"
            )
    with np.load(path) as payload:
        agent_template_flat, agent_tree = jax.tree_util.tree_flatten(agent_template)
        optimizer_template_flat, optimizer_tree = jax.tree_util.tree_flatten(optimizer_state_template)
        agent_flat = [jnp.asarray(payload[f"agent_{index}"]) for index in range(len(agent_template_flat))]
        optimizer_flat = [jnp.asarray(payload[f"optimizer_{index}"]) for index in range(len(optimizer_template_flat))]
        for label, actual_values, expected_values in (
            ("agent", agent_flat, agent_template_flat),
            ("optimizer", optimizer_flat, optimizer_template_flat),
        ):
            for index, (actual, expected) in enumerate(zip(actual_values, expected_values, strict=True)):
                if actual.shape != np.shape(expected):
                    raise ValueError(
                        f"checkpoint {label} leaf {index} shape {actual.shape} != runtime template {np.shape(expected)}"
                    )
        agent = jax.tree_util.tree_unflatten(agent_tree, agent_flat)
        optimizer_state = jax.tree_util.tree_unflatten(optimizer_tree, optimizer_flat)
        obs_rms = ObsRms(
            jnp.asarray(payload["obs_mean"]),
            jnp.asarray(payload["obs_var"]),
            jnp.asarray(payload["obs_count"]),
        )
        rng_key = jnp.asarray(payload["rng_key"])
        env_rng_key = jnp.asarray(payload["env_rng_key"]) if "env_rng_key" in payload.files else None
    return RestoredTrainingCheckpoint(agent, optimizer_state, obs_rms, rng_key, env_rng_key, metadata)


def load_training_actor_checkpoint(
    path: Path,
    *,
    agent_template: Any,
) -> RestoredTrainingCheckpoint:
    """Load actor/normalizer state without coupling a warm start to optimizer ABI.

    Reward-repair initialization intentionally resets the optimizer and value
    network.  Loading optimizer leaves solely to discover the old actor tree
    makes adding a zero-initialized adapter needlessly incompatible with older
    checkpoints, so this narrow loader verifies and restores only the state
    that is actually transferred.
    """

    path, metadata_path = resolve_training_checkpoint(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_payload_hash = metadata.get("training_payload_sha256")
    if expected_payload_hash is not None:
        actual_payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(expected_payload_hash) != actual_payload_hash:
            raise ValueError(
                "Stage-3 training checkpoint content fingerprint mismatch: "
                f"stored={expected_payload_hash} computed={actual_payload_hash}"
            )
    with np.load(path) as payload:
        template_flat, agent_tree = jax.tree_util.tree_flatten(agent_template)
        agent_keys = sorted(
            (name for name in payload.files if name.startswith("agent_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if len(agent_keys) != len(template_flat):
            raise ValueError(
                "checkpoint actor leaf count does not match its declared architecture: "
                f"checkpoint={len(agent_keys)} template={len(template_flat)}"
            )
        agent_flat = [jnp.asarray(payload[name]) for name in agent_keys]
        for index, (actual, expected) in enumerate(zip(agent_flat, template_flat, strict=True)):
            if actual.shape != np.shape(expected):
                raise ValueError(
                    f"checkpoint actor leaf {index} shape {actual.shape} "
                    f"!= source template {np.shape(expected)}"
                )
        agent = jax.tree_util.tree_unflatten(agent_tree, agent_flat)
        obs_rms = ObsRms(
            jnp.asarray(payload["obs_mean"]),
            jnp.asarray(payload["obs_var"]),
            jnp.asarray(payload["obs_count"]),
        )
        rng_key = jnp.asarray(payload["rng_key"])
        env_rng_key = jnp.asarray(payload["env_rng_key"]) if "env_rng_key" in payload.files else None
    return RestoredTrainingCheckpoint(agent, None, obs_rms, rng_key, env_rng_key, metadata)


def _residual_scale_vector_from_binding(
    binding: dict[str, Any],
    *,
    action_size: int,
    env_steps: int | None = None,
) -> np.ndarray:
    """Reconstruct target or scheduled effective residual authority exactly."""

    size = int(action_size)
    if size <= 0:
        raise ValueError("residual authority action_size must be positive")
    try:
        base_scale = float(binding["residual_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen residual binding has no valid residual_scale") from exc
    if not np.isfinite(base_scale) or not 0.0 <= base_scale <= 2.0:
        raise ValueError("frozen residual binding residual_scale is outside [0, 2]")
    target = np.full(size, base_scale, dtype=np.float64)
    seen: set[int] = set()
    for entry in binding.get("residual_scale_overrides", []):
        if not isinstance(entry, dict):
            raise ValueError("frozen residual scale override must be a mapping")
        try:
            index = int(entry["actuator_id"])
            scale = float(entry["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frozen residual scale override is malformed") from exc
        if index in seen or not 0 <= index < size:
            raise ValueError("frozen residual scale override actuator_id is invalid or duplicated")
        if not np.isfinite(scale) or not 0.0 <= scale <= 2.0:
            raise ValueError("frozen residual scale override is outside [0, 2]")
        seen.add(index)
        target[index] = scale
    recorded_target_hash = binding.get("residual_scale_vector_sha256")
    target_hash = hashlib.sha256(np.asarray(target, dtype="<f8").tobytes()).hexdigest()
    if recorded_target_hash is not None and recorded_target_hash != target_hash:
        raise ValueError("frozen residual target scale vector hash mismatch")

    schedule = binding.get("residual_scale_schedule")
    if schedule is None or env_steps is None:
        return target
    if not isinstance(schedule, dict) or schedule.get("schema_version") != (
        "incoming_hit_residual_authority_schedule_v1"
    ):
        raise ValueError("frozen residual authority schedule is malformed")
    if schedule.get("interpolation") != "linear_env_steps":
        raise ValueError("unsupported frozen residual authority interpolation")
    initial = target.copy()
    schedule_seen: set[int] = set()
    for entry in schedule.get("scheduled_actuators", []):
        if not isinstance(entry, dict):
            raise ValueError("scheduled residual actuator must be a mapping")
        try:
            index = int(entry["actuator_id"])
            initial_scale = float(entry["initial_scale"])
            target_scale = float(entry["target_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scheduled residual actuator is malformed") from exc
        if index in schedule_seen or not 0 <= index < size:
            raise ValueError("scheduled residual actuator_id is invalid or duplicated")
        if not np.isfinite(initial_scale) or not 0.0 <= initial_scale <= 2.0:
            raise ValueError("scheduled initial residual scale is outside [0, 2]")
        if target[index] != target_scale:
            raise ValueError("scheduled residual target scale differs from frozen binding")
        schedule_seen.add(index)
        initial[index] = initial_scale
    initial_hash = hashlib.sha256(np.asarray(initial, dtype="<f8").tobytes()).hexdigest()
    if schedule.get("initial_scale_vector_sha256") != initial_hash:
        raise ValueError("scheduled initial residual scale vector hash mismatch")
    try:
        ramp_steps = int(schedule["ramp_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scheduled residual ramp_steps is malformed") from exc
    if ramp_steps <= 0:
        raise ValueError("scheduled residual ramp_steps must be positive")
    progress = min(1.0, max(0.0, float(env_steps) / float(ramp_steps)))
    return initial + progress * (target - initial)


def _residual_binding_identity(binding: dict[str, Any]) -> dict[str, Any]:
    """Return frozen-base identity with only authority-amplitude fields removed."""

    result = dict(binding)
    for name in (
        "binding_sha256",
        "residual_scale",
        "residual_scale_overrides",
        "residual_scale_vector_sha256",
        "residual_scale_schedule",
    ):
        result.pop(name, None)
    return result


def _base_timing_binding_identity(binding: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen-base identity with only phase timing removed."""

    result = dict(binding)
    result.pop("binding_sha256", None)
    result.pop("phase_advance_s", None)
    return result


def _checkpoint_action_prior_binding(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a checkpoint's sealed action-prior lineage, if it has one.

    A time-interpolated prior is part of the actor's effective action, even
    though its knots are stored in ``TrainConfig`` rather than the parameter
    tree.  Actor-only initialization must therefore carry the exact same
    audited dataset forward instead of silently dropping or replacing it.
    """

    config = dict(metadata.get("config", {}) or {})
    if config.get("teacher_action_prior_mode", "none") != (
        "time_interpolated_frozen_plus_delta"
    ):
        return None
    if not config.get("teacher_prior_time_to_intercept_s") or not config.get(
        "teacher_prior_correction_raw"
    ):
        raise ValueError("actor checkpoint has an incomplete frozen action prior")

    report = dict(metadata.get("teacher_bc_pretrain_report", {}) or {})
    recorded_report_hash = report.get("report_sha256")
    unhashed_report = dict(report)
    unhashed_report.pop("report_sha256", None)
    if (
        report.get("schema_version")
        != "stage3_selected_correction_bc_pretrain_v1"
        or report.get("passed") is not True
        or recorded_report_hash != _stable_json_hash(unhashed_report)
    ):
        raise ValueError("actor checkpoint has an invalid action-prior report")

    binding = dict(report.get("teacher_binding", {}) or {})
    recorded_binding_hash = binding.get("binding_sha256")
    unhashed_binding = dict(binding)
    unhashed_binding.pop("binding_sha256", None)
    if not binding or recorded_binding_hash != _stable_json_hash(unhashed_binding):
        raise ValueError("actor checkpoint has an invalid action-prior binding")
    return binding


def _resume_action_prior_source(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], str | None, float | None]:
    """Recover the exact sealed dataset inputs needed to resume teacher use.

    A frozen time-interpolated prior and ordinary quality-imitation BC both
    refer to the checkpoint used when the CEM trajectory was certified, not
    to the checkpoint currently being resumed.  Rebuilding either binding
    with ``source_checkpoint_sha256=None`` silently changes its identity.
    """

    binding = _checkpoint_action_prior_binding(metadata)
    if binding is None:
        report = dict(metadata.get("teacher_bc_pretrain_report", {}) or {})
        recorded_report_hash = report.get("report_sha256")
        unsigned_report = dict(report)
        unsigned_report.pop("report_sha256", None)
        if (
            report.get("schema_version")
            != "stage3_selected_correction_bc_pretrain_v1"
            or report.get("passed") is not True
            or recorded_report_hash != _stable_json_hash(unsigned_report)
        ):
            raise ValueError("resume checkpoint has no valid sealed teacher-BC report")
        binding = dict(report.get("teacher_binding", {}) or {})
        recorded_binding_hash = binding.get("binding_sha256")
        unsigned_binding = dict(binding)
        unsigned_binding.pop("binding_sha256", None)
        if (
            binding.get("schema_version")
            not in {
                "stage3_quality_teacher_dataset_binding_v1",
                "stage3_cpu_certified_exploration_prior_binding_v1",
            }
            or recorded_binding_hash != _stable_json_hash(unsigned_binding)
        ):
            raise ValueError("resume checkpoint has no valid sealed teacher binding")
    source_checkpoint_sha256 = str(
        binding.get("source_checkpoint_sha256", "")
    ).strip() or None
    timing = dict(binding.get("base_timing_transfer", {}) or {})
    source_phase_advance_s: float | None = None
    if timing:
        source_phase_advance_s = float(
            timing.get("source_phase_advance_s", math.nan)
        )
        if not math.isfinite(source_phase_advance_s):
            raise ValueError("resume teacher timing transfer is invalid")
    return binding, source_checkpoint_sha256, source_phase_advance_s


def load_actor_initialization(
    path: Path,
    *,
    agent_template: Any,
    optimizer_state_template: Any,
    env: IncomingHitMjxEnv,
    base_timing_transfer_evidence: dict[str, Any] | None = None,
    quality_teacher_binding: dict[str, Any] | None = None,
    reset_correction_std: bool = False,
) -> tuple[RestoredTrainingCheckpoint, dict[str, Any]]:
    """Load an actor-only warm start across an intentional reward repair.

    The actor and observation normalizer may transfer when the physical scene,
    observation/action dimensions, frozen body prior, attachment and feed bank
    are identical.  A change to frozen-base phase timing is accepted only with
    a quality-teacher binding produced by an independent CPU MuJoCo replay.
    Value parameters, optimizer moments, RNG and curriculum state are
    deliberately not transferred.
    """

    # Keep the public argument for call-site compatibility.  Optimizer state is
    # deliberately not transferred during actor-only initialization.
    del optimizer_state_template
    metadata = load_training_checkpoint_metadata(Path(path))
    source_config = dict(metadata.get("config", {}) or {})
    if source_config:
        source_hidden = tuple(int(size) for size in metadata.get("hidden", source_config.get("hidden", ())))
        if not source_hidden:
            raise ValueError("actor initialization source checkpoint is missing hidden sizes")
        source_delta_hidden = tuple(int(size) for size in source_config.get("policy_delta_hidden", ()))
        source_refinement_hidden = tuple(
            int(size) for size in source_config.get("policy_refinement_delta_hidden", ())
        )
        source_correction_hidden = tuple(
            int(size) for size in source_config.get("policy_correction_hidden", ())
        )
        source_correction_std_init = tuple(
            float(value) for value in source_config.get("correction_std_init", ())
        )
        source_agent_template = init_agent(
            jax.random.PRNGKey(0),
            obs_size=int(metadata.get("obs_size", -1)),
            action_size=int(metadata.get("action_size", -1)),
            hidden=source_hidden,
            action_std_init=float(source_config.get("action_std_init", 0.35)),
            policy_delta_hidden=source_delta_hidden,
            policy_refinement_delta_hidden=source_refinement_hidden,
            policy_correction_hidden=source_correction_hidden,
            correction_action_size=len(tuple(source_config.get("policy_trainable_action_indices", ()))),
            correction_std_init=source_correction_std_init,
        )
    else:
        # Historical/unit-test checkpoints predate the serialized TrainConfig.
        # They cannot contain the new adapter, so use the runtime base tree.
        source_delta_hidden = ()
        source_agent_template = {
            name: value
            for name, value in agent_template.items()
            if name
            not in {
                "policy_delta",
                "policy_refinement_delta",
                "policy_correction",
                "correction_log_std",
            }
        }
    restored = load_training_actor_checkpoint(
        Path(path),
        agent_template=source_agent_template,
    )
    metadata = restored.metadata
    inherited_action_prior = _checkpoint_action_prior_binding(metadata)
    action_prior_lineage: dict[str, Any] | None = None
    if inherited_action_prior is not None:
        if quality_teacher_binding is None:
            raise ValueError(
                "actor initialization would drop the inherited frozen action prior"
            )
        if inherited_action_prior != quality_teacher_binding:
            raise ValueError(
                "actor initialization action-prior dataset differs from the source actor"
            )
        action_prior_lineage = {
            "schema_version": "stage3_actor_action_prior_lineage_v1",
            "mode": "exact_inherited_dataset_binding",
            "teacher_binding_sha256": inherited_action_prior["binding_sha256"],
        }
    for name, expected in (
        ("obs_size", int(env.observation_size)),
        ("action_size", int(env.action_size)),
    ):
        actual = int(metadata.get(name, -1))
        if actual != expected:
            raise ValueError(f"actor initialization {name} mismatch: checkpoint={actual}, runtime={expected}")
    source_control = dict(metadata.get("control_manifest", {}) or {})
    runtime_control = dict(getattr(env, "control_manifest", {}) or {})
    if not source_control or not runtime_control:
        raise ValueError("actor initialization requires source and runtime control manifests")
    for name in ("schema_version", "filter_finger_observation", "racket_attachment"):
        if source_control.get(name) != runtime_control.get(name):
            raise ValueError(f"actor initialization physical control field changed: {name}")
    source_frozen = source_control.get("frozen_base_residual")
    runtime_frozen = runtime_control.get("frozen_base_residual")
    authority_transfer: dict[str, Any] | None = None
    timing_transfer: dict[str, Any] | None = None
    if source_frozen != runtime_frozen:
        if not isinstance(source_frozen, dict) or not isinstance(runtime_frozen, dict):
            raise ValueError("actor initialization physical control field changed: frozen_base_residual")
        if _residual_binding_identity(source_frozen) == _residual_binding_identity(runtime_frozen):
            if runtime_frozen.get("residual_scale_schedule") is None:
                raise ValueError("actor initialization residual authority changed without a safe schedule")
            source_scale = _residual_scale_vector_from_binding(
                source_frozen,
                action_size=int(env.action_size),
                env_steps=int(metadata.get("env_steps", 0)),
            )
            runtime_initial_scale = _residual_scale_vector_from_binding(
                runtime_frozen,
                action_size=int(env.action_size),
                env_steps=0,
            )
            if not np.array_equal(source_scale, runtime_initial_scale):
                raise ValueError("actor initialization scheduled residual authority is not physically equivalent")
            runtime_target_scale = _residual_scale_vector_from_binding(
                runtime_frozen,
                action_size=int(env.action_size),
            )
            authority_transfer = {
                "schema_version": "stage3_residual_authority_transfer_v1",
                "mode": "exact_initial_action_scale_then_linear_ramp",
                "source_effective_scale_vector_sha256": hashlib.sha256(
                    np.asarray(source_scale, dtype="<f8").tobytes()
                ).hexdigest(),
                "runtime_initial_scale_vector_sha256": hashlib.sha256(
                    np.asarray(runtime_initial_scale, dtype="<f8").tobytes()
                ).hexdigest(),
                "runtime_target_scale_vector_sha256": hashlib.sha256(
                    np.asarray(runtime_target_scale, dtype="<f8").tobytes()
                ).hexdigest(),
                "changed_actuator_count": int(np.count_nonzero(runtime_target_scale != runtime_initial_scale)),
                "schedule": runtime_frozen["residual_scale_schedule"],
            }
        elif _base_timing_binding_identity(source_frozen) == _base_timing_binding_identity(runtime_frozen):
            evidence = dict(base_timing_transfer_evidence or {})
            recorded_evidence_hash = evidence.get("evidence_sha256")
            unhashed_evidence = {
                name: value
                for name, value in evidence.items()
                if name not in {"evidence_sha256", "teacher_cem_report_sha256"}
            }
            source_phase = float(source_frozen.get("phase_advance_s", math.nan))
            runtime_phase = float(runtime_frozen.get("phase_advance_s", math.nan))
            if (
                evidence.get("schema_version")
                != "stage3_teacher_verified_base_timing_transfer_v1"
                or evidence.get("verification_source")
                != "independent_cpu_mujoco_quality_replay_at_runtime_timing"
                or evidence.get("cpu_quality_verified") is not True
                or recorded_evidence_hash != _stable_json_hash(unhashed_evidence)
                or not isinstance(evidence.get("teacher_cem_report_sha256"), str)
                or len(evidence["teacher_cem_report_sha256"]) != 64
                or not math.isclose(
                    float(evidence.get("source_phase_advance_s", math.nan)),
                    source_phase,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    float(evidence.get("runtime_phase_advance_s", math.nan)),
                    runtime_phase,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "actor initialization base timing changed without matching quality-teacher evidence"
                )
            timing_transfer = {
                "schema_version": "stage3_actor_base_timing_transfer_v1",
                "mode": "quality_teacher_verified_runtime_phase",
                "source_phase_advance_s": source_phase,
                "runtime_phase_advance_s": runtime_phase,
                "teacher_cem_report_sha256": evidence["teacher_cem_report_sha256"],
                "teacher_timing_evidence_sha256": recorded_evidence_hash,
            }
        else:
            raise ValueError("actor initialization physical control field changed: frozen_base_residual")
    source_environment = dict(source_control.get("environment_abi", {}) or {})
    runtime_environment = dict(runtime_control.get("environment_abi", {}) or {})
    for mutable_reward_field in (
        "reward_weights",
        "reward_semantics",
        "return_constraints",
        "max_episode_steps",
    ):
        source_environment.pop(mutable_reward_field, None)
        runtime_environment.pop(mutable_reward_field, None)
    if timing_transfer is not None:
        source_environment_phase = float(
            source_environment.pop("swing_phase_advance_s", math.nan)
        )
        runtime_environment_phase = float(
            runtime_environment.pop("swing_phase_advance_s", math.nan)
        )
        if (
            not math.isclose(
                source_environment_phase,
                float(timing_transfer["source_phase_advance_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                runtime_environment_phase,
                float(timing_transfer["runtime_phase_advance_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("actor initialization base timing differs inside the environment ABI")
    impact_semantics_transfer: dict[str, Any] | None = None
    source_event_semantics = source_environment.get(
        "event_rebound_contact_semantics"
    )
    runtime_event_semantics = runtime_environment.get(
        "event_rebound_contact_semantics"
    )
    if source_event_semantics != runtime_event_semantics:
        teacher_binding = dict(quality_teacher_binding or {})
        recorded_binding_hash = teacher_binding.get("binding_sha256")
        unhashed_teacher_binding = dict(teacher_binding)
        unhashed_teacher_binding.pop("binding_sha256", None)
        teacher_schema = str(teacher_binding.get("schema_version", ""))
        teacher_backend = str(teacher_binding.get("training_backend", ""))
        quality_physics_evidence = bool(
            teacher_schema == "stage3_quality_teacher_dataset_binding_v1"
            and teacher_binding.get("training_backend_quality_verified") is True
            and teacher_binding.get("verification_source")
            == _TEACHER_VERIFICATION_SOURCE_BY_BACKEND.get(teacher_backend)
        )
        exploration_physics_evidence = bool(
            teacher_schema
            == "stage3_cpu_certified_exploration_prior_binding_v1"
            and teacher_backend == "warp"
            and teacher_binding.get("prior_role")
            == "bounded_exploration_prior_not_quality_teacher"
            and teacher_binding.get("quality_teacher") is False
            and teacher_binding.get("training_backend_quality_verified") is False
            and float(
                teacher_binding.get(
                    "training_backend_observed_teacher_success_rate", 0.0
                )
            )
            > 0.0
            and float(
                teacher_binding.get(
                    "training_backend_observed_no_fall_rate", 0.0
                )
            )
            >= 1.0
        )
        physics_evidence_valid = bool(
            source_event_semantics is None
            and runtime_event_semantics == _EVENT_REBOUND_CONTACT_SEMANTICS
            and recorded_binding_hash
            == _stable_json_hash(unhashed_teacher_binding)
            and teacher_binding.get("cpu_quality_verified") is True
            and teacher_binding.get("outgoing_velocity_semantics")
            == _OUTGOING_VELOCITY_SEMANTICS
            and teacher_binding.get("event_rebound_contact_semantics")
            == _EVENT_REBOUND_CONTACT_SEMANTICS
            and (quality_physics_evidence or exploration_physics_evidence)
        )
        if not physics_evidence_valid:
            raise ValueError(
                "actor initialization changed impact semantics without matching "
                "single-event CPU/training-backend evidence"
            )
        source_environment.pop("event_rebound_contact_semantics", None)
        runtime_environment.pop("event_rebound_contact_semantics", None)
        impact_semantics_transfer = {
            "schema_version": "stage3_actor_impact_semantics_transfer_v1",
            "mode": "legacy_unmarked_to_single_event_cooldown_v2",
            "source_event_rebound_contact_semantics": None,
            "runtime_event_rebound_contact_semantics": runtime_event_semantics,
            "teacher_binding_sha256": recorded_binding_hash,
        }
    if source_environment != runtime_environment:
        raise ValueError("actor initialization changed the physical environment ABI")
    runtime_feed_manifest = getattr(env, "feed_bank_manifest", None)
    source_feed_manifest = metadata.get("training_feed_manifest")
    if not isinstance(source_feed_manifest, dict) or not isinstance(runtime_feed_manifest, dict):
        raise ValueError("actor initialization requires source and runtime feed manifests")
    source_feed_producer = _actor_initialization_feed_producer(source_feed_manifest)
    runtime_feed_producer = _actor_initialization_feed_producer(runtime_feed_manifest)
    feed_transfer: dict[str, Any] | None = None
    if source_feed_producer != runtime_feed_producer:
        teacher_binding = dict(quality_teacher_binding or {})
        recorded_binding_hash = teacher_binding.get("binding_sha256")
        unhashed_teacher_binding = dict(teacher_binding)
        unhashed_teacher_binding.pop("binding_sha256", None)
        teacher_feed = str(teacher_binding.get("feed_fingerprint", ""))
        teacher_backend = str(teacher_binding.get("training_backend", ""))
        teacher_schema = str(teacher_binding.get("schema_version", ""))
        expected_verification_source = (
            _TEACHER_VERIFICATION_SOURCE_BY_BACKEND.get(teacher_backend)
        )
        runtime_backend = str(getattr(env, "impl", teacher_backend))
        runtime_fingerprints = [
            str(value) for value in runtime_feed_producer.get("sample_fingerprints", [])
        ]
        consumer_order = dict(runtime_feed_manifest.get("consumer_order", {}) or {})
        consumer_fingerprints = [
            str(value) for value in consumer_order.get("sample_fingerprints", [])
        ]
        source_payload, _source_metadata_path = resolve_training_checkpoint(Path(path))
        common_transfer_valid = bool(
            recorded_binding_hash == _stable_json_hash(unhashed_teacher_binding)
            and runtime_backend == teacher_backend
            and teacher_binding.get("source_checkpoint_sha256")
            == hashlib.sha256(source_payload.read_bytes()).hexdigest()
            and teacher_feed in runtime_fingerprints
            and consumer_order.get("mode") == "explicit_fingerprint_order"
            and consumer_fingerprints
            and consumer_fingerprints[0] == teacher_feed
        )
        quality_transfer_valid = bool(
            teacher_schema == "stage3_quality_teacher_dataset_binding_v1"
            and expected_verification_source is not None
            and teacher_binding.get("verification_source")
            == expected_verification_source
            and teacher_binding.get("training_backend_quality_verified") is True
        )
        exploration_transfer_valid = bool(
            teacher_schema
            == "stage3_cpu_certified_exploration_prior_binding_v1"
            and teacher_backend == "warp"
            and teacher_binding.get("verification_source")
            == "cpu_quality_plus_nonrobust_warp_exploration_replay"
            and teacher_binding.get("prior_role")
            == "bounded_exploration_prior_not_quality_teacher"
            and teacher_binding.get("quality_teacher") is False
            and teacher_binding.get("cpu_quality_verified") is True
            and teacher_binding.get("training_backend_quality_verified") is False
            and float(
                teacher_binding.get(
                    "training_backend_observed_teacher_success_rate", 0.0
                )
            )
            > 0.0
            and float(
                teacher_binding.get("training_backend_observed_no_fall_rate", 0.0)
            )
            >= 1.0
        )
        if not common_transfer_valid or not (
            quality_transfer_valid or exploration_transfer_valid
        ):
            raise ValueError(
                "actor initialization changed the training feed bank without matching "
                "quality-teacher or bounded-exploration evidence"
            )
        feed_transfer = {
            "schema_version": (
                "stage3_actor_feed_bank_transfer_v1"
                if quality_transfer_valid
                else "stage3_actor_feed_bank_exploration_transfer_v1"
            ),
            "mode": (
                "quality_teacher_first_then_gated_curriculum"
                if quality_transfer_valid
                else "cpu_certified_exploration_prior_first_then_gated_curriculum"
            ),
            "source_feed_producer_sha256": _stable_json_hash(source_feed_producer),
            "runtime_feed_producer_sha256": _stable_json_hash(runtime_feed_producer),
            "teacher_feed_fingerprint": teacher_feed,
            "teacher_binding_sha256": recorded_binding_hash,
        }
    source_policy_shapes = [np.shape(value) for value in jax.tree_util.tree_leaves(restored.agent["policy"])]
    runtime_policy_shapes = [np.shape(value) for value in jax.tree_util.tree_leaves(agent_template["policy"])]
    if source_policy_shapes != runtime_policy_shapes:
        raise ValueError("actor initialization base policy architecture changed")
    source_has_delta = "policy_delta" in restored.agent
    runtime_has_delta = "policy_delta" in agent_template
    if source_has_delta and not runtime_has_delta:
        raise ValueError("actor initialization would discard a learned policy delta adapter")
    if source_has_delta:
        source_delta_shapes = [
            np.shape(value) for value in jax.tree_util.tree_leaves(restored.agent["policy_delta"])
        ]
        runtime_delta_shapes = [
            np.shape(value) for value in jax.tree_util.tree_leaves(agent_template["policy_delta"])
        ]
        if source_delta_shapes != runtime_delta_shapes:
            raise ValueError("actor initialization policy delta adapter architecture changed")
    source_has_refinement = "policy_refinement_delta" in restored.agent
    runtime_has_refinement = "policy_refinement_delta" in agent_template
    if source_has_refinement and not runtime_has_refinement:
        raise ValueError("actor initialization would discard a learned policy refinement adapter")
    if source_has_refinement:
        source_refinement_shapes = [
            np.shape(value)
            for value in jax.tree_util.tree_leaves(restored.agent["policy_refinement_delta"])
        ]
        runtime_refinement_shapes = [
            np.shape(value)
            for value in jax.tree_util.tree_leaves(agent_template["policy_refinement_delta"])
        ]
        if source_refinement_shapes != runtime_refinement_shapes:
            raise ValueError("actor initialization policy refinement adapter architecture changed")
    source_has_correction = "policy_correction" in restored.agent
    runtime_has_correction = "policy_correction" in agent_template
    if source_has_correction and not runtime_has_correction:
        raise ValueError("actor initialization would discard a learned physical correction")
    if source_has_correction:
        source_correction_shapes = [
            np.shape(value) for value in jax.tree_util.tree_leaves(restored.agent["policy_correction"])
        ]
        runtime_correction_shapes = [
            np.shape(value) for value in jax.tree_util.tree_leaves(agent_template["policy_correction"])
        ]
        if source_correction_shapes != runtime_correction_shapes:
            raise ValueError("actor initialization physical correction architecture changed")
    if not isinstance(reset_correction_std, bool):
        raise ValueError("reset_correction_std must be boolean")
    correction_std_reset: dict[str, Any] | None = None
    if reset_correction_std:
        if not source_has_correction or not runtime_has_correction:
            raise ValueError(
                "correction exploration reset requires a learned source and runtime correction"
            )
        if "correction_log_std" not in restored.agent or "correction_log_std" not in agent_template:
            raise ValueError("correction exploration reset is missing log-standard-deviation parameters")
        source_log_std = np.asarray(restored.agent["correction_log_std"], dtype="<f4")
        runtime_log_std = np.asarray(agent_template["correction_log_std"], dtype="<f4")
        if (
            source_log_std.shape != runtime_log_std.shape
            or source_log_std.shape != (len(runtime_log_std),)
            or not np.isfinite(source_log_std).all()
            or not np.isfinite(runtime_log_std).all()
        ):
            raise ValueError("correction exploration reset has incompatible parameters")
        source_std = np.exp(source_log_std.astype(np.float64))
        runtime_std = np.exp(runtime_log_std.astype(np.float64))
        correction_std_reset = {
            "schema_version": "stage3_actor_correction_std_reset_v1",
            "mode": "runtime_configured_initial_std",
            "source_log_std_sha256": hashlib.sha256(source_log_std.tobytes()).hexdigest(),
            "runtime_log_std_sha256": hashlib.sha256(runtime_log_std.tobytes()).hexdigest(),
            "source_std_min": float(source_std.min()),
            "source_std_max": float(source_std.max()),
            "runtime_std_min": float(runtime_std.min()),
            "runtime_std_max": float(runtime_std.max()),
            "action_count": int(runtime_log_std.size),
        }
        restored = RestoredTrainingCheckpoint(
            agent={
                **restored.agent,
                "correction_log_std": agent_template["correction_log_std"],
            },
            optimizer_state=restored.optimizer_state,
            obs_rms=restored.obs_rms,
            rng_key=restored.rng_key,
            env_rng_key=restored.env_rng_key,
            metadata=restored.metadata,
        )
    payload_path, _ = resolve_training_checkpoint(Path(path))
    binding: dict[str, Any] = {
        "schema_version": (
            "stage3_actor_only_correction_exploration_repair_initialization_v1"
            if correction_std_reset is not None
            else (
                "stage3_actor_only_scheduled_authority_initialization_v1"
                if authority_transfer is not None
                else (
                    "stage3_actor_only_teacher_timing_initialization_v1"
                    if timing_transfer is not None
                    else "stage3_actor_only_reward_repair_initialization_v1"
                )
            )
        ),
        "source_checkpoint": str(payload_path),
        "source_payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "source_control_hash": source_control.get("control_hash"),
        "runtime_control_hash": runtime_control.get("control_hash"),
        "source_iteration": int(metadata.get("iteration", 0)),
        "source_env_steps": int(metadata.get("env_steps", 0)),
        "transferred": [
            "policy_actor",
            *( ["policy_delta_adapter"] if source_has_delta else [] ),
            *( ["policy_refinement_delta_adapter"] if source_has_refinement else [] ),
            *(
                [
                    "selected_physical_correction_mean"
                    if correction_std_reset is not None
                    else "selected_physical_correction"
                ]
                if source_has_correction
                else []
            ),
            "observation_normalizer",
        ],
        "reset": [
            "value_network",
            "optimizer_state",
            "rng",
            "curriculum_state",
            *(
                ["selected_physical_correction_exploration_std"]
                if correction_std_reset is not None
                else []
            ),
        ],
    }
    if runtime_has_delta and not source_has_delta:
        binding["initialized_zero"] = ["policy_delta_adapter"]
    if runtime_has_refinement and not source_has_refinement:
        binding.setdefault("initialized_zero", []).append("policy_refinement_delta_adapter")
    if runtime_has_correction and not source_has_correction:
        binding.setdefault("initialized_zero", []).append("selected_physical_correction")
    if authority_transfer is not None:
        binding["residual_authority_transfer"] = authority_transfer
    if timing_transfer is not None:
        binding["base_timing_transfer"] = timing_transfer
    if impact_semantics_transfer is not None:
        binding["impact_semantics_transfer"] = impact_semantics_transfer
    if feed_transfer is not None:
        binding["feed_bank_transfer"] = feed_transfer
    if action_prior_lineage is not None:
        binding["action_prior_lineage"] = action_prior_lineage
    if correction_std_reset is not None:
        binding["correction_std_reset"] = correction_std_reset
    binding["binding_sha256"] = _stable_json_hash(binding)
    return restored, binding


def _stable_json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _producer_feed_manifest(
    runtime_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the immutable producer identity from MJX's consumer ordering.

    ``feed-check`` signs the persisted feed artifact.  ``IncomingHitMjxEnv``
    then adds the exact order consumed by the curriculum.  Training must bind
    that complete runtime manifest without pretending the producer report
    already contained a consumer-only field.
    """

    if not isinstance(runtime_manifest, dict):
        raise ValueError("Stage-3 training feed manifest must be a JSON object")
    producer = dict(runtime_manifest)
    consumer_order = producer.pop("consumer_order", None)
    if not isinstance(consumer_order, dict):
        raise ValueError("Stage-3 runtime feed manifest has no consumer_order")
    if consumer_order.get("schema_version") != "incoming_hit_curriculum_feed_order_v1":
        raise ValueError("Stage-3 runtime feed consumer_order schema is incompatible")
    if consumer_order.get("mode") not in {
        "difficulty_sorted",
        "stored",
        "explicit_fingerprint_order",
    }:
        raise ValueError("Stage-3 runtime feed consumer_order mode is incompatible")
    producer_fingerprints = producer.get("sample_fingerprints")
    consumer_fingerprints = consumer_order.get("sample_fingerprints")
    if not isinstance(producer_fingerprints, list) or not isinstance(consumer_fingerprints, list):
        raise ValueError("Stage-3 feed manifests must contain sample_fingerprints")
    if sorted(str(value) for value in producer_fingerprints) != sorted(str(value) for value in consumer_fingerprints):
        raise ValueError("Stage-3 runtime consumer_order changed the feed-bank identity")
    return producer, consumer_order


def _actor_initialization_feed_producer(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return producer identity while accepting pre-ordering checkpoints.

    Old Stage-3 checkpoints were written before ``consumer_order`` existed.
    Actor-only warm starts may compare those immutable producer manifests, but
    production runtime/resume prerequisite validation continues to call the
    strict ``_producer_feed_manifest`` path above.
    """

    if not isinstance(manifest, dict):
        raise ValueError("actor initialization feed manifest must be a JSON object")
    if "consumer_order" not in manifest:
        return dict(manifest)
    producer, _consumer_order = _producer_feed_manifest(manifest)
    return producer


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _same_number(left: Any, right: Any, *, label: str) -> None:
    actual = _finite_float(left, label=label)
    expected = _finite_float(right, label=f"expected {label}")
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Stage-3 {label} is inconsistent with report evidence")


def _validate_preflight_predicates(preflight: dict[str, Any], *, paths: Any) -> None:
    required_true = (
        "scene_exists",
        "keyframe_found",
        "attachment_contract_passed",
        "configuration_contract_passed",
    )
    if any(preflight.get(name) is not True for name in required_true):
        raise ValueError("Stage-3 preflight has a failed scene/exact-child predicate")
    if preflight.get("missing_sites") != []:
        raise ValueError("Stage-3 preflight is missing required sites")
    if int(preflight.get("actuator_count", -1)) != 354:
        raise ValueError("Stage-3 preflight does not prove 354 actuators")
    if int(preflight.get("finger_joint_count", -1)) != 0 or preflight.get("finger_joint_names") != []:
        raise ValueError("Stage-3 preflight still contains finger joints")
    if int(preflight.get("finger_actuator_count", -1)) != 0 or preflight.get("finger_actuator_names") != []:
        raise ValueError("Stage-3 preflight still contains finger actuators")

    router = preflight.get("action_router")
    if not isinstance(router, dict) or router.get("schema_version") != "stage3_action_router_v2":
        raise ValueError("Stage-3 preflight action router schema is incompatible")
    if router.get("fixture_mode") != "rigid_tool_fingerless":
        raise ValueError("Stage-3 preflight does not prove the rigid-tool fixture")
    if router.get("partition_sizes") != [354, 0, 0]:
        raise ValueError("Stage-3 preflight does not prove the 354+0+0 router")
    all_names = router.get("all_actuator_names")
    owned_groups = (
        router.get("body_actuator_names"),
        router.get("right_grip_actuator_names"),
        router.get("left_neutral_actuator_names"),
    )
    expected_lengths = (354, 0, 0)
    if not isinstance(all_names, list) or len(all_names) != 354:
        raise ValueError("Stage-3 preflight action router has no full actuator list")
    if any(
        not isinstance(group, list) or len(group) != expected
        for group, expected in zip(owned_groups, expected_lengths, strict=True)
    ):
        raise ValueError("Stage-3 preflight action router partition names are incomplete")
    all_strings = [str(value) for value in all_names]
    owned_strings = [[str(value) for value in group] for group in owned_groups]
    if len(set(all_strings)) != 354:
        raise ValueError("Stage-3 preflight action router has duplicate actuators")
    flattened = [value for group in owned_strings for value in group]
    if len(set(flattened)) != 354 or set(flattened) != set(all_strings):
        raise ValueError("Stage-3 preflight action router ownership is not exhaustive")
    router_identity = {
        "schema_version": "stage3_action_router_v2",
        "fixture_mode": "rigid_tool_fingerless",
        "all": all_strings,
        "body": owned_strings[0],
        "right_grip": owned_strings[1],
        "left_neutral": owned_strings[2],
    }
    if router.get("schema_hash") != _stable_json_hash(router_identity):
        raise ValueError("Stage-3 preflight action router hash is invalid")

    attachment = preflight.get("racket_attachment")
    if not isinstance(attachment, dict) or attachment.get("schema_version") != "stage3_exact_child_attachment_v2":
        raise ValueError("Stage-3 preflight attachment schema is incompatible")
    recorded_attachment_hash = attachment.get("attachment_hash")
    attachment_unbound = dict(attachment)
    attachment_unbound.pop("attachment_hash", None)
    if recorded_attachment_hash != _stable_json_hash(attachment_unbound):
        raise ValueError("Stage-3 preflight attachment hash is invalid")
    checks = attachment.get("contract_checks")
    tolerances = attachment.get("contract_tolerances")
    if (
        attachment.get("attachment_mode") != "exact_child"
        or attachment.get("contract_passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(tolerances, dict)
        or attachment.get("parent_body_matches") is not True
        or int(attachment.get("racket_joint_count", -1)) != 0
        or int(attachment.get("racket_equality_constraint_count", -1)) != 0
        or attachment.get("hand_racket_contact_enabled") is not False
        or attachment.get("human_racket_explicit_contact_pairs") != 0
        or attachment.get("human_racket_mask_compatible_geom_pairs") != 0
        or attachment.get("racket_shuttle_contact_enabled") is not True
    ):
        raise ValueError("Stage-3 preflight does not prove exact-child/contact ownership")
    for metric, limit in tolerances.items():
        if _finite_float(attachment.get(metric), label=f"preflight {metric}") > _finite_float(
            limit, label=f"preflight {metric} tolerance"
        ):
            raise ValueError(f"Stage-3 preflight attachment exceeds {metric} tolerance")

    root_pos = preflight.get("root_pos")
    expected_root = preflight.get("expected_root_xy")
    if not isinstance(root_pos, list) or len(root_pos) < 2:
        raise ValueError("Stage-3 preflight has no root position")
    if not isinstance(expected_root, list) or len(expected_root) != 2:
        raise ValueError("Stage-3 preflight has no expected root position")
    configured_root = getattr(paths, "human_root_xy", None)
    if configured_root is not None and list(configured_root) != expected_root:
        raise ValueError("Stage-3 preflight expected root differs from the spec")
    for axis in range(2):
        if (
            abs(
                _finite_float(root_pos[axis], label=f"preflight root axis {axis}")
                - _finite_float(expected_root[axis], label=f"expected root axis {axis}")
            )
            >= 1e-6
        ):
            raise ValueError("Stage-3 preflight root placement failed")


def _validate_base_only_predicates(base_only: dict[str, Any]) -> None:
    if base_only.get("runner_stage") != "base-only-check":
        raise ValueError("Stage-3 base-only report runner stage is incompatible")
    residual_mode = base_only.get("control_mode") == "frozen_base_residual"
    expected_task_action = "all_zero_full_action_residual" if residual_mode else "all_zero_raw_latent"
    if base_only.get("task_action") != expected_task_action:
        raise ValueError("Stage-3 base-only report did not use zero task action")
    if base_only.get("shuttle_mode") != "parked_out_of_scene":
        raise ValueError("Stage-3 base-only report used an active shuttle")
    if residual_mode:
        if base_only.get("lambda_lab") is not None:
            raise ValueError("Stage-3 residual base-only report has a LAB lambda")
    else:
        _same_number(base_only.get("lambda_lab"), 0.0, label="base-only lambda")
    episodes = base_only.get("episodes")
    thresholds = base_only.get("thresholds")
    gates = base_only.get("gates")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Stage-3 base-only report has no rollout evidence")
    if not isinstance(thresholds, dict) or not isinstance(gates, dict):
        raise ValueError("Stage-3 base-only report has no thresholds/gates")
    required_steps = int(base_only.get("required_steps", 0))
    if required_steps <= 0:
        raise ValueError("Stage-3 base-only report has an invalid rollout length")

    threshold_names = [
        "min_rollout_count",
        "min_completion_rate",
        "min_finite_rate",
        "min_no_fall_rate",
        "min_root_height_m",
        "max_body_action_saturation_fraction",
        "max_full_action_saturation_fraction",
        "max_normalized_control_energy",
        "min_control_finite",
        "max_attachment_translation_drift_m",
        "max_attachment_rotation_drift_rad",
    ]
    if not residual_mode:
        threshold_names.append("max_lab_state_ood_fraction")
    values = {
        name: _finite_float(thresholds.get(name), label=f"base-only threshold {name}") for name in threshold_names
    }

    def episode_numbers(name: str) -> list[float]:
        return [
            _finite_float(episode.get(name), label=f"base-only episode {name}")
            for episode in episodes
            if isinstance(episode, dict)
        ]

    if any(not isinstance(episode, dict) for episode in episodes):
        raise ValueError("Stage-3 base-only rollout evidence is malformed")
    completion_rate = float(
        np.mean([int(episode.get("completed_steps", -1)) >= required_steps for episode in episodes])
    )
    finite_rate = float(np.mean([episode.get("finite") is True for episode in episodes]))
    no_fall_rate = float(np.mean([episode.get("body_fall") is False for episode in episodes]))
    metrics = {
        "rollout_count": float(len(episodes)),
        "completion_rate": completion_rate,
        "finite_rate": finite_rate,
        "no_fall_rate": no_fall_rate,
        "min_root_height_m": min(episode_numbers("min_root_height_m")),
        "max_body_action_saturation_fraction": max(episode_numbers("max_body_action_saturation_fraction")),
        "max_full_action_saturation_fraction": max(episode_numbers("max_full_action_saturation_fraction")),
        "max_normalized_control_energy": max(episode_numbers("max_normalized_control_energy")),
        "min_control_finite": min(episode_numbers("min_control_finite")),
        "max_attachment_translation_drift_m": max(episode_numbers("max_attachment_translation_drift_m")),
        "max_attachment_rotation_drift_rad": max(episode_numbers("max_attachment_rotation_drift_rad")),
    }
    if not residual_mode:
        metrics["max_lab_state_ood_fraction"] = max(episode_numbers("max_lab_state_ood_fraction"))
    for name, expected in metrics.items():
        _same_number(base_only.get(name), expected, label=f"base-only {name}")
    expected_gates = {
        "rollout_count": len(episodes) >= int(values["min_rollout_count"]),
        "completion_rate": completion_rate >= values["min_completion_rate"],
        "finite_rate": finite_rate >= values["min_finite_rate"],
        "no_fall_rate": no_fall_rate >= values["min_no_fall_rate"],
        "min_root_height_m": metrics["min_root_height_m"] >= values["min_root_height_m"],
        "body_action_saturation": metrics["max_body_action_saturation_fraction"]
        <= values["max_body_action_saturation_fraction"],
        "full_action_saturation": metrics["max_full_action_saturation_fraction"]
        <= values["max_full_action_saturation_fraction"],
        "normalized_control_energy": metrics["max_normalized_control_energy"]
        <= values["max_normalized_control_energy"],
        "control_finite": metrics["min_control_finite"] >= values["min_control_finite"],
        "attachment_translation_drift": metrics["max_attachment_translation_drift_m"]
        <= values["max_attachment_translation_drift_m"],
        "attachment_rotation_drift": metrics["max_attachment_rotation_drift_rad"]
        <= values["max_attachment_rotation_drift_rad"],
    }
    if not residual_mode:
        expected_gates["lab_state_ood_fraction"] = (
            metrics["max_lab_state_ood_fraction"] <= values["max_lab_state_ood_fraction"]
        )
    if gates != expected_gates or not all(expected_gates.values()):
        raise ValueError("Stage-3 base-only report has a failed or inconsistent gate")


def _validate_feed_check_predicates(
    feed_check: dict[str, Any],
    *,
    paths: Any,
    producer_training_manifest: dict[str, Any],
    consumer_order: dict[str, Any],
) -> None:
    manifests: dict[str, dict[str, Any]] = {}
    for label in ("train", "eval"):
        entry = feed_check.get(label)
        if not isinstance(entry, dict):
            raise ValueError(f"Stage-3 feed-check has no {label} evidence")
        manifest = entry.get("manifest")
        fingerprints = manifest.get("sample_fingerprints") if isinstance(manifest, dict) else None
        if not isinstance(manifest, dict) or not isinstance(fingerprints, list):
            raise ValueError(f"Stage-3 feed-check {label} manifest is malformed")
        bank_size = int(entry.get("bank_size", -1))
        expected_size = int(entry.get("expected_bank_size", -2))
        unique_count = len({str(value) for value in fingerprints})
        if bank_size != len(fingerprints) or bank_size != expected_size:
            raise ValueError(f"Stage-3 feed-check {label} count predicate failed")
        configured_size = getattr(
            paths,
            "eval_feed_bank_size" if label == "eval" else "feed_bank_size",
            None,
        )
        if configured_size is not None and expected_size != int(configured_size):
            raise ValueError(f"Stage-3 feed-check {label} count differs from the spec")
        if (
            entry.get("exact_count") is not True
            or entry.get("all_samples_unique") is not True
            or entry.get("all_in_window") is not True
            or entry.get("quality_passed") is not True
            or not isinstance(entry.get("quality"), dict)
            or entry["quality"].get("schema_version") != "incoming_shuttle_feed_quality_v2"
            or entry["quality"].get("passed") is not True
            or int(entry.get("unique_sample_count", -1)) != unique_count
            or unique_count != bank_size
        ):
            raise ValueError(f"Stage-3 feed-check {label} predicate failed")
        manifests[label] = manifest
    if manifests["train"] != producer_training_manifest:
        raise ValueError("Stage-3 training feed changed after feed-check")

    reported_order = feed_check["train"].get("consumer_order")
    if not isinstance(reported_order, dict) or reported_order.get("passed") is not True:
        raise ValueError("Stage-3 feed-check has no passing training consumer order")
    for key in ("schema_version", "mode", "sample_fingerprints"):
        if reported_order.get(key) != consumer_order.get(key):
            raise ValueError(f"Stage-3 feed-check consumer order changed: {key}")
    ordered_fingerprints = consumer_order.get("sample_fingerprints")
    if not isinstance(ordered_fingerprints, list) or not ordered_fingerprints:
        raise ValueError("Stage-3 runtime consumer order has no feed fingerprints")
    direct = dict(getattr(paths, "stage3_direct", {}) or {})
    configured_seeds = direct.get("seed_feed_fingerprints", [])
    if configured_seeds is None:
        configured_seeds = []
    configured_seeds = [str(value) for value in configured_seeds]
    producer_indices = {
        str(value): index
        for index, value in enumerate(manifests["train"]["sample_fingerprints"])
    }
    expected_seed_indices = [producer_indices[value] for value in configured_seeds]
    expected_prefix = ordered_fingerprints[: len(configured_seeds)] == configured_seeds
    content_sha256 = reported_order.get("content_sha256")
    if (
        reported_order.get("seed_feed_fingerprints") != configured_seeds
        or reported_order.get("seed_producer_indices") != expected_seed_indices
        or reported_order.get("seed_prefix_matches") is not expected_prefix
        or reported_order.get("first_fingerprint") != ordered_fingerprints[0]
        or not isinstance(content_sha256, str)
        or len(content_sha256) != 64
    ):
        raise ValueError("Stage-3 feed-check consumer-order evidence is inconsistent")

    train_values = [str(value) for value in manifests["train"]["sample_fingerprints"]]
    eval_values = [str(value) for value in manifests["eval"]["sample_fingerprints"]]
    overlap = sorted(set(train_values) & set(eval_values))
    expected_identity = {
        "bank_paths_distinct": True,
        "train_unique_sample_count": len(set(train_values)),
        "eval_unique_sample_count": len(set(eval_values)),
        "train_duplicate_count": len(train_values) - len(set(train_values)),
        "eval_duplicate_count": len(eval_values) - len(set(eval_values)),
        "train_eval_fingerprint_overlap_count": len(overlap),
        "train_eval_fingerprint_overlap": overlap,
    }
    feed_path = getattr(paths, "feed_bank_path", None)
    eval_feed_path = getattr(paths, "eval_feed_bank_path", None)
    if feed_path is not None and eval_feed_path is not None:
        expected_identity["bank_paths_distinct"] = Path(feed_path).resolve() != Path(eval_feed_path).resolve()
    for name, expected in expected_identity.items():
        if feed_check.get(name) != expected:
            raise ValueError(f"Stage-3 feed-check identity predicate changed: {name}")
    if not expected_identity["bank_paths_distinct"] or overlap:
        raise ValueError("Stage-3 train/eval feed banks overlap")


def validate_stage3_training_prerequisites(
    out_dir: Path,
    *,
    paths: Any,
    latent_checkpoint_dir: Path,
    control_manifest: dict[str, Any],
    training_feed_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require preflight, base-only and feed evidence before production PPO."""

    root = Path(out_dir).resolve()
    report_paths = {
        "preflight": root / "preflight_report.json",
        "base_only": root / "base_only_report.json",
        "feed_check": root / "feed_check_report.json",
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        if not path.is_file():
            raise ValueError(f"Stage-3 training requires {name} report: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Stage-3 {name} report is unreadable: {path}") from exc
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ValueError(f"Stage-3 {name} report did not pass: {path}")
        reports[name] = value

    preflight = reports["preflight"]
    if preflight.get("runner_type") != "incoming_shuttle_hit":
        raise ValueError("Stage-3 preflight report has the wrong runner type")
    for report_key, expected_path in (
        ("spec_path", Path(paths.spec_path)),
        ("scene_xml", Path(paths.scene_xml)),
    ):
        actual = Path(str(preflight.get(report_key, ""))).expanduser()
        if not actual.is_absolute():
            actual = REPO_ROOT / actual
        if actual.resolve() != expected_path.resolve():
            raise ValueError(f"Stage-3 preflight {report_key} changed")
    _validate_preflight_predicates(preflight, paths=paths)
    runtime_router_hash = control_manifest.get("router_schema_hash")
    if runtime_router_hash is not None and preflight["action_router"].get("schema_hash") != runtime_router_hash:
        raise ValueError("Stage-3 preflight router differs from the training runtime")
    runtime_attachment = control_manifest.get("racket_attachment")
    if isinstance(runtime_attachment, dict) and preflight["racket_attachment"].get(
        "attachment_hash"
    ) != runtime_attachment.get("attachment_hash"):
        raise ValueError("Stage-3 preflight attachment differs from the training runtime")

    base_only = reports["base_only"]
    if base_only.get("schema_version") != "stage3_base_only_v1":
        raise ValueError("Stage-3 base-only report schema is incompatible")
    _validate_base_only_predicates(base_only)
    recorded_latent = Path(str(base_only.get("latent_checkpoint", ""))).expanduser()
    if not recorded_latent.is_absolute():
        recorded_latent = REPO_ROOT / recorded_latent
    if recorded_latent.resolve() != Path(latent_checkpoint_dir).resolve():
        raise ValueError("Stage-3 base-only report used a different latent checkpoint")
    base_control = base_only.get("control_manifest")
    impact_recovery_v2 = (
        dict(control_manifest.get("environment_abi", {}) or {}).get("task_profile") == "impact_recovery_v2"
    )
    if not isinstance(base_control, dict):
        raise ValueError("Stage-3 base-only control manifest is missing")
    if impact_recovery_v2 and base_control.get("policy_abi_hash") != control_manifest.get("policy_abi_hash"):
        raise ValueError("Stage-3 base-only policy ABI changed")
    if not impact_recovery_v2 and base_control.get("control_hash") != control_manifest.get("control_hash"):
        raise ValueError("Stage-3 base-only control contract changed")

    producer_training_manifest, consumer_order = _producer_feed_manifest(training_feed_manifest)
    if "curriculum" in control_manifest:
        expected_mode = control_manifest.get(
            "curriculum_feed_order",
            "difficulty_sorted" if control_manifest.get("curriculum") is not None else "stored",
        )
        if consumer_order.get("mode") != expected_mode:
            raise ValueError("Stage-3 runtime feed order differs from the control curriculum")
    feed_check = reports["feed_check"]
    if feed_check.get("runner_stage") != "feed-check":
        raise ValueError("Stage-3 feed-check report schema is incompatible")
    _validate_feed_check_predicates(
        feed_check,
        paths=paths,
        producer_training_manifest=producer_training_manifest,
        consumer_order=consumer_order,
    )

    binding: dict[str, Any] = {
        "schema_version": "stage3_training_prerequisite_binding_v1",
        "action_family": (
            "latent_direct_ablation" if control_manifest.get("decoder_type") == "direct" else "fixed_synergy"
        ),
        "policy_action_size": control_manifest.get("task_action_dim"),
        "preflight_report_path": str(report_paths["preflight"]),
        "preflight_report_sha256": hashlib.sha256(report_paths["preflight"].read_bytes()).hexdigest(),
        "base_only_report_path": str(report_paths["base_only"]),
        "base_only_report_sha256": hashlib.sha256(report_paths["base_only"].read_bytes()).hexdigest(),
        "feed_check_report_path": str(report_paths["feed_check"]),
        "feed_check_report_sha256": hashlib.sha256(report_paths["feed_check"].read_bytes()).hexdigest(),
        "latent_checkpoint_fingerprint": control_manifest.get("latent_checkpoint_fingerprint"),
        "control_hash": control_manifest.get("control_hash"),
        "training_feed_producer_manifest_sha256": _stable_json_hash(producer_training_manifest),
        "training_feed_manifest_sha256": _stable_json_hash(training_feed_manifest),
        "verified": True,
    }
    required_binding_keys = ["latent_checkpoint_fingerprint", "control_hash"]
    if impact_recovery_v2:
        binding["policy_abi_hash"] = control_manifest.get("policy_abi_hash")
        required_binding_keys.append("policy_abi_hash")
    for key in required_binding_keys:
        if not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"Stage-3 prerequisite binding has no {key}")
    binding["binding_sha256"] = _stable_json_hash(binding)
    return binding


def validate_stage3_residual_training_prerequisites(
    out_dir: Path,
    *,
    paths: Any,
    base_policy_artifact: Path,
    control_manifest: dict[str, Any],
    training_feed_manifest: dict[str, Any],
    policy_update_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind evidence for a frozen body prior plus full-action residual PPO.

    This legacy-v1 action family still needs the same fail-closed production
    gates as LAB: the residual policy is allowed to change racket direction,
    but it may not silently change the body prior, attachment, feed bank or
    preflight scene underneath an optimizer/checkpoint.
    """

    root = Path(out_dir).resolve()
    report_paths = {
        "preflight": root / "preflight_report.json",
        "base_only": root / "base_only_report.json",
        "feed_check": root / "feed_check_report.json",
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        if not path.is_file():
            raise ValueError(f"Stage-3 frozen-base residual training requires {name} report: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Stage-3 frozen-base residual {name} report is unreadable: {path}") from exc
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ValueError(f"Stage-3 frozen-base residual {name} report did not pass: {path}")
        reports[name] = value

    preflight = reports["preflight"]
    if preflight.get("runner_type") != "incoming_shuttle_hit":
        raise ValueError("Stage-3 residual preflight report has the wrong runner type")
    for report_key, expected_path in (
        ("spec_path", Path(paths.spec_path)),
        ("scene_xml", Path(paths.scene_xml)),
    ):
        actual = Path(str(preflight.get(report_key, ""))).expanduser()
        if not actual.is_absolute():
            actual = REPO_ROOT / actual
        if actual.resolve() != expected_path.resolve():
            raise ValueError(f"Stage-3 residual preflight {report_key} changed")
    _validate_preflight_predicates(preflight, paths=paths)
    policy_update_contract_sha256 = None
    if policy_update_contract is not None:
        recorded_contract = preflight.get("policy_update_contract")
        if recorded_contract != policy_update_contract:
            raise ValueError("Stage-3 residual policy update contract changed after preflight")
        unbound_contract = dict(policy_update_contract)
        policy_update_contract_sha256 = unbound_contract.pop("contract_sha256", None)
        if policy_update_contract_sha256 != _stable_json_hash(unbound_contract):
            raise ValueError("Stage-3 residual policy update contract hash is invalid")
    runtime_router_hash = control_manifest.get("router_schema_hash")
    if runtime_router_hash is not None and preflight["action_router"].get("schema_hash") != runtime_router_hash:
        raise ValueError("Stage-3 residual preflight router differs from runtime")
    runtime_attachment = control_manifest.get("racket_attachment")
    if isinstance(runtime_attachment, dict) and preflight["racket_attachment"].get(
        "attachment_hash"
    ) != runtime_attachment.get("attachment_hash"):
        raise ValueError("Stage-3 residual preflight attachment differs from runtime")

    frozen_binding = control_manifest.get("frozen_base_residual")
    if (
        control_manifest.get("schema_version") != "incoming_hit_direct_action_v1"
        or not isinstance(frozen_binding, dict)
        or frozen_binding.get("schema_version") != "incoming_hit_frozen_base_residual_v1"
    ):
        raise ValueError("Stage-3 residual training requires the frozen-base residual control ABI")
    environment_abi = dict(control_manifest.get("environment_abi", {}) or {})
    policy_action_size = int(environment_abi.get("full_action_size", -1))
    if policy_action_size != int(frozen_binding.get("actor_action_size", -2)):
        raise ValueError("Stage-3 residual policy/base prior action dimensions are incompatible")

    base_only = reports["base_only"]
    if (
        base_only.get("schema_version") != "stage3_base_only_v1"
        or base_only.get("control_mode") != "frozen_base_residual"
    ):
        raise ValueError("Stage-3 residual base-only report schema is incompatible")
    _validate_base_only_predicates(base_only)
    expected_artifact = Path(base_policy_artifact).expanduser().resolve()
    recorded_artifact = Path(str(base_only.get("base_policy_artifact", ""))).expanduser()
    if not recorded_artifact.is_absolute():
        recorded_artifact = REPO_ROOT / recorded_artifact
    if recorded_artifact.resolve() != expected_artifact:
        raise ValueError("Stage-3 residual base-only report used a different base policy")
    if not math.isclose(
        float(base_only.get("residual_scale", float("nan"))),
        float(frozen_binding.get("residual_scale", float("nan"))),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Stage-3 residual base-only residual scale changed")
    base_control = base_only.get("control_manifest")
    if not isinstance(base_control, dict):
        raise ValueError("Stage-3 residual base-only control manifest is missing")
    # The CPU base-only rollout contains one parked shuttle and therefore has
    # neither a feed-order operation nor production seed fingerprints.  MJX's
    # actual stored order and seed identity are validated below against the
    # producer manifest.  Compare the common physical ABI after removing those
    # non-applicable fields and rebuilding the hash they deterministically
    # change.
    comparable_base_control = dict(base_control)
    comparable_base_control.pop("curriculum_feed_order", None)
    comparable_base_control.pop("seed_feed_fingerprints", None)
    comparable_base_control.pop("control_hash", None)
    comparable_base_control["control_hash"] = _stable_json_hash(comparable_base_control)
    comparable_runtime_control = dict(control_manifest)
    comparable_runtime_control.pop("curriculum_feed_order", None)
    comparable_runtime_control.pop("seed_feed_fingerprints", None)
    comparable_runtime_control.pop("control_hash", None)
    comparable_runtime_control["control_hash"] = _stable_json_hash(comparable_runtime_control)
    differing_control_fields = sorted(
        key
        for key in set(comparable_base_control) | set(comparable_runtime_control)
        if comparable_base_control.get(key) != comparable_runtime_control.get(key)
    )
    if differing_control_fields:
        raise ValueError("Stage-3 residual base-only control contract changed: " + ", ".join(differing_control_fields))

    producer_training_manifest, consumer_order = _producer_feed_manifest(training_feed_manifest)
    expected_feed_order = control_manifest.get(
        "curriculum_feed_order",
        "difficulty_sorted" if control_manifest.get("curriculum") is not None else "stored",
    )
    if consumer_order.get("mode") != expected_feed_order:
        raise ValueError("Stage-3 residual runtime feed order differs from curriculum")
    feed_check = reports["feed_check"]
    if feed_check.get("runner_stage") != "feed-check":
        raise ValueError("Stage-3 residual feed-check report schema is incompatible")
    _validate_feed_check_predicates(
        feed_check,
        paths=paths,
        producer_training_manifest=producer_training_manifest,
        consumer_order=consumer_order,
    )

    binding: dict[str, Any] = {
        "schema_version": "stage3_frozen_base_residual_prerequisite_binding_v1",
        "action_family": "frozen_base_residual",
        "policy_action_size": policy_action_size,
        "preflight_report_path": str(report_paths["preflight"]),
        "preflight_report_sha256": hashlib.sha256(report_paths["preflight"].read_bytes()).hexdigest(),
        "base_only_report_path": str(report_paths["base_only"]),
        "base_only_report_sha256": hashlib.sha256(report_paths["base_only"].read_bytes()).hexdigest(),
        "feed_check_report_path": str(report_paths["feed_check"]),
        "feed_check_report_sha256": hashlib.sha256(report_paths["feed_check"].read_bytes()).hexdigest(),
        "base_policy_artifact_path": str(expected_artifact),
        "base_policy_artifact_content_sha256": frozen_binding.get("artifact_content_sha256"),
        "frozen_base_binding_sha256": frozen_binding.get("binding_sha256"),
        "latent_checkpoint_fingerprint": None,
        "control_hash": control_manifest.get("control_hash"),
        "policy_update_contract_sha256": policy_update_contract_sha256,
        "training_feed_producer_manifest_sha256": _stable_json_hash(producer_training_manifest),
        "training_feed_manifest_sha256": _stable_json_hash(training_feed_manifest),
        "verified": True,
    }
    for key in (
        "base_policy_artifact_content_sha256",
        "frozen_base_binding_sha256",
        "control_hash",
    ):
        if not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"Stage-3 residual prerequisite binding has no {key}")
    binding["binding_sha256"] = _stable_json_hash(binding)
    return binding


def validate_stage3_direct_training_prerequisites(
    out_dir: Path,
    *,
    paths: Any,
    control_manifest: dict[str, Any],
    training_feed_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind production evidence for the pure 354-D Stage-3 policy.

    A direct policy has no latent checkpoint and therefore no meaningful
    ``base-only`` rollout.  It still has to pass the same scene/attachment and
    train-vs-heldout feed checks, and its exact target, control and policy ABI
    are sealed here before PPO can write a checkpoint.
    """

    root = Path(out_dir).resolve()
    report_paths = {
        "preflight": root / "preflight_report.json",
        "feed_check": root / "feed_check_report.json",
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        if not path.is_file():
            raise ValueError(f"Stage-3 direct training requires {name} report: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Stage-3 direct {name} report is unreadable: {path}") from exc
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ValueError(f"Stage-3 direct {name} report did not pass: {path}")
        reports[name] = value

    preflight = reports["preflight"]
    if preflight.get("runner_type") != "incoming_shuttle_hit":
        raise ValueError("Stage-3 direct preflight report has the wrong runner type")
    for report_key, expected_path in (
        ("spec_path", Path(paths.spec_path)),
        ("scene_xml", Path(paths.scene_xml)),
    ):
        actual = Path(str(preflight.get(report_key, ""))).expanduser()
        if not actual.is_absolute():
            actual = REPO_ROOT / actual
        if actual.resolve() != expected_path.resolve():
            raise ValueError(f"Stage-3 direct preflight {report_key} changed")
    _validate_preflight_predicates(preflight, paths=paths)
    runtime_router_hash = control_manifest.get("router_schema_hash")
    if runtime_router_hash is not None and preflight["action_router"].get("schema_hash") != runtime_router_hash:
        raise ValueError("Stage-3 direct preflight router differs from the training runtime")
    runtime_attachment = control_manifest.get("racket_attachment")
    if isinstance(runtime_attachment, dict) and preflight["racket_attachment"].get(
        "attachment_hash"
    ) != runtime_attachment.get("attachment_hash"):
        raise ValueError("Stage-3 direct preflight attachment differs from the training runtime")

    producer_training_manifest, consumer_order = _producer_feed_manifest(training_feed_manifest)
    if consumer_order.get("mode") != "stored":
        raise ValueError("Stage-3 direct runtime feed order must preserve the stored bank order")
    feed_check = reports["feed_check"]
    if feed_check.get("runner_stage") != "feed-check":
        raise ValueError("Stage-3 direct feed-check report schema is incompatible")
    _validate_feed_check_predicates(
        feed_check,
        paths=paths,
        producer_training_manifest=producer_training_manifest,
        consumer_order=consumer_order,
    )

    if control_manifest.get("schema_version") != "incoming_hit_direct_action_impact_recovery_v2":
        raise ValueError("Stage-3 direct training requires the impact/recovery full-action control ABI")
    environment_abi = dict(control_manifest.get("environment_abi", {}) or {})
    if environment_abi.get("task_profile") != "impact_recovery_v2":
        raise ValueError("Stage-3 direct training requires the impact/recovery task profile")
    if int(environment_abi.get("full_action_size", -1)) != 354:
        raise ValueError("Stage-3 direct policy must expose exactly 354 muscle actions")

    spec_path = Path(paths.spec_path).resolve()
    scene_path = Path(paths.scene_xml).resolve()
    if environment_abi.get("scene_sha256") != hashlib.sha256(scene_path.read_bytes()).hexdigest():
        raise ValueError("Stage-3 direct runtime scene differs from the preflight scene")
    target_path = Path(getattr(paths, "target_bank_path", "")).expanduser().resolve()
    if not target_path.is_file():
        raise ValueError("Stage-3 direct training target bank is missing")
    try:
        target = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-3 direct training target bank is unreadable") from exc
    if not isinstance(target, dict):
        raise ValueError("Stage-3 direct training target bank must be a JSON object")
    for key in ("bank_sha256", "source_fingerprint"):
        value = target.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Stage-3 direct training target bank has no valid {key}")
    if environment_abi.get("target_bank_sha256") != target["bank_sha256"]:
        raise ValueError("Stage-3 direct runtime target bank differs from the training target")

    binding: dict[str, Any] = {
        "schema_version": "stage3_direct_training_prerequisite_binding_v1",
        "action_family": "full_354",
        "policy_action_size": 354,
        "preflight_report_path": str(report_paths["preflight"]),
        "preflight_report_sha256": hashlib.sha256(report_paths["preflight"].read_bytes()).hexdigest(),
        "feed_check_report_path": str(report_paths["feed_check"]),
        "feed_check_report_sha256": hashlib.sha256(report_paths["feed_check"].read_bytes()).hexdigest(),
        "spec_path": str(spec_path),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "scene_path": str(scene_path),
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "training_target_path": str(target_path),
        "training_target_file_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "training_target_bank_sha256": target["bank_sha256"],
        "training_target_source_fingerprint": target["source_fingerprint"],
        "latent_checkpoint_fingerprint": None,
        "control_hash": control_manifest.get("control_hash"),
        "policy_abi_hash": control_manifest.get("policy_abi_hash"),
        "training_feed_producer_manifest_sha256": _stable_json_hash(producer_training_manifest),
        "training_feed_manifest_sha256": _stable_json_hash(training_feed_manifest),
        "verified": True,
    }
    for key in ("control_hash", "policy_abi_hash"):
        if not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"Stage-3 direct prerequisite binding has no {key}")
    binding["binding_sha256"] = _stable_json_hash(binding)
    return binding


def _read_version_completion(path: Path) -> dict[str, Any]:
    marker = path / "_COMPLETE.json"
    if not marker.is_file():
        raise FileNotFoundError(f"versioned checkpoint completion marker is missing: {marker}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != VERSIONED_CHECKPOINT_SCHEMA:
        raise ValueError("versioned checkpoint completion marker is incompatible")
    recorded = value.get("binding_sha256")
    unbound = dict(value)
    unbound.pop("binding_sha256", None)
    if recorded != _stable_json_hash(unbound):
        raise ValueError("versioned checkpoint completion binding hash mismatch")
    return value


def save_checkpoint(path: Path, agent, obs_rms: ObsRms, meta: dict[str, Any]) -> None:
    """Legacy inference-only checkpoint writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat, _ = jax.tree_util.tree_flatten(agent)
    payload = {f"param_{i}": np.asarray(p) for i, p in enumerate(flat)}
    payload["obs_mean"] = np.asarray(obs_rms.mean)
    payload["obs_var"] = np.asarray(obs_rms.var)
    payload["obs_count"] = np.asarray(obs_rms.count)
    np.savez(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, agent_template):
    with np.load(path) as payload:
        flat, treedef = jax.tree_util.tree_flatten(agent_template)
        flat = [jnp.asarray(payload[f"param_{i}"]) for i in range(len(flat))]
        agent = jax.tree_util.tree_unflatten(treedef, flat)
        obs_rms = ObsRms(
            jnp.asarray(payload["obs_mean"]),
            jnp.asarray(payload["obs_var"]),
            jnp.asarray(payload["obs_count"]),
        )
    return agent, obs_rms


def _init_stage3_wandb(
    out_dir: Path,
    *,
    cfg: TrainConfig,
    env: IncomingHitMjxEnv,
    resume_from: Path | None,
    initialization_binding: dict[str, Any] | None,
):
    """Start an optional, resumable W&B run for production Stage-3 jobs.

    The integration is deliberately opt-in so unit tests and offline runs do
    not acquire a network dependency.  Production launchers enable it with
    ``MUSCLEMIMIC_STAGE3_WANDB_PROJECT``.  The W&B run id is persisted next to
    the checkpoints, which keeps metrics on one run after process restarts.
    """

    project = os.environ.get("MUSCLEMIMIC_STAGE3_WANDB_PROJECT", "").strip()
    if not project:
        return None

    import wandb

    run_id_path = out_dir / "wandb_run_id.txt"
    configured_run_id = os.environ.get("MUSCLEMIMIC_STAGE3_WANDB_RUN_ID", "").strip()
    persisted_run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.is_file() else ""
    if configured_run_id and persisted_run_id and configured_run_id != persisted_run_id:
        raise ValueError(
            "configured Stage-3 W&B run id does not match the run already bound "
            f"to this output directory: configured={configured_run_id!r}, "
            f"persisted={persisted_run_id!r}"
        )
    run_id = configured_run_id or persisted_run_id or None
    wandb_dir = out_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    entity = os.environ.get("MUSCLEMIMIC_STAGE3_WANDB_ENTITY", "").strip() or None
    name = os.environ.get("MUSCLEMIMIC_STAGE3_WANDB_NAME", "").strip() or out_dir.name
    mode = os.environ.get("MUSCLEMIMIC_STAGE3_WANDB_MODE", "online").strip() or "online"
    control_manifest = dict(getattr(env, "control_manifest", {}) or {})
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(wandb_dir),
        config={
            "trainer": "incoming_hit_mjx_ppo_v3",
            "ppo": cfg._asdict(),
            "observation_size": int(env.observation_size),
            "action_size": int(env.action_size),
            "control_hash": getattr(env, "control_hash", None),
            "control_manifest": control_manifest,
            "policy_update_contract": getattr(env, "policy_update_contract", None),
            "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
            "actor_initialization": initialization_binding,
        },
    )
    run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
    (out_dir / "wandb_run.json").write_text(
        json.dumps(
            {
                "schema_version": "incoming_hit_wandb_binding_v1",
                "project": project,
                "entity": entity,
                "name": name,
                "run_id": run.id,
                "url": run.url,
                "mode": mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def train(
    env: IncomingHitMjxEnv,
    cfg: TrainConfig,
    out_dir: Path,
    *,
    log_every: int = 1,
    checkpoint_every: int = 10,
    resume_from: Path | None = None,
    initialize_policy_from: Path | None = None,
    teacher_dataset_path: Path | None = None,
    exploration_prior_dataset_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_per_iter = cfg.num_envs * cfg.rollout_steps
    target_iters, executed_step_target, unused_step_budget = _training_iteration_budget(
        total_env_steps=cfg.total_env_steps,
        steps_per_iteration=steps_per_iter,
        fresh_quality_teacher_bc=bool(
            teacher_dataset_path is not None
            and exploration_prior_dataset_path is None
            and resume_from is None
            and initialize_policy_from is not None
        ),
    )
    if not isinstance(cfg.reset_correction_std_on_actor_initialization, bool):
        raise ValueError(
            "reset_correction_std_on_actor_initialization must be boolean"
        )
    if (
        cfg.reset_correction_std_on_actor_initialization
        and cfg.policy_update_mode not in PHYSICAL_CORRECTION_POLICY_MODES
    ):
        raise ValueError(
            "correction exploration reset requires selected_physical_correction"
        )
    if (
        cfg.reset_correction_std_on_actor_initialization
        and initialize_policy_from is None
        and resume_from is None
    ):
        raise ValueError(
            "correction exploration reset requires actor initialization"
        )
    runtime_feed_manifest = getattr(env, "feed_bank_manifest", None)
    feed_manifest_required = bool(
        getattr(env, "expects_raw_latent", False)
        or getattr(env, "base_policy_artifact", None) is not None
        or getattr(env, "task_profile", "legacy_v1") == "impact_recovery_v2"
    )
    if feed_manifest_required:
        # There is no checkpoint yet, so validate only the runtime half here.
        # Exact feed identity is required for every production Stage-3 control
        # stack, including frozen-base residual training.
        if not isinstance(runtime_feed_manifest, dict):
            raise ValueError("Stage-3 training requires a verified feed-bank manifest")
    if (
        bool(getattr(env, "expects_raw_latent", False))
        or getattr(env, "base_policy_artifact", None) is not None
        or getattr(env, "task_profile", "legacy_v1") == "impact_recovery_v2"
    ):
        prerequisite_binding = getattr(env, "training_prerequisite_binding", None)
        if not isinstance(prerequisite_binding, dict) or prerequisite_binding.get("verified") is not True:
            raise ValueError("Stage-3 production training requires verified prerequisite evidence")
        recorded = prerequisite_binding.get("binding_sha256")
        unbound = dict(prerequisite_binding)
        unbound.pop("binding_sha256", None)
        if recorded != _stable_json_hash(unbound):
            raise ValueError("Stage-3 prerequisite binding hash mismatch")
    teacher_dataset: QualityTeacherDataset | None = None
    teacher_pretrain_report: dict[str, Any] | None = None
    if (
        teacher_dataset_path is not None
        and exploration_prior_dataset_path is not None
    ):
        raise ValueError(
            "quality teacher and CPU-certified exploration prior are mutually exclusive"
        )
    action_prior_dataset_path = (
        teacher_dataset_path
        if teacher_dataset_path is not None
        else exploration_prior_dataset_path
    )
    using_exploration_prior = exploration_prior_dataset_path is not None
    inherited_action_prior_binding: dict[str, Any] | None = None
    if action_prior_dataset_path is not None:
        expected_source_sha256 = None
        source_base_phase_advance_s = None
        if initialize_policy_from is not None:
            source_payload, _source_metadata = resolve_training_checkpoint(
                Path(initialize_policy_from)
            )
            expected_source_sha256 = hashlib.sha256(source_payload.read_bytes()).hexdigest()
            source_metadata = load_training_checkpoint_metadata(
                Path(initialize_policy_from)
            )
            inherited_action_prior_binding = _checkpoint_action_prior_binding(
                source_metadata
            )
            source_frozen = dict(
                dict(source_metadata.get("control_manifest", {}) or {}).get(
                    "frozen_base_residual", {}
                )
                or {}
            )
            if inherited_action_prior_binding is not None:
                expected_source_sha256 = str(
                    inherited_action_prior_binding.get(
                        "source_checkpoint_sha256", ""
                    )
                )
                inherited_timing = dict(
                    inherited_action_prior_binding.get(
                        "base_timing_transfer", {}
                    )
                    or {}
                )
                if inherited_timing:
                    source_base_phase_advance_s = float(
                        inherited_timing.get(
                            "source_phase_advance_s", math.nan
                        )
                    )
            elif source_frozen:
                source_base_phase_advance_s = float(
                    source_frozen.get("phase_advance_s", math.nan)
                )
        elif resume_from is not None:
            teacher_report_path = out_dir / "teacher_bc_pretrain_report.json"
            if not teacher_report_path.is_file():
                raise ValueError(
                    "resume teacher-BC run is missing its local pretrain binding"
                )
            resume_metadata = load_training_checkpoint_metadata(Path(resume_from))
            (
                inherited_action_prior_binding,
                expected_source_sha256,
                source_base_phase_advance_s,
            ) = _resume_action_prior_source(resume_metadata)
        elif resume_from is None:
            raise ValueError("a fresh action-prior run requires initialize_policy_from")
        teacher_dataset = load_quality_teacher_dataset(
            action_prior_dataset_path,
            selected_action_indices=tuple(cfg.policy_trainable_action_indices),
            correction_physical_scales=tuple(cfg.correction_physical_scales),
            source_checkpoint_sha256=expected_source_sha256,
            source_base_phase_advance_s=source_base_phase_advance_s,
            allow_cpu_certified_exploration_prior=using_exploration_prior,
        )
        if (
            inherited_action_prior_binding is not None
            and teacher_dataset.binding != inherited_action_prior_binding
        ):
            raise ValueError(
                "actor initialization action-prior dataset differs from the "
                "source checkpoint lineage"
            )
        if cfg.teacher_action_prior_mode == "time_interpolated_frozen_plus_delta":
            cfg = cfg._replace(
                teacher_prior_time_to_intercept_s=tuple(
                    float(value)
                    for value in teacher_dataset.time_to_intercept_s.tolist()
                ),
                teacher_prior_correction_raw=tuple(
                    tuple(float(value) for value in row)
                    for row in teacher_dataset.correction_raw.tolist()
                ),
            )
            # In prior mode the zero-initialized correction MLP is a learned
            # feedback delta, not an approximation of the open-loop teacher.
            # BC therefore anchors that delta at zero on verified states.
            teacher_dataset = QualityTeacherDataset(
                teacher_dataset.observation_normalized,
                np.zeros_like(teacher_dataset.correction_raw),
                teacher_dataset.sample_weight,
                teacher_dataset.time_to_intercept_s,
                teacher_dataset.binding,
            )
        elif cfg.teacher_action_prior_mode != "none":
            raise ValueError(
                "teacher_action_prior_mode must be none or "
                "time_interpolated_frozen_plus_delta"
            )
    elif cfg.teacher_action_prior_mode != "none":
        raise ValueError(
            "teacher action prior requires a sealed quality teacher or an explicit "
            "CPU-certified exploration-prior dataset"
        )
    if teacher_dataset is not None and resume_from is not None:
        teacher_report_path = out_dir / "teacher_bc_pretrain_report.json"
        if not teacher_report_path.is_file():
            raise ValueError("resume teacher-BC run is missing its local pretrain binding")
        teacher_pretrain_report = json.loads(
            teacher_report_path.read_text(encoding="utf-8")
        )
        recorded_teacher_report_sha = teacher_pretrain_report.get("report_sha256")
        teacher_report_unbound = dict(teacher_pretrain_report)
        teacher_report_unbound.pop("report_sha256", None)
        if recorded_teacher_report_sha != _stable_json_hash(teacher_report_unbound):
            raise ValueError("resume teacher-BC pretrain report hash mismatch")
        if teacher_pretrain_report.get("teacher_binding") != teacher_dataset.binding:
            raise ValueError("resume quality-teacher dataset differs from the local run binding")
    key = jax.random.PRNGKey(cfg.seed)
    key, k_agent, k_env = jax.random.split(key, 3)

    mx = env.put_model(cfg.num_envs)
    template = env.make_batched_template(cfg.num_envs)
    reset_fn = env.make_reset_fn(mx, cfg.num_envs)
    env_states = jax.jit(reset_fn)(k_env, template)

    agent = init_agent(
        k_agent,
        env.observation_size,
        env.action_size,
        cfg.hidden,
        cfg.action_std_init,
        policy_delta_hidden=cfg.policy_delta_hidden,
        policy_refinement_delta_hidden=cfg.policy_refinement_delta_hidden,
        policy_correction_hidden=cfg.policy_correction_hidden,
        correction_action_size=len(tuple(cfg.policy_trainable_action_indices)),
        correction_std_init=cfg.correction_std_init,
    )
    agent = apply_policy_exploration_contract(
        agent,
        action_size=env.action_size,
        trainable_action_indices=tuple(cfg.policy_trainable_action_indices),
        frozen_action_std=cfg.frozen_action_std,
    )
    optimizer = build_ppo_optimizer(
        agent,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate,
        actor_learning_rate=cfg.actor_learning_rate,
        critic_learning_rate=cfg.critic_learning_rate,
    )
    opt_state = optimizer.init(agent)
    obs_rms = ObsRms.create(env.observation_size)

    start_iteration = 0
    resumed_env_steps = 0
    curriculum_effective_steps = 0
    curriculum_gate_report: dict[str, Any] = {
        "checked": False,
        "passed": True,
        "phase": "fixed_feed",
    }
    curriculum_gate_window = _CompletedEpisodeGateWindow(
        min_completed_episodes=int(getattr(getattr(env, "curriculum", None), "gate_min_completed_episodes", 512)),
        max_iterations=int(getattr(getattr(env, "curriculum", None), "gate_window_iterations", 16)),
    )
    task_curriculum_enabled = bool(getattr(env, "task_curriculum_max_stage", None) is not None)
    task_stage_index = 0
    task_curriculum_is_complete = not task_curriculum_enabled
    task_gate_report: dict[str, Any] = {
        "checked": False,
        "passed": not task_curriculum_enabled,
        "failures": [],
    }
    initialization_binding: dict[str, Any] | None = None
    if resume_from is not None and initialize_policy_from is not None:
        raise ValueError("resume_from and initialize_policy_from are mutually exclusive")
    if initialize_policy_from is not None:
        initialized, initialization_binding = load_actor_initialization(
            Path(initialize_policy_from),
            agent_template=agent,
            optimizer_state_template=opt_state,
            env=env,
            base_timing_transfer_evidence=(
                None
                if teacher_dataset is None
                else teacher_dataset.binding.get("base_timing_transfer")
            ),
            quality_teacher_binding=(
                None if teacher_dataset is None else teacher_dataset.binding
            ),
            reset_correction_std=(
                cfg.reset_correction_std_on_actor_initialization
            ),
        )
        agent = {**agent, "policy": initialized.agent["policy"]}
        if "policy_delta" in initialized.agent:
            if "policy_delta" not in agent:
                raise ValueError("actor initialization contains an unexpected policy delta adapter")
            agent = {**agent, "policy_delta": initialized.agent["policy_delta"]}
        if "policy_refinement_delta" in initialized.agent:
            if "policy_refinement_delta" not in agent:
                raise ValueError("actor initialization contains an unexpected policy refinement delta adapter")
            agent = {
                **agent,
                "policy_refinement_delta": initialized.agent["policy_refinement_delta"],
            }
        if "policy_correction" in initialized.agent:
            if "policy_correction" not in agent:
                raise ValueError("actor initialization contains an unexpected physical correction")
            agent = {
                **agent,
                "policy_correction": initialized.agent["policy_correction"],
                "correction_log_std": initialized.agent["correction_log_std"],
            }
        obs_rms = initialized.obs_rms
        # The value function and log standard deviation retain their fresh
        # initialization, and optimizer moments are built after actor import.
        opt_state = optimizer.init(agent)
    if resume_from is not None:
        restored = load_training_checkpoint(
            Path(resume_from),
            agent_template=agent,
            optimizer_state_template=opt_state,
        )
        expected_control_hash = getattr(env, "control_hash", None)
        actual_control_hash = restored.metadata.get("control_hash")
        for name, expected, actual in (
            ("obs_size", env.observation_size, restored.metadata.get("obs_size")),
            ("action_size", env.action_size, restored.metadata.get("action_size")),
        ):
            if int(actual) != int(expected):
                raise ValueError(f"resume {name} mismatch: checkpoint={actual}, runtime={expected}")
        if expected_control_hash is not None and actual_control_hash != expected_control_hash:
            raise ValueError("resume Stage-3 control hash mismatch: latent/runtime/router/grip contract changed")
        checkpoint_curriculum = dict(restored.metadata.get("control_manifest", {}) or {}).get("curriculum")
        runtime_curriculum = dict(getattr(env, "control_manifest", {}) or {}).get("curriculum")
        if checkpoint_curriculum != runtime_curriculum:
            raise ValueError("resume Stage-3 curriculum configuration changed")
        runtime_prerequisites = getattr(env, "training_prerequisite_binding", None)
        if restored.metadata.get("training_prerequisite_binding") != runtime_prerequisites:
            if restored.metadata.get("checkpoint_stage") != "post_teacher_bc_pre_ppo":
                raise ValueError("resume Stage-3 prerequisite evidence changed")
            from musclemimic.badminton.stage3_reachability_release import (
                validate_static_ppo_prerequisite_extension,
            )

            validate_static_ppo_prerequisite_extension(
                checkpoint_binding=restored.metadata.get(
                    "training_prerequisite_binding"
                ),
                runtime_binding=runtime_prerequisites,
                checkpoint_payload_sha256=str(
                    restored.metadata.get("training_payload_sha256", "")
                ),
            )
        if restored.metadata.get("policy_update_contract") != getattr(env, "policy_update_contract", None):
            raise ValueError("resume Stage-3 policy update contract changed")
        validate_training_feed_manifest(
            runtime_feed_manifest,
            checkpoint_manifest=restored.metadata.get("training_feed_manifest"),
            required=feed_manifest_required,
        )
        checkpoint_config = dict(restored.metadata.get("config", {}) or {})
        runtime_config = cfg._asdict()
        # ``frozen_action_std`` was added after the original full-network and
        # v18 checkpoints.  Missing and explicit null are the same legacy
        # contract, so those checkpoints remain resumable.
        checkpoint_config.setdefault("frozen_action_std", None)
        checkpoint_config.setdefault("policy_delta_hidden", [])
        checkpoint_config.setdefault("policy_refinement_delta_hidden", [])
        checkpoint_config.setdefault("freeze_trainable_action_std", False)
        checkpoint_config.setdefault("successful_action_imitation_coef", 0.0)
        checkpoint_config.setdefault("actor_learning_rate", None)
        checkpoint_config.setdefault("critic_learning_rate", None)
        checkpoint_config.setdefault("max_abs_log_ratio", 10.0)
        checkpoint_config.setdefault("max_post_update_ratio_guard_fraction", 1.0)
        checkpoint_config.setdefault("max_post_update_kl_estimate", 1.0e9)
        checkpoint_config.setdefault("policy_correction_hidden", [])
        checkpoint_config.setdefault("correction_physical_scales", [])
        checkpoint_config.setdefault("correction_std_init", [])
        checkpoint_config.setdefault("correction_std_min", [])
        checkpoint_config.setdefault("correction_std_max", [])
        checkpoint_config.setdefault(
            "reset_correction_std_on_actor_initialization", False
        )
        checkpoint_config.setdefault("correction_window_open_s", 0.70)
        checkpoint_config.setdefault("correction_window_close_s", -0.10)
        checkpoint_config.setdefault("correction_window_smoothing_s", 0.05)
        checkpoint_config.setdefault("teacher_action_prior_mode", "none")
        checkpoint_config.setdefault("teacher_prior_time_to_intercept_s", [])
        checkpoint_config.setdefault("teacher_prior_correction_raw", [])
        checkpoint_config.setdefault("quality_success_min_outgoing_z_m_s", 0.5)
        checkpoint_config.setdefault("quality_success_min_forward_m_s", 2.0)
        checkpoint_config.setdefault(
            "quality_success_min_predicted_net_clearance_m", -1.0e9
        )
        checkpoint_config.setdefault(
            "quality_success_min_return_direction_signed_score", -1.0
        )
        checkpoint_config.setdefault(
            "quality_success_min_racket_face_forward_alignment", -1.0
        )
        checkpoint_config.setdefault(
            "quality_success_require_episode_no_fall", False
        )
        checkpoint_config.setdefault("quality_imitation_mode", "strict_success")
        checkpoint_config.setdefault("quality_imitation_min_weight", 0.0)
        checkpoint_config.setdefault(
            "quality_imitation_forward_softness_m_s", 1.0
        )
        checkpoint_config.setdefault(
            "quality_imitation_vertical_softness_m_s", 0.75
        )
        checkpoint_config.setdefault(
            "quality_imitation_clearance_softness_m", 0.75
        )
        checkpoint_config.setdefault("quality_imitation_direction_softness", 0.10)
        checkpoint_config.setdefault(
            "quality_imitation_require_episode_no_fall", False
        )
        checkpoint_config.setdefault("teacher_bc_pretrain_steps", 0)
        checkpoint_config.setdefault("teacher_bc_batch_size", 256)
        checkpoint_config.setdefault("teacher_bc_learning_rate", 3.0e-4)
        checkpoint_config.setdefault("teacher_bc_initial_coef", 0.0)
        checkpoint_config.setdefault("teacher_bc_final_coef", 0.0)
        checkpoint_config.setdefault("teacher_bc_decay_steps", 0)
        checkpoint_config.pop("total_env_steps", None)
        runtime_config.pop("total_env_steps", None)
        if json.loads(json.dumps(checkpoint_config)) != json.loads(json.dumps(runtime_config)):
            raise ValueError("resume PPO configuration changed; only total_env_steps may be increased")
        agent = restored.agent
        opt_state = restored.optimizer_state
        obs_rms = restored.obs_rms
        key = restored.rng_key
        if restored.env_rng_key is not None:
            env_states = env_states._replace(key=restored.env_rng_key)
        start_iteration = int(restored.metadata.get("iteration", 0))
        resumed_env_steps = int(restored.metadata.get("env_steps", 0))
        restored_curriculum_state = dict(restored.metadata.get("curriculum_state", {}) or {})
        curriculum_effective_steps = int(restored_curriculum_state.get("effective_steps", resumed_env_steps))
        curriculum_gate_report = dict(restored_curriculum_state.get("last_gate", curriculum_gate_report) or {})
        initialization_binding = restored.metadata.get("actor_initialization")
        restored_gate_window = restored_curriculum_state.get("episode_gate_window")
        if restored_gate_window is not None:
            curriculum_gate_window = _CompletedEpisodeGateWindow.from_state_dict(dict(restored_gate_window))
        restored_task_state = dict(restored.metadata.get("task_curriculum_state", {}) or {})
        if task_curriculum_enabled:
            if not restored_task_state:
                raise ValueError("resume checkpoint is missing Stage-3 v2 task curriculum state")
            task_stage_index = int(restored_task_state.get("stage_index", -1))
            env.task_curriculum_values(resumed_env_steps, stage_index=task_stage_index)
            task_curriculum_is_complete = bool(restored_task_state.get("complete", False))
            restored_max_stage = restored_task_state.get("max_stage")
            if restored_max_stage != env.task_curriculum_max_stage:
                from environment.overall_environment.src.stage3_task_curriculum_v2 import (
                    canonical_stage3_v2_curriculum,
                    stage_by_name,
                )

                stages = canonical_stage3_v2_curriculum()
                if stages.index(stage_by_name(restored_max_stage)) >= stages.index(
                    stage_by_name(env.task_curriculum_max_stage)
                ):
                    raise ValueError("resume may only expand the Stage-3 v2 curriculum max stage")
                task_curriculum_is_complete = False
            task_gate_report = dict(restored_task_state.get("last_gate", task_gate_report) or {})

    if teacher_dataset is not None and resume_from is None:
        agent, teacher_pretrain_report = pretrain_selected_correction_bc(
            agent,
            teacher_dataset,
            steps=int(cfg.teacher_bc_pretrain_steps),
            batch_size=int(cfg.teacher_bc_batch_size),
            learning_rate=float(cfg.teacher_bc_learning_rate),
            seed=int(cfg.seed),
        )
        (out_dir / "teacher_bc_pretrain_report.json").write_text(
            json.dumps(teacher_pretrain_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        opt_state = optimizer.init(agent)

    train_iteration = make_train_iteration(
        env,
        mx,
        cfg,
        optimizer,
        teacher_dataset=teacher_dataset,
    )

    if resumed_env_steps > cfg.total_env_steps:
        raise ValueError(
            f"resume checkpoint already reached {resumed_env_steps} env steps, "
            f"which exceeds target {cfg.total_env_steps}"
        )
    if resume_from is not None:
        expected_resumed_steps = start_iteration * steps_per_iter
        if resumed_env_steps != expected_resumed_steps:
            raise ValueError(
                "resume checkpoint iteration/env-step accounting is inconsistent: "
                f"iteration={start_iteration}, expected={expected_resumed_steps}, "
                f"reported={resumed_env_steps}"
            )
        if start_iteration > target_iters:
            raise ValueError(
                "resume checkpoint exceeds the requested absolute hard cap: "
                f"hard cap: checkpoint_steps={resumed_env_steps}, cap={cfg.total_env_steps}, "
                f"rollout_batch={steps_per_iter}"
            )
    metrics_path = out_dir / "metrics.jsonl"
    reconcile_metrics_history(
        metrics_path,
        checkpoint_iteration=start_iteration,
    )
    wandb_run = _init_stage3_wandb(
        out_dir,
        cfg=cfg,
        env=env,
        resume_from=resume_from,
        initialization_binding=initialization_binding,
    )

    def lab_curriculum_complete() -> bool:
        return bool(
            getattr(env, "curriculum", None) is None or curriculum_effective_steps >= env.curriculum.curriculum_end
        )

    def all_curricula_complete() -> bool:
        return lab_curriculum_complete() and task_curriculum_is_complete

    def checkpoint_metadata(
        *,
        iteration: int,
        env_steps: int,
        checkpoint_stage: str,
    ) -> dict[str, Any]:
        """Build one metadata contract for pre-PPO and iteration checkpoints.

        Teacher BC can be substantially better than the first PPO checkpoint
        on contact-sensitive tasks.  Keeping its exact actor/normalizer state
        makes that distinction measurable instead of forcing a reconstruction
        from a later, already-updated policy.
        """

        control_manifest = getattr(env, "control_manifest", {})
        curriculum_complete_at_checkpoint = all_curricula_complete()
        return {
            "checkpoint_version": "incoming_hit_training_v3",
            "checkpoint_stage": checkpoint_stage,
            "iteration": int(iteration),
            "env_steps": int(env_steps),
            "obs_size": env.observation_size,
            "action_size": env.action_size,
            "hidden": list(cfg.hidden),
            "config": cfg._asdict(),
            "control_hash": getattr(env, "control_hash", None),
            "control_manifest": control_manifest,
            "training_feed_manifest": getattr(env, "feed_bank_manifest", None),
            "training_prerequisite_binding": getattr(
                env, "training_prerequisite_binding", None
            ),
            "policy_update_contract": getattr(env, "policy_update_contract", None),
            "actor_initialization": initialization_binding,
            "teacher_bc_pretrain_report": teacher_pretrain_report,
            "base_policy_artifact": (
                None
                if getattr(env, "base_policy_artifact", None) is None
                else str(Path(env.base_policy_artifact).expanduser().resolve())
            ),
            "base_skill": getattr(env, "base_skill", None),
            "residual_scale": (
                None
                if getattr(env, "base_policy_artifact", None) is None
                else float(env.residual_scale)
            ),
            "curriculum_complete": curriculum_complete_at_checkpoint,
            "promotion_eligible": curriculum_complete_at_checkpoint,
            "resume_semantics": "iteration_boundary_fresh_environment_reset_v1",
            "curriculum_state": {
                "effective_steps": int(curriculum_effective_steps),
                "phase": (
                    env.curriculum.phase(curriculum_effective_steps)
                    if getattr(env, "curriculum", None) is not None
                    else "disabled"
                ),
                "lambda_lab": float(np.asarray(env_states.lambda_lab)),
                "active_feed_count": int(np.asarray(env_states.active_feed_count)),
                "last_gate": curriculum_gate_report,
                "episode_gate_window": curriculum_gate_window.state_dict(),
            },
            "task_curriculum_state": {
                "schema_version": "stage3_task_curriculum_state_v2",
                "max_stage": getattr(env, "task_curriculum_max_stage", None),
                "stage_index": int(task_stage_index),
                "stage": (
                    env.task_curriculum_values(
                        int(env_steps),
                        stage_index=task_stage_index,
                    ).stage_name
                    if task_curriculum_enabled
                    else "disabled"
                ),
                "complete": bool(task_curriculum_is_complete),
                "last_gate": task_gate_report,
            },
        }

    if teacher_dataset is not None and resume_from is None:
        save_versioned_training_checkpoint(
            out_dir,
            agent=agent,
            optimizer_state=opt_state,
            obs_rms=obs_rms,
            rng_key=key,
            env_rng_key=env_states.key,
            metadata=checkpoint_metadata(
                iteration=0,
                env_steps=0,
                checkpoint_stage="post_teacher_bc_pre_ppo",
            ),
        )

    if start_iteration == target_iters:
        curriculum_complete = all_curricula_complete()
        report = {
            "iterations": target_iters,
            "start_iteration": start_iteration,
            "requested_env_step_cap": int(cfg.total_env_steps),
            "env_steps": int(executed_step_target),
            "unused_env_step_budget": int(unused_step_budget),
            "curriculum_effective_steps": int(curriculum_effective_steps),
            "curriculum_phase": (
                env.curriculum.phase(curriculum_effective_steps)
                if getattr(env, "curriculum", None) is not None
                else "disabled"
            ),
            "curriculum_complete": curriculum_complete,
            "task_curriculum_phase": (
                env.task_curriculum_values(resumed_env_steps, stage_index=task_stage_index).stage_name
                if task_curriculum_enabled
                else "disabled"
            ),
            "task_curriculum_complete": task_curriculum_is_complete,
            "promotion_eligible": curriculum_complete,
            "extension_required": not curriculum_complete,
            "already_at_absolute_cap": True,
            "wall_seconds": 0.0,
            "final": {},
            "checkpoint": str(out_dir / "policy_latest.json"),
            "checkpoint_compatibility_alias": str(out_dir / "policy_latest.npz"),
            "metrics_file": str(metrics_path),
            "training_prerequisite_binding": getattr(env, "training_prerequisite_binding", None),
        }
        if wandb_run is not None:
            report["wandb"] = {
                "run_id": wandb_run.id,
                "url": wandb_run.url,
            }
        (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if wandb_run is not None:
            wandb_run.summary.update(report)
            wandb_run.finish()
        return report
    print(
        "Starting training... "
        f"iterations={target_iters - start_iteration} "
        f"target_env_steps={executed_step_target:,} "
        f"seed={cfg.seed} mode={cfg.policy_update_mode}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    t_start = time.time()
    for it in range(start_iteration + 1, target_iters + 1):
        if hasattr(env, "curriculum_values") and hasattr(env, "apply_curriculum"):
            values = env.curriculum_values(curriculum_effective_steps)
            task_values = (
                env.task_curriculum_values(
                    (it - 1) * steps_per_iter,
                    stage_index=task_stage_index,
                )
                if task_curriculum_enabled
                else env.task_curriculum_values((it - 1) * steps_per_iter)
            )
            env_states = env.apply_curriculum(
                env_states,
                env_steps=(it - 1) * steps_per_iter,
                lambda_lab=values.lambda_lab,
                active_feed_count=min(values.active_feed_count, task_values.active_feed_count),
                v2_stage_index=task_values.stage_index,
                v2_environment_mode=task_values.environment_mode_code,
                v2_reward_mask=task_values.reward_mask,
            )
        t0 = time.time()
        teacher_bc_coef = 0.0
        if teacher_dataset is not None:
            if int(cfg.teacher_bc_decay_steps) > 0:
                teacher_bc_progress = min(
                    1.0,
                    float((it - 1) * steps_per_iter) / float(cfg.teacher_bc_decay_steps),
                )
            else:
                teacher_bc_progress = 0.0
            teacher_bc_coef = float(cfg.teacher_bc_initial_coef) + teacher_bc_progress * (
                float(cfg.teacher_bc_final_coef) - float(cfg.teacher_bc_initial_coef)
            )
        agent, opt_state, obs_rms, env_states, key, metrics = train_iteration(
            agent,
            opt_state,
            obs_rms,
            env_states,
            key,
            teacher_bc_coef,
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        non_finite_metrics = sorted(name for name, value in metrics.items() if not np.isfinite(value))
        if non_finite_metrics:
            failure = {
                "schema_version": "incoming_hit_training_failure_v1",
                "iteration": int(it),
                "last_good_iteration": int(it - 1),
                "non_finite_metrics": non_finite_metrics,
            }
            (out_dir / "training_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"training/non_finite_metric_count": len(non_finite_metrics)},
                    step=int((it - 1) * steps_per_iter),
                )
                wandb_run.finish(exit_code=1)
            raise FloatingPointError("Stage-3 training produced non-finite metrics: " + ", ".join(non_finite_metrics))
        guard_fraction = float(
            metrics.get("ppo_post_update_ratio_guard_fraction", 0.0)
        )
        if guard_fraction > float(cfg.max_post_update_ratio_guard_fraction):
            failure = {
                "schema_version": "incoming_hit_training_failure_v1",
                "iteration": int(it),
                "last_good_iteration": int(it - 1),
                "reason": "post_update_trust_region_violation",
                "ppo_post_update_ratio_guard_fraction": guard_fraction,
                "max_post_update_ratio_guard_fraction": float(
                    cfg.max_post_update_ratio_guard_fraction
                ),
                "ppo_post_update_log_ratio_abs_max": float(
                    metrics["ppo_post_update_log_ratio_abs_max"]
                ),
            }
            (out_dir / "training_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"training/post_update_trust_region_violation": 1.0},
                    step=int((it - 1) * steps_per_iter),
                )
                wandb_run.finish(exit_code=1)
            raise FloatingPointError(
                "Stage-3 PPO post-update trust region guard exceeded: "
                f"{guard_fraction:.6f} > "
                f"{float(cfg.max_post_update_ratio_guard_fraction):.6f}"
            )
        kl_estimate = float(metrics.get("ppo_post_update_kl_estimate", 0.0))
        if kl_estimate > float(cfg.max_post_update_kl_estimate):
            failure = {
                "schema_version": "incoming_hit_training_failure_v1",
                "iteration": int(it),
                "last_good_iteration": int(it - 1),
                "reason": "post_update_kl_exceeded",
                "ppo_post_update_kl_estimate": kl_estimate,
                "max_post_update_kl_estimate": float(
                    cfg.max_post_update_kl_estimate
                ),
                "ppo_post_update_log_ratio_abs_max": float(
                    metrics["ppo_post_update_log_ratio_abs_max"]
                ),
            }
            (out_dir / "training_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"training/post_update_kl_exceeded": 1.0},
                    step=int((it - 1) * steps_per_iter),
                )
                wandb_run.finish(exit_code=1)
            raise FloatingPointError(
                "Stage-3 PPO post-update KL estimate exceeded: "
                f"{kl_estimate:.6f} > "
                f"{float(cfg.max_post_update_kl_estimate):.6f}"
            )
        metrics["iteration"] = it
        metrics["env_steps"] = it * steps_per_iter
        if getattr(env, "curriculum", None) is not None:
            curriculum_gate_window.update(metrics)
            gate_window_summary = curriculum_gate_window.summary()
            curriculum_effective_steps, curriculum_gate_report = env.curriculum.advance(
                effective_steps=curriculum_effective_steps,
                delta_steps=steps_per_iter,
                metrics=curriculum_gate_window.metrics_for_gate(),
            )
            curriculum_gate_report.update(
                {
                    "window_schema_version": curriculum_gate_window.schema_version,
                    "window_ready": bool(gate_window_summary["ready"]),
                    "window_episodes_finished": float(gate_window_summary["episodes_finished"]),
                    "window_hit_rate": float(gate_window_summary["hit_rate"]),
                    "window_crossed_net_rate": float(gate_window_summary["crossed_net_rate"]),
                    "window_fall_rate": float(gate_window_summary["fall_rate"]),
                    "window_positive_outgoing_z_rate_on_hit": float(
                        gate_window_summary["positive_outgoing_z_rate_on_hit"]
                    ),
                }
            )
            metrics["curriculum_effective_steps"] = curriculum_effective_steps
            metrics["curriculum_phase"] = env.curriculum.phase(curriculum_effective_steps)
            metrics["curriculum_gate_checked"] = bool(curriculum_gate_report.get("checked", False))
            metrics["curriculum_gate_passed"] = bool(curriculum_gate_report.get("passed", False))
            metrics["curriculum_gate_window_ready"] = bool(gate_window_summary["ready"])
            metrics["curriculum_gate_window_episodes"] = float(gate_window_summary["episodes_finished"])
            metrics["curriculum_gate_window_hit_rate"] = float(gate_window_summary["hit_rate"])
            metrics["curriculum_gate_window_crossed_net_rate"] = float(gate_window_summary["crossed_net_rate"])
            metrics["curriculum_gate_window_fall_rate"] = float(gate_window_summary["fall_rate"])
            metrics["curriculum_gate_window_positive_outgoing_z_rate_on_hit"] = float(
                gate_window_summary["positive_outgoing_z_rate_on_hit"]
            )
            if bool(curriculum_gate_report.get("checked", False)) and bool(curriculum_gate_report.get("passed", False)):
                curriculum_gate_window.clear()
        if task_curriculum_enabled and not task_curriculum_is_complete:
            from environment.overall_environment.src.stage3_task_curriculum_v2 import (
                canonical_stage3_v2_curriculum,
                promotion_failures,
                stage_by_name,
            )

            stages = canonical_stage3_v2_curriculum()
            maximum_index = stages.index(stage_by_name(env.task_curriculum_max_stage))
            current_stage = stages[task_stage_index]
            if task_stage_index + 1 < len(stages):
                eligible_at = stages[task_stage_index + 1].start_steps
            else:
                eligible_at = current_stage.start_steps + 5_000_000
            checked = int(metrics["env_steps"]) >= int(eligible_at)
            failures = promotion_failures(current_stage, metrics) if checked else ()
            passed = checked and not failures
            task_gate_report = {
                "checked": checked,
                "passed": passed,
                "stage": current_stage.name,
                "eligible_at_env_steps": int(eligible_at),
                "evaluated_at_env_steps": int(metrics["env_steps"]),
                "failures": list(failures),
            }
            if passed:
                if task_stage_index < maximum_index:
                    task_stage_index += 1
                else:
                    task_curriculum_is_complete = True
            metrics["task_curriculum_stage"] = stages[task_stage_index].name
            metrics["task_curriculum_stage_index"] = task_stage_index
            metrics["task_curriculum_gate_checked"] = checked
            metrics["task_curriculum_gate_passed"] = passed
            metrics["task_curriculum_complete"] = task_curriculum_is_complete
        metrics["iter_seconds"] = time.time() - t0
        metrics["env_steps_per_second"] = steps_per_iter / metrics["iter_seconds"]
        history.append(metrics)
        if it % log_every == 0:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=int(metrics["env_steps"]))
            print(
                f"iter {it}/{target_iters} steps={metrics['env_steps']:,} "
                f"reward={metrics['mean_reward']:.4f} hit={metrics['hit_rate']:.3f} "
                f"net={metrics['crossed_net_rate']:.3f} sps={metrics['env_steps_per_second']:,.0f}",
                flush=True,
            )
        if it % checkpoint_every == 0 or it == target_iters:
            save_versioned_training_checkpoint(
                out_dir,
                agent=agent,
                optimizer_state=opt_state,
                obs_rms=obs_rms,
                rng_key=key,
                env_rng_key=env_states.key,
                metadata=checkpoint_metadata(
                    iteration=it,
                    env_steps=it * steps_per_iter,
                    checkpoint_stage="ppo_iteration_boundary",
                ),
            )

    report = {
        "iterations": target_iters,
        "start_iteration": start_iteration,
        "requested_env_step_cap": int(cfg.total_env_steps),
        "env_steps": int(executed_step_target),
        "unused_env_step_budget": int(unused_step_budget),
        "curriculum_effective_steps": int(curriculum_effective_steps),
        "curriculum_phase": (
            env.curriculum.phase(curriculum_effective_steps)
            if getattr(env, "curriculum", None) is not None
            else "disabled"
        ),
        "curriculum_complete": bool(all_curricula_complete()),
        "promotion_eligible": bool(all_curricula_complete()),
        "extension_required": bool(not all_curricula_complete()),
        "task_curriculum_phase": (
            env.task_curriculum_values(executed_step_target, stage_index=task_stage_index).stage_name
            if task_curriculum_enabled
            else "disabled"
        ),
        "task_curriculum_complete": bool(task_curriculum_is_complete),
        "task_curriculum_last_gate": task_gate_report,
        "wall_seconds": time.time() - t_start,
        "final": history[-1] if history else {},
        "checkpoint": str(out_dir / "policy_latest.json"),
        "checkpoint_compatibility_alias": str(out_dir / "policy_latest.npz"),
        "metrics_file": str(metrics_path),
        "training_prerequisite_binding": getattr(env, "training_prerequisite_binding", None),
        "policy_update_contract": getattr(env, "policy_update_contract", None),
        "actor_initialization": initialization_binding,
    }
    if wandb_run is not None:
        report["wandb"] = {
            "run_id": wandb_run.id,
            "url": wandb_run.url,
        }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(report)
        wandb_run.finish()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="experiments/posttrain/incoming_shuttle_hit_v1.yaml")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument(
        "--base-policy-artifact", default=None, help="frozen base policy export dir (Stage 3 residual mode)"
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="residual amplitude (default: stage3_direct.residual_scale)",
    )
    parser.add_argument("--base-skill", default=None, help="skill name for a multi-skill base")
    parser.add_argument("--latent-checkpoint", default=None)
    parser.add_argument("--allow-unpromoted-latent", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--initialize-policy-from", type=Path, default=None)
    parser.add_argument(
        "--curriculum-max-stage",
        default=None,
        help="Clamp impact_recovery_v2 task curriculum at a canonical C0--C7 stage.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume the committed policy_latest pointer in --out-dir when present.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "musclemimic" / "badminton" / "scripts"))
    from run_incoming_shuttle_hit import (
        _build_stage3_lab_components,
        _ensure_feed_bank_artifact,
        _ensure_scene,
        load_incoming_hit_spec,
    )

    try:
        from run_incoming_shuttle_hit import _residual_scale_schedule
    except ImportError:
        # Lightweight injected runners used by downstream tests predate the
        # optional authority schedule and therefore represent the legacy
        # no-schedule contract.
        def _residual_scale_schedule(_paths):
            return {}

    try:
        from run_incoming_shuttle_hit import _seed_feed_fingerprints
    except ImportError:
        def _seed_feed_fingerprints(_paths):
            return ()

    try:
        from run_incoming_shuttle_hit import _policy_update_contract
    except ImportError:
        # Legacy injected runners use the historical full-network update.
        def _policy_update_contract(_paths, model):
            return {
                "schema_version": "stage3_policy_update_contract_v1",
                "mode": "full_network",
                "freeze_observation_normalizer": False,
                "trainable_actuator_names": [],
                "trainable_action_indices": [],
                "trainable_action_count": int(model.nu),
                "full_action_count": int(model.nu),
                "contract_sha256": None,
            }

    try:
        from run_incoming_shuttle_hit import _build_stage3_direct_curriculum
    except ImportError:
        # Compatibility for lightweight injected runners in unit tests and
        # older downstream wrappers.  No base artifact means no direct
        # curriculum, matching the historical behavior.
        def _build_stage3_direct_curriculum(_paths, *, base_policy_artifact):
            if base_policy_artifact is not None:
                raise ValueError("injected Stage-3 runner does not support a direct residual curriculum")
            return None

    paths = load_incoming_hit_spec(args.spec)
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    bank = feed_artifact.bank
    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=args.latent_checkpoint,
        allow_unpromoted=args.allow_unpromoted_latent,
    )
    direct_config = dict(getattr(paths, "stage3_direct", {}) or {})
    base_policy_artifact = args.base_policy_artifact
    configured_artifact = direct_config.get("base_policy_artifact")
    if base_policy_artifact is None and configured_artifact:
        candidate = Path(configured_artifact).expanduser()
        base_policy_artifact = str(candidate if candidate.is_absolute() else REPO_ROOT / candidate)
    residual_scale = (
        float(direct_config.get("residual_scale", 0.3)) if args.residual_scale is None else float(args.residual_scale)
    )
    residual_scale_overrides = direct_config.get("residual_scale_overrides", {})
    if residual_scale_overrides is None:
        residual_scale_overrides = {}
    if not isinstance(residual_scale_overrides, dict):
        raise ValueError("stage3_direct.residual_scale_overrides must be a mapping")
    direct_curriculum = _build_stage3_direct_curriculum(
        paths,
        base_policy_artifact=base_policy_artifact,
    )
    task_profile = getattr(paths, "task_profile", "legacy_v1")
    return_constraints = dict(getattr(paths, "return_constraints", {}) or {})
    min_return_clearance = return_constraints.get("min_clearance_m")

    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        return_net_x_m=float(return_constraints.get("net_x_m", 0.0)),
        return_net_height_m=float(return_constraints.get("net_height_m", 1.55)),
        min_return_net_clearance_m=(None if min_return_clearance is None else float(min_return_clearance)),
        desired_return_up_component=float(return_constraints.get("desired_up_component", 0.40)),
        ballistic_return_score_softness_m=float(return_constraints.get("ballistic_score_softness_m", 0.35)),
        clearance_prediction_mode=str(
            return_constraints.get(
                "clearance_prediction_mode",
                "vacuum_ballistic_v1",
            )
        ),
        shuttle_proximity_softness_m=float(return_constraints.get("shuttle_proximity_softness_m", 0.35)),
        timed_intercept_softness_m=float(return_constraints.get("timed_intercept_softness_m", 0.30)),
        direction_distance_softness_m=float(return_constraints.get("direction_distance_softness_m", 0.45)),
        contact_guidance_reward_mode=str(
            return_constraints.get("contact_guidance_reward_mode", "dense_per_step")
        ),
        contact_guidance_discount=float(
            return_constraints.get("contact_guidance_discount", 1.0)
        ),
        racket_velocity_direction_fraction=float(return_constraints.get("racket_velocity_direction_fraction", 0.30)),
        direction_reward_mode=str(return_constraints.get("direction_reward_mode", "positive_projection")),
        clearance_reward_mode=str(return_constraints.get("clearance_reward_mode", "positive_score")),
        hit_event_mode=str(return_constraints.get("hit_event_mode", "any_stringbed_contact")),
        racket_guidance_mode=str(return_constraints.get("racket_guidance_mode", "component_projection")),
        inverse_target_speed_m_s=float(return_constraints.get("inverse_target_speed_m_s", 12.0)),
        inverse_velocity_softness_m_s=float(return_constraints.get("inverse_velocity_softness_m_s", 6.0)),
        task_profile=task_profile,
        impact_target_bank=getattr(paths, "target_bank_path", None),
        recovery_horizon_steps=getattr(paths, "recovery_horizon_steps", 60),
        impl=args.impl,
        base_policy_artifact=base_policy_artifact,
        residual_scale=residual_scale,
        residual_scale_overrides=residual_scale_overrides,
        residual_scale_schedule=_residual_scale_schedule(paths),
        base_skill=args.base_skill,
        lab_controller=None if lab is None else lab.controller,
        lab_state_builder=None if lab is None else lab.state_builder,
        curriculum=lab.curriculum if lab is not None else direct_curriculum,
        curriculum_feed_order=(
            "difficulty_sorted" if lab is not None else str(direct_config.get("feed_order", "difficulty_sorted"))
        ),
        seed_feed_fingerprints=(() if lab is not None else _seed_feed_fingerprints(paths)),
        filter_finger_observation=None if lab is None else True,
        feed_bank_manifest=feed_artifact.manifest,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.76)),
        swing_phase_advance_s=float(direct_config.get("swing_phase_advance_s", 0.0)),
        task_curriculum_max_stage=(
            args.curriculum_max_stage
            if args.curriculum_max_stage is not None
            else ("C7_recovery" if task_profile == "impact_recovery_v2" else None)
        ),
    )
    policy_update_contract = _policy_update_contract(paths, env.model)
    env.policy_update_contract = policy_update_contract
    out_dir = args.out_dir if args.out_dir is not None else paths.output_dir / "train_gpu"
    if lab is not None:
        prerequisite_binding = validate_stage3_training_prerequisites(
            Path(out_dir),
            paths=paths,
            latent_checkpoint_dir=Path(lab.latent_checkpoint_dir),
            control_manifest=env.control_manifest,
            training_feed_manifest=env.feed_bank_manifest,
        )
        env.training_prerequisite_binding = prerequisite_binding
    elif task_profile == "impact_recovery_v2":
        env.training_prerequisite_binding = validate_stage3_direct_training_prerequisites(
            Path(out_dir),
            paths=paths,
            control_manifest=env.control_manifest,
            training_feed_manifest=env.feed_bank_manifest,
        )
    elif base_policy_artifact is not None:
        env.training_prerequisite_binding = validate_stage3_residual_training_prerequisites(
            Path(out_dir),
            paths=paths,
            base_policy_artifact=Path(base_policy_artifact),
            control_manifest=env.control_manifest,
            training_feed_manifest=env.feed_bank_manifest,
            policy_update_contract=policy_update_contract,
        )
    ppo = dict(paths.ppo_overrides)
    contact_guidance_reward_mode = str(
        getattr(
            env,
            "contact_guidance_reward_mode",
            return_constraints.get("contact_guidance_reward_mode", "dense_per_step"),
        )
    )
    contact_guidance_discount = float(
        getattr(
            env,
            "contact_guidance_discount",
            return_constraints.get("contact_guidance_discount", 1.0),
        )
    )
    if contact_guidance_reward_mode == "potential_event_direction":
        ppo_gamma = float(ppo.get("gamma", 0.99))
        if not math.isclose(
            contact_guidance_discount,
            ppo_gamma,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "potential_event_direction requires contact_guidance_discount "
                "to exactly match ppo.gamma"
            )
    cfg = TrainConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_env_steps=int(
            ppo.get("total_steps", 2_000_000) if args.total_env_steps is None else args.total_env_steps
        ),
        update_epochs=int(ppo.get("update_epochs", 4)),
        num_minibatches=int(ppo.get("num_minibatches", 8)),
        minibatch_size=int(ppo.get("minibatch_size", 0)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_coef=float(ppo.get("clip_coef", 0.2)),
        value_coef=float(ppo.get("value_coef", 0.5)),
        entropy_coef=float(ppo.get("entropy_coef", 0.001)),
        hidden=tuple(ppo.get("hidden_sizes", (256, 256))),
        action_std_init=float(ppo.get("action_std_init", 0.35)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        actor_learning_rate=(
            None if ppo.get("actor_learning_rate") is None else float(ppo["actor_learning_rate"])
        ),
        critic_learning_rate=(
            None if ppo.get("critic_learning_rate") is None else float(ppo["critic_learning_rate"])
        ),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        max_abs_log_ratio=float(ppo.get("max_abs_log_ratio", 10.0)),
        max_post_update_ratio_guard_fraction=float(
            ppo.get("max_post_update_ratio_guard_fraction", 1.0)
        ),
        max_post_update_kl_estimate=float(
            ppo.get("max_post_update_kl_estimate", 1.0e9)
        ),
        policy_update_mode=str(policy_update_contract["mode"]),
        policy_trainable_action_indices=tuple(policy_update_contract["trainable_action_indices"]),
        policy_delta_hidden=tuple(policy_update_contract.get("policy_delta_hidden_sizes", ())),
        policy_refinement_delta_hidden=tuple(
            policy_update_contract.get("policy_refinement_delta_hidden_sizes", ())
        ),
        policy_correction_hidden=tuple(
            policy_update_contract.get("policy_correction_hidden_sizes", ())
        ),
        correction_physical_scales=tuple(
            policy_update_contract.get("correction_physical_scales", ())
        ),
        correction_std_init=tuple(policy_update_contract.get("correction_std_init", ())),
        correction_std_min=tuple(policy_update_contract.get("correction_std_min", ())),
        correction_std_max=tuple(policy_update_contract.get("correction_std_max", ())),
        correction_window_open_s=float(
            policy_update_contract.get("correction_window", {}).get(
                "time_to_intercept_open_s", 0.70
            )
        ),
        correction_window_close_s=float(
            policy_update_contract.get("correction_window", {}).get(
                "time_to_intercept_close_s", -0.10
            )
        ),
        correction_window_smoothing_s=float(
            policy_update_contract.get("correction_window", {}).get("smoothing_s", 0.05)
        ),
        teacher_action_prior_mode=str(
            policy_update_contract.get("teacher_action_prior_mode", "none")
        ),
        quality_success_min_outgoing_z_m_s=float(
            policy_update_contract.get("quality_success", {}).get("min_outgoing_z_m_s", 0.5)
        ),
        quality_success_min_forward_m_s=float(
            policy_update_contract.get("quality_success", {}).get("min_forward_m_s", 2.0)
        ),
        quality_success_min_predicted_net_clearance_m=float(
            policy_update_contract.get("quality_success", {}).get(
                "min_predicted_net_clearance_m", -1.0e9
            )
        ),
        quality_success_min_return_direction_signed_score=float(
            policy_update_contract.get("quality_success", {}).get(
                "min_return_direction_signed_score", -1.0
            )
        ),
        quality_success_min_racket_face_forward_alignment=float(
            policy_update_contract.get("quality_success", {}).get(
                "min_racket_face_forward_alignment", -1.0
            )
        ),
        quality_success_require_episode_no_fall=bool(
            policy_update_contract.get("quality_success", {}).get(
                "require_episode_no_fall", False
            )
        ),
        quality_imitation_mode=str(
            policy_update_contract.get("quality_imitation", {}).get(
                "mode", "strict_success"
            )
        ),
        quality_imitation_min_weight=float(
            policy_update_contract.get("quality_imitation", {}).get(
                "min_weight", 0.0
            )
        ),
        quality_imitation_forward_softness_m_s=float(
            policy_update_contract.get("quality_imitation", {}).get(
                "forward_softness_m_s", 1.0
            )
        ),
        quality_imitation_vertical_softness_m_s=float(
            policy_update_contract.get("quality_imitation", {}).get(
                "vertical_softness_m_s", 0.75
            )
        ),
        quality_imitation_clearance_softness_m=float(
            policy_update_contract.get("quality_imitation", {}).get(
                "clearance_softness_m", 0.75
            )
        ),
        quality_imitation_direction_softness=float(
            policy_update_contract.get("quality_imitation", {}).get(
                "direction_softness", 0.10
            )
        ),
        quality_imitation_require_episode_no_fall=bool(
            policy_update_contract.get("quality_imitation", {}).get(
                "require_episode_no_fall", False
            )
        ),
        freeze_observation_normalizer=bool(policy_update_contract["freeze_observation_normalizer"]),
        frozen_action_std=policy_update_contract.get("frozen_action_std"),
        freeze_trainable_action_std=bool(
            policy_update_contract.get("freeze_trainable_action_std", False)
        ),
        successful_action_imitation_coef=float(
            policy_update_contract.get("successful_action_imitation_coef", 0.0)
        ),
        seed=args.seed,
    )
    resume_from = args.resume_from
    if resume_from is None and args.auto_resume:
        latest_pointer = Path(out_dir) / "policy_latest.json"
        legacy_latest = Path(out_dir) / "policy_latest.npz"
        if latest_pointer.is_file():
            resume_from = latest_pointer
        elif legacy_latest.is_file():
            resume_from = legacy_latest
    print("jax devices:", jax.devices())
    report = train(
        env,
        cfg,
        Path(out_dir),
        checkpoint_every=args.checkpoint_every,
        resume_from=resume_from,
        initialize_policy_from=args.initialize_policy_from,
    )
    print(json.dumps(report["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
