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


def init_agent(key, obs_size, action_size, hidden, action_std_init):
    k1, k2 = jax.random.split(key)
    policy = _init_mlp(k1, (obs_size, *hidden, action_size))
    value = _init_mlp(k2, (obs_size, *hidden, 1))
    policy[-1]["w"] = policy[-1]["w"] * 0.01
    return {
        "policy": policy,
        "value": value,
        "log_std": jnp.full((action_size,), jnp.log(action_std_init)),
    }


def _dist(params, obs):
    mean = _mlp(params["policy"], obs)
    std = jnp.exp(jnp.clip(params["log_std"], -5.0, 1.0))
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


def evaluate_actions(params, obs, raw_actions, *, squash_action: bool = True):
    mean, std = _dist(params, obs)
    if squash_action:
        squashed = jnp.tanh(raw_actions)
        logp = _tanh_normal_logprob(mean, std, raw_actions, squashed)
    else:
        logp = _normal_logprob(mean, std, raw_actions)
    entropy = jnp.sum(0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(std), axis=-1)
    value = _mlp(params["value"], obs)[..., 0]
    return logp, entropy, value


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
    max_grad_norm: float = 0.5
    hidden: tuple = (256, 256)
    action_std_init: float = 0.35
    seed: int = 0


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


def make_train_iteration(env: IncomingHitMjxEnv, mx, cfg: TrainConfig, optimizer):
    step_env = env.make_step_fn(mx, cfg.num_envs)
    squash_action = not bool(getattr(env, "expects_raw_latent", False))
    num_samples = cfg.rollout_steps * cfg.num_envs
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

    def rollout(agent, obs_rms, env_states, key):
        def body(carry, _):
            env_states, key = carry
            key, sub = jax.random.split(key)
            obs_norm = obs_rms.normalize(env_states.obs)
            action, raw, logp = sample_action(agent, obs_norm, sub, squash_action=squash_action)
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
                "landing_score": tr["landing_score"],
                "miss": tr["miss"],
                "body_fall": tr["body_fall"],
                "hit_event": tr["hit_event"],
                "landing_event": tr["landing_event"],
                "obs_raw": env_states.obs,
            }
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
                "muscle_power_abs_mean",
                "normalized_control_energy",
                "body_action_saturation_fraction",
                "full_action_saturation_fraction",
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

    def ppo_update(agent, opt_state, batch, key):
        def epoch(carry, _):
            agent, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, num_samples)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)

            def minibatch(carry, mb):
                agent, opt_state = carry

                def loss_fn(params):
                    logp, entropy, value = evaluate_actions(
                        params,
                        mb["obs_norm"],
                        mb["raw_action"],
                        squash_action=squash_action,
                    )
                    ratio = jnp.exp(logp - mb["logp"])
                    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                    pg1 = -adv * ratio
                    pg2 = -adv * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    policy_loss = jnp.maximum(pg1, pg2).mean()
                    value_loss = 0.5 * jnp.square(value - mb["returns"]).mean()
                    entropy_loss = -entropy.mean()
                    total = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
                    return total, (policy_loss, value_loss, entropy_loss)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent)
                updates, opt_state = optimizer.update(grads, opt_state, agent)
                agent = optax.apply_updates(agent, updates)
                agent = {
                    **agent,
                    "log_std": jnp.clip(agent["log_std"], -5.0, 1.0),
                }
                return (agent, opt_state), (loss, *aux)

            mbs = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, mb_size, *x.shape[1:]),
                shuffled,
            )
            (agent, opt_state), losses = jax.lax.scan(minibatch, (agent, opt_state), mbs)
            return (agent, opt_state, key), losses

        (agent, opt_state, key), losses = jax.lax.scan(epoch, (agent, opt_state, key), None, length=cfg.update_epochs)
        return agent, opt_state, key, losses

    @jax.jit
    def train_iteration(agent, opt_state, obs_rms, env_states, key):
        env_states, key, records = rollout(agent, obs_rms, env_states, key)
        advantages, returns = compute_rollout_gae(
            records,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        # Preserve the rollout-time normalization for every value target.  The
        # updated statistics apply only to the following iteration.
        obs_rms = obs_rms.update(records["obs_raw"])

        def flat(x):
            return x.reshape(-1, *x.shape[2:])

        batch = {
            "obs_norm": flat(records["obs_norm"]),
            "raw_action": flat(records["raw_action"]),
            "logp": flat(records["logp"]),
            "adv": flat(advantages),
            "returns": flat(returns),
        }
        agent, opt_state, key, losses = ppo_update(agent, opt_state, batch, key)

        done = records["done"]
        n_done = jnp.maximum(done.sum(), 1)
        metrics = {
            "mean_reward": records["reward"].mean(),
            "episodes_finished": done.sum(),
            "hit_rate": jnp.where(done, records["hit"], 0).sum() / n_done,
            "crossed_net_rate": jnp.where(done, records["crossed_net"], 0).sum() / n_done,
            "mean_landing_score": records["landing_score"].sum() / n_done,
            "miss_rate": jnp.where(done, records["miss"], 0).sum() / n_done,
            "fall_rate": jnp.where(done, records["body_fall"], 0).sum() / n_done,
            "loss": losses[0].mean(),
            "policy_loss": losses[1].mean(),
            "value_loss": losses[2].mean(),
            "entropy_loss": losses[3].mean(),
        }
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
            "net_clearance_m",
            "muscle_power_abs_mean",
            "normalized_control_energy",
            "body_action_saturation_fraction",
            "full_action_saturation_fraction",
            "bounded_residual_rms",
        ):
            if name in records:
                metrics[name] = records[name].mean()
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
    if consumer_order.get("mode") not in {"difficulty_sorted", "stored"}:
        raise ValueError("Stage-3 runtime feed consumer_order mode is incompatible")
    producer_fingerprints = producer.get("sample_fingerprints")
    consumer_fingerprints = consumer_order.get("sample_fingerprints")
    if not isinstance(producer_fingerprints, list) or not isinstance(consumer_fingerprints, list):
        raise ValueError("Stage-3 feed manifests must contain sample_fingerprints")
    if sorted(str(value) for value in producer_fingerprints) != sorted(str(value) for value in consumer_fingerprints):
        raise ValueError("Stage-3 runtime consumer_order changed the feed-bank identity")
    return producer, consumer_order


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
        "hard_weld_present",
        "weld_strength_passed",
    )
    if any(preflight.get(name) is not True for name in required_true):
        raise ValueError("Stage-3 preflight has a failed scene/weld predicate")
    if preflight.get("missing_sites") != []:
        raise ValueError("Stage-3 preflight is missing required sites")
    if int(preflight.get("actuator_count", -1)) != 416:
        raise ValueError("Stage-3 preflight does not prove 416 actuators")

    router = preflight.get("action_router")
    if not isinstance(router, dict) or router.get("schema_version") != ("stage3_action_router_v1"):
        raise ValueError("Stage-3 preflight action router schema is incompatible")
    if router.get("partition_sizes") != [354, 31, 31]:
        raise ValueError("Stage-3 preflight does not prove the 354+31+31 router")
    all_names = router.get("all_actuator_names")
    owned_groups = (
        router.get("body_actuator_names"),
        router.get("right_grip_actuator_names"),
        router.get("left_neutral_actuator_names"),
    )
    expected_lengths = (354, 31, 31)
    if not isinstance(all_names, list) or len(all_names) != 416:
        raise ValueError("Stage-3 preflight action router has no full actuator list")
    if any(
        not isinstance(group, list) or len(group) != expected
        for group, expected in zip(owned_groups, expected_lengths, strict=True)
    ):
        raise ValueError("Stage-3 preflight action router partition names are incomplete")
    all_strings = [str(value) for value in all_names]
    owned_strings = [[str(value) for value in group] for group in owned_groups]
    if len(set(all_strings)) != 416:
        raise ValueError("Stage-3 preflight action router has duplicate actuators")
    flattened = [value for group in owned_strings for value in group]
    if len(set(flattened)) != 416 or set(flattened) != set(all_strings):
        raise ValueError("Stage-3 preflight action router ownership is not exhaustive")
    router_identity = {
        "schema_version": "stage3_action_router_v1",
        "all": all_strings,
        "body": owned_strings[0],
        "right_grip": owned_strings[1],
        "left_neutral": owned_strings[2],
    }
    if router.get("schema_hash") != _stable_json_hash(router_identity):
        raise ValueError("Stage-3 preflight action router hash is invalid")

    attachment = preflight.get("racket_attachment")
    if not isinstance(attachment, dict) or attachment.get("schema_version") != ("stage3_attachment_v1"):
        raise ValueError("Stage-3 preflight attachment schema is incompatible")
    recorded_attachment_hash = attachment.get("attachment_hash")
    attachment_unbound = dict(attachment)
    attachment_unbound.pop("attachment_hash", None)
    if recorded_attachment_hash != _stable_json_hash(attachment_unbound):
        raise ValueError("Stage-3 preflight attachment hash is invalid")
    if (
        attachment.get("weld_active") is not True
        or attachment.get("contact_exclude_present") is not True
        or attachment.get("hand_racket_contact_enabled") is not False
        or attachment.get("human_racket_explicit_contact_pairs") != 0
        or attachment.get("human_racket_mask_compatible_geom_pairs") != 0
    ):
        raise ValueError("Stage-3 preflight does not prove hard-weld/zero-contact ownership")
    solref = attachment.get("weld_solref")
    maximum = _finite_float(
        preflight.get("max_weld_solref_time_constant_s"),
        label="preflight weld time-constant threshold",
    )
    if (
        not isinstance(solref, list)
        or not solref
        or _finite_float(solref[0], label="preflight weld solref") > maximum + 1e-12
    ):
        raise ValueError("Stage-3 preflight weld is weaker than the configured threshold")

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
    if base_only.get("task_action") != "all_zero_raw_latent":
        raise ValueError("Stage-3 base-only report did not use zero task action")
    if base_only.get("shuttle_mode") != "parked_out_of_scene":
        raise ValueError("Stage-3 base-only report used an active shuttle")
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

    threshold_names = (
        "min_rollout_count",
        "min_completion_rate",
        "min_finite_rate",
        "min_no_fall_rate",
        "min_root_height_m",
        "max_body_action_saturation_fraction",
        "max_full_action_saturation_fraction",
        "max_normalized_control_energy",
        "max_lab_state_ood_fraction",
        "min_control_finite",
        "max_attachment_translation_drift_m",
        "max_attachment_rotation_drift_rad",
    )
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
        "max_lab_state_ood_fraction": max(episode_numbers("max_lab_state_ood_fraction")),
        "min_control_finite": min(episode_numbers("min_control_finite")),
        "max_attachment_translation_drift_m": max(episode_numbers("max_attachment_translation_drift_m")),
        "max_attachment_rotation_drift_rad": max(episode_numbers("max_attachment_rotation_drift_rad")),
    }
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
        "lab_state_ood_fraction": metrics["max_lab_state_ood_fraction"] <= values["max_lab_state_ood_fraction"],
        "control_finite": metrics["min_control_finite"] >= values["min_control_finite"],
        "attachment_translation_drift": metrics["max_attachment_translation_drift_m"]
        <= values["max_attachment_translation_drift_m"],
        "attachment_rotation_drift": metrics["max_attachment_rotation_drift_rad"]
        <= values["max_attachment_rotation_drift_rad"],
    }
    if gates != expected_gates or not all(expected_gates.values()):
        raise ValueError("Stage-3 base-only report has a failed or inconsistent gate")


def _validate_feed_check_predicates(
    feed_check: dict[str, Any],
    *,
    paths: Any,
    producer_training_manifest: dict[str, Any],
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
            or int(entry.get("unique_sample_count", -1)) != unique_count
            or unique_count != bank_size
        ):
            raise ValueError(f"Stage-3 feed-check {label} predicate failed")
        manifests[label] = manifest
    if manifests["train"] != producer_training_manifest:
        raise ValueError("Stage-3 training feed changed after feed-check")

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
        expected_mode = "difficulty_sorted" if control_manifest.get("curriculum") is not None else "stored"
        if consumer_order.get("mode") != expected_mode:
            raise ValueError("Stage-3 runtime feed order differs from the control curriculum")
    feed_check = reports["feed_check"]
    if feed_check.get("runner_stage") != "feed-check":
        raise ValueError("Stage-3 feed-check report schema is incompatible")
    _validate_feed_check_predicates(
        feed_check,
        paths=paths,
        producer_training_manifest=producer_training_manifest,
    )

    binding: dict[str, Any] = {
        "schema_version": "stage3_training_prerequisite_binding_v1",
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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_feed_manifest = getattr(env, "feed_bank_manifest", None)
    feed_manifest_required = bool(getattr(env, "expects_raw_latent", False))
    if feed_manifest_required:
        # There is no checkpoint yet, so validate only the runtime half here.
        if not isinstance(runtime_feed_manifest, dict):
            raise ValueError("Stage-3 training requires a verified feed-bank manifest")
        prerequisite_binding = getattr(env, "training_prerequisite_binding", None)
        if not isinstance(prerequisite_binding, dict) or prerequisite_binding.get("verified") is not True:
            raise ValueError("Stage-3 LAB training requires verified preflight/base-only/feed evidence")
        recorded = prerequisite_binding.get("binding_sha256")
        unbound = dict(prerequisite_binding)
        unbound.pop("binding_sha256", None)
        if recorded != _stable_json_hash(unbound):
            raise ValueError("Stage-3 prerequisite binding hash mismatch")
    key = jax.random.PRNGKey(cfg.seed)
    key, k_agent, k_env = jax.random.split(key, 3)

    mx = env.put_model(cfg.num_envs)
    template = env.make_batched_template(cfg.num_envs)
    reset_fn = env.make_reset_fn(mx, cfg.num_envs)
    env_states = jax.jit(reset_fn)(k_env, template)

    agent = init_agent(k_agent, env.observation_size, env.action_size, cfg.hidden, cfg.action_std_init)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate),
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
    task_curriculum_enabled = bool(getattr(env, "task_curriculum_max_stage", None) is not None)
    task_stage_index = 0
    task_curriculum_is_complete = not task_curriculum_enabled
    task_gate_report: dict[str, Any] = {
        "checked": False,
        "passed": not task_curriculum_enabled,
        "failures": [],
    }
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
            raise ValueError("resume Stage-3 prerequisite evidence changed")
        validate_training_feed_manifest(
            runtime_feed_manifest,
            checkpoint_manifest=restored.metadata.get("training_feed_manifest"),
            required=feed_manifest_required,
        )
        checkpoint_config = dict(restored.metadata.get("config", {}) or {})
        runtime_config = cfg._asdict()
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

    train_iteration = make_train_iteration(env, mx, cfg, optimizer)

    steps_per_iter = cfg.num_envs * cfg.rollout_steps
    target_iters, executed_step_target, unused_step_budget = _full_batch_budget(
        total_env_steps=cfg.total_env_steps,
        steps_per_iteration=steps_per_iter,
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

    def lab_curriculum_complete() -> bool:
        return bool(
            getattr(env, "curriculum", None) is None or curriculum_effective_steps >= env.curriculum.curriculum_end
        )

    def all_curricula_complete() -> bool:
        return lab_curriculum_complete() and task_curriculum_is_complete

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
        (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return report
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
                lambda_lab=values.lambda_lab,
                active_feed_count=min(values.active_feed_count, task_values.active_feed_count),
                v2_stage_index=task_values.stage_index,
                v2_environment_mode=task_values.environment_mode_code,
                v2_reward_mask=task_values.reward_mask,
            )
        t0 = time.time()
        agent, opt_state, obs_rms, env_states, key, metrics = train_iteration(
            agent, opt_state, obs_rms, env_states, key
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
            raise FloatingPointError("Stage-3 training produced non-finite metrics: " + ", ".join(non_finite_metrics))
        metrics["iteration"] = it
        metrics["env_steps"] = it * steps_per_iter
        if getattr(env, "curriculum", None) is not None:
            curriculum_effective_steps, curriculum_gate_report = env.curriculum.advance(
                effective_steps=curriculum_effective_steps,
                delta_steps=steps_per_iter,
                metrics=metrics,
            )
            metrics["curriculum_effective_steps"] = curriculum_effective_steps
            metrics["curriculum_phase"] = env.curriculum.phase(curriculum_effective_steps)
            metrics["curriculum_gate_checked"] = bool(curriculum_gate_report.get("checked", False))
            metrics["curriculum_gate_passed"] = bool(curriculum_gate_report.get("passed", False))
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
            print(
                f"iter {it}/{target_iters} steps={metrics['env_steps']:,} "
                f"reward={metrics['mean_reward']:.4f} hit={metrics['hit_rate']:.3f} "
                f"net={metrics['crossed_net_rate']:.3f} sps={metrics['env_steps_per_second']:,.0f}",
                flush=True,
            )
        if it % checkpoint_every == 0 or it == target_iters:
            control_manifest = getattr(env, "control_manifest", {})
            curriculum_complete_at_checkpoint = all_curricula_complete()
            save_versioned_training_checkpoint(
                out_dir,
                agent=agent,
                optimizer_state=opt_state,
                obs_rms=obs_rms,
                rng_key=key,
                env_rng_key=env_states.key,
                metadata={
                    "checkpoint_version": "incoming_hit_training_v3",
                    "iteration": it,
                    "env_steps": it * steps_per_iter,
                    "obs_size": env.observation_size,
                    "action_size": env.action_size,
                    "hidden": list(cfg.hidden),
                    "config": cfg._asdict(),
                    "control_hash": getattr(env, "control_hash", None),
                    "control_manifest": control_manifest,
                    "training_feed_manifest": getattr(env, "feed_bank_manifest", None),
                    "training_prerequisite_binding": getattr(env, "training_prerequisite_binding", None),
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
                    },
                    "task_curriculum_state": {
                        "schema_version": "stage3_task_curriculum_state_v2",
                        "max_stage": getattr(env, "task_curriculum_max_stage", None),
                        "stage_index": int(task_stage_index),
                        "stage": (
                            env.task_curriculum_values(
                                it * steps_per_iter,
                                stage_index=task_stage_index,
                            ).stage_name
                            if task_curriculum_enabled
                            else "disabled"
                        ),
                        "complete": bool(task_curriculum_is_complete),
                        "last_gate": task_gate_report,
                    },
                },
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
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
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
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--base-skill", default=None, help="skill name for a multi-skill base")
    parser.add_argument("--latent-checkpoint", default=None)
    parser.add_argument("--allow-unpromoted-latent", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
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

    paths = load_incoming_hit_spec(args.spec)
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    bank = feed_artifact.bank
    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=args.latent_checkpoint,
        allow_unpromoted=args.allow_unpromoted_latent,
    )
    task_profile = getattr(paths, "task_profile", "legacy_v1")

    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        task_profile=task_profile,
        impact_target_bank=getattr(paths, "target_bank_path", None),
        recovery_horizon_steps=getattr(paths, "recovery_horizon_steps", 60),
        impl=args.impl,
        base_policy_artifact=args.base_policy_artifact,
        residual_scale=args.residual_scale,
        base_skill=args.base_skill,
        lab_controller=None if lab is None else lab.controller,
        lab_state_builder=None if lab is None else lab.state_builder,
        curriculum=None if lab is None else lab.curriculum,
        filter_finger_observation=None if lab is None else True,
        feed_bank_manifest=feed_artifact.manifest,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.55)),
        task_curriculum_max_stage=(
            args.curriculum_max_stage
            if args.curriculum_max_stage is not None
            else ("C7_recovery" if task_profile == "impact_recovery_v2" else None)
        ),
    )
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
    ppo = dict(paths.ppo_overrides)
    cfg = TrainConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_env_steps=int(
            ppo.get("total_steps", 2_000_000) if args.total_env_steps is None else args.total_env_steps
        ),
        update_epochs=int(ppo.get("update_epochs", 4)),
        minibatch_size=int(ppo.get("minibatch_size", 0)),
        hidden=tuple(ppo.get("hidden_sizes", (256, 256))),
        action_std_init=float(ppo.get("action_std_init", 0.35)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
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
    )
    print(json.dumps(report["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
