"""Adaptive start-state initialization utilities.

This module keeps the ASI update logic pure so the PPO runner can opt into it
without changing environment behavior when ASI is disabled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class FrameASIState:
    """State for per-trajectory, per-start-bucket ASI sampling."""

    logits: jnp.ndarray
    baseline: jnp.ndarray


def create_frame_asi_state(
    n_trajectories: int,
    num_buckets: int,
    baseline_init: float = 0.0,
) -> FrameASIState:
    """Create a uniform ASI state over `(trajectory, start_bucket)`."""
    shape = (int(n_trajectories), int(num_buckets))
    return FrameASIState(
        logits=jnp.zeros(shape, dtype=jnp.float32),
        baseline=jnp.full(shape, baseline_init, dtype=jnp.float32),
    )


def make_bucket_start_steps(
    split_points: jnp.ndarray,
    num_buckets: int,
    min_remaining_steps: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map each trajectory's ASI buckets to valid local start-frame indices.

    Args:
        split_points: `(n_trajectories + 1,)` cumulative trajectory boundaries.
        num_buckets: Number of ASI buckets per trajectory.
        min_remaining_steps: Required number of frames after the selected start.

    Returns:
        start_steps: `(n_trajectories, num_buckets)` local start indices.
        valid_mask: `(n_trajectories, num_buckets)` true when a trajectory is
            long enough for ASI sampling under `min_remaining_steps`.
    """
    lengths = jnp.diff(split_points).astype(jnp.int32)
    min_remaining_steps = jnp.maximum(jnp.asarray(min_remaining_steps, dtype=jnp.int32), 1)
    max_start = jnp.maximum(lengths - min_remaining_steps, 0)

    bucket_ids = jnp.arange(num_buckets, dtype=jnp.int32)
    denom = max(int(num_buckets) - 1, 1)
    start_steps = (max_start[:, None] * bucket_ids[None, :]) // denom

    valid_traj = lengths >= min_remaining_steps
    valid_mask = jnp.broadcast_to(valid_traj[:, None], start_steps.shape)
    return start_steps.astype(jnp.int32), valid_mask


def compute_frame_asi_probs(
    logits: jnp.ndarray,
    valid_mask: jnp.ndarray | None = None,
    uniform_mix: float = 0.1,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Convert ASI logits to valid categorical probabilities.

    Invalid buckets receive zero probability. Rows with no valid bucket fall
    back to a uniform distribution across all buckets.
    """
    logits = logits.astype(jnp.float32)
    if valid_mask is None:
        valid_mask = jnp.ones_like(logits, dtype=bool)
    else:
        valid_mask = valid_mask.astype(bool)

    row_has_valid = jnp.sum(valid_mask, axis=-1, keepdims=True) > 0
    safe_mask = jnp.where(row_has_valid, valid_mask, jnp.ones_like(valid_mask, dtype=bool))
    safe_count = jnp.maximum(jnp.sum(safe_mask, axis=-1, keepdims=True), 1)

    temperature = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), eps)
    masked_logits = jnp.where(safe_mask, logits / temperature, -1.0e9)
    softmax_probs = jax.nn.softmax(masked_logits, axis=-1)

    floor_probs = safe_mask.astype(jnp.float32) / safe_count.astype(jnp.float32)
    uniform_mix = jnp.clip(jnp.asarray(uniform_mix, dtype=jnp.float32), 0.0, 1.0)
    probs = (1.0 - uniform_mix) * softmax_probs + uniform_mix * floor_probs
    probs = jnp.where(safe_mask, probs, 0.0)
    return probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), eps)


def update_frame_asi_state(
    state: FrameASIState,
    probs: jnp.ndarray,
    init_traj_ids: jnp.ndarray,
    init_bucket_ids: jnp.ndarray,
    scores: jnp.ndarray,
    done_mask: jnp.ndarray,
    valid_mask: jnp.ndarray | None = None,
    alpha: float = 0.01,
    baseline_beta: float = 0.1,
    logit_clip: float = 5.0,
) -> FrameASIState:
    """Apply a REINFORCE-style ASI update from completed episodes."""
    n_trajectories, num_buckets = state.logits.shape
    num_segments = n_trajectories * num_buckets

    traj_ids = init_traj_ids.reshape(-1).astype(jnp.int32)
    bucket_ids = init_bucket_ids.reshape(-1).astype(jnp.int32)
    scores = scores.reshape(-1).astype(jnp.float32)
    done_mask = done_mask.reshape(-1).astype(bool)

    in_bounds = (
        (traj_ids >= 0)
        & (traj_ids < n_trajectories)
        & (bucket_ids >= 0)
        & (bucket_ids < num_buckets)
    )
    safe_traj_ids = jnp.clip(traj_ids, 0, n_trajectories - 1)
    safe_bucket_ids = jnp.clip(bucket_ids, 0, num_buckets - 1)

    if valid_mask is None:
        sample_valid = jnp.ones_like(done_mask, dtype=bool)
        update_mask = jnp.ones_like(state.logits, dtype=bool)
    else:
        update_mask = valid_mask.astype(bool)
        sample_valid = update_mask[safe_traj_ids, safe_bucket_ids]

    active = done_mask & in_bounds & sample_valid
    active_f = active.astype(jnp.float32)
    flat_ids = safe_traj_ids * num_buckets + safe_bucket_ids

    baseline_flat = state.baseline.reshape(-1)
    advantages = (scores - baseline_flat[flat_ids]) * active_f

    one_hot_adv = jax.ops.segment_sum(advantages, flat_ids, num_segments).reshape(
        n_trajectories, num_buckets
    )
    adv_by_traj = jax.ops.segment_sum(advantages, safe_traj_ids, n_trajectories)
    counts_by_traj = jax.ops.segment_sum(active_f, safe_traj_ids, n_trajectories)
    denom_by_traj = jnp.maximum(counts_by_traj, 1.0)

    grad = one_hot_adv - probs * adv_by_traj[:, None]
    grad = grad / denom_by_traj[:, None]
    grad = jnp.where(update_mask, grad, 0.0)

    new_logits = state.logits + jnp.asarray(alpha, dtype=jnp.float32) * grad
    logit_clip = jnp.asarray(logit_clip, dtype=jnp.float32)
    new_logits = jnp.clip(new_logits, -logit_clip, logit_clip)

    bucket_counts = jax.ops.segment_sum(active_f, flat_ids, num_segments).reshape(
        n_trajectories, num_buckets
    )
    score_sums = jax.ops.segment_sum(scores * active_f, flat_ids, num_segments).reshape(
        n_trajectories, num_buckets
    )
    mean_scores = score_sums / jnp.maximum(bucket_counts, 1.0)
    baseline_beta = jnp.clip(jnp.asarray(baseline_beta, dtype=jnp.float32), 0.0, 1.0)
    new_baseline = jnp.where(
        bucket_counts > 0,
        (1.0 - baseline_beta) * state.baseline + baseline_beta * mean_scores,
        state.baseline,
    )

    return FrameASIState(logits=new_logits, baseline=new_baseline)
