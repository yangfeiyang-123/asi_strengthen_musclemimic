"""Shared utilities for checkpoint state calculations.

Single source of truth for converting between:
- Optimizer steps (train_state.step)
- Update numbers (rollout iterations)
- Global timesteps (environment steps)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import serialization


@dataclass
class TrainingConfig:
    """Training configuration snapshot for checkpoints."""

    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int

    @classmethod
    def from_experiment_config(cls, exp_config):
        """Extract from OmegaConf experiment config."""
        return cls(
            num_envs=int(exp_config.num_envs),
            num_steps=int(exp_config.num_steps),
            num_minibatches=int(exp_config.num_minibatches),
            update_epochs=int(exp_config.update_epochs),
        )


def optimizer_step_to_update(optimizer_step: int, num_minibatches: int, update_epochs: int) -> int:
    """Convert optimizer step to update number.

    This is the canonical formula used throughout the codebase.
    Matches the training loop counter calculation.

    Args:
        optimizer_step: Current optimizer step (train_state.step)
        num_minibatches: Number of minibatches per update
        update_epochs: Number of epochs per update

    Returns:
        Update number (rollout iteration count)
    """
    return ((optimizer_step + 1) // num_minibatches) // update_epochs


def update_to_global_timestep(update_number: int, num_steps: int, num_envs: int) -> int:
    """Convert update number to global environment timesteps.

    Args:
        update_number: Update/rollout iteration number
        num_steps: Steps per rollout
        num_envs: Parallel environments

    Returns:
        Total environment steps across all environments
    """
    return update_number * num_steps * num_envs


def compute_checkpoint_metadata(optimizer_step: int, config: TrainingConfig, learning_rate: float) -> dict:
    """Compute all checkpoint metadata from optimizer step and config.

    Single source of truth for metadata calculation.

    Args:
        optimizer_step: Current optimizer step
        config: Training configuration
        learning_rate: Current learning rate

    Returns:
        Dictionary with all metadata fields
    """
    update_num = optimizer_step_to_update(optimizer_step, config.num_minibatches, config.update_epochs)
    global_ts = update_to_global_timestep(update_num, config.num_steps, config.num_envs)

    return {
        "step": optimizer_step,
        "update_number": update_num,
        "global_timestep": global_ts,
        "learning_rate": learning_rate,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "num_minibatches": config.num_minibatches,
        "update_epochs": config.update_epochs,
    }


def compute_resume_state(
    checkpoint_metadata: dict, current_config: TrainingConfig, total_timesteps: int
) -> tuple[int, int, bool]:
    """Compute resume state from checkpoint metadata and current config.

    Args:
        checkpoint_metadata: Loaded checkpoint metadata
        current_config: Current training configuration
        total_timesteps: Total timesteps for this training run

    Returns:
        Tuple of (completed_updates, remaining_updates, config_changed)

    Raises:
        ValueError: If checkpoint metadata is missing required fields
    """
    # Extract checkpoint state (strict - no defaults)
    try:
        completed_updates = int(checkpoint_metadata["update_number"])
        global_ts = int(checkpoint_metadata["global_timestep"])
        ckpt_num_envs = int(checkpoint_metadata["num_envs"])
        ckpt_num_steps = int(checkpoint_metadata["num_steps"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"Checkpoint missing required metadata fields. "
            f"Please create a new checkpoint with the updated format. Error: {e}"
        ) from e

    # Detect config change
    config_changed = ckpt_num_envs != current_config.num_envs or ckpt_num_steps != current_config.num_steps

    # Compute remaining updates with current config
    current_steps_per_update = current_config.num_steps * current_config.num_envs
    remaining_ts = max(total_timesteps - global_ts, 0)
    # Use ceiling division for remaining updates
    remaining_updates = (remaining_ts + current_steps_per_update - 1) // current_steps_per_update

    return completed_updates, remaining_updates, config_changed


def reset_lr_schedule_count(opt_state):
    """Reset ScaleByScheduleState.count to 0 while preserving other optimizer state.

    This allows the LR schedule to start fresh when resuming from a checkpoint,
    while keeping momentum/Adam statistics intact. Useful for finetuning where
    you want a fresh LR schedule but don't want to lose optimizer momentum.

    Args:
        opt_state: The optimizer state pytree (from optax)

    Returns:
        New optimizer state with schedule counters reset to 0
    """

    def reset_if_schedule_state(node):
        if isinstance(node, optax.ScaleByScheduleState):
            return optax.ScaleByScheduleState(count=jnp.zeros_like(node.count))
        return node

    return jax.tree_util.tree_map(
        reset_if_schedule_state,
        opt_state,
        is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
    )


def optimizer_schedule_counts(opt_state) -> tuple[jax.Array, ...]:
    """Return every Optax schedule counter in an optimizer state pytree."""
    counts = []

    def collect_schedule_state(node):
        if isinstance(node, optax.ScaleByScheduleState):
            counts.append(node.count)
        return node

    jax.tree_util.tree_map(
        collect_schedule_state,
        opt_state,
        is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
    )
    return tuple(counts)


def optimizer_schedule_step(opt_state, fallback_step):
    """Return the LR schedule's real step, falling back for fixed-LR optimizers.

    Muon can contain more than one scheduled inner optimizer.  Exact resume
    validates that all of those counters agree, so the first counter is the
    canonical value used for LR logging and checkpoint metadata.
    """
    counts = optimizer_schedule_counts(opt_state)
    return counts[0] if counts else fallback_step


def restore_optimizer_state(fresh_opt_state, loaded_opt_state):
    """Rehydrate a checkpoint optimizer state into the current Optax types.

    Orbax restores checkpoints saved without rich type metadata as nested
    dictionaries/lists.  Those containers hold the moments and schedule
    counters, but their pytree structure does not equal Optax's named-tuple
    structure.  Flax state-dict deserialization reconstructs the exact current
    optimizer types without discarding any checkpoint values.

    Raises:
        ValueError: if the loaded state cannot be mapped exactly onto the
            current optimizer structure.
    """
    fresh_structure = jax.tree_util.tree_structure(fresh_opt_state)
    loaded_structure = jax.tree_util.tree_structure(loaded_opt_state)
    if fresh_structure == loaded_structure:
        restored = loaded_opt_state
    else:
        # Orbax StandardRestore with rich types disabled materializes sequence
        # nodes as Python lists.  Flax's state-dict contract represents those
        # same tuple/list nodes as mappings with stringified integer keys.
        # Normalize containers only; array leaves remain the checkpoint values.
        def normalize_state_dict(target, value):
            # Orbax represents Optax EmptyState() as None.  Only map None to
            # an empty state dict when the current optimizer template proves
            # that this node is structurally empty.
            if value is None:
                target_state = serialization.to_state_dict(target)
                return {} if isinstance(target_state, dict) and not target_state else value

            target_fields = getattr(target, "_fields", None)
            if target_fields is not None and isinstance(value, dict):
                return {
                    key: normalize_state_dict(getattr(target, key), child)
                    for key, child in value.items()
                }

            if isinstance(target, list | tuple):
                if isinstance(value, dict):
                    children = [value[str(index)] for index in range(len(target))]
                elif isinstance(value, list | tuple):
                    children = list(value)
                else:
                    return value
                if len(children) != len(target):
                    raise ValueError(
                        "checkpoint optimizer sequence length does not match current optimizer: "
                        f"checkpoint={len(children)}, current={len(target)}"
                    )
                return {
                    str(index): normalize_state_dict(target[index], child)
                    for index, child in enumerate(children)
                }

            if isinstance(target, dict) and isinstance(value, dict):
                return {
                    key: normalize_state_dict(target[key], child)
                    if key in target else child
                    for key, child in value.items()
                }

            # Generic fallback for containers below custom Optax states.
            if isinstance(value, list | tuple):
                return {
                    str(index): normalize_state_dict(None, child)
                    for index, child in enumerate(value)
                }
            if isinstance(value, dict):
                return {
                    key: normalize_state_dict(None, child)
                    for key, child in value.items()
                }
            return value

        normalized_loaded_state = normalize_state_dict(fresh_opt_state, loaded_opt_state)
        try:
            restored = serialization.from_state_dict(fresh_opt_state, normalized_loaded_state)
        except Exception as exc:
            raise ValueError(
                "checkpoint optimizer state cannot be rehydrated into the current optimizer"
            ) from exc

    restored_structure = jax.tree_util.tree_structure(restored)
    if restored_structure != fresh_structure:
        raise ValueError(
            "rehydrated checkpoint optimizer structure does not match the current optimizer"
        )

    fresh_leaves = jax.tree_util.tree_leaves(fresh_opt_state)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if len(fresh_leaves) != len(restored_leaves):
        raise ValueError("rehydrated checkpoint optimizer has a different leaf count")
    for index, (fresh_leaf, restored_leaf) in enumerate(zip(fresh_leaves, restored_leaves)):
        fresh_shape = getattr(fresh_leaf, "shape", None)
        restored_shape = getattr(restored_leaf, "shape", None)
        if fresh_shape != restored_shape:
            raise ValueError(
                "rehydrated checkpoint optimizer leaf shape mismatch at "
                f"index {index}: checkpoint={restored_shape}, current={fresh_shape}"
            )
    return restored


def validate_optimizer_schedule_counts(opt_state, expected_step: int, *, require_schedule: bool = False) -> tuple[int, ...]:
    """Validate restored schedule counters and return their host integer values.

    A schedule may legitimately lag ``train_state.step``.  TrainState counts
    every attempted gradient application, while ``apply_if_finite`` can skip
    the inner optimizer.  Legacy checkpoints can also retain a fixed offset
    after an intentional/old-style schedule reset.  Exact resume must preserve
    that offset; it must not fabricate a newer schedule count.
    """
    counts = optimizer_schedule_counts(opt_state)
    if require_schedule and not counts:
        raise ValueError("annealed optimizer checkpoint contains no LR schedule counter")

    host_counts = tuple(int(jax.device_get(count)) for count in counts)
    if host_counts and any(count != host_counts[0] for count in host_counts[1:]):
        raise ValueError(f"optimizer LR schedule counters disagree: {host_counts}")
    if host_counts and host_counts[0] > int(expected_step):
        raise ValueError(
            "optimizer LR schedule counter is ahead of train_state.step: "
            f"schedule={host_counts[0]}, train_step={int(expected_step)}"
        )
    return host_counts
