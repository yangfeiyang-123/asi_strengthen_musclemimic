"""Explicit total-transition budgets for vectorized distillation collection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionBudget:
    num_envs: int
    vector_steps: int
    requested_transitions: int
    planned_transitions_before_trim: int
    legacy_num_steps: int | None


def resolve_collection_budget(
    *,
    num_envs: int,
    num_transitions: int | None,
    num_steps: int | None,
    default_transitions: int,
) -> CollectionBudget:
    """Resolve a bounded sample budget while preserving explicit legacy steps.

    ``num_steps`` means vector-environment steps and therefore produces
    ``num_steps * num_envs`` samples.  New production callers should always use
    ``num_transitions``; the last vector batch is trimmed exactly to that total.
    """
    envs = int(num_envs)
    if envs <= 0:
        raise ValueError("num_envs must be positive")
    if num_transitions is not None and num_steps is not None:
        raise ValueError("num_transitions and legacy num_steps are mutually exclusive")
    if num_transitions is None and num_steps is None:
        num_transitions = int(default_transitions)
    if num_transitions is not None:
        transitions = int(num_transitions)
        if transitions <= 0:
            raise ValueError("num_transitions must be positive")
        vector_steps = int(math.ceil(transitions / envs))
        legacy_steps = None
    else:
        assert num_steps is not None
        vector_steps = int(num_steps)
        if vector_steps <= 0:
            raise ValueError("num_steps must be positive")
        transitions = vector_steps * envs
        legacy_steps = vector_steps
    return CollectionBudget(
        num_envs=envs,
        vector_steps=vector_steps,
        requested_transitions=transitions,
        planned_transitions_before_trim=vector_steps * envs,
        legacy_num_steps=legacy_steps,
    )
