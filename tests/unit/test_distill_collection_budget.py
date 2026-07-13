from __future__ import annotations

import pytest

from musclemimic.distill.collection_budget import resolve_collection_budget


def test_total_transition_budget_ceil_steps_and_requires_final_trim():
    budget = resolve_collection_budget(
        num_envs=256,
        num_transitions=1_000_001,
        num_steps=None,
        default_transitions=1_000_000,
    )

    assert budget.vector_steps == 3907
    assert budget.requested_transitions == 1_000_001
    assert budget.planned_transitions_before_trim == 1_000_192
    assert budget.planned_transitions_before_trim - budget.requested_transitions == 191
    assert budget.legacy_num_steps is None


def test_legacy_vector_steps_are_explicitly_expanded_to_samples():
    budget = resolve_collection_budget(
        num_envs=256,
        num_transitions=None,
        num_steps=200_000,
        default_transitions=1_000_000,
    )

    assert budget.vector_steps == 200_000
    assert budget.requested_transitions == 51_200_000
    assert budget.legacy_num_steps == 200_000


def test_budget_rejects_ambiguous_or_nonpositive_inputs():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_collection_budget(
            num_envs=8,
            num_transitions=100,
            num_steps=10,
            default_transitions=100,
        )
    with pytest.raises(ValueError, match="positive"):
        resolve_collection_budget(
            num_envs=8,
            num_transitions=0,
            num_steps=None,
            default_transitions=100,
        )
