from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.grip.train_right_hand_racket_grip_policy import (  # noqa: E402
    PolicyValueNet,
    _sample_action,
    _tanh_normal_logprob,
)


def test_sample_action_returns_bounded_action_and_finite_logprob():
    rng = np.random.default_rng(0)
    model = PolicyValueNet(obs_size=4, action_size=3, hidden_sizes=(8,), action_std_init=0.35)

    action, logprob, value = _sample_action(torch, model, np.zeros(4, dtype=np.float32), "cpu", rng)

    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert np.isfinite(logprob)
    assert np.isfinite(value)


def test_tanh_normal_logprob_applies_squash_correction():
    mean = torch.zeros((1, 2))
    std = torch.ones((1, 2))
    raw_action = torch.tensor([[0.5, -0.25]])
    squashed_action = torch.tanh(raw_action)
    distribution = torch.distributions.Normal(mean, std)

    corrected = _tanh_normal_logprob(distribution, raw_action, squashed_action, torch)
    uncorrected = distribution.log_prob(raw_action).sum(axis=-1)

    assert torch.isfinite(corrected).all()
    assert corrected.item() > uncorrected.item()
