from __future__ import annotations

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from musclemimic.runner.engine import build_logging_callback
from musclemimic.runner.logging import ExperimentHooks


class _FakeWandb:
    def __init__(self):
        self.calls: list[tuple[dict, int]] = []

    def log(self, payload, *, step):
        self.calls.append((dict(payload), int(step)))


def _callback(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = OmegaConf.create(
        {
            "experiment": {
                "algorithm": "PPOJax",
                "reset_logging_timestep": False,
            }
        }
    )
    callback = build_logging_callback(
        env=object(),
        config=config,
        agent_conf=SimpleNamespace(),
        use_wandb=True,
        hooks=ExperimentHooks(),
    )
    return callback, fake_wandb


def test_validation_only_callback_does_not_overwrite_training_metrics(monkeypatch):
    callback, fake_wandb = _callback(monkeypatch)

    callback(
        {
            "has_validation_update": True,
            "max_timestep": 9_994_240,
            "jax_raw_timestep": 9_994_240,
            "val_mean_episode_return": 12.5,
            "val_mean_episode_length": 42.0,
        }
    )

    payload, step = fake_wandb.calls[-1]
    assert step == 9_994_240
    assert payload["Validation/Mean Episode Return"] == 12.5
    assert payload["Validation/Mean Episode Length"] == 42.0
    assert "Mean Episode Return" not in payload
    assert "Mean Episode Length" not in payload
    assert "Learning Rate" not in payload


def test_training_callback_keeps_training_metrics(monkeypatch):
    callback, fake_wandb = _callback(monkeypatch)

    callback(
        {
            "max_timestep": 20_480,
            "jax_raw_timestep": 20_480,
            "mean_episode_return": 3.5,
            "mean_episode_length": 8.0,
            "learning_rate": 2.0e-4,
        }
    )

    payload, step = fake_wandb.calls[-1]
    assert step == 20_480
    assert payload["Mean Episode Return"] == 3.5
    assert payload["Mean Episode Length"] == 8.0
    assert payload["Learning Rate"] == 2.0e-4
