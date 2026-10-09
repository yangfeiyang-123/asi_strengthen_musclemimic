from __future__ import annotations

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from musclemimic.runner import engine
from musclemimic.runner.logging import ExperimentHooks


class _FakeRun:
    def __init__(self):
        self.metric_definitions: list[tuple[str, dict]] = []

    def define_metric(self, name: str, **kwargs):
        self.metric_definitions.append((name, kwargs))


class _FakeWandbModule:
    def __init__(self):
        self.run = _FakeRun()
        self.init_kwargs = None
        self.logs = []

    def login(self):
        return True

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def Video(self, path, *, format):
        return {"path": path, "format": format}

    def log(self, payload, *, step):
        self.logs.append((dict(payload), int(step)))


def test_setup_wandb_uses_current_timestep_as_default_step_metric(monkeypatch, tmp_path):
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = OmegaConf.create(
        {
            "wandb": {
                "mode": "online",
                "project": "musclemimic",
                "dir": str(tmp_path / "wandb"),
            }
        }
    )

    enabled, run = engine.setup_wandb(config)

    assert enabled is True
    assert run is fake_wandb.run
    assert fake_wandb.run.metric_definitions == [
        ("Current Timestep", {}),
        ("*", {"step_metric": "Current Timestep", "step_sync": True}),
    ]


def test_validation_video_carries_current_timestep():
    fake_wandb = _FakeWandbModule()

    ExperimentHooks().on_validation_video(
        use_wandb=True,
        wandb=fake_wandb,
        video_path="/tmp/validation.mp4",
        timestep=359_792_640,
    )

    payload, step = fake_wandb.logs[-1]
    assert step == 359_792_640
    assert payload["Current Timestep"] == 359_792_640
    assert payload["Validation/Video"] == {
        "path": "/tmp/validation.mp4",
        "format": "mp4",
    }
