from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from musclemimic.runner import engine


class _DummyHooks:
    def build_video_recorder(self, result_dir, config):
        return None


def test_setup_jax_cache_honors_explicit_data_disk_path(monkeypatch, tmp_path):
    cache_dir = tmp_path / "jax-cache"
    updates = []
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(engine.jax.config, "update", lambda key, value: updates.append((key, value)))

    engine.setup_jax_cache()

    expected = str(cache_dir.resolve())
    assert cache_dir.is_dir()
    assert engine.os.environ["JAX_COMPILATION_CACHE_DIR"] == expected
    assert updates == [("jax_compilation_cache_dir", expected)]


def _make_config(*, auto_resume: bool = True, resume_from: str | None = None):
    return OmegaConf.create(
        {
            "wandb": {"mode": "disabled"},
            "experiment": {
                "resume_from": resume_from,
                "auto_resume": auto_resume,
                "run_id": None,
                "checkpoint_root": None,
                "checkpoint_dir": "checkpoints",
                "online_logging_interval": 1,
                "validation": {"active": False},
            },
        }
    )


def _patch_run_experiment_dependencies(monkeypatch, tmp_path: Path, captured: dict):
    runtime = SimpleNamespace(
        output_dir=str(tmp_path / "outputs" / "run1"),
        cwd=str(tmp_path / "launch"),
    )
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(runtime=runtime)))

    monkeypatch.setattr(engine, "setup_jax_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "setup_wandb", lambda _config: (False, None))
    monkeypatch.setattr(engine, "_auto_set_skip_body_data", lambda _config: None)
    monkeypatch.setattr(engine, "instantiate_env", lambda _config: SimpleNamespace(th=None))
    monkeypatch.setattr(engine, "_can_share_trajectory_handler", lambda _config: False)
    monkeypatch.setattr(engine, "instantiate_validation_env", lambda _config, share_trajectory=False: SimpleNamespace())
    monkeypatch.setattr(engine, "_maybe_share_validation_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "pick_algorithm", lambda _config: SimpleNamespace())
    monkeypatch.setattr(
        engine,
        "build_agent_conf",
        lambda _algorithm_cls, _env, config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(engine, "build_metrics_handler", lambda _config, _env: object())
    monkeypatch.setattr(engine, "build_logging_callback", lambda *_args, **_kwargs: (lambda *_a, **_k: None))
    monkeypatch.setattr(engine, "config_hash", lambda _cfg: "hash123")
    monkeypatch.setattr(
        engine,
        "resolve_checkpoint_dir",
        lambda **_kwargs: str(tmp_path / "checkpoints" / "hash123"),
    )
    monkeypatch.setattr(engine, "validate_checkpoint_compatibility", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(engine, "write_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "compute_training_rngs", lambda _config: [0])
    monkeypatch.setattr(
        engine,
        "run_training",
        lambda _train_fn, _rngs, **_kwargs: {"ok": True},
    )

    def fake_resume_or_fresh(
        env,
        agent_conf,
        algorithm_cls,
        config,
        mh,
        logging_callback,
        logging_interval=1,
        val_env=None,
        apply_resume_resets=True,
    ):
        captured["apply_resume_resets"] = apply_resume_resets
        captured["resume_from"] = config.experiment.resume_from
        return lambda _rng: {"ok": True}

    monkeypatch.setattr(engine, "resume_or_fresh", fake_resume_or_fresh)


def test_run_experiment_local_auto_resume_skips_resume_resets(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    latest = tmp_path / "checkpoints" / "hash123" / "checkpoint_42"
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: str(latest))

    config = _make_config(auto_resume=True, resume_from="/external/checkpoint_10")
    engine.run_experiment(config, _DummyHooks())

    assert captured["apply_resume_resets"] is False
    assert captured["resume_from"] == str(latest)


def test_run_experiment_explicit_resume_keeps_resume_resets(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: None)

    explicit = "/external/checkpoint_10"
    config = _make_config(auto_resume=True, resume_from=explicit)
    engine.run_experiment(config, _DummyHooks())

    assert captured["apply_resume_resets"] is True
    assert captured["resume_from"] == explicit


def test_parent_rehash_failure_happens_before_run_manifest_commit(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        engine,
        "bind_explicit_parent_checkpoint",
        lambda *_args, **_kwargs: {"binding_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        engine,
        "validate_explicit_parent_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("explicit parent checkpoint changed")
        ),
    )
    manifest_writes: list[object] = []
    monkeypatch.setattr(
        engine,
        "write_manifest",
        lambda *_args, **_kwargs: manifest_writes.append(True),
    )

    config = _make_config(auto_resume=True, resume_from="/external/checkpoint_10")
    with pytest.raises(ValueError, match="parent checkpoint changed"):
        engine.run_experiment(config, _DummyHooks())

    assert manifest_writes == []


def test_configure_action_training_outputs_sets_dataset_training_dirs(tmp_path):
    config = OmegaConf.create(
        {
            "wandb": {"mode": "disabled", "tags": ["fullbody"]},
            "experiment": {
                "training_action": "backhand_clear",
                "checkpoint_root": "/legacy/checkpoints",
                "validation": {"active": False},
            },
        }
    )

    training_root = engine.configure_action_training_outputs(config, str(tmp_path))

    expected = tmp_path / "datasets" / "backhand_clear" / "training"
    assert training_root == str(expected)
    assert config.experiment.training_root == str(expected)
    assert config.experiment.checkpoint_root == str(expected / "checkpoints")
    assert config.experiment.validation_video_dir == str(expected / "validation_videos")
    assert config.wandb.dir == str(expected / "wandb")
    assert config.wandb.name == "backhand_clear"
    assert "backhand_clear" in list(config.wandb.tags)


def test_run_experiment_passes_action_training_root_to_checkpoint_resolver(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: None)

    def fake_resolve_checkpoint_dir(**kwargs):
        captured["resolve_kwargs"] = kwargs
        return str(tmp_path / "resolved" / "hash123")

    monkeypatch.setattr(engine, "resolve_checkpoint_dir", fake_resolve_checkpoint_dir)

    config = _make_config(auto_resume=True)
    config.experiment.training_action = "backhand_light"
    engine.run_experiment(config, _DummyHooks())

    expected = tmp_path / "launch" / "datasets" / "backhand_light" / "training"
    kwargs = captured["resolve_kwargs"]
    assert kwargs["training_root"] == str(expected)
    assert config.experiment.training_root == str(expected)
    assert config.experiment.checkpoint_root == str(expected / "checkpoints")
