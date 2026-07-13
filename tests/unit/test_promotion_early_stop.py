from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from musclemimic.badminton.promotion_early_stop import (
    load_promotion_progress,
    record_validation,
    resolve_promotion_early_stop,
    validation_chunk_length,
)
from musclemimic.algorithms.ppo.runner import (
    _balanced_validation_trajectory_indices,
    _run_strict_promotion_validation,
)
from musclemimic.badminton.training_gates import CANONICAL_PROMOTION_THRESHOLDS
from musclemimic.runner.engine import run_training


def _config(tmp_path, *, stage="stage1", baseline=None, n_seeds=1):
    promotion = {
        "stage": stage,
        "auto_stop": True,
        "consecutive_validations": 3,
        "progress_path": None,
        "baseline_metrics_path": baseline,
        **dict(CANONICAL_PROMOTION_THRESHOLDS[stage]),
    }
    return OmegaConf.create(
        {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "n_seeds": n_seeds,
            "save_checkpoints": True,
            "checkpoints_on_validation": True,
            "validation": {
                "active": True,
                "cover_all_trajectories": True,
                "deterministic": True,
                "start_from_beginning": True,
                "num_envs": 5,
                "amass_dataset_conf": {
                    "rel_dataset_path": [f"motion-{index}" for index in range(5)]
                },
            },
            "promotion": promotion,
        }
    )


def _stage1_metrics(*, passing=True):
    return {
        "val_early_termination_rate": 0.01 if passing else 0.20,
        "val_frame_coverage": 0.99,
        "val_err_rpos": 0.05,
        "val_action_saturation_fraction": 0.01,
        "val_activation_energy": 0.20,
    }


def _record(settings, progress, *, metrics, update, global_timestep):
    return record_validation(
        settings,
        progress,
        metrics=metrics,
        update_number=update,
        global_timestep=global_timestep,
        checkpoint_identity={
            "update_number": update,
            "global_timestep": global_timestep,
            "config_hash": settings.config_hash,
        },
        validation_provenance={
            "semantics": "evaluate_all_once_per_heldout_v1"
        },
    )


def test_validation_chunk_length_stops_exactly_on_boundaries():
    assert validation_chunk_length(0, 100, 32) == 32
    assert validation_chunk_length(30, 100, 32) == 2
    assert validation_chunk_length(32, 7, 32) == 7
    assert validation_chunk_length(32, 0, 32) == 0
    with pytest.raises(ValueError, match="positive"):
        validation_chunk_length(0, 1, 0)


def test_balanced_validation_indices_cover_every_heldout_motion():
    indices = _balanced_validation_trajectory_indices(20, 5)
    assert indices.tolist() == [0, 1, 2, 3, 4] * 4
    with pytest.raises(ValueError, match="must be >="):
        _balanced_validation_trajectory_indices(4, 5)


def test_progress_persists_streak_and_rolls_back_to_resumed_checkpoint(tmp_path):
    settings = resolve_promotion_early_stop(_config(tmp_path))
    progress = load_promotion_progress(settings, checkpoint_update=0)

    for update in (10, 20, 30):
        progress = _record(
            settings, progress, metrics=_stage1_metrics(), update=update,
            global_timestep=update * 100,
        )

    assert progress["stopped_early"] is True
    assert progress["consecutive_pass_streak"] == 3
    persisted = json.loads(settings.progress_path.read_text())
    assert persisted["stop_reason"] == "consecutive_promotion_pass"
    assert persisted["validations"] == [event["metrics"] for event in persisted["history"]]

    at_second_checkpoint = load_promotion_progress(settings, checkpoint_update=20)
    assert at_second_checkpoint["consecutive_pass_streak"] == 2
    assert at_second_checkpoint["stopped_early"] is False

    at_stop_checkpoint = load_promotion_progress(settings, checkpoint_update=30)
    assert at_stop_checkpoint["consecutive_pass_streak"] == 3
    assert at_stop_checkpoint["stopped_early"] is True


def test_failed_validation_resets_consecutive_streak(tmp_path):
    settings = resolve_promotion_early_stop(_config(tmp_path))
    progress = load_promotion_progress(settings, checkpoint_update=0)
    for update, passing in ((10, True), (20, True), (30, False), (40, True)):
        progress = _record(
            settings, progress, metrics=_stage1_metrics(passing=passing),
            update=update, global_timestep=update * 100,
        )
    assert progress["consecutive_pass_streak"] == 1
    assert progress["stopped_early"] is False


def test_minimum_validation_count_delays_stop_even_after_three_passes(tmp_path):
    cfg = _config(tmp_path)
    cfg.promotion.min_validations_before_stop = 5
    settings = resolve_promotion_early_stop(cfg)
    progress = load_promotion_progress(settings, checkpoint_update=0)
    for update in range(1, 6):
        progress = _record(
            settings, progress, metrics=_stage1_metrics(), update=update,
            global_timestep=update * 100,
        )
        assert progress["stopped_early"] is (update >= 5)


def test_stage2_online_gate_uses_required_stage1_baseline(tmp_path):
    baseline = tmp_path / "stage1.json"
    baseline.write_text(
        json.dumps(
            {
                "history": [
                    {
                        "update_number": 10,
                        "global_timestep": 1000,
                        "metrics": {"val_err_rpos": 0.10},
                    }
                ]
            }
        )
    )
    settings = resolve_promotion_early_stop(
        _config(tmp_path, stage="stage2", baseline=str(baseline))
    )
    progress = load_promotion_progress(settings, checkpoint_update=0)
    metrics = {
        "val_early_termination_rate": 0.01,
        "val_frame_coverage": 0.99,
        "val_err_racket_pos": 0.02,
        "val_err_racket_rot": 0.10,
        "val_err_rpos": 0.105,
    }
    progress = _record(
        settings, progress, metrics=metrics, update=10, global_timestep=1000,
    )
    assert progress["history"][-1]["passed"] is True
    degradation = progress["history"][-1]["gate_report"]["evaluations"][0][-1]
    assert degradation["name"] == "body_metric_relative_degradation"
    assert degradation["value"] == pytest.approx(0.05)

    with pytest.raises(ValueError, match="baseline_metrics_path"):
        resolve_promotion_early_stop(_config(tmp_path, stage="stage2"))


def test_stage2_progress_rejects_baseline_content_mutation(tmp_path):
    baseline = tmp_path / "stage1.json"
    baseline.write_text(
        json.dumps({"validations": [{"val_err_rpos": 0.10}]}),
        encoding="utf-8",
    )
    settings = resolve_promotion_early_stop(
        _config(tmp_path, stage="stage2", baseline=str(baseline))
    )
    progress = load_promotion_progress(settings, checkpoint_update=0)
    assert progress["baseline_metrics_sha256"] == settings.baseline_metrics_sha256

    baseline.write_text(
        json.dumps({"validations": [{"val_err_rpos": 0.20}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline metrics changed"):
        load_promotion_progress(settings, checkpoint_update=0)
    with pytest.raises(ValueError, match="baseline metrics changed"):
        _record(
            settings,
            progress,
            metrics={
                "val_early_termination_rate": 0.01,
                "val_frame_coverage": 0.99,
                "val_err_racket_pos": 0.02,
                "val_err_racket_rot": 0.10,
                "val_err_rpos": 0.10,
            },
            update=10,
            global_timestep=1000,
        )


def test_auto_stop_rejects_multiple_seeds_and_missing_checkpointing(tmp_path):
    with pytest.raises(ValueError, match="single seed"):
        resolve_promotion_early_stop(_config(tmp_path, n_seeds=2))
    cfg = _config(tmp_path)
    cfg.save_checkpoints = False
    with pytest.raises(ValueError, match="checkpointing"):
        resolve_promotion_early_stop(cfg)


def test_auto_stop_rejects_missing_or_drifted_yaml_threshold(tmp_path):
    missing = _config(tmp_path)
    del missing.promotion.max_activation_energy
    with pytest.raises(ValueError, match="missing canonical threshold"):
        resolve_promotion_early_stop(missing)

    drifted = _config(tmp_path)
    drifted.promotion.max_relative_site_position_error_m = 0.10
    with pytest.raises(ValueError, match="threshold drift"):
        resolve_promotion_early_stop(drifted)


def test_engine_host_control_path_does_not_apply_outer_jit(monkeypatch):
    monkeypatch.setattr(
        "musclemimic.runner.engine.jax.jit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("outer jit called")),
    )
    key = jnp.asarray([0, 1], dtype=jnp.uint32)
    assert run_training(lambda value: ("ran", value), key, host_controlled=True)[0] == "ran"


def test_online_promotion_calls_strict_evaluate_all_once_semantics(monkeypatch):
    class TrajectoryHandler:
        n_trajectories = 5

        @staticmethod
        def len_trajectory(index):
            return 100 + index

    val_env = SimpleNamespace(
        th=TrajectoryHandler(),
        info=SimpleNamespace(horizon=50),
        _mdp_info=SimpleNamespace(horizon=50),
    )
    seen = {}

    def fake_evaluate_all(**kwargs):
        seen.update(kwargs)
        return {"val_frame_coverage": 1.0}

    monkeypatch.setattr(
        "musclemimic.algorithms.ppo.runner._run_validation_all",
        fake_evaluate_all,
    )
    result = _run_strict_promotion_validation(
        network="network",
        train_state=SimpleNamespace(params="params", run_stats="stats"),
        val_env=val_env,
        config={
            "validation": {
                "deterministic": True,
                "start_from_beginning": True,
                "num_envs": 5,
            }
        },
        eval_seed=17,
    )

    assert result == {"val_frame_coverage": 1.0}
    assert seen["traj_env"] is seen["val_env"] is val_env
    assert seen["deterministic"] is True
    assert seen["num_envs"] == 5
    assert val_env.info.horizon == 50
    assert val_env._mdp_info.horizon == 104
