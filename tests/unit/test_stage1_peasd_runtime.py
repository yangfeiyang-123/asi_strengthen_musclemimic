"""Focused runtime, fixed-budget, and canonical-evaluator contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from musclemimic.physiology.emg_anchor import (
    EmgAnchorSpec,
    emg_synergy_metrics,
)
from musclemimic.physiology.emg_consistency_runtime import curriculum_weight
from musclemimic.algorithms.common.networks import RunningMeanStd
from musclemimic.algorithms.ppo import runner as ppo_runner
from musclemimic.algorithms.ppo.runner import (
    _apply_frozen_eval_policy,
    _requires_stage1_endpoint_review_set,
    _run_strict_promotion_validation,
)
from musclemimic.runner.engine import bind_stage1_peasd_fixed_budget_contract
from musclemimic.runner.stage1_peasd_validation import (
    STAGE1_PEASD_VALIDATION_METRIC_KEYS,
    STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS,
    _validate_training_treatment_samples,
    build_stage1_peasd_training_treatment_sample,
    stage1_peasd_training_treatment_targets,
)
from scripts.evaluate_stage1_peasd import _t0_diagnostics_config

ROOT = Path(__file__).resolve().parents[2]


def _clear_t0_config():
    return OmegaConf.create(
        {
            "experiment": {
                "training_action": "forehandClear_standard",
                "stage1_peasd": {
                    "schema_version": "stage1_peasd_lite_matched_arm_v1",
                    "arm": "T0",
                    "action_id": "forehandClear_standard",
                    "canonical_seeds": [0, 1, 2],
                    "fresh_optimizer_required": True,
                    "parent_initialization_checkpoint": None,
                },
                "seeds": [0],
                "n_seeds": 1,
                "auto_resume": False,
                "resume_from": None,
                "reset_optimizer_on_resume": True,
                "promotion": {"auto_stop": False},
                "total_timesteps": 320_000_000,
                "num_envs": 256,
                "ppo_config": {"num_steps": 80},
                "save_checkpoints": True,
                "checkpoints_on_validation": True,
                "validation": {
                    "active": True,
                    "deterministic": True,
                    "start_from_beginning": True,
                    "num": 32,
                },
                "env_params": {
                    "reward_params": {
                        "emg_consistency": {
                            "enabled": False,
                            "mode": "off",
                            "arm": "T0",
                            "action_id": "forehand_high_clear",
                            "reference_cache": None,
                            "mapping_path": None,
                            "anchor_weight_max": 0.0,
                            "synergy_weight_max": 0.0,
                            "start_update": 1000,
                            "ramp_updates": 4000,
                            "tube_kappa": 1.0,
                            "huber_delta": 1.0,
                            "synergy_phase_shuffle_offset_bins": 0,
                        }
                    }
                },
            }
        }
    )


def test_real_clear_budget_records_scheduled_count_and_independent_endpoint() -> None:
    config = _clear_t0_config()
    contract = bind_stage1_peasd_fixed_budget_contract(config)

    assert contract["num_updates"] == 15_625
    assert contract["validation_interval_updates"] == 488
    assert contract["scheduled_validation_count"] == 32
    assert contract["endpoint_requires_independent_validation"] is True
    assert contract["expected_history_count"] == 33
    assert contract["expected_endpoint_update_number"] == 15_625
    assert contract["expected_endpoint_global_timestep"] == 320_000_000


@pytest.mark.parametrize(
    ("field", "value"),
    (("resume_lr_override", 1e-4), ("extend_completed_run", True)),
)
def test_fixed_budget_refuses_resume_and_extension_side_channels(field, value) -> None:
    config = _clear_t0_config()
    config.experiment[field] = value
    with pytest.raises(ValueError, match="no parent/resume/extension"):
        bind_stage1_peasd_fixed_budget_contract(config)


def test_t0_posthoc_config_preserves_loss_spec_but_never_enables_reward(tmp_path) -> None:
    tube = tmp_path / "tube"
    tube.mkdir()
    config = _clear_t0_config()
    bind_stage1_peasd_fixed_budget_contract(config)

    diagnostic = _t0_diagnostics_config(
        config.experiment,
        reference_cache=tube,
    )

    assert diagnostic["enabled"] is True
    assert diagnostic["mode"] == "diagnostics_only"
    assert diagnostic["arm"] == "T0"
    assert diagnostic["action_id"] == "forehand_high_clear"
    assert diagnostic["anchor_weight_max"] == 0.0
    assert diagnostic["synergy_weight_max"] == 0.0
    assert diagnostic["tube_kappa"] == 1.0
    assert diagnostic["huber_delta"] == 1.0
    assert diagnostic["start_update"] == 1000
    assert diagnostic["ramp_updates"] == 4000


def test_t4_shifted_reward_can_be_good_while_real_reference_is_bad() -> None:
    # One measured channel and one coefficient are enough to make the
    # scientific distinction explicit.  At phase bin 0 the real reference is
    # low; T4's half-cycle bin 2 reference matches the simulated activation.
    spec = EmgAnchorSpec(
        projection=jnp.asarray([[1.0]], dtype=jnp.float32),
        activation_addresses=jnp.asarray([0], dtype=jnp.int32),
        synergy_projection=jnp.asarray([[1.0]], dtype=jnp.float32),
        anchor_mean=jnp.zeros((1, 4, 1), dtype=jnp.float32),
        anchor_scale=jnp.ones((1, 4, 1), dtype=jnp.float32),
        anchor_valid=jnp.ones((1, 4, 1), dtype=jnp.float32),
        amplitude_confidence=jnp.ones((1, 1), dtype=jnp.float32),
        synergy_mean=jnp.asarray([[[0.1], [0.2], [0.9], [0.3]]], dtype=jnp.float32),
        synergy_scale=jnp.full((1, 4, 1), 0.01, dtype=jnp.float32),
        synergy_valid=jnp.ones((1, 4, 1), dtype=jnp.float32),
        phase_bin_count=4,
    )
    activation = jnp.asarray([0.9], dtype=jnp.float32)
    shifted = emg_synergy_metrics(
        activation,
        spec,
        action_index=0,
        phase=0.0,
        phase_bin_offset=2,
    )
    real = emg_synergy_metrics(
        activation,
        spec,
        action_index=0,
        phase=0.0,
        phase_bin_offset=0,
    )

    assert float(shifted.loss) < float(real.loss)
    assert float(real.intensity_loss) > 0.0


def test_emg_curriculum_and_metric_contract_are_jit_safe_and_complete() -> None:
    fn = jax.jit(
        lambda update: curriculum_weight(
            update,
            maximum=0.05,
            start_update=1000,
            ramp_updates=4000,
            backend=jnp,
        )
    )
    assert float(fn(jnp.asarray(999))) == pytest.approx(0.0)
    assert float(fn(jnp.asarray(3000))) == pytest.approx(0.025)
    assert float(fn(jnp.asarray(5000))) == pytest.approx(0.05)
    assert "val_activation_saturation_fraction" in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    assert "val_emg_anchor_correlation" in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    assert "val_emg_synergy_real_reference_loss" in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    assert "val_emg_curriculum_factor_anchor" in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    assert (
        "val_penalty_emg_consistency_effective_after_total_clip"
        in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    )
    assert (
        "val_penalty_emg_consistency_effective_after_reward_floor"
        in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    )


def test_strict_eval_freezes_run_stats_and_is_padding_batch_independent() -> None:
    normalizer = RunningMeanStd()
    variables = normalizer.init(jax.random.key(0), jnp.zeros((1, 3), dtype=jnp.float32))
    real_observation = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    padded_observations = jnp.concatenate(
        [
            real_observation,
            jnp.asarray([[99.0, -99.0, 50.0], [-50.0, 20.0, 80.0]], dtype=jnp.float32),
        ],
        axis=0,
    )

    single = _apply_frozen_eval_policy(
        normalizer,
        variables.get("params", {}),
        variables["run_stats"],
        real_observation,
    )
    padded = _apply_frozen_eval_policy(
        normalizer,
        variables.get("params", {}),
        variables["run_stats"],
        padded_observations,
    )

    assert jnp.allclose(single, padded[0])
    assert variables["run_stats"]["count"] == pytest.approx(1.000001)


def test_strict_eval_clamps_vector_width_to_heldout_trajectory_count(monkeypatch) -> None:
    calls = []

    def fake_run_validation_all(**kwargs):
        calls.append(kwargs)
        return {"val_mean_episode_return": 1.0}

    monkeypatch.setattr(ppo_runner, "_run_validation_all", fake_run_validation_all)
    trajectory_handler = SimpleNamespace(
        n_trajectories=3,
        len_trajectory=lambda _index: 8,
    )
    env = SimpleNamespace(
        th=trajectory_handler,
        info=SimpleNamespace(horizon=8),
        _mdp_info=SimpleNamespace(horizon=8),
    )
    train_state = SimpleNamespace(params={}, run_stats={}, step=jnp.asarray(0))
    config = OmegaConf.create(
        {
            "validation": {
                "deterministic": True,
                "start_from_beginning": True,
                "num_envs": 20,
            },
            "stage1_peasd": None,
        }
    )

    overprovisioned = _run_strict_promotion_validation(
        network=object(),
        train_state=train_state,
        val_env=env,
        config=config,
        eval_seed=7,
    )
    config.validation.num_envs = 3
    exact = _run_strict_promotion_validation(
        network=object(),
        train_state=train_state,
        val_env=env,
        config=config,
        eval_seed=7,
    )

    assert overprovisioned == exact
    assert [call["num_envs"] for call in calls] == [3, 3]
    assert all(call["normalize_env"] is False for call in calls)


def test_training_treatment_samples_bind_actual_curriculum_rollouts() -> None:
    config = _clear_t0_config()
    contract = bind_stage1_peasd_fixed_budget_contract(config)
    targets = stage1_peasd_training_treatment_targets(config.experiment)
    assert [target["rollout_update_index"] for target in targets] == [0, 1000, 5000, 15624]

    zero_metrics = {key: 0.0 for key in STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS}
    samples = [
        build_stage1_peasd_training_treatment_sample(
            experiment=config.experiment,
            rollout_update_index=int(target["rollout_update_index"]),
            training_metrics=zero_metrics,
        )
        for target in targets
    ]
    validated = _validate_training_treatment_samples(
        samples,
        experiment=config.experiment,
        completed_updates=int(contract["num_updates"]),
    )
    assert validated[-1]["roles"] == ["endpoint_rollout"]

    tampered = [dict(sample) for sample in samples]
    tampered[-1] = {
        **tampered[-1],
        "metrics": {**tampered[-1]["metrics"], "emg_anchor_weight": 0.01},
    }
    with pytest.raises(ValueError, match="delivered emg_anchor_weight"):
        _validate_training_treatment_samples(
            tampered,
            experiment=config.experiment,
            completed_updates=int(contract["num_updates"]),
        )


@pytest.mark.parametrize(
    ("arm", "anchor_max", "synergy_max"),
    (
        ("T0", 0.0, 0.0),
        ("T1", 0.03, 0.0),
        ("T2", 0.0, 0.02),
        ("T3", 0.03, 0.02),
        ("T4", 0.03, 0.02),
    ),
)
def test_all_arms_delivered_treatment_samples_cover_zero_based_schedule(
    arm, anchor_max, synergy_max
) -> None:
    config = _clear_t0_config()
    config.experiment.stage1_peasd.arm = arm
    reward = config.experiment.env_params.reward_params.emg_consistency
    reward.arm = arm
    reward.enabled = arm != "T0"
    reward.mode = "off" if arm == "T0" else "reward"
    reward.anchor_weight_max = anchor_max
    reward.synergy_weight_max = synergy_max
    if arm == "T3":
        config.experiment.validation.update(
            {
                "visual_review_kind": "stage1_body",
                "cycle_video_trajectories": True,
                "video_length": 400,
            }
        )
    contract = bind_stage1_peasd_fixed_budget_contract(config)
    targets = stage1_peasd_training_treatment_targets(config.experiment)
    samples = []
    for target in targets:
        update_index = int(target["rollout_update_index"])
        anchor_weight = float(
            curriculum_weight(
                update_index,
                maximum=anchor_max,
                start_update=1000,
                ramp_updates=4000,
                backend=jnp,
            )
        )
        synergy_weight = float(
            curriculum_weight(
                update_index,
                maximum=synergy_max,
                start_update=1000,
                ramp_updates=4000,
                backend=jnp,
            )
        )
        active = anchor_weight > 0.0 or synergy_weight > 0.0
        metrics = {key: 0.0 for key in STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS}
        metrics.update(
            {
                "emg_anchor_weight": anchor_weight,
                "emg_synergy_weight": synergy_weight,
                "emg_curriculum_factor_anchor": (
                    anchor_weight / anchor_max if anchor_max else 0.0
                ),
                "emg_curriculum_factor_synergy": (
                    synergy_weight / synergy_max if synergy_max else 0.0
                ),
                "penalty_emg_anchor_raw": -0.5 * anchor_weight,
                "penalty_emg_anchor_after_local_clip": -0.5 * anchor_weight,
                "penalty_emg_synergy_raw": -0.5 * synergy_weight,
                "penalty_emg_synergy_after_local_clip": -0.5 * synergy_weight,
                "penalty_emg_consistency_after_local_clip": -0.5
                * (anchor_weight + synergy_weight),
                "penalty_emg_consistency_effective_after_total_clip": -0.4
                * (anchor_weight + synergy_weight),
                "penalty_emg_consistency_effective_after_reward_floor": -0.3
                * (anchor_weight + synergy_weight),
                "emg_consistency_penalty_masked_fraction": 0.2 if active else 0.0,
                "emg_consistency_final_reward_masked_fraction": 0.4 if active else 0.0,
                "emg_anchor_valid_channel_fraction": 1.0 if arm != "T0" else 0.0,
                "emg_synergy_real_reference_intensity": 0.5 if arm != "T0" else 0.0,
            }
        )
        samples.append(
            build_stage1_peasd_training_treatment_sample(
                experiment=config.experiment,
                rollout_update_index=update_index,
                training_metrics=metrics,
            )
        )

    validated = _validate_training_treatment_samples(
        samples,
        experiment=config.experiment,
        completed_updates=int(contract["num_updates"]),
    )
    assert [sample["rollout_update_index"] for sample in validated] == [
        0,
        1000,
        5000,
        15624,
    ]
    assert validated[-1]["metrics"]["emg_anchor_weight"] == pytest.approx(anchor_max)
    assert validated[-1]["metrics"]["emg_synergy_weight"] == pytest.approx(synergy_max)
    if arm == "T3":
        erased = [dict(sample) for sample in samples]
        erased[-1] = {
            **erased[-1],
            "metrics": {
                **erased[-1]["metrics"],
                "emg_consistency_final_reward_masked_fraction": 1.0,
            },
        }
        with pytest.raises(ValueError, match="erased by the final reward floor"):
            _validate_training_treatment_samples(
                erased,
                experiment=config.experiment,
                completed_updates=int(contract["num_updates"]),
            )


def test_pre_registered_t3_seed0_requires_exact_endpoint_review_set() -> None:
    assert _requires_stage1_endpoint_review_set(
        OmegaConf.create({"stage1_peasd": {"arm": "T3"}, "seeds": [0]})
    )
    assert not _requires_stage1_endpoint_review_set(
        OmegaConf.create({"stage1_peasd": {"arm": "T3"}, "seeds": [1]})
    )
    assert not _requires_stage1_endpoint_review_set(
        OmegaConf.create({"stage1_peasd": {"arm": "T4"}, "seeds": [0]})
    )


def test_canonical_launcher_routes_stage1_eval_without_starting_training(tmp_path) -> None:
    log_path = tmp_path / "eval.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUSCLEMIMIC_JAX_CACHE_KEY": "stage1_peasd_eval_test",
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache"),
            "MUSCLEMIMIC_TRAIN_LOG": str(log_path),
            "MUSCLEMIMIC_DRY_RUN": "1",
        }
    )
    completed = subprocess.run(
        [
            str(ROOT / "scripts/run_fullbody_training.sh"),
            "--stage1-peasd-eval",
            "--checkpoint",
            str(tmp_path / "checkpoint_15625"),
            "--reference-cache",
            str(tmp_path / "tube"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout + completed.stderr
    assert "mode=stage1-peasd-eval" in output
    assert "workload=read-only-evaluation (training is disabled)" in output
    assert "scripts/evaluate_stage1_peasd.py" in output
    assert "evaluation was not started and training is disabled" in output
    assert "fullbody/experiment.py" not in output


@pytest.mark.parametrize("gpu_value", ["0,1", "gpu0", "-1", " 0"])
def test_canonical_launcher_requires_one_numeric_physical_gpu(
    tmp_path,
    gpu_value: str,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_value,
            "MUSCLEMIMIC_JAX_CACHE_KEY": "invalid_gpu_contract_test",
            "MUSCLEMIMIC_TRAIN_LOG": str(tmp_path / "invalid-gpu.log"),
            "MUSCLEMIMIC_DRY_RUN": "1",
        }
    )
    completed = subprocess.run(
        [
            str(ROOT / "scripts/run_fullbody_training.sh"),
            "--config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "must be one non-negative physical GPU index" in completed.stderr
