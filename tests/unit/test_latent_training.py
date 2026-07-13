from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.dataset import write_split_shard
from musclemimic.latent_muscle.action_mask import ActionMask


def _require_latent_deps():
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")


def _write_latent_dataset(path):
    data = {
        "student_obs": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "reference_features": np.array(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [0.0, 1.0],
                [0.1, 1.0],
            ],
            dtype=np.float32,
        ),
        "teacher_action": np.array(
            [
                [0.2, -0.2],
                [0.25, -0.15],
                [-0.2, 0.2],
                [-0.25, 0.15],
            ],
            dtype=np.float32,
        ),
        "teacher_mu": np.array(
            [
                [0.2, -0.2],
                [0.25, -0.15],
                [-0.2, 0.2],
                [-0.25, 0.15],
            ],
            dtype=np.float32,
        ),
        "traj_no": np.array([0, 0, 1, 1], dtype=np.int32),
        "subtraj_step_no": np.array([0, 1, 0, 1], dtype=np.int32),
    }
    ctrlrange = [[0.0, 1.0], [0.0, 1.0]]
    body_semantic = {
        "schema_version": "body_obs_v1",
        "total_size": 3,
        "kinematic_size": 3,
        "muscle_size": 0,
        "touch_size": 0,
        "goal_size": 0,
        "other_size": 0,
        "action_size": 2,
        "root_joint_name": "root",
        "joint_names": [],
        "actuator_names": ["hip", "shoulder"],
        "touch_sensor_names": [],
        "observation_names": ["state"],
        "student_filtered": True,
        "channels": [
            {
                "name": f"state[{index}]",
                "entry": "state",
                "entry_type": "test",
                "entry_offset": index,
                "student_index": index,
                "category": "kinematic",
            }
            for index in range(3)
        ],
    }
    body_schema = body_semantic | {
        "semantic_hash": ordered_schema_hash(kind="body_observation", payload=body_semantic),
        "provenance": {"teacher_ckpt": "/tmp/teacher"},
    }
    write_split_shard(
        path,
        data,
        split="train",
        shard_idx=0,
        metadata={
            "actuator_names": ["hip", "shoulder"],
            "actuator_ctrlrange": ctrlrange,
            "ctrlrange_schema_hash": ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": ["hip", "shoulder"], "ctrlrange": ctrlrange},
            ),
            "body_obs_schema": body_schema,
            "body_obs_schema_hash": body_schema["semantic_hash"],
        },
    )


def test_kl_warmup_schedule_reaches_target_after_warmup():
    from musclemimic.latent_muscle.train_latent import kl_warmup_weight

    assert kl_warmup_weight(step=0, target=0.2, warmup_steps=10) == 0.0
    assert kl_warmup_weight(step=5, target=0.2, warmup_steps=10) == 0.1
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=10) == 0.2
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=0) == 0.2


def test_temporal_smoothness_tracks_teacher_delta_instead_of_flattening_motion():
    from musclemimic.latent_muscle.train_latent import teacher_delta_smooth_mse

    teacher = np.array([[[0.0], [0.5], [1.0], [0.25]]], dtype=np.float32)
    matching_student = teacher.copy()
    flat_student = np.zeros_like(teacher)

    assert float(teacher_delta_smooth_mse(matching_student, teacher)) == 0.0
    assert float(teacher_delta_smooth_mse(flat_student, teacher)) > 0.0


def test_latent_checkpoint_roundtrip_preserves_action_mask_manifest(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint, save_latent_checkpoint

    mask = ActionMask.from_correction_actuators(
        all_actuator_names=["hip", "finger", "shoulder"],
        correction_actuator_names=["finger"],
    )
    variables = {"params": {"w": np.array([1.0, 2.0], dtype=np.float32)}}

    save_latent_checkpoint(
        tmp_path,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state={"step": np.array(3, dtype=np.int32)},
        action_mask=mask,
        config={"latent_dim": 2, "action_min": -1.0, "action_max": 1.0},
        train_metrics=[{"step": 1, "total_loss": 0.5}],
        eval_metrics={"action_mse": 0.25},
        obs_norm={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        action_norm={"min": -1.0, "max": 1.0},
    )

    loaded = load_latent_checkpoint(tmp_path)

    assert (tmp_path / "encoder.msgpack").is_file()
    assert (tmp_path / "prior.msgpack").is_file()
    assert (tmp_path / "decoder.msgpack").is_file()
    assert (tmp_path / "optimizer_state.msgpack").is_file()
    assert (tmp_path / "latent_config.yaml").is_file()
    assert (tmp_path / "train_metrics.csv").is_file()
    assert (tmp_path / "eval_metrics.json").is_file()
    assert loaded["action_mask"]["decoder_action_dim"] == 2
    assert loaded["action_mask"]["full_action_dim"] == 3
    np.testing.assert_array_equal(loaded["encoder_variables"]["params"]["w"], variables["params"]["w"])

    action_norm_path = tmp_path / "action_norm.json"
    original_action_norm = action_norm_path.read_bytes()
    action_norm_path.write_bytes(original_action_norm + b"\n")
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_latent_checkpoint(tmp_path, runtime_only=True)
    action_norm_path.write_bytes(original_action_norm)


def test_passed_production_checkpoint_cannot_omit_strict_closed_loop_report(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint, save_latent_checkpoint

    mask = ActionMask.from_correction_actuators(
        all_actuator_names=["hip", "shoulder"],
        correction_actuator_names=[],
    )
    variables = {"params": {"w": np.array([1.0], dtype=np.float32)}}
    save_latent_checkpoint(
        tmp_path,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state={"step": np.array(0, dtype=np.int32)},
        action_mask=mask,
        config={"latent_dim": 1, "require_closed_loop_metrics": True},
        train_metrics=[],
        eval_metrics={"promotion": {"passed": True}},
    )

    with pytest.raises(ValueError, match="missing closed_loop_metrics.json"):
        load_latent_checkpoint(tmp_path)


def test_latent_train_one_batch_writes_checkpoint_and_metrics(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig, train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir)
    output_dir = tmp_path / "latent_run"

    result = train_latent(
        LatentTrainConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            latent_dim=2,
            hidden_layer_dims=(8,),
            batch_size=2,
            horizon=2,
            num_steps=3,
            learning_rate=1e-2,
            seed=0,
            kl_weight=1e-3,
            kl_warmup_steps=2,
            smooth_weight=0.1,
            log_interval=0,
            action_mask={
                "all_actuator_names": ["hip", "shoulder"],
                "correction_actuator_names": [],
            },
            closed_loop_evaluator=lambda _context: {
                "fall_or_early_termination_rate": 0.0,
                "lambda_025_050_no_fall_rate": 1.0,
            },
            promotion_gates={
                "closed_loop_max_fall_or_early_termination_rate": 0.05,
                "closed_loop_min_lambda_025_050_no_fall_rate": 0.95,
            },
        )
    )

    checkpoint_dir = output_dir / "latent_checkpoint"
    metrics = json.loads((checkpoint_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    rows = (checkpoint_dir / "train_metrics.csv").read_text(encoding="utf-8").strip().splitlines()

    assert result.checkpoint_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "encoder.msgpack").is_file()
    assert (checkpoint_dir / "prior.msgpack").is_file()
    assert (checkpoint_dir / "decoder.msgpack").is_file()
    assert len(rows) >= 2
    assert np.isfinite(metrics["action_mse"])
    assert metrics["action_min"] == -1.0
    assert metrics["action_max"] == 1.0
    assert np.isfinite(metrics["posterior_action_mse"])
    assert np.isfinite(metrics["prior_mean_action_mse"])
    assert metrics["closed_loop_fall_or_early_termination_rate"] == 0.0
    assert (checkpoint_dir / "state_schema.json").is_file()
    assert (checkpoint_dir / "action_schema.json").is_file()
    assert (checkpoint_dir / "motion_split.json").is_file()
    assert (checkpoint_dir / "body_obs_schema.json").is_file()
    assert (checkpoint_dir / "checkpoint_fingerprint.txt").is_file()
    obs_norm = json.loads((checkpoint_dir / "obs_norm.json").read_text(encoding="utf-8"))
    assert obs_norm["count"] == 4
    assert obs_norm["source_split"] == "train"

    from musclemimic.latent_muscle.runtime import (
        LatentCheckpointCompatibilityError,
        LatentMuscleRuntime,
    )

    runtime = LatentMuscleRuntime.from_checkpoint(
        checkpoint_dir,
        runtime_body_actuator_names=["hip", "shoulder"],
    )
    assert runtime.body_obs_schema["actuator_names"] == ["hip", "shoulder"]
    assert runtime.body_obs_schema["action_size"] == 2
    state = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    prior_mu, prior_sigma = runtime.prior_numpy(state)
    prior_mu_raw, prior_raw_sigma = runtime.prior_raw_numpy(state)
    action = runtime.decode_numpy(state, prior_mu)
    assert prior_mu.shape == (1, 2)
    assert prior_sigma.shape == (1, 2)
    assert action.shape == (1, 2)
    np.testing.assert_allclose(prior_mu_raw, prior_mu)
    expected_sigma = np.clip(
        np.log1p(np.exp(-np.abs(prior_raw_sigma))) + np.maximum(prior_raw_sigma, 0.0),
        runtime.sigma_min,
        runtime.sigma_max,
    )
    np.testing.assert_allclose(prior_sigma, expected_sigma, rtol=1e-6)

    import jax
    import jax.numpy as jnp

    jitted = jax.jit(runtime.prior_mean_action_jax)(jnp.asarray(state))
    np.testing.assert_allclose(np.asarray(jitted), runtime.prior_mean_action_numpy(state), rtol=1e-5)
    with pytest.raises(LatentCheckpointCompatibilityError, match="actuator names"):
        LatentMuscleRuntime.from_checkpoint(
            checkpoint_dir,
            runtime_body_actuator_names=["shoulder", "hip"],
        )

    # Deployment bundles intentionally omit posterior/optimizer training state.
    (checkpoint_dir / "encoder.msgpack").unlink()
    (checkpoint_dir / "optimizer_state.msgpack").unlink()
    runtime_only = LatentMuscleRuntime.from_checkpoint(checkpoint_dir)
    assert runtime_only.checkpoint_dir == str(checkpoint_dir)
    assert runtime_only.prior_mean_action_numpy(state).shape == (1, 2)
    assert runtime.body_obs_schema["root_joint_name"] == "root"
    np.testing.assert_allclose(runtime.body_ctrlrange, [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(runtime.body_action_to_ctrl_numpy([[-1.0, 1.0]]), [[0.0, 1.0]])
    assert len(runtime.checkpoint_fingerprint) == 64
    assert runtime.control_manifest["checkpoint_fingerprint"] == runtime.checkpoint_fingerprint
