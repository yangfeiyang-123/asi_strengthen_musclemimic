from __future__ import annotations

import json

import numpy as np
import pytest

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
    write_split_shard(path, data, split="train", shard_idx=0)


def test_kl_warmup_schedule_reaches_target_after_warmup():
    from musclemimic.latent_muscle.train_latent import kl_warmup_weight

    assert kl_warmup_weight(step=0, target=0.2, warmup_steps=10) == 0.0
    assert kl_warmup_weight(step=5, target=0.2, warmup_steps=10) == 0.1
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=10) == 0.2
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=0) == 0.2


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
