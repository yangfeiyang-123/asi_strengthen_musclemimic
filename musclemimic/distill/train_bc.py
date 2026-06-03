"""Offline behavior cloning trainer for distilled student policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.checkpoint_manager import CheckpointMetadata, UnifiedCheckpointManager
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.ppo.config import PPOAgentState
from musclemimic.distill.dataset import DistillDataset
from musclemimic.distill.losses import bc_loss, distribution_log_std, distribution_mean
from musclemimic.runner.engine import instantiate_env


@dataclass
class BCTrainResult:
    checkpoint_path: str
    metadata_path: str
    final_train_action_mse: float
    final_val_action_mse: float | None


def _batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {name: jnp.asarray(value) for name, value in batch.items()}


def _ensure_student_filter(config: Any) -> None:
    with open_dict(config):
        if "student_obs_filter" not in config.experiment:
            config.experiment.student_obs_filter = {}
        config.experiment.student_obs_filter.enabled = True
        config.experiment.student_obs_filter.drop_goal_lookahead = True
        config.experiment.student_obs_filter.keep_motion_phase = True
        config.experiment.student_obs_filter.require_goal_group = True
        config.experiment.student_obs_filter.require_motion_phase = True


def validate_dataset_matches_student_env(*, dataset: DistillDataset, env: Any, config: Any) -> int:
    """Validate that BC shards match the configured student policy input shape."""
    wrapped_env = wrap_env(env, config.experiment)
    expected_dim = int(wrapped_env.info.observation_space.shape[0])
    if int(dataset.student_obs_dim) != expected_dim:
        raise ValueError(
            "distill dataset student_obs_dim does not match configured student env: "
            f"dataset student_obs_dim={int(dataset.student_obs_dim)} expected={expected_dim}"
        )
    return expected_dim


def evaluate_bc_loss(
    train_state: TrainState,
    network: Any,
    dataset: DistillDataset,
    batch_size: int,
    gaussian_kl_weight: float = 0.0,
) -> dict[str, float]:
    sums = {"total_loss": 0.0, "action_mse": 0.0, "value_mse": 0.0, "gaussian_kl": 0.0}
    count = 0
    for batch in dataset.iter_batches(batch_size=batch_size, shuffle=False, repeat=False):
        jbatch = _batch_to_jax(batch)
        pi, value = network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            jbatch["student_obs"],
        )
        losses = bc_loss(
            student_mu=distribution_mean(pi),
            teacher_action=jbatch["teacher_action"],
            student_value=value,
            teacher_value=jbatch.get("teacher_value"),
            student_log_std=distribution_log_std(pi),
            teacher_mu=jbatch.get("teacher_mu"),
            teacher_log_std=jbatch.get("teacher_log_std"),
            gaussian_kl_weight=float(gaussian_kl_weight),
        )
        n = int(batch["student_obs"].shape[0])
        for key in sums:
            sums[key] += float(losses[key]) * n
        count += n
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_bc(
    *,
    config: Any,
    dataset_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 4096,
    num_steps: int = 200_000,
    lr: float = 3e-4,
    seed: int = 0,
    value_distill_weight: float = 0.1,
    gaussian_kl_weight: float = 0.0,
    log_interval: int = 100,
) -> BCTrainResult:
    """Train a PPO-compatible student checkpoint from teacher rollout shards."""
    _ensure_student_filter(config)
    dataset = DistillDataset(dataset_dir, split="train", seed=seed)
    try:
        val_dataset = DistillDataset(dataset_dir, split="val", seed=seed)
    except FileNotFoundError:
        val_dataset = None

    env = instantiate_env(config)
    expected_student_obs_dim = validate_dataset_matches_student_env(dataset=dataset, env=env, config=config)
    agent_conf = PPOJax.init_agent_conf(env, config)
    tx = optax.adam(float(lr))

    rng = jax.random.PRNGKey(int(seed))
    init_obs = jnp.zeros((dataset.student_obs_dim,), dtype=jnp.float32)
    init_vars = agent_conf.network.init(rng, init_obs)
    train_state = TrainState.create(
        apply_fn=agent_conf.network.apply,
        params=init_vars["params"],
        run_stats=init_vars["run_stats"],
        tx=tx,
    )

    @jax.jit
    def train_step(ts: TrainState, batch: dict[str, jax.Array]):
        def loss_fn(params, run_stats):
            (pi, value), updates = agent_conf.network.apply(
                {"params": params, "run_stats": run_stats},
                batch["student_obs"],
                mutable=["run_stats"],
            )
            losses = bc_loss(
                student_mu=distribution_mean(pi),
                teacher_action=batch["teacher_action"],
                student_value=value,
                teacher_value=batch.get("teacher_value"),
                student_log_std=distribution_log_std(pi),
                teacher_mu=batch.get("teacher_mu"),
                teacher_log_std=batch.get("teacher_log_std"),
                action_mse_weight=1.0,
                value_distill_weight=float(value_distill_weight),
                gaussian_kl_weight=float(gaussian_kl_weight),
            )
            return losses["total_loss"], (losses, updates["run_stats"])

        (loss, (losses, new_run_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts.params,
            ts.run_stats,
        )
        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(run_stats=new_run_stats)
        return ts, losses | {"total_loss": loss}

    last_losses = None
    batch_iter = dataset.iter_batches(batch_size=batch_size, shuffle=True, repeat=True)
    for step in range(1, int(num_steps) + 1):
        batch = _batch_to_jax(next(batch_iter))
        train_state, last_losses = train_step(train_state, batch)
        if log_interval and step % int(log_interval) == 0:
            print(
                f"[bc] step={step} "
                f"loss={float(last_losses['total_loss']):.6f} "
                f"action_mse={float(last_losses['action_mse']):.6f} "
                f"value_mse={float(last_losses['value_mse']):.6f} "
                f"gaussian_kl={float(last_losses['gaussian_kl']):.6f}"
            )

    output_path = Path(output_dir)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = str(checkpoint_dir)
        config.experiment.bc_dataset_dir = str(dataset_dir)

    agent_conf = PPOJax._agent_conf(config, agent_conf.network, tx)
    agent_state = PPOAgentState(train_state=train_state)
    ckpt_metadata = CheckpointMetadata(
        step=int(train_state.step),
        update_number=int(num_steps),
        global_timestep=0,
        target_global_timestep=0,
        learning_rate=float(lr),
        num_envs=int(config.experiment.get("num_envs", 1)),
        num_steps=int(config.experiment.get("num_steps", config.experiment.get("ppo_config", {}).get("num_steps", 1))),
        num_minibatches=int(config.experiment.get("num_minibatches", config.experiment.get("ppo_config", {}).get("num_minibatches", 1))),
        update_epochs=int(config.experiment.get("update_epochs", config.experiment.get("ppo_config", {}).get("update_epochs", 1))),
        backend=str(config.experiment.env_params.get("mjx_backend", "jax")),
        env_name=str(config.experiment.env_params.get("env_name", "")),
    )
    manager = UnifiedCheckpointManager(str(checkpoint_dir), max_to_keep=5, async_save=False)
    try:
        checkpoint_path = manager.save_checkpoint(int(train_state.step), agent_conf, agent_state, ckpt_metadata)
    finally:
        manager.close()

    train_metrics = evaluate_bc_loss(
        train_state,
        agent_conf.network,
        dataset,
        batch_size=batch_size,
        gaussian_kl_weight=gaussian_kl_weight,
    )
    val_metrics = (
        evaluate_bc_loss(
            train_state,
            agent_conf.network,
            val_dataset,
            batch_size=batch_size,
            gaussian_kl_weight=gaussian_kl_weight,
        )
        if val_dataset
        else None
    )

    metadata = {
        "dataset_dir": str(dataset_dir),
        "student_obs_dim": dataset.student_obs_dim,
        "expected_student_obs_dim": expected_student_obs_dim,
        "action_dim": dataset.action_dim,
        "bc_steps": int(num_steps),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "value_distill_weight": float(value_distill_weight),
        "gaussian_kl_weight": float(gaussian_kl_weight),
        "train": train_metrics,
        "val": val_metrics,
        "dataset_metadata": dataset.metadata,
        "student_obs_filter": OmegaConf.to_container(config.experiment.student_obs_filter, resolve=True),
    }
    metadata_path = output_path / "distill_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return BCTrainResult(
        checkpoint_path=checkpoint_path,
        metadata_path=str(metadata_path),
        final_train_action_mse=float(train_metrics["action_mse"]),
        final_val_action_mse=float(val_metrics["action_mse"]) if val_metrics else None,
    )


def load_config_for_bc(config_path: str | Path):
    """Load a Hydra fullbody config from a path, falling back to plain YAML."""
    config_name = str(config_path)
    config_path = Path(config_name)
    fullbody_dir = Path(__file__).resolve().parents[2] / "fullbody"
    try:
        import hydra
        from hydra import compose, initialize_config_dir

        if config_path.is_file():
            rel = config_path.resolve().relative_to(fullbody_dir.resolve()).with_suffix("")
            config_name = str(rel)
        with initialize_config_dir(version_base=None, config_dir=str(fullbody_dir.resolve())):
            return compose(config_name=config_name.removesuffix(".yaml"))
    except Exception:
        return OmegaConf.load(config_path)
