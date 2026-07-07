"""Minimal latent distillation trainer for posterior/prior/decoder modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from musclemimic.distill.dataset import SequenceDistillDataset
from musclemimic.latent_muscle.action_mask import ActionMask
from musclemimic.latent_muscle.checkpoint import save_latent_checkpoint
from musclemimic.latent_muscle.losses import latent_distillation_loss, positive_sigma
from musclemimic.latent_muscle.networks import (
    ConditionalPrior,
    LatentDecoder,
    PosteriorEncoder,
    reparameterize_gaussian,
)


@dataclass(frozen=True)
class LatentTrainConfig:
    dataset_dir: str
    output_dir: str
    latent_dim: int = 32
    hidden_layer_dims: tuple[int, ...] = (512, 256)
    batch_size: int = 256
    horizon: int = 8
    num_steps: int = 100_000
    learning_rate: float = 3e-4
    seed: int = 0
    kl_weight: float = 1e-3
    kl_warmup_steps: int = 10_000
    free_bits: float = 0.0
    smooth_weight: float = 0.0
    bound_weight: float = 0.0
    action_min: float = -1.0
    action_max: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    log_interval: int = 100
    action_mask: dict[str, Any] | None = None


@dataclass(frozen=True)
class LatentTrainResult:
    checkpoint_dir: str
    final_total_loss: float
    final_action_mse: float


def kl_warmup_weight(*, step: int, target: float, warmup_steps: int) -> float:
    if int(warmup_steps) <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, float(step) / float(warmup_steps)))


def train_latent(config: LatentTrainConfig) -> LatentTrainResult:
    dataset = SequenceDistillDataset(config.dataset_dir, split="train", seed=config.seed)
    action_dim = int(dataset.action_dim)
    action_mask = _build_action_mask(config.action_mask, action_dim)
    if action_mask.body_size != action_dim:
        raise ValueError(
            f"decoder action_dim={action_dim} must equal action mask body size={action_mask.body_size}"
        )

    posterior = PosteriorEncoder(
        latent_dim=int(config.latent_dim),
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    prior = ConditionalPrior(
        latent_dim=int(config.latent_dim),
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    decoder = LatentDecoder(
        action_dim=action_dim,
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        bounded_action=True,
    )

    rng = jax.random.PRNGKey(int(config.seed))
    init_batch = next(dataset.iter_sequence_batches(batch_size=1, horizon=config.horizon, shuffle=False))
    init_state = jnp.asarray(init_batch["student_obs"].reshape(-1, dataset.student_obs_dim), dtype=jnp.float32)
    init_ref = jnp.asarray(init_batch["reference_features"].reshape(-1, dataset.reference_features_dim), dtype=jnp.float32)
    init_latent = jnp.zeros((init_state.shape[0], int(config.latent_dim)), dtype=jnp.float32)
    rng, post_rng, prior_rng, dec_rng = jax.random.split(rng, 4)
    variables = {
        "encoder": posterior.init(post_rng, init_state, init_ref),
        "prior": prior.init(prior_rng, init_state),
        "decoder": decoder.init(dec_rng, init_state, init_latent),
    }
    tx = optax.adam(float(config.learning_rate))
    opt_state = tx.init(variables)
    batch_iter = dataset.iter_sequence_batches(
        batch_size=int(config.batch_size),
        horizon=int(config.horizon),
        shuffle=True,
        repeat=True,
    )
    metrics_history: list[dict[str, Any]] = []

    def loss_fn(params, batch, step_rng, kl_weight):
        state = jnp.asarray(batch["student_obs"], dtype=jnp.float32)
        reference = jnp.asarray(batch["reference_features"], dtype=jnp.float32)
        teacher_action = jnp.asarray(batch["teacher_action"], dtype=jnp.float32)
        batch_size, horizon = state.shape[:2]
        flat_state = state.reshape((batch_size * horizon, state.shape[-1]))
        flat_reference = reference.reshape((batch_size * horizon, reference.shape[-1]))
        flat_teacher_action = teacher_action.reshape((batch_size * horizon, teacher_action.shape[-1]))

        posterior_mu, posterior_raw_sigma = posterior.apply(params["encoder"], flat_state, flat_reference)
        prior_mu, prior_raw_sigma = prior.apply(params["prior"], flat_state)
        z = reparameterize_gaussian(
            step_rng,
            posterior_mu,
            posterior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        predicted_action = decoder.apply(params["decoder"], flat_state, z)
        losses = latent_distillation_loss(
            predicted_action=predicted_action,
            teacher_action=flat_teacher_action,
            posterior_mu=posterior_mu,
            posterior_raw_sigma=posterior_raw_sigma,
            prior_mu=prior_mu,
            prior_raw_sigma=prior_raw_sigma,
            action_weight=1.0,
            kl_weight=kl_weight,
            free_bits=float(config.free_bits),
            smooth_weight=0.0,
            bound_weight=float(config.bound_weight),
            action_min=float(config.action_min),
            action_max=float(config.action_max),
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        pred_seq = predicted_action.reshape((batch_size, horizon, teacher_action.shape[-1]))
        if horizon > 1 and float(config.smooth_weight):
            smooth_mse = jnp.mean(jnp.square(pred_seq[:, 1:] - pred_seq[:, :-1]))
        else:
            smooth_mse = jnp.asarray(0.0, dtype=losses["total_loss"].dtype)
        posterior_sigma = positive_sigma(
            posterior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        prior_sigma = positive_sigma(
            prior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        total = losses["total_loss"] + float(config.smooth_weight) * smooth_mse
        diagnostics = losses | {
            "total_loss": total,
            "smooth_mse": smooth_mse,
            "posterior_sigma_mean": jnp.mean(posterior_sigma),
            "posterior_sigma_min": jnp.min(posterior_sigma),
            "posterior_sigma_max": jnp.max(posterior_sigma),
            "prior_sigma_mean": jnp.mean(prior_sigma),
            "prior_sigma_min": jnp.min(prior_sigma),
            "prior_sigma_max": jnp.max(prior_sigma),
            "posterior_prior_mu_l2": jnp.mean(jnp.linalg.norm(posterior_mu - prior_mu, axis=-1)),
            "latent_dim_active_count": jnp.sum(jnp.std(z, axis=0) > 1e-3),
            "decoder_action_mean": jnp.mean(predicted_action),
            "decoder_action_std": jnp.std(predicted_action),
            "decoder_action_min": jnp.min(predicted_action),
            "decoder_action_max": jnp.max(predicted_action),
        }
        return total, diagnostics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    final_metrics: dict[str, Any] | None = None
    for step in range(1, int(config.num_steps) + 1):
        batch = _batch_to_jax(next(batch_iter))
        rng, step_rng = jax.random.split(rng)
        current_kl_weight = kl_warmup_weight(
            step=step,
            target=float(config.kl_weight),
            warmup_steps=int(config.kl_warmup_steps),
        )
        (_loss, metrics), grads = grad_fn(variables, batch, step_rng, current_kl_weight)
        updates, opt_state = tx.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        final_metrics = _metrics_to_float(metrics | {"step": step, "kl_weight": current_kl_weight})
        metrics_history.append(final_metrics)
        if config.log_interval and step % int(config.log_interval) == 0:
            print(
                f"[latent] step={step} loss={final_metrics['total_loss']:.6f} "
                f"action_mse={final_metrics['action_mse']:.6f} kl={final_metrics['kl']:.6f}"
            )

    eval_metrics = _evaluate_once(
        variables=variables,
        posterior=posterior,
        prior=prior,
        decoder=decoder,
        dataset=dataset,
        config=config,
    )
    checkpoint_dir = Path(config.output_dir) / "latent_checkpoint"
    save_latent_checkpoint(
        checkpoint_dir,
        encoder_variables=variables["encoder"],
        prior_variables=variables["prior"],
        decoder_variables=variables["decoder"],
        optimizer_state=opt_state,
        action_mask=action_mask,
        config=asdict(config) | {
            "student_obs_dim": int(dataset.student_obs_dim),
            "reference_features_dim": int(dataset.reference_features_dim),
            "action_dim": action_dim,
        },
        train_metrics=metrics_history,
        eval_metrics=eval_metrics,
        obs_norm={
            "mean": np.mean(dataset.arrays["student_obs"], axis=0),
            "std": np.std(dataset.arrays["student_obs"], axis=0) + 1e-8,
        },
        action_norm={"min": float(config.action_min), "max": float(config.action_max)},
    )
    final_metrics = final_metrics or {"total_loss": float("nan"), "action_mse": float("nan")}
    return LatentTrainResult(
        checkpoint_dir=str(checkpoint_dir),
        final_total_loss=float(final_metrics["total_loss"]),
        final_action_mse=float(final_metrics["action_mse"]),
    )


def _evaluate_once(*, variables, posterior, prior, decoder, dataset, config: LatentTrainConfig) -> dict[str, Any]:
    batch = next(dataset.iter_sequence_batches(batch_size=int(config.batch_size), horizon=int(config.horizon), shuffle=False))
    state = jnp.asarray(batch["student_obs"], dtype=jnp.float32)
    reference = jnp.asarray(batch["reference_features"], dtype=jnp.float32)
    teacher_action = jnp.asarray(batch["teacher_action"], dtype=jnp.float32)
    flat_state = state.reshape((-1, state.shape[-1]))
    flat_reference = reference.reshape((-1, reference.shape[-1]))
    flat_teacher_action = teacher_action.reshape((-1, teacher_action.shape[-1]))
    posterior_mu, posterior_raw_sigma = posterior.apply(variables["encoder"], flat_state, flat_reference)
    prior_mu, prior_raw_sigma = prior.apply(variables["prior"], flat_state)
    predicted_action = decoder.apply(variables["decoder"], flat_state, posterior_mu)
    losses = latent_distillation_loss(
        predicted_action=predicted_action,
        teacher_action=flat_teacher_action,
        posterior_mu=posterior_mu,
        posterior_raw_sigma=posterior_raw_sigma,
        prior_mu=prior_mu,
        prior_raw_sigma=prior_raw_sigma,
        kl_weight=float(config.kl_weight),
        free_bits=float(config.free_bits),
        bound_weight=float(config.bound_weight),
        action_min=float(config.action_min),
        action_max=float(config.action_max),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    return _metrics_to_float(
        losses
        | {
            "action_min": float(config.action_min),
            "action_max": float(config.action_max),
            "num_eval_samples": int(flat_state.shape[0]),
        }
    )


def _batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _metrics_to_float(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: float(value) if np.ndim(np.asarray(value)) == 0 else np.asarray(value).tolist() for key, value in metrics.items()}


def _build_action_mask(config: dict[str, Any] | None, action_dim: int) -> ActionMask:
    if config is None:
        all_names = [f"action_{index}" for index in range(int(action_dim))]
        correction_names: list[str] = []
    else:
        all_names = list(config.get("all_actuator_names") or [f"action_{index}" for index in range(int(action_dim))])
        correction_names = list(config.get("correction_actuator_names") or [])
    return ActionMask.from_correction_actuators(
        all_actuator_names=all_names,
        correction_actuator_names=correction_names,
    )
