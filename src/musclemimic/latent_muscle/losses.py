"""Losses for LATENT-style muscle skill distillation."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp


def positive_sigma(raw_sigma, *, sigma_min: float = 0.05, sigma_max: float = 2.0):
    """Convert an unconstrained scale parameter into a positive sigma.

    Applies softplus then clips to [sigma_min, sigma_max], matching the
    LAB deployment transform exactly so KL training sees identical sigma
    values.
    """
    return jnp.clip(jax.nn.softplus(raw_sigma), float(sigma_min), float(sigma_max))


def gaussian_diag_kl_from_raw_sigma(
    posterior_mu,
    posterior_raw_sigma,
    prior_mu,
    prior_raw_sigma,
    *,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
) -> jnp.ndarray:
    """Mean KL(q(z|s,r) || p(z|s)) for diagonal Gaussians.

    Both sigma inputs are unconstrained parameters converted with softplus
    and clamped to [sigma_min, sigma_max], matching the LAB transform used
    at deployment time.
    """
    posterior_sigma = positive_sigma(posterior_raw_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
    prior_sigma = positive_sigma(prior_raw_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
    posterior_var = jnp.square(posterior_sigma)
    prior_var = jnp.square(prior_sigma)
    per_dim = (
        jnp.log(prior_sigma)
        - jnp.log(posterior_sigma)
        + (posterior_var + jnp.square(posterior_mu - prior_mu)) / (2.0 * prior_var)
        - 0.5
    )
    return jnp.mean(jnp.sum(per_dim, axis=-1))


def latent_distillation_loss(
    *,
    predicted_action,
    teacher_action,
    posterior_mu,
    prior_mu,
    posterior_raw_sigma=None,
    prior_raw_sigma=None,
    posterior_log_sigma=None,
    prior_log_sigma=None,
    previous_predicted_action=None,
    action_weight: float = 1.0,
    kl_weight: float = 1e-3,
    free_bits: float = 0.0,
    smooth_weight: float = 0.0,
    bound_weight: float = 0.0,
    action_min: float = -1.0,
    action_max: float = 1.0,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    sample_weight=None,
    predicted_physical_excitation=None,
    teacher_physical_excitation=None,
    physical_excitation_weight: float = 0.0,
    residual_excitation=None,
    residual_l1_weight: float = 0.0,
    residual_l2_weight: float = 0.0,
    baseline_excitation=None,
    baseline_l1_weight: float = 0.0,
    baseline_l2_weight: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Combine reconstruction, KL, physical, residual, smoothness, and bounds.

    ``sample_weight`` is applied to action-space/physical/residual terms while
    KL remains uniformly averaged.  This lets short impact phases receive more
    reconstruction weight without changing the probabilistic prior itself.
    """
    posterior_raw_sigma = _resolve_raw_sigma(
        "posterior",
        raw_sigma=posterior_raw_sigma,
        log_sigma=posterior_log_sigma,
    )
    prior_raw_sigma = _resolve_raw_sigma(
        "prior",
        raw_sigma=prior_raw_sigma,
        log_sigma=prior_log_sigma,
    )
    predicted_action = jnp.asarray(predicted_action)
    teacher_action = jnp.asarray(teacher_action)
    action_mse = _weighted_feature_mean(
        jnp.square(predicted_action - teacher_action), sample_weight
    )
    kl_per_sample = gaussian_diag_kl_per_sample(
        posterior_mu,
        posterior_raw_sigma,
        prior_mu,
        prior_raw_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    if float(free_bits) > 0.0:
        latent_dim = jnp.asarray(posterior_mu).shape[-1]
        kl_per_sample = jnp.maximum(kl_per_sample, float(free_bits) * float(latent_dim))
    kl = jnp.mean(kl_per_sample)
    if previous_predicted_action is None or float(smooth_weight) == 0.0:
        smooth_mse = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        smooth_mse = _weighted_feature_mean(
            jnp.square(predicted_action - jnp.asarray(previous_predicted_action)),
            sample_weight,
        )

    if float(bound_weight) == 0.0:
        bound_violation = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        below = jnp.square(jnp.minimum(predicted_action - float(action_min), 0.0))
        above = jnp.square(jnp.maximum(predicted_action - float(action_max), 0.0))
        bound_violation = _weighted_feature_mean(below + above, sample_weight)

    if predicted_physical_excitation is None and teacher_physical_excitation is None:
        physical_excitation_mse = jnp.asarray(0.0, dtype=action_mse.dtype)
    elif predicted_physical_excitation is None or teacher_physical_excitation is None:
        raise ValueError(
            "predicted_physical_excitation and teacher_physical_excitation must be supplied together"
        )
    else:
        predicted_physical = jnp.asarray(predicted_physical_excitation)
        teacher_physical = jnp.asarray(teacher_physical_excitation)
        if predicted_physical.shape != teacher_physical.shape:
            raise ValueError(
                "predicted/teacher physical excitation shapes differ: "
                f"{predicted_physical.shape} vs {teacher_physical.shape}"
            )
        physical_excitation_mse = _weighted_feature_mean(
            jnp.square(predicted_physical - teacher_physical), sample_weight
        )

    if residual_excitation is None:
        residual_l1 = jnp.asarray(0.0, dtype=action_mse.dtype)
        residual_l2 = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        residual = jnp.asarray(residual_excitation)
        residual_l1 = _weighted_feature_mean(jnp.abs(residual), sample_weight)
        residual_l2 = _weighted_feature_mean(jnp.square(residual), sample_weight)

    if baseline_excitation is None:
        baseline_l1 = jnp.asarray(0.0, dtype=action_mse.dtype)
        baseline_l2 = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        baseline = jnp.asarray(baseline_excitation)
        baseline_l1 = _weighted_feature_mean(jnp.abs(baseline), sample_weight)
        baseline_l2 = _weighted_feature_mean(jnp.square(baseline), sample_weight)

    total = (
        float(action_weight) * action_mse
        + float(kl_weight) * kl
        + float(smooth_weight) * smooth_mse
        + float(bound_weight) * bound_violation
        + float(physical_excitation_weight) * physical_excitation_mse
        + float(residual_l1_weight) * residual_l1
        + float(residual_l2_weight) * residual_l2
        + float(baseline_l1_weight) * baseline_l1
        + float(baseline_l2_weight) * baseline_l2
    )
    return {
        "total_loss": total,
        "action_mse": action_mse,
        "kl": kl,
        "smooth_mse": smooth_mse,
        "bound_violation": bound_violation,
        "physical_excitation_mse": physical_excitation_mse,
        "residual_l1": residual_l1,
        "residual_l2": residual_l2,
        "baseline_l1": baseline_l1,
        "baseline_l2": baseline_l2,
    }


def gaussian_diag_kl_per_sample(
    posterior_mu,
    posterior_raw_sigma,
    prior_mu,
    prior_raw_sigma,
    *,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
) -> jnp.ndarray:
    posterior_sigma = positive_sigma(posterior_raw_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
    prior_sigma = positive_sigma(prior_raw_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
    posterior_var = jnp.square(posterior_sigma)
    prior_var = jnp.square(prior_sigma)
    per_dim = (
        jnp.log(prior_sigma)
        - jnp.log(posterior_sigma)
        + (posterior_var + jnp.square(posterior_mu - prior_mu)) / (2.0 * prior_var)
        - 0.5
    )
    return jnp.sum(per_dim, axis=-1)


def _resolve_raw_sigma(name: str, *, raw_sigma, log_sigma):
    if raw_sigma is not None:
        if log_sigma is not None:
            raise ValueError(f"pass either {name}_raw_sigma or {name}_log_sigma, not both")
        return raw_sigma
    if log_sigma is None:
        raise TypeError(f"missing required keyword argument: {name}_raw_sigma")
    warnings.warn(
        f"{name}_log_sigma is deprecated; pass {name}_raw_sigma because the value is "
        "an unconstrained softplus scale parameter, not a log standard deviation.",
        DeprecationWarning,
        stacklevel=3,
    )
    return log_sigma


def _weighted_feature_mean(values, sample_weight=None):
    """Reduce feature dimensions per sample, then take a safe weighted mean."""

    values = jnp.asarray(values)
    if values.ndim == 0:
        return values
    per_sample = values
    if values.ndim > 1:
        per_sample = jnp.mean(values, axis=tuple(range(1, values.ndim)))
    if sample_weight is None:
        return jnp.mean(per_sample)
    weights = jnp.asarray(sample_weight, dtype=per_sample.dtype)
    if weights.shape != per_sample.shape:
        try:
            weights = jnp.broadcast_to(weights, per_sample.shape)
        except ValueError as exc:
            raise ValueError(
                f"sample_weight shape {weights.shape} cannot broadcast to {per_sample.shape}"
            ) from exc
    weights = jnp.maximum(weights, 0.0)
    return jnp.sum(weights * per_sample) / jnp.maximum(jnp.sum(weights), 1e-12)
