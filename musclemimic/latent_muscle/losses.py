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
) -> dict[str, jnp.ndarray]:
    """Combine teacher action reconstruction, prior KL, smoothness, and bounds."""
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
    action_mse = jnp.mean(jnp.square(predicted_action - teacher_action))
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
        smooth_mse = jnp.mean(jnp.square(predicted_action - jnp.asarray(previous_predicted_action)))

    if float(bound_weight) == 0.0:
        bound_violation = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        below = jnp.square(jnp.minimum(predicted_action - float(action_min), 0.0))
        above = jnp.square(jnp.maximum(predicted_action - float(action_max), 0.0))
        bound_violation = jnp.mean(below + above)

    total = (
        float(action_weight) * action_mse
        + float(kl_weight) * kl
        + float(smooth_weight) * smooth_mse
        + float(bound_weight) * bound_violation
    )
    return {
        "total_loss": total,
        "action_mse": action_mse,
        "kl": kl,
        "smooth_mse": smooth_mse,
        "bound_violation": bound_violation,
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
