"""Latent Action Barrier helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LABActionOutput:
    """Diagnostics from a LAB-constrained latent action decode."""

    action: np.ndarray
    body_action: np.ndarray
    latent: np.ndarray
    raw_latent: np.ndarray
    prior_mu: np.ndarray
    prior_sigma: np.ndarray
    correction: np.ndarray | None = None


def latent_action_barrier(
    *,
    raw_latent,
    prior_mu,
    prior_raw_sigma=None,
    prior_log_sigma=None,
    lambda_lab: float,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
):
    """Map PPO raw latent residuals into the state-conditioned prior box.

    The sigma argument is an unconstrained scale parameter consumed by
    softplus, not a log standard deviation. ``prior_log_sigma`` remains as a
    deprecated compatibility alias.
    """
    if prior_raw_sigma is None:
        if prior_log_sigma is None:
            raise TypeError("missing required keyword argument: prior_raw_sigma")
        warnings.warn(
            "prior_log_sigma is deprecated; pass prior_raw_sigma because the value is "
            "an unconstrained softplus scale parameter, not a log standard deviation.",
            DeprecationWarning,
            stacklevel=2,
        )
        prior_raw_sigma = prior_log_sigma
    elif prior_log_sigma is not None:
        raise ValueError("pass either prior_raw_sigma or prior_log_sigma, not both")
    raw_latent = jnp.asarray(raw_latent)
    prior_mu = jnp.asarray(prior_mu)
    prior_raw_sigma = jnp.asarray(prior_raw_sigma)
    sigma = jnp.clip(jax.nn.softplus(prior_raw_sigma), float(sigma_min), float(sigma_max))
    return prior_mu + float(lambda_lab) * sigma * jnp.tanh(raw_latent)


class LABActionWrapper:
    """Decode high-level latent actions through prior, LAB, and body decoder."""

    def __init__(
        self,
        *,
        prior: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        decoder: Callable[[np.ndarray, np.ndarray], np.ndarray],
        lambda_lab: float,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        combine_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if float(lambda_lab) <= 0.0:
            raise ValueError("lambda_lab must be positive")
        if float(sigma_min) <= 0.0:
            raise ValueError("sigma_min must be positive")
        if float(sigma_max) < float(sigma_min):
            raise ValueError("sigma_max must be >= sigma_min")
        self.prior = prior
        self.decoder = decoder
        self.lambda_lab = float(lambda_lab)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.combine_fn = combine_fn

    def __call__(self, state: np.ndarray, high_level_action: Any, *, return_info: bool = False):
        state_array = np.asarray(state, dtype=float)
        raw_latent, correction = _split_high_level_action(high_level_action)
        prior_mu, prior_raw_sigma = self.prior(state_array)
        prior_mu = _as_finite_array("prior_mu", prior_mu)
        prior_raw_sigma = _as_finite_array("prior_raw_sigma", prior_raw_sigma)
        raw_latent = _as_finite_array("raw_latent", raw_latent)
        if raw_latent.shape != prior_mu.shape:
            raise ValueError(f"raw_latent shape {raw_latent.shape} must match prior_mu shape {prior_mu.shape}")
        if prior_raw_sigma.shape != prior_mu.shape:
            raise ValueError(
                f"prior_raw_sigma shape {prior_raw_sigma.shape} must match prior_mu shape {prior_mu.shape}"
            )

        prior_sigma = np.clip(_softplus(prior_raw_sigma), self.sigma_min, self.sigma_max)
        latent = prior_mu + self.lambda_lab * prior_sigma * np.tanh(raw_latent)
        body_action = _as_finite_array("body_action", self.decoder(state_array, latent))
        if correction is None:
            action = body_action
            correction_array = None
        elif self.combine_fn is None:
            raise ValueError("high_level_action provided correction but LABActionWrapper has no combine_fn")
        else:
            correction_array = _as_finite_array("correction", correction)
            action = _as_finite_array("action", self.combine_fn(body_action, correction_array))

        output = LABActionOutput(
            action=action,
            body_action=body_action,
            latent=latent,
            raw_latent=raw_latent,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            correction=correction_array,
        )
        return output if return_info else output.action


def _split_high_level_action(high_level_action: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(high_level_action, dict):
        if "latent" not in high_level_action:
            raise ValueError("high_level_action dict must contain a 'latent' entry")
        correction = high_level_action.get("correction")
        return np.asarray(high_level_action["latent"], dtype=float), None if correction is None else np.asarray(correction)
    return np.asarray(high_level_action, dtype=float), None


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def _as_finite_array(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array
