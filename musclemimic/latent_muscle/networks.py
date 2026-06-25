"""Flax modules for latent muscle skill distillation."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


def _activation(name: str):
    try:
        return getattr(nn, name)
    except AttributeError as exc:
        raise ValueError(f"Activation function {name!r} not found in flax.linen") from exc


class _MLP(nn.Module):
    hidden_layer_dims: Sequence[int]
    activation: str = "tanh"
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        act = _activation(self.activation)
        for index, dim in enumerate(self.hidden_layer_dims):
            x = nn.Dense(
                int(dim),
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"dense_{index}",
            )(x)
            if self.use_layernorm:
                x = nn.LayerNorm(epsilon=self.layernorm_eps, name=f"ln_{index}")(x)
            x = act(x)
        return x


class PosteriorEncoder(nn.Module):
    """q(z | state, reference_features) used only during distillation."""

    latent_dim: int
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    use_layernorm: bool = False
    sigma_min: float = 0.05
    sigma_max: float = 2.0

    @nn.compact
    def __call__(self, state, reference_features):
        x = jnp.concatenate([jnp.asarray(state), jnp.asarray(reference_features)], axis=-1)
        x = _MLP(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name="trunk",
        )(x)
        mu = nn.Dense(self.latent_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="mu")(x)
        raw_sigma = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(_inverse_softplus(0.5)),
            name="raw_sigma",
        )(x)
        raw_sigma = _clamp_raw_sigma(raw_sigma, self.sigma_min, self.sigma_max)
        return mu, raw_sigma


class ConditionalPrior(nn.Module):
    """p(z | state), used by LAB during high-level PPO deployment."""

    latent_dim: int
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    use_layernorm: bool = False
    sigma_min: float = 0.05
    sigma_max: float = 2.0

    @nn.compact
    def __call__(self, state):
        x = _MLP(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name="trunk",
        )(jnp.asarray(state))
        mu = nn.Dense(self.latent_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="mu")(x)
        raw_sigma = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(_inverse_softplus(0.5)),
            name="raw_sigma",
        )(x)
        raw_sigma = _clamp_raw_sigma(raw_sigma, self.sigma_min, self.sigma_max)
        return mu, raw_sigma


class LatentDecoder(nn.Module):
    """D(state, z) -> body action."""

    action_dim: int
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    bounded_action: bool = True
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, state, latent):
        x = jnp.concatenate([jnp.asarray(state), jnp.asarray(latent)], axis=-1)
        x = _MLP(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name="trunk",
        )(x)
        raw = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="action")(x)
        return jnp.tanh(raw) if self.bounded_action else raw


def reparameterize_gaussian(rng, mu, raw_sigma, *, sigma_min: float = 0.05, sigma_max: float = 2.0):
    """Sample z = mu + clip(softplus(raw_sigma), sigma_min, sigma_max) * eps."""
    eps = jax.random.normal(rng, shape=jnp.shape(mu), dtype=jnp.asarray(mu).dtype)
    sigma = jnp.clip(jax.nn.softplus(raw_sigma), float(sigma_min), float(sigma_max))
    return mu + sigma * eps


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(float(value))))


def _clamp_raw_sigma(raw_sigma, sigma_min: float, sigma_max: float):
    raw_min = _inverse_softplus(float(sigma_min))
    raw_max = _inverse_softplus(float(sigma_max))
    return jnp.clip(raw_sigma, raw_min, raw_max)
