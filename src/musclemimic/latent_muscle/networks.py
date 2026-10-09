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
    """q(z | state, reference_features[, emg_context]) used only during distillation.

    ``emg_context`` is privileged training-time input: a phase-queried, low
    dimensional EMG coordination summary, never a raw waveform.  It is absent
    from :class:`ConditionalPrior`, so the deployed prior/decoder pair never
    requires EMG and the KL term is what transfers the structure.
    """

    latent_dim: int
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    use_layernorm: bool = False
    sigma_min: float = 0.05
    sigma_max: float = 2.0

    @nn.compact
    def __call__(self, state, reference_features, emg_context=None):
        pieces = [jnp.asarray(state), jnp.asarray(reference_features)]
        if emg_context is not None:
            pieces.append(jnp.asarray(emg_context))
        x = jnp.concatenate(pieces, axis=-1)
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


class SynergyHead(nn.Module):
    """G(z) -> non-negative synergy coefficients, a readout not a constraint.

    The head deliberately does not force the leading latent dimensions to equal
    the measured synergy coefficients: the latent must still encode unmeasured
    muscles, balance, dynamics compensation and reference detail.  A single
    softplus layer keeps the physiological signal interpretable while leaving
    the rest of the latent capacity free.
    """

    synergy_dim: int

    @nn.compact
    def __call__(self, latent):
        return nn.softplus(
            nn.Dense(
                int(self.synergy_dim),
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name="synergy",
            )(jnp.asarray(latent))
        )


def emg_context_dropout(rng, context, *, dropout_rate: float):
    """Zero the whole EMG context per sample with probability ``dropout_rate``.

    Modality dropout, not element dropout: the posterior must stay usable when
    the channel is absent, which both narrows the prior/posterior deployment gap
    and simulates missing electrodes.  ``dropout_rate=0`` is an exact no-op and
    ``1.0`` blanks every row, which is the negative control.
    """

    values = jnp.asarray(context)
    rate = float(dropout_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("emg context dropout_rate must lie in [0, 1]")
    if rate == 0.0:
        return values, jnp.ones(values.shape[:-1], dtype=values.dtype)
    keep = (jax.random.uniform(rng, shape=values.shape[:-1], dtype=values.dtype) >= rate).astype(values.dtype)
    return values * keep[..., None], keep


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
