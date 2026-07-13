"""Train-only observation normalization shared by latent training/runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray
    count: int
    epsilon: float = 1e-6
    clip: float = 10.0

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError(f"normalizer mean/std must be equal rank-1 shapes, got {mean.shape}/{std.shape}")
        if int(self.count) <= 0:
            raise ValueError("normalizer count must be positive")
        if float(self.epsilon) <= 0.0 or float(self.clip) <= 0.0:
            raise ValueError("normalizer epsilon and clip must be positive")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("normalizer mean/std must be finite and std must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "clip", float(self.clip))

    @property
    def state_dim(self) -> int:
        return int(self.mean.size)

    @classmethod
    def fit(cls, values: Any, *, epsilon: float = 1e-6, clip: float = 10.0) -> "ObservationNormalizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"normalizer training values must have shape (N, D), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("normalizer training values contain non-finite values")
        variance = np.var(array, axis=0)
        std = np.sqrt(np.maximum(variance, float(epsilon) ** 2))
        return cls(
            mean=np.mean(array, axis=0).astype(np.float32),
            std=std.astype(np.float32),
            count=int(array.shape[0]),
            epsilon=float(epsilon),
            clip=float(clip),
        )

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "ObservationNormalizer":
        if payload is None:
            raise ValueError("latent checkpoint is missing obs_norm.json")
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            count=int(payload.get("count", 1)),
            epsilon=float(payload.get("epsilon", 1e-6)),
            clip=float(payload.get("clip", 10.0)),
        )

    def normalize_numpy(self, state: Any) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32)
        self._check_last_dim(array)
        return np.clip((array - self.mean) / self.std, -self.clip, self.clip)

    def normalize_jax(self, state):
        array = jnp.asarray(state, dtype=jnp.float32)
        if array.shape[-1] != self.state_dim:
            raise ValueError(f"state last dimension must be {self.state_dim}, got {array.shape}")
        return jnp.clip(
            (array - jnp.asarray(self.mean)) / jnp.asarray(self.std),
            -self.clip,
            self.clip,
        )

    def to_manifest(self, *, source_split: str = "train") -> dict[str, Any]:
        return {
            "schema_version": "observation_normalizer_v1",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "source_split": str(source_split),
            "state_dim": self.state_dim,
        }

    def _check_last_dim(self, value: np.ndarray) -> None:
        if value.ndim < 1 or value.shape[-1] != self.state_dim:
            raise ValueError(f"state last dimension must be {self.state_dim}, got {value.shape}")
