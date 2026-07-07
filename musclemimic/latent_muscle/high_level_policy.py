"""High-level latent policy adapters for LAB deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HighLevelLatentAction:
    latent: np.ndarray
    correction: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        payload = {"latent": np.asarray(self.latent, dtype=float)}
        if self.correction is not None:
            payload["correction"] = np.asarray(self.correction, dtype=float)
        return payload


class HighLevelLatentPolicy:
    """Adapter around a policy whose ``act`` output is a raw latent action."""

    def __init__(self, policy: Any, *, latent_dim: int, correction_dim: int = 0) -> None:
        self.policy = policy.eval() if hasattr(policy, "eval") else policy
        self.latent_dim = int(latent_dim)
        self.correction_dim = int(correction_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.correction_dim < 0:
            raise ValueError("correction_dim must be non-negative")

    def eval(self):
        return self

    def act(self, obs: np.ndarray) -> HighLevelLatentAction:
        raw = np.asarray(self.policy.act(obs), dtype=float)
        expected = self.latent_dim + self.correction_dim
        if raw.shape != (expected,):
            raise ValueError(f"high-level policy action must have shape ({expected},), got {raw.shape}")
        latent = raw[: self.latent_dim]
        correction = raw[self.latent_dim :] if self.correction_dim else None
        return HighLevelLatentAction(latent=latent, correction=correction)
