"""Environment adapter for high-level latent actions decoded through LAB."""

from __future__ import annotations

from typing import Any

import numpy as np

from musclemimic.latent_muscle.action_mask import ActionMask


class LABEnvWrapper:
    """Wrap an env so ``step`` accepts high-level latent/correction actions."""

    def __init__(
        self,
        *,
        env: Any,
        action_wrapper: Any,
        action_mask: ActionMask | None = None,
        state_adapter=None,
    ) -> None:
        self.env = env
        self.action_wrapper = action_wrapper
        self.action_mask = action_mask
        self.state_adapter = state_adapter

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def step(self, state, high_level_action):
        lab_state = np.asarray(state, dtype=float)
        if self.state_adapter is not None:
            lab_state = np.asarray(self.state_adapter(lab_state), dtype=float)
        output = self.action_wrapper(lab_state, high_level_action, return_info=True)
        full_action = np.asarray(output.action, dtype=float)
        if self.action_mask is not None and full_action.shape != (self.action_mask.action_size,):
            raise ValueError(
                f"LAB decoded action shape {full_action.shape} does not match "
                f"full_action_dim ({self.action_mask.action_size},)"
            )
        result = self.env.step(state, full_action)
        return _attach_lab_info(result, output)


def _attach_lab_info(result, output):
    lab_info = {
        "raw_latent_norm": float(np.linalg.norm(output.raw_latent)),
        "lab_latent_norm": float(np.linalg.norm(output.latent)),
        "prior_sigma_mean": float(np.mean(output.prior_sigma)),
        "body_action_norm": float(np.linalg.norm(output.body_action)),
        "correction_action_norm": 0.0 if output.correction is None else float(np.linalg.norm(output.correction)),
    }
    if not isinstance(result, tuple) or len(result) == 0:
        return result
    items = list(result)
    if isinstance(items[-1], dict):
        info = dict(items[-1])
        info["lab"] = lab_info
        items[-1] = info
    else:
        items.append({"lab": lab_info})
    return tuple(items)
