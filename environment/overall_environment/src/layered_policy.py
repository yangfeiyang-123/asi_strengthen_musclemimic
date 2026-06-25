from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.overall_environment.src.layered_control import LayeredActuatorRouter


@dataclass(frozen=True)
class LayeredPolicyOutput:
    body_action: np.ndarray
    grip_action: np.ndarray
    full_action: np.ndarray


class LatentBodyPolicy:
    """Body policy adapter: high-level latent residual -> LAB decoded body action."""

    def __init__(
        self,
        *,
        high_level_policy,
        lab_wrapper,
        state_adapter=None,
        expected_body_size: int | None = None,
    ) -> None:
        self.high_level_policy = high_level_policy.eval() if hasattr(high_level_policy, "eval") else high_level_policy
        self.lab_wrapper = lab_wrapper
        self.state_adapter = state_adapter
        self._expected_body_size = expected_body_size

    @classmethod
    def validated(
        cls,
        *,
        high_level_policy,
        lab_wrapper,
        router: "LayeredActuatorRouter",
        state_adapter=None,
    ) -> "LatentBodyPolicy":
        return cls(
            high_level_policy=high_level_policy,
            lab_wrapper=lab_wrapper,
            state_adapter=state_adapter,
            expected_body_size=router.body_size,
        )

    def eval(self):
        return self

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=float)
        raw_latent = np.asarray(self.high_level_policy.act(obs_array), dtype=float)
        state = obs_array if self.state_adapter is None else np.asarray(self.state_adapter(obs_array), dtype=float)
        if not np.isfinite(state).all():
            raise ValueError("LAB state contains non-finite values")
        body_action = np.asarray(self.lab_wrapper(state, raw_latent), dtype=float)
        if not np.isfinite(body_action).all():
            raise ValueError("LAB decoded body action contains non-finite values")
        if self._expected_body_size is not None and body_action.shape != (self._expected_body_size,):
            raise ValueError(
                f"LAB decoder output shape {body_action.shape} does not match "
                f"expected body_size ({self._expected_body_size},)"
            )
        return body_action


class LayeredPolicy:
    def __init__(self, *, body_policy, grip_policy, router: LayeredActuatorRouter) -> None:
        self.body_policy = body_policy.eval()
        self.grip_policy = grip_policy
        self.router = router

    def act(self, obs: np.ndarray) -> LayeredPolicyOutput:
        body_action = np.asarray(self.body_policy.act(obs), dtype=float)
        grip_action = np.asarray(self.grip_policy.act(obs), dtype=float)
        full_action = self.router.merge(body_action=body_action, grip_action=grip_action)
        return LayeredPolicyOutput(body_action=body_action, grip_action=grip_action, full_action=full_action)


class ZeroGripPolicy:
    def __init__(self, action_size: int) -> None:
        self.action_size = int(action_size)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_size, dtype=float)


RandomGripPolicy = ZeroGripPolicy


class TorchGripPolicy:
    def __init__(self, *, model, obs_rms) -> None:
        self.model = model
        self.obs_rms = obs_rms

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch

        obs_norm = self.obs_rms.normalize(np.asarray(obs, dtype=float))
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            mean, _, _ = self.model(obs_tensor)
            action = torch.clamp(mean, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy().astype(float)


def load_grip_policy(path, *, obs_size: int, action_size: int, allow_random_init: bool = False):
    policy_path = Path(path)
    if not policy_path.is_file():
        if allow_random_init:
            return ZeroGripPolicy(action_size)
        raise FileNotFoundError(policy_path)

    import torch
    from src.grip.train_right_hand_racket_grip_policy import PolicyValueNet, RunningMeanStd

    checkpoint = torch.load(policy_path, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    if int(metrics.get("obs_size", obs_size)) != int(obs_size):
        raise ValueError("grip policy obs_size does not match target obs_size")
    if int(metrics.get("action_size", action_size)) != int(action_size):
        raise ValueError("grip policy action_size does not match target action_size")

    ppo = metrics.get("ppo", {})
    hidden_sizes = tuple(int(value) for value in ppo.get("hidden_sizes", (256, 256)))
    action_std_init = float(ppo.get("action_std_init", 0.35))
    model = PolicyValueNet(obs_size, action_size, hidden_sizes, action_std_init)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs_rms = RunningMeanStd((obs_size,))
    state = checkpoint["obs_rms"]
    obs_rms.mean = np.asarray(state["mean"], dtype=np.float64)
    obs_rms.var = np.asarray(state["var"], dtype=np.float64)
    obs_rms.count = float(state["count"])
    return TorchGripPolicy(model=model, obs_rms=obs_rms)
