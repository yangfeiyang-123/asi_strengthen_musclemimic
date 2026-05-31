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


class RandomGripPolicy:
    def __init__(self, action_size: int) -> None:
        self.action_size = int(action_size)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_size, dtype=float)


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
            return RandomGripPolicy(action_size)
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
