from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.paths import REPO_ROOT, reference_json_path, scene_xml_path, target_config_path
from src.grip.right_hand_racket_grip_env import DEFAULT_TRAINING_CONFIG, RightHandRacketGripEnv

DEFAULT_POLICY_DIR = REPO_ROOT / "outputs" / "right_hand_racket_grip" / "policy"


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 1024
    minibatch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    action_std_init: float = 0.35
    seed: int = 0


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta**2 * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


def train_policy(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    training_config: str | Path = DEFAULT_TRAINING_CONFIG,
    out_dir: str | Path = DEFAULT_POLICY_DIR,
    *,
    total_steps: int | None = None,
    rollout_steps: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch.set_num_threads(1)

    ppo_config = _load_ppo_config(Path(training_config))
    if total_steps is not None:
        ppo_config = _replace_config(ppo_config, total_steps=_positive_int(total_steps, "total_steps"))
    if rollout_steps is not None:
        ppo_config = _replace_config(ppo_config, rollout_steps=_positive_int(rollout_steps, "rollout_steps"))
    if seed is not None:
        ppo_config = _replace_config(ppo_config, seed=int(seed))
    _validate_ppo_config(ppo_config)

    rng = np.random.default_rng(ppo_config.seed)
    torch.manual_seed(ppo_config.seed)
    env = RightHandRacketGripEnv(xml, targets, reference, training_config)
    obs, info = env.reset()
    obs_size = int(obs.size)
    action_size = int(env.action_size)
    obs_rms = RunningMeanStd((obs_size,))
    obs_rms.update(obs)

    model = PolicyValueNet(obs_size, action_size, ppo_config.hidden_sizes, ppo_config.action_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    update_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    last_info = info
    summaries: list[dict[str, float]] = []

    while global_step < ppo_config.total_steps:
        rollout_target = min(ppo_config.rollout_steps, ppo_config.total_steps - global_step)
        rollout = _empty_rollout(rollout_target, obs_size, action_size)
        for step in range(rollout_target):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalize(obs)
            action, logprob, value = _sample_action(torch, model, obs_norm, device, rng)
            next_obs, reward, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)

            rollout["obs"][step] = obs_norm
            rollout["actions"][step] = action
            rollout["logprobs"][step] = logprob
            rollout["rewards"][step] = float(reward)
            rollout["dones"][step] = float(done)
            rollout["values"][step] = value

            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                obs, last_info = env.reset()
            if global_step >= ppo_config.total_steps:
                break

        next_obs_norm = obs_rms.normalize(obs)
        with torch.no_grad():
            next_value = float(model.value(_tensor(torch, next_obs_norm, device).unsqueeze(0)).item())
        advantages, returns = _gae(rollout["rewards"], rollout["dones"], rollout["values"], next_value, ppo_config)
        update_summary = _ppo_update(torch, model, optimizer, rollout, advantages, returns, ppo_config, device)
        update_index += 1
        update_summary.update(
            {
                "update": float(update_index),
                "global_step": float(global_step),
                "mean_episode_return": _mean_last(completed_returns, 10),
                "mean_episode_length": _mean_last(completed_lengths, 10),
                "mean_rollout_reward": float(np.mean(rollout["rewards"])),
                "mean_site_error_m": float(last_info["mean_site_error_m"]),
                "contact_count": float(last_info["contact_count"]),
                "max_handle_penetration_m": float(last_info["max_handle_penetration_m"]),
                "grip_slip_m": float(last_info["grip_slip_m"]),
                "v_shape_error": float(last_info["v_shape_error"]),
            }
        )
        summaries.append(update_summary)
        print(json.dumps(update_summary, sort_keys=True), flush=True)

    metrics = {
        "mode": "ppo_right_hand_racket_grip",
        "xml": str(Path(xml)),
        "targets": str(Path(targets)),
        "reference": str(Path(reference)),
        "training_config": str(Path(training_config)),
        "out_dir": str(out_path),
        "ppo": asdict(ppo_config),
        "obs_size": obs_size,
        "action_size": action_size,
        "global_step": int(global_step),
        "updates": int(update_index),
        "mean_episode_return_last10": _mean_last(completed_returns, 10),
        "mean_episode_length_last10": _mean_last(completed_lengths, 10),
        "last_info": _json_safe(last_info),
        "updates_detail": summaries,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, out_path / "policy_latest.pt")
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...],
        action_std_init: float,
    ) -> None:
        super().__init__()
        self.policy_body = _mlp(obs_size, hidden_sizes)
        last_size = hidden_sizes[-1] if hidden_sizes else obs_size
        self.mean_head = nn.Linear(last_size, action_size)
        self.value_body = _mlp(obs_size, hidden_sizes)
        self.value_head = nn.Linear(last_size, 1)
        self.log_std = nn.Parameter(torch_log_std(action_std_init, action_size))

    def forward(self, obs):
        features = self.policy_body(obs)
        mean = torch_tanh(self.mean_head(features))
        value = self.value(obs)
        return mean, self.log_std.expand_as(mean), value

    def value(self, obs):
        return self.value_head(self.value_body(obs)).squeeze(-1)


def _mlp(input_size: int, hidden_sizes: tuple[int, ...]):
    layers = []
    last_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_size, hidden_size))
        layers.append(nn.Tanh())
        last_size = hidden_size
    return nn.Sequential(*layers)


def torch_log_std(action_std_init: float, action_size: int):
    return torch.full((action_size,), math.log(action_std_init), dtype=torch.float32)


def torch_tanh(value):
    return torch.tanh(value)


def _sample_action(torch, model: PolicyValueNet, obs_norm: np.ndarray, device: str, rng: np.random.Generator):
    with torch.no_grad():
        obs_tensor = _tensor(torch, obs_norm, device).unsqueeze(0)
        mean, log_std, value = model(obs_tensor)
        std = torch.exp(log_std)
        noise = torch.as_tensor(rng.standard_normal(mean.shape), dtype=torch.float32, device=device)
        raw_action = mean + noise * std
        distribution = torch.distributions.Normal(mean, std)
        logprob = distribution.log_prob(raw_action).sum(axis=-1)
        action = torch.clamp(raw_action, -1.0, 1.0)
    return (
        action.squeeze(0).cpu().numpy().astype(np.float64),
        float(logprob.item()),
        float(value.item()),
    )


def _empty_rollout(steps: int, obs_size: int, action_size: int) -> dict[str, np.ndarray]:
    return {
        "obs": np.zeros((steps, obs_size), dtype=np.float32),
        "actions": np.zeros((steps, action_size), dtype=np.float32),
        "logprobs": np.zeros((steps,), dtype=np.float32),
        "rewards": np.zeros((steps,), dtype=np.float32),
        "dones": np.zeros((steps,), dtype=np.float32),
        "values": np.zeros((steps,), dtype=np.float32),
    }


def _gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    config: PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - float(dones[step])
        next_values = next_value if step == rewards.shape[0] - 1 else float(values[step + 1])
        delta = float(rewards[step]) + config.gamma * next_values * next_nonterminal - float(values[step])
        last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def _ppo_update(
    torch,
    model: PolicyValueNet,
    optimizer,
    rollout: dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    config: PPOConfig,
    device: str,
) -> dict[str, float]:
    obs = _tensor(torch, rollout["obs"], device)
    actions = _tensor(torch, rollout["actions"], device)
    old_logprobs = _tensor(torch, rollout["logprobs"], device)
    returns_tensor = _tensor(torch, returns, device)
    advantages_tensor = _tensor(torch, advantages, device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    batch_size = obs.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(config.update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            mean, log_std, values = model(obs[batch_indices])
            std = torch.exp(log_std)
            distribution = torch.distributions.Normal(mean, std)
            new_logprob = distribution.log_prob(actions[batch_indices]).sum(axis=-1)
            entropy = distribution.entropy().sum(axis=-1).mean()
            log_ratio = new_logprob - old_logprobs[batch_indices]
            ratio = torch.exp(log_ratio)
            minibatch_advantages = advantages_tensor[batch_indices]
            unclipped = -minibatch_advantages * ratio
            clipped = -minibatch_advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            policy_loss = torch.max(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(values - returns_tensor[batch_indices]).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return {
        "loss": _mean_or_zero(losses),
        "policy_loss": _mean_or_zero(policy_losses),
        "value_loss": _mean_or_zero(value_losses),
        "entropy": _mean_or_zero(entropies),
    }


def _tensor(torch, values: np.ndarray, device: str):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _load_ppo_config(path: Path) -> PPOConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"training config root must be a mapping: {path}")
    ppo_raw = raw.get("ppo", {})
    if not isinstance(ppo_raw, dict):
        raise ValueError("training config ppo must be a mapping")
    values = {**asdict(PPOConfig()), **ppo_raw}
    hidden_sizes = values.get("hidden_sizes", (256, 256))
    if isinstance(hidden_sizes, list):
        hidden_sizes = tuple(int(value) for value in hidden_sizes)
    values["hidden_sizes"] = hidden_sizes
    return PPOConfig(**values)


def _replace_config(config: PPOConfig, **updates: object) -> PPOConfig:
    values = asdict(config)
    values.update(updates)
    hidden_sizes = values["hidden_sizes"]
    if isinstance(hidden_sizes, list):
        values["hidden_sizes"] = tuple(hidden_sizes)
    return PPOConfig(**values)


def _validate_ppo_config(config: PPOConfig) -> None:
    _positive_int(config.total_steps, "ppo.total_steps")
    _positive_int(config.rollout_steps, "ppo.rollout_steps")
    _positive_int(config.minibatch_size, "ppo.minibatch_size")
    _positive_int(config.update_epochs, "ppo.update_epochs")
    for name in ("gamma", "gae_lambda", "clip_coef", "value_coef", "learning_rate", "max_grad_norm", "action_std_init"):
        _positive_float(getattr(config, name), f"ppo.{name}")
    if config.entropy_coef < 0:
        raise ValueError("ppo.entropy_coef must be >= 0")
    if not config.hidden_sizes or any(int(size) <= 0 for size in config.hidden_sizes):
        raise ValueError("ppo.hidden_sizes must contain positive layer sizes")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def _mean_last(values: list[float] | list[int], count: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values[-count:], dtype=float)))


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a right-hand badminton racket grip policy with CPU MuJoCo PPO.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--reference", type=Path, default=reference_json_path(), help="Grip reference JSON path.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG, help="Training YAML path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_POLICY_DIR, help="Output policy directory.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override ppo.total_steps.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override ppo.rollout_steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override ppo.seed.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = train_policy(
        args.xml,
        args.targets,
        args.reference,
        args.training_config,
        args.out_dir,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
