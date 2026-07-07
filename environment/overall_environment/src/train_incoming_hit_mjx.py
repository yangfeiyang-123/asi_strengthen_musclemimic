"""Standalone GPU PPO trainer for the incoming-shuttle hit task.

PureJaxRL-style: the whole train iteration (rollout via vmapped env step +
GAE + minibatched PPO updates) is one jitted function. Independent from the
musclemimic trajectory-tracking pipeline.

Policy/value: shared-input MLPs; tanh-squashed Gaussian policy with a
state-independent learned log-std (same semantics as the CPU
``PolicyValueNet``). Observation normalization uses a running mean/std
updated from each rollout batch.

Run from the repository root (GPU env: source configs/env.sh):

    .venv/bin/python -m environment.overall_environment.src.train_incoming_hit_mjx \
        --spec experiments/posttrain/incoming_shuttle_hit_v1.yaml \
        --num-envs 512 --total-env-steps 2000000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    EnvState,
    IncomingHitMjxEnv,
)


# ---------------------------------------------------------------------------
# networks (plain pytree params)
# ---------------------------------------------------------------------------


def _init_mlp(key, sizes):
    params = []
    for k_in, k_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / k_in)
        params.append(
            {
                "w": jax.random.normal(sub, (k_in, k_out)) * scale,
                "b": jnp.zeros(k_out),
            }
        )
    return params


def _mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]


def init_agent(key, obs_size, action_size, hidden, action_std_init):
    k1, k2 = jax.random.split(key)
    policy = _init_mlp(k1, (obs_size, *hidden, action_size))
    value = _init_mlp(k2, (obs_size, *hidden, 1))
    policy[-1]["w"] = policy[-1]["w"] * 0.01
    return {
        "policy": policy,
        "value": value,
        "log_std": jnp.full((action_size,), jnp.log(action_std_init)),
    }


def _dist(params, obs):
    mean = _mlp(params["policy"], obs)
    std = jnp.exp(params["log_std"])
    return mean, std


def sample_action(params, obs, key):
    mean, std = _dist(params, obs)
    raw = mean + std * jax.random.normal(key, mean.shape)
    action = jnp.tanh(raw)
    logp = _tanh_normal_logprob(mean, std, raw, action)
    return action, raw, logp


def _tanh_normal_logprob(mean, std, raw, squashed):
    base = -0.5 * (((raw - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    correction = jnp.log(1.0 - squashed**2 + 1e-6)
    return jnp.sum(base - correction, axis=-1)


def evaluate_actions(params, obs, raw_actions):
    mean, std = _dist(params, obs)
    squashed = jnp.tanh(raw_actions)
    logp = _tanh_normal_logprob(mean, std, raw_actions, squashed)
    entropy = jnp.sum(0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(std), axis=-1)
    value = _mlp(params["value"], obs)[..., 0]
    return logp, entropy, value


# ---------------------------------------------------------------------------
# running obs normalization
# ---------------------------------------------------------------------------


class ObsRms(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @staticmethod
    def create(obs_size):
        return ObsRms(jnp.zeros(obs_size), jnp.ones(obs_size), jnp.asarray(1e-4))

    def update(self, batch: jnp.ndarray) -> "ObsRms":
        batch = batch.reshape(-1, batch.shape[-1])
        b_mean = batch.mean(0)
        b_var = batch.var(0)
        b_count = batch.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta**2 * self.count * b_count / tot
        return ObsRms(mean, m2 / tot, tot)

    def normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip((obs - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class TrainConfig(NamedTuple):
    num_envs: int = 512
    rollout_steps: int = 64
    total_env_steps: int = 2_000_000
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: tuple = (256, 256)
    action_std_init: float = 0.35
    seed: int = 0


def make_train_iteration(env: IncomingHitMjxEnv, mx, cfg: TrainConfig, optimizer):
    step_env = env.make_step_fn(mx, cfg.num_envs)

    def rollout(agent, obs_rms, env_states, key):
        def body(carry, _):
            env_states, key = carry
            key, sub = jax.random.split(key)
            obs_norm = obs_rms.normalize(env_states.obs)
            action, raw, logp = sample_action(
                agent, obs_norm, sub
            )
            value = _mlp(agent["value"], obs_norm)[..., 0]
            next_states, tr = step_env(env_states, action)
            record = {
                "obs_norm": obs_norm,
                "raw_action": raw,
                "logp": logp,
                "value": value,
                "reward": tr["reward"],
                "done": tr["done"],
                "terminated": tr["terminated"],
                "hit": tr["hit"],
                "crossed_net": tr["crossed_net"],
                "landing_score": tr["landing_score"],
                "obs_raw": env_states.obs,
            }
            return (next_states, key), record

        (env_states, key), records = jax.lax.scan(
            body, (env_states, key), None, length=cfg.rollout_steps
        )
        return env_states, key, records

    def gae(records, last_value):
        def body(carry, step):
            adv = carry
            delta = (
                step["reward"]
                + cfg.gamma * step["next_value"] * (1.0 - step["terminated"])
                - step["value"]
            )
            adv = delta + cfg.gamma * cfg.gae_lambda * (1.0 - step["done"]) * adv
            return adv, adv

        next_values = jnp.concatenate(
            [records["value"][1:], last_value[None]], axis=0
        )
        steps = {
            "reward": records["reward"],
            "value": records["value"],
            "next_value": next_values,
            "done": records["done"].astype(jnp.float32),
            "terminated": records["terminated"].astype(jnp.float32),
        }
        _, advantages = jax.lax.scan(
            body,
            jnp.zeros_like(last_value),
            jax.tree_util.tree_map(lambda x: x[::-1], steps),
        )
        advantages = advantages[::-1]
        returns = advantages + records["value"]
        return advantages, returns

    def ppo_update(agent, opt_state, batch, key):
        num_samples = cfg.rollout_steps * cfg.num_envs
        mb_size = num_samples // cfg.num_minibatches

        def epoch(carry, _):
            agent, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, num_samples)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)

            def minibatch(carry, mb):
                agent, opt_state = carry

                def loss_fn(params):
                    logp, entropy, value = evaluate_actions(
                        params, mb["obs_norm"], mb["raw_action"]
                    )
                    ratio = jnp.exp(logp - mb["logp"])
                    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                    pg1 = -adv * ratio
                    pg2 = -adv * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    policy_loss = jnp.maximum(pg1, pg2).mean()
                    value_loss = 0.5 * jnp.square(value - mb["returns"]).mean()
                    entropy_loss = -entropy.mean()
                    total = (
                        policy_loss
                        + cfg.value_coef * value_loss
                        + cfg.entropy_coef * entropy_loss
                    )
                    return total, (policy_loss, value_loss, entropy_loss)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent)
                updates, opt_state = optimizer.update(grads, opt_state, agent)
                agent = optax.apply_updates(agent, updates)
                return (agent, opt_state), (loss, *aux)

            mbs = jax.tree_util.tree_map(
                lambda x: x[: mb_size * cfg.num_minibatches].reshape(
                    cfg.num_minibatches, mb_size, *x.shape[1:]
                ),
                shuffled,
            )
            (agent, opt_state), losses = jax.lax.scan(minibatch, (agent, opt_state), mbs)
            return (agent, opt_state, key), losses

        (agent, opt_state, key), losses = jax.lax.scan(
            epoch, (agent, opt_state, key), None, length=cfg.update_epochs
        )
        return agent, opt_state, key, losses

    @jax.jit
    def train_iteration(agent, opt_state, obs_rms, env_states, key):
        env_states, key, records = rollout(agent, obs_rms, env_states, key)
        obs_rms = obs_rms.update(records["obs_raw"])
        last_obs_norm = obs_rms.normalize(env_states.obs)
        last_value = _mlp(agent["value"], last_obs_norm)[..., 0]
        advantages, returns = gae(records, last_value)

        flat = lambda x: x.reshape(-1, *x.shape[2:])
        batch = {
            "obs_norm": flat(records["obs_norm"]),
            "raw_action": flat(records["raw_action"]),
            "logp": flat(records["logp"]),
            "adv": flat(advantages),
            "returns": flat(returns),
        }
        agent, opt_state, key, losses = ppo_update(agent, opt_state, batch, key)

        done = records["done"]
        n_done = jnp.maximum(done.sum(), 1)
        metrics = {
            "mean_reward": records["reward"].mean(),
            "episodes_finished": done.sum(),
            "hit_rate": jnp.where(done, records["hit"], 0).sum() / n_done,
            "crossed_net_rate": jnp.where(done, records["crossed_net"], 0).sum() / n_done,
            "mean_landing_score": records["landing_score"].sum() / n_done,
            "loss": losses[0].mean(),
            "policy_loss": losses[1].mean(),
            "value_loss": losses[2].mean(),
            "entropy_loss": losses[3].mean(),
        }
        return agent, opt_state, obs_rms, env_states, key, metrics

    return train_iteration


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, agent, obs_rms: ObsRms, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat, treedef = jax.tree_util.tree_flatten(agent)
    payload = {f"param_{i}": np.asarray(p) for i, p in enumerate(flat)}
    payload["obs_mean"] = np.asarray(obs_rms.mean)
    payload["obs_var"] = np.asarray(obs_rms.var)
    payload["obs_count"] = np.asarray(obs_rms.count)
    np.savez(path, **payload)
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, agent_template):
    with np.load(path) as payload:
        flat, treedef = jax.tree_util.tree_flatten(agent_template)
        flat = [jnp.asarray(payload[f"param_{i}"]) for i in range(len(flat))]
        agent = jax.tree_util.tree_unflatten(treedef, flat)
        obs_rms = ObsRms(
            jnp.asarray(payload["obs_mean"]),
            jnp.asarray(payload["obs_var"]),
            jnp.asarray(payload["obs_count"]),
        )
    return agent, obs_rms


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def train(
    env: IncomingHitMjxEnv,
    cfg: TrainConfig,
    out_dir: Path,
    *,
    log_every: int = 1,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    key, k_agent, k_env = jax.random.split(key, 3)

    mx = env.put_model(cfg.num_envs)
    template = env.make_batched_template(cfg.num_envs)
    reset_fn = env.make_reset_fn(mx, cfg.num_envs)
    env_states = jax.jit(reset_fn)(k_env, template)

    agent = init_agent(k_agent, env.observation_size, env.action_size, cfg.hidden, cfg.action_std_init)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate),
    )
    opt_state = optimizer.init(agent)
    obs_rms = ObsRms.create(env.observation_size)

    train_iteration = make_train_iteration(env, mx, cfg, optimizer)

    steps_per_iter = cfg.num_envs * cfg.rollout_steps
    num_iters = max(1, cfg.total_env_steps // steps_per_iter)
    metrics_path = out_dir / "metrics.jsonl"
    history: list[dict[str, float]] = []
    t_start = time.time()
    for it in range(1, num_iters + 1):
        t0 = time.time()
        agent, opt_state, obs_rms, env_states, key, metrics = train_iteration(
            agent, opt_state, obs_rms, env_states, key
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["iteration"] = it
        metrics["env_steps"] = it * steps_per_iter
        metrics["iter_seconds"] = time.time() - t0
        metrics["env_steps_per_second"] = steps_per_iter / metrics["iter_seconds"]
        history.append(metrics)
        if it % log_every == 0:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"iter {it}/{num_iters} steps={metrics['env_steps']:,} "
                f"reward={metrics['mean_reward']:.4f} hit={metrics['hit_rate']:.3f} "
                f"net={metrics['crossed_net_rate']:.3f} sps={metrics['env_steps_per_second']:,.0f}",
                flush=True,
            )
        if it % checkpoint_every == 0 or it == num_iters:
            save_checkpoint(
                out_dir / "policy_latest.npz",
                agent,
                obs_rms,
                {
                    "iteration": it,
                    "env_steps": it * steps_per_iter,
                    "obs_size": env.observation_size,
                    "action_size": env.action_size,
                    "hidden": list(cfg.hidden),
                    "config": cfg._asdict(),
                },
            )

    report = {
        "iterations": num_iters,
        "env_steps": num_iters * steps_per_iter,
        "wall_seconds": time.time() - t_start,
        "final": history[-1] if history else {},
        "checkpoint": str(out_dir / "policy_latest.npz"),
        "metrics_file": str(metrics_path),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", default="experiments/posttrain/incoming_shuttle_hit_v1.yaml"
    )
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--total-env-steps", type=int, default=2_000_000)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--base-policy-artifact", default=None, help="frozen base policy export dir (Stage 3 residual mode)")
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--base-skill", default=None, help="skill name for a multi-skill base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "musclemimic" / "badminton" / "scripts"))
    from run_incoming_shuttle_hit import _ensure_feed_bank, _ensure_scene, load_incoming_hit_spec

    paths = load_incoming_hit_spec(args.spec)
    _ensure_scene(paths)
    bank = _ensure_feed_bank(paths)

    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        impl=args.impl,
        base_policy_artifact=args.base_policy_artifact,
        residual_scale=args.residual_scale,
        base_skill=args.base_skill,
    )
    ppo = dict(paths.ppo_overrides)
    cfg = TrainConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_env_steps=args.total_env_steps,
        update_epochs=int(ppo.get("update_epochs", 4)),
        hidden=tuple(ppo.get("hidden_sizes", (256, 256))),
        action_std_init=float(ppo.get("action_std_init", 0.35)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        seed=args.seed,
    )
    out_dir = args.out_dir if args.out_dir is not None else paths.output_dir / "train_gpu"
    print("jax devices:", jax.devices())
    report = train(env, cfg, Path(out_dir), checkpoint_every=args.checkpoint_every)
    print(json.dumps(report["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
