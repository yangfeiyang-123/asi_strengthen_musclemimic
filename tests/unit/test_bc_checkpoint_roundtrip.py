"""Roundtrip coverage for PPO-compatible BC checkpoints."""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from loco_mujoco.core.utils.env import Box
from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.checkpoint_manager import CheckpointMetadata, UnifiedCheckpointManager
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.ppo.config import PPOAgentState
from musclemimic.runner.eval_utils import load_checkpoint


class _TinyEnv:
    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        obs_space = Box(low=-jnp.ones(obs_dim), high=jnp.ones(obs_dim))
        act_space = Box(low=-jnp.ones(act_dim), high=jnp.ones(act_dim))
        self.info = types.SimpleNamespace(action_space=act_space, observation_space=obs_space)
        self.mdp_info = types.SimpleNamespace(observation_space=obs_space)


def _tiny_config():
    return OmegaConf.create(
        {
            "experiment": {
                "total_timesteps": 16,
                "num_envs": 2,
                "env_params": {
                    "env_name": "TinyDistillEnv",
                    "mjx_backend": "jax",
                },
                "ppo_config": {
                    "num_steps": 2,
                    "update_epochs": 1,
                    "num_minibatches": 1,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "clip_eps_vf": 0.2,
                    "init_std": 1.0,
                    "learnable_std": True,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                },
                "validation": {"num": 2},
                "actor_hidden_layers": [8],
                "critic_hidden_layers": [8],
                "activation": "tanh",
                "use_layernorm": False,
                "layernorm_eps": 1e-5,
                "lr": 3e-4,
                "anneal_lr": False,
                "lr_schedule_type": "linear",
                "warmup_steps": None,
                "min_lr_ratio": 0.0,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "optimizer_type": "adamw",
            }
        }
    )


def test_ppo_compatible_bc_checkpoint_loads_and_forwards(tmp_path):
    env = _TinyEnv()
    config = _tiny_config()
    agent_conf = PPOJax.init_agent_conf(env, config)
    init_vars = agent_conf.network.init(jax.random.PRNGKey(0), jnp.zeros((4,), dtype=jnp.float32))
    train_state = TrainState.create(
        apply_fn=agent_conf.network.apply,
        params=init_vars["params"],
        run_stats=init_vars["run_stats"],
        tx=optax.adam(3e-4),
    )
    agent_state = PPOAgentState(train_state=train_state)
    metadata = CheckpointMetadata(
        step=int(train_state.step),
        update_number=1,
        global_timestep=0,
        target_global_timestep=0,
        learning_rate=3e-4,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        backend="jax",
        env_name="TinyDistillEnv",
    )
    manager = UnifiedCheckpointManager(str(tmp_path / "checkpoints"), max_to_keep=5, async_save=False)
    try:
        checkpoint_path = manager.save_checkpoint(int(train_state.step), agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_config, loaded_agent_state, loaded_metadata = load_checkpoint(checkpoint_path)
    loaded_agent_conf = PPOJax.init_agent_conf(env, loaded_config)
    pi, value = loaded_agent_conf.network.apply(
        {
            "params": loaded_agent_state.train_state.params,
            "run_stats": loaded_agent_state.train_state.run_stats,
        },
        jnp.zeros((4,), dtype=jnp.float32),
    )

    assert loaded_metadata.env_name == "TinyDistillEnv"
    assert pi.mean().shape == (2,)
    assert value.shape == ()
