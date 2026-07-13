"""Optional MyoFullBody smoke tests for student observation filtering.

These tests require the local GMR/AMASS caches referenced by the badminton
ForehandClear config. They skip cleanly when those resources are unavailable.
"""

import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path


@pytest.mark.integration
def test_forehandclear_student_filter_phase_smoke():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import wrap_env
    from musclemimic.runner.engine import instantiate_env

    fullbody_dir = Path(__file__).resolve().parents[2] / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(fullbody_dir)):
        cfg = compose(config_name="config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo")

    cfg.experiment.env_params.num_envs = 1
    cfg.experiment.num_envs = 1
    # This is a one-environment shape smoke, not the production PPO batch.
    # Keep its reduced 1 * 80 rollout divisible after shrinking from the
    # canonical 256 environments.
    cfg.experiment.ppo_config.num_minibatches = 1
    cfg.experiment.normalize_env = False
    cfg.experiment.validation.active = False

    try:
        env = instantiate_env(cfg)
    except Exception as exc:
        pytest.skip(f"ForehandClear MyoFullBody resources unavailable: {exc}")

    wrapped = wrap_env(env, cfg.experiment)
    obs, _state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    obs = jnp.asarray(obs)

    assert obs.shape[-1] == wrapped.info.observation_space.shape[0]
    phase = obs[..., -1]
    assert bool(jnp.all(phase >= 0.0))
    assert bool(jnp.all(phase <= 1.0))

    agent_conf = PPOJax.init_agent_conf(env, cfg)
    init_vars = agent_conf.network.init(jax.random.PRNGKey(1), jnp.zeros((wrapped.info.observation_space.shape[0],)))
    assert "params" in init_vars
