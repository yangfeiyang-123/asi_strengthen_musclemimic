"""Zero-shot racket probe for a body-only Stage-1 checkpoint.

Runs the policy unchanged inside ``MjxMyoFullBodyRacket`` (rigid racket, grip v2 preset /
attachment v4 contract, ``RacketMimicReward`` with the ``derived_rigid`` racket reference) on
the 20 held-out aug100 motions at several racket mass scales, and reports how many motions
terminate early plus the racket position / rotation tracking error.  This is the starting
point of the racket mass curriculum: no training happens here.

Usage (from the repo root)::

    source .venv/bin/activate && source configs/env.sh
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python experiments/stage2/racket_zero_shot_t3.py --masses 0.25,1.0

Output: ``outputs/stage2_racket_zero_shot/<ckpt-tag>_zero_shot.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_CKPT = REPO / "checkpoints/stage1/seed0/T3/checkpoint_39063"
OUT_DIR = REPO / "outputs/stage2_racket_zero_shot"
RACKET_ERR_KEYS = ("err_racket_pos", "err_racket_rot")


def _patch_config_for_racket(config, mass_scale: float):
    """Turn a body-only checkpoint config into the racket-holding validation config."""
    from omegaconf import OmegaConf

    OmegaConf.set_struct(config, False)
    env_params = config.experiment.env_params
    env_params.env_name = "MjxMyoFullBodyRacket"
    env_params.reward_type = "RacketMimicReward"
    env_params.racket_mass_scale = float(mass_scale)
    reward = env_params.reward_params
    # Same racket weights as base/conf_fullbody_badminton_racket_gmr.yaml.
    reward.racket_pos_w_sum = 0.3
    reward.racket_pos_w_exp = 50.0
    reward.racket_rot_w_sum = 0.15
    reward.racket_rot_w_exp = 5.0
    # The EMG reward is a Stage-1 body-only treatment; it is not part of the racket stage.
    if "emg_consistency" in reward:
        reward.pop("emg_consistency")
    return config


def _unwrap(state):
    seen = 0
    while hasattr(state, "env_state") and seen < 64:
        state = state.env_state
        seen += 1
    return state


def run_mass(checkpoint: Path, mass_scale: float) -> dict:
    import jax
    import jax.numpy as jnp

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import wrap_env
    from musclemimic.algorithms.ppo.runner import (
        _apply_frozen_eval_policy,
        _reset_eval_all_batch_jitted,
        _tree_where_batch,
    )
    from musclemimic.runner.engine import instantiate_validation_env
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint, resolve_checkpoint_path

    t0 = time.time()
    config, agent_state, _meta = load_checkpoint(str(Path(resolve_checkpoint_path(str(checkpoint))).resolve()))
    config = _patch_config_for_racket(config, mass_scale)

    env = instantiate_validation_env(config, share_trajectory=False)
    assert env is not None and getattr(env, "th", None) is not None, "no held-out split in checkpoint config"
    print(
        f"[mass {mass_scale}] env built: obs={env.info.observation_space.shape} "
        f"act={env.info.action_space.shape} n_traj={env.th.n_trajectories}",
        flush=True,
    )
    agent_conf = PPOJax.init_agent_conf(env, config)
    state = align_agent_state(agent_state, agent_conf)
    params = state.train_state.params
    run_stats = state.train_state.run_stats
    network = agent_conf.network

    if env.mjx_enabled and env.th.is_numpy:
        env.th.to_jax()
    n_traj = int(env.th.n_trajectories)
    lens = [int(env.th.len_trajectory(i)) for i in range(n_traj)]
    horizon = max(lens)
    if int(env.info.horizon) < horizon:
        env._mdp_info.horizon = horizon
    val_env = wrap_env(env, config.experiment)

    idx = jnp.arange(n_traj, dtype=jnp.int32)
    reset_keys = jax.random.split(jax.random.key(0), n_traj)
    obs, env_state = _reset_eval_all_batch_jitted(val_env, reset_keys, idx)

    def _rollout(params, run_stats, obs, env_state):
        def body(carry, _):
            cur_obs, cur_state, completed, rng = carry
            rng, _ = jax.random.split(rng)
            pi, _value = _apply_frozen_eval_policy(network, params, run_stats, cur_obs)
            action = jnp.where(completed[:, None], 0.0, pi.mode())
            next_obs, reward, absorbing, done, info, next_state, _transition = val_env.step_with_transition(
                cur_state, action
            )
            valid = ~completed
            next_completed = completed | done
            next_state = _tree_where_batch(completed, cur_state, next_state)
            next_obs = jnp.where(completed[:, None], cur_obs, next_obs)
            out = {"valid": valid, "absorbing": absorbing & valid, "reward": jnp.where(valid, reward, 0.0)}
            for key in RACKET_ERR_KEYS:
                if key in info:
                    out[key] = jnp.where(valid, info[key], jnp.nan)
            return (next_obs, next_state, next_completed, rng), out

        init = (obs, env_state, jnp.zeros(obs.shape[0], dtype=bool), jax.random.key(1))
        _, scan_out = jax.lax.scan(body, init, None, horizon)
        return scan_out

    scan_out = jax.jit(_rollout)(params, run_stats, obs, env_state)
    scan_out = {k: np.asarray(v) for k, v in scan_out.items()}
    valid = scan_out["valid"]
    steps = valid.sum(0)
    lens_arr = np.asarray(lens)
    early = steps < (lens_arr - 1)
    row = {
        "mass_scale": mass_scale,
        "n_trajectories": n_traj,
        "early_termination_n": int(early.sum()),
        "early_termination_idx": [int(i) for i in np.flatnonzero(early)],
        "coverage_mean": float((steps / np.maximum(lens_arr - 1, 1)).mean()),
        "reward_mean_per_step": float(scan_out["reward"].sum() / max(int(valid.sum()), 1)),
    }
    for key in RACKET_ERR_KEYS:
        if key in scan_out:
            row[key + "_mean"] = float(np.nanmean(scan_out[key]))
            row[key + "_p90"] = float(np.nanpercentile(scan_out[key], 90))
    row["per_traj_steps"] = [int(s) for s in steps]
    row["per_traj_len"] = lens
    row["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in row.items() if not k.startswith("per_")}, ensure_ascii=False), flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--masses", default="0.25,1.0", help="comma-separated racket mass scales")
    parser.add_argument("--tag", default="t3_seed0", help="output file prefix")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for mass in (float(x) for x in args.masses.split(",")):
        results[str(mass)] = run_mass(args.checkpoint, mass)
    out_path = OUT_DIR / f"{args.tag}_zero_shot.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
