#!/usr/bin/env python3
"""Replay Stage-1 endpoints on the held-out split and save kinematics + activation.

Per checkpoint -> outputs/stage1_endpoint_compare/kin/<arm>_s<seed>.npz with, per trajectory i:
  act_traj{i}      (T,354)  ordered muscle activation (data.act via actuator_actadr)
  qpos_traj{i}     (T,nq)   qvel_traj{i} (T,nv)   site_traj{i} (T,17,3) mimic-site world positions
  ref_qpos_traj{i} (L,nq)   ref_qvel_traj{i} (L,nv) ref_site_traj{i} (L,17,3)  human reference (retargeted)
plus joint_names, site_names, actuator_names, dt, episode/trajectory lengths, early flags.
The MuJoCo model is saved once as kin/model.mjb for offline rendering.

Usage:  source configs/env.sh; CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/stage1/capture_endpoint_kinematics.py [--split train|val] [--only T0:0 T3:0]
  --split train evaluates the 80 training motions (fit, not generalisation) into kin_train/; per-trajectory EMG metrics go to <arm>_s<seed>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CKPT_ROOT = REPO / "checkpoints/stage1"
OUT_ROOT = REPO / "outputs/stage1_endpoint_compare"
OUT = OUT_ROOT / "kin"
TUBE_MANIFEST = "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear/emg_reference_manifest.json"
ARMS = ("T0", "T1", "T2", "T3", "T4")


def discover_leaves():
    out = []
    for seed_dir in sorted(CKPT_ROOT.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        for arm in ARMS:
            leaf = seed_dir / arm / "checkpoint_39063"
            if leaf.is_dir():
                out.append((arm, seed, leaf))
    return out


def capture(arm, seed, leaf, save_model: bool, split: str = "val"):
    import jax
    import jax.numpy as jnp
    import mujoco

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import wrap_env
    from musclemimic.algorithms.ppo.runner import (
        _apply_frozen_eval_policy,
        _reset_eval_all_batch_jitted,
        _tree_where_batch,
    )
    from musclemimic.physiology.runtime_binding import resolve_ordered_policy_muscle_layout
    from musclemimic.runner.engine import instantiate_validation_env
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint, resolve_checkpoint_path

    t0 = time.time()
    ck = Path(resolve_checkpoint_path(str(leaf))).resolve(strict=True)
    config, agent_state, _ = load_checkpoint(str(ck))
    from omegaconf import OmegaConf, open_dict
    if split == "train":
        # evaluate the 80 training motions with the validation machinery (deterministic, frame-0 start)
        train_paths = list(config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
        with open_dict(config):
            config.experiment.validation.amass_dataset_conf.rel_dataset_path = train_paths
    motion_names = [str(Path(p).name) for p in config.experiment.validation.amass_dataset_conf.rel_dataset_path]
    experiment = config.experiment
    validation = experiment.get("validation", {})
    env = instantiate_validation_env(config, share_trajectory=False)
    # EMG diagnostics with the verified tube (real phase), same as compare_endpoints_80_20.py
    from musclemimic.physiology.emg_anchor import emg_anchor_metrics, emg_synergy_metrics
    from musclemimic.physiology.emg_consistency_runtime import compile_emg_consistency_runtime
    raw_emg = dict(OmegaConf.to_container(config.experiment.env_params.reward_params.emg_consistency, resolve=True))
    raw_emg.update(enabled=True, arm="T3", reference_cache=TUBE_MANIFEST, mapping_path=None, synergy_phase_shuffle_offset_bins=0,
                   anchor_weight_max=0.02, synergy_weight_max=0.05)
    for k in ("anchor_inside_weight", "anchor_burst_weight", "anchor_shape_weight", "anchor_scale_floor", "anchor_channel_loss_cap"):
        raw_emg.pop(k, None)
    runtime = compile_emg_consistency_runtime(env, raw_emg, base_dir=REPO)
    spec = runtime.spec; action_index = int(runtime.action_index); kappa = float(runtime.config.tube_kappa); huber = float(runtime.config.huber_delta)

    def _metrics(act, phase):
        a = emg_anchor_metrics(act, spec, action_index=action_index, phase=phase, kappa=kappa, huber_delta=huber)
        s = emg_synergy_metrics(act, spec, action_index=action_index, phase=phase, phase_bin_offset=0, shape_weight=1.0, intensity_weight=0.25, kappa=kappa, huber_delta=huber)
        return {"anchor_loss": a.loss, "anchor_corr": a.pattern_correlation, "synergy_loss": s.loss, "synergy_shape_loss": s.shape_loss,
                "synergy_intensity_loss": s.intensity_loss, "synergy_shape_cosine": s.shape_cosine, "projected": a.projected_activation, "synergy_coeff": s.coefficients}
    metrics_fn = jax.jit(jax.vmap(_metrics))
    model = env._model
    layout = resolve_ordered_policy_muscle_layout(env, model=model)
    act_addr = jnp.asarray(np.asarray(layout.activation_addresses, dtype=np.int32))
    th = env.th
    info = th.traj.info
    site_names = list(info.site_names)
    site_ids = jnp.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names], dtype=jnp.int32)
    # reference (numpy copies before any device transfer)
    td = th.traj.data
    split_points = np.asarray(td.split_points)
    ref_qpos = np.asarray(td.qpos); ref_qvel = np.asarray(td.qvel); ref_site = np.asarray(td.site_xpos)

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    train_state = agent_state.train_state
    network = agent_conf.network
    if getattr(env, "mjx_enabled", False) and th.is_numpy:
        th.to_jax()
    n_traj = int(th.n_trajectories)
    traj_lens = [int(th.len_trajectory(i)) for i in range(n_traj)]
    max_h = max(traj_lens)
    if int(env.info.horizon) < max_h:
        env._mdp_info.horizon = max_h
    num_envs = min(int(validation.get("num_envs") or n_traj), n_traj)
    venv = wrap_env(env, experiment)

    def _unwrap(s):
        k = 0
        while hasattr(s, "env_state") and k < 64:
            s = s.env_state; k += 1
        return s

    def _rollout(params, run_stats, obs, env_state, horizon):
        def body(carry, _):
            cur_obs, cur_state, completed, rng, rs = carry
            was = completed
            rng, _ = jax.random.split(rng)
            pi, _ = _apply_frozen_eval_policy(network, params, rs, cur_obs)
            action = jnp.where(was[:, None], 0.0, pi.mode())
            nobs, reward, absorbing, done, _i, nstate, tstate = venv.step_with_transition(cur_state, action)
            valid = ~was
            completed = was | done
            nstate = _tree_where_batch(was, cur_state, nstate)
            nobs = jnp.where(was[:, None], cur_obs, nobs)
            d = _unwrap(tstate).data
            out = {
                "act": jnp.take(d.act, act_addr, axis=-1),
                "qpos": d.qpos, "qvel": d.qvel,
                "site": jnp.take(d.site_xpos, site_ids, axis=-2),
                "done": done, "absorbing": absorbing, "valid": valid,
            }
            return (nobs, nstate, completed, rng, rs), out
        _, so = jax.lax.scan(body, (obs, env_state, jnp.zeros(obs.shape[0], bool), jax.random.key(0), run_stats), None, horizon)
        return so

    rollout = jax.jit(_rollout, static_argnums=(4,))
    rng = jax.random.key(int(validation.get("eval_seed", 0)))
    save = {
        "joint_names": np.asarray(list(info.joint_names)), "site_names": np.asarray(site_names),
        "actuator_names": np.asarray([str(n) for n in layout.actuator_names]),
        "dt": np.asarray(1.0 / float(info.frequency)), "run_id": np.asarray(str(experiment.get("run_id"))),
    }
    ep_lens, tr_lens, earlies, per_traj = [], [], [], []
    for b0 in range(0, n_traj, num_envs):
        idx = list(range(b0, min(b0 + num_envs, n_traj))); active = len(idx)
        while len(idx) < num_envs:
            idx.append(idx[-1])
        rng, brng, _ = jax.random.split(rng, 3)
        obs, st = _reset_eval_all_batch_jitted(venv, jax.random.split(brng, num_envs), jnp.asarray(idx, dtype=jnp.int32))
        so = jax.device_get(rollout(train_state.params, train_state.run_stats, obs, st, max_h))
        valid = np.asarray(so["valid"])
        for lane in range(active):
            ti = int(idx[lane]); lv = valid[:, lane]; L = int(lv.sum())
            done = np.asarray(so["done"][:, lane]); absorbing = np.asarray(so["absorbing"][:, lane])
            any_done = bool(done.any()); first = int(np.argmax(done)) + 1 if any_done else max_h
            early = any_done and bool(absorbing[max(first - 1, 0)]) and L < traj_lens[ti]
            for k in ("act", "qpos", "qvel", "site"):
                save[f"{k}_traj{ti}"] = np.asarray(so[k][:, lane])[lv].astype(np.float32)
            act = save[f"act_traj{ti}"]
            phase = np.clip(np.arange(L, dtype=np.float32) / max(traj_lens[ti] - 1, 1), 0.0, 1.0)
            m = jax.device_get(metrics_fn(jnp.asarray(act), jnp.asarray(phase)))
            save[f"phase_traj{ti}"] = phase; save[f"projected_traj{ti}"] = np.asarray(m["projected"], np.float32); save[f"synergy_coeff_traj{ti}"] = np.asarray(m["synergy_coeff"], np.float32)
            per_traj.append({"traj": ti, "motion": motion_names[ti] if ti < len(motion_names) else "", "split": split, "traj_len": traj_lens[ti], "ep_len": L,
                             "coverage": L / traj_lens[ti], "early": bool(early),
                             **{k: float(np.mean(m[k])) for k in ("anchor_loss", "anchor_corr", "synergy_loss", "synergy_shape_loss", "synergy_intensity_loss", "synergy_shape_cosine")}})
            s, e = split_points[ti], split_points[ti + 1]
            save[f"ref_qpos_traj{ti}"] = ref_qpos[s:e].astype(np.float32)
            save[f"ref_qvel_traj{ti}"] = ref_qvel[s:e].astype(np.float32)
            save[f"ref_site_traj{ti}"] = ref_site[s:e].astype(np.float32)
            ep_lens.append(L); tr_lens.append(traj_lens[ti]); earlies.append(early)
            print(f"  [{arm} s{seed}] traj {ti}: {L}/{traj_lens[ti]} early={early}", flush=True)
    save["episode_lengths"] = np.asarray(ep_lens, np.int32); save["traj_lengths"] = np.asarray(tr_lens, np.int32)
    save["early_terminated"] = np.asarray(earlies, bool)
    out_dir = OUT if split == "val" else OUT_ROOT / f"kin_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save["motion_names"] = np.asarray(motion_names); save["split"] = np.asarray(split)
    np.savez_compressed(out_dir / f"{arm}_s{seed}.npz", **save)
    (out_dir / f"{arm}_s{seed}.json").write_text(json.dumps({"arm": arm, "seed": seed, "split": split, "run_id": str(experiment.get("run_id")), "per_traj": per_traj}, indent=1, ensure_ascii=False), encoding="utf-8")
    if save_model:
        mujoco.mj_saveModel(model, str(OUT / "model.mjb"), None)
        (OUT / "model_meta.json").write_text(json.dumps({
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "activation_addresses": [int(a) for a in layout.activation_addresses],
            "actuator_names": [str(n) for n in layout.actuator_names],
        }), encoding="utf-8")
    print(f"[done] {arm} s{seed} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    args = ap.parse_args()
    leaves = discover_leaves()
    if args.only:
        want = {(a, int(s)) for a, s in (x.split(":") for x in args.only)}
        leaves = [l for l in leaves if (l[0], l[1]) in want]
    print(f"[plan] {[(a, s) for a, s, _ in leaves]}", flush=True)
    out_dir = OUT if args.split == "val" else OUT_ROOT / f"kin_{args.split}"
    for k, (arm, seed, leaf) in enumerate(leaves):
        if (out_dir / f"{arm}_s{seed}.json").exists():
            print(f"[skip] {arm} s{seed}", flush=True); continue
        capture(arm, seed, leaf, save_model=(k == 0 or not (OUT / "model.mjb").exists()), split=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
