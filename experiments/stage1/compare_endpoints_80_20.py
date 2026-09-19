#!/usr/bin/env python3
"""对 checkpoints/stage1 下 12 个 800M endpoint 做统一的 held-out rollout 对比（探索性，非正式门）。

每个 checkpoint：
  1. 从 checkpoint 自带的 resolved config 构建 held-out 验证环境（20 条 80/20 验证轨迹，
     deterministic policy，frame-0 起点），与官方验证同一路径；
  2. 逐步采集 354 维 ordered activation（data.act 经 actuator_actadr）与归一化进度；
  3. 用同一份 verified tube（真实相位，不做 T4 平移）计算 15 通道 anchor 与协同指标：
     anchor loss / violation / pattern correlation / 逐通道 loss，
     synergy loss / shape loss / intensity loss / shape cosine；
  4. 跟踪与安全指标直接取 checkpoints/stage1/COMPLETED_20260918.json 的 endpoint_metrics
     （官方验证在同一 held-out 集上算出），并用本次 rollout 的 coverage / early 交叉核对。

输出：outputs/stage1_endpoint_compare/{<arm>_s<seed>.npz, summary.json, summary.md}

用法（仓库根目录）：
    source configs/env.sh
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/stage1/compare_endpoints_80_20.py
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
COMPLETED = CKPT_ROOT / "COMPLETED_20260918.json"
TUBE_MANIFEST = "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear/emg_reference_manifest.json"
ARMS = ("T0", "T1", "T2", "T3", "T4")
TRACK_KEYS = (
    "val_err_joint_pos",
    "val_err_rpos",
    "val_err_root_xyz",
    "val_frame_coverage",
    "val_early_termination_rate",
    "val_action_saturation_fraction",
    "val_activation_saturation_fraction",
    "val_activation_energy",
    "val_emg_synergy_real_reference_loss",
)


def discover_leaves() -> list[tuple[str, int, Path]]:
    leaves = []
    for seed_dir in sorted(CKPT_ROOT.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        for arm in ARMS:
            leaf = seed_dir / arm / "checkpoint_39063"
            if leaf.is_dir():
                leaves.append((arm, seed, leaf))
    return leaves


def emg_config_from(config: dict, *, arm_label: str) -> dict:
    raw = dict(config["experiment"]["env_params"]["reward_params"]["emg_consistency"])
    raw.update(
        enabled=True,
        arm="T3",  # 真实相位、两条腿都可观测；只用于诊断，不进奖励
        reference_cache=TUBE_MANIFEST,
        mapping_path=None,
        synergy_phase_shuffle_offset_bins=0,
        anchor_weight_max=0.02,
        synergy_weight_max=0.05,
    )
    raw["_diagnostic_for"] = arm_label
    raw.pop("_diagnostic_for")
    return raw


def rollout_leaf(arm: str, seed: int, leaf: Path, emg_template: dict | None, out_dir: Path) -> dict:
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import wrap_env
    from musclemimic.algorithms.ppo.runner import (
        _apply_frozen_eval_policy,
        _reset_eval_all_batch_jitted,
        _tree_where_batch,
    )
    from musclemimic.physiology.emg_anchor import emg_anchor_metrics, emg_synergy_metrics
    from musclemimic.physiology.emg_consistency_runtime import compile_emg_consistency_runtime
    from musclemimic.physiology.runtime_binding import resolve_ordered_policy_muscle_layout
    from musclemimic.runner.engine import instantiate_validation_env
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint, resolve_checkpoint_path

    t0 = time.time()
    checkpoint_path = Path(resolve_checkpoint_path(str(leaf))).resolve(strict=True)
    config, agent_state, _meta = load_checkpoint(str(checkpoint_path))
    experiment = config.experiment
    run_id = str(experiment.get("run_id"))
    validation = experiment.get("validation", {})
    assert bool(validation.get("deterministic", False)) and bool(validation.get("start_from_beginning", False))

    env = instantiate_validation_env(config, share_trajectory=False)
    assert env is not None and getattr(env, "th", None) is not None, "no held-out split"
    layout = resolve_ordered_policy_muscle_layout(env, model=env._model)
    assert layout.width == 354
    addresses = np.asarray(layout.activation_addresses, dtype=np.int32)

    raw_emg = emg_template if emg_template is not None else emg_config_from(
        OmegaConf.to_container(config, resolve=True), arm_label=arm
    )
    runtime = compile_emg_consistency_runtime(env, raw_emg, base_dir=REPO)
    assert runtime is not None, "EMG runtime did not compile"
    spec = runtime.spec
    action_index = int(runtime.action_index)
    kappa = float(runtime.config.tube_kappa)
    huber = float(runtime.config.huber_delta)

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    train_state = agent_state.train_state
    network = agent_conf.network

    if getattr(env, "mjx_enabled", False) and env.th.is_numpy:
        env.th.to_jax()
    n_traj = int(env.th.n_trajectories)
    traj_lens = [int(env.th.len_trajectory(i)) for i in range(n_traj)]
    max_horizon = max(traj_lens)
    if int(env.info.horizon) < max_horizon:
        env._mdp_info.horizon = max_horizon
    num_envs = min(int(validation.get("num_envs") or n_traj), n_traj)
    val_env = wrap_env(env, experiment)
    act_addr = jnp.asarray(addresses, dtype=jnp.int32)

    def _unwrap(state):
        seen = 0
        while hasattr(state, "env_state") and seen < 64:
            state = state.env_state
            seen += 1
        return state

    def _rollout(params, run_stats, obs, env_state, horizon):
        num = obs.shape[0]

        def _body(carry, _):
            cur_obs, cur_state, completed, rng, rs = carry
            was_completed = completed
            rng, _ = jax.random.split(rng)
            pi, _v = _apply_frozen_eval_policy(network, params, rs, cur_obs)
            action = jnp.where(was_completed[:, None], 0.0, pi.mode())
            next_obs, reward, absorbing, done, _info, next_state, transition_state = val_env.step_with_transition(
                cur_state, action
            )
            valid = ~was_completed
            completed = was_completed | done
            next_state = _tree_where_batch(was_completed, cur_state, next_state)
            next_obs = jnp.where(was_completed[:, None], cur_obs, next_obs)
            mjx_state = _unwrap(transition_state)
            act = jnp.take(mjx_state.data.act, act_addr, axis=-1)
            act = jnp.where(was_completed[:, None], 0.0, act)
            out = {"act": act, "reward": jnp.where(valid, reward, 0.0), "done": done, "absorbing": absorbing, "valid": valid}
            return (next_obs, next_state, completed, rng, rs), out

        _, scan_out = jax.lax.scan(_body, (obs, env_state, jnp.zeros(num, dtype=bool), jax.random.key(0), run_stats), None, horizon)
        return scan_out

    rollout_fn = jax.jit(_rollout, static_argnums=(4,))

    def _metrics(act, phase):
        a = emg_anchor_metrics(act, spec, action_index=action_index, phase=phase, kappa=kappa, huber_delta=huber)
        s = emg_synergy_metrics(
            act, spec, action_index=action_index, phase=phase, phase_bin_offset=0,
            shape_weight=1.0, intensity_weight=0.25, kappa=kappa, huber_delta=huber,
        )
        return {
            "anchor_loss": a.loss, "anchor_violation": a.violation_fraction, "anchor_corr": a.pattern_correlation,
            "anchor_channel_loss": a.channel_loss, "synergy_loss": s.loss, "synergy_shape_loss": s.shape_loss,
            "synergy_intensity_loss": s.intensity_loss, "synergy_shape_cosine": s.shape_cosine,
            "synergy_coeff": s.coefficients, "projected": a.projected_activation,
        }

    metrics_fn = jax.jit(jax.vmap(_metrics))

    eval_seed = int(validation.get("eval_seed", 0))
    rng = jax.random.key(eval_seed)
    per_traj = []
    save = {"activation_addresses": addresses, "actuator_names": np.asarray([str(n) for n in layout.actuator_names]),
            "channel_names": np.asarray(list(spec.channel_names))}
    for batch_start in range(0, n_traj, num_envs):
        idx = list(range(batch_start, min(batch_start + num_envs, n_traj)))
        active = len(idx)
        while len(idx) < num_envs:
            idx.append(idx[-1])
        rng, batch_rng, _ = jax.random.split(rng, 3)
        reset_keys = jax.random.split(batch_rng, num_envs)
        obs, env_state = _reset_eval_all_batch_jitted(val_env, reset_keys, jnp.asarray(idx, dtype=jnp.int32))
        out = jax.device_get(rollout_fn(train_state.params, train_state.run_stats, obs, env_state, max_horizon))
        valid = np.asarray(out["valid"])
        for lane in range(active):
            ti = int(idx[lane])
            lv = valid[:, lane]
            ep_len = int(lv.sum())
            act = np.asarray(out["act"][:, lane, :])[lv].astype(np.float32)
            done = np.asarray(out["done"][:, lane]); absorbing = np.asarray(out["absorbing"][:, lane])
            any_done = bool(done.any())
            first_done = int(np.argmax(done)) + 1 if any_done else max_horizon
            early = any_done and bool(absorbing[max(first_done - 1, 0)]) and ep_len < traj_lens[ti]
            phase = np.clip(np.arange(ep_len, dtype=np.float32) / max(traj_lens[ti] - 1, 1), 0.0, 1.0)
            m = jax.device_get(metrics_fn(jnp.asarray(act), jnp.asarray(phase)))
            rec = {"traj": ti, "traj_len": traj_lens[ti], "ep_len": ep_len, "coverage": ep_len / traj_lens[ti],
                   "early": bool(early), "return": float(np.asarray(out["reward"][:, lane])[lv].sum())}
            for k in ("anchor_loss", "anchor_violation", "anchor_corr", "synergy_loss", "synergy_shape_loss",
                      "synergy_intensity_loss", "synergy_shape_cosine"):
                rec[k] = float(np.mean(m[k]))
            rec["anchor_channel_loss"] = np.mean(np.asarray(m["anchor_channel_loss"]), axis=0).tolist()
            per_traj.append(rec)
            save[f"act_traj{ti}"] = act
            save[f"phase_traj{ti}"] = phase
            save[f"projected_traj{ti}"] = np.asarray(m["projected"], dtype=np.float32)
            save[f"synergy_coeff_traj{ti}"] = np.asarray(m["synergy_coeff"], dtype=np.float32)
            print(f"  [{arm} s{seed}] traj {ti}: len {ep_len}/{traj_lens[ti]} early={early} "
                  f"anchor={rec['anchor_loss']:.4f} syn={rec['synergy_loss']:.4f} cos={rec['synergy_shape_cosine']:.3f}", flush=True)

    def _mean(k):
        return float(np.mean([r[k] for r in per_traj]))

    summary = {
        "arm": arm, "seed": seed, "run_id": run_id, "checkpoint": str(checkpoint_path.relative_to(REPO)),
        "n_traj": n_traj, "eval_seed": eval_seed, "elapsed_s": round(time.time() - t0, 1),
        "rollout": {
            "coverage": float(sum(r["ep_len"] for r in per_traj) / sum(r["traj_len"] for r in per_traj)),
            "early_termination_rate": float(np.mean([r["early"] for r in per_traj])),
            "return_mean": _mean("return"),
        },
        "physiology_traj_mean": {k: _mean(k) for k in (
            "anchor_loss", "anchor_violation", "anchor_corr", "synergy_loss", "synergy_shape_loss",
            "synergy_intensity_loss", "synergy_shape_cosine")},
        "anchor_channel_loss_traj_mean": np.mean([r["anchor_channel_loss"] for r in per_traj], axis=0).tolist(),
        "channel_names": list(spec.channel_names),
        "per_traj": per_traj,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{arm}_s{seed}.npz", **save)
    (out_dir / f"{arm}_s{seed}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {arm} s{seed} in {summary['elapsed_s']}s: {json.dumps(summary['physiology_traj_mean'])}", flush=True)
    return summary


def write_report(summaries: list[dict], out_dir: Path) -> None:
    completed = {r["run_id"]: r for r in json.load(open(COMPLETED))["runs"]}
    rows = []
    for s in sorted(summaries, key=lambda x: (x["arm"], x["seed"])):
        off = completed.get(s["run_id"], {}).get("endpoint_metrics", {})
        p = s["physiology_traj_mean"]
        rows.append({
            "arm": s["arm"], "seed": s["seed"],
            "joint_pos": off.get("val_err_joint_pos"), "rpos": off.get("val_err_rpos"),
            "coverage_official": off.get("val_frame_coverage"), "coverage_rollout": s["rollout"]["coverage"],
            "early_official": off.get("val_early_termination_rate"), "early_rollout": s["rollout"]["early_termination_rate"],
            "act_energy": off.get("val_activation_energy"), "act_sat": off.get("val_activation_saturation_fraction"),
            **p,
        })
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "runs": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")

    def f(v, nd=3):
        return "—" if v is None else f"{v:.{nd}f}"

    lines = ["# Stage-1 800M endpoint 对比（80/20 held-out，同一 verified tube，真实相位）", "",
             "跟踪/安全列来自 `checkpoints/stage1/COMPLETED_20260918.json` 的官方 endpoint 验证；生理列来自本次 rollout（逐轨迹均值再平均）。",
             "anchor/synergy 越低越好，cosine 与 corr 越高越好。T0 训练时无肌电项，其生理列是 post-hoc 诊断。", "",
             "| arm | seed | joint_pos↓ | coverage(官/本) | early(官/本) | act_energy | anchor_loss↓ | anchor_corr↑ | synergy_loss↓ | shape_cos↑ | intensity_loss↓ |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seed']} | {f(r['joint_pos'])} | {f(r['coverage_official'])}/{f(r['coverage_rollout'])} | "
                     f"{f(r['early_official'],2)}/{f(r['early_rollout'],2)} | {f(r['act_energy'])} | {f(r['anchor_loss'],4)} | "
                     f"{f(r['anchor_corr'])} | {f(r['synergy_loss'],4)} | {f(r['synergy_shape_cosine'])} | {f(r['synergy_intensity_loss'],4)} |")
    lines += ["", "## 按 arm 平均（seed 数见括号）", "",
              "| arm | n | joint_pos | anchor_loss | anchor_corr | synergy_loss | shape_cos |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        def m(k):
            vals = [r[k] for r in sub if r[k] is not None]
            return float(np.mean(vals)) if vals else None
        lines.append(f"| {arm} | {len(sub)} | {f(m('joint_pos'))} | {f(m('anchor_loss'),4)} | {f(m('anchor_corr'))} | {f(m('synergy_loss'),4)} | {f(m('synergy_shape_cosine'))} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO / "outputs/stage1_endpoint_compare"))
    parser.add_argument("--only", nargs="*", default=None, help="e.g. T3:0 T4:0")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    leaves = discover_leaves()
    if args.only:
        wanted = {(a, int(s)) for a, s in (x.split(":") for x in args.only)}
        leaves = [l for l in leaves if (l[0], l[1]) in wanted]
    print(f"[plan] {len(leaves)} leaves: {[(a, s) for a, s, _ in leaves]}", flush=True)
    summaries = []
    for arm, seed, leaf in leaves:
        done_json = out_dir / f"{arm}_s{seed}.json"
        if args.report_only or done_json.exists():
            if done_json.exists():
                summaries.append(json.loads(done_json.read_text(encoding="utf-8")))
                print(f"[skip] {arm} s{seed} already done", flush=True)
            continue
        summaries.append(rollout_leaf(arm, seed, leaf, None, out_dir))
    write_report(summaries, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
