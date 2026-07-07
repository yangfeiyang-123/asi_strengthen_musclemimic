#!/usr/bin/env python3
"""Run incoming-shuttle hit stages: preflight, feed-check, physics-smoke, PPO training.

This task is independent from the musclemimic trajectory-tracking pipeline: the
environment owns its MuJoCo scene and physics loop, and training uses the
standalone PPO utilities shared with the grip-policy trainer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class IncomingHitPaths:
    spec_path: Path
    scene_xml: Path
    build_if_missing: bool
    human_root_xy: tuple[float, float]
    feed_bank_path: Path
    feed_bank_size: int
    feed_seed: int
    eval_feed_bank_path: Path
    eval_feed_bank_size: int
    eval_feed_seed: int
    feed_kwargs: dict[str, Any]
    hit_window_kwargs: dict[str, Any]
    control_substeps: int
    max_episode_steps: int
    reward_weights: dict[str, float]
    ppo_overrides: dict[str, Any]
    output_dir: Path


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_incoming_hit_spec(spec_path: str | Path) -> IncomingHitPaths:
    resolved_spec = _resolve(spec_path)
    data = yaml.safe_load(resolved_spec.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_spec} must contain a mapping")
    if data.get("runner_type") != "incoming_shuttle_hit":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")

    scene = data.get("scene", {})
    feed = dict(data.get("feed", {}))
    eval_feed = dict(data.get("eval_feed", {}))
    window = dict(data.get("hit_window", {}))
    episode = dict(data.get("episode", {}))
    reward = dict(data.get("reward", {}))
    ppo = dict(data.get("ppo", {}))
    for section, name in ((scene, "scene"), (feed, "feed"), (episode, "episode")):
        if not isinstance(section, dict):
            raise ValueError(f"{name} must contain a mapping")

    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    human_root_xy = tuple(float(v) for v in scene.get("human_root_xy", (-3.35, 0.0)))

    feed_bank_path = _resolve(feed.pop("bank_path", "outputs/incoming_shuttle_hit/feed_bank.npz"))
    feed_bank_size = int(feed.pop("bank_size", 512))
    feed_seed = int(feed.pop("seed", 7))
    eval_feed_bank_path = _resolve(eval_feed.pop("bank_path", str(feed_bank_path) + ".eval.npz"))
    eval_feed_bank_size = int(eval_feed.pop("bank_size", 128))
    eval_feed_seed = int(eval_feed.pop("seed", feed_seed + 1000))

    feed_kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in feed.items()}
    window_kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in window.items()}

    return IncomingHitPaths(
        spec_path=resolved_spec,
        scene_xml=_resolve(scene["xml"]),
        build_if_missing=bool(scene.get("build_if_missing", True)),
        human_root_xy=human_root_xy,
        feed_bank_path=feed_bank_path,
        feed_bank_size=feed_bank_size,
        feed_seed=feed_seed,
        eval_feed_bank_path=eval_feed_bank_path,
        eval_feed_bank_size=eval_feed_bank_size,
        eval_feed_seed=eval_feed_seed,
        feed_kwargs=feed_kwargs,
        hit_window_kwargs=window_kwargs,
        control_substeps=int(episode.get("control_substeps", 10)),
        max_episode_steps=int(episode.get("max_episode_steps", 300)),
        reward_weights={str(key): float(value) for key, value in reward.items()},
        ppo_overrides=ppo,
        output_dir=output_dir,
    )


def _feed_config(paths: IncomingHitPaths):
    from environment.overall_environment.src.shuttle_feeder import FeedConfig

    return FeedConfig(**paths.feed_kwargs)


def _hit_window(paths: IncomingHitPaths):
    from environment.overall_environment.src.shuttle_feeder import HitWindow

    return HitWindow(**paths.hit_window_kwargs)


def _ensure_scene(paths: IncomingHitPaths) -> None:
    if paths.scene_xml.is_file():
        return
    if not paths.build_if_missing:
        raise FileNotFoundError(f"scene XML missing and build_if_missing is false: {paths.scene_xml}")
    from environment.overall_environment.src.incoming_scene import build_incoming_hit_scene

    build_incoming_hit_scene(paths.scene_xml, human_root_xy=paths.human_root_xy)


def _ensure_feed_bank(paths: IncomingHitPaths, *, evaluation: bool = False) -> list[Any]:
    from environment.overall_environment.src.shuttle_feeder import (
        build_feed_bank,
        load_feed_bank,
        save_feed_bank,
    )

    bank_path = paths.eval_feed_bank_path if evaluation else paths.feed_bank_path
    bank_size = paths.eval_feed_bank_size if evaluation else paths.feed_bank_size
    seed = paths.eval_feed_seed if evaluation else paths.feed_seed
    if bank_path.is_file():
        bank = load_feed_bank(bank_path)
        if len(bank) >= bank_size:
            return bank
    bank = build_feed_bank(bank_size, seed, _feed_config(paths), _hit_window(paths))
    save_feed_bank(bank_path, bank)
    return bank


def _make_env(paths: IncomingHitPaths, *, feed_bank: list[Any] | None, seed: int = 0, **overrides: Any):
    from environment.overall_environment.src.incoming_shuttle_hit_env import IncomingShuttleHitEnv

    kwargs: dict[str, Any] = dict(
        feed_bank=feed_bank,
        feed_config=_feed_config(paths),
        hit_window=_hit_window(paths),
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        seed=seed,
    )
    kwargs.update(overrides)
    return IncomingShuttleHitEnv(paths.scene_xml, **kwargs)


def preflight(paths: IncomingHitPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    import mujoco

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "preflight"
    out_path.mkdir(parents=True, exist_ok=True)
    _ensure_scene(paths)

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    weld_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        for i in range(model.neq)
    ]
    hard_weld = "overall_right_hand_racket_soft_weld"
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_adr = int(model.jnt_qposadr[root_joint])
    required_sites = ["overall_stringbed_center_site", "overall_cork_contact_site", "rh_palm_grip_site"]
    missing_sites = [
        name for name in required_sites
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0
    ]

    report = {
        "runner_type": "incoming_shuttle_hit",
        "spec_path": str(paths.spec_path),
        "scene_xml": str(paths.scene_xml),
        "scene_exists": paths.scene_xml.is_file(),
        "output_dir": str(out_path),
        "keyframe_found": key_id >= 0,
        "hard_weld_present": hard_weld in weld_names,
        "root_pos": [float(v) for v in data.qpos[root_adr : root_adr + 3]],
        "expected_root_xy": list(paths.human_root_xy),
        "missing_sites": missing_sites,
        "actuator_count": int(model.nu),
        "timestep_s": float(model.opt.timestep),
        "reward_weights": paths.reward_weights,
        "feed_bank_path": str(paths.feed_bank_path),
        "eval_feed_bank_path": str(paths.eval_feed_bank_path),
    }
    report["passed"] = bool(
        report["scene_exists"]
        and report["keyframe_found"]
        and report["hard_weld_present"]
        and not missing_sites
        and abs(report["root_pos"][0] - paths.human_root_xy[0]) < 1e-6
    )
    (out_path / "preflight_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def feed_check(paths: IncomingHitPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "feed_check"
    out_path.mkdir(parents=True, exist_ok=True)
    window = _hit_window(paths)

    report: dict[str, Any] = {"runner_stage": "feed-check"}
    for label, evaluation in (("train", False), ("eval", True)):
        bank = _ensure_feed_bank(paths, evaluation=evaluation)
        points = np.stack([sample.intercept_point for sample in bank])
        times = np.array([sample.intercept_time_s for sample in bank])
        speeds = np.array([np.linalg.norm(sample.intercept_velocity) for sample in bank])
        inside = window.contains(points)
        report[label] = {
            "bank_size": len(bank),
            "all_in_window": bool(inside.all()),
            "intercept_point_mean": points.mean(axis=0).tolist(),
            "intercept_point_min": points.min(axis=0).tolist(),
            "intercept_point_max": points.max(axis=0).tolist(),
            "intercept_time_mean_s": float(times.mean()),
            "intercept_time_range_s": [float(times.min()), float(times.max())],
            "intercept_speed_mean_m_s": float(speeds.mean()),
            "intercept_speed_range_m_s": [float(speeds.min()), float(speeds.max())],
        }
    report["passed"] = bool(report["train"]["all_in_window"] and report["eval"]["all_in_window"])
    (out_path / "feed_check_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def physics_smoke(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    episodes: int = 3,
    record_video: bool = False,
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "physics_smoke"
    out_path.mkdir(parents=True, exist_ok=True)
    _ensure_scene(paths)
    bank = _ensure_feed_bank(paths)

    # body_fall termination is disabled so the full shuttle flight is observable
    # under zero muscle activity.
    env = _make_env(paths, feed_bank=bank, seed=0, terminate_on_body_fall=False)
    zero_action = np.zeros(env.action_size, dtype=float)
    episode_reports = []
    finite = True
    frames: list[np.ndarray] = []
    renderer = None
    if record_video:
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")

    for episode in range(int(episodes)):
        obs, info = env.reset(feed_index=episode)
        start_x = float(info["feed_intercept_point"][0])
        launch_x = float(env.feed.launch_pos[0])
        min_speed = math.inf
        max_speed = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            finite = finite and bool(np.isfinite(obs).all()) and math.isfinite(float(reward))
            speed = float(np.linalg.norm(info["flight"]["shuttle_velocity"]))
            min_speed = min(min_speed, speed)
            max_speed = max(max_speed, speed)
            if renderer is not None and episode == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
        flight = info["flight"]
        episode_reports.append(
            {
                "launch_x": launch_x,
                "intercept_x": start_x,
                "termination_reason": info.get("termination_reason"),
                "final_shuttle_xyz": np.asarray(flight["shuttle_xyz"]).tolist(),
                "final_state": info["state"],
                "steps": int(info["step_count"]),
                "max_speed_m_s": max_speed,
                "landing_speed_m_s": speed,
                "crossed_to_player_half": bool(float(flight["shuttle_xyz"][0]) < 0.0),
            }
        )

    if frames:
        import imageio.v2 as imageio

        video_path = out_path / "physics_smoke.mp4"
        imageio.mimsave(video_path, frames, fps=int(1.0 / (env.model.opt.timestep * env.control_substeps)), macro_block_size=None)

    landed = all(r["termination_reason"] in {"miss", "landed"} for r in episode_reports)
    crossed = all(r["crossed_to_player_half"] for r in episode_reports)
    # drag caps the shuttle speed near terminal velocity by touchdown
    aero_effective = all(r["landing_speed_m_s"] <= 7.5 for r in episode_reports)
    report = {
        "runner_stage": "physics-smoke",
        "episodes": episode_reports,
        "finite": bool(finite),
        "all_landed": landed,
        "all_crossed_to_player_half": crossed,
        "aero_effective": aero_effective,
        "passed": bool(finite and landed and crossed and aero_effective),
    }
    (out_path / "physics_smoke_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _run_ppo(
    paths: IncomingHitPaths,
    *,
    out_dir: Path,
    total_steps: int,
    rollout_steps: int,
    ppo_overrides: dict[str, Any] | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_every_updates: int = 20,
) -> dict[str, Any]:
    import torch

    from src.grip.train_right_hand_racket_grip_policy import (
        PPOConfig,
        PolicyValueNet,
        RunningMeanStd,
        _empty_rollout,
        _gae,
        _mean_last,
        _ppo_update,
        _sample_action,
        _tensor,
    )

    torch.set_num_threads(max(1, os.cpu_count() // 2 if os.cpu_count() else 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    bank = _ensure_feed_bank(paths)
    env = _make_env(paths, feed_bank=bank, seed=seed)
    obs, last_info = env.reset()
    obs_size = int(obs.size)
    action_size = int(env.action_size)
    obs_rms = RunningMeanStd((obs_size,))
    obs_rms.update(obs)

    config_kwargs: dict[str, Any] = dict(
        total_steps=int(total_steps),
        rollout_steps=int(rollout_steps),
        minibatch_size=min(int(rollout_steps), 256),
        seed=int(seed),
    )
    for key, value in (ppo_overrides or {}).items():
        if key in {"total_steps", "rollout_steps"}:
            continue
        if key == "hidden_sizes":
            value = tuple(int(v) for v in value)
        config_kwargs[key] = value
    ppo_config = PPOConfig(**config_kwargs)

    model = PolicyValueNet(obs_size, action_size, ppo_config.hidden_sizes, ppo_config.action_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)

    global_step = 0
    update_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    episode_hits: list[float] = []
    episode_crossed: list[float] = []
    episode_landing_scores: list[float] = []
    current_hit = 0.0
    current_crossed = 0.0
    current_landing = 0.0
    summaries: list[dict[str, float]] = []
    finite = True

    while global_step < ppo_config.total_steps:
        rollout_target = min(ppo_config.rollout_steps, ppo_config.total_steps - global_step)
        rollout = _empty_rollout(rollout_target, obs_size, action_size)
        for step in range(rollout_target):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalize(obs)
            action, logprob, value = _sample_action(torch, model, obs_norm, device, rng)
            next_obs, reward, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)
            finite = finite and bool(np.isfinite(next_obs).all()) and bool(np.isfinite(reward))

            rollout["obs"][step] = obs_norm
            rollout["actions"][step] = action
            rollout["logprobs"][step] = logprob
            rollout["rewards"][step] = float(reward)
            rollout["dones"][step] = float(done)
            rollout["values"][step] = value

            terms = last_info.get("reward_terms", {})
            current_hit = max(current_hit, 1.0 if terms.get("hit_bonus", 0.0) > 0.0 else 0.0)
            current_crossed = max(current_crossed, 1.0 if terms.get("crossed_net", 0.0) > 0.0 else 0.0)
            if terms.get("landing_region", 0.0) != 0.0:
                current_landing = float(terms["landing_region"])

            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                episode_hits.append(current_hit)
                episode_crossed.append(current_crossed)
                episode_landing_scores.append(current_landing)
                episode_return = 0.0
                episode_length = 0
                current_hit = current_crossed = current_landing = 0.0
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
                "mean_rollout_reward": float(np.mean(rollout["rewards"])),
                "mean_episode_return": _mean_last(completed_returns, 10),
                "mean_episode_length": _mean_last(completed_lengths, 10),
                "hit_rate": _mean_last(episode_hits, 20),
                "crossed_net_rate": _mean_last(episode_crossed, 20),
                "mean_landing_score": _mean_last(episode_landing_scores, 20),
            }
        )
        summaries.append(update_summary)
        if update_index % max(1, int(checkpoint_every_updates)) == 0 or global_step >= ppo_config.total_steps:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_rms_mean": obs_rms.mean,
                    "obs_rms_var": obs_rms.var,
                    "obs_size": obs_size,
                    "action_size": action_size,
                    "hidden_sizes": list(ppo_config.hidden_sizes),
                    "global_step": global_step,
                },
                out_dir / "policy_latest.pt",
            )

    report = {
        "runner_stage": "train",
        "global_steps": int(global_step),
        "updates": int(update_index),
        "finite": bool(finite),
        "episodes_completed": len(completed_returns),
        "mean_episode_return_last10": _mean_last(completed_returns, 10),
        "hit_rate_last20": _mean_last(episode_hits, 20),
        "crossed_net_rate_last20": _mean_last(episode_crossed, 20),
        "mean_landing_score_last20": _mean_last(episode_landing_scores, 20),
        "last_update": summaries[-1] if summaries else {},
        "checkpoint": str(out_dir / "policy_latest.pt"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe({"summaries": summaries, "report": report}), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def train_tiny(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    total_steps: int = 256,
    rollout_steps: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "tiny_train"
    _ensure_scene(paths)
    report = _run_ppo(
        paths,
        out_dir=out_path,
        total_steps=total_steps,
        rollout_steps=rollout_steps,
        ppo_overrides={"update_epochs": 1, "hidden_sizes": (64, 64), "action_std_init": 0.25},
        seed=seed,
        device=device,
    )
    report["runner_stage"] = "train-tiny"
    (out_path / "tiny_train_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def train(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "train"
    _ensure_scene(paths)
    overrides = dict(paths.ppo_overrides)
    total_steps = int(overrides.get("total_steps", 2_000_000))
    rollout_steps = int(overrides.get("rollout_steps", 1024))
    return _run_ppo(
        paths,
        out_dir=out_path,
        total_steps=total_steps,
        rollout_steps=rollout_steps,
        ppo_overrides=overrides,
        seed=seed,
        device=device,
    )


def train_gpu(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    num_envs: int = 512,
    rollout_steps: int = 64,
    total_env_steps: int | None = None,
    impl: str = "warp",
    base_policy_artifact: str | None = None,
    residual_scale: float = 0.3,
    base_skill: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """GPU-parallel PPO on the MJX badminton env (warp backend by default).

    Requires the sanitized GPU environment (source BadmintonMimic/configs/env.sh,
    which also prepends the cuda-compat 12.4 libraries needed by Warp).
    When ``base_policy_artifact`` is set, the PPO action becomes a residual on
    top of a frozen distilled base policy (Stage 3).
    """
    from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (
        IncomingHitMjxEnv,
    )
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        TrainConfig,
        train as train_mjx,
    )

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "train_gpu"
    _ensure_scene(paths)
    bank = _ensure_feed_bank(paths)

    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        impl=impl,
        base_policy_artifact=base_policy_artifact,
        residual_scale=residual_scale,
        base_skill=base_skill,
    )
    ppo = dict(paths.ppo_overrides)
    if total_env_steps is None:
        total_env_steps = int(ppo.get("total_steps", 2_000_000))
    cfg = TrainConfig(
        num_envs=int(num_envs),
        rollout_steps=int(rollout_steps),
        total_env_steps=int(total_env_steps),
        update_epochs=int(ppo.get("update_epochs", 4)),
        hidden=tuple(ppo.get("hidden_sizes", (256, 256))),
        action_std_init=float(ppo.get("action_std_init", 0.35)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        seed=int(seed),
    )
    report = train_mjx(env, cfg, out_path)
    report["runner_stage"] = "train-gpu"
    report["impl"] = impl
    return report


def evaluate(
    paths: IncomingHitPaths,
    *,
    checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
    episodes: int = 8,
    record_video: bool = False,
) -> dict[str, Any]:
    """Replay a train-gpu checkpoint deterministically on the CPU reference env."""
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "evaluate"
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        Path(checkpoint)
        if checkpoint is not None
        else paths.output_dir / "train_gpu" / "policy_latest.npz"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
    hidden = [int(h) for h in meta["config"]["hidden"]]

    with np.load(ckpt_path) as payload:
        params = [payload[k] for k in sorted(payload.files, key=lambda s: (len(s), s)) if k.startswith("param_")]
        obs_mean = payload["obs_mean"]
        obs_var = payload["obs_var"]
    # tree_flatten order over the agent dict (sorted keys): log_std first, then
    # policy layers as (b, w) pairs, then value layers
    n_policy = len(hidden) + 1

    def mlp_forward(obs_norm: np.ndarray) -> np.ndarray:
        x = obs_norm
        for i in range(n_policy):
            b, w = params[1 + 2 * i], params[2 + 2 * i]
            x = x @ w + b
            if i < n_policy - 1:
                x = np.tanh(x)
        return np.tanh(x)

    from environment.overall_environment.src.incoming_shuttle_hit_env import IncomingShuttleHitEnv

    bank = _ensure_feed_bank(paths, evaluation=True)
    env = IncomingShuttleHitEnv(
        paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        terminate_on_body_fall=True,
        seed=123,
    )
    renderer = None
    frames: list[Any] = []
    if record_video:
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")
        env.model.vis.global_.offwidth = 1280
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=1280)
        camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")

    results = []
    for episode in range(int(episodes)):
        obs, info = env.reset(feed_index=episode)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_norm = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10.0, 10.0)
            action = mlp_forward(obs_norm)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if renderer is not None and episode == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
        results.append(
            {
                "episode": episode,
                "return": episode_return,
                "steps": int(info["step_count"]),
                "termination_reason": info.get("termination_reason"),
                "hit": bool(info.get("hit_closing_speed_m_s", 0.0) > 0.0),
                "landing_region": info.get("landing_region"),
            }
        )

    if frames:
        import imageio.v2 as imageio

        imageio.mimsave(out_path / "evaluate_episode0.mp4", frames, fps=100, macro_block_size=None)

    report = {
        "runner_stage": "evaluate",
        "checkpoint": str(ckpt_path),
        "episodes": results,
        "mean_return": float(np.mean([r["return"] for r in results])),
        "hit_rate": float(np.mean([1.0 if r["hit"] else 0.0 for r in results])),
    }
    (out_path / "evaluate_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="BadmintonMimic/experiments/posttrain/incoming_shuttle_hit_v1.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "feed-check", "physics-smoke", "train-tiny", "train", "train-gpu", "evaluate"),
        default="preflight",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="checkpoint npz for evaluate")
    parser.add_argument("--steps", type=int, default=256, help="total PPO steps for train-tiny")
    parser.add_argument("--rollout-steps", type=int, default=64, help="rollout length for train-tiny/train-gpu")
    parser.add_argument("--episodes", type=int, default=3, help="episodes for physics-smoke")
    parser.add_argument("--num-envs", type=int, default=512, help="parallel envs for train-gpu")
    parser.add_argument("--total-env-steps", type=int, default=None, help="env steps for train-gpu (default: spec ppo.total_steps)")
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp", help="MJX backend for train-gpu")
    parser.add_argument("--base-policy-artifact", default=None, help="frozen base policy export dir (Stage 3 residual mode)")
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--base-skill", default=None, help="skill name for a multi-skill base")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = load_incoming_hit_spec(args.spec)
    if args.stage == "preflight":
        report = preflight(paths, out_dir=args.out_dir)
    elif args.stage == "feed-check":
        report = feed_check(paths, out_dir=args.out_dir)
    elif args.stage == "physics-smoke":
        report = physics_smoke(
            paths, out_dir=args.out_dir, episodes=args.episodes, record_video=args.record_video
        )
    elif args.stage == "train-tiny":
        report = train_tiny(
            paths,
            out_dir=args.out_dir,
            total_steps=args.steps,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == "evaluate":
        report = evaluate(
            paths,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            episodes=args.episodes,
            record_video=args.record_video,
        )
    elif args.stage == "train-gpu":
        report = train_gpu(
            paths,
            out_dir=args.out_dir,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            total_env_steps=args.total_env_steps,
            impl=args.impl,
            base_policy_artifact=args.base_policy_artifact,
            residual_scale=args.residual_scale,
            base_skill=args.base_skill,
            seed=args.seed,
        )
    else:
        report = train(paths, out_dir=args.out_dir, seed=args.seed, device=args.device)

    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
