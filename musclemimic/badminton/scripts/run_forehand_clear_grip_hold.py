#!/usr/bin/env python3
"""Run ForehandClear grip-hold diagnostics, replay smoke checks, and tiny PPO runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BODY_CONTROL_SUBSTEPS = 10


@dataclass(frozen=True)
class GripHoldPaths:
    spec_path: Path
    runner_type: str
    resume_from: Path
    body_policy_artifact: Path | None
    scene_xml: Path
    grip_seed: Path
    output_dir: Path
    reward_weights: dict[str, float]


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_grip_hold_spec(spec_path: str | Path) -> GripHoldPaths:
    resolved_spec = _resolve(spec_path)
    data = yaml.safe_load(resolved_spec.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_spec} must contain a mapping")
    if data.get("runner_type") != "forehand_clear_grip_hold":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")

    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    body_policy = data.get("body_policy", {})
    if not isinstance(body_policy, dict):
        raise ValueError("body_policy must contain a mapping")
    artifact = body_policy.get("artifact")
    reward_weights = data.get("reward", {})
    if not isinstance(reward_weights, dict):
        raise ValueError("reward must contain a mapping")
    return GripHoldPaths(
        spec_path=resolved_spec,
        runner_type=str(data["runner_type"]),
        resume_from=_resolve(data["resume_from"]),
        body_policy_artifact=_resolve(artifact) if artifact else None,
        scene_xml=_resolve(data["scene"]["xml"]),
        grip_seed=_resolve(data["grip_seed"]["path"]),
        output_dir=output_dir,
        reward_weights={str(key): float(value) for key, value in reward_weights.items()},
    )


def preflight(paths: GripHoldPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "runner_type": paths.runner_type,
        "spec_path": str(paths.spec_path),
        "resume_from": str(paths.resume_from),
        "scene_xml": str(paths.scene_xml),
        "grip_seed": str(paths.grip_seed),
        "body_policy_artifact": str(paths.body_policy_artifact) if paths.body_policy_artifact else None,
        "output_dir": str(out_path),
        "reward_weights": paths.reward_weights,
        "checkpoint_exists": paths.resume_from.is_dir(),
        "body_policy_artifact_exists": bool(paths.body_policy_artifact and paths.body_policy_artifact.is_dir()),
        "scene_exists": paths.scene_xml.is_file(),
        "grip_seed_exists": paths.grip_seed.is_file(),
    }
    (out_path / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def record_reset_video(*, paths: GripHoldPaths, out_dir: Path) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    video_path = diagnostics_dir / "reset_grip_hold.mp4"

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        frames = []
        for _ in range(30):
            renderer.update_scene(data)
            frames.append(renderer.render())
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=None)
    finally:
        renderer.close()
    return video_path


def diagnostic_reset(
    paths: GripHoldPaths,
    *,
    out_dir: str | Path | None = None,
    recorder=record_reset_video,
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    report = preflight(paths, out_dir=out_path)
    video_path = recorder(paths=paths, out_dir=out_path)
    report["reset_video"] = str(video_path)
    (out_path / "diagnostic_reset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def record_replay_video(
    paths: GripHoldPaths,
    *,
    out_dir: str | Path | None = None,
    steps: int = 300,
    fps: int = 30,
    render_stride: int = 1,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if render_stride <= 0:
        raise ValueError(f"render_stride must be positive, got {render_stride}")

    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    from environment.overall_environment.src.overall_grip_hold_env import OverallGripHoldEnv

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "replay_video"
    video_dir = out_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(paths, out_dir=out_path)

    env = OverallGripHoldEnv(
        paths.scene_xml,
        residual_groups=["right_hand_fingers"],
        max_episode_steps=max(steps + 1, 8),
        body_policy_artifact=paths.body_policy_artifact,
        body_checkpoint=paths.resume_from,
        reward_weights=paths.reward_weights,
    )
    env.reset()
    action = np.zeros(env.action_size, dtype=float)
    video_path = video_dir / "real_policy_no_servo_replay.mp4"
    frames: list[np.ndarray] = []
    rewards: list[float] = []
    last_info: dict[str, Any] = {}
    finite = True
    terminated = False
    truncated = False
    first_termination_step: int | None = None

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        camera = "overall_view" if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view") >= 0 else None
        renderer.update_scene(env.data, camera=camera)
        frames.append(renderer.render())
        for step in range(steps):
            _, reward, terminated, truncated, last_info = env.step(action)
            rewards.append(float(reward))
            finite = finite and bool(np.isfinite(env.data.qpos).all()) and bool(np.isfinite(env.data.qvel).all())
            if step % render_stride == 0 or step == steps - 1:
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
            if (terminated or truncated) and first_termination_step is None:
                first_termination_step = int(step + 1)
            if not finite:
                break
    finally:
        renderer.close()

    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
    report.update(
        {
            "runner_stage": "replay-video",
            "video_path": str(video_path),
            "policy_source": "real",
            "steps_requested": int(steps),
            "steps_completed": int(len(rewards)),
            "frames": int(len(frames)),
            "fps": int(fps),
            "render_stride": int(render_stride),
            "finite": bool(finite),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "first_termination_step": first_termination_step,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "last_info": _json_safe(last_info),
            "pose_servo_enabled": bool(last_info.get("pose_servo_enabled", False)),
            "reset_mode": last_info.get("reset_mode"),
            "body_goal_obs_source": last_info.get("body_goal_obs_source"),
            "racket_drop": bool(last_info.get("racket_drop", False)),
            "palm_to_grip_m": float(last_info.get("palm_to_grip_m", float("nan"))),
        }
    )
    (out_path / "replay_video_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "config" / "metadata"
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_shape_size(checkpoint_dir: str | Path, key: str) -> int:
    metadata_path = Path(checkpoint_dir) / "train_state" / "_METADATA"
    if not metadata_path.is_file():
        return 0
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = data.get("tree_metadata", {}).get(key, {})
    shape = value.get("value_metadata", {}).get("write_shape", [])
    if len(shape) != 1:
        return 0
    return int(shape[0])


def _fallback_disable_fingers_manifest(
    checkpoint_dir: str | Path,
    metadata: dict[str, Any],
    scene_actuator_names: list[str],
):
    from environment.overall_environment.src.action_manifest import ActionManifest
    from environment.overall_environment.src.layered_control import RIGHT_HAND_FINGER_ACTUATORS

    left_hand_fingers = {f"{name}_left" for name in RIGHT_HAND_FINGER_ACTUATORS}
    finger_actuators = RIGHT_HAND_FINGER_ACTUATORS | left_hand_fingers
    source_names = [name for name in scene_actuator_names if name not in finger_actuators]
    action_size = _metadata_shape_size(checkpoint_dir, "('params', 'actor', 'Dense_16', 'bias')") or len(source_names)
    if action_size != len(source_names):
        raise ValueError(
            f"fallback action manifest mismatch: checkpoint action size {action_size}, "
            f"derived actuator names {len(source_names)}"
        )
    env_params = metadata["experiment"]["env_params"]
    return ActionManifest.from_env_params(
        env_params,
        actuator_names=source_names,
        obs_size=_metadata_shape_size(checkpoint_dir, "('run_stats', 'RunningMeanStd_0', 'mean')"),
        obs_fields=[],
    )


def _reconstruct_manifest_for_precheck(
    checkpoint_dir: str | Path,
    metadata: dict[str, Any],
    scene_actuator_names: list[str],
):
    from environment.overall_environment.src.action_manifest import reconstruct_action_manifest

    try:
        return reconstruct_action_manifest(checkpoint_dir)
    except ModuleNotFoundError as exc:
        if exc.name not in {"jax", "flax", "brax", "mujoco_mjx"}:
            raise
        return _fallback_disable_fingers_manifest(checkpoint_dir, metadata, scene_actuator_names)


def replay_precheck(paths: GripHoldPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    import mujoco

    from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter
    from environment.overall_environment.src.control_scaling import (
        apply_checkpoint_ctrl_ranges_to_model,
        normalized_action_to_model_ctrl,
    )
    from environment.overall_environment.src.layered_control import actuator_names_from_model

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report = preflight(paths, out_dir=out_path)
    metadata = checkpoint_metadata(paths.resume_from)
    tags = metadata.get("wandb", {}).get("tags", [])
    env_params = metadata.get("experiment", {}).get("env_params", {})
    goal_params = env_params.get("goal_params", {})
    dataset_conf = (
        metadata.get("experiment", {})
        .get("task_factory", {})
        .get("params", {})
        .get("amass_dataset_conf", {})
    )
    scene_model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    ctrl_range_report = apply_checkpoint_ctrl_ranges_to_model(scene_model, paths.resume_from)
    scene_actuator_names = actuator_names_from_model(scene_model)
    body_manifest = _reconstruct_manifest_for_precheck(paths.resume_from, metadata, scene_actuator_names)
    adapter = CheckpointToFullActionAdapter(body_manifest.actuator_names, scene_actuator_names)
    adapter_report = adapter.report()
    report.update(
        {
            "runner_stage": "replay-precheck",
            "checkpoint_tags": tags,
            "checkpoint_tags_match": "forehand_clear" in tags,
            "base_env_name": env_params.get("env_name"),
            "base_disable_fingers": bool(env_params.get("disable_fingers", False)),
            "base_goal_type": env_params.get("goal_type"),
            "base_goal_lookahead": goal_params.get("n_step_lookahead"),
            "base_goal_stride": goal_params.get("n_step_stride"),
            "base_sites_for_mimic": goal_params.get("sites_for_mimic", []),
            "base_motion_paths": dataset_conf.get("rel_dataset_path", []),
            "checkpoint_action_size": body_manifest.action_size,
            "checkpoint_obs_size": body_manifest.obs_size,
            "checkpoint_disable_fingers": body_manifest.disable_fingers,
            "scene_action_size": int(scene_model.nu),
            "body_ctrl_range_overrides": asdict(ctrl_range_report),
            "action_adapter_ready": True,
            "adapter_mapped_count": adapter_report.mapped_count,
            "adapter_extra_in_target_count": len(adapter_report.extra_in_target),
            "adapter_extra_in_target": adapter_report.extra_in_target,
            "policy_replay_ready": False,
            "blocked_reason": (
                "Frozen policy replay still needs checkpoint actor loading, observation compatibility "
                "checks, and layered body/grip action routing."
            ),
        }
    )
    (out_path / "replay_precheck_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def replay_smoke(
    paths: GripHoldPaths,
    *,
    out_dir: str | Path | None = None,
    steps: int = 100,
    policy_source: str = "fake",
    pose_servo_debug: bool = False,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if policy_source not in {"fake", "real"}:
        raise ValueError(f"policy_source must be 'fake' or 'real', got {policy_source!r}")

    import mujoco

    from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter
    from environment.overall_environment.src.control_scaling import (
        apply_checkpoint_ctrl_ranges_to_model,
        normalized_action_to_model_ctrl,
    )
    from environment.overall_environment.src.layered_control import actuator_names_from_model
    from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
    from environment.overall_environment.src.training_scene import (
        build_training_scene_report,
        validate_training_scene_report,
    )

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report = replay_precheck(paths, out_dir=out_path)
    report["runner_stage"] = "replay-smoke"
    report["policy_source"] = policy_source
    report["pose_servo_enabled"] = bool(pose_servo_debug)
    report["servo_scope"] = "all_debug" if pose_servo_debug else "none"

    try:
        training_scene_report = build_training_scene_report(paths.scene_xml)
        validate_training_scene_report(training_scene_report)
    except Exception as exc:
        report.update(
            {
                "fake_policy_replay_ready": False,
                "policy_replay_ready": False,
                "steps_requested": steps,
                "steps_completed": 0,
                "finite": False,
                "blocked_reason": f"training scene validation failed: {exc}",
            }
        )
        _write_replay_smoke_report(out_path, report)
        return report

    if policy_source == "real":
        from environment.overall_environment.src.body_obs_adapter import BodyObsAdapter
        from environment.overall_environment.src.frozen_body_policy import (
            FrozenBodyPolicy,
            FrozenBodyPolicyArtifactError,
            validate_actor_checkpoint_shapes,
        )
        from environment.overall_environment.src.overall_grip_hold_env import align_racket_grip_to_palm
        from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider

        env = OverallBadmintonEnvironment(paths.scene_xml)
        obs, _ = env.reset()
        ctrl_range_report = apply_checkpoint_ctrl_ranges_to_model(env.model, paths.resume_from)
        keyframe_grip_to_palm = _grip_to_palm_vector(env.model, env.data)
        obs_adapter = BodyObsAdapter.from_checkpoint(paths.resume_from)
        obs_report = obs_adapter.check_compatibility(overall_obs_size=int(obs.size))
        shape_report = validate_actor_checkpoint_shapes(paths.resume_from)
        artifact_path = paths.body_policy_artifact
        if artifact_path is None or not artifact_path.is_dir():
            report.update(
                {
                    "fake_policy_replay_ready": False,
                    "policy_replay_ready": False,
                    "overall_obs_size": int(obs.size),
                    "body_obs_compatibility": asdict(obs_report),
                    "actor_checkpoint_shapes": asdict(shape_report),
                    "steps_requested": steps,
                    "steps_completed": 0,
                    "finite": False,
                    "blocked_reason": f"body policy artifact not found: {artifact_path}",
                }
            )
            _write_replay_smoke_report(out_path, report)
            return report

        metadata = checkpoint_metadata(paths.resume_from)
        scene_actuator_names = actuator_names_from_model(env.model)
        body_manifest = _reconstruct_manifest_for_precheck(paths.resume_from, metadata, scene_actuator_names)
        adapter = CheckpointToFullActionAdapter(body_manifest.actuator_names, scene_actuator_names)
        try:
            body_policy = FrozenBodyPolicy.load_from_export(artifact_path)
        except FrozenBodyPolicyArtifactError as exc:
            report.update(
                {
                    "fake_policy_replay_ready": False,
                    "policy_replay_ready": False,
                    "overall_obs_size": int(obs.size),
                    "body_obs_compatibility": asdict(obs_report),
                    "actor_checkpoint_shapes": asdict(shape_report),
                    "steps_requested": steps,
                    "steps_completed": 0,
                    "finite": False,
                    "blocked_reason": str(exc),
                }
            )
            _write_replay_smoke_report(out_path, report)
            return report

        goal_provider = TrajectoryGoalProvider.from_checkpoint(paths.resume_from)
        goal_provider.reset(traj_step=0)
        reset_reference = goal_provider.apply_reference_state_to(env.model, env.data, traj_step=0)
        align_racket_grip_to_palm(
            env.model,
            env.data,
            reference_grip_to_palm=keyframe_grip_to_palm,
        )
        env._servo_qpos = np.array(env.data.qpos, dtype=float)
        obs = np.concatenate([np.array(env.data.qpos, dtype=float), np.array(env.data.qvel, dtype=float)])
        goal_size = int(goal_provider.goal_size)
        finite = bool(np.isfinite(obs).all())
        steps_completed = 0
        max_abs_obs = float(np.max(np.abs(obs))) if obs.size else 0.0
        max_palm_to_grip_m = _palm_to_grip_distance(env.model, env.data)
        raw_body_action_max_abs = 0.0
        clipped_full_action_max_abs = 0.0
        servo_force_norm_max = 0.0
        body_obs_max_abs = 0.0
        goal_obs_mean_abs_values: list[float] = []
        raw_body_action_mean_abs_values: list[float] = []
        full_ctrl_saturation_count = 0
        full_ctrl_count = 0
        body_obs_size = 0
        body_action_size = 0
        for _ in range(steps):
            goal_obs = goal_provider.build(env.model, env.data)
            body_obs = obs_adapter.build_from_mujoco(env.model, env.data, goal_obs=goal_obs)
            body_action = body_policy.act(body_obs)
            full_action = adapter.adapt(body_action)
            full_ctrl = normalized_action_to_model_ctrl(env.model, full_action)

            body_obs_size = int(body_obs.size)
            body_action_size = int(body_action.size)
            if body_obs.size:
                body_obs_max_abs = max(body_obs_max_abs, float(np.max(np.abs(body_obs))))
            if goal_obs.size:
                goal_obs_mean_abs_values.append(float(np.mean(np.abs(goal_obs))))
            if body_action.size:
                raw_body_action_max_abs = max(raw_body_action_max_abs, float(np.max(np.abs(body_action))))
                raw_body_action_mean_abs_values.append(float(np.mean(np.abs(body_action))))
            if full_ctrl.size:
                clipped_full_action_max_abs = max(
                    clipped_full_action_max_abs,
                    float(np.max(np.abs(full_ctrl))),
                )
                limited = np.asarray(env.model.actuator_ctrllimited, dtype=bool)
                if limited.any():
                    lower = np.asarray(env.model.actuator_ctrlrange[:, 0], dtype=float)
                    upper = np.asarray(env.model.actuator_ctrlrange[:, 1], dtype=float)
                    full_ctrl_saturation_count += int(
                        np.count_nonzero(
                            np.isclose(full_ctrl[limited], lower[limited], atol=1e-6)
                            | np.isclose(full_ctrl[limited], upper[limited], atol=1e-6)
                        )
                    )
                    full_ctrl_count += int(np.count_nonzero(limited))

            obs, servo_norm = _step_control_interval(
                env,
                full_ctrl,
                pose_servo=pose_servo_debug,
                control_substeps=DEFAULT_BODY_CONTROL_SUBSTEPS,
            )
            servo_force_norm_max = max(servo_force_norm_max, servo_norm)
            steps_completed += 1
            goal_provider.advance()
            obs_finite = bool(np.isfinite(obs).all())
            finite = finite and obs_finite
            if obs.size:
                max_abs_obs = max(max_abs_obs, float(np.max(np.abs(obs))))
            max_palm_to_grip_m = max(max_palm_to_grip_m, _palm_to_grip_distance(env.model, env.data))
            if not obs_finite:
                break

        final_palm_to_grip_m = _palm_to_grip_distance(env.model, env.data)
        racket_drop = (not np.isfinite(final_palm_to_grip_m)) or final_palm_to_grip_m > 0.25
        ready = bool(finite and steps_completed == steps and not racket_drop)
        report.update(
            {
                "fake_policy_replay_ready": False,
                "policy_replay_ready": ready,
                "overall_obs_size": int(obs.size),
                "body_obs_compatibility": asdict(obs_report),
                "actor_checkpoint_shapes": asdict(shape_report),
                "body_obs_size": body_obs_size,
                "body_action_size": body_action_size,
                "goal_obs_size": goal_size,
                "goal_obs_source": goal_provider.source,
                "goal_motion_path": goal_provider.motion_path,
                "reset_mode": "trajectory",
                "reset_traj_step": int(reset_reference.traj_step),
                "control_substeps": DEFAULT_BODY_CONTROL_SUBSTEPS,
                "physics_timestep_s": float(env.model.opt.timestep),
                "policy_control_dt_s": float(env.model.opt.timestep * DEFAULT_BODY_CONTROL_SUBSTEPS),
                "body_ctrl_range_overrides": asdict(ctrl_range_report),
                "goal_traj_step": goal_provider.last_built_step,
                "goal_next_traj_step": goal_provider.traj_step,
                "goal_traj_len": goal_provider.traj_len,
                "steps_requested": steps,
                "steps_completed": steps_completed,
                "finite": bool(finite),
                "scene_validation_ready": True,
                "max_abs_obs": max_abs_obs,
                "max_palm_to_grip_m": max_palm_to_grip_m,
                "final_palm_to_grip_m": final_palm_to_grip_m,
                "racket_drop": bool(racket_drop),
                "raw_body_action_max_abs": raw_body_action_max_abs,
                "clipped_full_action_max_abs": clipped_full_action_max_abs,
                "servo_force_norm_max": servo_force_norm_max,
                "body_obs_max_abs": body_obs_max_abs,
                "goal_obs_mean_abs": float(np.mean(goal_obs_mean_abs_values)) if goal_obs_mean_abs_values else 0.0,
                "raw_body_action_mean_abs": float(np.mean(raw_body_action_mean_abs_values))
                if raw_body_action_mean_abs_values
                else 0.0,
                "full_ctrl_saturation_rate": float(full_ctrl_saturation_count / max(full_ctrl_count, 1)),
                "blocked_reason": "" if ready else "real body policy replay-smoke did not stay finite/stable",
            }
        )
        _write_replay_smoke_report(out_path, report)
        return report

    metadata = checkpoint_metadata(paths.resume_from)
    env = OverallBadmintonEnvironment(paths.scene_xml)
    obs, _ = env.reset()
    ctrl_range_report = apply_checkpoint_ctrl_ranges_to_model(env.model, paths.resume_from)
    scene_actuator_names = actuator_names_from_model(env.model)
    body_manifest = _reconstruct_manifest_for_precheck(paths.resume_from, metadata, scene_actuator_names)
    adapter = CheckpointToFullActionAdapter(body_manifest.actuator_names, scene_actuator_names)
    fake_body_action = np.zeros(body_manifest.action_size, dtype=float)
    full_action = adapter.adapt(fake_body_action)
    full_ctrl = normalized_action_to_model_ctrl(env.model, full_action)

    finite = bool(np.isfinite(obs).all() and np.isfinite(full_ctrl).all())
    steps_completed = 0
    max_abs_obs = float(np.max(np.abs(obs))) if obs.size else 0.0
    max_palm_to_grip_m = _palm_to_grip_distance(env.model, env.data)
    servo_force_norm_max = 0.0
    for _ in range(steps):
        obs, servo_norm = _step_control_interval(
            env,
            full_ctrl,
            pose_servo=pose_servo_debug,
            control_substeps=DEFAULT_BODY_CONTROL_SUBSTEPS,
        )
        servo_force_norm_max = max(servo_force_norm_max, servo_norm)
        steps_completed += 1
        obs_finite = bool(np.isfinite(obs).all())
        finite = finite and obs_finite
        if obs.size:
            max_abs_obs = max(max_abs_obs, float(np.max(np.abs(obs))))
        max_palm_to_grip_m = max(max_palm_to_grip_m, _palm_to_grip_distance(env.model, env.data))
        if not obs_finite:
            break

    final_palm_to_grip_m = _palm_to_grip_distance(env.model, env.data)
    racket_drop = (not np.isfinite(final_palm_to_grip_m)) or final_palm_to_grip_m > 0.25
    report.update(
        {
            "fake_policy_replay_ready": bool(finite and steps_completed == steps),
            "policy_replay_ready": False,
            "steps_requested": steps,
            "steps_completed": steps_completed,
            "finite": bool(finite),
            "scene_validation_ready": True,
            "max_abs_obs": max_abs_obs,
            "max_palm_to_grip_m": max_palm_to_grip_m,
            "final_palm_to_grip_m": final_palm_to_grip_m,
            "racket_drop": bool(racket_drop),
            "servo_force_norm_max": servo_force_norm_max,
            "control_substeps": DEFAULT_BODY_CONTROL_SUBSTEPS,
            "physics_timestep_s": float(env.model.opt.timestep),
            "policy_control_dt_s": float(env.model.opt.timestep * DEFAULT_BODY_CONTROL_SUBSTEPS),
            "body_ctrl_range_overrides": asdict(ctrl_range_report),
            "blocked_reason": (
                "fake body replay-smoke passed; real checkpoint replay is still blocked by "
                "body observation adapter and actor loading"
            ),
        }
    )
    _write_replay_smoke_report(out_path, report)
    return report


def _palm_to_grip_distance(model, data) -> float:
    import mujoco

    palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    if palm_site < 0 or grip_site < 0:
        return float("inf")
    return float(np.linalg.norm(data.site_xpos[palm_site] - data.site_xpos[grip_site]))


def _grip_to_palm_vector(model, data) -> np.ndarray:
    import mujoco

    palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    if palm_site < 0 or grip_site < 0:
        raise ValueError("missing palm or grip site")
    return np.asarray(data.site_xpos[grip_site] - data.site_xpos[palm_site], dtype=float)


def _step_control_interval(env, ctrl: np.ndarray, *, pose_servo: bool, control_substeps: int) -> tuple[np.ndarray, float]:
    obs = np.zeros(env.model.nq + env.model.nv, dtype=float)
    servo_force_norm_max = 0.0
    for _ in range(int(control_substeps)):
        obs, _ = env.step(ctrl=ctrl, pose_servo=pose_servo)
        servo_force_norm_max = max(servo_force_norm_max, float(np.linalg.norm(env.data.qfrc_applied)))
    return obs, servo_force_norm_max


def _write_replay_smoke_report(out_path: Path, report: dict[str, Any]) -> None:
    (out_path / "replay_smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def train_tiny(
    paths: GripHoldPaths,
    *,
    out_dir: str | Path | None = None,
    total_steps: int = 128,
    rollout_steps: int = 32,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")

    import torch

    from environment.overall_environment.src.overall_grip_hold_env import OverallGripHoldEnv
    from src.grip.train_right_hand_racket_grip_policy import (
        PPOConfig,
        PolicyValueNet,
        RunningMeanStd,
        _empty_rollout,
        _gae,
        _json_safe,
        _mean_last,
        _ppo_update,
        _sample_action,
        _tensor,
    )

    torch.set_num_threads(1)
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "tiny_train"
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = OverallGripHoldEnv(
        paths.scene_xml,
        residual_groups=["right_hand_fingers"],
        max_episode_steps=max(rollout_steps, 8),
        body_policy_artifact=paths.body_policy_artifact,
        body_checkpoint=paths.resume_from,
        reward_weights=paths.reward_weights,
    )
    obs, last_info = env.reset()
    obs_size = int(obs.size)
    action_size = int(env.action_size)
    obs_rms = RunningMeanStd((obs_size,))
    obs_rms.update(obs)
    ppo_config = PPOConfig(
        total_steps=int(total_steps),
        rollout_steps=int(rollout_steps),
        minibatch_size=int(rollout_steps),
        update_epochs=1,
        hidden_sizes=(64, 64),
        seed=int(seed),
        action_std_init=0.25,
    )
    model = PolicyValueNet(obs_size, action_size, ppo_config.hidden_sizes, ppo_config.action_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)

    global_step = 0
    update_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    summaries: list[dict[str, float]] = []
    finite = True
    last_step_info = dict(last_info)

    while global_step < ppo_config.total_steps:
        rollout_target = min(ppo_config.rollout_steps, ppo_config.total_steps - global_step)
        rollout = _empty_rollout(rollout_target, obs_size, action_size)
        for step in range(rollout_target):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalize(obs)
            action, logprob, value = _sample_action(torch, model, obs_norm, device, rng)
            next_obs, reward, terminated, truncated, last_info = env.step(action)
            last_step_info = dict(last_info)
            done = bool(terminated or truncated)
            finite = finite and bool(np.isfinite(next_obs).all()) and bool(np.isfinite(reward))

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
                "mean_rollout_reward": float(np.mean(rollout["rewards"])),
                "mean_episode_return": _mean_last(completed_returns, 10),
                "mean_episode_length": _mean_last(completed_lengths, 10),
                "grip_slip_m": float(last_step_info["grip_slip_m"]),
                "hand_handle_contact_count": float(last_step_info["hand_handle_contact_count"]),
                "racket_drop": float(bool(last_step_info["racket_drop"])),
                "body_fall": float(bool(last_step_info["body_fall"])),
            }
        )
        summaries.append(update_summary)

    metrics = {
        "runner_stage": "train-tiny",
        "policy_source": env.body_policy_source,
        "scene_xml": str(paths.scene_xml),
        "resume_from": str(paths.resume_from),
        "output_dir": str(out_path),
        "obs_size": obs_size,
        "action_size": action_size,
        "global_step": int(global_step),
        "updates": int(update_index),
        "finite": bool(finite),
        "mean_episode_return_last10": _mean_last(completed_returns, 10),
        "mean_episode_length_last10": _mean_last(completed_lengths, 10),
        "last_info": _json_safe(last_step_info),
        "updates_detail": _json_safe(summaries),
    }
    checkpoint_path = out_path / "policy_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics["policy_checkpoint"] = str(checkpoint_path)
    (out_path / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="experiments/posttrain/forehand_clear_grip_hold_v1.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "reset-video", "replay-precheck", "replay-smoke", "replay-video", "train-tiny"),
        default="preflight",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps", type=int, default=100, help="Number of MuJoCo steps for replay-smoke.")
    parser.add_argument(
        "--policy-source",
        choices=("fake", "real"),
        default="fake",
        help="Use zero fake body actions for replay-smoke, or fail fast for the real checkpoint path.",
    )
    parser.add_argument(
        "--pose-servo-debug",
        action="store_true",
        help="Enable legacy pose servo during replay-smoke for debugging only.",
    )
    parser.add_argument("--total-steps", type=int, default=128, help="Total PPO steps for train-tiny.")
    parser.add_argument("--rollout-steps", type=int, default=32, help="Rollout length for train-tiny.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train-tiny.")
    parser.add_argument("--device", default="cpu", help="Torch device for train-tiny.")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for replay-video.")
    parser.add_argument("--render-stride", type=int, default=1, help="Render every N simulation steps for replay-video.")
    args = parser.parse_args()

    paths = load_grip_hold_spec(args.spec)
    if args.stage == "reset-video":
        report = diagnostic_reset(paths, out_dir=args.out_dir)
    elif args.stage == "replay-precheck":
        report = replay_precheck(paths, out_dir=args.out_dir)
    elif args.stage == "replay-smoke":
        report = replay_smoke(
            paths,
            out_dir=args.out_dir,
            steps=args.steps,
            policy_source=args.policy_source,
            pose_servo_debug=args.pose_servo_debug,
        )
    elif args.stage == "replay-video":
        report = record_replay_video(
            paths,
            out_dir=args.out_dir,
            steps=args.steps,
            fps=args.video_fps,
            render_stride=args.render_stride,
        )
    elif args.stage == "train-tiny":
        report = train_tiny(
            paths,
            out_dir=args.out_dir,
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            device=args.device,
        )
    else:
        report = preflight(paths, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
