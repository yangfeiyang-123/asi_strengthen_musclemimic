#!/usr/bin/env python3
"""Run ForehandClear grip-hold diagnostics.

This script does not train yet. Training requires checkpoint action manifests,
name-based action adaptation, observation compatibility checks, and layered
body/grip action routing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class GripHoldPaths:
    spec_path: Path
    runner_type: str
    resume_from: Path
    scene_xml: Path
    grip_seed: Path
    output_dir: Path


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
    return GripHoldPaths(
        spec_path=resolved_spec,
        runner_type=str(data["runner_type"]),
        resume_from=_resolve(data["resume_from"]),
        scene_xml=_resolve(data["scene"]["xml"]),
        grip_seed=_resolve(data["grip_seed"]["path"]),
        output_dir=output_dir,
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
        "output_dir": str(out_path),
        "checkpoint_exists": paths.resume_from.is_dir(),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    parser.add_argument("--stage", choices=("preflight", "reset-video", "replay-precheck"), default="preflight")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    paths = load_grip_hold_spec(args.spec)
    if args.stage == "reset-video":
        report = diagnostic_reset(paths, out_dir=args.out_dir)
    elif args.stage == "replay-precheck":
        report = replay_precheck(paths, out_dir=args.out_dir)
    else:
        report = preflight(paths, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
