#!/usr/bin/env python3
"""Prepare and validate ForehandClear grip-hold dependencies before PPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musclemimic.badminton.scripts.run_forehand_clear_grip_hold import load_grip_hold_spec
from environment.overall_environment.src.frozen_body_policy import (
    export_frozen_body_policy,
    load_frozen_body_policy_manifest,
)
from environment.overall_environment.src.training_scene import (
    build_training_scene_report,
    validate_training_scene_report,
)
from environment.overall_environment.src.trajectory_goal_provider import TrajectoryGoalProvider


def prepare_artifacts(
    spec: str | Path,
    *,
    out_dir: str | Path | None = None,
    build_training_scene: bool = False,
    export_frozen_policy: bool = False,
    check_trajectory_cache: bool = False,
    check_grip_seed: bool = False,
) -> dict[str, Any]:
    paths = load_grip_hold_spec(spec)
    spec_data = _load_yaml(paths.spec_path)
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "prepare_report"
    out_path.mkdir(parents=True, exist_ok=True)

    if build_training_scene:
        from environment.overall_environment.src.build_overall_environment import build_overall_scene

        build_overall_scene(
            paths.scene_xml,
            grip_seed=paths.grip_seed,
            mode="training",
        )
    if export_frozen_policy:
        if paths.body_policy_artifact is None:
            raise ValueError("spec has no body_policy.artifact")
        export_frozen_body_policy(paths.resume_from, paths.body_policy_artifact, restore_tensors=True)

    _require(paths.resume_from.is_dir(), f"checkpoint missing: {paths.resume_from}")
    _require(paths.scene_xml.is_file(), f"training scene missing: {paths.scene_xml}")
    if check_grip_seed:
        _require(paths.grip_seed.is_file(), f"grip seed missing: {paths.grip_seed}")
    _require(paths.body_policy_artifact is not None, "spec has no body_policy.artifact")
    artifact = paths.body_policy_artifact
    _require(artifact.is_dir(), f"body policy artifact missing: {artifact}")
    _require((artifact / "params.npz").is_file(), f"artifact missing params.npz: {artifact}")
    _require((artifact / "run_stats.npz").is_file(), f"artifact missing run_stats.npz: {artifact}")

    manifest = load_frozen_body_policy_manifest(artifact)
    spec_checkpoint = _resolve_from_spec(paths.spec_path, spec_data.get("body_policy", {}).get("checkpoint", paths.resume_from))
    artifact_source_checkpoint_matches_spec = Path(manifest.source_checkpoint).resolve() == spec_checkpoint.resolve()
    _require(
        artifact_source_checkpoint_matches_spec,
        f"artifact source checkpoint != spec body_policy.checkpoint: {manifest.source_checkpoint} != {spec_checkpoint}",
    )

    scene_report = build_training_scene_report(paths.scene_xml)
    validate_training_scene_report(scene_report)
    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    actuation_enabled = bool(model.nu > 0 and not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION))
    _require(actuation_enabled, "training scene lacks enabled actuators")
    hand_racket_contact_allowed = not scene_report.has_fullbody_racket_exclude
    _require(hand_racket_contact_allowed, "Full Body - overall_racket contact exclude exists")

    goal_size = 0
    trajectory_cache_ok = False
    goal_provider: TrajectoryGoalProvider | None = None
    if check_trajectory_cache:
        goal_provider = TrajectoryGoalProvider.from_checkpoint(paths.resume_from)
        trajectory_cache_ok = goal_provider.cache_path.is_file()
        goal_size = int(goal_provider.goal_size)
        _require(trajectory_cache_ok, f"trajectory cache missing: {goal_provider.cache_path}")
    else:
        try:
            goal_provider = TrajectoryGoalProvider.from_checkpoint(paths.resume_from)
            trajectory_cache_ok = goal_provider.cache_path.is_file()
            goal_size = int(goal_provider.goal_size) if trajectory_cache_ok else 0
        except Exception:
            trajectory_cache_ok = False
            goal_size = 0

    report = {
        "spec_path": str(paths.spec_path),
        "checkpoint_exists": paths.resume_from.is_dir(),
        "body_policy_artifact": str(artifact),
        "artifact_source_checkpoint_matches_spec": artifact_source_checkpoint_matches_spec,
        "params_npz_exists": (artifact / "params.npz").is_file(),
        "run_stats_npz_exists": (artifact / "run_stats.npz").is_file(),
        "training_scene_ok": True,
        "actuation_enabled": actuation_enabled,
        "hand_racket_contact_allowed": hand_racket_contact_allowed,
        "trajectory_cache_ok": trajectory_cache_ok,
        "trajectory_cache_path": str(goal_provider.cache_path) if goal_provider is not None else None,
        "grip_seed_ok": paths.grip_seed.is_file(),
        "grip_seed_path": str(paths.grip_seed),
        "body_obs_size": int(manifest.actor_spec.obs_size),
        "goal_size": goal_size,
        "body_action_size": int(manifest.actor_spec.action_size),
        "overall_action_size": int(model.nu),
        "scene_validation": {
            "keyframes": scene_report.keyframes,
            "actuator_count": scene_report.actuator_count,
            "missing_sites": scene_report.missing_sites,
            "missing_geoms": scene_report.missing_geoms,
            "has_fullbody_racket_exclude": scene_report.has_fullbody_racket_exclude,
        },
    }
    (out_path / "prepare_artifacts_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _resolve_from_spec(spec_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    parser.add_argument("--build-training-scene", action="store_true")
    parser.add_argument("--export-frozen-policy", action="store_true")
    parser.add_argument("--check-trajectory-cache", action="store_true")
    parser.add_argument("--check-grip-seed", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    report = prepare_artifacts(
        args.spec,
        out_dir=args.out_dir,
        build_training_scene=args.build_training_scene,
        export_frozen_policy=args.export_frozen_policy,
        check_trajectory_cache=args.check_trajectory_cache,
        check_grip_seed=args.check_grip_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
