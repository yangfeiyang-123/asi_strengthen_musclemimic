#!/usr/bin/env python3
"""Dedicated ForehandClear static-hit preflight and physics-smoke runner."""

from __future__ import annotations

import argparse
import json
import math
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
class StaticHitPaths:
    spec_path: Path
    output_dir: Path
    resume_from: Path
    scene_xml: Path
    grip_policy_checkpoint: Path
    shuttle_qpos: np.ndarray
    runner_type: str
    validation: dict[str, Any]
    player_half_sign: int


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _midpoint(config: dict[str, Any], low_key: str, high_key: str, default_low: float, default_high: float) -> float:
    low = float(config.get(low_key, default_low))
    high = float(config.get(high_key, default_high))
    if high < low:
        raise ValueError(f"{high_key} must be >= {low_key}, got {high} < {low}")
    return 0.5 * (low + high)


def _normalize_static_shuttle_qpos(value: Any, *, source: str) -> np.ndarray:
    qpos = np.asarray(value, dtype=float)
    if qpos.shape != (7,):
        raise ValueError(f"{source} must have shape (7,), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{source} must contain only finite values")
    quat_norm = float(np.linalg.norm(qpos[3:7]))
    if quat_norm <= 1e-8:
        raise ValueError(f"{source} quaternion must be nonzero")
    qpos = qpos.copy()
    qpos[3:7] /= quat_norm
    return qpos


def _build_static_shuttle_qpos(
    scene: dict[str, Any],
    impact_target: dict[str, Any],
    shuttle: dict[str, Any],
) -> np.ndarray:
    explicit_qpos = shuttle.get("static_qpos", shuttle.get("qpos"))
    if explicit_qpos is not None:
        return _normalize_static_shuttle_qpos(explicit_qpos, source="shuttle.static_qpos")

    root_xy = np.asarray(scene.get("root_start_xy", [-2.5, 0.0]), dtype=float)
    if root_xy.shape != (2,) or not np.isfinite(root_xy).all():
        raise ValueError(f"scene.root_start_xy must have shape (2,), got {root_xy.shape}")

    regularization = impact_target.get("regularization", {})
    if not isinstance(regularization, dict):
        regularization = {}
    forward_offset = _midpoint(regularization, "min_forward_offset_m", "max_forward_offset_m", 0.28, 0.90)
    side_offset = _midpoint(regularization, "min_racket_side_offset_m", "max_racket_side_offset_m", 0.16, 0.70)
    height = float(shuttle.get("static_height_m", impact_target.get("static_height_m", 1.4)))
    if height <= 1.0:
        raise ValueError(f"static shuttle height must be above 1.0m, got {height}")

    player_half_sign = int(scene.get("player_half_sign", -1))
    if player_half_sign == 0:
        raise ValueError("scene.player_half_sign must be nonzero")
    forward_sign = -player_half_sign
    side_sign = int(shuttle.get("side_sign", 1))
    if side_sign == 0:
        raise ValueError("shuttle.side_sign must be nonzero")

    return np.asarray(
        [
            float(root_xy[0] + forward_sign * forward_offset),
            float(root_xy[1] + side_sign * side_offset),
            height,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def load_static_hit_spec(spec_path: str | Path) -> StaticHitPaths:
    resolved_spec = _resolve(spec_path)
    data = yaml.safe_load(resolved_spec.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_spec} must contain a YAML mapping")
    if data.get("runner_type") != "static_hit_staging":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")
    if data.get("action") != "ForehandClearStaticHit":
        raise ValueError(f"unsupported action: {data.get('action')!r}")

    scene = data.get("scene", {})
    impact_target = data.get("impact_target", {})
    shuttle = data.get("shuttle", {})
    grip_policy = data.get("grip_policy", {})
    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    validation = dict(data.get("validation", {}))
    validation.setdefault("no_pose_servo", True)
    validation.setdefault("no_racket_drop", True)
    validation.setdefault("no_fall", True)
    validation.setdefault("impact", {})
    validation["impact"].setdefault("require_detected", True)
    validation.setdefault("flight", {})
    if "target_landing_region" in validation["flight"]:
        validation["flight"].setdefault("require_over_net", True)

    return StaticHitPaths(
        spec_path=resolved_spec,
        output_dir=output_dir,
        resume_from=_resolve(data["resume_from"]),
        scene_xml=_resolve(scene["xml"]),
        grip_policy_checkpoint=_resolve(grip_policy["checkpoint"]),
        shuttle_qpos=_build_static_shuttle_qpos(scene, impact_target, shuttle),
        runner_type="static_hit",
        validation=validation,
        player_half_sign=int(scene.get("player_half_sign", -1)),
    )


def preflight(paths: StaticHitPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    from environment.overall_environment.src.training_scene import (
        build_training_scene_report,
        validate_training_scene_report,
    )

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "static_hit_preflight"
    out_path.mkdir(parents=True, exist_ok=True)
    scene_report = None
    scene_error = ""
    try:
        scene_report = build_training_scene_report(paths.scene_xml)
        validate_training_scene_report(scene_report)
    except Exception as exc:
        scene_error = str(exc)

    report = {
        "runner_type": paths.runner_type,
        "spec_path": str(paths.spec_path),
        "output_dir": str(out_path),
        "resume_from": str(paths.resume_from),
        "scene_xml": str(paths.scene_xml),
        "grip_policy_checkpoint": str(paths.grip_policy_checkpoint),
        "shuttle_qpos": paths.shuttle_qpos,
        "checkpoint_exists": paths.resume_from.is_dir(),
        "scene_exists": paths.scene_xml.is_file(),
        "grip_policy_exists": paths.grip_policy_checkpoint.is_file(),
        "scene_validation_error": scene_error,
        "scene_validation_ready": scene_report is not None and not scene_error,
        "actuation_enabled": bool(scene_report and scene_report.actuator_count > 0),
        "hand_racket_contact_allowed": bool(scene_report and not scene_report.has_fullbody_racket_exclude),
        "pose_servo_allowed": False,
        "validation": paths.validation,
    }
    (out_path / "preflight_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def physics_smoke(
    paths: StaticHitPaths,
    *,
    out_dir: str | Path | None = None,
    steps: int = 120,
) -> dict[str, Any]:
    import mujoco

    from environment.overall_environment.src.overall_env import OverallBadmintonEnvironment
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv, StaticShuttleTarget

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "static_hit_physics_smoke"
    out_path.mkdir(parents=True, exist_ok=True)
    report = preflight(paths, out_dir=out_path)

    base = OverallBadmintonEnvironment(paths.scene_xml)
    obs, _base_info = base.reset()
    shuttle_joint = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    if shuttle_joint < 0:
        raise ValueError("static-hit scene missing overall_shuttle_free joint")
    qpos_adr = int(base.model.jnt_qposadr[shuttle_joint])
    qvel_adr = int(base.model.jnt_dofadr[shuttle_joint])
    target_qpos = np.asarray(paths.shuttle_qpos, dtype=float).copy()

    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=StaticShuttleTarget(qpos_adr=qpos_adr, qvel_adr=qvel_adr, qpos=target_qpos),
        impact_phase=0.5,
        phase_tolerance=0.08,
        episode_steps=max(int(steps), 2),
        player_half_sign=paths.player_half_sign,
    )
    obs, reset_info = env.reset()
    finite = bool(np.isfinite(obs).all())
    last_info: dict[str, Any] = dict(reset_info)
    terminated = False
    truncated = False
    for _ in range(int(steps)):
        obs, reward, terminated, truncated, last_info = env.step(ctrl=np.zeros(base.model.nu, dtype=float))
        finite = finite and bool(np.isfinite(obs).all()) and math.isfinite(float(reward))
        if terminated or truncated or not finite:
            break

    flight = dict(last_info.get("flight", {}))
    smoke = {
        **report,
        "runner_stage": "physics-smoke",
        "steps_requested": int(steps),
        "steps_completed": int(last_info.get("step_count", env.step_index)),
        "finite": bool(finite),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "pose_servo_enabled": False,
        "racket_drop": bool(last_info.get("racket_drop", False)),
        "body_fall": bool(last_info.get("body_fall", False)),
        "impact_detected": env.release_step is not None,
        "release_step": env.release_step,
        "over_net": bool(flight.get("crossed_net", False)),
        "landing_region": flight.get("region"),
        "target_landing_region": paths.validation.get("flight", {}).get("target_landing_region"),
        "last_info": _json_safe(last_info),
    }
    smoke["acceptance"] = static_hit_acceptance(smoke, paths.validation)
    (out_path / "physics_smoke_report.json").write_text(
        json.dumps(_json_safe(smoke), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return smoke


def static_hit_acceptance(report: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if validation.get("require_finite", True) and not bool(report.get("finite", False)):
        failures.append("nonfinite")
    if validation.get("no_pose_servo", True) and bool(report.get("pose_servo_enabled", False)):
        failures.append("pose_servo_enabled")
    if validation.get("no_racket_drop", True) and bool(report.get("racket_drop", False)):
        failures.append("racket_drop")
    if validation.get("no_fall", True) and bool(report.get("body_fall", False)):
        failures.append("body_fall")

    impact_cfg = validation.get("impact", {}) if isinstance(validation.get("impact", {}), dict) else {}
    if impact_cfg.get("require_detected", False) and not bool(report.get("impact_detected", False)):
        failures.append("impact_not_detected")

    flight_cfg = validation.get("flight", {}) if isinstance(validation.get("flight", {}), dict) else {}
    if flight_cfg.get("require_over_net", False) and not bool(report.get("over_net", False)):
        failures.append("not_over_net")
    target_region = flight_cfg.get("target_landing_region")
    if target_region and report.get("landing_region") != target_region:
        failures.append("landing_region_mismatch")

    return {"passed": not failures, "failures": failures}


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
        default="BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml",
    )
    parser.add_argument("--stage", choices=("preflight", "physics-smoke"), default="preflight")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    paths = load_static_hit_spec(args.spec)
    if args.stage == "physics-smoke":
        report = physics_smoke(paths, out_dir=args.out_dir, steps=args.steps)
    else:
        report = preflight(paths, out_dir=args.out_dir)
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
