from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.paths import reference_json_path, scene_xml_path, target_config_path
from src.grip.right_hand_racket_grip_env import RightHandRacketGripEnv
from src.grip.validate_right_hand_racket_grip import (
    acceptance_thresholds,
    json_safe,
    pass_checks,
    perturb_and_recover,
    racket_pose,
    rotation_angle_deg,
    safe_env_step,
)


def evaluate(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    episodes: int = 3,
    steps: int = 1000,
) -> dict[str, Any]:
    """Run a deterministic zero-action reference-hold evaluation."""
    episode_count = _positive_int(episodes, "episodes")
    step_count = _positive_int(steps, "steps")
    thresholds = acceptance_thresholds(targets)
    env = RightHandRacketGripEnv(xml, targets, reference)
    zero_action = np.zeros(env.action_size, dtype=float)

    episode_metrics: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    all_site_errors: list[float] = []
    all_contacts: list[int] = []
    all_raw_contacts: list[int] = []
    all_translation_drifts: list[float] = []
    all_orientation_drifts: list[float] = []
    all_recovery_site_errors: list[float] = []
    all_recovery_orientation_drifts: list[float] = []
    reward_terms_by_name: dict[str, list[float]] = {}
    finite = True
    failure_error = None
    steps_executed = 0
    final_info: dict[str, Any] | None = None
    final_checks: dict[str, bool] | None = None

    for episode in range(episode_count):
        obs, info = env.reset()
        start_pos, start_rot = racket_pose(env)
        finite = finite and _array_finite(obs) and _info_finite(info)
        rewards: list[float] = []
        site_errors: list[float] = [float(info["mean_site_error_m"])]
        contacts: list[int] = [int(info["contact_count"])]
        raw_contacts: list[int] = [int(info["raw_contact_count"])]
        truncated = False
        terminated = False

        for _ in range(step_count):
            step_result = safe_env_step(env, zero_action)
            if not step_result["ok"]:
                finite = False
                failure_error = failure_error or step_result["error"]
                break
            obs = step_result["obs"]
            reward = step_result["reward"]
            terminated = step_result["terminated"]
            truncated = step_result["truncated"]
            info = step_result["info"]
            reward_float = float(reward)
            rewards.append(reward_float)
            site_errors.append(float(info["mean_site_error_m"]))
            contacts.append(int(info["contact_count"]))
            raw_contacts.append(int(info["raw_contact_count"]))
            all_rewards.append(reward_float)
            all_site_errors.append(float(info["mean_site_error_m"]))
            all_contacts.append(int(info["contact_count"]))
            all_raw_contacts.append(int(info["raw_contact_count"]))
            for name, value in info.get("reward_terms", {}).items():
                reward_terms_by_name.setdefault(name, []).append(float(value))
            finite = finite and _array_finite(obs) and math.isfinite(reward_float) and _info_finite(info)
            steps_executed += 1
            if terminated or truncated:
                break

        final_info = info
        end_pos, end_rot = racket_pose(env)
        translation_drift = float(np.linalg.norm(end_pos - start_pos))
        orientation_drift = rotation_angle_deg(end_rot @ start_rot.T)
        recovery_metrics = (
            perturb_and_recover(env, zero_action, thresholds)
            if finite
            else {
                "recovery_mean_site_error_m": None,
                "recovery_orientation_drift_deg": None,
                "perturb_steps_executed": 0,
                "recovery_steps_executed": 0,
                "finite": False,
                "error": None,
            }
        )
        if recovery_metrics["finite"]:
            all_recovery_site_errors.append(float(recovery_metrics["recovery_mean_site_error_m"]))
            all_recovery_orientation_drifts.append(float(recovery_metrics["recovery_orientation_drift_deg"]))
        else:
            finite = False
            failure_error = failure_error or recovery_metrics.get("error")
        all_translation_drifts.append(translation_drift)
        all_orientation_drifts.append(orientation_drift)
        final_checks = pass_checks(
            float(info["mean_site_error_m"]),
            translation_drift,
            orientation_drift,
            int(info["contact_count"]),
            finite,
            thresholds,
            recovery_metrics["recovery_mean_site_error_m"],
            recovery_metrics["recovery_orientation_drift_deg"],
        )
        episode_metrics.append(
            {
                "episode": episode,
                "steps_executed": len(rewards),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "mean_reward": _mean_or_zero(rewards),
                "mean_site_error_m": _mean_or_zero(site_errors),
                "contact_count": int(contacts[-1]),
                "raw_contact_count": int(raw_contacts[-1]),
                "max_contact_count": int(max(contacts, default=0)),
                "translation_drift_m": translation_drift,
                "orientation_drift_deg": orientation_drift,
                "recovery_mean_site_error_m": recovery_metrics["recovery_mean_site_error_m"],
                "recovery_orientation_drift_deg": recovery_metrics["recovery_orientation_drift_deg"],
                "finite": bool(finite),
            }
        )

    final_info = final_info or {"site_errors_m": {}, "contact_count": 0, "raw_contact_count": 0, "mean_site_error_m": 0.0}
    recovery_mean_site_error = _mean_or_none(all_recovery_site_errors)
    recovery_orientation_drift = _mean_or_none(all_recovery_orientation_drifts)
    final_checks = final_checks or pass_checks(
        float(final_info["mean_site_error_m"]),
        _mean_or_zero(all_translation_drifts),
        _mean_or_zero(all_orientation_drifts),
        int(final_info["contact_count"]),
        finite,
        thresholds,
        recovery_mean_site_error,
        recovery_orientation_drift,
    )
    return {
        "mode": "zero_action_reference_hold",
        "xml": str(Path(xml)),
        "targets": str(Path(targets)),
        "reference": str(Path(reference)),
        "episodes": episode_count,
        "steps_requested": step_count,
        "steps_executed": int(steps_executed),
        "mean_reward": _mean_or_zero(all_rewards),
        "mean_site_error_m": _mean_or_zero(all_site_errors),
        "contact_count": int(round(_mean_or_zero(all_contacts))),
        "raw_contact_count": int(round(_mean_or_zero(all_raw_contacts))),
        "max_contact_count": int(max(all_contacts, default=0)),
        "reward_terms_mean": {
            name: _mean_or_zero(values)
            for name, values in sorted(reward_terms_by_name.items())
        },
        "site_errors_m": _float_dict(final_info["site_errors_m"]),
        "translation_drift_m": _mean_or_zero(all_translation_drifts),
        "orientation_drift_deg": _mean_or_zero(all_orientation_drifts),
        "recovery_mean_site_error_m": recovery_mean_site_error,
        "recovery_orientation_drift_deg": recovery_orientation_drift,
        "thresholds": thresholds,
        "pass": final_checks,
        "acceptance_pass": bool(all(final_checks.values())),
        "finite": bool(finite),
        "failure_error": failure_error,
        "episodes_detail": episode_metrics,
    }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def _array_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def _info_finite(info: dict[str, Any]) -> bool:
    for key in ("mean_site_error_m", "contact_count", "raw_contact_count"):
        value = info.get(key)
        if isinstance(value, int):
            continue
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return False
    site_errors = info.get("site_errors_m", {})
    if not isinstance(site_errors, dict):
        return False
    return all(isinstance(value, int | float) and math.isfinite(float(value)) for value in site_errors.values())


def _mean_or_zero(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the right-hand racket grip environment.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--reference", type=Path, default=reference_json_path(), help="Grip reference JSON path.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of deterministic evaluation episodes.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum zero-action steps per episode.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = evaluate(args.xml, args.targets, args.reference, episodes=args.episodes, steps=args.steps)
    print(json.dumps(json_safe(metrics), allow_nan=False, indent=2, sort_keys=True))
    return 0 if metrics["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
