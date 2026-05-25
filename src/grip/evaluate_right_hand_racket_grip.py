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


def evaluate(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    *,
    episodes: int = 3,
    steps: int = 1000,
) -> dict[str, Any]:
    """Run a deterministic zero-action reference-hold evaluation."""
    episode_count = _positive_int(episodes, "episodes")
    step_count = _positive_int(steps, "steps")
    env = RightHandRacketGripEnv(xml, targets, reference)
    zero_action = np.zeros(env.action_size, dtype=float)

    episode_metrics: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    all_site_errors: list[float] = []
    all_contacts: list[int] = []
    finite = True
    steps_executed = 0

    for episode in range(episode_count):
        obs, info = env.reset()
        finite = finite and _array_finite(obs) and _info_finite(info)
        rewards: list[float] = []
        site_errors: list[float] = [float(info["mean_site_error_m"])]
        contacts: list[int] = [int(info["contact_count"])]
        truncated = False
        terminated = False

        for _ in range(step_count):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            reward_float = float(reward)
            rewards.append(reward_float)
            site_errors.append(float(info["mean_site_error_m"]))
            contacts.append(int(info["contact_count"]))
            all_rewards.append(reward_float)
            all_site_errors.append(float(info["mean_site_error_m"]))
            all_contacts.append(int(info["contact_count"]))
            finite = finite and _array_finite(obs) and math.isfinite(reward_float) and _info_finite(info)
            steps_executed += 1
            if terminated or truncated:
                break

        episode_metrics.append(
            {
                "episode": episode,
                "steps_executed": len(rewards),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "mean_reward": _mean_or_zero(rewards),
                "mean_site_error_m": _mean_or_zero(site_errors),
                "contact_count": int(contacts[-1]),
                "max_contact_count": int(max(contacts, default=0)),
                "finite": bool(finite),
            }
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
        "max_contact_count": int(max(all_contacts, default=0)),
        "finite": bool(finite),
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
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
