from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.paths import reference_json_path, scene_xml_path, target_config_path
from src.grip.right_hand_racket_grip_env import RightHandRacketGripEnv
from src.grip.target_config import load_grip_target_config


def validate_grip(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    *,
    steps: int = 200,
) -> dict[str, Any]:
    """Validate reference-hold grip stability against target-config thresholds."""
    step_count = _positive_int(steps, "steps")
    env = RightHandRacketGripEnv(xml, targets, reference)
    zero_action = np.zeros(env.action_size, dtype=float)
    obs, info = env.reset()
    start_pos, start_rot = _racket_pose(env)

    finite = _array_finite(obs) and _info_finite(info)
    last_info = info
    terminated = False
    truncated = False
    steps_executed = 0

    for _ in range(step_count):
        obs, reward, terminated, truncated, last_info = env.step(zero_action)
        finite = finite and _array_finite(obs) and math.isfinite(float(reward)) and _info_finite(last_info)
        steps_executed += 1
        if terminated or truncated:
            break

    end_pos, end_rot = _racket_pose(env)
    translation_drift = float(np.linalg.norm(end_pos - start_pos))
    orientation_drift = _rotation_angle_deg(end_rot @ start_rot.T)
    mean_site_error = float(last_info["mean_site_error_m"])
    contact_count = int(last_info["contact_count"])
    thresholds = _acceptance_thresholds(targets)
    pass_checks = _pass_checks(
        mean_site_error,
        translation_drift,
        orientation_drift,
        contact_count,
        finite,
        thresholds,
    )

    return {
        "mode": "zero_action_reference_hold",
        "xml": str(Path(xml)),
        "targets": str(Path(targets)),
        "reference": str(Path(reference)),
        "steps_requested": step_count,
        "steps_executed": int(steps_executed),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "mean_site_error_m": mean_site_error,
        "translation_drift_m": translation_drift,
        "orientation_drift_deg": orientation_drift,
        "contact_count": contact_count,
        "finite": bool(finite),
        "thresholds": thresholds,
        "pass": pass_checks,
        "acceptance_pass": bool(all(pass_checks.values())),
    }


def _racket_pose(env: RightHandRacketGripEnv) -> tuple[np.ndarray, np.ndarray]:
    if env.model_map.racket_body is None:
        raise ValueError("missing racket body in model map")
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model_map.racket_body)
    if body_id < 0:
        raise ValueError(f"missing racket body {env.model_map.racket_body!r}")
    return (
        np.array(env.data.xpos[body_id], dtype=float),
        np.array(env.data.xmat[body_id], dtype=float).reshape(3, 3),
    )


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    cosine = (float(np.trace(rotation_matrix)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def _acceptance_thresholds(targets: str | Path) -> dict[str, float | int | None]:
    raw_acceptance = load_grip_target_config(targets).raw.get("training_acceptance")
    if not isinstance(raw_acceptance, dict):
        raw_acceptance = {}
    return {
        "max_mean_site_error_m": _optional_float(raw_acceptance.get("max_mean_site_error_m")),
        "max_racket_translation_drift_m_2s": _optional_float(
            raw_acceptance.get("max_racket_translation_drift_m_2s")
        ),
        "max_racket_orientation_drift_deg_2s": _optional_float(
            raw_acceptance.get("max_racket_orientation_drift_deg_2s")
        ),
        "min_handle_contacts": _optional_int(raw_acceptance.get("min_handle_contacts")),
    }


def _pass_checks(
    mean_site_error: float,
    translation_drift: float,
    orientation_drift: float,
    contact_count: int,
    finite: bool,
    thresholds: dict[str, float | int | None],
) -> dict[str, bool]:
    return {
        "mean_site_error_m": _leq_optional(mean_site_error, thresholds["max_mean_site_error_m"]),
        "translation_drift_m": _leq_optional(
            translation_drift,
            thresholds["max_racket_translation_drift_m_2s"],
        ),
        "orientation_drift_deg": _leq_optional(
            orientation_drift,
            thresholds["max_racket_orientation_drift_deg_2s"],
        ),
        "contact_count": _geq_optional(contact_count, thresholds["min_handle_contacts"]),
        "finite": bool(finite),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"acceptance threshold must be finite, got {value!r}")
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"acceptance threshold must be an integer, got {value!r}")
    number = int(value)
    if number != value:
        raise ValueError(f"acceptance threshold must be an integer, got {value!r}")
    return number


def _leq_optional(value: float, threshold: float | int | None) -> bool:
    if threshold is None:
        return True
    return bool(value <= float(threshold))


def _geq_optional(value: int, threshold: float | int | None) -> bool:
    if threshold is None:
        return True
    return bool(value >= int(threshold))


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


def validation_exit_code(metrics: dict[str, Any], *, strict: bool = False) -> int:
    if metrics.get("finite") is not True:
        return 1
    if strict and metrics.get("acceptance_pass") is not True:
        return 2
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate right-hand racket grip reference-hold stability.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--reference", type=Path, default=reference_json_path(), help="Grip reference JSON path.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum zero-action validation steps.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any configured acceptance threshold fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = validate_grip(args.xml, args.targets, args.reference, steps=args.steps)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return validation_exit_code(metrics, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
