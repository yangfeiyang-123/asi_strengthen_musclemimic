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

FLOAT_ACCEPTANCE_KEYS = (
    "max_mean_site_error_m",
    "max_racket_translation_drift_m_2s",
    "max_racket_orientation_drift_deg_2s",
    "perturb_force_n",
    "perturb_torque_nm",
    "perturb_recovery_s",
    "max_recovery_site_error_m",
    "max_recovery_orientation_error_deg",
)
INT_ACCEPTANCE_KEYS = ("min_handle_contacts",)
DEFAULT_PERTURB_DURATION_S = 0.05
DEFAULT_PERTURB_RECOVERY_S = 0.5
DEFAULT_MAX_HANDLE_PENETRATION_M = 0.002


def validate_grip(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    *,
    steps: int = 200,
) -> dict[str, Any]:
    """Validate reference-hold grip stability against target-config thresholds."""
    step_count = _positive_int(steps, "steps")
    thresholds = acceptance_thresholds(targets)
    env = RightHandRacketGripEnv(xml, targets, reference)
    zero_action = np.zeros(env.action_size, dtype=float)
    obs, info = env.reset()
    start_pos, start_rot = racket_pose(env)

    finite = _array_finite(obs) and _info_finite(info)
    last_info = info
    max_illegal_handle_contact_count = int(info["illegal_handle_contact_count"])
    max_handle_penetration_m = float(info["max_handle_penetration_m"])
    terminated = False
    truncated = False
    steps_executed = 0
    failure_error = None

    for _ in range(step_count):
        step_result = safe_env_step(env, zero_action)
        if not step_result["ok"]:
            finite = False
            failure_error = step_result["error"]
            break
        obs = step_result["obs"]
        reward = step_result["reward"]
        terminated = step_result["terminated"]
        truncated = step_result["truncated"]
        last_info = step_result["info"]
        finite = finite and _array_finite(obs) and math.isfinite(float(reward)) and _info_finite(last_info)
        max_illegal_handle_contact_count = max(
            max_illegal_handle_contact_count,
            int(last_info["illegal_handle_contact_count"]),
        )
        max_handle_penetration_m = max(
            max_handle_penetration_m,
            float(last_info["max_handle_penetration_m"]),
        )
        steps_executed += 1
        if terminated or truncated:
            break

    end_pos, end_rot = racket_pose(env)
    translation_drift = float(np.linalg.norm(end_pos - start_pos))
    orientation_drift = rotation_angle_deg(end_rot @ start_rot.T)
    recovery_metrics = perturb_and_recover(env, zero_action, thresholds) if finite else _failed_recovery_metrics()
    if not recovery_metrics["finite"]:
        finite = False
        failure_error = failure_error or recovery_metrics.get("error")
    mean_site_error = float(last_info["mean_site_error_m"])
    contact_count = int(last_info["contact_count"])
    checks = pass_checks(
        mean_site_error,
        translation_drift,
        orientation_drift,
        contact_count,
        max_illegal_handle_contact_count,
        max_handle_penetration_m,
        finite,
        thresholds,
        recovery_metrics["recovery_mean_site_error_m"],
        recovery_metrics["recovery_orientation_drift_deg"],
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
        "illegal_handle_contact_count": max_illegal_handle_contact_count,
        "max_handle_penetration_m": max_handle_penetration_m,
        "raw_contact_count": int(last_info["raw_contact_count"]),
        "site_errors_m": _float_dict(last_info["site_errors_m"]),
        "recovery_mean_site_error_m": recovery_metrics["recovery_mean_site_error_m"],
        "recovery_orientation_drift_deg": recovery_metrics["recovery_orientation_drift_deg"],
        "perturb_steps_executed": recovery_metrics["perturb_steps_executed"],
        "recovery_steps_executed": recovery_metrics["recovery_steps_executed"],
        "finite": bool(finite),
        "failure_error": failure_error,
        "thresholds": thresholds,
        "pass": checks,
        "acceptance_pass": bool(all(checks.values())),
    }


def racket_pose(env: RightHandRacketGripEnv) -> tuple[np.ndarray, np.ndarray]:
    if env.model_map.racket_body is None:
        raise ValueError("missing racket body in model map")
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model_map.racket_body)
    if body_id < 0:
        raise ValueError(f"missing racket body {env.model_map.racket_body!r}")
    return (
        np.array(env.data.xpos[body_id], dtype=float),
        np.array(env.data.xmat[body_id], dtype=float).reshape(3, 3),
    )


def racket_body_id(env: RightHandRacketGripEnv) -> int:
    if env.model_map.racket_body is None:
        raise ValueError("missing racket body in model map")
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model_map.racket_body)
    if body_id < 0:
        raise ValueError(f"missing racket body {env.model_map.racket_body!r}")
    return int(body_id)


def rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    cosine = (float(np.trace(rotation_matrix)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def acceptance_thresholds(targets: str | Path) -> dict[str, float | int | None]:
    raw_acceptance = load_grip_target_config(targets).raw.get("training_acceptance")
    if not isinstance(raw_acceptance, dict):
        raw_acceptance = {}
    thresholds: dict[str, float | int | None] = {
        key: _optional_nonnegative_float(raw_acceptance.get(key), f"training_acceptance.{key}")
        for key in FLOAT_ACCEPTANCE_KEYS
    }
    thresholds.update(
        {
            key: _optional_nonnegative_int(raw_acceptance.get(key), f"training_acceptance.{key}")
            for key in INT_ACCEPTANCE_KEYS
        }
    )
    return thresholds


def pass_checks(
    mean_site_error: float,
    translation_drift: float,
    orientation_drift: float,
    contact_count: int,
    illegal_handle_contact_count: int,
    max_handle_penetration_m: float,
    finite: bool,
    thresholds: dict[str, float | int | None],
    recovery_mean_site_error: float | None,
    recovery_orientation_drift: float | None,
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
        "illegal_handle_contact_count": illegal_handle_contact_count == 0,
        "max_handle_penetration_m": max_handle_penetration_m <= DEFAULT_MAX_HANDLE_PENETRATION_M,
        "recovery_mean_site_error_m": _leq_optional(
            recovery_mean_site_error,
            thresholds["max_recovery_site_error_m"],
        ),
        "recovery_orientation_drift_deg": _leq_optional(
            recovery_orientation_drift,
            thresholds["max_recovery_orientation_error_deg"],
        ),
        "finite": bool(finite),
    }


def safe_env_step(env: RightHandRacketGripEnv, action: np.ndarray) -> dict[str, Any]:
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except Exception as exc:  # noqa: BLE001 - preserve JSON CLI behavior for simulation failures.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "obs": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


def perturb_and_recover(
    env: RightHandRacketGripEnv,
    zero_action: np.ndarray,
    thresholds: dict[str, float | int | None],
) -> dict[str, Any]:
    body_id = racket_body_id(env)
    start_pos, start_rot = racket_pose(env)
    force_n = _threshold_float_or_default(thresholds, "perturb_force_n", 0.0)
    torque_nm = _threshold_float_or_default(thresholds, "perturb_torque_nm", 0.0)
    recovery_s = _threshold_float_or_default(thresholds, "perturb_recovery_s", DEFAULT_PERTURB_RECOVERY_S)
    control_dt = float(env.model.opt.timestep) * float(env.control_substeps)
    perturb_steps = max(1, int(math.ceil(DEFAULT_PERTURB_DURATION_S / control_dt)))
    recovery_steps = int(math.ceil(recovery_s / control_dt)) if recovery_s > 0.0 else 0
    finite = True
    last_info = env._info()
    error = None
    perturb_steps_executed = 0
    recovery_steps_executed = 0

    env.data.xfrc_applied[body_id, :3] = np.array([force_n, 0.0, 0.0], dtype=float)
    env.data.xfrc_applied[body_id, 3:] = np.array([0.0, 0.0, torque_nm], dtype=float)
    try:
        for _ in range(perturb_steps):
            result = safe_env_step(env, zero_action)
            if not result["ok"]:
                finite = False
                error = result["error"]
                break
            last_info = result["info"]
            finite = finite and _array_finite(result["obs"]) and math.isfinite(result["reward"]) and _info_finite(last_info)
            perturb_steps_executed += 1
    finally:
        env.data.xfrc_applied[body_id, :] = 0.0

    if finite:
        for _ in range(recovery_steps):
            result = safe_env_step(env, zero_action)
            if not result["ok"]:
                finite = False
                error = result["error"]
                break
            last_info = result["info"]
            finite = finite and _array_finite(result["obs"]) and math.isfinite(result["reward"]) and _info_finite(last_info)
            recovery_steps_executed += 1

    _, end_rot = racket_pose(env)
    return {
        "recovery_mean_site_error_m": float(last_info["mean_site_error_m"]) if finite else None,
        "recovery_orientation_drift_deg": rotation_angle_deg(end_rot @ start_rot.T) if finite else None,
        "perturb_steps_executed": int(perturb_steps_executed),
        "recovery_steps_executed": int(recovery_steps_executed),
        "finite": bool(finite),
        "error": error,
    }


def _failed_recovery_metrics() -> dict[str, Any]:
    return {
        "recovery_mean_site_error_m": None,
        "recovery_orientation_drift_deg": None,
        "perturb_steps_executed": 0,
        "recovery_steps_executed": 0,
        "finite": False,
        "error": None,
    }


def _optional_nonnegative_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a JSON number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{context} must be >= 0, got {value!r}")
    return number


def _threshold_float_or_default(
    thresholds: dict[str, float | int | None],
    key: str,
    default: float,
) -> float:
    value = thresholds.get(key)
    return default if value is None else float(value)


def _optional_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer >= 0, got {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"{context} must be an integer >= 0, got {value!r}")
    number = int(value)
    if number < 0:
        raise ValueError(f"{context} must be an integer >= 0, got {value!r}")
    return number


def _leq_optional(value: float | None, threshold: float | int | None) -> bool:
    if threshold is None:
        return True
    if value is None or not math.isfinite(float(value)):
        return False
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
    for key in (
        "mean_site_error_m",
        "v_shape_error",
        "anti_panhandle_error",
        "anti_thumb_grip_error",
        "thumb_index_y_gap_m",
        "v_bisector_theta_deg",
        "palm_theta_deg",
        "thumb_theta_deg",
        "racket_translation_error_m",
        "racket_orientation_error_deg",
        "grip_slip_m",
        "reference_pose_error",
        "joint_limit_margin_cost",
        "contact_count",
        "illegal_handle_contact_count",
        "max_handle_penetration_m",
        "raw_contact_count",
    ):
        value = info.get(key)
        if isinstance(value, int):
            continue
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return False
    site_errors = info.get("site_errors_m", {})
    if not isinstance(site_errors, dict):
        return False
    return all(isinstance(value, int | float) and math.isfinite(float(value)) for value in site_errors.values())


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def validation_exit_code(metrics: dict[str, Any], *, strict: bool = False) -> int:
    if metrics.get("finite") is not True:
        return 1
    if strict and metrics.get("acceptance_pass") is not True:
        return 2
    return 0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
    print(json.dumps(json_safe(metrics), allow_nan=False, indent=2, sort_keys=True))
    return validation_exit_code(metrics, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
