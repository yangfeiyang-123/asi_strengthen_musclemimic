from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.grip_objectives import mean_site_error, weighted_site_target_residuals
from src.grip.hand_racket_model_map import HandRacketModelMap, load_model_map
from src.grip.paths import reference_json_path, scene_xml_path, target_config_path
from src.grip.target_config import GripTargetConfig, load_grip_target_config

_REGULARIZATION_WEIGHT = 0.02
_IK_MEAN_THRESHOLD_M = 0.03
_TRANSLATION_BOUND_RADIUS_M = 0.5
_ROTVEC_BOUND_RAD = np.pi


def solve_reference(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    out: str | Path = reference_json_path(),
    *,
    max_nfev: int = 200,
) -> dict[str, Any]:
    xml_path = Path(xml)
    targets_path = Path(targets)
    out_path = Path(out)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model_map = load_model_map(model)
    if not model_map.ok:
        raise ValueError(f"unresolved MuJoCo model names: {model_map.missing}")

    target_config = load_grip_target_config(targets_path)
    initial_qpos, keyframe_name = _initial_qpos(model)
    qpos = initial_qpos.copy()
    qvel = np.zeros(model.nv, dtype=float)
    qpos_indices, lower, upper = right_hand_qpos_indices_and_bounds(model, model_map.right_hand_joint_names)
    racket_qpos_adr = racket_freejoint_qpos_address(model, model_map)

    local_target_sites = racket_local_target_positions(target_config)
    initial_hand_sites = hand_site_positions(model, data, qpos, model_map)
    initial_racket_translation = _centroid_aligned_racket_translation(initial_hand_sites, local_target_sites)
    initial_rotvec = np.zeros(3, dtype=float)
    initial_hand_values = qpos[qpos_indices].copy()
    initial_values = np.concatenate([initial_hand_values, initial_racket_translation, initial_rotvec])
    variable_lower = np.concatenate(
        [
            lower,
            initial_racket_translation - _TRANSLATION_BOUND_RADIUS_M,
            np.full(3, -_ROTVEC_BOUND_RAD, dtype=float),
        ],
    )
    variable_upper = np.concatenate(
        [
            upper,
            initial_racket_translation + _TRANSLATION_BOUND_RADIUS_M,
            np.full(3, _ROTVEC_BOUND_RAD, dtype=float),
        ],
    )
    weights = {name: target_config.target_weight(name) for name in local_target_sites}

    def residual(values: np.ndarray) -> np.ndarray:
        _apply_optimization_values(qpos, qpos_indices, racket_qpos_adr, values)
        current_sites = hand_site_positions(model, data, qpos, model_map)
        target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
        site_residuals = weighted_site_target_residuals(current_sites, target_sites, weights)
        regularization = (values[: len(qpos_indices)] - initial_hand_values) * _REGULARIZATION_WEIGHT
        return np.concatenate([site_residuals, regularization])

    result = least_squares(
        residual,
        initial_values,
        bounds=(variable_lower, variable_upper),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    _apply_optimization_values(qpos, qpos_indices, racket_qpos_adr, result.x)
    current_sites = hand_site_positions(model, data, qpos, model_map)
    target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
    site_errors = {
        name: float(np.linalg.norm(current_sites[name] - target_sites[name]))
        for name in sorted(target_sites)
    }
    weighted_residual = weighted_site_target_residuals(current_sites, target_sites, weights)
    regularization = (result.x[: len(qpos_indices)] - initial_hand_values) * _REGULARIZATION_WEIGHT
    mean_error = mean_site_error(current_sites, target_sites)
    max_error = max(site_errors.values(), default=0.0)
    max_error_site = max(site_errors, key=site_errors.get) if site_errors else None
    training_mean_threshold = training_mean_threshold_m(target_config)
    meets_ik_mean_threshold = mean_error_meets_threshold(mean_error, _IK_MEAN_THRESHOLD_M)
    meets_training_mean_threshold = mean_error_meets_threshold(mean_error, training_mean_threshold)
    notes = _quality_notes(
        mean_error,
        max_error,
        max_error_site,
        site_errors.get("palm"),
        meets_ik_mean_threshold,
        training_mean_threshold,
        meets_training_mean_threshold,
    )

    output = {
        "xml": str(xml_path),
        "keyframe_name": keyframe_name,
        "qpos": _float_list(qpos),
        "qvel": _float_list(qvel),
        "racket_freejoint_qpos": racket_freejoint_qpos(model, qpos, model_map),
        "right_hand_joint_names": list(model_map.right_hand_joint_names),
        "site_errors_m": site_errors,
        "mean_site_error_m": mean_error,
        "max_site_error_m": max_error,
        "meets_ik_mean_threshold": meets_ik_mean_threshold,
        "meets_training_mean_threshold": meets_training_mean_threshold,
        "objective_breakdown": {
            "mean_site_error_m": mean_error,
            "max_site_error_m": max_error,
            "max_site_error_name": max_error_site,
            "ik_mean_threshold_m": _IK_MEAN_THRESHOLD_M,
            "training_mean_threshold_m": training_mean_threshold,
            "weighted_site_residual_norm": float(np.linalg.norm(weighted_residual)),
            "regularization_norm": float(np.linalg.norm(regularization)),
            "least_squares_cost": float(result.cost),
            "nfev": int(result.nfev),
            "optimizer_success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
        },
        "notes": notes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "out": str(out_path),
        "mean_site_error_m": mean_error,
        "max_site_error_m": max_error,
        "meets_ik_mean_threshold": meets_ik_mean_threshold,
        "meets_training_mean_threshold": meets_training_mean_threshold,
        "nfev": int(result.nfev),
        "optimizer_success": bool(result.success),
        "cost": float(result.cost),
    }


def _initial_qpos(model: mujoco.MjModel) -> tuple[np.ndarray, str | None]:
    if model.nkey > 0:
        keyframe_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0)
        return np.array(model.key_qpos[0], dtype=float), keyframe_name
    return np.array(model.qpos0, dtype=float), None


def right_hand_qpos_indices_and_bounds(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing joint {joint_name!r}")
        width = _joint_qpos_width(model, joint_id)
        if width != 1:
            raise ValueError(f"right-hand joint {joint_name!r} has qpos width {width}; expected scalar joint")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        indices.append(qpos_adr)
        if bool(model.jnt_limited[joint_id]):
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
        else:
            lower.append(-np.inf)
            upper.append(np.inf)

    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate right-hand qpos indices resolved from joints: {indices}")
    return np.array(indices, dtype=int), np.array(lower, dtype=float), np.array(upper, dtype=float)


def hand_site_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    model_map: HandRacketModelMap,
) -> dict[str, np.ndarray]:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    positions: dict[str, np.ndarray] = {}
    for logical_name, site_name in model_map.hand_sites.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"missing hand site {site_name!r}")
        positions[logical_name] = np.array(data.site_xpos[site_id], dtype=float)
    return positions


def racket_local_target_positions(target_config: GripTargetConfig) -> dict[str, np.ndarray]:
    return {
        name: target_config.target_xyz(name)
        for name in sorted(target_config.target_points_racket_local)
    }


def racket_local_targets_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    target_config: GripTargetConfig,
    model_map: HandRacketModelMap,
) -> dict[str, np.ndarray]:
    if model_map.racket_body is None:
        raise ValueError("missing racket body in model map")

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, model_map.racket_body)
    if body_id < 0:
        raise ValueError(f"missing racket body {model_map.racket_body!r}")
    racket_pos = np.array(data.xpos[body_id], dtype=float)
    racket_xmat = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)

    return {
        name: racket_pos + racket_xmat @ target_config.target_xyz(name)
        for name in sorted(target_config.target_points_racket_local)
    }


def racket_freejoint_qpos_address(model: mujoco.MjModel, model_map: HandRacketModelMap) -> int:
    if model_map.racket_freejoint is None:
        raise ValueError("missing racket freejoint in model map")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_map.racket_freejoint)
    if joint_id < 0:
        raise ValueError(f"missing racket freejoint {model_map.racket_freejoint!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"racket joint {model_map.racket_freejoint!r} is not a freejoint")
    return int(model.jnt_qposadr[joint_id])


def racket_freejoint_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    model_map: HandRacketModelMap,
) -> list[float] | None:
    if model_map.racket_freejoint is None:
        return None
    try:
        qpos_adr = racket_freejoint_qpos_address(model, model_map)
    except ValueError:
        return None
    return _float_list(qpos[qpos_adr : qpos_adr + 7])


def training_mean_threshold_m(target_config: GripTargetConfig) -> float | None:
    training_acceptance = target_config.raw.get("training_acceptance")
    if not isinstance(training_acceptance, dict):
        return None
    threshold = training_acceptance.get("max_mean_site_error_m")
    if threshold is None:
        return None
    return float(threshold)


def mean_error_meets_threshold(mean_error: float, threshold: float | None) -> bool | None:
    if threshold is None:
        return None
    return mean_error <= threshold


def quality_exit_code(result: dict[str, Any], *, allow_poor_reference: bool) -> int:
    if allow_poor_reference:
        return 0
    return 0 if result.get("meets_ik_mean_threshold") is True else 2


def _apply_optimization_values(
    qpos: np.ndarray,
    hand_qpos_indices: np.ndarray,
    racket_qpos_adr: int,
    values: np.ndarray,
) -> None:
    hand_count = len(hand_qpos_indices)
    qpos[hand_qpos_indices] = values[:hand_count]
    qpos[racket_qpos_adr : racket_qpos_adr + 3] = values[hand_count : hand_count + 3]
    qpos[racket_qpos_adr + 3 : racket_qpos_adr + 7] = _quat_wxyz_from_rotvec(values[hand_count + 3 : hand_count + 6])


def _centroid_aligned_racket_translation(
    hand_sites: dict[str, np.ndarray],
    local_target_sites: dict[str, np.ndarray],
) -> np.ndarray:
    names = sorted(local_target_sites)
    hand_centroid = np.mean([hand_sites[name] for name in names], axis=0)
    local_target_centroid = np.mean([local_target_sites[name] for name in names], axis=0)
    return np.asarray(hand_centroid - local_target_centroid, dtype=float)


def _quat_wxyz_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _quality_notes(
    mean_error: float,
    max_error: float,
    max_error_site: str | None,
    palm_error: float | None,
    meets_ik_mean_threshold: bool,
    training_mean_threshold: float | None,
    meets_training_mean_threshold: bool | None,
) -> list[str]:
    ik_status = "met" if meets_ik_mean_threshold else "not met"
    notes = [
        "Static reference optimized right-hand qpos and racket freejoint translation/orientation.",
        f"IK mean site error threshold {_IK_MEAN_THRESHOLD_M:.3f} m is {ik_status}; mean error is {mean_error:.6f} m.",
        f"Max site error is {max_error:.6f} m at {max_error_site or '<none>'}.",
        f"Regularization weight to initial right-hand pose: {_REGULARIZATION_WEIGHT}.",
    ]
    if palm_error is not None:
        notes.append(f"Palm site error is {palm_error:.6f} m.")
    if training_mean_threshold is None:
        notes.append("Training mean site error threshold is unavailable in target config.")
    else:
        training_status = "met" if meets_training_mean_threshold else "not met"
        notes.append(
            f"Training mean site error threshold {training_mean_threshold:.3f} m is {training_status}.",
        )
    return notes


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    if joint_type in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
        return 1
    raise ValueError(f"unsupported MuJoCo joint type {joint_type} for joint id {joint_id}")


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a static right-hand racket grip reference.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--out", type=Path, default=reference_json_path(), help="Output reference JSON path.")
    parser.add_argument("--max-nfev", type=int, default=200, help="Maximum least-squares function evaluations.")
    parser.add_argument(
        "--allow-poor-reference",
        action="store_true",
        help="Return exit code 0 even when the solved reference misses the IK mean-error threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = solve_reference(args.xml, args.targets, args.out, max_nfev=args.max_nfev)
    print(json.dumps(result, indent=2, sort_keys=True))
    return quality_exit_code(result, allow_poor_reference=args.allow_poor_reference)


if __name__ == "__main__":
    raise SystemExit(main())
