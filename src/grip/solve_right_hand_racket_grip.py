from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.grip_objectives import mean_site_error, weighted_site_target_residuals
from src.grip.hand_racket_model_map import HandRacketModelMap, load_model_map
from src.grip.paths import reference_json_path, scene_xml_path, target_config_path
from src.grip.target_config import GripTargetConfig, load_grip_target_config

_REGULARIZATION_WEIGHT = 0.05


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

    target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
    weights = {name: target_config.target_weight(name) for name in target_sites}
    initial_values = qpos[qpos_indices].copy()

    def residual(values: np.ndarray) -> np.ndarray:
        qpos[qpos_indices] = values
        current_sites = hand_site_positions(model, data, qpos, model_map)
        site_residuals = weighted_site_target_residuals(current_sites, target_sites, weights)
        regularization = (values - initial_values) * _REGULARIZATION_WEIGHT
        return np.concatenate([site_residuals, regularization])

    result = least_squares(
        residual,
        initial_values,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    qpos[qpos_indices] = result.x
    current_sites = hand_site_positions(model, data, qpos, model_map)
    site_errors = {
        name: float(np.linalg.norm(current_sites[name] - target_sites[name]))
        for name in sorted(target_sites)
    }
    weighted_residual = weighted_site_target_residuals(current_sites, target_sites, weights)
    regularization = (result.x - initial_values) * _REGULARIZATION_WEIGHT
    mean_error = mean_site_error(current_sites, target_sites)
    max_error = max(site_errors.values(), default=0.0)

    output = {
        "xml": str(xml_path),
        "keyframe_name": keyframe_name,
        "qpos": _float_list(qpos),
        "qvel": _float_list(qvel),
        "racket_freejoint_qpos": racket_freejoint_qpos(model, qpos, model_map),
        "right_hand_joint_names": list(model_map.right_hand_joint_names),
        "site_errors_m": site_errors,
        "objective_breakdown": {
            "mean_site_error_m": mean_error,
            "max_site_error_m": max_error,
            "weighted_site_residual_norm": float(np.linalg.norm(weighted_residual)),
            "regularization_norm": float(np.linalg.norm(regularization)),
            "least_squares_cost": float(result.cost),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
        },
        "notes": [
            "Static reference optimized right-hand qpos only.",
            "Racket pose and non-right-hand joints remain at the initial keyframe/model pose.",
            f"Regularization weight to initial right-hand pose: {_REGULARIZATION_WEIGHT}.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "out": str(out_path),
        "mean_site_error_m": mean_error,
        "max_site_error_m": max_error,
        "nfev": int(result.nfev),
        "success": bool(result.success),
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


def racket_freejoint_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    model_map: HandRacketModelMap,
) -> list[float] | None:
    if model_map.racket_freejoint is None:
        return None
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_map.racket_freejoint)
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return None
    qpos_adr = int(model.jnt_qposadr[joint_id])
    return _float_list(qpos[qpos_adr : qpos_adr + 7])


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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = solve_reference(args.xml, args.targets, args.out, max_nfev=args.max_nfev)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
