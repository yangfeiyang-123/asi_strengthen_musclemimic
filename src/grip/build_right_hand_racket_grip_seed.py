from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.grip.grip_seed import joint_shape_metrics
from src.grip.hand_racket_model_map import HandRacketModelMap, load_model_map
from src.grip.paths import grip_seed_json_path, scene_xml_path, target_config_path
from src.grip.solve_right_hand_racket_grip import (
    hand_site_positions,
    handle_max_penetration,
    racket_freejoint_qpos,
    racket_freejoint_qpos_address,
    racket_local_targets_to_world,
    right_hand_qpos_indices_and_bounds,
)
from src.grip.target_config import GripTargetConfig, load_grip_target_config


CURL_PRIORS = {
    "mcp4_flexion_r": 0.35,
    "pm4_flexion_r": 1.05,
    "md4_flexion_r": 0.65,
    "mcp5_flexion_r": 0.30,
    "pm5_flexion_r": 0.85,
    "md5_flexion_r": 0.55,
}


def build_grip_seed(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    out: str | Path = grip_seed_json_path(),
    *,
    initial_reference: str | Path | None = None,
    max_nfev: int = 120,
    render: bool = True,
) -> dict[str, Any]:
    xml_path = Path(xml)
    targets_path = Path(targets)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model_map = load_model_map(model)
    if not model_map.ok:
        raise ValueError(f"unresolved MuJoCo model names: {model_map.missing}")

    target_config = load_grip_target_config(targets_path)
    qpos = _initial_qpos(model, initial_reference)
    qvel = np.zeros(model.nv, dtype=float)
    hand_indices, lower, upper = right_hand_qpos_indices_and_bounds(model, model_map.right_hand_joint_names)
    racket_adr = racket_freejoint_qpos_address(model, model_map)
    initial_values = np.concatenate([qpos[hand_indices], qpos[racket_adr : racket_adr + 7]])

    def residual(values: np.ndarray) -> np.ndarray:
        qpos[hand_indices] = values[: len(hand_indices)]
        qpos[racket_adr : racket_adr + 7] = _normalized_freejoint(values[len(hand_indices) :])
        current_sites = hand_site_positions(model, data, qpos, model_map)
        target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
        site_residuals = np.concatenate(
            [
                (current_sites[name] - target_sites[name]) * target_config.target_weight(name)
                for name in sorted(target_sites)
            ],
        )
        return np.concatenate(
            [
                site_residuals,
                _curl_prior_residuals(model, qpos, weight=0.35),
                _lower_bound_residuals(model, qpos, model_map.right_hand_joint_names, weight=0.08),
                _penetration_residual(model, data, qpos, model_map),
            ],
        )

    variable_lower = np.concatenate([lower, np.full(7, -np.inf, dtype=float)])
    variable_upper = np.concatenate([upper, np.full(7, np.inf, dtype=float)])
    result = least_squares(
        residual,
        initial_values,
        bounds=(variable_lower, variable_upper),
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    qpos[hand_indices] = result.x[: len(hand_indices)]
    qpos[racket_adr : racket_adr + 7] = _normalized_freejoint(result.x[len(hand_indices) :])
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    site_errors = _site_errors(model, data, qpos, target_config, model_map)
    contact_metrics = _contact_metrics(model, data, model_map)
    shape_metrics = joint_shape_metrics(model, qpos, model_map.right_hand_joint_names)
    seed_scene = out_path.parent / "right_hand_racket_grip_seed_scene.xml"
    shutil.copyfile(xml_path, seed_scene)
    _write_keyframe(seed_scene, "right_hand_racket_grip_seed", qpos)

    visualization_paths: list[str] = []
    if render:
        visualization_paths = _render_seed_views(model, data, out_path.parent / "visualization")

    raw = {
        "schema_version": 1,
        "source_xml": str(xml_path),
        "target_config": str(targets_path),
        "qpos": _float_list(qpos),
        "qvel": _float_list(qvel),
        "right_hand_joint_names": list(model_map.right_hand_joint_names),
        "racket_freejoint_name": str(model_map.racket_freejoint),
        "racket_freejoint_qpos": racket_freejoint_qpos(model, qpos, model_map),
        "site_errors_m": site_errors,
        "joint_shape_metrics": shape_metrics,
        "contact_metrics": contact_metrics,
        "visualization_paths": visualization_paths,
        "generation_command": _generation_command(xml_path, targets_path, out_path, initial_reference, max_nfev),
        "objective": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "message": str(result.message),
        },
    }
    out_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = out_path.parent / "right_hand_racket_grip_seed_report.json"
    report = {
        "out": str(out_path),
        **raw["objective"],
        "site_errors_m": site_errors,
        "contact_metrics": contact_metrics,
        "joint_shape_metrics": shape_metrics,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "out": str(out_path),
        "report": str(report_path),
        "seed_scene": str(seed_scene),
        "nfev": int(result.nfev),
    }


def _initial_qpos(model: mujoco.MjModel, initial_reference: str | Path | None) -> np.ndarray:
    if initial_reference is None:
        return np.array(model.qpos0, dtype=float)
    raw = json.loads(Path(initial_reference).read_text(encoding="utf-8"))
    qpos = np.asarray(raw["qpos"], dtype=float)
    if qpos.shape != (model.nq,):
        raise ValueError(f"initial reference qpos must have shape ({model.nq},), got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("initial reference qpos must be finite")
    return qpos.copy()


def _normalized_freejoint(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    quat = result[3:7]
    norm = float(np.linalg.norm(quat))
    if norm == 0.0 or not np.isfinite(norm):
        result[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        result[3:7] = quat / norm
    return result


def _curl_prior_residuals(model: mujoco.MjModel, qpos: np.ndarray, *, weight: float) -> np.ndarray:
    residuals = []
    for joint_name, target in CURL_PRIORS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        residuals.append((float(qpos[adr]) - target) * weight)
    return np.asarray(residuals, dtype=float)


def _lower_bound_residuals(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    *,
    weight: float,
) -> np.ndarray:
    residuals = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        lower = float(model.jnt_range[joint_id, 0])
        residuals.append(max(0.0, 0.03 - (float(qpos[adr]) - lower)) * weight)
    return np.asarray(residuals, dtype=float)


def _penetration_residual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    model_map: HandRacketModelMap,
) -> np.ndarray:
    penetration = handle_max_penetration(model, data, qpos, model_map)
    return np.array([max(0.0, penetration - 0.003) * 8.0], dtype=float)


def _site_errors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    target_config: GripTargetConfig,
    model_map: HandRacketModelMap,
) -> dict[str, float]:
    current_sites = hand_site_positions(model, data, qpos, model_map)
    target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
    return {
        name: float(np.linalg.norm(current_sites[name] - target_sites[name]))
        for name in sorted(target_sites)
    }


def _contact_metrics(model: mujoco.MjModel, data: mujoco.MjData, model_map: HandRacketModelMap) -> dict[str, float | int]:
    handle_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in model_map.handle_geoms
    }
    handle_ids.discard(-1)
    contacts = 0
    max_penetration = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) in handle_ids or int(contact.geom2) in handle_ids:
            contacts += 1
            max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
    return {
        "raw_handle_contacts": contacts,
        "max_handle_penetration_m": max_penetration,
    }


def _write_keyframe(xml_path: Path, key_name: str, qpos: np.ndarray) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.attrib.get("name") == key_name:
            keyframe.remove(key)
    ET.SubElement(
        keyframe,
        "key",
        {"name": key_name, "qpos": " ".join(f"{value:.17g}" for value in qpos)},
    )
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _render_seed_views(model: mujoco.MjModel, data: mujoco.MjData, out_dir: Path) -> list[str]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    paths = []
    palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    if palm_site < 0:
        raise ValueError("missing site 'rh_palm_grip_site'")

    for name, azimuth in (("seed_grip_closeup_front.png", 70), ("seed_grip_closeup_side.png", 20)):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = data.site_xpos[palm_site]
        cam.distance = 0.35
        cam.azimuth = azimuth
        cam.elevation = -8
        renderer.update_scene(data, camera=cam)
        path = out_dir / name
        Image.fromarray(renderer.render()).save(path)
        paths.append(str(path))
    renderer.close()
    return paths


def _generation_command(
    xml: Path,
    targets: Path,
    out: Path,
    initial_reference: str | Path | None,
    max_nfev: int,
) -> list[str]:
    command = [
        "python",
        "-m",
        "src.grip.build_right_hand_racket_grip_seed",
        "--xml",
        str(xml),
        "--targets",
        str(targets),
        "--out",
        str(out),
        "--max-nfev",
        str(max_nfev),
    ]
    if initial_reference is not None:
        command.extend(["--initial-reference", str(initial_reference)])
    return command


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible right-hand racket grip seed artifact.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path())
    parser.add_argument("--targets", type=Path, default=target_config_path())
    parser.add_argument("--out", type=Path, default=grip_seed_json_path())
    parser.add_argument("--initial-reference", type=Path, default=None)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_grip_seed(
        args.xml,
        args.targets,
        args.out,
        initial_reference=args.initial_reference,
        max_nfev=args.max_nfev,
        render=not args.no_render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
