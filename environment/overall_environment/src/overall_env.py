from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

READY_KEYFRAME = "overall_ready"
OVERALL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = OVERALL_ROOT / "assets" / "overall_badminton_scene.xml"


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))


class OverallBadmintonEnvironment:
    """Reset-only MuJoCo wrapper for the composed badminton scene."""

    def __init__(self, xml: str | Path = DEFAULT_XML_PATH) -> None:
        self.xml_path = Path(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self._servo_qpos = np.array(self.model.qpos0, dtype=float)
        self.keyframe_name = READY_KEYFRAME
        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
        if self.keyframe_id < 0:
            raise ValueError(f"missing keyframe {self.keyframe_name!r} in {self.xml_path}")

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        self._servo_qpos = np.array(self.data.qpos, dtype=float)
        return self._observation(), self._info()

    def step(
        self,
        ctrl: np.ndarray | None = None,
        *,
        pose_servo: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if ctrl is not None:
            ctrl_array = np.asarray(ctrl, dtype=float)
            if ctrl_array.shape != (self.model.nu,):
                raise ValueError(f"ctrl must have shape ({self.model.nu},), got {ctrl_array.shape}")
            self.data.ctrl[:] = ctrl_array
        self.data.qfrc_applied[:] = 0.0
        if pose_servo:
            _apply_pose_servo(self.model, self.data, self._servo_qpos)
        mujoco.mj_step(self.model, self.data)
        return self._observation(), self._info()

    def _observation(self) -> np.ndarray:
        return np.concatenate([np.array(self.data.qpos, dtype=float), np.array(self.data.qvel, dtype=float)])

    def _info(self) -> dict[str, Any]:
        return {
            "keyframe": self.keyframe_name,
            "has_court": _object_exists(self.model, mujoco.mjtObj.mjOBJ_BODY, "overall_court_static"),
            "has_racket": _object_exists(self.model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket"),
            "has_shuttlecock": _object_exists(self.model, mujoco.mjtObj.mjOBJ_BODY, "overall_shuttle"),
            "shuttle_cork_height_m": _site_height(
                self.model,
                self.data,
                "overall_cork_contact_site",
            ),
        }


def _object_exists(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> bool:
    return mujoco.mj_name2id(model, obj_type, name) >= 0


def _site_height(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"missing site {name!r}")
    return float(data.site_xpos[site_id, 2])


def _apply_pose_servo(model: mujoco.MjModel, data: mujoco.MjData, reference_qpos: np.ndarray) -> None:
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name == "overall_shuttle_free":
            continue
        joint_type = int(model.jnt_type[joint_id])
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            _apply_freejoint_servo(data, reference_qpos, qadr, dadr)
        elif joint_type in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            error = reference_qpos[qadr] - data.qpos[qadr]
            force = 12.0 * error - 1.2 * data.qvel[dadr]
            data.qfrc_applied[dadr] += float(np.clip(force, -4.0, 4.0))


def _apply_freejoint_servo(data: mujoco.MjData, reference_qpos: np.ndarray, qadr: int, dadr: int) -> None:
    position_error = reference_qpos[qadr : qadr + 3] - data.qpos[qadr : qadr + 3]
    force = 140.0 * position_error - 28.0 * data.qvel[dadr : dadr + 3]
    data.qfrc_applied[dadr : dadr + 3] += np.clip(force, -180.0, 180.0)

    q_current_inverse = np.zeros(4)
    q_error = np.zeros(4)
    mujoco.mju_negQuat(q_current_inverse, data.qpos[qadr + 3 : qadr + 7])
    mujoco.mju_mulQuat(q_error, reference_qpos[qadr + 3 : qadr + 7], q_current_inverse)
    if q_error[0] < 0.0:
        q_error *= -1.0
    torque = 36.0 * q_error[1:4] - 7.2 * data.qvel[dadr + 3 : dadr + 6]
    data.qfrc_applied[dadr + 3 : dadr + 6] += np.clip(torque, -40.0, 40.0)


def _configure_viewer_visuals(viewer: Any, *, debug_visuals: bool = False) -> None:
    if debug_visuals:
        viewer.opt.geomgroup[:] = 1
        viewer.opt.sitegroup[:] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
        return

    viewer.opt.geomgroup[:] = 0
    for group_id in (0, 1, 2):
        viewer.opt.geomgroup[group_id] = 1
    viewer.opt.sitegroup[:] = 0
    for flag in (
        mujoco.mjtVisFlag.mjVIS_ACTUATOR,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
        mujoco.mjtVisFlag.mjVIS_JOINT,
        mujoco.mjtVisFlag.mjVIS_SKIN,
        mujoco.mjtVisFlag.mjVIS_TENDON,
        mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
    ):
        viewer.opt.flags[flag] = 0


def launch_viewer(
    env: OverallBadmintonEnvironment,
    *,
    simulate: bool = False,
    pose_servo: bool = True,
    debug_visuals: bool = False,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
        _configure_viewer_visuals(viewer, debug_visuals=debug_visuals)
        while viewer.is_running():
            if simulate:
                env.step(pose_servo=pose_servo)
            viewer.sync()
            time.sleep(0.01)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the overall badminton environment.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH, help="Overall scene XML path.")
    parser.add_argument("--build-if-missing", action="store_true", help="Generate the default XML when absent.")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer window after reset.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance physics while the viewer is open. Uses a pose servo unless --free-simulate is set.",
    )
    parser.add_argument(
        "--free-simulate",
        action="store_true",
        help="Disable the pose servo during --simulate and run raw MuJoCo physics.",
    )
    parser.add_argument(
        "--simulate-steps",
        type=int,
        default=0,
        help="Run this many simulation steps after reset before printing the final smoke-test JSON.",
    )
    parser.add_argument(
        "--debug-visuals",
        action="store_true",
        help="Show all MuJoCo visual groups plus joints, tendons, sites, and actuators.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if args.build_if_missing and not args.xml.exists():
        from environment.overall_environment.src.build_overall_environment import build_overall_scene

        build_overall_scene(args.xml)
    env = OverallBadmintonEnvironment(args.xml)
    obs, info = env.reset()
    for _ in range(args.simulate_steps):
        obs, info = env.step(pose_servo=not args.free_simulate)
    print(json.dumps({"obs_size": int(obs.size), **info}, indent=2, sort_keys=True))
    if args.viewer:
        launch_viewer(
            env,
            simulate=args.simulate,
            pose_servo=not args.free_simulate,
            debug_visuals=args.debug_visuals,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
