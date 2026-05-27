from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.build_overall_environment import READY_KEYFRAME, build_overall_scene
from environment.overall_environment.src.paths import default_overall_scene_path


class OverallBadmintonEnvironment:
    """Reset-only MuJoCo wrapper for the composed badminton scene."""

    def __init__(self, xml: str | Path = default_overall_scene_path()) -> None:
        self.xml_path = Path(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.keyframe_name = READY_KEYFRAME
        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
        if self.keyframe_id < 0:
            raise ValueError(f"missing keyframe {self.keyframe_name!r} in {self.xml_path}")

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def step(self, ctrl: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if ctrl is not None:
            ctrl_array = np.asarray(ctrl, dtype=float)
            if ctrl_array.shape != (self.model.nu,):
                raise ValueError(f"ctrl must have shape ({self.model.nu},), got {ctrl_array.shape}")
            self.data.ctrl[:] = ctrl_array
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


def launch_viewer(env: OverallBadmintonEnvironment, *, simulate: bool = False) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
        while viewer.is_running():
            if simulate:
                mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the overall badminton environment.")
    parser.add_argument("--xml", type=Path, default=default_overall_scene_path(), help="Overall scene XML path.")
    parser.add_argument("--build-if-missing", action="store_true", help="Generate the default XML when absent.")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer window after reset.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance physics while the viewer is open. By default the viewer is static.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if args.build_if_missing and not args.xml.exists():
        build_overall_scene(args.xml)
    env = OverallBadmintonEnvironment(args.xml)
    obs, info = env.reset()
    print(json.dumps({"obs_size": int(obs.size), **info}, indent=2, sort_keys=True))
    if args.viewer:
        launch_viewer(env, simulate=args.simulate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
