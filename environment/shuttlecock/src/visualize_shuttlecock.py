#!/usr/bin/env python3
"""Open the MuJoCo shuttlecock asset in an interactive viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "assets" / "shuttlecock_mujoco.xml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="MuJoCo XML file to visualize.",
    )
    parser.add_argument(
        "--camera",
        default="shuttle_closeup",
        help="Camera name to select when the viewer starts.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to keep the viewer open. Use 0 for interactive until closed.",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="Slowly spin the shuttlecock around its vertical axis for inspection.",
    )
    return parser


def _require_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - depends on local install.
        raise SystemExit(
            "MuJoCo Python viewer is not available. Install it with `pip install mujoco` "
            "inside the environment where you want to visualize the shuttlecock."
        ) from exc
    return mujoco


def _select_camera(mujoco, model, viewer, camera_name: str) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
            for idx in range(model.ncam)
        ]
        raise SystemExit(f"Camera not found: {camera_name!r}. Available cameras: {names}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def _set_slow_spin(model, data, body_name: str = "shuttle", yaw_rate_rad_s: float = 0.6) -> None:
    joint_id = model.body_jntadr[mujoco_body_id(model, body_name)]
    dof_start = model.jnt_dofadr[joint_id]
    data.qvel[dof_start + 5] = yaw_rate_rad_s


def mujoco_body_id(model, body_name: str) -> int:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name!r}")
    return int(body_id)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mujoco = _require_mujoco()

    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Put the shuttle in a stable inspection pose above the floor.
    data.qpos[:7] = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
    if args.spin:
        _set_slow_spin(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _select_camera(mujoco, model, viewer, args.camera)
        start = time.monotonic()
        while viewer.is_running():
            if args.duration > 0.0 and time.monotonic() - start >= args.duration:
                break
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
