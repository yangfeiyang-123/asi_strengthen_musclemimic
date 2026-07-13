"""Render the fingers-enabled racket grip: MyoFullBodyRacket with disable_fingers
=False, right hand posed at the optimized forehand grip, at a mid-swing frame.
Visual proof that the fingers wrap the handle while the racket stays rigidly held.

Run:
  source configs/env.sh
  MUJOCO_GL=egl JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python rl_training_environment/render_grip_fingers.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

from musclemimic.environments.humanoids.myofullbody_racket import (
    MyoFullBodyRacket,
    RACKET_BODY_NAME,
)

import mujoco  # noqa: E402
from loco_mujoco.trajectory import Trajectory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import qpos_from_trajectory_frame  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
W, H = 1100, 850


def _swing_pose(model: mujoco.MjModel) -> np.ndarray:
    """Model-layout qpos at a mid-swing frame of a forehand-clear clip."""
    clips = sorted(
        p for p in glob.glob(
            "datasets/forehandClear_standard/muscle_trajectory/optimized/*.npz"
        ) if not p.endswith("_analysis.npz")
    )
    if not clips:
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        return d.qpos.copy()
    traj = Trajectory.load(clips[0], backend=np)
    n_frames = np.asarray(traj.data.qpos).shape[0]
    frame = int(n_frames * 0.6)
    print(f"pose source: {Path(clips[0]).name}  frame {frame}/{n_frames}")
    # Map by joint name: the clip is retargeted for the bare-hand MyoFullBody,
    # while this model has finger joints inserted mid-qpos (see pose_utils).
    return qpos_from_trajectory_frame(model, traj, frame)


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = MyoFullBodyRacket(disable_fingers=False, no_skybox=False)
    model = env._model
    data = mujoco.MjData(model)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    model.vis.headlight.specular[:] = 0.2

    # body pose from the swing clip (fingerless trajectory -> fingers at default)...
    data.qpos[:] = _swing_pose(model)
    # ...then close the right hand onto the grip pose, exactly as the training
    # init handler does at every reset.
    addrs = env.grip_finger_qpos_addrs
    targets = env.grip_finger_targets
    data.qpos[addrs] = targets
    mujoco.mj_forward(model, data)
    print(f"posed {len(addrs)} right-hand finger joints to the grip pose")

    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thirdmc_r")
    look = np.array(data.xpos[hand], dtype=float)

    views = {
        "grip_closeup": dict(azimuth=110, elevation=-12, distance=0.55, lookat=data.xpos[hand]),
        "grip_closeup_side": dict(azimuth=20, elevation=-8, distance=0.55, lookat=data.xpos[hand]),
        "swing_3q": dict(azimuth=135, elevation=-12, distance=3.0, lookat=look),
    }

    cam = mujoco.MjvCamera()
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        for name, v in views.items():
            cam.lookat[:] = v["lookat"]
            cam.azimuth, cam.elevation, cam.distance = v["azimuth"], v["elevation"], v["distance"]
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            out = OUT_DIR / f"racket_grip_fingers_{name}.png"
            _save_png(img, out)
            print(f"  {name:18s} -> {out.name}")


def _save_png(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)


if __name__ == "__main__":
    main()
