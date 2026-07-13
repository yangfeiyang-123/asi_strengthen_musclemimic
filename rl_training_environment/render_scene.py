"""Render multi-view images of the racket-holding RL training scene.

The scene is MyoFullBodyRacket: the muscle-actuated full-body humanoid rigidly
holding a badminton racket, posed at a mid-swing frame of a forehand-clear
trajectory. Outputs individual views plus a composite grid into this directory.

Run:
  source configs/env.sh
  MUJOCO_GL=egl JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python rl_training_environment/render_scene.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

# Import the environment first so loco_mujoco fully initializes before we touch
# loco_mujoco.trajectory (avoids a partial-init circular import).
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


def _load_swing_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Return a mid-swing qpos frame from a forehand-clear clip (or model default)."""
    clips = sorted(
        p
        for p in glob.glob(
            "datasets/forehandClear_standard/muscle_trajectory/optimized/*.npz"
        )
        if not p.endswith("_analysis.npz")
    )
    if not clips:
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        return d.qpos.copy()
    traj = Trajectory.load(clips[0], backend=np)
    n_frames = np.asarray(traj.data.qpos).shape[0]
    # pick a frame near peak swing (~60% through the clip)
    frame = int(n_frames * 0.6)
    print(f"pose source: {Path(clips[0]).name}  frame {frame}/{n_frames}")
    # Map by joint name (robust to models whose qpos layout extends the clip's).
    return qpos_from_trajectory_frame(model, traj, frame)


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = MyoFullBodyRacket(disable_fingers=True, no_skybox=False)
    model = env._model
    data = mujoco.MjData(model)

    # enlarge the offscreen framebuffer so high-res renders fit
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    # brighten so the dark muscle geoms read clearly against the sky/floor
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    model.vis.headlight.specular[:] = 0.2

    qpos = _load_swing_qpos(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    # look-at point: racket body if present, else pelvis
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    if rid < 0:
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    center = np.array(data.xpos[rid], dtype=float)
    # frame the whole body: center between pelvis and racket, mid height
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = (np.array(data.xpos[pelvis]) + center) / 2.0
    look[2] = 0.9

    views = {
        "front": dict(azimuth=90, elevation=-10, distance=3.2),
        "side": dict(azimuth=0, elevation=-10, distance=3.2),
        "back_45": dict(azimuth=225, elevation=-15, distance=3.2),
        "top": dict(azimuth=90, elevation=-75, distance=3.4),
        "swing_3q": dict(azimuth=135, elevation=-12, distance=3.0),
        "racket_closeup": dict(azimuth=110, elevation=-8, distance=1.3),
    }

    cam = mujoco.MjvCamera()
    imgs = {}
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        for name, v in views.items():
            cam.lookat[:] = center if name == "racket_closeup" else look
            cam.azimuth = v["azimuth"]
            cam.elevation = v["elevation"]
            cam.distance = v["distance"]
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            imgs[name] = img
            out = OUT_DIR / f"racket_scene_{name}.png"
            _save_png(img, out)
            subj = float((img.mean(-1) < 245).mean()) * 100
            print(f"  {name:14s} -> {out.name}  subject%={subj:4.1f}")

    _save_grid(imgs, OUT_DIR / "racket_scene_multiview.png")
    print(f"grid -> {OUT_DIR / 'racket_scene_multiview.png'}")


def _save_png(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)


def _save_grid(imgs: dict, path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    order = ["front", "side", "back_45", "swing_3q", "top", "racket_closeup"]
    order = [k for k in order if k in imgs]
    cols, rows = 3, 2
    cw = max(i.shape[1] for i in imgs.values())
    ch = max(i.shape[0] for i in imgs.values())
    grid = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for idx, name in enumerate(order):
        r, c = divmod(idx, cols)
        tile = Image.fromarray(imgs[name])
        grid.paste(tile, (c * cw, r * ch))
        draw.text((c * cw + 12, r * ch + 10), name, fill=(20, 20, 20))
    grid.save(path)


if __name__ == "__main__":
    main()
