#!/usr/bin/env python3
"""Render side-by-side videos: human reference (retargeted) | simulated arms, muscles coloured by activation.

Reads outputs/stage1_endpoint_compare/kin/{model.mjb, <arm>_s<seed>.npz}.
Usage: MUJOCO_GL=egl .venv/bin/python experiments/stage1/render_endpoint_videos.py --arms T0:0 T3:0 --trajs 0 3 --fps 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
KIN = REPO / "outputs/stage1_endpoint_compare/kin"
VID = REPO / "outputs/stage1_endpoint_compare/videos"
LABEL = {"T0": "T0  no EMG", "T1": "T1  anchor only", "T2": "T2  synergy only", "T3": "T3  PEASD-Lite", "T4": "T4  phase-shifted"}


def make_renderer(model, size):
    r = mujoco.Renderer(model, size, size)
    return r


def scene_options(show_muscles: bool):
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = show_muscles
    opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = show_muscles
    opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    return opt


def camera(lookat, azimuth=140.0, elevation=-12.0, distance=2.6):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth = azimuth; cam.elevation = elevation; cam.distance = distance
    return cam


def render_pose(renderer, model, data, qpos, act, addresses, opt, cam):
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    if act is not None:
        data.act[:] = 0.0
        data.act[addresses] = act
        data.ctrl[:] = 0.0
    else:
        data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render()


def label(img, text, sub=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:  # noqa: BLE001
        font = small = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    d.text((10, 6), text, fill=(255, 255, 255), font=font)
    if sub:
        d.text((10, im.height - 24), sub, fill=(255, 255, 0), font=small)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["T0:0", "T3:0"])
    ap.add_argument("--trajs", nargs="+", type=int, default=[0])
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=140.0)
    args = ap.parse_args()
    VID.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(KIN / "model.mjb"))
    meta = json.load(open(KIN / "model_meta.json"))
    addresses = np.asarray(meta["activation_addresses"], dtype=np.int32)
    data = mujoco.MjData(model)
    renderer = make_renderer(model, args.size)
    arms = [(a, int(s)) for a, s in (x.split(":") for x in args.arms)]
    kins = {k: np.load(KIN / f"{k[0]}_s{k[1]}.npz") for k in arms}
    ref_src = kins[arms[0]]
    site_names = [str(s) for s in ref_src["site_names"]]
    pelvis = site_names.index("pelvis_mimic")
    opt_ref = scene_options(False); opt_sim = scene_options(True)
    for ti in args.trajs:
        ref_q = ref_src[f"ref_qpos_traj{ti}"]; ref_site = ref_src[f"ref_site_traj{ti}"]
        L = ref_q.shape[0]
        sims = {k: (z[f"qpos_traj{ti}"], z[f"act_traj{ti}"]) for k, z in kins.items()}
        out = VID / (f"traj{ti}_ref_" + "_".join(f"{a}s{s}" for a, s in arms) + ".mp4")
        writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
        for t in range(L):
            look = ref_site[t, pelvis].copy(); look[2] = 0.9
            cam = camera(look, azimuth=args.azimuth)
            panels = [label(render_pose(renderer, model, data, ref_q[t], None, addresses, opt_ref, cam),
                            "Human reference (retargeted)", f"progress {t / max(L - 1, 1):.2f}")]
            for k in arms:
                q, a = sims[k]
                if t < q.shape[0]:
                    img = render_pose(renderer, model, data, q[t], a[t], addresses, opt_sim, cam)
                    sub = f"mean act {a[t].mean():.2f}"
                else:
                    img = np.zeros((args.size, args.size, 3), np.uint8) + 40
                    sub = "episode ended (fell / terminated)"
                panels.append(label(img, LABEL[k[0]] + f"  seed{k[1]}", sub))
            writer.append_data(np.concatenate(panels, axis=1))
        writer.close()
        print("wrote", out, f"({L} frames)", flush=True)


if __name__ == "__main__":
    main()
