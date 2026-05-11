#!/usr/bin/env python3
"""Render MyoFullBody retarget cache files to short preview videos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_paths(project_root: Path, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("MUSCLEMIMIC_AMASS_PATH", str(project_root / "data" / "amass_npz"))
    os.environ.setdefault("AMASS_PATH", os.environ["MUSCLEMIMIC_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_CONVERTED_AMASS_PATH", str(repo_root / "caches" / "AMASS"))
    os.environ.setdefault("CONVERTED_AMASS_PATH", os.environ["MUSCLEMIMIC_CONVERTED_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def _make_model(project_root: Path, repo_root: Path):
    _configure_paths(project_root, repo_root)
    from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory

    conf = AMASSDatasetConf(["badminton/train/forehand_clear_clip1_merged_poses"])
    conf.retargeting_method = "gmr"
    conf.gmr_config = {
        "src_human": "smplh",
        "target_fps": 30,
        "solver": "daqp",
        "damping": 0.5,
        "offset_to_ground": False,
        "use_velocity_limit": False,
        "use_fitted_shape": True,
        "shape_fitting_iterations": 500,
    }
    env = ImitationFactory.make("MyoFullBody", amass_dataset_conf=conf, headless=True)
    return env._model, mujoco.MjData(env._model)


def _default_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 145.0
    cam.elevation = -18.0
    cam.distance = 4.2
    cam.lookat[:] = np.array([0.0, 0.0, 1.0])
    return cam


def render_cache(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cache_path: Path,
    output_path: Path,
    width: int,
    height: int,
    stride: int,
    fps: int,
    output_format: str,
) -> None:
    motion = np.load(cache_path, allow_pickle=True)
    qpos = np.asarray(motion["qpos"], dtype=np.float64)
    frame_ids = list(range(0, len(qpos), stride))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _default_camera()
    if output_format == "gif":
        rendered_frames: list[Image.Image] = []
        for frame_id in frame_ids:
            data.qpos[:] = qpos[frame_id]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.9])
            renderer.update_scene(data, camera=camera)
            rendered_frames.append(Image.fromarray(renderer.render()))
        if not rendered_frames:
            raise RuntimeError(f"No frames rendered for {cache_path}")
        rendered_frames[0].save(
            output_path,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        renderer.close()
        print(f"[OK] {cache_path} -> {output_path} ({len(frame_ids)} frames)")
        return

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {output_path}")

    try:
        for frame_id in frame_ids:
            data.qpos[:] = qpos[frame_id]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.9])
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    print(f"[OK] {cache_path} -> {output_path} ({len(frame_ids)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", action="append", required=True, help="Motion name without .npz")
    parser.add_argument("--output-dir", type=Path, default=Path("BadmintonMimic/outputs/vis/cache_preview"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    repo_root = _repo_root()
    project_root = _project_root()
    model, data = _make_model(project_root, repo_root)
    cache_root = repo_root / "caches" / "AMASS" / "MyoFullBody" / "gmr"

    for motion in args.motion:
        cache_path = cache_root / f"{motion}.npz"
        video_name = motion.replace("/", "_") + f".{args.format}"
        render_cache(
            model,
            data,
            cache_path,
            args.output_dir / video_name,
            args.width,
            args.height,
            args.stride,
            args.fps,
            args.format,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
