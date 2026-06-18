#!/usr/bin/env python3
"""Render MyoFullBody retarget cache files to short preview videos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before importing mujoco/JAX-backed modules. This script is used
# on headless servers, so default to EGL/CPU unless the caller overrides it.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(disable_fingers=True, no_skybox=True)
    return env._model, mujoco.MjData(env._model)


def _default_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Match the repo's default follow-view recorder instead of the older
    # high side camera.
    cam.azimuth = 90.0
    cam.elevation = 0.0
    cam.distance = 3.4
    cam.lookat[:] = np.array([0.0, 0.0, 1.0])
    return cam


def _resolve_output_fps(cache_frequency: float, stride: int, requested_fps: float | None) -> float:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if requested_fps is not None:
        return float(requested_fps)
    return float(cache_frequency) / float(stride)


def _select_frame_ids(n_frames: int, cache_frequency: float, stride: int, sample_fps: float | None) -> list[int]:
    if sample_fps is None:
        return list(range(0, n_frames, stride))
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    duration = n_frames / float(cache_frequency)
    sample_times = np.arange(0.0, duration, 1.0 / float(sample_fps))
    frame_ids = np.rint(sample_times * float(cache_frequency)).astype(int)
    frame_ids = np.clip(frame_ids, 0, n_frames - 1)
    return np.unique(frame_ids).tolist()


def render_cache(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cache_path: Path,
    output_path: Path,
    width: int,
    height: int,
    stride: int,
    fps: float | None,
    sample_fps: float | None,
    output_format: str,
) -> None:
    motion = np.load(cache_path, allow_pickle=True)
    qpos = np.asarray(motion["qpos"], dtype=np.float64)
    cache_frequency = float(np.asarray(motion["frequency"]).reshape(-1)[0]) if "frequency" in motion else 30.0
    output_fps = float(sample_fps) if fps is None and sample_fps is not None else _resolve_output_fps(cache_frequency, stride, fps)
    frame_ids = _select_frame_ids(len(qpos), cache_frequency, stride, sample_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _default_camera()
    if output_format == "gif":
        rendered_frames: list[Image.Image] = []
        for frame_id in frame_ids:
            data.qpos[:] = qpos[frame_id]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.45])
            renderer.update_scene(data, camera=camera)
            rendered_frames.append(Image.fromarray(renderer.render()))
        if not rendered_frames:
            raise RuntimeError(f"No frames rendered for {cache_path}")
        rendered_frames[0].save(
            output_path,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=int(1000 / output_fps),
            loop=0,
        )
        renderer.close()
        print(f"[OK] {cache_path} -> {output_path} ({len(frame_ids)} frames, {output_fps:g} fps)")
        return

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {output_path}")

    try:
        for frame_id in frame_ids:
            data.qpos[:] = qpos[frame_id]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.45])
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    print(f"[OK] {cache_path} -> {output_path} ({len(frame_ids)} frames, {output_fps:g} fps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", action="append", required=True, help="Motion name without .npz")
    parser.add_argument("--output-dir", type=Path, default=Path("BadmintonMimic/outputs/vis/cache_preview"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Sample cache by time at this fps before writing. Use 30 to create source-video-compatible previews.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video fps. Defaults to --sample-fps when set, otherwise cache frequency divided by --stride.",
    )
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
            args.sample_fps,
            args.format,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
