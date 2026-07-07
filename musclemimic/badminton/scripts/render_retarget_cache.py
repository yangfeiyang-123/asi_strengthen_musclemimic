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


def _set_env_pair(primary: str, alias: str, value: Path, *, override: bool) -> None:
    if override:
        os.environ[primary] = str(value)
        os.environ[alias] = str(value)
        return
    os.environ.setdefault(primary, str(value))
    os.environ.setdefault(alias, os.environ[primary])


def _configure_paths(
    project_root: Path,
    repo_root: Path,
    amass_root: Path | None = None,
    converted_amass_root: Path | None = None,
    gmr_cache_root: Path | None = None,
) -> None:
    sys.path.insert(0, str(repo_root))
    _set_env_pair(
        "MUSCLEMIMIC_AMASS_PATH",
        "AMASS_PATH",
        amass_root or repo_root / "datasets" / "_global" / "amass_npz",
        override=amass_root is not None,
    )
    _set_env_pair(
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
        "CONVERTED_AMASS_PATH",
        converted_amass_root or repo_root / "datasets" / "_global" / "muscle_trajectory" / "gmr_cache",
        override=converted_amass_root is not None,
    )
    if gmr_cache_root is not None:
        os.environ["MUSCLEMIMIC_GMR_CACHE_PATH"] = str(gmr_cache_root)
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def _make_model(
    project_root: Path,
    repo_root: Path,
    amass_root: Path | None = None,
    converted_amass_root: Path | None = None,
    gmr_cache_root: Path | None = None,
):
    _configure_paths(
        project_root,
        repo_root,
        amass_root=amass_root,
        converted_amass_root=converted_amass_root,
        gmr_cache_root=gmr_cache_root,
    )
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


def _default_cache_root(repo_root: Path) -> Path:
    gmr_cache_root = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH")
    if gmr_cache_root:
        return Path(gmr_cache_root)

    converted_root = os.environ.get("CONVERTED_AMASS_PATH") or os.environ.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if converted_root:
        return Path(converted_root) / "MyoFullBody" / "gmr"

    return repo_root / "datasets" / "_global" / "muscle_trajectory" / "gmr_cache" / "MyoFullBody" / "gmr"


def _resolve_cache_path(cache_root: Path, motion: str) -> Path:
    return cache_root / f"{motion.removesuffix('.npz')}.npz"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", action="append", required=True, help="Motion name without .npz")
    parser.add_argument("--output-dir", type=Path, default=Path("BadmintonMimic/outputs/vis/cache_preview"))
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=None,
        help="AMASS-style NPZ root used while constructing the MyoFullBody model.",
    )
    parser.add_argument(
        "--converted-amass-root",
        type=Path,
        default=None,
        help="Legacy converted AMASS root. Defaults to datasets/_global/muscle_trajectory/gmr_cache.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Direct MyoFullBody GMR cache root, e.g. datasets/<action>/muscle_trajectory/gmr_cache.",
    )
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
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    project_root = _project_root()
    model, data = _make_model(
        project_root,
        repo_root,
        amass_root=args.amass_root,
        converted_amass_root=args.converted_amass_root,
        gmr_cache_root=args.cache_root,
    )
    cache_root = args.cache_root or _default_cache_root(repo_root)

    for motion in args.motion:
        cache_path = _resolve_cache_path(cache_root, motion)
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
