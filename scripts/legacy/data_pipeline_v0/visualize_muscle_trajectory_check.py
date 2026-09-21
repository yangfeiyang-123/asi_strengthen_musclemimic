#!/usr/bin/env python3
"""Render sampled muscle_trajectory caches for quick raw/optimized checks."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT_DIR = REPO_ROOT / "musclemimic" / "badminton" / "scripts"
OPTIMIZED_WHAM_ROOT = Path("/data3/yangfeiyang/WorkSpace/optimized_wham")
sys.path.insert(0, str(RENDER_SCRIPT_DIR))
sys.path.insert(0, str(OPTIMIZED_WHAM_ROOT))

from render_retarget_cache import _make_model, render_cache  # noqa: E402
from lib.video_fps import detect_video_fps  # noqa: E402


DEFAULT_SKIP_DIRS = {"_global", "_index", "temp_visualize_check"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class RenderJob(NamedTuple):
    action: str
    variant: str
    motion: str
    fps: float


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def _action_dirs(datasets_root: Path, skip_dirs: set[str]) -> list[Path]:
    return sorted(
        [path for path in datasets_root.iterdir() if path.is_dir() and path.name not in skip_dirs],
        key=lambda path: _natural_key(path.name),
    )


def _iter_raw_videos(action_dir: Path) -> list[Path]:
    raw_video_root = action_dir / "raw_video"
    if not raw_video_root.exists():
        return []
    return sorted(
        [
            path
            for path in raw_video_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ],
        key=lambda path: _natural_key(str(path.relative_to(raw_video_root))),
    )


def _sequence_name_for_video(action_dir: Path, video: Path) -> str:
    raw_video_root = action_dir / "raw_video"
    videos = _iter_raw_videos(action_dir)
    same_stem_count = sum(1 for candidate in videos if candidate.stem == video.stem)
    if same_stem_count <= 1:
        return video.stem
    try:
        relative = video.relative_to(raw_video_root).with_suffix("")
    except ValueError:
        return video.stem
    return "__".join(relative.parts)


def _select_videos(action_dir: Path, limit_per_action: int) -> list[tuple[str, Path, float]]:
    raw_dir = action_dir / "muscle_trajectory" / "raw"
    optimized_dir = action_dir / "muscle_trajectory" / "optimized"
    selected: list[tuple[str, Path, float]] = []
    for video in _iter_raw_videos(action_dir):
        motion = _sequence_name_for_video(action_dir, video)
        if (raw_dir / f"{motion}.npz").exists() and (optimized_dir / f"{motion}.npz").exists():
            fps = detect_video_fps(video)
            if fps is None:
                raise RuntimeError(f"Could not detect source-video FPS: {video}")
            selected.append((motion, video, float(fps)))
            if len(selected) >= limit_per_action:
                break
    return selected


def _write_manifest(output_root: Path, selections: dict[str, list[tuple[str, Path, float]]]) -> None:
    manifest = output_root / "selection_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("action\tmotion\tfps\tvideo\n")
        for action, selected in selections.items():
            for motion, video, fps in selected:
                handle.write(f"{action}\t{motion}\t{fps:g}\t{video}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "temp_visualize_check",
    )
    parser.add_argument("--limit-per-action", type=int, default=5)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Override visualization FPS. Defaults to each source video's detected FPS.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.limit_per_action <= 0:
        raise ValueError("--limit-per-action must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must be in [0, --num-workers)")

    datasets_root = args.datasets_root.resolve()
    output_root = args.output_root.resolve()

    selections: dict[str, list[tuple[str, Path, float]]] = {}
    for action_dir in _action_dirs(datasets_root, DEFAULT_SKIP_DIRS | {output_root.name}):
        selected = _select_videos(action_dir, args.limit_per_action)
        if selected:
            selections[action_dir.name] = selected
        else:
            print(f"[SKIP] {action_dir.name}: no paired raw/optimized muscle_trajectory npz files", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_root, selections)

    all_jobs: list[RenderJob] = []
    for action, selected in selections.items():
        for variant in ("raw", "optimized"):
            for motion, _video, fps in selected:
                sample_fps = args.sample_fps if args.sample_fps is not None else fps
                all_jobs.append(RenderJob(action, variant, motion, sample_fps))

    jobs = [
        job
        for job_index, job in enumerate(all_jobs)
        if job_index % args.num_workers == args.worker_index
    ]

    total = len(all_jobs)
    print(f"[INFO] selected {sum(len(v) for v in selections.values())} source videos across {len(selections)} actions", flush=True)
    print(f"[INFO] rendering {total} videos to {output_root}", flush=True)
    print(
        f"[INFO] worker {args.worker_index}/{args.num_workers}: {len(jobs)} assigned videos",
        flush=True,
    )

    model, data = _make_model(REPO_ROOT / "musclemimic" / "badminton", REPO_ROOT)
    rendered = 0
    skipped = 0

    for job in jobs:
        cache_root = datasets_root / job.action / "muscle_trajectory" / job.variant
        out_dir = output_root / job.action / "muscle_trajectory" / job.variant
        output_path = out_dir / f"{job.motion}.{args.format}"
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"[SKIP] existing {output_path}", flush=True)
            continue
        print(f"[RUN] {job.action}/{job.variant}/{job.motion} @ {job.fps:g} fps", flush=True)
        render_cache(
            model,
            data,
            cache_root / f"{job.motion}.npz",
            output_path,
            args.width,
            args.height,
            args.stride,
            args.fps,
            job.fps,
            args.format,
        )
        rendered += 1

    print(f"[DONE] rendered={rendered} skipped_existing={skipped} output_root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
