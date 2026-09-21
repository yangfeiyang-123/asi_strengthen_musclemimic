#!/usr/bin/env python3
"""Run original WHAM+DPVO and initial MuscleMimic retargeting for dataset videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_WHAM_ROOT = Path("/data3/yangfeiyang/WorkSpace/WHAM")
DEFAULT_DPVO_ROOT = Path("/data3/yangfeiyang/WorkSpace/optimized_wham/third-party/DPVO")
DEFAULT_VITPOSE_ROOT = Path("/data3/yangfeiyang/WorkSpace/optimized_wham/third-party/ViTPose")
DEFAULT_MUSCLEMIMIC_RUNNER = (str(REPO_ROOT / ".venv" / "bin" / "python3"),)
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SKIP_ACTION_DIRS = {"_global", "_index", "temp_visualize_check"}


@dataclass(frozen=True)
class VideoTask:
    index: int
    action_dir: Path
    video: Path
    sequence: str
    fps: float

    @property
    def action(self) -> str:
        return self.action_dir.name

    @property
    def initial_wham_root(self) -> Path:
        # Keep the spelling requested by the user: wham/inital_wham.
        return self.action_dir / "wham" / "inital_wham"

    @property
    def wham_dir(self) -> Path:
        return self.initial_wham_root / self.sequence

    @property
    def wham_pkl(self) -> Path:
        return self.wham_dir / "wham_output.pkl"

    @property
    def amass_npz(self) -> Path:
        return self.initial_wham_root / f"{self.sequence}.npz"

    @property
    def initial_gmr_root(self) -> Path:
        return self.action_dir / "muscle_trajectory" / "initial"

    @property
    def initial_gmr_npz(self) -> Path:
        return self.initial_gmr_root / f"{self.sequence}.npz"


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def _action_dirs(datasets_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [datasets_root / action for action in requested]
    return sorted(
        [
            path
            for path in datasets_root.iterdir()
            if path.is_dir() and path.name not in SKIP_ACTION_DIRS
        ],
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


def _detect_video_fps(video: Path) -> float:
    sys.path.insert(0, str(Path("/data3/yangfeiyang/WorkSpace/optimized_wham")))
    from lib.video_fps import detect_video_fps

    fps = detect_video_fps(video)
    if fps is None:
        raise RuntimeError(f"Could not detect FPS for {video}")
    return float(fps)


def discover_tasks(datasets_root: Path, actions: list[str] | None) -> list[VideoTask]:
    tasks: list[VideoTask] = []
    for action_dir in _action_dirs(datasets_root, actions):
        videos = _iter_raw_videos(action_dir)
        if not videos:
            print(f"[SKIP] {action_dir.name}: no raw videos", flush=True)
            continue
        for video in videos:
            tasks.append(
                VideoTask(
                    index=len(tasks),
                    action_dir=action_dir,
                    video=video,
                    sequence=_sequence_name_for_video(action_dir, video),
                    fps=_detect_video_fps(video),
                )
            )
    return tasks


def assigned_tasks(tasks: list[VideoTask], num_workers: int, worker_index: int) -> list[VideoTask]:
    if num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError("--worker-index must be in [0, --num-workers)")
    return [task for task in tasks if task.index % num_workers == worker_index]


def _write_status(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_joblib(path: Path):
    import joblib

    return joblib.load(path)


def _assert_dpvo_slam(task: VideoTask) -> None:
    import numpy as np

    slam_path = task.wham_dir / "slam_results.pth"
    if not slam_path.exists():
        raise RuntimeError(f"Missing DPVO slam_results: {slam_path}")
    slam = np.asarray(_load_joblib(slam_path))
    if slam.ndim != 2 or slam.shape[1] < 7:
        raise RuntimeError(f"Unexpected slam_results shape for {task.sequence}: {slam.shape}")
    local_only = np.zeros_like(slam)
    local_only[:, 3] = 1.0
    if np.allclose(slam[:, :7], local_only[:, :7], atol=1e-7):
        raise RuntimeError(f"DPVO appears inactive/local-only for {task.sequence}")


def _load_wham_runtime(wham_root: Path, dpvo_root: Path, vitpose_root: Path):
    os.chdir(wham_root)
    sys.path.insert(0, str(vitpose_root))
    sys.path.insert(0, str(dpvo_root))
    sys.path.insert(0, str(wham_root))

    import torch
    from configs.config import get_cfg_defaults
    from lib.models import build_body_model, build_network
    import lib.models.preproc.slam as slam_mod
    import demo as wham_demo

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; WHAM+DPVO must run on GPU.")
    if not (dpvo_root / "config" / "default.yaml").exists():
        raise RuntimeError(f"DPVO config not found: {dpvo_root}")
    if not (vitpose_root / "mmpose").exists():
        raise RuntimeError(f"ViTPose/mmpose not found: {vitpose_root}")
    if not (wham_root / "checkpoints" / "dpvo.pth").exists():
        raise RuntimeError(f"DPVO checkpoint not found under {wham_root / 'checkpoints'}")

    slam_mod.DPVO_DIR = str(dpvo_root)
    wham_demo._run_global = True
    wham_demo.args = SimpleNamespace(run_smplify=False)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(wham_root / "configs" / "yamls" / "demo.yaml"))
    cfg.DEVICE = "cuda"
    print(f"[WHAM] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    return cfg, network, wham_demo


def run_wham_phase(
    tasks: list[VideoTask],
    *,
    wham_root: Path,
    dpvo_root: Path,
    vitpose_root: Path,
    force: bool,
    status_path: Path,
) -> None:
    if not tasks:
        return
    cfg, network, wham_demo = _load_wham_runtime(wham_root, dpvo_root, vitpose_root)
    for task in tasks:
        try:
            if task.wham_pkl.exists() and not force:
                _assert_dpvo_slam(task)
                print(f"[SKIP wham] {task.action}/{task.sequence}", flush=True)
                continue
            if force and task.wham_dir.exists():
                shutil.rmtree(task.wham_dir)
            task.wham_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RUN wham] {task.index}: {task.action}/{task.sequence} @ {task.fps:g} fps", flush=True)
            wham_demo.run(
                cfg,
                str(task.video),
                str(task.wham_dir),
                network,
                calib=None,
                run_global=True,
                save_pkl=True,
                visualize=False,
            )
            _assert_dpvo_slam(task)
            _write_status(status_path, {"phase": "wham", "status": "ok", "index": task.index, "action": task.action, "sequence": task.sequence})
        except Exception as exc:
            _write_status(
                status_path,
                {
                    "phase": "wham",
                    "status": "error",
                    "index": task.index,
                    "action": task.action,
                    "sequence": task.sequence,
                    "error": repr(exc),
                },
            )
            raise


def _run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def run_convert_phase(
    tasks: list[VideoTask],
    *,
    musclemimic_runner: list[str],
    force: bool,
    status_path: Path,
) -> None:
    for task in tasks:
        if not task.wham_pkl.exists():
            raise FileNotFoundError(f"Missing WHAM output: {task.wham_pkl}")
        if task.amass_npz.exists() and not force:
            print(f"[SKIP convert] {task.action}/{task.sequence}", flush=True)
            continue
        task.initial_wham_root.mkdir(parents=True, exist_ok=True)
        command = [
            *musclemimic_runner,
            str(REPO_ROOT / "musclemimic" / "badminton" / "scripts" / "convert_wham_to_amass.py"),
            "--input",
            str(task.wham_pkl),
            "--output",
            str(task.amass_npz),
            "--fps",
            str(int(task.fps) if float(task.fps).is_integer() else task.fps),
            "--force-fps",
            "--video",
            str(task.video),
            "--gender",
            "neutral",
            "--merge-tracks",
        ]
        _run_command(command, REPO_ROOT)
        _write_status(status_path, {"phase": "convert", "status": "ok", "index": task.index, "action": task.action, "sequence": task.sequence})


def _fps_key(fps: float) -> str:
    text = f"{fps:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def run_retarget_phase(
    tasks: list[VideoTask],
    *,
    musclemimic_runner: list[str],
    force: bool,
    status_path: Path,
) -> None:
    grouped: dict[tuple[Path, float], list[VideoTask]] = {}
    for task in tasks:
        if not task.amass_npz.exists():
            raise FileNotFoundError(f"Missing AMASS initial npz: {task.amass_npz}")
        grouped.setdefault((task.action_dir, round(float(task.fps), 6)), []).append(task)

    for (action_dir, fps), group_tasks in sorted(grouped.items(), key=lambda item: (_natural_key(item[0][0].name), item[0][1])):
        missing = [task for task in group_tasks if not task.initial_gmr_npz.exists()]
        if not missing and not force:
            print(f"[SKIP retarget] {action_dir.name} fps={fps:g} count={len(group_tasks)}", flush=True)
            continue
        if force:
            for task in group_tasks:
                if task.initial_gmr_npz.exists():
                    task.initial_gmr_npz.unlink()

        manifest = action_dir / "manifests" / "initial" / f"initial_fps_{_fps_key(fps)}.txt"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(task.sequence for task in group_tasks) + "\n", encoding="utf-8")
        (action_dir / "muscle_trajectory" / "initial").mkdir(parents=True, exist_ok=True)

        command = [
            *musclemimic_runner,
            str(REPO_ROOT / "musclemimic" / "badminton" / "scripts" / "run_retarget.py"),
            "--manifest",
            str(manifest),
            "--amass-root",
            str(action_dir / "wham" / "inital_wham"),
            "--gmr-cache-root",
            str(action_dir / "muscle_trajectory" / "initial"),
            "--fps",
            str(int(fps) if float(fps).is_integer() else fps),
        ]
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("JAX_PLATFORM_NAME", "cpu")
        _run_command(command, REPO_ROOT, env=env)
        for task in group_tasks:
            if not task.initial_gmr_npz.exists():
                raise RuntimeError(f"Retarget did not create {task.initial_gmr_npz}")
            _write_status(status_path, {"phase": "retarget", "status": "ok", "index": task.index, "action": task.action, "sequence": task.sequence})


def audit(tasks: list[VideoTask]) -> int:
    missing_wham = [task for task in tasks if not task.wham_pkl.exists()]
    missing_amass = [task for task in tasks if not task.amass_npz.exists()]
    missing_gmr = [task for task in tasks if not task.initial_gmr_npz.exists()]
    print(f"[AUDIT] videos={len(tasks)}")
    print(f"[AUDIT] missing_wham={len(missing_wham)} missing_amass={len(missing_amass)} missing_initial_gmr={len(missing_gmr)}")
    by_action: dict[str, list[VideoTask]] = {}
    for task in tasks:
        by_action.setdefault(task.action, []).append(task)
    for action in sorted(by_action, key=_natural_key):
        action_tasks = by_action[action]
        wham_done = sum(task.wham_pkl.exists() for task in action_tasks)
        gmr_done = sum(task.initial_gmr_npz.exists() for task in action_tasks)
        print(f"[AUDIT] {action}: videos={len(action_tasks)} wham={wham_done} initial_gmr={gmr_done}")
    return 0 if not missing_wham and not missing_amass and not missing_gmr else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--wham-root", type=Path, default=DEFAULT_WHAM_ROOT)
    parser.add_argument("--dpvo-root", type=Path, default=DEFAULT_DPVO_ROOT)
    parser.add_argument("--vitpose-root", type=Path, default=DEFAULT_VITPOSE_ROOT)
    parser.add_argument("--musclemimic-runner", nargs="+", default=list(DEFAULT_MUSCLEMIMIC_RUNNER))
    parser.add_argument("--actions", nargs="*", default=None)
    parser.add_argument("--phase", choices=["wham", "convert", "retarget", "all", "audit"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--force-wham", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--force-retarget", action="store_true")
    parser.add_argument("--status-log", type=Path, default=REPO_ROOT / "logs" / "initial_wham_dpvo_status.jsonl")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

    tasks = discover_tasks(args.datasets_root.resolve(), args.actions)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    selected = assigned_tasks(tasks, args.num_workers, args.worker_index)
    print(
        f"[INFO] discovered={len(tasks)} assigned={len(selected)} "
        f"worker={args.worker_index}/{args.num_workers} phase={args.phase}",
        flush=True,
    )

    if args.phase == "audit":
        return audit(tasks)
    if args.phase in {"wham", "all"}:
        run_wham_phase(
            selected,
            wham_root=args.wham_root.resolve(),
            dpvo_root=args.dpvo_root.resolve(),
            vitpose_root=args.vitpose_root.resolve(),
            force=args.force_wham,
            status_path=args.status_log,
        )
    if args.phase in {"convert", "all"}:
        run_convert_phase(
            selected,
            musclemimic_runner=args.musclemimic_runner,
            force=args.force_convert,
            status_path=args.status_log,
        )
    if args.phase in {"retarget", "all"}:
        run_retarget_phase(
            selected,
            musclemimic_runner=args.musclemimic_runner,
            force=args.force_retarget,
            status_path=args.status_log,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
