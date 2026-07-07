#!/usr/bin/env python3
"""Stage locally-optimized muscle trajectories into the GMR cache for tracking.

The badminton motion clips referenced by the mainline tracking configs
(``rel_dataset_path``) are not present in the GMR cache, and re-retargeting is
blocked (no SMPL model / HF 404). But ``datasets/<action>/muscle_trajectory/
optimized/*.npz`` already ARE retargeted MyoFullBody trajectories. This tool
symlinks (or copies) those npz into the GMR cache under stable relative names
so ``ImitationFactory`` loads them from cache instead of retargeting.

Layout produced (default cache root, per action ``forehandClear_standard``):

    <gmr_cache>/MyoFullBody/gmr/skill/forehandClear_standard/<clip>.npz

The matching ``rel_dataset_path`` entries are ``skill/forehandClear_standard/<clip>``.
Use ``--emit-manifest`` to also write train/val list files consumed by the
config generator.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Dedicated flat cache root consumed via MUSCLEMIMIC_GMR_CACHE_PATH. When that
# env var is set, ImitationFactory resolves rel_dataset_path to
# ``<cache_root>/<rel>.npz`` directly (no MyoFullBody/gmr subdir), so we stage
# clips at ``<cache_root>/skill/<action>/<clip>.npz`` and the matching
# rel_dataset_path is ``skill/<action>/<clip>``.
DEFAULT_CACHE_ROOT = REPO_ROOT / "datasets" / "_global" / "muscle_trajectory" / "skill_cache"
SKILL_NAMESPACE = "skill"


def _optimized_clips(action_dir: Path) -> list[Path]:
    opt_dir = action_dir / "muscle_trajectory" / "optimized"
    if not opt_dir.is_dir():
        raise FileNotFoundError(f"no optimized trajectory dir: {opt_dir}")
    clips = sorted(
        p for p in opt_dir.glob("*.npz") if not p.stem.endswith("_analysis")
    )
    if not clips:
        raise FileNotFoundError(f"no optimized *.npz (excluding *_analysis) in {opt_dir}")
    return clips


def _validate_clip(path: Path) -> dict[str, object]:
    import numpy as np
    import loco_mujoco.core.observations.base  # noqa: F401  break circular import
    from loco_mujoco.trajectory.dataclasses import Trajectory

    traj = Trajectory.load(str(path), backend=np)
    qpos = np.asarray(traj.data.qpos)
    return {
        "frames": int(qpos.shape[0]),
        "qpos_dim": int(qpos.shape[1]),
        "frequency_hz": float(traj.info.frequency),
        "njnt": int(traj.info.model.njnt),
        "nsite": int(traj.info.model.nsite),
    }


def stage_action(
    action: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    link_mode: str = "symlink",
    validate: bool = True,
    val_ratio: float = 0.2,
    seed: int = 0,
    max_clips: int | None = None,
) -> dict[str, object]:
    action_dir = REPO_ROOT / "datasets" / action
    clips = _optimized_clips(action_dir)
    if max_clips is not None:
        clips = clips[: max_clips]

    dest_dir = cache_root / SKILL_NAMESPACE / action
    dest_dir.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    staged: list[dict[str, object]] = []
    for clip in clips:
        rel = f"{SKILL_NAMESPACE}/{action}/{clip.stem}"
        dest = dest_dir / f"{clip.stem}.npz"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if link_mode == "symlink":
            dest.symlink_to(clip.resolve())
        elif link_mode == "copy":
            import shutil

            shutil.copy2(clip, dest)
        else:
            raise ValueError(f"link_mode must be symlink|copy, got {link_mode!r}")
        entry: dict[str, object] = {"rel_dataset_path": rel, "source": str(clip)}
        if validate:
            entry.update(_validate_clip(clip))
        rel_paths.append(rel)
        staged.append(entry)

    rng = random.Random(seed)
    shuffled = list(rel_paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:]) or sorted(rel_paths)
    if not val:
        val = train[:1]

    return {
        "action": action,
        "cache_dir": str(dest_dir),
        "num_clips": len(clips),
        "train": train,
        "val": val,
        "clips": staged,
    }


def _emit_manifest(report: dict[str, object], manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "train_list.txt").write_text(
        "\n".join(report["train"]) + "\n", encoding="utf-8"
    )
    (manifest_dir / "val_list.txt").write_text(
        "\n".join(report["val"]) + "\n", encoding="utf-8"
    )
    (manifest_dir / "stage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, help="dataset action dir under datasets/")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "--emit-manifest",
        type=Path,
        default=None,
        help="write train_list.txt/val_list.txt/stage_report.json here",
    )
    args = parser.parse_args()

    report = stage_action(
        args.action,
        cache_root=args.cache_root,
        link_mode=args.link_mode,
        validate=not args.no_validate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_clips=args.max_clips,
    )
    if args.emit_manifest is not None:
        _emit_manifest(report, args.emit_manifest)
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
