#!/usr/bin/env python3
"""Backfill the true frame rate into legacy WHAM pkls that predate fps storage.

WHAM now writes ``mocap_framerate`` into ``wham_output.pkl`` (demo.py), but pkls produced
before that carry no fps, forcing every downstream step to be told the rate by hand — the
root of the "jump played at 2x speed" bug. This scans existing ``wham/raw_wham/<seq>/
wham_output.pkl`` files under a dataset action, probes the fps of the matching
``raw_video/<seq>`` clip, and writes it back into every track of the pkl (in place). No
WHAM re-run needed; the pkls become self-describing so future retargets need no --fps.

Usage:
  backfill_pkl_fps.py --datasets-root <root> [--action JumpAndSmash] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fps_utils import VIDEO_SUFFIXES, detect_video_fps, fps_from_record  # noqa: E402

DEFAULT_DATASETS_ROOT = Path("/data3/yangfeiyang/WorkSpace/musclemimic/datasets")


def _find_video(action_dir: Path, sequence: str) -> Path | None:
    """Locate the raw video for a sequence (handles nested `a__b` -> a/b layouts)."""
    raw_root = action_dir / "raw_video"
    rel = Path(*sequence.split("__"))
    candidates = [raw_root / rel.with_suffix(s) for s in VIDEO_SUFFIXES]
    candidates += [raw_root / f"{sequence}{s}" for s in VIDEO_SUFFIXES]
    for c in candidates:
        if c.exists():
            return c
    # last resort: any video whose stem matches the last path segment
    stem = sequence.split("__")[-1]
    for c in raw_root.rglob("*"):
        if c.is_file() and c.suffix.lower() in VIDEO_SUFFIXES and c.stem == stem:
            return c
    return None


def _write_fps_into_pkl(pkl_path: Path, fps: float) -> bool:
    """Write mocap_framerate/fps into every track of a WHAM pkl. Returns True if changed."""
    results = joblib.load(pkl_path)
    if not isinstance(results, dict):
        return False
    changed = False
    for value in results.values():
        if isinstance(value, dict):
            for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
                value[key] = float(fps)
            changed = True
    if changed:
        joblib.dump(results, pkl_path)
    return changed


def backfill(datasets_root: Path, actions: list[str] | None, *, dry_run: bool, force: bool) -> int:
    action_dirs = (
        [datasets_root / a for a in actions]
        if actions
        else sorted(p for p in datasets_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    )
    n_done = n_skip = n_fail = 0
    for action_dir in action_dirs:
        raw_wham = action_dir / "wham" / "raw_wham"
        if not raw_wham.exists():
            continue
        for seq_dir in sorted(p for p in raw_wham.iterdir() if p.is_dir()):
            pkl = seq_dir / "wham_output.pkl"
            if not pkl.exists():
                continue
            sequence = seq_dir.name
            existing = fps_from_record(_first_track(pkl))
            if existing and not force:
                print(f"[skip] {action_dir.name}/{sequence}: already has fps={existing:g}")
                n_skip += 1
                continue
            video = _find_video(action_dir, sequence)
            if video is None:
                print(f"[FAIL] {action_dir.name}/{sequence}: no matching raw video")
                n_fail += 1
                continue
            fps = detect_video_fps(video)
            if not fps:
                print(f"[FAIL] {action_dir.name}/{sequence}: could not probe fps from {video.name}")
                n_fail += 1
                continue
            if dry_run:
                print(f"[dry ] {action_dir.name}/{sequence}: would set fps={fps:g} (from {video.name})")
            else:
                _write_fps_into_pkl(pkl, fps)
                print(f"[ok  ] {action_dir.name}/{sequence}: set fps={fps:g} (from {video.name})")
            n_done += 1
    print(f"\ndone={n_done} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


def _first_track(pkl_path: Path):
    try:
        results = joblib.load(pkl_path)
    except Exception:
        return None
    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, dict):
                return value
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    ap.add_argument("--action", action="append", default=None, help="Action dir name (repeatable); default: all")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would change")
    ap.add_argument("--force", action="store_true", help="Overwrite even if the pkl already has an fps field")
    args = ap.parse_args()
    return backfill(args.datasets_root, args.action, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
