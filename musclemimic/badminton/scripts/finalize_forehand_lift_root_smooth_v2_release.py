#!/usr/bin/env python3
"""Bind numeric and manually reviewed visual QC into the v2 release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from musclemimic.badminton.scripts.prepare_forehand_lift_root_smooth_v2 import (
    TRAIN_MOTIONS,
    VAL_MOTIONS,
    VARIANT,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = float(payload["format"]["duration"])
    frames = int(stream.get("nb_frames") or 0)
    if path.stat().st_size < 10_000 or duration <= 0.0 or frames < 10:
        raise ValueError(f"invalid or incomplete visual QC video: {path}")
    return {
        "path": str(path.relative_to(_repo_root())),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "average_frame_rate": str(stream["avg_frame_rate"]),
        "frame_count": frames,
        "duration_s": duration,
    }


def finalize(dataset_root: Path, *, reviewer: str) -> tuple[dict, dict]:
    repo_root = _repo_root()
    dataset_root = dataset_root.resolve()
    manifest_root = dataset_root / "manifests" / VARIANT
    numeric_path = manifest_root / "numeric_qc_report.json"
    source_release_path = manifest_root / "source_release.json"
    video_root = dataset_root / "training" / "qc" / f"{VARIANT}_videos"
    storyboard_root = dataset_root / "training" / "qc" / f"{VARIANT}_storyboards"
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if numeric.get("passed") is not True or numeric.get("errors"):
        raise ValueError("numeric QC report is not a clean pass")

    videos = []
    storyboards = []
    for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS):
        video = video_root / f"{motion}.mp4"
        storyboard = storyboard_root / f"{motion}.png"
        if not storyboard.is_file() or storyboard.stat().st_size < 10_000:
            raise ValueError(f"missing or incomplete visual storyboard: {storyboard}")
        videos.append({"motion": motion, **_probe_video(video)})
        storyboards.append(
            {
                "motion": motion,
                "path": str(storyboard.relative_to(repo_root)),
                "sha256": _sha256(storyboard),
                "size_bytes": storyboard.stat().st_size,
            }
        )

    reviewed_at = datetime.now(UTC).isoformat()
    visual = {
        "schema_version": "forehand_lift_root_smooth_v2_visual_qc_v1",
        "variant": VARIANT,
        "reviewed_at": reviewed_at,
        "reviewer": reviewer,
        "review_method": "six uniformly sampled frames per motion plus complete preview playback artifacts",
        "required_observations": {
            "continuous_root_and_limb_motion": True,
            "upright_noncollapsed_body": True,
            "no_visible_floor_penetration_or_floating": True,
            "forehand_lift_action_preserved": True,
            "all_train_and_validation_motions_reviewed": True,
        },
        "findings": [],
        "videos": videos,
        "storyboards": storyboards,
        "passed": True,
    }
    visual_path = manifest_root / "visual_qc_report.json"
    visual_path.write_text(
        json.dumps(visual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config_path = (
        repo_root
        / "fullbody/config_specific_task/stage1_body/"
        "conf_fullbody_forehand_lift_optimized_root_smooth_v2.yaml"
    )
    release = {
        "schema_version": "forehand_lift_root_smooth_v2_release_v1",
        "dataset": "forehandLift",
        "variant": VARIANT,
        "created_at": reviewed_at,
        "source_fps": 60,
        "cache_fps": 100,
        "train_motions": list(TRAIN_MOTIONS),
        "validation_motions": list(VAL_MOTIONS),
        "source_release": {
            "path": str(source_release_path.relative_to(repo_root)),
            "sha256": _sha256(source_release_path),
        },
        "numeric_qc": {
            "path": str(numeric_path.relative_to(repo_root)),
            "sha256": _sha256(numeric_path),
            "passed": True,
        },
        "visual_qc": {
            "path": str(visual_path.relative_to(repo_root)),
            "sha256": _sha256(visual_path),
            "passed": True,
        },
        "retarget_contract": {
            "method": "gmr",
            "target_fps": 60,
            "cache_fps": 100,
            "damping": 0.5,
            "offset_to_ground": False,
            "grounding_mode": "global",
            "use_velocity_limit": False,
            "clear_cache_during_materialization": True,
            "clear_cache_during_training": False,
        },
        "implementation": {
            "source_recipe": "musclemimic/badminton/scripts/prepare_forehand_lift_root_smooth_v2.py",
            "source_recipe_sha256": _sha256(
                repo_root / "musclemimic/badminton/scripts/prepare_forehand_lift_root_smooth_v2.py"
            ),
            "retarget_launcher": "musclemimic/badminton/scripts/run_retarget.py",
            "retarget_launcher_sha256": _sha256(repo_root / "musclemimic/badminton/scripts/run_retarget.py"),
            "retarget_core_sha256": _sha256(repo_root / "loco_mujoco/smpl/retargeting.py"),
            "training_config": str(config_path.relative_to(repo_root)),
            "training_config_sha256": _sha256(config_path),
        },
        "motions": [
            {
                "motion": row["motion"],
                "split": row["split"],
                "source_sha256": row["source_sha256"],
                "cache_sha256": row["cache_sha256"],
                "analysis_sha256": row["analysis_sha256"],
                "root_rms_ratio_repaired_over_legacy": row["root_rms_ratio_repaired_over_legacy"],
            }
            for row in numeric["motions"]
        ],
        "passed": True,
    }
    release_path = manifest_root / "release_manifest.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return visual, release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_repo_root() / "datasets" / "forehandLift",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--confirm-visual-pass",
        action="store_true",
        help="Required assertion that every generated storyboard has been manually reviewed.",
    )
    args = parser.parse_args()
    if not args.confirm_visual_pass:
        raise SystemExit("refusing to finalize without --confirm-visual-pass")
    visual, release = finalize(args.dataset_root, reviewer=args.reviewer)
    print(
        json.dumps(
            {
                "visual_passed": visual["passed"],
                "release_passed": release["passed"],
                "motion_count": len(release["motions"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
