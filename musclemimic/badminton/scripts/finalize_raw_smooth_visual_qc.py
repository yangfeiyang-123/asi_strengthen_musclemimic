"""Bind the reviewed raw_smooth_v1 previews to the final data release.

This command does not render or modify motion data.  It verifies every preview
with ffprobe, hashes the reviewed images/videos and writes a deterministic
visual-QC report.  The explicit confirmation flag prevents a decode check from
being mistaken for a visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


DATASET = "forehandClear_standard"
VARIANT = "raw_smooth_v1"
OVERRIDE_MOTIONS = (
    "6月2日-1",
    "6月2日-2",
    "6月2日-3",
    "6月2日-4",
    "6月2日-5",
    "6月2日-7",
)
VISUAL_REPORT_RELATIVE_PATH = Path(
    "datasets/forehandClear_standard/manifests/raw_smooth_v1/visual_qc_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,duration,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    stream = streams[0]
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    frames = int(stream["nb_frames"])
    duration = float(stream["duration"])
    if fps != 60.0 or frames <= 0 or duration <= 0.0:
        raise ValueError(
            f"invalid real-time preview: {path} fps={fps} frames={frames} "
            f"duration={duration}"
        )
    if int(stream["width"]) != 640 or int(stream["height"]) != 480:
        raise ValueError(f"unexpected preview resolution: {path}")
    if abs(duration - frames / fps) > 1.0 / fps + 1e-6:
        raise ValueError(f"preview duration/frame mismatch: {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
    }


def build_report(repo_root: Path, *, confirm_all_reviewed: bool) -> dict[str, Any]:
    if not confirm_all_reviewed:
        raise ValueError(
            "refusing to issue visual signoff without --confirm-all-reviewed"
        )
    release_path = repo_root / (
        "datasets/forehandClear_standard/manifests/raw_smooth_v1/"
        "release_manifest.json"
    )
    qc_path = release_path.with_name("qc_report.json")
    video_dir = repo_root / "outputs/vis/raw_smooth_v1_60fps"
    release = _load_json(release_path)
    qc = _load_json(qc_path)
    if release.get("schema_version") != "forehand_clear_raw_smooth_release_v3":
        raise ValueError("visual signoff requires the final release-v3 schema")
    if not qc.get("passed") or not qc.get("clean_passed"):
        raise ValueError("visual signoff requires clean strict numeric QC")
    if qc.get("warnings") or qc.get("hard_errors"):
        raise ValueError("visual signoff refuses numeric QC warnings/errors")

    qc_by_motion = {
        str(item["motion"]): item for item in qc.get("motions", [])
    }
    motion_records: list[dict[str, Any]] = []
    for item in release.get("motions", []):
        motion = str(item["motion"])
        video_path = video_dir / f"raw_smooth_v1_{motion}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing release preview: {video_path}")
        if motion not in qc_by_motion:
            raise ValueError(f"numeric QC has no record for {motion}")
        # QC-v3 stores semantic swing metrics directly on each motion record.
        # Older drafts expected a nested ``metrics`` mapping and silently
        # signed 27 null values, which defeated the numeric/visual binding.
        metrics = qc_by_motion[motion]
        semantic_metrics = {
            "right_hand_path_length_m": metrics.get("right_hand_path_length_m"),
            "max_right_hand_displacement_m": metrics.get(
                "max_right_hand_displacement_m"
            ),
        }
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in semantic_metrics.values()
        ):
            raise ValueError(f"numeric QC has invalid semantic swing metrics for {motion}")
        motion_records.append(
            {
                "motion": motion,
                "split": item["split"],
                "cache_sha256": item["cache"]["sha256"],
                "video_path": str(video_path.relative_to(repo_root)),
                "video_sha256": _sha256(video_path),
                "video": _probe_video(video_path),
                "semantic_swing_metrics": semantic_metrics,
                "reviewed": True,
                "passed": True,
            }
        )

    expected = {str(item["motion"]) for item in release.get("motions", [])}
    actual = {item["motion"] for item in motion_records}
    if len(motion_records) != 27 or expected != actual:
        raise ValueError("visual signoff requires exactly all 27 release motions")

    reviewed_artifact_paths = [
        video_dir / "all_27_midframes_final.png",
        video_dir / "final_six_fallback_contact_sheets.png",
        *(
            video_dir / "contact_sheets_final" / f"{motion}.jpg"
            for motion in OVERRIDE_MOTIONS
        ),
        video_dir / "contact_sheets" / "6月2日-7_left_elbow_60fps.png",
    ]
    reviewed_artifacts = []
    for path in reviewed_artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing reviewed visual artifact: {path}")
        reviewed_artifacts.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
            }
        )

    total_frames = sum(int(item["video"]["frames"]) for item in motion_records)
    total_duration = sum(
        float(item["video"]["duration_seconds"]) for item in motion_records
    )
    return {
        "schema_version": "raw_smooth_v1_visual_qc_v2",
        "dataset": DATASET,
        "variant": VARIANT,
        "passed": True,
        "reviewer_type": "codex_machine_assisted_visual_inspection",
        "release": {
            "path": str(release_path.relative_to(repo_root)),
            "file_sha256": _sha256(release_path),
            "release_sha256": release["release_sha256"],
        },
        "numeric_qc": {
            "path": str(qc_path.relative_to(repo_root)),
            "file_sha256": _sha256(qc_path),
            "schema_version": qc["schema_version"],
            "clean_passed": True,
            "warning_count": 0,
            "hard_error_count": 0,
        },
        "review_contract": {
            "all_27_midframe_mosaic_reviewed": True,
            "six_override_full_motion_contact_sheets_reviewed": True,
            "six_override_real_time_videos_reviewed": True,
            "rapid_left_elbow_window_for_6月2日-7_reviewed_at_fps": 60,
            "checks": [
                "major_forehand_swing_is_present",
                "root_posture_is_upright_and_continuous",
                "right_hand_path_is_continuous",
                "rapid_joint_window_has_no_visible_pose_teleport",
                "start_and_end_pose_are_finite",
            ],
        },
        "reviewed_artifacts": reviewed_artifacts,
        "video_summary": {
            "video_count": len(motion_records),
            "all_decodable": True,
            "all_fps": 60.0,
            "all_resolution": [640, 480],
            "total_frames": total_frames,
            "total_duration_seconds": total_duration,
        },
        "motions": motion_records,
        "notes": (
            "All 27 release previews were reviewed at real-time 60 fps and in the "
            "final mid-frame mosaic. The six protected-cache overrides were also "
            "reviewed as full-motion contact sheets. Each contains a complete "
            "forehand swing with continuous upright root and right-hand motion. "
            "The 6月2日-7 left-elbow high-speed window is visually continuous and "
            "not a pose teleport. Learned rollout smoothness remains a separate "
            "Stage-1 promotion requirement."
        ),
    }


def validate_report(repo_root: Path, report_path: Path) -> dict[str, Any]:
    """Revalidate the immutable visual signoff without issuing a new one."""

    repo_root = repo_root.resolve()
    report_path = report_path.resolve()
    report = _load_json(report_path)
    release_path = repo_root / (
        "datasets/forehandClear_standard/manifests/raw_smooth_v1/"
        "release_manifest.json"
    )
    qc_path = release_path.with_name("qc_report.json")
    release = _load_json(release_path)
    qc = _load_json(qc_path)

    errors: list[str] = []
    if report.get("schema_version") != "raw_smooth_v1_visual_qc_v2":
        errors.append("visual report schema is incompatible")
    if report.get("dataset") != DATASET or report.get("variant") != VARIANT:
        errors.append("visual report dataset/variant is not canonical")
    if report.get("passed") is not True:
        errors.append("visual report is not signed as passed")
    review_contract = report.get("review_contract")
    required_review_flags = (
        "all_27_midframe_mosaic_reviewed",
        "six_override_full_motion_contact_sheets_reviewed",
        "six_override_real_time_videos_reviewed",
    )
    if not isinstance(review_contract, dict) or any(
        review_contract.get(name) is not True for name in required_review_flags
    ):
        errors.append("visual report review contract is incomplete")
    elif review_contract.get(
        "rapid_left_elbow_window_for_6月2日-7_reviewed_at_fps"
    ) != 60:
        errors.append("visual report rapid-joint review was not performed at 60 fps")

    release_binding = report.get("release")
    expected_release_relative = str(release_path.relative_to(repo_root))
    if not isinstance(release_binding, dict):
        errors.append("visual report has no release binding")
    else:
        if release_binding.get("path") != expected_release_relative:
            errors.append("visual report names a different release manifest")
        if release_binding.get("file_sha256") != _sha256(release_path):
            errors.append("visual report release manifest hash changed")
        if release_binding.get("release_sha256") != release.get("release_sha256"):
            errors.append("visual report release content identity changed")

    qc_binding = report.get("numeric_qc")
    expected_qc_relative = str(qc_path.relative_to(repo_root))
    if not isinstance(qc_binding, dict):
        errors.append("visual report has no numeric-QC binding")
    else:
        if qc_binding.get("path") != expected_qc_relative:
            errors.append("visual report names a different numeric-QC report")
        if qc_binding.get("file_sha256") != _sha256(qc_path):
            errors.append("visual report numeric-QC hash changed")
        if qc_binding.get("schema_version") != qc.get("schema_version"):
            errors.append("visual report numeric-QC schema changed")
        if (
            qc_binding.get("clean_passed") is not True
            or qc_binding.get("warning_count") != 0
            or qc_binding.get("hard_error_count") != 0
        ):
            errors.append("visual report does not bind warning-free numeric QC")
    if (
        qc.get("passed") is not True
        or qc.get("clean_passed") is not True
        or qc.get("warnings")
        or qc.get("hard_errors")
    ):
        errors.append("current numeric-QC report is not clean")

    release_motions = release.get("motions")
    report_motions = report.get("motions")
    qc_motions = qc.get("motions")
    if not isinstance(release_motions, list) or len(release_motions) != 27:
        errors.append("release manifest does not contain exactly 27 motions")
        release_motions = []
    if not isinstance(report_motions, list) or len(report_motions) != 27:
        errors.append("visual report does not contain exactly 27 motions")
        report_motions = []
    qc_by_motion = {
        str(item.get("motion")): item
        for item in qc_motions or []
        if isinstance(item, dict)
    }
    report_by_motion = {
        str(item.get("motion")): item
        for item in report_motions
        if isinstance(item, dict)
    }
    if len(report_by_motion) != len(report_motions):
        errors.append("visual report has duplicate or malformed motion records")
    for release_motion in release_motions:
        if not isinstance(release_motion, dict):
            errors.append("release manifest has a malformed motion record")
            continue
        motion = str(release_motion.get("motion"))
        record = report_by_motion.get(motion)
        numeric = qc_by_motion.get(motion)
        if record is None or numeric is None:
            errors.append(f"visual/QC motion record is missing: {motion}")
            continue
        expected_video = (
            repo_root
            / "outputs/vis/raw_smooth_v1_60fps"
            / f"raw_smooth_v1_{motion}.mp4"
        )
        if record.get("split") != release_motion.get("split"):
            errors.append(f"visual report split changed: {motion}")
        if record.get("cache_sha256") != release_motion.get("cache", {}).get(
            "sha256"
        ):
            errors.append(f"visual report cache identity changed: {motion}")
        if record.get("video_path") != str(expected_video.relative_to(repo_root)):
            errors.append(f"visual report video path changed: {motion}")
        elif not expected_video.is_file() or record.get("video_sha256") != _sha256(
            expected_video
        ):
            errors.append(f"visual report video content changed: {motion}")
        if record.get("reviewed") is not True or record.get("passed") is not True:
            errors.append(f"visual review is incomplete: {motion}")
        semantic = record.get("semantic_swing_metrics")
        if not isinstance(semantic, dict):
            errors.append(f"visual report has no semantic swing metrics: {motion}")
            continue
        for metric_name in (
            "right_hand_path_length_m",
            "max_right_hand_displacement_m",
        ):
            try:
                actual = float(semantic[metric_name])
                expected = float(numeric[metric_name])
            except (KeyError, TypeError, ValueError):
                errors.append(f"visual report has invalid {metric_name}: {motion}")
                continue
            if not math.isfinite(actual) or actual != expected:
                errors.append(f"visual report {metric_name} changed: {motion}")

    reviewed_artifacts = report.get("reviewed_artifacts")
    if not isinstance(reviewed_artifacts, list) or len(reviewed_artifacts) != 9:
        errors.append("visual report does not bind all nine reviewed artifacts")
    else:
        seen_paths: set[str] = set()
        for artifact in reviewed_artifacts:
            if not isinstance(artifact, dict) or not isinstance(
                artifact.get("path"), str
            ):
                errors.append("visual report has a malformed reviewed artifact")
                continue
            relative = str(artifact["path"])
            if relative in seen_paths:
                errors.append(f"visual report repeats reviewed artifact: {relative}")
                continue
            seen_paths.add(relative)
            path = (repo_root / relative).resolve()
            try:
                path.relative_to(repo_root)
            except ValueError:
                errors.append(f"reviewed artifact escapes repository: {relative}")
                continue
            if not path.is_file() or artifact.get("sha256") != _sha256(path):
                errors.append(f"reviewed artifact content changed: {relative}")

    passed = not errors
    return {
        "schema_version": "raw_smooth_v1_visual_qc_validation_v1",
        "passed": passed,
        "errors": errors,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "release_sha256": release.get("release_sha256"),
        "motion_count": len(report_motions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=VISUAL_REPORT_RELATIVE_PATH)
    parser.add_argument("--confirm-all-reviewed", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Revalidate an existing signed report without issuing a new signoff.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    if args.validate:
        validation = validate_report(repo_root, output)
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        if not validation["passed"]:
            raise SystemExit(2)
        return
    report = build_report(
        repo_root, confirm_all_reviewed=bool(args.confirm_all_reviewed)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "release_sha256": report["release"]["release_sha256"],
                **report["video_summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
