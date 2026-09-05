#!/usr/bin/env python3
"""Materialize the ForehandLift vertical-root-smoothed expert source release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


SCHEMA_VERSION = "forehand_lift_root_smooth_release_v2"
VARIANT = "optimized_root_smooth_v2"
SOURCE_FPS = 60.0
TARGET_MARGIN = 0.95
MAX_VERTICAL_CHANGE_M = 0.06
WINDOW_CANDIDATES = tuple(range(5, 42, 2))

TRAIN_MOTIONS = (
    "forehandLift-1",
    "forehandLift-3",
    "forehandLift-4",
    "forehandLift-5",
    "5月13日-1",
    "5月13日-2",
    "5月13日-4",
    "5月13日-5",
    "5月13日-6",
    "5月13日-8",
    "5月13日-9",
    "5月13日-10",
)
VAL_MOTIONS = (
    "forehandLift-2",
    "forehandLift-6",
    "5月13日-3",
    "5月13日-7",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vertical_rms_acceleration(values: np.ndarray, fps: float) -> float:
    vertical = np.asarray(values, dtype=np.float64).reshape(-1)
    if vertical.size < 3:
        raise ValueError("vertical root trajectory needs at least three frames")
    return float(np.sqrt(np.mean(np.diff(vertical, n=2) ** 2)) * float(fps) ** 2)


def smooth_vertical_root(
    values: np.ndarray,
    *,
    fps: float,
    target_rms_acceleration: float,
    margin: float = TARGET_MARGIN,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertical = np.asarray(values, dtype=np.float64).reshape(-1)
    current = vertical_rms_acceleration(vertical, fps)
    target = float(target_rms_acceleration) * float(margin)
    if current <= target:
        return np.array(vertical, copy=True), {
            "filter": "identity",
            "window_length": 1,
            "polyorder": None,
            "rms_acceleration_before_mps2": current,
            "rms_acceleration_after_mps2": current,
            "target_rms_acceleration_mps2": float(target_rms_acceleration),
            "target_margin": float(margin),
            "max_vertical_change_m": 0.0,
        }

    for window in WINDOW_CANDIDATES:
        if window >= vertical.size:
            break
        candidate = savgol_filter(vertical, window_length=window, polyorder=3, mode="interp")
        rms = vertical_rms_acceleration(candidate, fps)
        max_change = float(np.max(np.abs(candidate - vertical)))
        if rms <= target:
            if max_change > MAX_VERTICAL_CHANGE_M:
                raise ValueError(
                    f"root smoothing needs {max_change:.4f} m change, above "
                    f"{MAX_VERTICAL_CHANGE_M:.4f} m safety limit"
                )
            return candidate, {
                "filter": "savgol",
                "window_length": int(window),
                "polyorder": 3,
                "rms_acceleration_before_mps2": current,
                "rms_acceleration_after_mps2": rms,
                "target_rms_acceleration_mps2": float(target_rms_acceleration),
                "target_margin": float(margin),
                "max_vertical_change_m": max_change,
            }
    raise ValueError(
        f"no safe Savitzky-Golay window reached root RMS target {target:.6f} m/s^2 "
        f"from {current:.6f} m/s^2"
    )


def _load_upstream_quality(path: Path, motion: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        target = float(report["before"]["root"]["rms_accel_mps2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{motion}: upstream quality report lacks root RMS baseline") from exc
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(f"{motion}: invalid upstream root RMS baseline {target!r}")
    return report


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def materialize_release(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    source_root = root / "wham" / "optimized_wham"
    output_root = root / "temp" / VARIANT
    manifest_root = root / "manifests" / VARIANT
    expected_paths = (
        root / "temp" / VARIANT,
        root / "manifests" / VARIANT,
    )
    if (output_root.resolve(), manifest_root.resolve()) != expected_paths:
        raise ValueError("release paths must not escape the dataset root or use symlink aliases")

    motions = (*TRAIN_MOTIONS, *VAL_MOTIONS)
    if set(TRAIN_MOTIONS) & set(VAL_MOTIONS) or len(set(motions)) != 16:
        raise ValueError("ForehandLift train/validation split must contain 16 unique motions")

    records: list[dict[str, Any]] = []
    for motion in motions:
        source_path = source_root / f"{motion}.npz"
        quality_path = source_root / motion / "reference_bundle" / "quality_report.json"
        if not source_path.is_file() or not quality_path.is_file():
            raise FileNotFoundError(f"{motion}: missing source or upstream quality report")
        quality = _load_upstream_quality(quality_path, motion)
        with np.load(source_path, allow_pickle=True) as source:
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
        if "trans" not in arrays or arrays["trans"].ndim != 2 or arrays["trans"].shape[1] != 3:
            raise ValueError(f"{motion}: source trans must have shape [T,3]")
        rates = [
            float(np.asarray(arrays[field]).reshape(-1)[0])
            for field in ("mocap_framerate", "mocap_frame_rate")
            if field in arrays
        ]
        if len(rates) != 2 or any(not np.isclose(rate, SOURCE_FPS) for rate in rates):
            raise ValueError(f"{motion}: expected both source FPS fields to equal 60, got {rates}")
        original_trans = np.asarray(arrays["trans"], dtype=np.float64)
        repaired_vertical, repair = smooth_vertical_root(
            original_trans[:, 2],
            fps=SOURCE_FPS,
            target_rms_acceleration=float(quality["before"]["root"]["rms_accel_mps2"]),
        )
        repaired_trans = np.array(original_trans, copy=True)
        repaired_trans[:, 2] = repaired_vertical
        arrays["trans"] = repaired_trans.astype(np.asarray(arrays["trans"]).dtype, copy=False)
        if not np.array_equal(arrays["trans"][:, :2], np.asarray(original_trans[:, :2], dtype=arrays["trans"].dtype)):
            raise AssertionError(f"{motion}: horizontal root motion changed")

        output_path = output_root / f"{motion}.npz"
        _save_npz(output_path, arrays)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "motion": motion,
            "source_path": str(source_path.relative_to(_repo_root())),
            "source_sha256": sha256_file(source_path),
            "output_path": str(output_path.relative_to(_repo_root())),
            "output_sha256": sha256_file(output_path),
            "preserved": ["poses", "root_orientation", "horizontal_root", "frame_count", "fps"],
            "modified": "trans[:,2] only",
            "repair": repair,
            "upstream_quality_tier": quality.get("quality_tier"),
            "upstream_failed_gates": [
                str(item.get("name")) for item in quality.get("failed_gates", [])
            ],
        }
        sidecar_path = output_root / f"{motion}.source_provenance.json"
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(sidecar)

    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "train_source_list.txt").write_text(
        "".join(f"{motion}\n" for motion in TRAIN_MOTIONS), encoding="utf-8"
    )
    (manifest_root / "val_source_list.txt").write_text(
        "".join(f"{motion}\n" for motion in VAL_MOTIONS), encoding="utf-8"
    )
    (manifest_root / "all_source_list.txt").write_text(
        "".join(f"{motion}\n" for motion in motions), encoding="utf-8"
    )
    cache_prefix = f"forehandLift/muscle_trajectory/{VARIANT}"
    (manifest_root / "train_list.txt").write_text(
        "".join(f"{cache_prefix}/{motion}\n" for motion in TRAIN_MOTIONS), encoding="utf-8"
    )
    (manifest_root / "val_list.txt").write_text(
        "".join(f"{cache_prefix}/{motion}\n" for motion in VAL_MOTIONS), encoding="utf-8"
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "forehandLift",
        "variant": VARIANT,
        "source_fps": SOURCE_FPS,
        "source_count": len(records),
        "train_count": len(TRAIN_MOTIONS),
        "validation_count": len(VAL_MOTIONS),
        "repair_contract": {
            "axis": "AMASS z (vertical)",
            "filter": "adaptive shortest Savitzky-Golay window",
            "polyorder": 3,
            "target": "95% of pre-lower-body-optimization vertical RMS acceleration",
            "maximum_vertical_change_m": MAX_VERTICAL_CHANGE_M,
        },
        "motions": records,
    }
    release_path = manifest_root / "source_release.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_repo_root() / "datasets" / "forehandLift",
    )
    args = parser.parse_args()
    release = materialize_release(args.dataset_root)
    print(
        f"[OK] materialized {release['source_count']} sources in "
        f"{args.dataset_root / 'temp' / VARIANT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
