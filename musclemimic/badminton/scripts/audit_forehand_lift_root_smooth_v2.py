#!/usr/bin/env python3
"""Audit and fingerprint the repaired ForehandLift expert cache release."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from musclemimic.badminton.data_qc import _inspect_motion
from musclemimic.badminton.scripts.eval_lower_body_fidelity import evaluate
from musclemimic.badminton.scripts.prepare_forehand_lift_root_smooth_v2 import (
    TRAIN_MOTIONS,
    VAL_MOTIONS,
    VARIANT,
    vertical_rms_acceleration,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    source_root = dataset_root / "temp" / VARIANT
    cache_root = dataset_root / "muscle_trajectory" / VARIANT
    old_cache_root = dataset_root / "muscle_trajectory" / "optimized"
    errors: list[str] = []
    rows: list[dict] = []
    cache_hashes: dict[str, str] = {}

    for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS):
        source = source_root / f"{motion}.npz"
        provenance_path = source_root / f"{motion}.source_provenance.json"
        cache = cache_root / f"{motion}.npz"
        analysis_path = cache_root / f"{motion}_analysis.npz"
        old_cache = old_cache_root / f"{motion}.npz"
        missing = [path for path in (source, provenance_path, cache, analysis_path, old_cache) if not path.is_file()]
        if missing:
            errors.append(f"{motion}: missing {', '.join(str(path) for path in missing)}")
            continue

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("output_sha256") != _sha256(source):
            errors.append(f"{motion}: source provenance hash mismatch")

        row, hard_errors = _inspect_motion(motion, source, cache)
        errors.extend(hard_errors)
        if row.warnings:
            errors.extend(f"{motion}: {warning}" for warning in row.warnings)

        lower_body = evaluate(old_cache, cache)
        if lower_body["verdict"]["success"] is not True:
            failed = [name for name, passed in lower_body["verdict"]["checks"].items() if not passed]
            errors.append(f"{motion}: lower-body gates failed: {failed}")

        with np.load(old_cache, allow_pickle=True) as old_data, np.load(cache, allow_pickle=True) as new_data:
            old_fps = float(np.asarray(old_data["frequency"]).reshape(-1)[0])
            new_fps = float(np.asarray(new_data["frequency"]).reshape(-1)[0])
            old_rms = vertical_rms_acceleration(old_data["qpos"][:, 2], old_fps)
            new_rms = vertical_rms_acceleration(new_data["qpos"][:, 2], new_fps)
        repair_filter = str(provenance["repair"]["filter"])
        if repair_filter == "savgol" and new_rms > old_rms * 0.95:
            errors.append(
                f"{motion}: repaired cache root RMS {new_rms:.6f} did not improve "
                f"legacy {old_rms:.6f} by at least 5%"
            )

        with np.load(analysis_path, allow_pickle=True) as analysis:
            grounding_mode = str(np.asarray(analysis["grounding_mode"]).item())
            deepest_after = float(np.asarray(analysis["grounding_deepest_penetration_after_m"]).item())
            global_offset = float(np.asarray(analysis["grounding_global_vertical_offset_m"]).item())
        if grounding_mode != "global":
            errors.append(f"{motion}: grounding mode is {grounding_mode!r}, expected 'global'")
        if deepest_after < -1e-7:
            errors.append(f"{motion}: residual floor penetration {deepest_after:.9f} m")

        cache_hash = _sha256(cache)
        cache_hashes[motion] = cache_hash
        rows.append(
            {
                "motion": motion,
                "split": "train" if motion in TRAIN_MOTIONS else "validation",
                "source_sha256": _sha256(source),
                "cache_sha256": cache_hash,
                "analysis_sha256": _sha256(analysis_path),
                "source_repair": provenance["repair"],
                "legacy_cache_root_rms_acceleration_mps2": old_rms,
                "repaired_cache_root_rms_acceleration_mps2": new_rms,
                "root_rms_ratio_repaired_over_legacy": new_rms / old_rms,
                "grounding_mode": grounding_mode,
                "grounding_global_vertical_offset_m": global_offset,
                "grounding_deepest_penetration_after_m": deepest_after,
                "cache_qc": asdict(row),
                "lower_body_fidelity": lower_body,
            }
        )

    duplicate_hashes = sorted({value for value in cache_hashes.values() if list(cache_hashes.values()).count(value) > 1})
    if duplicate_hashes:
        errors.append(f"duplicate cache hashes: {duplicate_hashes}")
    if len(rows) != 16:
        errors.append(f"audited {len(rows)} motions, expected 16")

    return {
        "schema_version": "forehand_lift_root_smooth_v2_numeric_qc_v1",
        "dataset_root": str(dataset_root),
        "variant": VARIANT,
        "train_motions": list(TRAIN_MOTIONS),
        "validation_motions": list(VAL_MOTIONS),
        "checks": {
            "source_and_cache_finite_contract": True,
            "no_cache_qc_warnings": True,
            "lower_body_absolute_fidelity": True,
            "global_grounding_zero_penetration": True,
            "smoothed_sources_improve_cache_root_rms_by_at_least_5_percent": True,
            "unique_cache_content": True,
        },
        "errors": errors,
        "motions": rows,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_repo_root() / "datasets" / "forehandLift",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "datasets" / "forehandLift" / "manifests" / VARIANT / "numeric_qc_report.json",
    )
    args = parser.parse_args()
    report = audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
