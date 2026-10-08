"""Fail-closed promotion metrics for held-out event/racket reference bundles.

This module intentionally consumes manifests rather than tracking caches.  The
event-reference gate therefore certifies the measured/fused source artifacts
before resampling or policy collection can obscure their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.data.event_lookup import EventReferenceLookup
from musclemimic.badminton.data.reference_bundle import load_reference_bundle
from musclemimic.badminton.data_qc import inspect_event_racket_bundle, validate_session_split
from musclemimic.badminton.json_contract import load_json_strict

EVENT_REFERENCE_QC_SCHEMA_VERSION = "event_reference_promotion_metrics_v1"


def build_event_reference_metrics(
    train_manifests: Sequence[str | Path],
    val_manifests: Sequence[str | Path],
    *,
    min_racket_confidence: float = 0.0,
    train_event_bank: str | Path | None = None,
    val_event_bank: str | Path | None = None,
) -> dict[str, Any]:
    """Validate event bundles and return metrics consumed by ``event_reference_v2``.

    ``reference_count`` is the smaller split size so a large training bank can
    never hide an undersized held-out bank.  Uncertainty metrics are worst-case
    bounds reported by the source/annotation pipeline, not estimates inferred
    from confidence scores.
    """

    train_paths = _require_manifest_paths(train_manifests, split="train")
    val_paths = _require_manifest_paths(val_manifests, split="validation")
    train_rows = _inspect_split(
        train_paths,
        split="train",
        min_racket_confidence=min_racket_confidence,
    )
    val_rows = _inspect_split(
        val_paths,
        split="validation",
        min_racket_confidence=min_racket_confidence,
    )
    all_rows = [*train_rows, *val_rows]
    split_report = validate_session_split(train_rows, val_rows)

    trial_keys = [
        (
            str(row["provenance"]["subject_id"]),
            str(row["provenance"]["session_id"]),
            str(row["provenance"]["trial_id"]),
        )
        for row in all_rows
    ]
    fingerprints = [str(row["content_fingerprint"]) for row in all_rows]
    unique_trials = len(set(trial_keys)) == len(trial_keys)
    unique_content = len(set(fingerprints)) == len(fingerprints)
    no_legacy = all(not bool(row["provenance"]["legacy_fallback"]) for row in all_rows)
    reports_passed = [bool(row["qc"]["passed"]) for row in all_rows]
    manual_reviews_passed = all(
        str(row["provenance"]["manual_review_status"]).strip().lower() in {"passed", "approved"} for row in all_rows
    )
    train_source_videos = {str(row["provenance"]["source_video_id"]).strip() for row in train_rows}
    validation_source_videos = {str(row["provenance"]["source_video_id"]).strip() for row in val_rows}
    source_video_overlap = sorted(train_source_videos & validation_source_videos)
    source_videos_disjoint = not source_video_overlap
    event_valid_rate = float(np.mean(reports_passed))
    finite_rates = [float(row["racket_state_finite_rate"]) for row in all_rows]
    binding_verified = bool(
        split_report["passed"]
        and unique_trials
        and unique_content
        and no_legacy
        and all(reports_passed)
        and manual_reviews_passed
        and source_videos_disjoint
    )
    train_fingerprints = [str(row["content_fingerprint"]) for row in train_rows]
    validation_fingerprints = [str(row["content_fingerprint"]) for row in val_rows]
    train_bank = _bind_event_bank(
        train_event_bank,
        expected_bundle_fingerprints=train_fingerprints,
        split="train",
    )
    validation_bank = _bind_event_bank(
        val_event_bank,
        expected_bundle_fingerprints=validation_fingerprints,
        split="validation",
    )
    train_control_dt = train_bank["control_dt"]
    validation_control_dt = validation_bank["control_dt"]
    control_dt_binding = bool(
        train_control_dt is not None
        and validation_control_dt is not None
        and np.isclose(train_control_dt, validation_control_dt, atol=1e-8, rtol=0.0)
    )
    bank_binding = bool(
        train_bank["verified"]
        and validation_bank["verified"]
        and train_bank["bank_fingerprint"] != validation_bank["bank_fingerprint"]
        and control_dt_binding
    )
    binding_verified &= bank_binding

    payload: dict[str, Any] = {
        "schema_version": EVENT_REFERENCE_QC_SCHEMA_VERSION,
        "reference_count": min(len(train_rows), len(val_rows)),
        "train_reference_count": len(train_rows),
        "validation_reference_count": len(val_rows),
        "event_valid_rate": event_valid_rate,
        "impact_position_uncertainty_m": max(
            float(row["provenance"]["impact_position_uncertainty_m"]) for row in all_rows
        ),
        "impact_timing_uncertainty_s": max(float(row["provenance"]["impact_timing_uncertainty_s"]) for row in all_rows),
        "racket_state_finite_rate": min(finite_rates),
        "artifact_binding_verified": float(binding_verified),
        "split_disjoint_verified": float(bool(split_report["passed"])),
        "unique_trial_identity_verified": float(unique_trials),
        "unique_content_verified": float(unique_content),
        "legacy_fallback_absent": float(no_legacy),
        "manual_review_passed": float(manual_reviews_passed),
        "source_video_split_disjoint_verified": float(source_videos_disjoint),
        "source_video_id_overlap": source_video_overlap,
        "event_bank_binding_verified": float(bank_binding),
        "control_dt_binding_verified": float(control_dt_binding),
        "event_reference_control_dt": train_control_dt if control_dt_binding else None,
        "train_event_reference_bank": train_bank,
        "validation_event_reference_bank": validation_bank,
        "train_event_reference_bank_fingerprint": train_bank["bank_fingerprint"],
        "validation_event_reference_bank_fingerprint": validation_bank["bank_fingerprint"],
        "train_reference_bundle_fingerprints": train_fingerprints,
        "validation_reference_bundle_fingerprints": validation_fingerprints,
        "train_reference_set_fingerprint": _json_sha256(
            {"split": "train", "ordered_bundle_fingerprints": train_fingerprints}
        ),
        "validation_reference_set_fingerprint": _json_sha256(
            {
                "split": "validation",
                "ordered_bundle_fingerprints": validation_fingerprints,
            }
        ),
        "session_split_qc": split_report,
        "train": train_rows,
        "validation": val_rows,
    }
    payload["metrics_fingerprint"] = _json_sha256(payload)
    return payload


def write_event_reference_metrics(
    output: str | Path,
    *,
    train_manifests: Sequence[str | Path],
    val_manifests: Sequence[str | Path],
    min_racket_confidence: float = 0.0,
    train_event_bank: str | Path | None = None,
    val_event_bank: str | Path | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_event_reference_metrics(
        train_manifests,
        val_manifests,
        min_racket_confidence=min_racket_confidence,
        train_event_bank=train_event_bank,
        val_event_bank=val_event_bank,
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _require_manifest_paths(values: Sequence[str | Path], *, split: str) -> tuple[Path, ...]:
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    if not paths:
        raise ValueError(f"{split} event-reference manifest list must be non-empty")
    duplicates = sorted(str(path) for path in paths if paths.count(path) > 1)
    if duplicates:
        raise ValueError(f"{split} event-reference manifest list has duplicates: {duplicates}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{split} event-reference manifests do not exist: {missing}")
    return paths


def _inspect_split(
    paths: Sequence[Path],
    *,
    split: str,
    min_racket_confidence: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        bundle = load_reference_bundle(path)
        if bundle.provenance is None or bundle.content_fingerprint is None:
            raise ValueError(f"{split} bundle is not event-reference v2: {path}")
        qc = inspect_event_racket_bundle(
            bundle,
            min_racket_confidence=float(min_racket_confidence),
        )
        racket = bundle.racket
        if racket is None:
            raise ValueError(f"{split} bundle has no racket state: {path}")
        racket_arrays = (
            racket.position_world,
            racket.quaternion_world,
            racket.linear_velocity_world,
            racket.angular_velocity_world,
            racket.stringbed_normal_world,
            racket.stringbed_center_world,
            racket.confidence,
        )
        finite_count = sum(int(np.count_nonzero(np.isfinite(value))) for value in racket_arrays)
        element_count = sum(int(value.size) for value in racket_arrays)
        rows.append(
            {
                "split": split,
                "manifest": str(path),
                "content_fingerprint": bundle.content_fingerprint,
                "provenance": dict(bundle.provenance),
                "racket_reference_source": racket.source,
                "racket_state_finite_rate": float(finite_count / element_count),
                "qc": qc,
            }
        )
    return rows


def _bind_event_bank(
    source: str | Path | None,
    *,
    expected_bundle_fingerprints: Sequence[str],
    split: str,
) -> dict[str, Any]:
    if source is None:
        return {
            "verified": False,
            "split": split,
            "bank_fingerprint": None,
            "bundle_fingerprints": [],
            "motion_uids": [],
            "motion_paths": [],
            "control_dt": None,
            "reference_fps": [],
            "effective_ref_strides": [],
            "reason": "event bank manifest was not supplied",
        }
    lookup = EventReferenceLookup.from_manifest(source)
    entries = sorted(lookup.entries, key=lambda entry: entry.traj_no)
    expected_trajectories = list(range(len(entries)))
    actual_trajectories = [entry.traj_no for entry in entries]
    actual_fingerprints = [entry.reference_bundle_content_fingerprint for entry in entries]
    control_dts = [float(entry.control_dt) for entry in entries]
    timing_consistent = all(np.isclose(value, control_dts[0], atol=1e-8, rtol=0.0) for value in control_dts)
    return {
        "verified": bool(
            actual_trajectories == expected_trajectories
            and actual_fingerprints == list(expected_bundle_fingerprints)
            and timing_consistent
        ),
        "split": split,
        "bank_fingerprint": lookup.fingerprint,
        "bundle_fingerprints": actual_fingerprints,
        "motion_uids": [int(entry.motion_uid) for entry in entries],
        "motion_paths": [entry.motion_path for entry in entries],
        "control_dt": control_dts[0] if timing_consistent else None,
        "reference_fps": [float(entry.reference_fps) for entry in entries],
        "effective_ref_strides": [float(entry.effective_ref_stride) for entry in entries],
        "trajectory_indices": actual_trajectories,
        "reason": None,
    }


def _load_manifest_list(path: str | Path) -> tuple[Path, ...]:
    source = Path(path).expanduser().resolve()
    payload = load_json_strict(source)
    if isinstance(payload, Mapping):
        payload = payload.get("manifests")
    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        raise ValueError(f"manifest list must be a JSON string array or {{'manifests': [...]}}: {source}")
    return tuple(
        (source.parent / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        for value in payload
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate event/racket v2 bundles and emit promotion metrics.")
    parser.add_argument("--train-manifests-json", type=Path, required=True)
    parser.add_argument("--val-manifests-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-event-bank", type=Path, required=True)
    parser.add_argument("--val-event-bank", type=Path, required=True)
    parser.add_argument("--min-racket-confidence", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train = _load_manifest_list(args.train_manifests_json)
    validation = _load_manifest_list(args.val_manifests_json)
    payload = build_event_reference_metrics(
        train,
        validation,
        min_racket_confidence=args.min_racket_confidence,
        train_event_bank=args.train_event_bank,
        val_event_bank=args.val_event_bank,
    )
    if args.dry_run:
        print(
            json.dumps(
                {"dry_run": True, "output": str(args.output), "metrics": payload},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
