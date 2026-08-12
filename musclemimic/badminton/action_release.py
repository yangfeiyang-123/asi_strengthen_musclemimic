"""Validate the immutable trajectory release owned by an action registry row.

The historical forehand-clear release has a bespoke, byte-for-byte rebuild
validator.  ForehandLift uses a different release schema and ChinaJump's
accepted ``optimized_qc10`` split predates the JSON release format.  This
module provides one read-only entry point without pretending those provenance
formats are interchangeable.

Every successful report binds the declared ordered split and the bytes of each
source/cache file.  The ChinaJump report also records that its upstream review
is a legacy QC decision document rather than a structured release manifest;
downstream papers must preserve that evidence limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import ActionSpec, action_choices, resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "musclemimic_action_release_validation_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _read_split(path: Path) -> tuple[str, ...]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            rows.append(Path(value.removesuffix(".npz")).name)
    return tuple(rows)


def _file_inventory(spec: ActionSpec) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    source_root = spec.dataset_root / spec.source_namespace
    cache_root = spec.dataset_root / spec.cache_namespace
    for motion in spec.all_motions:
        source = source_root / f"{motion}.npz"
        cache = cache_root / f"{motion}.npz"
        missing = [str(path) for path in (source, cache) if not path.is_file()]
        if missing:
            errors.append(f"{motion}: missing registered source/cache files: {missing}")
            continue
        if source.is_symlink() or cache.is_symlink():
            errors.append(f"{motion}: registered source/cache must not be symlinks")
            continue
        rows.append(
            {
                "motion": motion,
                "split": "train" if motion in spec.train_motions else "validation",
                "source_path": str(source.relative_to(REPO_ROOT)),
                "source_sha256": _sha256(source),
                "cache_path": str(cache.relative_to(REPO_ROOT)),
                "cache_sha256": _sha256(cache),
            }
        )
    return rows, errors


def _verify_bound_file(binding: Any, *, label: str, errors: list[str]) -> None:
    if not isinstance(binding, Mapping):
        errors.append(f"{label} binding is missing")
        return
    raw_path = str(binding.get("path", "")).strip()
    expected = str(binding.get("sha256", "")).strip()
    path = (REPO_ROOT / raw_path).resolve() if raw_path else Path()
    if not raw_path or not path.is_file():
        errors.append(f"{label} file is missing: {raw_path!r}")
    elif _sha256(path) != expected:
        errors.append(f"{label} content hash differs from release manifest")
    if binding.get("passed") is False:
        errors.append(f"{label} did not pass")


def _validate_forehand_clear(spec: ActionSpec, release_path: Path) -> tuple[dict[str, Any], list[str]]:
    from musclemimic.badminton.scripts.data_release import validate_release_manifest
    from musclemimic.badminton.scripts.finalize_raw_smooth_visual_qc import (
        validate_report as validate_visual_qc_report,
    )

    result = validate_release_manifest(spec.dataset_root, release_path)
    errors = [str(value) for value in result.get("errors", ())]
    visual_path = release_path.with_name("visual_qc_report.json")
    visual = validate_visual_qc_report(REPO_ROOT, visual_path)
    errors.extend(str(value) for value in visual.get("errors", ()))
    return {
        "upstream_schema": result.get("schema_version"),
        "upstream_release_sha256": result.get("release_sha256"),
        "review_evidence_kind": "structured_release_and_visual_qc",
        "visual_qc_path": str(visual_path.relative_to(REPO_ROOT)),
        "visual_qc_sha256": visual.get("report_sha256"),
        "formal_release_manifest": True,
    }, errors


def _validate_forehand_lift(spec: ActionSpec, release_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"ForehandLift release manifest is unreadable: {exc}"]
    if not isinstance(payload, Mapping):
        return {}, ["ForehandLift release manifest must be a JSON object"]
    expected = {
        "dataset": spec.action_id,
        "variant": spec.data_variant,
        "train_motions": list(spec.train_motions),
        "validation_motions": list(spec.val_motions),
        "source_fps": 60,
        "cache_fps": 100,
        "passed": True,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(f"ForehandLift release {field} differs from action registry")
    records = payload.get("motions")
    by_motion = {
        str(row.get("motion")): row
        for row in records
        if isinstance(row, Mapping) and str(row.get("motion", ""))
    } if isinstance(records, list) else {}
    if tuple(by_motion) != spec.all_motions:
        errors.append("ForehandLift release motion order differs from action registry")
    inventory, inventory_errors = _file_inventory(spec)
    errors.extend(inventory_errors)
    for row in inventory:
        declared = by_motion.get(row["motion"], {})
        for field in ("split", "source_sha256", "cache_sha256"):
            if declared.get(field) != row[field]:
                errors.append(f"{row['motion']}: release {field} differs from current registered file")
    _verify_bound_file(payload.get("source_release"), label="ForehandLift source release", errors=errors)
    _verify_bound_file(payload.get("numeric_qc"), label="ForehandLift numeric QC", errors=errors)
    _verify_bound_file(payload.get("visual_qc"), label="ForehandLift visual QC", errors=errors)
    return {
        "upstream_schema": payload.get("schema_version"),
        "upstream_release_sha256": _sha256(release_path),
        "review_evidence_kind": "structured_release_and_visual_qc",
        "visual_qc_path": str(payload.get("visual_qc", {}).get("path", "")),
        "visual_qc_sha256": str(payload.get("visual_qc", {}).get("sha256", "")),
        "formal_release_manifest": True,
    }, errors


def _validate_chinajump(spec: ActionSpec, evidence_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not evidence_path.is_file():
        return {}, [f"ChinaJump QC decision document is missing: {evidence_path}"]
    train_path = spec.dataset_root / "manifests" / "optimized_qc10_train.txt"
    val_path = spec.dataset_root / "manifests" / "optimized_qc10_val.txt"
    try:
        train = _read_split(train_path)
        val = _read_split(val_path)
    except OSError as exc:
        return {}, [f"ChinaJump split manifest is unreadable: {exc}"]
    if train != spec.train_motions:
        errors.append("ChinaJump train split differs from action registry")
    if val != spec.val_motions:
        errors.append("ChinaJump validation split differs from action registry")
    return {
        "upstream_schema": "chinajump_optimized_qc10_legacy_decision_v1",
        "upstream_release_sha256": _sha256(evidence_path),
        "review_evidence_kind": "legacy_qc_decision_document",
        "formal_release_manifest": False,
        "evidence_limitations": [
            "no_structured_json_release_manifest",
            "no_content_bound_structured_visual_qc_report",
        ],
        "split_manifests": {
            "train": {"path": str(train_path.relative_to(REPO_ROOT)), "sha256": _sha256(train_path)},
            "validation": {"path": str(val_path.relative_to(REPO_ROOT)), "sha256": _sha256(val_path)},
        },
    }, errors


def validate_action_release(action: str | ActionSpec) -> dict[str, Any]:
    """Return a content-bound, action-aware release validation report."""

    spec = action if isinstance(action, ActionSpec) else resolve(action)
    release_path = (REPO_ROOT / spec.release_manifest).resolve()
    inventory, inventory_errors = _file_inventory(spec)
    if spec.slug == "forehand_clear":
        upstream, errors = _validate_forehand_clear(spec, release_path)
    elif spec.slug == "forehand_lift":
        upstream, errors = _validate_forehand_lift(spec, release_path)
    elif spec.slug == "chinajump":
        upstream, errors = _validate_chinajump(spec, release_path)
    else:  # registry construction currently prevents this, but keep the gate explicit.
        upstream, errors = {}, [f"no release validator is registered for {spec.slug!r}"]
    errors.extend(inventory_errors)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "data_variant": spec.data_variant,
        "release_evidence_path": str(release_path),
        "release_evidence_sha256": _sha256(release_path) if release_path.is_file() else None,
        "train_motions": list(spec.train_motions),
        "validation_motions": list(spec.val_motions),
        "file_inventory": inventory,
        **upstream,
        "errors": errors,
        "passed": not errors,
    }
    unsigned = dict(payload)
    payload["release_binding_sha256"] = _fingerprint(unsigned)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=action_choices(), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = validate_action_release(args.action)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0 if report["passed"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
