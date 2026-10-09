"""CPU-only QC and promotion metrics for physical teacher rollout shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.data.event_lookup import (
    EVENT_LOOKUP_FIELDS,
    EventReferenceLookup,
)
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.dataset import PhysicalDistillDataset
from musclemimic.distill.provenance import validate_dataset_manifest

PHYSICAL_QC_SCHEMA_VERSION = "physical_rollout_promotion_metrics_v1"
BODY_ONLY_PHYSICAL_QC_SCHEMA_VERSION = "body_only_physical_rollout_promotion_metrics_v1"
BODY_ONLY_PHASE_FREE_CONTRACT = "body_only_phase_free_rollout_contract_v1"
BODY_ONLY_SUPPORTED_CLAIMS = (
    "immutable_train_validation_rollout_integrity",
    "teacher_checkpoint_consistency",
    "motion_split_disjointness",
    "finite_physical_muscle_signals",
    "teacher_action_saturation",
)
BODY_ONLY_EXCLUDED_CLAIMS = (
    "event_alignment",
    "impact_alignment",
    "phase_conditioned_performance",
    "ready_or_recovery_phase_performance",
    "racket_or_shuttle_performance",
    "stage2_task_causal_effects",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def build_body_only_physical_rollout_metrics(
    train_source: str | Path,
    val_source: str | Path,
    *,
    teacher_checkpoint_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build the phase-free physical-rollout contract used by body-only tasks.

    This is intentionally a separate schema from the event-aligned forehand
    contract.  It proves immutable dataset, teacher, split, finite-signal, and
    saturation properties without manufacturing an event, impact, racket, or
    Stage-2 causal interpretation for actions such as ChinaJump.
    """

    train_manifest = validate_dataset_manifest(train_source)
    val_manifest = validate_dataset_manifest(val_source)
    train = PhysicalDistillDataset(train_source, split="train", require_event_fields=False)
    val = PhysicalDistillDataset(val_source, split="val", require_event_fields=False)
    _reject_event_conditioning(train, split="train")
    _reject_event_conditioning(val, split="validation")

    expected = (
        None
        if teacher_checkpoint_fingerprint is None
        else _require_sha256(
            teacher_checkpoint_fingerprint,
            field="teacher_checkpoint_fingerprint argument",
        )
    )
    train_metrics = _body_only_split_metrics(
        train,
        manifest=train_manifest,
        expected_split="train",
    )
    val_metrics = _body_only_split_metrics(
        val,
        manifest=val_manifest,
        expected_split="val",
    )
    train_teacher = _metadata_teacher_sha256(train.metadata, split="train")
    val_teacher = _metadata_teacher_sha256(val.metadata, split="validation")
    train_manifest_teacher = _manifest_teacher_sha256(train_manifest, split="train")
    val_manifest_teacher = _manifest_teacher_sha256(val_manifest, split="validation")
    checkpoint_binding = bool(
        train_teacher and train_teacher == val_teacher == train_manifest_teacher == val_manifest_teacher
    )
    if expected is not None:
        checkpoint_binding &= train_teacher == expected

    train_motions = set(train_metrics["motion_uids"])
    val_motions = set(val_metrics["motion_uids"])
    split_disjoint = bool(train_motions and val_motions and not (train_motions & val_motions))
    payload = {
        "schema_version": BODY_ONLY_PHYSICAL_QC_SCHEMA_VERSION,
        "contract": BODY_ONLY_PHASE_FREE_CONTRACT,
        "claim_scope": {
            "supported": list(BODY_ONLY_SUPPORTED_CLAIMS),
            "excluded": list(BODY_ONLY_EXCLUDED_CLAIMS),
        },
        "rollout_count": min(train_metrics["rollout_count"], val_metrics["rollout_count"]),
        "sample_count": min(train_metrics["sample_count"], val_metrics["sample_count"]),
        "finite_rate": min(train_metrics["finite_rate"], val_metrics["finite_rate"]),
        "action_saturation_fraction": max(
            train_metrics["action_saturation_fraction"],
            val_metrics["action_saturation_fraction"],
        ),
        "immutable_manifest_binding_verified": 1.0,
        "checkpoint_binding_verified": float(checkpoint_binding),
        "split_disjoint_verified": float(split_disjoint),
        "phase_free_contract_verified": 1.0,
        "teacher_checkpoint_fingerprint": train_teacher or val_teacher,
        "expected_teacher_checkpoint_fingerprint": expected,
        "train_motion_uids": sorted(train_motions),
        "validation_motion_uids": sorted(val_motions),
        "motion_uid_overlap": sorted(train_motions & val_motions),
        "train_dataset_manifest_fingerprint": train_metrics["dataset_manifest_fingerprint"],
        "validation_dataset_manifest_fingerprint": val_metrics["dataset_manifest_fingerprint"],
        "train": train_metrics,
        "validation": val_metrics,
    }
    payload["metrics_fingerprint"] = _json_sha256(payload)
    return payload


def write_body_only_physical_rollout_metrics(
    output: str | Path,
    *,
    train_source: str | Path,
    val_source: str | Path,
    teacher_checkpoint_fingerprint: str | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_body_only_physical_rollout_metrics(
        train_source,
        val_source,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def build_physical_rollout_metrics(
    train_source: str | Path,
    val_source: str | Path,
    *,
    teacher_checkpoint_fingerprint: str | None = None,
    event_reference_metrics: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if event_reference_metrics is None:
        raise ValueError("physical rollout QC requires content-bound event-reference promotion metrics")
    train = PhysicalDistillDataset(train_source, split="train", require_event_fields=True)
    val = PhysicalDistillDataset(val_source, split="val", require_event_fields=True)
    expected = (
        None
        if teacher_checkpoint_fingerprint is None
        else _require_sha256(
            teacher_checkpoint_fingerprint,
            field="teacher_checkpoint_fingerprint argument",
        )
    )
    event_evidence = _load_event_reference_evidence(event_reference_metrics)
    train_metrics = _split_metrics(
        train,
        expected_bank_fingerprint=event_evidence.get("train_event_reference_bank_fingerprint"),
    )
    val_metrics = _split_metrics(
        val,
        expected_bank_fingerprint=event_evidence.get("validation_event_reference_bank_fingerprint"),
    )
    train_teacher = _metadata_teacher_sha256(train.metadata, split="train")
    val_teacher = _metadata_teacher_sha256(val.metadata, split="validation")
    checkpoint_binding = bool(train_teacher and train_teacher == val_teacher)
    if expected:
        checkpoint_binding &= train_teacher == expected
    train_motions = set(_unique_ints(train.arrays.get("motion_uid")))
    val_motions = set(_unique_ints(val.arrays.get("motion_uid")))
    split_disjoint = bool(train_motions and val_motions and not (train_motions & val_motions))
    train_event = _metadata_event_binding(
        train.metadata,
        split="train",
        expected_bundle_fingerprints=event_evidence.get("train_reference_bundle_fingerprints", ()),
        expected_bank_fingerprint=event_evidence.get("train_event_reference_bank_fingerprint"),
    )
    val_event = _metadata_event_binding(
        val.metadata,
        split="validation",
        expected_bundle_fingerprints=event_evidence.get("validation_reference_bundle_fingerprints", ()),
        expected_bank_fingerprint=event_evidence.get("validation_event_reference_bank_fingerprint"),
    )
    event_binding = bool(
        event_evidence
        and float(event_evidence.get("artifact_binding_verified", 0.0)) == 1.0
        and float(event_evidence.get("event_bank_binding_verified", 0.0)) == 1.0
        and train_event["verified"]
        and val_event["verified"]
        and train_metrics["exact_event_reference_verified"]
        and val_metrics["exact_event_reference_verified"]
        and train_event["bank_fingerprint"] != val_event["bank_fingerprint"]
    )
    checkpoint_binding &= split_disjoint and event_binding
    payload = {
        "schema_version": PHYSICAL_QC_SCHEMA_VERSION,
        "rollout_count": min(train_metrics["rollout_count"], val_metrics["rollout_count"]),
        "sample_count": min(train_metrics["sample_count"], val_metrics["sample_count"]),
        "finite_rate": min(train_metrics["finite_rate"], val_metrics["finite_rate"]),
        "reference_alignment_rate": min(
            train_metrics["reference_alignment_rate"],
            val_metrics["reference_alignment_rate"],
        ),
        "exact_event_reference_rate": min(
            train_metrics["exact_event_reference_rate"],
            val_metrics["exact_event_reference_rate"],
        ),
        "action_saturation_fraction": max(
            train_metrics["action_saturation_fraction"],
            val_metrics["action_saturation_fraction"],
        ),
        "checkpoint_binding_verified": float(checkpoint_binding),
        "split_disjoint_verified": float(split_disjoint),
        "event_reference_binding_verified": float(event_binding),
        "event_reference_metrics_fingerprint": event_evidence.get("metrics_fingerprint"),
        "train_event_reference_binding": train_event,
        "validation_event_reference_binding": val_event,
        "teacher_checkpoint_fingerprint": train_teacher or val_teacher,
        "expected_teacher_checkpoint_fingerprint": expected,
        "train_motion_uids": sorted(train_motions),
        "validation_motion_uids": sorted(val_motions),
        "motion_uid_overlap": sorted(train_motions & val_motions),
        "train": train_metrics,
        "validation": val_metrics,
    }
    payload["metrics_fingerprint"] = _json_sha256(payload)
    return payload


def write_physical_rollout_metrics(
    output: str | Path,
    *,
    train_source: str | Path,
    val_source: str | Path,
    teacher_checkpoint_fingerprint: str | None = None,
    event_reference_metrics: str | Path | Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_physical_rollout_metrics(
        train_source,
        val_source,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
        event_reference_metrics=event_reference_metrics,
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _reject_event_conditioning(
    dataset: PhysicalDistillDataset,
    *,
    split: str,
) -> None:
    forbidden_arrays = {
        *EVENT_LOOKUP_FIELDS,
        "event_reference_frame",
        "phase_id",
        "phase_local",
        "phase_global",
        "time_to_impact_s",
        "time_from_impact_s",
        "impact_flag",
    }
    present_arrays = sorted(forbidden_arrays & set(dataset.arrays))
    if present_arrays:
        raise ValueError(f"{split} body-only phase-free rollout contains event/phase arrays {present_arrays}")
    present_metadata = sorted(key for key in dataset.metadata if str(key).startswith("event_reference_"))
    if present_metadata:
        raise ValueError(f"{split} body-only phase-free rollout contains event-reference metadata {present_metadata}")


def _body_only_split_metrics(
    dataset: PhysicalDistillDataset,
    *,
    manifest: Mapping[str, Any],
    expected_split: str,
) -> dict[str, Any]:
    arrays = dataset.arrays
    if "rollout_uid" not in arrays:
        raise ValueError("body-only physical rollout QC requires stable rollout_uid in every shard")
    if "motion_uid" not in arrays:
        raise ValueError("body-only physical rollout QC requires stable motion_uid in every shard")
    rollout_uids = np.asarray(arrays["rollout_uid"], dtype=np.int64)
    motion_uids = np.asarray(arrays["motion_uid"], dtype=np.int64)
    if np.any(rollout_uids < 0) or np.any(motion_uids < 0):
        raise ValueError("body-only physical rollout QC requires non-negative rollout_uid and motion_uid")

    manifest_motion_uids: set[int] = set()
    manifest_motion_paths: set[str] = set()
    collections = manifest.get("collections")
    if not isinstance(collections, list) or not collections:
        raise ValueError(f"{expected_split} immutable dataset manifest has no collection records")
    for collection in collections:
        contract = collection.get("contract") if isinstance(collection, Mapping) else None
        if not isinstance(contract, Mapping):
            raise ValueError(f"{expected_split} immutable dataset manifest has an invalid collection contract")
        if contract.get("split") != expected_split:
            raise ValueError(
                f"{expected_split} immutable dataset manifest contains collection for split={contract.get('split')!r}"
            )
        paths = contract.get("motion_paths")
        uids = contract.get("motion_uids")
        if not isinstance(paths, list) or not isinstance(uids, list) or len(paths) != len(uids) or not paths:
            raise ValueError(f"{expected_split} immutable dataset manifest has an incomplete motion split")
        manifest_motion_paths.update(str(path) for path in paths)
        manifest_motion_uids.update(int(uid) for uid in uids)

    observed_motion_uids = set(_unique_ints(motion_uids))
    if observed_motion_uids != manifest_motion_uids:
        raise ValueError(
            f"{expected_split} rollout motion_uid inventory differs from its immutable "
            f"manifest: rollout={sorted(observed_motion_uids)} "
            f"manifest={sorted(manifest_motion_uids)}"
        )
    finite_fields = (
        "student_obs",
        "teacher_action",
        *PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS,
    )
    finite_masks = []
    for field in finite_fields:
        value = np.asarray(arrays[field])
        finite_masks.append(np.all(np.isfinite(value).reshape(value.shape[0], -1), axis=1))
    finite_rate = float(np.mean(np.logical_and.reduce(finite_masks)))
    teacher_action = np.asarray(arrays["teacher_action"], dtype=np.float64)
    saturation = float(np.mean(np.abs(teacher_action) >= 1.0 - 1e-6))
    return {
        "split": dataset.split,
        "sample_count": int(dataset.num_samples),
        "rollout_count": int(np.unique(rollout_uids).size),
        "finite_rate": finite_rate,
        "action_saturation_fraction": saturation,
        "source": str(dataset.dataset_dir.resolve()),
        "dataset_manifest_fingerprint": _require_sha256(
            manifest.get("manifest_fingerprint"),
            field=f"{expected_split} dataset manifest_fingerprint",
        ),
        "motion_uids": sorted(observed_motion_uids),
        "motion_paths": sorted(manifest_motion_paths),
    }


def _manifest_teacher_sha256(
    manifest: Mapping[str, Any],
    *,
    split: str,
) -> str:
    teacher = manifest.get("teacher_checkpoint")
    if not isinstance(teacher, Mapping):
        raise ValueError(f"{split} immutable dataset manifest lacks teacher_checkpoint")
    return _require_sha256(
        teacher.get("sha256"),
        field=f"{split} dataset manifest teacher checkpoint sha256",
    )


def _split_metrics(
    dataset: PhysicalDistillDataset,
    *,
    expected_bank_fingerprint: Any,
) -> dict[str, Any]:
    arrays = dataset.arrays
    if "rollout_uid" not in arrays:
        raise ValueError("physical rollout QC requires stable rollout_uid in every shard")
    required_reference = {
        *EVENT_LOOKUP_FIELDS,
        "event_reference_frame",
        "traj_no",
        "subtraj_step_no",
        "motion_uid",
    }
    missing = sorted(required_reference - set(arrays))
    if missing:
        raise ValueError(f"physical rollout QC lacks event-reference fields {missing}")
    physical_fields = tuple(PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS)
    finite_masks = []
    for field in physical_fields:
        value = np.asarray(arrays[field])
        finite_masks.append(np.all(np.isfinite(value).reshape(value.shape[0], -1), axis=1))
    finite_rate = float(np.mean(np.logical_and.reduce(finite_masks)))
    phase_id = np.asarray(arrays["phase_id"], dtype=np.int32)
    phase_local = np.asarray(arrays["phase_local"], dtype=np.float64)
    time_to = np.asarray(arrays["time_to_impact_s"], dtype=np.float64)
    time_from = np.asarray(arrays["time_from_impact_s"], dtype=np.float64)
    impact = np.asarray(arrays["impact_flag"], dtype=bool)
    confidence = np.asarray(arrays["reference_confidence"], dtype=np.float64)
    aligned = (
        (phase_id >= 0)
        & (phase_id <= 5)
        & np.isfinite(phase_local)
        & (phase_local >= 0.0)
        & (phase_local <= 1.0)
        & np.isfinite(time_to)
        & np.isfinite(time_from)
        & np.isclose(time_to, -time_from, atol=1e-5, rtol=0.0)
        & (~impact | (phase_id == 3))
        & np.isfinite(confidence)
        & (confidence >= 0.0)
        & (confidence <= 1.0)
    )
    exact_event = _exact_event_reference_audit(
        dataset,
        expected_bank_fingerprint=expected_bank_fingerprint,
    )
    aligned &= exact_event["sample_match_mask"]
    teacher_action = np.asarray(arrays["teacher_action"], dtype=np.float64)
    saturation = float(np.mean(np.abs(teacher_action) >= 1.0 - 1e-6))
    shard_paths = tuple(Path(path) for path in dataset.shard_paths)
    return {
        "split": dataset.split,
        "sample_count": int(dataset.num_samples),
        "rollout_count": int(np.unique(np.asarray(arrays["rollout_uid"], dtype=np.int64)).size),
        "finite_rate": finite_rate,
        "reference_alignment_rate": float(np.mean(aligned)),
        "exact_event_reference_rate": exact_event["match_rate"],
        "exact_event_reference_verified": exact_event["verified"],
        "exact_event_reference_audit": exact_event["report"],
        "action_saturation_fraction": saturation,
        "source": str(dataset.dataset_dir.resolve()),
        "source_shards": [path.name for path in shard_paths],
        "source_content_fingerprint": _json_sha256(
            {
                "metadata": dataset.metadata,
                "shards": {path.name: _file_sha256(path) for path in shard_paths},
            }
        ),
        "event_reference_bank_fingerprint": dataset.metadata.get("event_reference_bank_fingerprint"),
    }


def _exact_event_reference_audit(
    dataset: PhysicalDistillDataset,
    *,
    expected_bank_fingerprint: Any,
) -> dict[str, Any]:
    """Recompute every persisted event label from immutable rollout coordinates."""

    split = "validation" if dataset.split == "val" else dataset.split
    manifest_value = str(dataset.metadata.get("event_reference_bank_manifest", "")).strip()
    if not manifest_value:
        raise ValueError(f"{split} metadata lacks event_reference_bank_manifest for exact replay")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = dataset.dataset_dir / manifest_path
    lookup = EventReferenceLookup.from_manifest(manifest_path.resolve())
    metadata_control_dt = float(dataset.metadata.get("event_reference_control_dt", np.nan))
    lookup.validate_control_dt(metadata_control_dt)
    expected_bank = _require_sha256(
        expected_bank_fingerprint,
        field=f"expected {split} event reference bank fingerprint",
    )
    metadata_bank = _require_sha256(
        dataset.metadata.get("event_reference_bank_fingerprint"),
        field=f"{split} metadata.event_reference_bank_fingerprint",
    )
    entries = sorted(lookup.entries, key=lambda entry: entry.traj_no)
    bank_inventory = {
        "bundle_fingerprints": [entry.reference_bundle_content_fingerprint for entry in entries],
        "motion_uids": [int(entry.motion_uid) for entry in entries],
        "motion_paths": [entry.motion_path for entry in entries],
        "control_dt": metadata_control_dt,
    }
    metadata_inventory = {
        "bundle_fingerprints": dataset.metadata.get("event_reference_bundle_fingerprints"),
        "motion_uids": dataset.metadata.get("event_reference_bank_motion_uids"),
        "motion_paths": dataset.metadata.get("event_reference_bank_motion_paths"),
        "control_dt": dataset.metadata.get("event_reference_control_dt"),
    }
    inventory_matches = bank_inventory == metadata_inventory

    arrays = dataset.arrays
    expected = lookup.lookup_batch(
        traj_no=np.asarray(arrays["traj_no"]),
        subtraj_step_no=np.asarray(arrays["subtraj_step_no"]),
        motion_uid=np.asarray(arrays["motion_uid"]),
    )
    fields = (*EVENT_LOOKUP_FIELDS, "event_reference_frame")
    sample_match = np.ones(dataset.num_samples, dtype=bool)
    field_rates: dict[str, float] = {}
    for field in fields:
        observed = np.asarray(arrays[field])
        reference = np.asarray(expected[field])
        if observed.shape != reference.shape:
            raise ValueError(
                f"{split} exact event field {field} shape differs from bank replay: "
                f"{observed.shape} != {reference.shape}"
            )
        if observed.dtype.kind in {"b", "i", "u"} or reference.dtype.kind in {
            "b",
            "i",
            "u",
        }:
            matched = observed == reference
        else:
            matched = np.isclose(observed, reference, rtol=0.0, atol=1e-6)
        if matched.ndim > 1:
            matched = np.all(matched.reshape(matched.shape[0], -1), axis=1)
        matched = np.asarray(matched, dtype=bool)
        sample_match &= matched
        field_rates[field] = float(np.mean(matched))

    bank_matches = lookup.fingerprint == metadata_bank == expected_bank
    rate = float(np.mean(sample_match))
    verified = bool(bank_matches and inventory_matches and np.all(sample_match))
    return {
        "sample_match_mask": sample_match,
        "match_rate": rate,
        "verified": verified,
        "report": {
            "schema_version": "exact_event_reference_replay_audit_v1",
            "verified": verified,
            "bank_fingerprint": lookup.fingerprint,
            "metadata_bank_fingerprint": metadata_bank,
            "expected_bank_fingerprint": expected_bank,
            "bank_fingerprint_matches": bank_matches,
            "ordered_bank_inventory_matches_metadata": inventory_matches,
            "control_dt": metadata_control_dt,
            "sample_count": int(dataset.num_samples),
            "matched_sample_count": int(np.count_nonzero(sample_match)),
            "field_match_rates": field_rates,
        },
    }


def _unique_ints(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in np.unique(np.asarray(value, dtype=np.int64)).tolist()]


def _load_event_reference_evidence(
    source: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source is None:
        return {}
    payload = dict(source) if isinstance(source, Mapping) else load_json_strict(source)
    if not isinstance(payload, dict):
        raise ValueError("event-reference metrics must contain a JSON object")
    if payload.get("schema_version") != "event_reference_promotion_metrics_v1":
        raise ValueError("event-reference metrics schema is unsupported")
    supplied = _require_sha256(
        payload.get("metrics_fingerprint"),
        field="event-reference metrics_fingerprint",
    )
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("metrics_fingerprint", None)
    if supplied != _json_sha256(fingerprint_payload):
        raise ValueError("event-reference metrics_fingerprint mismatch")
    for field in (
        "train_reference_bundle_fingerprints",
        "validation_reference_bundle_fingerprints",
    ):
        values = payload.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"event-reference metrics lacks non-empty {field}")
        digests = [_require_sha256(value, field=f"{field} entry") for value in values]
        if len(set(digests)) != len(digests):
            raise ValueError(f"event-reference metrics {field} contains duplicates")
    return payload


def _metadata_event_binding(
    metadata: Mapping[str, Any],
    *,
    split: str,
    expected_bundle_fingerprints: Sequence[Any],
    expected_bank_fingerprint: Any,
) -> dict[str, Any]:
    bank_fingerprint = _require_sha256(
        metadata.get("event_reference_bank_fingerprint"),
        field=f"{split} metadata.event_reference_bank_fingerprint",
    )
    actual_raw = metadata.get("event_reference_bundle_fingerprints")
    if not isinstance(actual_raw, list) or not actual_raw:
        raise ValueError(f"{split} metadata lacks event reference bundle inventory")
    actual = [_require_sha256(value, field=f"{split} event reference bundle fingerprint") for value in actual_raw]
    expected = [
        _require_sha256(value, field=f"expected {split} event reference fingerprint")
        for value in expected_bundle_fingerprints
    ]
    expected_bank = _require_sha256(
        expected_bank_fingerprint,
        field=f"expected {split} event reference bank fingerprint",
    )
    motion_uids = metadata.get("event_reference_bank_motion_uids")
    motion_paths = metadata.get("event_reference_bank_motion_paths")
    identity_complete = bool(
        isinstance(motion_uids, list)
        and isinstance(motion_paths, list)
        and len(motion_uids) == len(actual)
        and len(motion_paths) == len(actual)
        and len({int(value) for value in motion_uids}) == len(actual)
        and len({str(value) for value in motion_paths}) == len(actual)
    )
    return {
        "verified": bool(expected and actual == expected and bank_fingerprint == expected_bank and identity_complete),
        "bank_fingerprint": bank_fingerprint,
        "bundle_fingerprints": actual,
        "motion_uids": [] if not isinstance(motion_uids, list) else motion_uids,
        "motion_paths": [] if not isinstance(motion_paths, list) else motion_paths,
        "ordered_inventory_matches_event_qc": bool(expected and actual == expected),
        "bank_fingerprint_matches_event_qc": bank_fingerprint == expected_bank,
        "motion_identity_complete": identity_complete,
    }


def _metadata_teacher_sha256(metadata: Mapping[str, Any], *, split: str) -> str:
    digest = _require_sha256(
        metadata.get("teacher_checkpoint_fingerprint"),
        field=f"{split} metadata.teacher_checkpoint_fingerprint",
    )
    content = metadata.get("teacher_checkpoint_content")
    if not isinstance(content, Mapping):
        raise ValueError(f"{split} metadata lacks teacher_checkpoint_content audit record")
    if content.get("schema_version") != "checkpoint_content_fingerprint_v1":
        raise ValueError(f"{split} teacher_checkpoint_content schema is unsupported")
    content_digest = _require_sha256(
        content.get("sha256"),
        field=f"{split} metadata.teacher_checkpoint_content.sha256",
    )
    if content_digest != digest:
        raise ValueError(f"{split} compact teacher fingerprint differs from its content record")
    files = content.get("files")
    if (
        not isinstance(files, list)
        or not files
        or int(content.get("num_files", -1)) != len(files)
        or int(content.get("num_bytes", -1)) < 0
    ):
        raise ValueError(f"{split} teacher_checkpoint_content inventory is incomplete")
    return digest


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-checkpoint-fingerprint", default=None)
    parser.add_argument(
        "--qc-contract",
        choices=("event-aligned", "body-only-phase-free"),
        default="event-aligned",
        help=("Select the existing event-aligned contract or the separately versioned body-only, phase-free contract."),
    )
    parser.add_argument("--event-reference-metrics", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.qc_contract == "body-only-phase-free":
        if args.event_reference_metrics is not None:
            parser.error("--event-reference-metrics is forbidden for the body-only-phase-free QC contract")
        path = write_body_only_physical_rollout_metrics(
            args.output,
            train_source=args.train,
            val_source=args.val,
            teacher_checkpoint_fingerprint=args.teacher_checkpoint_fingerprint,
        )
    else:
        if args.event_reference_metrics is None:
            parser.error("the event-aligned QC contract requires --event-reference-metrics")
        path = write_physical_rollout_metrics(
            args.output,
            train_source=args.train,
            val_source=args.val,
            teacher_checkpoint_fingerprint=args.teacher_checkpoint_fingerprint,
            event_reference_metrics=args.event_reference_metrics,
        )
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
