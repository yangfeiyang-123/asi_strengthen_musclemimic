"""Fail-closed provenance contract for primitive-only synergy sources.

The early Stage-1 controller is allowed to consume a basis fitted only from
primitive motions.  A basis artifact alone cannot establish that fact, so this
module stores and verifies the exact source inventory and the identities of all
contracts that affect the fitted physical-excitation matrix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.motion_identity import (
    normalize_relative_motion_path,
    stable_motion_uid,
)

PRIMITIVE_SOURCE_MANIFEST_SCHEMA_VERSION = "primitive_synergy_source_manifest_v2"
RANK_SELECTION_RULE_SCHEMA_VERSION = "early_control_rank_selection_rule_v1"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "primitive_only",
        "contains_target_skill_rollouts",
        "target_skill_id",
        "excluded_target_motions",
        "primitive_task_ids",
        "primitive_source_kinds",
        "primitive_trial_ids",
        "train_motion_uids",
        "validation_motion_uids",
        "source_dataset_fingerprint",
        "source_checkpoint_fingerprints",
        "source_checkpoint_contents",
        "primitive_required_phase_ids",
        "primitive_phase_schema_fingerprints",
        "model_hash",
        "actuator_schema_hash",
        "control_range_hash",
        "transform_ctrlrange_schema_hash",
        "preprocessing_fingerprint",
        "phase_weight_fingerprint",
        "NMF_seeds",
        "rank_selection_rule",
        "manifest_fingerprint",
    }
)
_RANK_SELECTION_RULE = {
    "schema_version": RANK_SELECTION_RULE_SCHEMA_VERSION,
    "strategy": "lowest_rank_passing_all_gates",
    "requires_all_vaf_gates": True,
    "requires_all_stability_gates": True,
    "fallback_allowed": False,
}


@dataclass(frozen=True)
class PrimitiveSourceManifest:
    """A validated primitive-only source manifest and its content identity."""

    manifest: dict[str, Any]
    fingerprint: str
    path: Path


def canonical_rank_selection_rule() -> dict[str, Any]:
    """Return the only rank-selection rule eligible for early control."""

    return copy.deepcopy(_RANK_SELECTION_RULE)


def build_primitive_source_manifest(
    *,
    target_skill_id: str,
    excluded_target_motion_paths: Sequence[str],
    primitive_task_ids: Sequence[str],
    primitive_source_kinds: Mapping[str, str],
    primitive_trial_ids: Sequence[str],
    train_motion_uids: Sequence[int],
    validation_motion_uids: Sequence[int],
    source_checkpoint_fingerprints: Mapping[str, str],
    source_checkpoint_contents: Mapping[str, Mapping[str, Any]],
    primitive_required_phase_ids: Mapping[str, Sequence[int]],
    primitive_phase_schema_fingerprints: Mapping[str, str],
    source_dataset_fingerprint: str,
    model_hash: str,
    actuator_schema_hash: str,
    control_range_hash: str,
    transform_ctrlrange_schema_hash: str,
    preprocessing_fingerprint: str,
    phase_weight_fingerprint: str,
    nmf_seeds: Sequence[int],
    rank_selection_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and self-fingerprint a primitive-only source manifest.

    The two scope booleans are intentionally not caller configurable.  This
    builder cannot be used to bless target-skill rollouts as primitive data.
    """

    tasks = list(primitive_task_ids)
    checkpoints = _validate_task_fingerprint_inventory(
        source_checkpoint_fingerprints,
        tasks,
        field="source_checkpoint_fingerprints",
    )
    required_phases = _validate_required_phase_inventory(
        primitive_required_phase_ids,
        tasks,
        field="primitive_required_phase_ids",
        accept_sequences=True,
    )
    phase_schemas = _validate_task_fingerprint_inventory(
        primitive_phase_schema_fingerprints,
        tasks,
        field="primitive_phase_schema_fingerprints",
    )
    checkpoint_contents = _validate_checkpoint_content_inventory(
        source_checkpoint_contents,
        checkpoints=checkpoints,
        primitive_tasks=tasks,
        field="source_checkpoint_contents",
    )
    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SOURCE_MANIFEST_SCHEMA_VERSION,
        "primitive_only": True,
        "contains_target_skill_rollouts": False,
        "target_skill_id": target_skill_id,
        "excluded_target_motions": _target_motion_inventory(excluded_target_motion_paths),
        "primitive_task_ids": tasks,
        "primitive_source_kinds": dict(primitive_source_kinds),
        "primitive_trial_ids": list(primitive_trial_ids),
        "train_motion_uids": list(train_motion_uids),
        "validation_motion_uids": list(validation_motion_uids),
        "source_checkpoint_fingerprints": checkpoints,
        "source_checkpoint_contents": checkpoint_contents,
        "primitive_required_phase_ids": required_phases,
        "primitive_phase_schema_fingerprints": phase_schemas,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "model_hash": model_hash,
        "actuator_schema_hash": actuator_schema_hash,
        "control_range_hash": control_range_hash,
        "transform_ctrlrange_schema_hash": transform_ctrlrange_schema_hash,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "phase_weight_fingerprint": phase_weight_fingerprint,
        "NMF_seeds": list(nmf_seeds),
        "rank_selection_rule": dict(
            canonical_rank_selection_rule() if rank_selection_rule is None else rank_selection_rule
        ),
    }
    payload["manifest_fingerprint"] = primitive_source_manifest_fingerprint(payload)
    return validate_primitive_source_manifest(payload)


def save_primitive_source_manifest(
    path: str | Path,
    *,
    target_skill_id: str,
    excluded_target_motion_paths: Sequence[str],
    primitive_task_ids: Sequence[str],
    primitive_source_kinds: Mapping[str, str],
    primitive_trial_ids: Sequence[str],
    train_motion_uids: Sequence[int],
    validation_motion_uids: Sequence[int],
    source_checkpoint_fingerprints: Mapping[str, str],
    source_checkpoint_contents: Mapping[str, Mapping[str, Any]],
    primitive_required_phase_ids: Mapping[str, Sequence[int]],
    primitive_phase_schema_fingerprints: Mapping[str, str],
    source_dataset_fingerprint: str,
    model_hash: str,
    actuator_schema_hash: str,
    control_range_hash: str,
    transform_ctrlrange_schema_hash: str,
    preprocessing_fingerprint: str,
    phase_weight_fingerprint: str,
    nmf_seeds: Sequence[int],
    rank_selection_rule: Mapping[str, Any] | None = None,
) -> PrimitiveSourceManifest:
    """Write a canonical source manifest and reload it through strict JSON."""

    manifest_path = _manifest_path(path)
    payload = build_primitive_source_manifest(
        target_skill_id=target_skill_id,
        excluded_target_motion_paths=excluded_target_motion_paths,
        primitive_task_ids=primitive_task_ids,
        primitive_source_kinds=primitive_source_kinds,
        primitive_trial_ids=primitive_trial_ids,
        train_motion_uids=train_motion_uids,
        validation_motion_uids=validation_motion_uids,
        source_checkpoint_fingerprints=source_checkpoint_fingerprints,
        source_checkpoint_contents=source_checkpoint_contents,
        primitive_required_phase_ids=primitive_required_phase_ids,
        primitive_phase_schema_fingerprints=primitive_phase_schema_fingerprints,
        source_dataset_fingerprint=source_dataset_fingerprint,
        model_hash=model_hash,
        actuator_schema_hash=actuator_schema_hash,
        control_range_hash=control_range_hash,
        transform_ctrlrange_schema_hash=transform_ctrlrange_schema_hash,
        preprocessing_fingerprint=preprocessing_fingerprint,
        phase_weight_fingerprint=phase_weight_fingerprint,
        nmf_seeds=nmf_seeds,
        rank_selection_rule=rank_selection_rule,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return load_primitive_source_manifest(
        manifest_path,
        expected_fingerprint=str(payload["manifest_fingerprint"]),
    )


def load_primitive_source_manifest(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> PrimitiveSourceManifest:
    """Load a manifest with duplicate-key and optional identity verification."""

    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"primitive source manifest does not exist: {manifest_path}")
    payload = load_json_strict(manifest_path)
    validated = validate_primitive_source_manifest(
        payload,
        expected_fingerprint=expected_fingerprint,
    )
    return PrimitiveSourceManifest(
        manifest=validated,
        fingerprint=str(validated["manifest_fingerprint"]),
        path=manifest_path,
    )


def validate_primitive_source_manifest(
    payload: Mapping[str, Any],
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate the complete primitive-source contract without fallback."""

    if not isinstance(payload, Mapping):
        raise ValueError("primitive source manifest must be a JSON object")
    if any(not isinstance(field, str) for field in payload):
        raise ValueError("primitive source manifest field names must be strings")
    fields = set(payload)
    missing = sorted(_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_FIELDS)
    if missing:
        raise ValueError(f"primitive source manifest is missing fields: {missing}")
    if unknown:
        raise ValueError(f"primitive source manifest has unknown fields: {unknown}")
    if payload.get("schema_version") != PRIMITIVE_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported primitive source manifest schema")
    if payload.get("primitive_only") is not True:
        raise ValueError("primitive source manifest requires primitive_only=true")
    if payload.get("contains_target_skill_rollouts") is not False:
        raise ValueError("primitive source manifest requires contains_target_skill_rollouts=false")

    primitive_tasks = _require_unique_nonempty_strings(
        payload["primitive_task_ids"],
        "primitive_task_ids",
    )
    target_skill_id = payload["target_skill_id"]
    if not isinstance(target_skill_id, str) or not target_skill_id.strip():
        raise ValueError("target_skill_id must be a non-empty string")
    if target_skill_id.strip().casefold() in {task.strip().casefold() for task in primitive_tasks}:
        raise ValueError("primitive_task_ids must exclude target_skill_id")
    source_kinds = payload["primitive_source_kinds"]
    if not isinstance(source_kinds, Mapping) or set(source_kinds) != set(primitive_tasks):
        raise ValueError("primitive_source_kinds keys must exactly match primitive_task_ids")
    if any(value != "primitive" for value in source_kinds.values()):
        raise ValueError("every primitive_source_kinds value must be 'primitive'")
    _require_unique_nonempty_strings(payload["primitive_trial_ids"], "primitive_trial_ids")
    train_uids = _require_motion_uids(payload["train_motion_uids"], "train_motion_uids")
    validation_uids = _require_motion_uids(
        payload["validation_motion_uids"],
        "validation_motion_uids",
    )
    overlap = train_uids & validation_uids
    if overlap:
        formatted = sorted(repr(value) for value in overlap)
        raise ValueError(f"primitive train/validation motion_uids overlap: {formatted}")
    excluded_target_uids = {
        int(item["motion_uid"]) for item in _validate_target_motion_inventory(payload["excluded_target_motions"])
    }
    target_overlap = (train_uids | validation_uids) & excluded_target_uids
    if target_overlap:
        formatted = sorted(repr(value) for value in target_overlap)
        raise ValueError(f"primitive source motion_uids overlap excluded target motion_uids: {formatted}")

    checkpoints = payload["source_checkpoint_fingerprints"]
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("source_checkpoint_fingerprints must be a non-empty JSON object")
    if set(checkpoints) != set(primitive_tasks):
        raise ValueError("source_checkpoint_fingerprints keys must exactly match primitive_task_ids")
    for source_id, fingerprint in checkpoints.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source checkpoint identifiers must be non-empty strings")
        _require_sha256(fingerprint, f"source_checkpoint_fingerprints[{source_id!r}]")
    _validate_checkpoint_content_inventory(
        payload["source_checkpoint_contents"],
        checkpoints=checkpoints,
        primitive_tasks=primitive_tasks,
        field="source_checkpoint_contents",
    )
    _validate_required_phase_inventory(
        payload["primitive_required_phase_ids"],
        primitive_tasks,
        field="primitive_required_phase_ids",
        accept_sequences=False,
    )
    _validate_task_fingerprint_inventory(
        payload["primitive_phase_schema_fingerprints"],
        primitive_tasks,
        field="primitive_phase_schema_fingerprints",
    )

    for field in (
        "source_dataset_fingerprint",
        "model_hash",
        "actuator_schema_hash",
        "control_range_hash",
        "transform_ctrlrange_schema_hash",
        "preprocessing_fingerprint",
        "phase_weight_fingerprint",
    ):
        _require_sha256(payload[field], field)

    seeds = payload["NMF_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("NMF_seeds must contain at least two distinct non-negative integers")
    _validate_rank_selection_rule(payload["rank_selection_rule"])

    recorded_fingerprint = _require_sha256(
        payload["manifest_fingerprint"],
        "manifest_fingerprint",
    )
    actual_fingerprint = primitive_source_manifest_fingerprint(payload)
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError("primitive source manifest_fingerprint mismatch")
    if expected_fingerprint is not None:
        expected = _require_sha256(expected_fingerprint, "expected_fingerprint")
        if actual_fingerprint != expected:
            raise ValueError(
                "primitive source manifest differs from expected_fingerprint: "
                f"expected={expected} actual={actual_fingerprint}"
            )

    return copy.deepcopy(dict(payload))


def primitive_source_manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a manifest, excluding its self hash."""

    canonical = {str(key): value for key, value in payload.items() if key != "manifest_fingerprint"}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_primitive_source_manifest_from_splits(
    output_path: str | Path,
    *,
    train_source: str | Path,
    validation_source: str | Path,
    target_skill_id: str,
    excluded_target_motion_paths: Sequence[str],
    source_checkpoint_fingerprints: Mapping[str, str],
    fit_config: Any,
    model_hash: str | None = None,
) -> PrimitiveSourceManifest:
    """Derive a source manifest from compliant primitive train/val shards."""

    # Lazy imports avoid a module cycle: fit imports the manifest loader for
    # runtime validation, while this optional builder reuses fit fingerprints.
    from musclemimic.distill.action_schema import actuator_schema_hash
    from musclemimic.synergy.fit import (
        load_synergy_split,
        synergy_phase_weight_fingerprint,
        synergy_preprocessing_fingerprint,
    )
    from musclemimic.synergy.schema import ctrlrange_schema_hash

    train = load_synergy_split(train_source, split="train")
    validation = load_synergy_split(validation_source, split="val")
    if train.muscle_names != validation.muscle_names:
        raise ValueError("primitive train/validation actuator names differ")
    if set(train.source_files) & set(validation.source_files):
        raise ValueError("primitive train/validation source shard files overlap")
    if train.motion_ids is None or validation.motion_ids is None:
        raise ValueError("primitive source manifest requires motion_uid/traj_no in both splits")
    if train.motion_id_field != "motion_uid" or validation.motion_id_field != "motion_uid":
        raise ValueError("primitive source manifest requires stable motion_uid, never local traj_no")

    task_ids, trial_ids = _primitive_sample_inventory(train.arrays, validation.arrays)
    checkpoints = _validate_task_fingerprint_inventory(
        source_checkpoint_fingerprints,
        task_ids,
        field="source_checkpoint_fingerprints",
    )
    checkpoint_contents = _matching_split_metadata_contract(
        train.metadata,
        validation.metadata,
        field="source_checkpoint_contents",
    )
    checkpoint_contents = _validate_checkpoint_content_inventory(
        checkpoint_contents,
        checkpoints=checkpoints,
        primitive_tasks=task_ids,
        field="source_checkpoint_contents",
    )
    required_phases = _matching_split_metadata_contract(
        train.metadata,
        validation.metadata,
        field="primitive_required_phase_ids",
    )
    required_phases = _validate_required_phase_inventory(
        required_phases,
        task_ids,
        field="primitive_required_phase_ids",
        accept_sequences=False,
    )
    phase_schema_fingerprints = _matching_split_metadata_contract(
        train.metadata,
        validation.metadata,
        field="primitive_phase_schema_fingerprints",
    )
    phase_schema_fingerprints = _validate_task_fingerprint_inventory(
        phase_schema_fingerprints,
        task_ids,
        field="primitive_phase_schema_fingerprints",
    )
    for split, metadata in (("train", train.metadata), ("validation", validation.metadata)):
        if metadata.get("source_checkpoint_fingerprints") != checkpoints:
            raise ValueError(f"primitive {split} metadata checkpoint inventory differs from CLI input")
    _validate_required_phase_coverage(
        train.arrays,
        required_phases=required_phases,
        split="train",
    )
    _validate_required_phase_coverage(
        validation.arrays,
        required_phases=required_phases,
        split="validation",
    )

    metadata_model_hashes = {
        str(metadata.get("model_hash", metadata.get("model_fingerprint", "")))
        for metadata in (train.metadata, validation.metadata)
    }
    if len(metadata_model_hashes) != 1 or "" in metadata_model_hashes:
        raise ValueError("primitive train/validation metadata require one identical model hash")
    metadata_model_hash = next(iter(metadata_model_hashes))
    resolved_model_hash = metadata_model_hash if model_hash is None else str(model_hash)
    if resolved_model_hash != metadata_model_hash:
        raise ValueError("explicit model hash differs from primitive shard metadata")

    source_control_hashes = {
        str(metadata.get("ctrlrange_schema_hash", "")) for metadata in (train.metadata, validation.metadata)
    }
    if len(source_control_hashes) != 1 or "" in source_control_hashes:
        raise ValueError("primitive train/validation metadata require one source ctrlrange schema hash")
    train_ctrlrange = np.asarray(train.metadata.get("actuator_ctrlrange"), dtype=np.float64)
    validation_ctrlrange = np.asarray(
        validation.metadata.get("actuator_ctrlrange"),
        dtype=np.float64,
    )
    expected_shape = (len(train.muscle_names), 2)
    if (
        train_ctrlrange.shape != expected_shape
        or validation_ctrlrange.shape != expected_shape
        or not np.array_equal(train_ctrlrange, validation_ctrlrange)
    ):
        raise ValueError("primitive train/validation actuator ctrlrange differs")

    split_provenance = {
        "train": train.provenance(),
        "validation": validation.provenance(),
    }
    dataset_fingerprint = _json_sha256(split_provenance)
    return save_primitive_source_manifest(
        output_path,
        target_skill_id=target_skill_id,
        excluded_target_motion_paths=excluded_target_motion_paths,
        primitive_task_ids=task_ids,
        primitive_source_kinds=dict.fromkeys(task_ids, "primitive"),
        primitive_trial_ids=trial_ids,
        train_motion_uids=sorted(int(value) for value in np.unique(train.motion_ids)),
        validation_motion_uids=sorted(int(value) for value in np.unique(validation.motion_ids)),
        source_checkpoint_fingerprints=checkpoints,
        source_checkpoint_contents=checkpoint_contents,
        primitive_required_phase_ids=required_phases,
        primitive_phase_schema_fingerprints=phase_schema_fingerprints,
        source_dataset_fingerprint=dataset_fingerprint,
        model_hash=resolved_model_hash,
        actuator_schema_hash=actuator_schema_hash(train.muscle_names),
        control_range_hash=next(iter(source_control_hashes)),
        transform_ctrlrange_schema_hash=ctrlrange_schema_hash(
            train.muscle_names,
            train_ctrlrange,
        ),
        preprocessing_fingerprint=synergy_preprocessing_fingerprint(fit_config),
        phase_weight_fingerprint=synergy_phase_weight_fingerprint(fit_config),
        nmf_seeds=tuple(int(value) for value in fit_config.seeds),
    )


def _primitive_sample_inventory(
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    required = {
        "phase_id",
        "motion_uid",
        "task_id",
        "trial_id",
        "source_kind",
        "success",
        "quality_weight",
    }
    split_tasks: dict[str, set[str]] = {}
    split_trials: dict[str, set[str]] = {}
    trial_to_task: dict[str, str] = {}
    trial_to_motion: dict[str, int] = {}
    for split, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"primitive {split} shards lack source/QC fields: {missing}")
        count = int(np.asarray(arrays["phase_id"]).shape[0])
        fields = {name: np.asarray(arrays[name]) for name in required - {"phase_id"}}
        if any(value.shape != (count,) for value in fields.values()):
            raise ValueError(f"primitive {split} source/QC fields must have shape [{count}]")
        for label in ("task_id", "trial_id", "source_kind"):
            if fields[label].dtype.kind not in {"U", "S"}:
                raise ValueError(f"primitive {split} {label} must use a string dtype")
        motion_uid = fields["motion_uid"]
        if np.issubdtype(motion_uid.dtype, np.bool_) or not np.issubdtype(
            motion_uid.dtype,
            np.integer,
        ):
            raise ValueError(f"primitive {split} motion_uid must use an integer dtype")
        if any(str(value) != "primitive" for value in fields["source_kind"].tolist()):
            raise ValueError(f"primitive {split} shards contain non-primitive source_kind")
        success = np.asarray(fields["success"], dtype=np.float64)
        quality = np.asarray(fields["quality_weight"], dtype=np.float64)
        if not np.all(success == 1.0):
            raise ValueError(f"primitive {split} shards contain unsuccessful samples")
        if not np.all(np.isfinite(quality)) or np.any(quality <= 0.0):
            raise ValueError(f"primitive {split} quality_weight must be finite and positive")
        task_values = np.asarray(
            [str(value) for value in fields["task_id"].tolist()],
            dtype=object,
        )
        trial_values = np.asarray(
            [str(value) for value in fields["trial_id"].tolist()],
            dtype=object,
        )
        split_tasks[split] = set(task_values.tolist())
        split_trials[split] = set(trial_values.tolist())
        minimum_trials = 2 if split == "train" else 1
        for task in split_tasks[split]:
            task_trials = set(trial_values[task_values == task].tolist())
            if len(task_trials) < minimum_trials:
                raise ValueError(
                    f"primitive {split} task {task!r} requires at least {minimum_trials} distinct trial(s)"
                )
            for trial in task_trials:
                previous = trial_to_task.setdefault(trial, task)
                if previous != task:
                    raise ValueError("primitive trial_id cannot belong to multiple tasks")
                trial_motions = {int(value) for value in motion_uid[trial_values == trial].tolist()}
                if len(trial_motions) != 1:
                    raise ValueError("primitive trial_id must bind exactly one motion_uid")
                motion = next(iter(trial_motions))
                previous_motion = trial_to_motion.setdefault(trial, motion)
                if previous_motion != motion:
                    raise ValueError("primitive trial_id motion binding changed across shards")
    if split_tasks["train"] != split_tasks["validation"]:
        raise ValueError("primitive train/validation must cover the same complete task inventory")
    if split_trials["train"] & split_trials["validation"]:
        raise ValueError("primitive train/validation trial_id inventories overlap")
    tasks = split_tasks["train"]
    trials = split_trials["train"] | split_trials["validation"]
    if not tasks or not trials or any(not value.strip() for value in tasks | trials):
        raise ValueError("primitive task/trial inventory must be non-empty")
    return sorted(tasks), sorted(trials)


def _matching_split_metadata_contract(
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
    *,
    field: str,
) -> Any:
    train_value = train_metadata.get(field)
    validation_value = validation_metadata.get(field)
    if train_value is None or validation_value is None:
        raise ValueError(f"primitive train/validation metadata require {field}")
    if _json_sha256(train_value) != _json_sha256(validation_value):
        raise ValueError(f"primitive train/validation metadata {field} differ")
    return copy.deepcopy(train_value)


def _validate_required_phase_coverage(
    arrays: Mapping[str, np.ndarray],
    *,
    required_phases: Mapping[str, Sequence[int]],
    split: str,
) -> None:
    tasks = np.asarray(arrays["task_id"])
    phases = np.asarray(arrays["phase_id"])
    if tasks.shape != phases.shape:
        raise ValueError(f"primitive {split} task_id and phase_id shapes differ")
    if np.issubdtype(phases.dtype, np.bool_) or not np.issubdtype(
        phases.dtype,
        np.integer,
    ):
        raise ValueError(f"primitive {split} phase_id must use an integer dtype")
    task_strings = np.asarray([str(value) for value in tasks.tolist()], dtype=object)
    for task, task_required_phases in required_phases.items():
        observed = {int(value) for value in phases[task_strings == task].tolist()}
        missing = sorted(set(task_required_phases) - observed)
        if missing:
            raise ValueError(f"primitive {split} task {task!r} is missing required phase_ids: {missing}")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-skill-id", required=True)
    parser.add_argument(
        "--excluded-target-motion-path",
        action="append",
        required=True,
        dest="excluded_target_motion_paths",
        help="repeat one normalized target motion path to exclude by stable motion UID",
    )
    parser.add_argument("--source-checkpoints-json", required=True)
    parser.add_argument("--model-hash", default=None)
    parser.add_argument("--normalization", choices=["channel_max", "channel_l2", "none"], default="channel_max")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-8)
    parser.add_argument("--phase-weights-json", default=None)
    parser.add_argument("--nmf-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checkpoints = load_json_strict(args.source_checkpoints_json)
    if not isinstance(checkpoints, Mapping):
        raise ValueError("source checkpoints JSON must contain a task-to-fingerprint object")
    phase_weights = None
    if args.phase_weights_json:
        raw_phase_weights = load_json_strict(args.phase_weights_json)
        if not isinstance(raw_phase_weights, Mapping):
            raise ValueError("phase weights JSON must contain an object")
        phase_weights = {int(key): float(value) for key, value in raw_phase_weights.items()}
    from musclemimic.synergy.fit import SynergyFitConfig

    fit_config = SynergyFitConfig(
        seeds=tuple(args.nmf_seeds),
        normalization=args.normalization,
        near_zero_threshold=args.near_zero_threshold,
        phase_weights=phase_weights,
    )
    source = save_primitive_source_manifest_from_splits(
        args.output,
        train_source=args.train,
        validation_source=args.val,
        target_skill_id=args.target_skill_id,
        excluded_target_motion_paths=args.excluded_target_motion_paths,
        source_checkpoint_fingerprints=checkpoints,
        fit_config=fit_config,
        model_hash=args.model_hash,
    )
    print(
        json.dumps(
            {"source_manifest": str(source.path.resolve()), "fingerprint": source.fingerprint},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _manifest_path(path: str | Path) -> Path:
    supplied = Path(path)
    return supplied if supplied.suffix == ".json" else supplied / SOURCE_MANIFEST_FILENAME


def _target_motion_inventory(paths: Sequence[str]) -> list[dict[str, Any]]:
    normalized = [normalize_relative_motion_path(str(path)) for path in paths]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("excluded target motion paths must be non-empty and unique")
    return [{"path": path, "motion_uid": stable_motion_uid(path)} for path in normalized]


def _validate_target_motion_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("excluded_target_motions must be a non-empty JSON array")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "motion_uid"}:
            raise ValueError("excluded_target_motions entries require exactly path and motion_uid")
        path = normalize_relative_motion_path(str(item["path"]))
        if item["path"] != path:
            raise ValueError("excluded target motion paths must already be normalized")
        motion_uid = item["motion_uid"]
        if type(motion_uid) is not int or motion_uid != stable_motion_uid(path):
            raise ValueError("excluded target motion_uid differs from its stable path identity")
        result.append({"path": path, "motion_uid": motion_uid})
    paths = [item["path"] for item in result]
    uids = [item["motion_uid"] for item in result]
    if len(paths) != len(set(paths)) or len(uids) != len(set(uids)):
        raise ValueError("excluded target motion inventory contains duplicates/collisions")
    return result


def _validate_required_phase_inventory(
    value: Any,
    primitive_tasks: Sequence[str],
    *,
    field: str,
    accept_sequences: bool,
) -> dict[str, list[int]]:
    if not isinstance(value, Mapping) or set(value) != set(primitive_tasks):
        raise ValueError(f"{field} keys must exactly match primitive_task_ids")
    result: dict[str, list[int]] = {}
    for task in primitive_tasks:
        phases = value[task]
        valid_container = isinstance(phases, list) or (
            accept_sequences and isinstance(phases, Sequence) and not isinstance(phases, str | bytes | bytearray)
        )
        if not valid_container or not phases:
            raise ValueError(f"{field}[{task!r}] must be a non-empty integer array")
        phase_ids = list(phases)
        if any(type(phase) is not int or phase < 0 for phase in phase_ids):
            raise ValueError(f"{field}[{task!r}] entries must be non-negative integers")
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError(f"{field}[{task!r}] must not contain duplicates")
        result[task] = sorted(phase_ids)
        if not accept_sequences and phase_ids != result[task]:
            raise ValueError(f"{field}[{task!r}] must be sorted")
    return result


def _validate_task_fingerprint_inventory(
    value: Any,
    primitive_tasks: Sequence[str],
    *,
    field: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(primitive_tasks):
        raise ValueError(f"{field} keys must exactly match primitive_task_ids")
    return {task: _require_sha256(value[task], f"{field}[{task!r}]") for task in primitive_tasks}


def _validate_checkpoint_content_inventory(
    value: Any,
    *,
    checkpoints: Mapping[str, Any],
    primitive_tasks: Sequence[str],
    field: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(primitive_tasks):
        raise ValueError(f"{field} keys must exactly match primitive_task_ids")
    result: dict[str, dict[str, Any]] = {}
    required_record_fields = {
        "schema_version",
        "supplied_path",
        "resolved_path",
        "sha256",
        "num_files",
        "num_bytes",
        "files",
    }
    required_file_fields = {"path", "sha256", "num_bytes"}
    for task in primitive_tasks:
        record = value[task]
        if not isinstance(record, Mapping) or set(record) != required_record_fields:
            raise ValueError(f"{field}[{task!r}] must be an exact checkpoint content audit record")
        if record.get("schema_version") != "checkpoint_content_fingerprint_v1":
            raise ValueError(f"{field}[{task!r}] checkpoint content schema is unsupported")
        for path_field in ("supplied_path", "resolved_path"):
            path_value = record.get(path_field)
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError(f"{field}[{task!r}].{path_field} must be non-empty")
        content_sha = _require_sha256(
            record.get("sha256"),
            f"{field}[{task!r}].sha256",
        )
        if content_sha != checkpoints.get(task):
            raise ValueError(f"{field}[{task!r}].sha256 differs from source_checkpoint_fingerprints")
        files = record.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{field}[{task!r}].files must be non-empty")
        num_files = record.get("num_files")
        num_bytes = record.get("num_bytes")
        if type(num_files) is not int or num_files != len(files):
            raise ValueError(f"{field}[{task!r}].num_files differs from files")
        if type(num_bytes) is not int or num_bytes < 0:
            raise ValueError(f"{field}[{task!r}].num_bytes must be non-negative")
        seen_paths: set[str] = set()
        summed_bytes = 0
        canonical_files: list[dict[str, Any]] = []
        for index, item in enumerate(files):
            if not isinstance(item, Mapping) or set(item) != required_file_fields:
                raise ValueError(f"{field}[{task!r}].files[{index}] must be an exact file audit record")
            path = item.get("path")
            if not isinstance(path, str) or not path.strip() or path in seen_paths:
                raise ValueError(f"{field}[{task!r}] file paths must be non-empty and unique")
            seen_paths.add(path)
            item_bytes = item.get("num_bytes")
            if type(item_bytes) is not int or item_bytes < 0:
                raise ValueError(f"{field}[{task!r}].files[{index}].num_bytes must be non-negative")
            summed_bytes += item_bytes
            canonical_files.append(
                {
                    "path": path,
                    "sha256": _require_sha256(
                        item.get("sha256"),
                        f"{field}[{task!r}].files[{index}].sha256",
                    ),
                    "num_bytes": item_bytes,
                }
            )
        if summed_bytes != num_bytes:
            raise ValueError(f"{field}[{task!r}].num_bytes differs from file inventory")
        result[task] = {
            "schema_version": "checkpoint_content_fingerprint_v1",
            "supplied_path": record["supplied_path"],
            "resolved_path": record["resolved_path"],
            "sha256": content_sha,
            "num_files": num_files,
            "num_bytes": num_bytes,
            "files": canonical_files,
        }
    return result


def _require_unique_nonempty_strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _require_motion_uids(value: Any, field: str) -> set[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON array")
    canonical: list[int] = []
    for item in value:
        if type(item) is not int or item < 0 or item > np.iinfo(np.int64).max:
            raise ValueError(f"{field} entries must be non-negative int64 integers")
        canonical.append(item)
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{field} must not contain duplicates")
    return set(canonical)


def _validate_rank_selection_rule(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("rank_selection_rule must be a JSON object")
    if dict(value) != _RANK_SELECTION_RULE:
        raise ValueError(
            "rank_selection_rule must select the lowest rank passing every VAF and stability gate without fallback"
        )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 fingerprint")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
