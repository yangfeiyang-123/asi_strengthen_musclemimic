"""Content-addressed provenance for distillation datasets and checkpoints.

Production distillation is intentionally stricter than the legacy NPZ loader:
every collection is staged outside the visible shard set, then committed to an
immutable inventory.  A resume is accepted only when the inventory on disk is
byte-for-byte identical to ``dataset_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.motion_identity import normalize_motion_path, stable_motion_uid

DATASET_MANIFEST = "dataset_manifest.json"
DATASET_MANIFEST_SCHEMA = "distill_dataset_manifest_v2"
TEACHER_PROMOTION_BINDING_SCHEMA = "stage2_teacher_promotion_binding_v1"
STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA = "stage1_teacher_promotion_binding_v1"
TEST_ONLY_TEACHER_PROMOTION_SCHEMA = "test_only_unpromoted_teacher_v1"
DEFAULT_TEACHER_PROMOTION_STAGE = "stage2"
STAGE1_BODY_ONLY_TEACHER_ROLE = "body_only"
VERIFIED_STAGE1_PROMOTION_EVIDENCE = "verified_stage1_promotion_v1"
VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE = "verified_stage1_peasd_promotion_v1"
VERIFIED_STAGE2_PROMOTION_EVIDENCE = "verified_stage2_promotion_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256_text(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return text


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_stage1_peasd_reference_promotion(
    binding: Any,
    *,
    expected_promotion: str | Path | None = None,
    expected_tube: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the PEASD promotion/tube binding captured by a collection.

    The compact record lives inside the immutable collection request.  Paths
    alone are never trusted: the promotion, its source graph, and (when
    supplied) the currently selected tube are all revalidated before the
    compact record is compared byte-for-byte.
    """

    if not isinstance(binding, Mapping):
        raise ValueError("distill collection lacks Stage-1 PEASD reference promotion")
    path = Path(str(binding.get("path", ""))).expanduser().resolve(strict=True)
    if expected_promotion is not None and path != Path(expected_promotion).expanduser().resolve(
        strict=True
    ):
        raise ValueError("distill collection uses a different Stage-1 PEASD promotion")
    from musclemimic.badminton.stage1_peasd_gate import (
        validate_stage1_peasd_teacher_promotion,
    )

    promotion = validate_stage1_peasd_teacher_promotion(
        path,
        expected_tube=expected_tube,
    )
    rebuilt = {
        "path": str(path),
        "content_sha256": file_sha256(path),
        "binding_sha256": promotion["binding_sha256"],
        "emg_reference_binding": promotion["emg_reference_binding"],
    }
    if _jsonable(dict(binding)) != rebuilt:
        raise ValueError(
            "distill collection Stage-1 PEASD promotion/tube binding changed"
        )
    return rebuilt


def _checkpoint_descends_from_stage1_peasd_reference(
    checkpoint: Any,
    reference: Mapping[str, Any],
) -> bool:
    if not isinstance(checkpoint, Mapping):
        return False
    lineage: Any = checkpoint.get("parent_checkpoint_lineage")
    while isinstance(lineage, Mapping):
        promotion = lineage.get("promotion")
        if isinstance(promotion, Mapping) and (
            promotion.get("evidence_kind")
            == VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE
            and promotion.get("artifact_content_sha256")
            == reference.get("content_sha256")
            and promotion.get("artifact_binding_sha256")
            == reference.get("binding_sha256")
        ):
            return True
        lineage = lineage.get("parent_checkpoint_lineage")
    return False


def checkpoint_content_fingerprint(path: str | Path, *, canonicalize: bool = True) -> dict[str, Any]:
    """Return a content fingerprint for one concrete checkpoint directory.

    PPO callers may supply a checkpoint root; it is resolved to the exact
    ``checkpoint_<step>`` before hashing so a later checkpoint cannot silently
    change the identity of the teacher used by an existing dataset.
    """
    supplied = str(path)
    if supplied.startswith("hf://"):
        if not canonicalize:
            raise ValueError("remote checkpoint fingerprints require canonicalization")
        from musclemimic.runner.checkpointing import _canonicalize_resume_path

        concrete = Path(_canonicalize_resume_path(supplied)).resolve()
    else:
        candidate = Path(supplied).expanduser()
        if canonicalize:
            from musclemimic.runner.checkpointing import _canonicalize_resume_path

            concrete = Path(_canonicalize_resume_path(str(candidate))).resolve()
        else:
            concrete = candidate.resolve()
    if not concrete.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {concrete}")

    files = sorted(
        (item for item in concrete.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(concrete).as_posix(),
    )
    if not files:
        raise ValueError(f"checkpoint directory has no files to fingerprint: {concrete}")
    digest = hashlib.sha256()
    total_bytes = 0
    records: list[dict[str, Any]] = []
    for item in files:
        relative = item.relative_to(concrete).as_posix()
        payload = item.read_bytes()
        item_hash = hashlib.sha256(payload).hexdigest()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        total_bytes += len(payload)
        records.append({"path": relative, "sha256": item_hash, "num_bytes": len(payload)})
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": supplied,
        "resolved_path": str(concrete),
        "sha256": digest.hexdigest(),
        "num_files": len(records),
        "num_bytes": int(total_bytes),
        # The file inventory makes changes auditable without trusting only one
        # opaque directory digest.
        "files": records,
    }


def checkpoint_fingerprint_matches(record: Mapping[str, Any]) -> bool:
    resolved = record.get("resolved_path")
    if not isinstance(resolved, str) or not resolved:
        return False
    try:
        actual = checkpoint_content_fingerprint(resolved, canonicalize=False)
    except (OSError, ValueError):
        return False
    return (
        actual["sha256"] == record.get("sha256")
        and actual["num_files"] == record.get("num_files")
        and actual["num_bytes"] == record.get("num_bytes")
    )


def validate_direct_acceptance_record(record: Any) -> dict[str, Any]:
    """Recheck the strict shape of a direct-student acceptance result."""
    if not isinstance(record, Mapping) or record.get("passed") is not True:
        raise ValueError("selected direct policy has no passed acceptance evidence")
    if record.get("failed") != [] or record.get("missing") != []:
        raise ValueError("selected direct acceptance contains failed or missing gates")
    values = record.get("values")
    thresholds = record.get("thresholds")
    required_values = {
        "return_ratio",
        "completion_ratio",
        "early_termination_delta",
        "err_rpos_relative_degradation",
        "err_racket_pos_relative_degradation",
        "err_racket_rot_relative_degradation",
        "convergence_normalized_abs_slope",
        "convergence_normalized_span",
        "temporal_usable_sequence_count",
        "temporal_best_lag_steps",
        "temporal_max_abs_motion_best_lag_steps",
        "temporal_lag_mse_improvement_fraction",
    }
    if not isinstance(values, Mapping) or not required_values.issubset(values):
        raise ValueError("selected direct acceptance values are incomplete")
    if any(not np.isfinite(float(values[key])) for key in required_values):
        raise ValueError("selected direct acceptance contains non-finite values")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("selected direct acceptance thresholds are missing")
    convergence = record.get("convergence")
    temporal = record.get("temporal")
    for name, gate, checks_key in (
        ("convergence", convergence, "evidence_checks"),
        ("temporal", temporal, "checks"),
    ):
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ValueError(f"selected direct acceptance {name} gate is not passed")
        checks = gate.get(checks_key)
        if not isinstance(checks, Mapping) or not checks or any(value is not True for value in checks.values()):
            raise ValueError(f"selected direct acceptance {name} checks are incomplete")
    return dict(record)


def validate_stage2_teacher_promotion(
    manifest_path: str | Path,
    *,
    teacher_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and bind the complete Stage-2 promotion artifact."""
    path = Path(manifest_path).expanduser().resolve(strict=True)
    resolved_teacher = Path(str(teacher_checkpoint.get("resolved_path", ""))).resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    from musclemimic.badminton.racket_mass_curriculum import (
        PROMOTION_SCHEMA_VERSION as MASS_PROMOTION_SCHEMA_VERSION,
    )
    from musclemimic.badminton.racket_mass_curriculum import (
        validate_mass_promoted_artifact,
    )

    if isinstance(raw, Mapping) and raw.get("schema_version") == MASS_PROMOTION_SCHEMA_VERSION:
        payload = validate_mass_promoted_artifact(
            path,
            expected_stage="mass_100",
            expected_checkpoint=resolved_teacher,
        )
        teacher_role = "racket_mass_100"
    else:
        # The legacy Stage-2 validator remains byte-for-byte strict; only the
        # separately versioned final mass-rung schema receives this alternate
        # path.
        from musclemimic.badminton.promotion_artifact import (
            validate_promoted_artifact,
        )

        payload = validate_promoted_artifact(
            path,
            expected_stage="stage2",
            expected_checkpoint=resolved_teacher,
        )
        teacher_role = "legacy_stage2"
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-2 promotion artifact has no checkpoint identity")
    if Path(str(checkpoint.get("checkpoint_path", ""))).resolve(strict=True) != resolved_teacher:
        raise ValueError("Stage-2 promotion artifact points to a different teacher checkpoint")
    binding = {
        "schema_version": TEACHER_PROMOTION_BINDING_SCHEMA,
        "path": str(path),
        "content_sha256": file_sha256(path),
        "binding_sha256": payload.get("binding_sha256"),
        "stage": "stage2",
        "teacher_checkpoint_sha256": str(teacher_checkpoint.get("sha256", "")),
        # Embed the complete validated artifact so run/update/config, numerical
        # metrics, visual review, and Stage-1 parent identities remain available
        # at every downstream layer without trusting a path-only handoff.
        "artifact": payload,
    }
    if teacher_role != "legacy_stage2":
        binding["teacher_role"] = teacher_role
    return binding


def validate_stage1_teacher_promotion(
    manifest_path: str | Path,
    *,
    teacher_checkpoint: Mapping[str, Any],
    teacher_role: str,
) -> dict[str, Any]:
    """Validate a formal Stage-1 body-only teacher without relabeling it.

    Stage-1 is deliberately represented by its own binding schema.  This
    keeps the historical Stage-2 binding byte-for-byte unchanged and makes a
    downstream reader prove which promotion contract was actually used.
    """

    role = str(teacher_role).strip().lower()
    if role != STAGE1_BODY_ONLY_TEACHER_ROLE:
        raise ValueError(
            "Stage-1 distill teachers require explicit teacher_role='body_only'"
        )
    path = Path(manifest_path).expanduser().resolve(strict=True)
    resolved_teacher = Path(
        str(teacher_checkpoint.get("resolved_path", ""))
    ).resolve(strict=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-1 teacher promotion manifest is unreadable") from exc
    from musclemimic.badminton.stage1_peasd_gate import (
        PEASD_TEACHER_PROMOTION_SCHEMA_VERSION,
        validate_stage1_peasd_teacher_promotion,
    )

    if isinstance(raw, Mapping) and raw.get("schema_version") == (
        PEASD_TEACHER_PROMOTION_SCHEMA_VERSION
    ):
        payload = validate_stage1_peasd_teacher_promotion(
            path,
            expected_checkpoint=resolved_teacher,
        )
        promotion_kind = VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE
    else:
        from musclemimic.badminton.promotion_artifact import validate_promoted_artifact

        payload = validate_promoted_artifact(
            path,
            expected_stage="stage1",
            expected_checkpoint=resolved_teacher,
        )
        promotion_kind = VERIFIED_STAGE1_PROMOTION_EVIDENCE
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-1 promotion artifact has no checkpoint identity")
    if Path(str(checkpoint.get("checkpoint_path", ""))).resolve(
        strict=True
    ) != resolved_teacher:
        raise ValueError(
            "Stage-1 promotion artifact points to a different teacher checkpoint"
        )
    result = {
        "schema_version": STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA,
        "path": str(path),
        "content_sha256": file_sha256(path),
        "binding_sha256": payload.get("binding_sha256"),
        "stage": "stage1",
        "teacher_role": role,
        "teacher_checkpoint_sha256": str(teacher_checkpoint.get("sha256", "")),
        "artifact": payload,
    }
    if promotion_kind == VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE:
        result["promotion_kind"] = promotion_kind
    return result


def validate_teacher_promotion_manifest(
    manifest_path: str | Path,
    *,
    teacher_checkpoint: Mapping[str, Any],
    expected_stage: str = DEFAULT_TEACHER_PROMOTION_STAGE,
    teacher_role: str | None = None,
) -> dict[str, Any]:
    """Validate a promoted teacher under an explicitly selected stage.

    Omitting ``expected_stage`` preserves the established Stage-2 behavior.
    Selecting Stage-1 requires the explicit ``body_only`` role so it cannot be
    mistaken for a racket/full-chain Stage-2 teacher.
    """

    stage = str(expected_stage).strip().lower()
    if stage == "stage1":
        if teacher_role is None:
            raise ValueError(
                "Stage-1 teacher promotion requires explicit teacher_role='body_only'"
            )
        return validate_stage1_teacher_promotion(
            manifest_path,
            teacher_checkpoint=teacher_checkpoint,
            teacher_role=teacher_role,
        )
    if stage != "stage2":
        raise ValueError("teacher promotion stage must be 'stage1' or 'stage2'")
    binding = validate_stage2_teacher_promotion(
        manifest_path,
        teacher_checkpoint=teacher_checkpoint,
    )
    if teacher_role is not None:
        requested_role = str(teacher_role).strip().lower()
        actual_role = str(binding.get("teacher_role", "legacy_stage2"))
        if requested_role != actual_role:
            raise ValueError(
                "Stage-2 teacher promotion role mismatch: "
                f"artifact={actual_role!r} requested={requested_role!r}"
            )
    return binding


def test_only_unpromoted_teacher_binding(
    teacher_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TEST_ONLY_TEACHER_PROMOTION_SCHEMA,
        "test_only": True,
        "teacher_checkpoint_sha256": str(teacher_checkpoint.get("sha256", "")),
    }


def validate_teacher_promotion_binding(
    binding: Any,
    *,
    teacher_checkpoint: Mapping[str, Any],
    require_promoted: bool,
    expected_stage: str | None = None,
    expected_teacher_role: str | None = None,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError("distill dataset is missing teacher promotion evidence")
    schema = binding.get("schema_version")
    if schema == TEST_ONLY_TEACHER_PROMOTION_SCHEMA:
        if require_promoted:
            raise ValueError("test-only unpromoted teacher cannot enter production promotion")
        if binding.get("test_only") is not True or binding.get("teacher_checkpoint_sha256") != teacher_checkpoint.get(
            "sha256"
        ):
            raise ValueError("test-only teacher binding does not match dataset teacher")
        return dict(binding)
    if schema == TEACHER_PROMOTION_BINDING_SCHEMA:
        rebuilt = validate_stage2_teacher_promotion(
            str(binding.get("path", "")),
            teacher_checkpoint=teacher_checkpoint,
        )
        change_label = "Stage-2"
        actual_role = str(rebuilt.get("teacher_role", "legacy_stage2"))
    elif schema == STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA:
        rebuilt = validate_stage1_teacher_promotion(
            str(binding.get("path", "")),
            teacher_checkpoint=teacher_checkpoint,
            teacher_role=str(binding.get("teacher_role", "")),
        )
        change_label = "Stage-1"
        actual_role = str(rebuilt["teacher_role"])
    else:
        raise ValueError("teacher promotion binding schema is invalid")
    if _jsonable(dict(binding)) != rebuilt:
        raise ValueError(
            f"{change_label} teacher promotion binding or one of its sources changed"
        )
    actual_stage = str(rebuilt.get("stage", "")).lower()
    if expected_stage is not None and actual_stage != str(expected_stage).strip().lower():
        raise ValueError(
            "teacher promotion stage mismatch: "
            f"binding={actual_stage!r} expected={str(expected_stage).strip().lower()!r}"
        )
    if expected_teacher_role is not None and actual_role != str(
        expected_teacher_role
    ).strip().lower():
        raise ValueError(
            "teacher promotion role mismatch: "
            f"binding={actual_role!r} expected={str(expected_teacher_role).strip().lower()!r}"
        )
    return rebuilt


def teacher_promotion_evidence_kind(binding: Mapping[str, Any]) -> str:
    """Return the promotion metric token for a validated binding."""

    if binding.get("schema_version") == TEST_ONLY_TEACHER_PROMOTION_SCHEMA:
        return "test_only_unpromoted_teacher"
    stage = str(binding.get("stage", "")).lower()
    schema = binding.get("schema_version")
    if stage == "stage1" and schema == STAGE1_TEACHER_PROMOTION_BINDING_SCHEMA:
        if binding.get("teacher_role") != STAGE1_BODY_ONLY_TEACHER_ROLE:
            raise ValueError("Stage-1 teacher promotion binding lacks body_only role")
        kind = binding.get("promotion_kind", VERIFIED_STAGE1_PROMOTION_EVIDENCE)
        if kind not in {
            VERIFIED_STAGE1_PROMOTION_EVIDENCE,
            VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE,
        }:
            raise ValueError("Stage-1 teacher promotion evidence kind is invalid")
        return str(kind)
    if stage == "stage2" and schema == TEACHER_PROMOTION_BINDING_SCHEMA:
        return VERIFIED_STAGE2_PROMOTION_EVIDENCE
    raise ValueError("teacher promotion binding stage/schema is inconsistent")


def stable_run_uid(*, output_dir: str | Path, teacher_sha256: str, tag: str = "distill") -> str:
    return canonical_json_sha256(
        {
            "kind": "distill_run_uid_v1",
            "tag": str(tag),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "teacher_sha256": str(teacher_sha256),
        }
    )[:24]


def _manifest_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if key != "manifest_fingerprint"}


def _with_manifest_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _jsonable(dict(payload))
    result["manifest_fingerprint"] = canonical_json_sha256(_manifest_without_hash(result))
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_dataset_manifest(dataset_dir: str | Path, *, validate: bool = True) -> dict[str, Any]:
    root = Path(dataset_dir)
    path = root / DATASET_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"distill dataset is missing {DATASET_MANIFEST}: {root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported distill dataset manifest schema: {payload.get('schema_version')!r}")
    expected_hash = canonical_json_sha256(_manifest_without_hash(payload))
    if payload.get("manifest_fingerprint") != expected_hash:
        raise ValueError("distill dataset manifest_fingerprint mismatch")
    if validate:
        validate_dataset_manifest(root, payload=payload)
    return payload


def _all_dataset_shards(root: Path) -> list[Path]:
    return sorted(
        {
            *root.glob("shard_*.npz"),
            *root.glob("train_*.npz"),
            *root.glob("val_*.npz"),
            *root.glob("test_*.npz"),
        },
        key=lambda item: item.name,
    )


def _shard_record(path: Path, *, split: str | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as shard:
        if "student_obs" not in shard.files or "teacher_action" not in shard.files:
            raise ValueError(f"distill shard lacks required arrays: {path}")
        sample_count = int(shard["student_obs"].shape[0])
        if int(shard["teacher_action"].shape[0]) != sample_count:
            raise ValueError(f"distill shard sample dimensions differ: {path}")
        fields = sorted(str(name) for name in shard.files)
    inferred_split = split
    if inferred_split is None:
        inferred_split = next(
            (name for name in ("train", "val", "test") if path.name.startswith(f"{name}_")),
            None,
        )
    return {
        "filename": path.name,
        "split": inferred_split,
        "sha256": file_sha256(path),
        "num_bytes": int(path.stat().st_size),
        "num_samples": sample_count,
        "fields": fields,
    }


def validate_dataset_manifest(
    dataset_dir: str | Path,
    *,
    payload: Mapping[str, Any] | None = None,
    expected_teacher: Mapping[str, Any] | None = None,
    expected_teacher_promotion: Mapping[str, Any] | None = None,
    expected_stage1_peasd_promotion: str | Path | None = None,
    expected_emg_reference: str | Path | None = None,
    require_promoted_teacher: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest = dict(payload) if payload is not None else load_dataset_manifest(root, validate=False)
    expected_hash = canonical_json_sha256(_manifest_without_hash(manifest))
    if manifest.get("manifest_fingerprint") != expected_hash:
        raise ValueError("distill dataset manifest_fingerprint mismatch")
    recorded = manifest.get("shards")
    if not isinstance(recorded, list):
        raise ValueError("distill dataset manifest shards must be a list")
    by_name = {str(item.get("filename")): item for item in recorded if isinstance(item, Mapping)}
    if len(by_name) != len(recorded):
        raise ValueError("distill dataset manifest contains duplicate/invalid shard records")
    actual_paths = _all_dataset_shards(root)
    actual_names = [path.name for path in actual_paths]
    if actual_names != sorted(by_name):
        raise ValueError(
            f"distill dataset shard set differs from immutable manifest: disk={actual_names} manifest={sorted(by_name)}"
        )
    actual_records = [_shard_record(path) for path in actual_paths]
    for actual in actual_records:
        expected = by_name[actual["filename"]]
        for key in ("split", "sha256", "num_bytes", "num_samples", "fields"):
            if _jsonable(actual[key]) != _jsonable(expected.get(key)):
                raise ValueError(
                    "distill dataset shard provenance mismatch: "
                    f"shard={actual['filename']} key={key} "
                    f"disk={actual[key]!r} manifest={expected.get(key)!r}"
                )
    totals = manifest.get("totals") or {}
    expected_samples = sum(int(item["num_samples"]) for item in actual_records)
    if (
        int(totals.get("num_shards", -1)) != len(actual_records)
        or int(totals.get("num_samples", -1)) != expected_samples
    ):
        raise ValueError("distill dataset manifest totals do not match exact shard inventory")
    metadata_record = manifest.get("metadata")
    metadata_path = root / "metadata.json"
    if actual_records:
        if not isinstance(metadata_record, Mapping) or not metadata_path.is_file():
            raise ValueError("distill dataset manifest lacks immutable metadata.json provenance")
        if metadata_record.get("sha256") != file_sha256(metadata_path) or int(
            metadata_record.get("num_bytes", -1)
        ) != int(metadata_path.stat().st_size):
            raise ValueError("distill dataset metadata.json provenance mismatch")
    elif metadata_record is not None:
        raise ValueError("empty distill dataset manifest must not claim metadata provenance")
    body_contract_payload = manifest.get("body_synergy_contract")
    if body_contract_payload is not None:
        from musclemimic.synergy.multistage_contract import BodySynergyContractV2

        body_contract = BodySynergyContractV2.from_manifest(body_contract_payload)
        if manifest.get("body_synergy_contract_fingerprint") != (
            body_contract.contract_fingerprint
        ):
            raise ValueError(
                "distill dataset BodySynergyContractV2 fingerprint mismatch"
            )
        if manifest.get("body_synergy_portable_core_fingerprint") != (
            body_contract.portable_decoder_core_fingerprint
        ):
            raise ValueError(
                "distill dataset portable decoder core fingerprint mismatch"
            )
        _require_sha256_text(
            manifest.get("frozen_body_decoder_fingerprint"),
            "distill frozen_body_decoder_fingerprint",
        )
        if actual_records:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in (
                "body_synergy_contract",
                "body_synergy_contract_fingerprint",
                "body_synergy_portable_core_fingerprint",
                "frozen_body_decoder_fingerprint",
            ):
                if _jsonable(metadata_payload.get(key)) != _jsonable(
                    manifest.get(key)
                ):
                    raise ValueError(
                        f"distill dataset metadata {key} differs from immutable manifest"
                    )
    elif any(
        manifest.get(key) is not None
        for key in (
            "body_synergy_contract_fingerprint",
            "body_synergy_portable_core_fingerprint",
            "frozen_body_decoder_fingerprint",
        )
    ):
        raise ValueError(
            "distill dataset has partial body synergy contract provenance"
        )
    teacher = manifest.get("teacher_checkpoint")
    if not isinstance(teacher, Mapping) or not isinstance(teacher.get("sha256"), str):
        raise ValueError("distill dataset manifest lacks teacher checkpoint provenance")
    if expected_teacher is not None and teacher.get("sha256") != expected_teacher.get("sha256"):
        raise ValueError(
            "distill dataset teacher checkpoint fingerprint mismatch: "
            f"dataset={teacher.get('sha256')} expected={expected_teacher.get('sha256')}"
        )
    teacher_promotion = validate_teacher_promotion_binding(
        manifest.get("teacher_promotion"),
        teacher_checkpoint=teacher,
        require_promoted=bool(require_promoted_teacher),
    )
    if expected_teacher_promotion is not None and _jsonable(dict(expected_teacher_promotion)) != teacher_promotion:
        raise ValueError("dataset Stage-2 teacher promotion manifest differs from requested manifest")
    if not isinstance(manifest.get("run_uid"), str) or not manifest["run_uid"]:
        raise ValueError("distill dataset manifest lacks a non-empty run_uid")
    collections = manifest.get("collections")
    if not isinstance(collections, list):
        raise ValueError("distill dataset manifest collections must be a list")
    collection_ids: set[str] = set()
    assigned_shards: list[str] = []
    peasd_reference_bindings: list[dict[str, Any]] = []
    for collection in collections:
        if not isinstance(collection, Mapping):
            raise ValueError("distill dataset collection record is invalid")
        collection_id = collection.get("collection_id")
        contract = collection.get("contract")
        if not isinstance(collection_id, str) or collection_id in collection_ids:
            raise ValueError("distill dataset collection IDs must be unique strings")
        collection_ids.add(collection_id)
        if not isinstance(contract, Mapping) or contract.get("collection_id") != collection_id:
            raise ValueError(f"distill collection contract identity mismatch: {collection_id}")
        if collection.get("contract_fingerprint") != canonical_json_sha256(contract):
            raise ValueError(f"distill collection contract fingerprint mismatch: {collection_id}")
        if contract.get("teacher_checkpoint_sha256") != teacher.get("sha256"):
            raise ValueError(f"distill collection teacher fingerprint mismatch: {collection_id}")
        if contract.get("teacher_promotion_content_sha256") != teacher_promotion.get("content_sha256") or contract.get(
            "teacher_promotion_binding_sha256"
        ) != teacher_promotion.get("binding_sha256"):
            if (
                teacher_promotion.get("test_only") is not True
                or contract.get("test_only_unpromoted_teacher") is not True
            ):
                raise ValueError(f"distill collection teacher promotion mismatch: {collection_id}")
        request = contract.get("request")
        if not isinstance(request, Mapping):
            raise ValueError(f"distill collection request is invalid: {collection_id}")
        raw_peasd_reference = request.get("stage1_peasd_reference_promotion")
        if raw_peasd_reference is not None:
            peasd_reference_bindings.append(
                validate_stage1_peasd_reference_promotion(
                    raw_peasd_reference,
                    expected_promotion=expected_stage1_peasd_promotion,
                    expected_tube=expected_emg_reference,
                )
            )
        elif expected_stage1_peasd_promotion is not None:
            raise ValueError(
                f"distill collection lacks the selected Stage-1 PEASD promotion: {collection_id}"
            )
        motion_paths = contract.get("motion_paths")
        motion_uids = contract.get("motion_uids")
        if not isinstance(motion_paths, list) or not motion_paths:
            raise ValueError(f"distill collection has no motion split: {collection_id}")
        normalized_paths = [normalize_motion_path(item) for item in motion_paths]
        if normalized_paths != motion_paths or len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError(f"distill collection motion paths are not canonical/unique: {collection_id}")
        if motion_uids != [int(stable_motion_uid(item)) for item in normalized_paths]:
            raise ValueError(f"distill collection motion UID mismatch: {collection_id}")
        if contract.get("motion_split_fingerprint") != canonical_json_sha256(
            {"split": contract.get("split"), "motion_paths": normalized_paths}
        ):
            raise ValueError(f"distill collection motion split fingerprint mismatch: {collection_id}")
        if not isinstance(contract.get("config_fingerprint"), str):
            raise ValueError(f"distill collection config fingerprint is missing: {collection_id}")
        student = contract.get("student_checkpoint")
        student_sha = contract.get("student_checkpoint_sha256")
        if student is None:
            if student_sha is not None:
                raise ValueError(f"distill collection student fingerprint is inconsistent: {collection_id}")
        elif not isinstance(student, Mapping) or student.get("sha256") != student_sha:
            raise ValueError(f"distill collection student checkpoint mismatch: {collection_id}")
        names = collection.get("shards")
        if not isinstance(names, list) or any(name not in by_name for name in names):
            raise ValueError(f"distill collection references unknown shards: {collection_id}")
        assigned_shards.extend(str(name) for name in names)
        if int(collection.get("num_shards", -1)) != len(names) or int(collection.get("num_samples", -1)) != sum(
            int(by_name[name]["num_samples"]) for name in names
        ):
            raise ValueError(f"distill collection shard/sample totals mismatch: {collection_id}")
    if sorted(assigned_shards) != sorted(by_name) or len(assigned_shards) != len(set(assigned_shards)):
        raise ValueError("distill dataset shards are not assigned exactly once to collections")
    if peasd_reference_bindings:
        if len(peasd_reference_bindings) != len(collections):
            raise ValueError(
                "distill dataset mixes PEASD-bound and unbound collection provenance"
            )
        first_reference = peasd_reference_bindings[0]
        if any(binding != first_reference for binding in peasd_reference_bindings[1:]):
            raise ValueError(
                "distill dataset collections use different Stage-1 PEASD promotion/tube bindings"
            )
        promotion_artifact = teacher_promotion.get("artifact")
        promotion_kind = teacher_promotion.get("promotion_kind")
        if promotion_kind == VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE:
            expected_reference = {
                "path": teacher_promotion.get("path"),
                "content_sha256": teacher_promotion.get("content_sha256"),
                "binding_sha256": teacher_promotion.get("binding_sha256"),
                "emg_reference_binding": (
                    promotion_artifact.get("emg_reference_binding")
                    if isinstance(promotion_artifact, Mapping)
                    else None
                ),
            }
            if first_reference != expected_reference:
                raise ValueError(
                    "body-only distill collection reference promotion differs from its PEASD teacher"
                )
        elif teacher_promotion.get("teacher_role") == "racket_mass_100":
            checkpoint = (
                promotion_artifact.get("checkpoint")
                if isinstance(promotion_artifact, Mapping)
                else None
            )
            if not _checkpoint_descends_from_stage1_peasd_reference(
                checkpoint,
                first_reference,
            ):
                raise ValueError(
                    "racket distill teacher ancestry does not contain the collection's "
                    "Stage-1 PEASD promotion"
                )
        else:
            raise ValueError(
                "Stage-1 PEASD reference collection cannot use an unrelated teacher promotion"
            )
    return manifest


@dataclass
class DistillCollectionTransaction:
    dataset_dir: Path
    staging_dir: Path
    collection_id: str
    contract: dict[str, Any]
    manifest: dict[str, Any]
    already_complete: bool = False

    @property
    def output_dir(self) -> Path:
        return self.staging_dir

    @property
    def existing_paths(self) -> list[Path]:
        record = next(
            (item for item in self.manifest.get("collections", []) if item.get("collection_id") == self.collection_id),
            None,
        )
        return [] if record is None else [self.dataset_dir / name for name in record.get("shards", [])]

    def commit(self, staged_paths: Iterable[str | Path]) -> list[Path]:
        if self.already_complete:
            return self.existing_paths
        current = load_dataset_manifest(self.dataset_dir, validate=True)
        if current["manifest_fingerprint"] != self.manifest["manifest_fingerprint"]:
            raise ValueError("distill dataset changed concurrently during collection")

        staged = sorted((Path(path) for path in staged_paths), key=lambda item: item.name)
        if not staged or any(path.parent.resolve() != self.staging_dir.resolve() for path in staged):
            raise ValueError("collection commit requires non-empty shards from its isolated staging directory")
        staged_records = [_shard_record(path, split=self.contract.get("split")) for path in staged]
        requested_samples = (self.contract.get("request") or {}).get("num_transitions")
        staged_samples = sum(int(item["num_samples"]) for item in staged_records)
        if requested_samples is not None and staged_samples != int(requested_samples):
            raise ValueError(
                "distill collection sample count differs from immutable request: "
                f"staged={staged_samples} requested={int(requested_samples)}"
            )
        split = self.contract.get("split")
        prefix = f"{split}_" if split in {"train", "val", "test"} else "shard_"
        existing = _all_dataset_shards(self.dataset_dir)
        indices = [
            int(path.stem.rsplit("_", 1)[-1])
            for path in existing
            if path.name.startswith(prefix) and path.stem.rsplit("_", 1)[-1].isdigit()
        ]
        next_index = 0 if not indices else max(indices) + 1
        destinations: list[Path] = []
        for offset, source in enumerate(staged):
            destination = self.dataset_dir / f"{prefix}{next_index + offset:06d}.npz"
            if destination.exists():
                raise FileExistsError(f"distill shard destination already exists: {destination}")
            os.replace(source, destination)
            destinations.append(destination)

        # Preserve the existing aggregate ABI metadata while adding the staged
        # split/collector metadata.  The immutable manifest remains the source
        # of truth for inventory and sample counts.
        from musclemimic.distill.dataset import _infer_metadata, _merge_dataset_metadata

        metadata_path = self.dataset_dir / "metadata.json"
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        staged_metadata_path = self.staging_dir / "metadata.json"
        staged_metadata = (
            json.loads(staged_metadata_path.read_text(encoding="utf-8")) if staged_metadata_path.is_file() else {}
        )
        merged = _merge_dataset_metadata(existing_metadata, staged_metadata)
        merged = _infer_metadata(self.dataset_dir, merged)
        _atomic_write_json(metadata_path, merged)

        records = list(current.get("shards") or [])
        new_records = [_shard_record(path, split=split) for path in destinations]
        records.extend(new_records)
        collection_record = {
            "collection_id": self.collection_id,
            "contract": self.contract,
            "contract_fingerprint": canonical_json_sha256(self.contract),
            "shards": [path.name for path in destinations],
            "num_shards": len(destinations),
            "num_samples": sum(int(item["num_samples"]) for item in new_records),
        }
        collections = list(current.get("collections") or [])
        if any(item.get("collection_id") == self.collection_id for item in collections):
            raise ValueError(f"collection ID was committed concurrently: {self.collection_id}")
        collections.append(collection_record)
        updated = {
            **_manifest_without_hash(current),
            "collections": collections,
            "shards": records,
            "totals": {
                "num_shards": len(records),
                "num_samples": sum(int(item["num_samples"]) for item in records),
            },
            "metadata": {
                "filename": "metadata.json",
                "sha256": file_sha256(metadata_path),
                "num_bytes": int(metadata_path.stat().st_size),
            },
        }
        updated = _with_manifest_hash(updated)
        _atomic_write_json(self.dataset_dir / DATASET_MANIFEST, updated)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        validate_dataset_manifest(self.dataset_dir, payload=updated)
        self.manifest = updated
        self.already_complete = True
        return destinations


def begin_collection(
    *,
    dataset_dir: str | Path,
    teacher_checkpoint: Mapping[str, Any],
    collector: str,
    split: str | None,
    seed: int,
    motion_paths: Iterable[str],
    config_payload: Any,
    request_payload: Mapping[str, Any],
    resume: bool,
    run_uid: str | None = None,
    dagger_iteration: int | None = None,
    student_checkpoint: Mapping[str, Any] | None = None,
    teacher_promotion: Mapping[str, Any] | None = None,
    teacher_promotion_stage: str | None = None,
    teacher_promotion_role: str | None = None,
    allow_test_only_unpromoted_teacher: bool = False,
    body_synergy_contract: Mapping[str, Any] | None = None,
    frozen_body_decoder_fingerprint: str | None = None,
) -> DistillCollectionTransaction:
    root = Path(dataset_dir)
    normalized_paths = [normalize_motion_path(path) for path in motion_paths]
    if not normalized_paths or len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("distill collection requires a non-empty unique motion path list")
    if collector == "dagger_student_rollout_teacher_relabel" and dagger_iteration is None:
        raise ValueError("DAgger collection requires an explicit dagger_iteration")
    if teacher_promotion is None:
        if not allow_test_only_unpromoted_teacher:
            raise ValueError("production distill collection requires --teacher_promotion_manifest")
        resolved_teacher_promotion = test_only_unpromoted_teacher_binding(teacher_checkpoint)
    else:
        resolved_teacher_promotion = validate_teacher_promotion_binding(
            teacher_promotion,
            teacher_checkpoint=teacher_checkpoint,
            require_promoted=True,
            expected_stage=teacher_promotion_stage,
            expected_teacher_role=teacher_promotion_role,
        )
    resolved_body_contract = None
    resolved_frozen_decoder_fingerprint = None
    if body_synergy_contract is not None:
        from musclemimic.synergy.multistage_contract import BodySynergyContractV2

        resolved_contract = BodySynergyContractV2.from_manifest(
            body_synergy_contract
        )
        resolved_body_contract = resolved_contract.to_manifest()
        resolved_frozen_decoder_fingerprint = _require_sha256_text(
            frozen_body_decoder_fingerprint,
            "frozen_body_decoder_fingerprint",
        )
    elif frozen_body_decoder_fingerprint not in (None, ""):
        raise ValueError(
            "frozen_body_decoder_fingerprint requires body_synergy_contract"
        )
    root.mkdir(parents=True, exist_ok=True)
    resolved_run_uid = str(
        run_uid
        or stable_run_uid(
            output_dir=root,
            teacher_sha256=str(teacher_checkpoint["sha256"]),
        )
    )
    if not resolved_run_uid:
        raise ValueError("distill run_uid must be non-empty")
    collection_id = (
        f"dagger_{split or 'all'}_{int(dagger_iteration):06d}"
        if dagger_iteration is not None
        else f"teacher_{split or 'all'}"
    )
    contract = {
        "schema_version": "distill_collection_contract_v2",
        "collection_id": collection_id,
        "collector": str(collector),
        "split": split,
        "seed": int(seed),
        "dagger_iteration": None if dagger_iteration is None else int(dagger_iteration),
        "motion_paths": normalized_paths,
        "motion_uids": [int(stable_motion_uid(path)) for path in normalized_paths],
        "motion_split_fingerprint": canonical_json_sha256({"split": split, "motion_paths": normalized_paths}),
        "config_fingerprint": canonical_json_sha256(config_payload),
        "request": _jsonable(dict(request_payload)),
        "teacher_checkpoint_sha256": str(teacher_checkpoint["sha256"]),
        "teacher_promotion_content_sha256": resolved_teacher_promotion.get("content_sha256"),
        "teacher_promotion_binding_sha256": resolved_teacher_promotion.get("binding_sha256"),
        "test_only_unpromoted_teacher": bool(resolved_teacher_promotion.get("test_only", False)),
        "student_checkpoint_sha256": (None if student_checkpoint is None else str(student_checkpoint["sha256"])),
        "student_checkpoint": (None if student_checkpoint is None else _jsonable(dict(student_checkpoint))),
        "body_synergy_contract": resolved_body_contract,
        "body_synergy_contract_fingerprint": (
            None
            if resolved_body_contract is None
            else resolved_body_contract["contract_fingerprint"]
        ),
        "body_synergy_portable_core_fingerprint": (
            None
            if resolved_body_contract is None
            else resolved_body_contract["portable_decoder_core_fingerprint"]
        ),
        "frozen_body_decoder_fingerprint": resolved_frozen_decoder_fingerprint,
    }
    manifest_path = root / DATASET_MANIFEST
    if not resume:
        # Fresh means fresh: metadata, a stale DAgger shard, a prior manifest,
        # or an interrupted staging directory all require an explicit resume or
        # a new output directory.
        entries = sorted(item.name for item in root.iterdir())
        if entries:
            raise ValueError(
                "fresh distill collection refuses a non-empty dataset directory; "
                f"found={entries}. Use explicit resume only for the same manifest."
            )
        manifest = _with_manifest_hash(
            {
                "schema_version": DATASET_MANIFEST_SCHEMA,
                "run_uid": resolved_run_uid,
                "teacher_checkpoint": _jsonable(dict(teacher_checkpoint)),
                "teacher_promotion": resolved_teacher_promotion,
                "body_synergy_contract": resolved_body_contract,
                "body_synergy_contract_fingerprint": (
                    None
                    if resolved_body_contract is None
                    else resolved_body_contract["contract_fingerprint"]
                ),
                "body_synergy_portable_core_fingerprint": (
                    None
                    if resolved_body_contract is None
                    else resolved_body_contract[
                        "portable_decoder_core_fingerprint"
                    ]
                ),
                "frozen_body_decoder_fingerprint": (
                    resolved_frozen_decoder_fingerprint
                ),
                "collections": [],
                "shards": [],
                "totals": {"num_shards": 0, "num_samples": 0},
                "metadata": None,
            }
        )
        _atomic_write_json(manifest_path, manifest)
    else:
        manifest = load_dataset_manifest(root, validate=True)
        if manifest.get("run_uid") != resolved_run_uid:
            raise ValueError(
                f"distill resume run_uid mismatch: manifest={manifest.get('run_uid')} requested={resolved_run_uid}"
            )
        recorded_teacher = manifest.get("teacher_checkpoint") or {}
        if recorded_teacher.get("sha256") != teacher_checkpoint.get("sha256"):
            raise ValueError("distill resume teacher checkpoint content fingerprint mismatch")
        if _jsonable(manifest.get("teacher_promotion")) != _jsonable(resolved_teacher_promotion):
            raise ValueError("distill resume Stage-2 teacher promotion binding mismatch")
        for key, expected in (
            ("body_synergy_contract", resolved_body_contract),
            (
                "body_synergy_contract_fingerprint",
                None
                if resolved_body_contract is None
                else resolved_body_contract["contract_fingerprint"],
            ),
            (
                "body_synergy_portable_core_fingerprint",
                None
                if resolved_body_contract is None
                else resolved_body_contract[
                    "portable_decoder_core_fingerprint"
                ],
            ),
            (
                "frozen_body_decoder_fingerprint",
                resolved_frozen_decoder_fingerprint,
            ),
        ):
            if _jsonable(manifest.get(key)) != _jsonable(expected):
                raise ValueError(
                    f"distill resume {key} binding mismatch"
                )

    completed = next(
        (item for item in manifest.get("collections", []) if item.get("collection_id") == collection_id),
        None,
    )
    if completed is not None:
        if completed.get("contract_fingerprint") != canonical_json_sha256(contract):
            raise ValueError(f"distill collection {collection_id} already exists with a different immutable contract")
        return DistillCollectionTransaction(
            dataset_dir=root,
            staging_dir=root / ".distill_staging" / collection_id,
            collection_id=collection_id,
            contract=contract,
            manifest=manifest,
            already_complete=True,
        )

    staging = root / ".distill_staging" / collection_id
    if staging.exists():
        marker = staging / "collection_contract.json"
        if not marker.is_file():
            raise ValueError(f"untrusted distill staging directory lacks contract marker: {staging}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        previous_contract = previous.get("contract") if isinstance(previous, Mapping) else None
        if canonical_json_sha256(previous_contract) != canonical_json_sha256(contract):
            raise ValueError(f"stale distill staging contract mismatch: {staging}")
        owner_pid = previous.get("owner_pid")
        if isinstance(owner_pid, int) and owner_pid > 0:
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise ValueError(f"cannot verify active distill staging owner pid={owner_pid}") from exc
            else:
                raise ValueError(f"distill collection is already active in another process pid={owner_pid}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(
        staging / "collection_contract.json",
        {"contract": contract, "owner_pid": os.getpid()},
    )
    return DistillCollectionTransaction(
        dataset_dir=root,
        staging_dir=staging,
        collection_id=collection_id,
        contract=contract,
        manifest=manifest,
    )
