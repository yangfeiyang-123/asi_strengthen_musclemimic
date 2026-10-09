"""Validation video recording utilities for training workflows."""

import hashlib
import json
import re
import secrets
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.badminton.visual_review import (
    CANDIDATE_SCHEMA_VERSION,
    REVIEW_SCHEMA_VERSION,
    REVIEW_KINDS,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
)
from musclemimic.utils import setup_headless_rendering_if_needed
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import TaskFactory


def _candidate_motion_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    return normalized[:-4] if normalized.endswith(".npz") else normalized


ENDPOINT_REVIEW_SET_SCHEMA_VERSION = "stage1_peasd_endpoint_visual_review_set_v1"
STAGE2_ENDPOINT_REVIEW_SET_SCHEMA_VERSION = "stage2_endpoint_visual_review_set_v1"
STAGE1_BLIND_PACKAGE_SCHEMA_VERSION = "stage1_peasd_blind_package_v1"
STAGE1_BLIND_MAPPING_SCHEMA_VERSION = "stage1_peasd_blind_private_mapping_v1"
STAGE1_BLIND_REVIEW_SCHEMA_VERSION = "stage1_peasd_blind_review_v1"
_OPAQUE_ID = re.compile(r"^[0-9a-f]{32}$")
_ARM_IDENTITY_TOKEN = re.compile(r"(?<![a-z0-9])t[0-4](?![a-z0-9])", re.IGNORECASE)
_SEED_IDENTITY_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:seed[ _-]*[0-9]+|s[0-9]+)(?![a-z0-9])",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reviewer_sensitive_values(
    *,
    candidate: Mapping | None = None,
    motions: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return private identity fragments that may not enter a blind package."""

    values: set[str] = set()
    if candidate is not None:
        for raw in candidate.values():
            if not isinstance(raw, str) or not raw.strip():
                continue
            value = raw.strip()
            values.add(value)
            if "/" in value or "\\" in value:
                path = Path(value)
                values.add(path.name)
                values.add(path.parent.name)
    for raw in motions or ():
        value = str(raw).strip()
        if not value:
            continue
        path = Path(value)
        values.update((value, path.name, path.stem))
    # Tiny fragments create accidental substring matches and do not provide a
    # meaningful identity check.  Canonical arm/seed tokens are handled by
    # dedicated boundary-aware expressions below.
    return tuple(sorted(value for value in values if len(value) >= 4))


def _reject_reviewer_visible_identity(
    payload: Mapping,
    *,
    candidate: Mapping | None = None,
    motions: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Reject identity-bearing fields or values from reviewer-visible JSON."""

    def walk(value) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).lower()
                if (
                    "checkpoint" in key
                    or "run_id" in key
                    or "runid" in key
                    or key in {"arm", "seed", "motion", "source"}
                    or key.endswith(("_arm", "_seed", "_motion", "_source"))
                ):
                    raise ValueError(
                        f"Stage1 reviewer-visible JSON leaks identity field {raw_key!r}"
                    )
                walk(child)
        elif isinstance(value, list | tuple):
            for child in value:
                walk(child)

    walk(payload)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).lower()
    if _ARM_IDENTITY_TOKEN.search(encoded):
        raise ValueError("Stage1 reviewer-visible JSON leaks a treatment-arm identity")
    if _SEED_IDENTITY_TOKEN.search(encoded):
        raise ValueError("Stage1 reviewer-visible JSON leaks a seed identity")
    for private_value in _reviewer_sensitive_values(
        candidate=candidate,
        motions=motions,
    ):
        if private_value.lower() in encoded:
            raise ValueError("Stage1 reviewer-visible JSON leaks a private endpoint identity")


def _resolve_artifact_path(raw: str, *, relative_to: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=True)


def validate_stage1_endpoint_review_set(
    source: str | Path | Mapping,
    *,
    expected_candidate: Mapping | None = None,
    expected_motions: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Validate the complete frozen-policy visual candidate set."""

    source_path = None if isinstance(source, Mapping) else Path(source).expanduser().resolve(strict=True)
    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else json.loads(source_path.read_text(encoding="utf-8"))
    )
    expected_keys = {
        "schema_version",
        "review_kind",
        "candidate",
        "deterministic_policy",
        "run_stats_frozen",
        "validation_motion_paths",
        "clips",
        "binding_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Stage1 endpoint review-set schema mismatch")
    if (
        payload.get("schema_version") != ENDPOINT_REVIEW_SET_SCHEMA_VERSION
        or payload.get("review_kind") != STAGE1_REVIEW_KIND
        or payload.get("deterministic_policy") is not True
        or payload.get("run_stats_frozen") is not True
    ):
        raise ValueError("Stage1 endpoint review set is not frozen deterministic evidence")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if payload.get("binding_sha256") != _canonical_sha256(unsigned):
        raise ValueError("Stage1 endpoint review-set binding mismatch")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("Stage1 endpoint review set has no checkpoint candidate")
    if expected_candidate is None:
        from musclemimic.badminton.promotion_artifact import checkpoint_identity

        expected_candidate = checkpoint_identity(str(candidate.get("checkpoint_path", "")))
    if dict(candidate) != dict(expected_candidate):
        raise ValueError("Stage1 endpoint review set belongs to another checkpoint")
    motions = payload.get("validation_motion_paths")
    clips = payload.get("clips")
    if not isinstance(motions, list) or not motions or not isinstance(clips, list):
        raise ValueError("Stage1 endpoint review set lacks held-out motions/clips")
    if expected_motions is not None and motions != [str(item) for item in expected_motions]:
        raise ValueError("Stage1 endpoint review set differs from the held-out split")
    if len(clips) != len(motions):
        raise ValueError("Stage1 endpoint review set is incomplete")
    base_dir = source_path.parent if source_path is not None else Path.cwd()
    for index, (motion, raw_clip) in enumerate(zip(motions, clips, strict=True)):
        if not isinstance(raw_clip, Mapping) or set(raw_clip) != {
            "motion",
            "artifact",
            "artifact_content_sha256",
        }:
            raise ValueError(f"Stage1 endpoint review clip {index} schema mismatch")
        if raw_clip.get("motion") != motion:
            raise ValueError("Stage1 endpoint review clip order/motion mismatch")
        artifact = Path(str(raw_clip.get("artifact", ""))).expanduser()
        if not artifact.is_absolute():
            artifact = base_dir / artifact
        artifact = artifact.resolve(strict=True)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"Stage1 endpoint review artifact is missing: {artifact}")
        if raw_clip.get("artifact_content_sha256") != _sha256_file(artifact):
            raise ValueError("Stage1 endpoint review artifact content changed")
    return payload


def validate_stage1_peasd_blind_package(source: str | Path | Mapping) -> dict:
    """Validate reviewer-visible package metadata and anonymous video bytes."""

    source_path = None if isinstance(source, Mapping) else Path(source).expanduser().resolve(strict=True)
    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else json.loads(source_path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "package_id",
        "clips",
        "package_content_sha256",
    }:
        raise ValueError("Stage1 blind package schema mismatch")
    if payload.get("schema_version") != STAGE1_BLIND_PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 blind package schema")
    package_id = str(payload.get("package_id", ""))
    if not _OPAQUE_ID.fullmatch(package_id):
        raise ValueError("Stage1 blind package_id is not opaque")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Stage1 blind package has no clips")
    seen_ids: set[str] = set()
    seen_artifacts: set[str] = set()
    base_dir = source_path.parent if source_path is not None else Path.cwd()
    normalized_clips = []
    for index, raw_clip in enumerate(clips):
        if not isinstance(raw_clip, Mapping) or set(raw_clip) != {
            "opaque_clip_id",
            "artifact",
            "artifact_content_sha256",
        }:
            raise ValueError(f"Stage1 blind package clip {index} schema mismatch")
        opaque_id = str(raw_clip.get("opaque_clip_id", ""))
        artifact_text = str(raw_clip.get("artifact", ""))
        artifact = Path(artifact_text)
        if (
            not _OPAQUE_ID.fullmatch(opaque_id)
            or artifact.is_absolute()
            or artifact.parts != ("clips", f"{opaque_id}.mp4")
            or ".." in artifact.parts
        ):
            raise ValueError("Stage1 blind package exposes a non-opaque artifact identity")
        if opaque_id in seen_ids or artifact_text in seen_artifacts:
            raise ValueError("Stage1 blind package repeats an opaque clip")
        seen_ids.add(opaque_id)
        seen_artifacts.add(artifact_text)
        artifact_path = (base_dir / artifact).resolve(strict=True)
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise ValueError("Stage1 blind package clip is missing or empty")
        artifact_sha = str(raw_clip.get("artifact_content_sha256", ""))
        if artifact_sha != _sha256_file(artifact_path):
            raise ValueError("Stage1 blind package video bytes changed")
        normalized_clips.append(dict(raw_clip))
    core = {
        "schema_version": STAGE1_BLIND_PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "clips": normalized_clips,
    }
    if payload.get("package_content_sha256") != _canonical_sha256(core):
        raise ValueError("Stage1 blind package content binding mismatch")
    _reject_reviewer_visible_identity(payload)
    return payload


def _validate_stage1_peasd_blind_review_template(
    source: str | Path,
    *,
    package: Mapping,
    candidate: Mapping,
    motions: list[str] | tuple[str, ...],
) -> dict:
    path = Path(source).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "package_id",
        "package_manifest",
        "package_content_sha256",
        "reviewer_id",
        "clips",
        "passed",
    }:
        raise ValueError("Stage1 blind review template schema mismatch")
    if (
        path.name != "review_template.json"
        or payload.get("schema_version") != STAGE1_BLIND_REVIEW_SCHEMA_VERSION
        or payload.get("package_id") != package.get("package_id")
        or payload.get("package_manifest") != "package_manifest.json"
        or payload.get("package_content_sha256")
        != package.get("package_content_sha256")
        or payload.get("reviewer_id") is not None
        or payload.get("passed") is not None
    ):
        raise ValueError("Stage1 blind review template is not pristine/anonymous")
    rows = payload.get("clips")
    if not isinstance(rows, list) or len(rows) != len(package["clips"]):
        raise ValueError("Stage1 blind review template has an incomplete clip set")
    for index, (row, clip) in enumerate(zip(rows, package["clips"], strict=True)):
        if not isinstance(row, Mapping) or set(row) != {
            "opaque_clip_id",
            "artifact",
            "major_swing_complete",
            "root_tracking_spike_free",
            "right_hand_tracking_spike_free",
            "passed",
            "notes",
        }:
            raise ValueError(f"Stage1 blind review template clip {index} schema mismatch")
        if (
            row.get("opaque_clip_id") != clip.get("opaque_clip_id")
            or row.get("artifact") != clip.get("artifact")
            or any(
                row.get(field) is not None
                for field in (
                    "major_swing_complete",
                    "root_tracking_spike_free",
                    "right_hand_tracking_spike_free",
                    "passed",
                    "notes",
                )
            )
        ):
            raise ValueError("Stage1 blind review template is pre-filled or relabelled")
    _reject_reviewer_visible_identity(
        payload,
        candidate=candidate,
        motions=motions,
    )
    return payload


def validate_stage1_peasd_blind_mapping(
    source: str | Path | Mapping,
    *,
    expected_candidate: Mapping | None = None,
    expected_motions: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Validate private opaque-to-endpoint mapping and both copies' bytes."""

    source_path = (
        None
        if isinstance(source, Mapping)
        else Path(source).expanduser().resolve(strict=True)
    )
    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else json.loads(source_path.read_text(encoding="utf-8"))
    )
    expected_keys = {
        "schema_version",
        "package_id",
        "review_package_dir",
        "package_manifest_path",
        "package_manifest_content_sha256",
        "package_content_sha256",
        "review_template_path",
        "review_template_content_sha256",
        "review_submission_path",
        "endpoint_review_set_path",
        "endpoint_review_set_content_sha256",
        "endpoint_review_set_binding_sha256",
        "candidate",
        "clips",
        "binding_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Stage1 blind private mapping schema mismatch")
    if payload.get("schema_version") != STAGE1_BLIND_MAPPING_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 blind private mapping schema")
    supplied_binding = str(payload.get("binding_sha256", ""))
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied_binding != _canonical_sha256(unsigned):
        raise ValueError("Stage1 blind private mapping binding mismatch")
    package_manifest_path = Path(str(payload["package_manifest_path"])).expanduser().resolve(strict=True)
    if source_path is not None and (
        source_path.name != "stage1_blind_private_mapping.json"
        or source_path.parent != package_manifest_path.parent.parent
        or source_path.is_relative_to(package_manifest_path.parent)
    ):
        raise ValueError("Stage1 private mapping is not sealed outside the reviewer package")
    if payload["package_manifest_content_sha256"] != _sha256_file(package_manifest_path):
        raise ValueError("Stage1 blind package manifest changed")
    package = validate_stage1_peasd_blind_package(package_manifest_path)
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("Stage1 blind mapping has no candidate identity")
    endpoint_motions = expected_motions
    if (
        payload.get("package_id") != package.get("package_id")
        or payload.get("package_content_sha256") != package.get("package_content_sha256")
        or Path(str(payload["review_package_dir"])).expanduser().resolve(strict=True)
        != package_manifest_path.parent
        or Path(str(payload["review_template_path"])).expanduser().resolve(strict=True)
        != package_manifest_path.parent / "review_template.json"
        or Path(str(payload["review_submission_path"])).expanduser().resolve(strict=True)
        != package_manifest_path.parent / "review.json"
    ):
        raise ValueError("Stage1 blind mapping points to a foreign reviewer package")
    review_template_path = Path(str(payload["review_template_path"])).expanduser().resolve(
        strict=True
    )
    review_submission_path = Path(
        str(payload["review_submission_path"])
    ).expanduser().resolve(strict=True)
    if payload.get("review_template_content_sha256") != _sha256_file(
        review_template_path
    ):
        raise ValueError("Stage1 blind review template changed")
    endpoint_path = Path(str(payload["endpoint_review_set_path"])).expanduser().resolve(strict=True)
    if payload["endpoint_review_set_content_sha256"] != _sha256_file(endpoint_path):
        raise ValueError("Stage1 endpoint review set changed after blinding")
    endpoint = validate_stage1_endpoint_review_set(
        endpoint_path,
        expected_candidate=expected_candidate,
        expected_motions=expected_motions,
    )
    endpoint_motions = list(endpoint["validation_motion_paths"])
    if payload.get("endpoint_review_set_binding_sha256") != endpoint.get("binding_sha256"):
        raise ValueError("Stage1 blind mapping endpoint binding mismatch")
    if not isinstance(candidate, Mapping) or dict(candidate) != dict(endpoint["candidate"]):
        raise ValueError("Stage1 blind mapping candidate differs from endpoint")
    _validate_stage1_peasd_blind_review_template(
        review_template_path,
        package=package,
        candidate=candidate,
        motions=endpoint_motions,
    )
    submission = json.loads(review_submission_path.read_text(encoding="utf-8"))
    if not isinstance(submission, Mapping):
        raise ValueError("Stage1 blind review submission must be a JSON object")
    _reject_reviewer_visible_identity(
        submission,
        candidate=candidate,
        motions=endpoint_motions,
    )
    mapping_clips = payload.get("clips")
    package_clips = package["clips"]
    endpoint_clips = endpoint["clips"]
    if (
        not isinstance(mapping_clips, list)
        or len(mapping_clips) != len(package_clips)
        or len(mapping_clips) != len(endpoint_clips)
    ):
        raise ValueError("Stage1 blind private mapping has an incomplete clip set")
    for index, (mapped, blinded, original) in enumerate(
        zip(mapping_clips, package_clips, endpoint_clips, strict=True)
    ):
        if not isinstance(mapped, Mapping) or set(mapped) != {
            "opaque_clip_id",
            "anonymous_artifact_path",
            "anonymous_artifact_content_sha256",
            "source_motion",
            "source_artifact_path",
            "source_artifact_content_sha256",
        }:
            raise ValueError(f"Stage1 blind mapping clip {index} schema mismatch")
        anonymous = Path(str(mapped["anonymous_artifact_path"])).expanduser().resolve(strict=True)
        original_path = Path(str(mapped["source_artifact_path"])).expanduser().resolve(strict=True)
        endpoint_original_path = _resolve_artifact_path(
            str(original["artifact"]),
            relative_to=endpoint_path.parent,
        )
        if (
            mapped.get("opaque_clip_id") != blinded.get("opaque_clip_id")
            or anonymous != (package_manifest_path.parent / str(blinded["artifact"])).resolve(strict=True)
            or mapped.get("anonymous_artifact_content_sha256")
            != blinded.get("artifact_content_sha256")
            or mapped.get("source_motion") != original.get("motion")
            or original_path
            != endpoint_original_path
            or mapped.get("source_artifact_content_sha256")
            != original.get("artifact_content_sha256")
            or _sha256_file(anonymous) != _sha256_file(original_path)
        ):
            raise ValueError("Stage1 blind mapping cannot reconstruct exact endpoint clip bytes")
    return payload


def build_stage1_peasd_blind_review_package(
    *,
    endpoint_review_set: str | Path,
    output_dir: str | Path,
    expected_candidate: Mapping | None = None,
    expected_motions: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Create anonymous reviewer media plus a separately sealed private map."""

    endpoint_path = Path(endpoint_review_set).expanduser().resolve(strict=True)
    endpoint = validate_stage1_endpoint_review_set(
        endpoint_path,
        expected_candidate=expected_candidate,
        expected_motions=expected_motions,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    mapping_path = root / "stage1_blind_private_mapping.json"
    if mapping_path.exists():
        existing = validate_stage1_peasd_blind_mapping(
            mapping_path,
            expected_candidate=endpoint["candidate"],
            expected_motions=endpoint["validation_motion_paths"],
        )
        if (
            Path(str(existing["endpoint_review_set_path"]))
            .expanduser()
            .resolve(strict=True)
            != endpoint_path
        ):
            raise ValueError("existing Stage1 blind mapping belongs to another endpoint review set")
        return {
            "package_manifest": Path(existing["package_manifest_path"]),
            "review_template": Path(existing["review_template_path"]),
            "review_submission": Path(existing["review_submission_path"]),
            "private_mapping": mapping_path,
        }

    package_id = secrets.token_hex(16)
    package_root = root / "stage1_blind_review_package"
    if package_root.exists():
        raise ValueError("unbound Stage1 blind review package already exists")
    temporary_root = root / f".stage1_blind_review_package.{package_id}.tmp"
    clips_dir = temporary_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=False)
    package_clips = []
    mapping_clips = []
    for original in endpoint["clips"]:
        opaque_id = secrets.token_hex(16)
        relative_artifact = Path("clips") / f"{opaque_id}.mp4"
        source_artifact = _resolve_artifact_path(
            str(original["artifact"]),
            relative_to=endpoint_path.parent,
        )
        anonymous_artifact = clips_dir / f"{opaque_id}.mp4"
        shutil.copyfile(source_artifact, anonymous_artifact)
        artifact_sha = _sha256_file(anonymous_artifact)
        if artifact_sha != original["artifact_content_sha256"]:
            raise RuntimeError("anonymous Stage1 review copy differs from source video bytes")
        package_clips.append(
            {
                "opaque_clip_id": opaque_id,
                "artifact": relative_artifact.as_posix(),
                "artifact_content_sha256": artifact_sha,
            }
        )
        mapping_clips.append(
            {
                "opaque_clip_id": opaque_id,
                "anonymous_artifact_path": str(package_root / relative_artifact),
                "anonymous_artifact_content_sha256": artifact_sha,
                "source_motion": str(original["motion"]),
                "source_artifact_path": str(source_artifact),
                "source_artifact_content_sha256": str(original["artifact_content_sha256"]),
            }
        )
    package_core = {
        "schema_version": STAGE1_BLIND_PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "clips": package_clips,
    }
    package = {
        **package_core,
        "package_content_sha256": _canonical_sha256(package_core),
    }
    _atomic_write_json(temporary_root / "package_manifest.json", package)
    review_template = {
        "schema_version": STAGE1_BLIND_REVIEW_SCHEMA_VERSION,
        "package_id": package_id,
        "package_manifest": "package_manifest.json",
        "package_content_sha256": package["package_content_sha256"],
        "reviewer_id": None,
        "clips": [
            {
                "opaque_clip_id": clip["opaque_clip_id"],
                "artifact": clip["artifact"],
                "major_swing_complete": None,
                "root_tracking_spike_free": None,
                "right_hand_tracking_spike_free": None,
                "passed": None,
                "notes": None,
            }
            for clip in package_clips
        ],
        "passed": None,
    }
    _reject_reviewer_visible_identity(
        package,
        candidate=endpoint["candidate"],
        motions=endpoint["validation_motion_paths"],
    )
    _reject_reviewer_visible_identity(
        review_template,
        candidate=endpoint["candidate"],
        motions=endpoint["validation_motion_paths"],
    )
    _atomic_write_json(temporary_root / "review_template.json", review_template)
    # The pristine template remains hash-bound by the private mapping.  The
    # reviewer fills this separate anonymous submission in place.
    _atomic_write_json(temporary_root / "review.json", review_template)
    validate_stage1_peasd_blind_package(temporary_root / "package_manifest.json")
    _validate_stage1_peasd_blind_review_template(
        temporary_root / "review_template.json",
        package=package,
        candidate=endpoint["candidate"],
        motions=endpoint["validation_motion_paths"],
    )
    temporary_root.replace(package_root)
    package_manifest_path = package_root / "package_manifest.json"
    review_template_path = package_root / "review_template.json"
    review_submission_path = package_root / "review.json"
    mapping_unsigned = {
        "schema_version": STAGE1_BLIND_MAPPING_SCHEMA_VERSION,
        "package_id": package_id,
        "review_package_dir": str(package_root),
        "package_manifest_path": str(package_manifest_path),
        "package_manifest_content_sha256": _sha256_file(package_manifest_path),
        "package_content_sha256": package["package_content_sha256"],
        "review_template_path": str(review_template_path),
        "review_template_content_sha256": _sha256_file(review_template_path),
        "review_submission_path": str(review_submission_path),
        "endpoint_review_set_path": str(endpoint_path),
        "endpoint_review_set_content_sha256": _sha256_file(endpoint_path),
        "endpoint_review_set_binding_sha256": endpoint["binding_sha256"],
        "candidate": dict(endpoint["candidate"]),
        "clips": mapping_clips,
    }
    mapping = {**mapping_unsigned, "binding_sha256": _canonical_sha256(mapping_unsigned)}
    _atomic_write_json(mapping_path, mapping)
    validate_stage1_peasd_blind_mapping(
        mapping_path,
        expected_candidate=endpoint["candidate"],
        expected_motions=endpoint["validation_motion_paths"],
    )
    return {
        "package_manifest": package_manifest_path,
        "review_template": review_template_path,
        "review_submission": review_submission_path,
        "private_mapping": mapping_path,
    }


def validate_stage1_peasd_blind_review(
    review_source: str | Path,
    *,
    private_mapping: str | Path,
    expected_candidate: Mapping,
    expected_motions: list[str] | tuple[str, ...],
) -> dict:
    """Reconstruct a formal visual review without trusting claimed blinding."""

    review_path = Path(review_source).expanduser().resolve(strict=True)
    mapping = validate_stage1_peasd_blind_mapping(
        private_mapping,
        expected_candidate=expected_candidate,
        expected_motions=expected_motions,
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or set(review) != {
        "schema_version",
        "package_id",
        "package_manifest",
        "package_content_sha256",
        "reviewer_id",
        "clips",
        "passed",
    }:
        raise ValueError("formal Stage1 PEASD requires the opaque blind review schema")
    if review.get("schema_version") != STAGE1_BLIND_REVIEW_SCHEMA_VERSION:
        raise ValueError("formal Stage1 PEASD rejects legacy/self-declared visual blinding")
    package_manifest_path = Path(mapping["package_manifest_path"]).resolve(strict=True)
    if (
        review_path.name != "review.json"
        or review.get("package_id") != mapping.get("package_id")
        or review.get("package_manifest") != "package_manifest.json"
        or review.get("package_content_sha256") != mapping.get("package_content_sha256")
        or review_path.parent != package_manifest_path.parent
    ):
        raise ValueError("Stage1 blind review belongs to a foreign reviewer package")
    package = validate_stage1_peasd_blind_package(package_manifest_path)
    reviewer_id = str(review.get("reviewer_id", "") or "").strip()
    if not reviewer_id:
        raise ValueError("Stage1 blind review requires an identified reviewer")
    raw_clips = review.get("clips")
    if not isinstance(raw_clips, list) or len(raw_clips) != len(package["clips"]):
        raise ValueError("Stage1 blind review is missing one or more opaque clips")
    decisions = []
    for index, (raw, packaged, mapped) in enumerate(
        zip(raw_clips, package["clips"], mapping["clips"], strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != {
            "opaque_clip_id",
            "artifact",
            "major_swing_complete",
            "root_tracking_spike_free",
            "right_hand_tracking_spike_free",
            "passed",
            "notes",
        }:
            raise ValueError(f"Stage1 blind review clip {index} schema mismatch")
        if (
            raw.get("opaque_clip_id") != packaged.get("opaque_clip_id")
            or raw.get("artifact") != packaged.get("artifact")
        ):
            raise ValueError("Stage1 blind review clip identity/order changed")
        for field in (
            "major_swing_complete",
            "root_tracking_spike_free",
            "right_hand_tracking_spike_free",
            "passed",
        ):
            if raw.get(field) is not True:
                raise ValueError(f"Stage1 blind review clips[{index}].{field} must be true")
        notes = str(raw.get("notes", "") or "").strip()
        if not notes:
            raise ValueError(f"Stage1 blind review clips[{index}].notes is required")
        anonymous_path = (
            package_manifest_path.parent / str(packaged["artifact"])
        ).resolve(strict=True)
        if _sha256_file(anonymous_path) != packaged["artifact_content_sha256"]:
            raise ValueError("Stage1 blind review video bytes changed after review")
        decisions.append(
            {
                "review_kind": STAGE1_REVIEW_KIND,
                "motion": mapped["source_motion"],
                "artifact": str(anonymous_path),
                "artifact_content_sha256": packaged["artifact_content_sha256"],
                "major_swing_complete": True,
                "root_tracking_spike_free": True,
                "right_hand_tracking_spike_free": True,
                "passed": True,
                "notes": notes,
                "candidate": dict(expected_candidate),
            }
        )
    if review.get("passed") is not True:
        raise ValueError("Stage1 blind review top-level passed must be true")
    _reject_reviewer_visible_identity(
        review,
        candidate=expected_candidate,
        motions=expected_motions,
    )
    reconstructed = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_kind": STAGE1_REVIEW_KIND,
        "candidate": dict(expected_candidate),
        "clips": decisions,
        "passed": True,
    }
    from musclemimic.badminton.visual_review import validate_visual_review

    report = validate_visual_review(
        reconstructed,
        required_clips=len(expected_motions),
        expected_motions=expected_motions,
        required_review_kind=STAGE1_REVIEW_KIND,
        expected_candidate=expected_candidate,
    )
    if report.get("passed") is not True:
        raise ValueError(
            "Stage1 blind review could not reconstruct the endpoint review: "
            + "; ".join(str(item) for item in report.get("errors", ()))
        )
    return {
        "review": review,
        "private_mapping": mapping,
        "package": package,
        "reconstructed_visual_review": reconstructed,
        "validation_report": report,
    }


class ValidationVideoRecorder:
    """
    Host-side utility to record short evaluation rollouts during training.

    It reconstructs a standalone env (MJX by default) with headless rendering and
    invokes PPOJax.play_policy(record=True) to write a video using the built-in
    VideoRecorder. Designed to be called from the training logging callback.
    """

    def __init__(
        self,
        video_dir: str,
        frequency: int = 10,
        length: int = 500,
        deterministic: bool = True,
        cycle_trajectories: bool = False,
        max_recordings: int | None = None,
        review_kind: str | None = None,
    ):
        """
        Args:
            video_dir: Base directory where videos are written.
            frequency: Record every N validation callbacks.
            length: Number of steps to record per episode.
            deterministic: Use deterministic policy for reproducibility.
        """
        self.video_dir = video_dir
        self.frequency = max(1, int(frequency))
        self.length = max(1, int(length))
        self.deterministic = deterministic
        self.cycle_trajectories = bool(cycle_trajectories)
        self.max_recordings = (
            None if max_recordings is None else max(0, int(max_recordings))
        )
        if review_kind is not None and review_kind not in REVIEW_KINDS:
            raise ValueError(f"unsupported validation visual review kind: {review_kind!r}")
        self.review_kind = review_kind

    def _selected_validation_motion(self, agent_conf, validation_number: int):
        if not self.cycle_trajectories:
            return None
        validation = getattr(agent_conf.config.experiment, "validation", {})
        dataset = validation.get("amass_dataset_conf", {})
        paths = list(dataset.get("rel_dataset_path", []) or [])
        if not paths:
            return None
        recording_number = max(0, int(validation_number) // self.frequency - 1)
        trajectory_index = recording_number % len(paths)
        return trajectory_index, str(paths[trajectory_index])

    def _write_visual_candidate(
        self,
        *,
        motion: str,
        artifact: str,
        validation_number: int,
        timestep: int | None,
        candidate_identity: Mapping | None = None,
    ) -> None:
        if self.review_kind is None:
            return
        prefix = "stage1" if self.review_kind == STAGE1_REVIEW_KIND else "stage2"
        path = Path(self.video_dir) / f"{prefix}_visual_review_candidates.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "review_kind": self.review_kind,
            "motion": motion,
            "artifact": artifact,
            "validation_number": int(validation_number),
            "timestep": None if timestep is None else int(timestep),
            "major_swing_complete": None,
            "root_tracking_spike_free": None,
            "right_hand_tracking_spike_free": None,
            "passed": None,
            "notes": None,
        }
        if candidate_identity is not None:
            row["candidate"] = dict(candidate_identity)
        if self.review_kind == STAGE2_REVIEW_KIND:
            row.update(
                {
                    "racket_head_trajectory_ok": None,
                    "racket_face_orientation_ok": None,
                }
            )

        existing: list[dict] = []
        if path.is_file():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    previous = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid visual candidate JSONL at {path}:{line_number}"
                    ) from exc
                # Legacy candidate rows are readable historical output, but
                # cannot enter the structured production manifest.
                if (
                    isinstance(previous, dict)
                    and previous.get("schema_version") == CANDIDATE_SCHEMA_VERSION
                    and previous.get("review_kind") == self.review_kind
                ):
                    existing.append(previous)

        motion_key = _candidate_motion_key(motion)
        existing = [
            previous
            for previous in existing
            if _candidate_motion_key(str(previous.get("motion", ""))) != motion_key
        ]
        for previous in existing:
            if str(previous.get("artifact", "")).strip() == artifact.strip():
                raise ValueError(
                    "visual candidate artifact is already assigned to another motion: "
                    f"{artifact}"
                )
        existing.append(row)
        motion_keys = [_candidate_motion_key(str(item.get("motion", ""))) for item in existing]
        artifacts = [str(item.get("artifact", "")).strip() for item in existing]
        if len(set(motion_keys)) != len(motion_keys):
            raise ValueError("visual candidate motions must remain unique")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("visual candidate artifacts must remain unique")

        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in existing
            ),
            encoding="utf-8",
        )
        temp.replace(path)

    def _build_env_params(self, agent_conf, tag: str) -> dict:
        """Clone training env params and add recording-specific settings."""
        # Copy training env params.
        env_params = dict(agent_conf.config.experiment.env_params)

        # Switch to the MuJoCo CPU env for recording.
        env_name = env_params.get("env_name", "")
        if isinstance(env_name, str) and env_name.startswith("Mjx"):
            env_params["env_name"] = env_name.replace("Mjx", "", 1)
        # Drop MJX-only parameters.
        for k in ("mjx_backend", "num_envs", "nconmax", "njmax"):
            if k in env_params:
                env_params.pop(k, None)

        # Apply validation terminal-state settings.
        if hasattr(agent_conf.config.experiment, "validation"):
            validation_config = agent_conf.config.experiment.validation
            env_params["terminal_state_type"] = validation_config.get("terminal_state_type", "NoTerminalStateHandler")
            env_params["terminal_state_params"] = dict(validation_config.get("terminal_state_params", {}))
        else:
            env_params["terminal_state_type"] = "NoTerminalStateHandler"
            env_params["terminal_state_params"] = {}

        # Configure headless recording.
        env_params["headless"] = True
        # Enable goal visualization during recording.
        env_params["visualize_goal"] = True
        # Match recording FPS to the control rate.
        timestep = env_params.get("timestep", 0.002)
        n_substeps = env_params.get("n_substeps", 5)
        control_dt = timestep * n_substeps
        fps = int(round(1.0 / control_dt))
        env_params["recorder_params"] = {
            "path": self.video_dir,
            "tag": tag,
            "video_name": f"{env_params.get('env_name', 'env')}",
            "fps": fps,
            "compress": True,
        }

        # Mirror visualization settings into goal_params.
        goal_params = dict(env_params.get("goal_params", {}))
        goal_params["visualize_goal"] = True
        # Enable enhanced goal visualization when supported.
        goal_params.setdefault("enable_enhanced_visualization", True)
        goal_params.setdefault("target_geom_rgba", [0.471, 0.38, 0.812, 0.6])
        env_params["goal_params"] = goal_params

        # Use visualization-specific goal classes.
        env_name = env_params.get("env_name", "")
        sites = goal_params.get("sites_for_mimic", [])
        if "Bimanual" in env_name:
            env_params["goal_type"] = "GoalBimanualTrajMimicv2"
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites
        elif "MyoFullBody" in env_name:
            # Fullbody uses GoalTrajMimicv2.
            env_params["goal_type"] = "GoalTrajMimicv2"
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites

        # Reuse training timing parameters.
        for k in ("timestep", "n_substeps"):
            if k in agent_conf.config.experiment.env_params:
                env_params[k] = agent_conf.config.experiment.env_params[k]

        # Start each validation rollout at trajectory step 0.
        if hasattr(agent_conf.config.experiment, "validation"):
            if agent_conf.config.experiment.validation.get("start_from_beginning", False):
                if "th_params" not in env_params:
                    env_params["th_params"] = {}
                env_params["th_params"]["start_from_random_step"] = False

        return env_params

    def _build_task_params(self, agent_conf) -> dict:
        """Clone task params and apply validation-specific dataset overrides."""
        raw_task_params = agent_conf.config.experiment.task_factory.params
        if OmegaConf.is_config(raw_task_params):
            task_params = OmegaConf.to_container(raw_task_params, resolve=True)
        else:
            task_params = dict(raw_task_params) if raw_task_params else {}

        if hasattr(agent_conf.config.experiment, "validation"):
            validation_config = agent_conf.config.experiment.validation
            for key in ("amass_dataset_conf", "dataset_conf", "trajectory_dataset_conf"):
                val_dataset = validation_config.get(key, None)
                if val_dataset is not None:
                    task_params[key] = (
                        OmegaConf.to_container(val_dataset, resolve=True)
                        if OmegaConf.is_config(val_dataset)
                        else val_dataset
                    )

        amass_conf = task_params.get("amass_dataset_conf")
        if isinstance(amass_conf, dict):
            amass_conf = dict(amass_conf)
            motion_paths = list(amass_conf.get("rel_dataset_path", []) or [])
            if self.cycle_trajectories and motion_paths:
                # Every held-out motion must be present so the scheduled
                # recorder cycle can generate one candidate for each clip.
                amass_conf["max_motions"] = len(motion_paths)
            else:
                amass_conf.setdefault("max_motions", 3)
            task_params["amass_dataset_conf"] = amass_conf
        return task_params

    def record_episode(
        self,
        agent_conf,
        agent_state,
        validation_number: int,
        timestep: int | None = None,
        *,
        motion_index: int | None = None,
        candidate_identity: Mapping | None = None,
    ) -> str | None:
        """
        Record a single short rollout if the frequency condition matches.

        Args:
            agent_conf: PPO agent configuration (contains network and saved config).
            agent_state: PPO agent state; only params and run_stats are used.
            validation_number: Current validation counter (1-based).
            timestep: Global training timestep for naming.

        Returns:
            Path to the recorded video file if available, else None.
        """
        if motion_index is None and validation_number % self.frequency != 0:
            return None
        recording_number = int(validation_number) // self.frequency
        if (
            motion_index is None
            and self.max_recordings is not None
            and recording_number > self.max_recordings
        ):
            return None

        setup_headless_rendering_if_needed()

        # Always use MuJoCo CPU env for evaluation visualization
        use_mujoco = True

        # Build a recording tag.
        time_tag = time.strftime("%Y%m%d_%H%M%S")
        if motion_index is None:
            selected = self._selected_validation_motion(agent_conf, validation_number)
        else:
            validation = getattr(agent_conf.config.experiment, "validation", {})
            dataset = validation.get("amass_dataset_conf", {})
            paths = list(dataset.get("rel_dataset_path", []) or [])
            if not 0 <= int(motion_index) < len(paths):
                raise ValueError(
                    "review-set motion index is outside the validation motion list"
                )
            selected = (int(motion_index), str(paths[int(motion_index)]))
        trajectory_tag = "" if selected is None else f"_traj{selected[0]}"
        tag = (
            f"validation_{validation_number}{trajectory_tag}_"
            f"t{timestep if timestep is not None else 0}_{time_tag}"
        )

        # Build the evaluation environment.
        factory = TaskFactory.get_factory_cls(agent_conf.config.experiment.task_factory.name)
        env_params = self._build_env_params(agent_conf, tag)
        task_params = self._build_task_params(agent_conf)

        # Isolate StatefulObject indices for the recorder env.
        saved_instances = StatefulObject._instances.copy()
        StatefulObject._instances.clear()
        env = None
        try:
            print(f"[ValidationVideo] Building eval env for recording (tag={tag})...")
            # Create the recorder environment.
            try:
                env = factory.make(**env_params, **task_params)
            except Exception as e:
                print(f"[ValidationVideo] ERROR: Failed to create environment: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

            if selected is not None:
                trajectory_index, motion_path = selected
                if not hasattr(env, "th") or env.th is None:
                    raise ValueError(
                        "cycle_trajectories requires a trajectory-backed validation environment"
                    )
                if trajectory_index >= int(env.th.n_trajectories):
                    raise ValueError(
                        "selected validation trajectory is absent from recorder environment: "
                        f"index={trajectory_index} count={env.th.n_trajectories}"
                    )
                env.th.fixed_start_conf = [int(trajectory_index), 0]
                env.th.use_fixed_start = True
                env.th.random_start = False

            fps = env_params["recorder_params"]["fps"]
            print(f"[ValidationVideo] Eval env ready; starting rollout for {self.length} steps @ {fps} fps")

            # Keep isolation active through the internal reset in play_policy.
            PPOJax.play_policy(
                env,
                agent_conf,
                agent_state,
                n_envs=1,
                n_steps=self.length,
                render=True,  # must be True for recording to emit frames
                record=True,
                deterministic=self.deterministic,
                use_mujoco=use_mujoco,
                wrap_env=True,
                train_state_seed=0,
                freeze_run_stats=candidate_identity is not None,
            )
        finally:
            # Restore global state even on failure.
            if env is not None:
                env.stop()
            StatefulObject._instances = saved_instances

        # Return the recorded video path when available.
        video_path = env.video_file_path if env is not None else None
        if video_path and selected is not None:
            self._write_visual_candidate(
                motion=selected[1],
                artifact=str(video_path),
                validation_number=validation_number,
                timestep=timestep,
                candidate_identity=candidate_identity,
            )
        return video_path

    def record_review_set(
        self,
        *,
        agent_conf,
        agent_state,
        validation_number: int,
        timestep: int,
        candidate_identity: Mapping,
    ) -> list[str]:
        """Record every held-out motion from one frozen promotion candidate."""

        if self.review_kind is None:
            raise ValueError("review-set recording requires a structured review kind")
        validation = getattr(agent_conf.config.experiment, "validation", {})
        paths = list(
            validation.get("amass_dataset_conf", {}).get("rel_dataset_path", [])
            or []
        )
        if not paths:
            raise ValueError("review-set recording has no held-out validation motions")
        artifacts: list[str] = []
        for motion_index in range(len(paths)):
            artifact = self.record_episode(
                agent_conf=agent_conf,
                agent_state=agent_state,
                validation_number=validation_number,
                timestep=timestep,
                motion_index=motion_index,
                candidate_identity=candidate_identity,
            )
            if not artifact:
                raise RuntimeError(
                    f"review-set recording produced no artifact for {paths[motion_index]}"
                )
            artifacts.append(artifact)
        resolved_artifacts: list[Path] = []
        for artifact in artifacts:
            artifact_path = Path(artifact).expanduser().resolve(strict=True)
            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                raise RuntimeError(f"review-set artifact is missing or empty: {artifact_path}")
            resolved_artifacts.append(artifact_path)

        # Seal one complete set only after every held-out motion has rendered.
        # The adjacent template can be filled by a blinded reviewer without
        # manually reconstructing checkpoint or clip identities.
        prefix = "stage1" if self.review_kind == STAGE1_REVIEW_KIND else "stage2"
        clips = [
            {
                "motion": str(motion),
                "artifact": str(artifact),
                "artifact_content_sha256": _sha256_file(artifact),
            }
            for motion, artifact in zip(paths, resolved_artifacts, strict=True)
        ]
        unsigned = {
            "schema_version": (
                ENDPOINT_REVIEW_SET_SCHEMA_VERSION
                if self.review_kind == STAGE1_REVIEW_KIND
                else STAGE2_ENDPOINT_REVIEW_SET_SCHEMA_VERSION
            ),
            "review_kind": self.review_kind,
            "candidate": dict(candidate_identity),
            "deterministic_policy": True,
            "run_stats_frozen": True,
            "validation_motion_paths": [str(path) for path in paths],
            "clips": clips,
        }
        review_set = {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}
        review_set_path = Path(self.video_dir) / f"{prefix}_endpoint_review_set.json"
        _atomic_write_json(review_set_path, review_set)
        if self.review_kind == STAGE1_REVIEW_KIND:
            validate_stage1_endpoint_review_set(
                review_set_path,
                expected_candidate=candidate_identity,
                expected_motions=[str(path) for path in paths],
            )
            blind_paths = build_stage1_peasd_blind_review_package(
                endpoint_review_set=review_set_path,
                output_dir=self.video_dir,
                expected_candidate=candidate_identity,
                expected_motions=[str(path) for path in paths],
            )
            print(
                "[ValidationVideo] Stage1 reviewer package (share this directory only): "
                f"{blind_paths['package_manifest'].parent}"
            )
            print(
                "[ValidationVideo] Reviewer fills anonymous submission (leave template pristine): "
                f"{blind_paths['review_submission']}"
            )
            print(
                "[ValidationVideo] Stage1 private identity mapping (do not share): "
                f"{blind_paths['private_mapping']}"
            )
            return artifacts

        # Stage2 retains the general identity-bearing structured-review schema.
        # Formal Stage1 PEASD never writes this template because it would expose
        # the selected treatment arm, seed, checkpoint and motion labels.
        template = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "review_kind": self.review_kind,
            "candidate": dict(candidate_identity),
            "blinding": {
                "reviewer_id": None,
                "reviewer_blinded_to_arm": None,
                "reviewer_blinded_to_seed": None,
            },
            "source_review_set": {
                "path": str(review_set_path.resolve()),
                "content_sha256": _sha256_file(review_set_path),
                "binding_sha256": review_set["binding_sha256"],
            },
            "clips": [
                {
                    "review_kind": self.review_kind,
                    "candidate": dict(candidate_identity),
                    **clip,
                    "major_swing_complete": None,
                    "root_tracking_spike_free": None,
                    "right_hand_tracking_spike_free": None,
                    **(
                        {
                            "racket_head_trajectory_ok": None,
                            "racket_face_orientation_ok": None,
                        }
                        if self.review_kind == STAGE2_REVIEW_KIND
                        else {}
                    ),
                    "passed": None,
                    "notes": None,
                }
                for clip in clips
            ],
            "passed": None,
        }
        _atomic_write_json(
            Path(self.video_dir) / f"{prefix}_visual_review_template.json",
            template,
        )
        return artifacts
