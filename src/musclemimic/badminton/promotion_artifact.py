"""Cryptographically bound Stage-1/Stage-2 promotion artifacts.

The numerical gate, checkpoint and human visual review are produced by
different processes.  A path-only hand-off is unsafe: a newer checkpoint or a
review copied from another run could silently satisfy the next stage.  This
module creates and validates one immutable manifest that binds all three to the
same run, PPO update, global timestep and configuration hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "forehand_clear_promoted_artifact_v2"
CHECKPOINT_IDENTITY_SCHEMA_VERSION = "forehand_clear_checkpoint_identity_v1"
SUPPORTED_STAGES = frozenset({"stage1", "stage2"})
PROMOTION_PROGRESS_SCHEMA_VERSION = "forehand_clear_promotion_progress_v3"


def sha256_path(path: str | Path) -> str:
    """Return a content hash for a file or a complete directory tree.

    Directory names, file names, sizes and bytes are included.  Symlinks are
    resolved before hashing so a retargeted ``latest`` pointer cannot preserve
    the old digest while changing its payload.
    """

    source = Path(path).expanduser()
    if source.is_symlink():
        source = source.resolve(strict=True)
    if source.is_file():
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(f"artifact path does not exist: {source}")

    digest = hashlib.sha256()
    digest.update(b"directory-tree-v1\0")
    entries = sorted(
        (entry for entry in source.rglob("*") if entry.is_file() or entry.is_symlink()),
        key=lambda entry: entry.relative_to(source).as_posix(),
    )
    if not entries:
        raise ValueError(f"checkpoint directory is empty: {source}")
    for entry in entries:
        relative = entry.relative_to(source).as_posix().encode("utf-8")
        resolved = entry.resolve(strict=True) if entry.is_symlink() else entry
        if not resolved.is_file():
            raise ValueError(f"checkpoint tree contains a non-file symlink: {entry}")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(int(resolved.stat().st_size).to_bytes(8, "big"))
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: str | Path) -> dict[str, Any]:
    """Read an Orbax PPO checkpoint and return its immutable identity."""

    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Stage-1/Stage-2 checkpoint must be a directory: {path}")
    metadata_path = path / "metadata" / "metadata"
    if not metadata_path.is_file():
        raise ValueError(f"checkpoint metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint metadata is unreadable: {metadata_path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a JSON object")
    try:
        update_number = int(metadata["update_number"])
        global_timestep = int(metadata["global_timestep"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint metadata requires integer update_number/global_timestep"
        ) from exc
    if update_number < 0 or global_timestep < 0:
        raise ValueError("checkpoint update/global timestep must be non-negative")

    checkpoint_dir = path.parent.resolve()
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"checkpoint run manifest is missing: {manifest_path}")
    try:
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint run manifest is unreadable: {manifest_path}") from exc
    config_hash = str(run_manifest.get("config_hash", ""))
    if not config_hash:
        raise ValueError("checkpoint run manifest has no config_hash")

    parent_lineage = run_manifest.get("parent_checkpoint_lineage")
    experiment_config = run_manifest.get("experiment_config")
    nested_lineage = None
    if isinstance(experiment_config, Mapping):
        parent_config = experiment_config.get("parent_checkpoint_lineage")
        if isinstance(parent_config, Mapping):
            nested_lineage = parent_config.get("identity")
    if parent_lineage is None:
        parent_lineage = nested_lineage
    elif nested_lineage != parent_lineage:
        raise ValueError(
            "checkpoint run manifest parent lineage differs from experiment config"
        )
    if parent_lineage is not None:
        from musclemimic.runner.checkpointing import (
            validate_parent_checkpoint_lineage,
        )

        parent_lineage = validate_parent_checkpoint_lineage(parent_lineage)

    # The directory name is a storage location, not a run identity. Prefer the
    # bound manifest declarations and use config_hash as the stable fallback
    # for older manifests, so copying an immutable run tree does not change its
    # content lineage.
    config_run_id = None
    if isinstance(experiment_config, Mapping):
        raw_run_id = experiment_config.get("run_id")
        if raw_run_id not in (None, ""):
            config_run_id = str(raw_run_id)
    manifest_run_identity = run_manifest.get("run_identity")
    identity_run_id = None
    if isinstance(manifest_run_identity, Mapping):
        raw_identity_id = manifest_run_identity.get("experiment_id")
        if raw_identity_id not in (None, ""):
            identity_run_id = str(raw_identity_id)
        if manifest_run_identity.get("config_hash") != config_hash:
            raise ValueError("checkpoint run identity config hash is inconsistent")
        if manifest_run_identity.get("parent_checkpoint_lineage") != parent_lineage:
            raise ValueError("checkpoint run identity parent lineage is inconsistent")
        unsigned_run_identity = {
            "schema_version": manifest_run_identity.get("schema_version"),
            "experiment_id": identity_run_id,
            "config_hash": config_hash,
            "parent_checkpoint_lineage": parent_lineage,
        }
        if manifest_run_identity.get("binding_sha256") != _mapping_sha256(
            unsigned_run_identity
        ):
            raise ValueError("checkpoint run identity binding hash is stale")
    if config_run_id is not None and identity_run_id is not None:
        if config_run_id != identity_run_id:
            raise ValueError("checkpoint run identity differs from configured run_id")
    stable_run_id = config_run_id or identity_run_id or config_hash

    result = {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
        "checkpoint_path": str(path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_content_sha256": sha256_path(path),
        "metadata_content_sha256": sha256_path(metadata_path),
        "run_manifest_content_sha256": sha256_path(manifest_path),
        "update_number": update_number,
        "global_timestep": global_timestep,
        "target_global_timestep": int(metadata.get("target_global_timestep", 0) or 0),
        "config_hash": config_hash,
        "run_id": stable_run_id,
    }
    if parent_lineage is not None:
        result["parent_checkpoint_lineage"] = parent_lineage
    return result


def build_promoted_artifact(
    *,
    stage: str,
    checkpoint: str | Path,
    promotion_progress: str | Path,
    visual_review: str | Path,
    parent_promoted_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed promotion manifest from current source artifacts."""

    stage_key = str(stage).lower()
    if stage_key not in SUPPORTED_STAGES:
        raise ValueError(f"unsupported promoted stage: {stage!r}")
    identity = checkpoint_identity(checkpoint)
    progress_path = Path(promotion_progress).expanduser().resolve(strict=True)
    review_path = Path(visual_review).expanduser().resolve(strict=True)
    progress = _load_mapping(progress_path, "promotion progress")
    review = _load_mapping(review_path, "visual review")
    if progress.get("schema_version") != PROMOTION_PROGRESS_SCHEMA_VERSION:
        raise ValueError("promotion progress schema is incompatible")

    parent_binding: dict[str, Any] | None = None
    if stage_key == "stage2":
        if parent_promoted_artifact is None:
            raise ValueError("Stage-2 promotion requires its Stage-1 promoted parent")
        parent_path = Path(parent_promoted_artifact).expanduser().resolve(strict=True)
        parent = validate_promoted_artifact(parent_path, expected_stage="stage1")
        parent_checkpoint = parent.get("checkpoint")
        if not isinstance(parent_checkpoint, Mapping):
            raise ValueError("Stage-1 promoted parent has no checkpoint identity")
        if not _lineage_contains_checkpoint(
            identity.get("parent_checkpoint_lineage"),
            parent_checkpoint,
        ):
            raise ValueError(
                "Stage-2 checkpoint lineage does not contain its promoted Stage-1 parent"
            )
        baseline_value = progress.get("baseline_metrics_path")
        baseline_sha256 = progress.get("baseline_metrics_sha256")
        if not isinstance(baseline_value, str) or not baseline_value:
            raise ValueError("Stage-2 promotion progress has no Stage-1 baseline path")
        baseline_path = Path(baseline_value).expanduser().resolve(strict=True)
        if not isinstance(baseline_sha256, str) or baseline_sha256 != sha256_path(
            baseline_path
        ):
            raise ValueError("Stage-2 baseline metrics content identity is missing or stale")
        parent_progress = parent.get("promotion_progress")
        if not isinstance(parent_progress, Mapping):
            raise ValueError("Stage-1 promoted parent has no progress identity")
        parent_progress_path = Path(str(parent_progress.get("path", ""))).expanduser().resolve(
            strict=True
        )
        if baseline_path != parent_progress_path:
            raise ValueError("Stage-2 baseline is not its promoted Stage-1 parent progress")
        if baseline_sha256 != parent_progress.get("content_sha256"):
            raise ValueError("Stage-2 baseline content differs from promoted Stage-1 progress")
        parent_binding = {
            "path": str(parent_path),
            "content_sha256": sha256_path(parent_path),
            "binding_sha256": parent.get("binding_sha256"),
            "checkpoint": parent_checkpoint,
            "promotion_progress_content_sha256": parent_progress.get("content_sha256"),
        }
    elif parent_promoted_artifact is not None:
        raise ValueError("Stage-1 promotion must not declare a promoted parent")

    if str(progress.get("stage", "")).lower() != stage_key:
        raise ValueError("promotion progress belongs to a different stage")
    if Path(str(progress.get("checkpoint_dir", ""))).expanduser().resolve() != Path(
        identity["checkpoint_dir"]
    ):
        raise ValueError("promotion progress and checkpoint belong to different runs")
    if str(progress.get("config_hash", "")) != identity["config_hash"]:
        raise ValueError("promotion progress config hash differs from checkpoint")
    history = progress.get("history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], Mapping):
        raise ValueError("promotion progress requires a non-empty history")
    tail = dict(history[-1])
    _require_same_candidate(tail, identity, label="promotion progress tail")
    provenance = tail.get("validation_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("semantics") != (
        "evaluate_all_once_per_heldout_v1"
    ):
        raise ValueError("promotion progress tail lacks strict held-out provenance")
    if tail.get("passed") is not True:
        raise ValueError("latest promotion validation did not pass")
    if progress.get("stopped_early") is not True:
        raise ValueError("promotion progress has not completed its consecutive gate")
    recorded_identity = tail.get("checkpoint_identity")
    if not isinstance(recorded_identity, Mapping):
        raise ValueError("promotion progress tail has no checkpoint identity")
    _require_identity_equal(recorded_identity, identity, label="promotion progress")

    candidate = review.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("production visual review requires a candidate identity")
    _require_identity_equal(candidate, identity, label="visual review")
    from musclemimic.badminton.visual_review import (
        STAGE1_REVIEW_KIND,
        STAGE2_REVIEW_KIND,
        validate_visual_review,
    )

    review_report = validate_visual_review(
        review,
        required_clips=5,
        required_review_kind=(
            STAGE1_REVIEW_KIND if stage_key == "stage1" else STAGE2_REVIEW_KIND
        ),
        expected_candidate=identity,
    )
    if review_report["passed"] is not True:
        raise ValueError(
            "structured visual review did not pass: "
            + "; ".join(str(error) for error in review_report["errors"])
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage_key,
        "parent_promoted_artifact": parent_binding,
        "checkpoint": identity,
        "promotion_progress": {
            "path": str(progress_path),
            "content_sha256": sha256_path(progress_path),
            "validation_count": int(progress.get("validation_count", len(history))),
            "consecutive_pass_streak": int(progress.get("consecutive_pass_streak", 0)),
            "tail": tail,
        },
        "visual_review": {
            "path": str(review_path),
            "content_sha256": sha256_path(review_path),
            "schema_version": review.get("schema_version"),
            "review_kind": review.get("review_kind"),
            "candidate": dict(candidate),
        },
    }
    payload["binding_sha256"] = _mapping_sha256(payload)
    return payload


def validate_promoted_artifact(
    manifest: str | Path,
    *,
    expected_stage: str,
    expected_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute every source hash before a downstream stage consumes it."""

    path = Path(manifest).expanduser().resolve(strict=True)
    payload = _load_mapping(path, "promoted artifact")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("promoted artifact schema is incompatible")
    if str(payload.get("stage", "")).lower() != str(expected_stage).lower():
        raise ValueError("promoted artifact belongs to a different stage")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("promoted artifact checkpoint identity is missing")
    checkpoint_path = Path(str(checkpoint.get("checkpoint_path", "")))
    if expected_checkpoint is not None:
        expected = Path(expected_checkpoint).expanduser().resolve(strict=True)
        if checkpoint_path.expanduser().resolve(strict=True) != expected:
            raise ValueError("promoted artifact points to a different checkpoint")

    rebuilt = build_promoted_artifact(
        stage=str(payload["stage"]),
        checkpoint=checkpoint_path,
        promotion_progress=str(payload.get("promotion_progress", {}).get("path", "")),
        visual_review=str(payload.get("visual_review", {}).get("path", "")),
        parent_promoted_artifact=(
            None
            if payload.get("parent_promoted_artifact") is None
            else str(payload.get("parent_promoted_artifact", {}).get("path", ""))
        ),
    )
    if rebuilt != payload:
        raise ValueError("promoted artifact or one of its bound sources changed")
    return rebuilt


def write_promoted_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _require_same_candidate(
    value: Mapping[str, Any], identity: Mapping[str, Any], *, label: str
) -> None:
    for key in ("update_number", "global_timestep"):
        try:
            matches = int(value[key]) == int(identity[key])
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"{label} {key} differs from checkpoint")


def _require_identity_equal(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    required = (
        "checkpoint_path",
        "checkpoint_dir",
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
        "update_number",
        "global_timestep",
        "config_hash",
        "run_id",
    )
    for key in required:
        if key not in actual or actual[key] != expected[key]:
            raise ValueError(f"{label} candidate {key} differs from checkpoint")
    if actual.get("parent_checkpoint_lineage") != expected.get(
        "parent_checkpoint_lineage"
    ):
        raise ValueError(
            f"{label} candidate parent checkpoint lineage differs from checkpoint"
        )


def _lineage_contains_checkpoint(
    lineage: Any,
    expected_checkpoint: Mapping[str, Any],
) -> bool:
    """Return whether a direct/transitive parent is the promoted checkpoint."""

    if not isinstance(lineage, Mapping):
        return False
    expected_keys = (
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
        "update_number",
        "global_timestep",
        "target_global_timestep",
        "config_hash",
        "run_id",
    )
    current = lineage.get("checkpoint")
    if isinstance(current, Mapping) and all(
        current.get(key) == expected_checkpoint.get(key) for key in expected_keys
    ):
        return True
    return _lineage_contains_checkpoint(
        lineage.get("parent_checkpoint_lineage"),
        expected_checkpoint,
    )


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(SUPPORTED_STAGES))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--promotion-progress", required=True)
    parser.add_argument("--visual-review", required=True)
    parser.add_argument("--parent-promoted-artifact", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    payload = build_promoted_artifact(
        stage=args.stage,
        checkpoint=args.checkpoint,
        promotion_progress=args.promotion_progress,
        visual_review=args.visual_review,
        parent_promoted_artifact=args.parent_promoted_artifact,
    )
    write_promoted_artifact(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
