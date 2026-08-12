from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

# Fields excluded from config hash (they don't affect training identity)
_HASH_EXCLUDE_FIELDS = frozenset(
    {
        "resume_from",
        "reset_logging_timestep",
        "checkpoint_dir",
        "checkpoint_root",
        "training_root",
        "validation_video_dir",
        "run_id",
        "auto_resume",
        # Injected from checkpoint metadata after the canonical run hash has
        # already been established.  It controls resume continuity but is not a
        # different training identity.
        "resume_lr_override",
        # Runtime-only opt-in used to add one more configured training budget after
        # a checkpoint has reached its stored hard cap.  It must not fork the run
        # identity because all model/data/optimizer hyperparameters stay unchanged.
        "extend_completed_run",
    }
)

_DATASET_ACTION_RE = re.compile(r"(?:^|[/\\])datasets[/\\]([^/\\]+)(?:[/\\]|$)")
_IGNORED_DATASET_ACTIONS = {"_global", "_index"}
_CANONICAL_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)")
PARENT_CHECKPOINT_LINEAGE_SCHEMA_VERSION = "resume_parent_checkpoint_lineage_v1"
PARENT_PROMOTION_BINDING_SCHEMA_VERSION = "resume_parent_promotion_binding_v1"
STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION = "stage1_source_tree_snapshot_v1"
STAGE1_SOURCE_TREE_SCOPES = (
    "fullbody",
    "musclemimic",
    "scripts",
    "configs",
    "analysis/latent_synergy",
    "environment/overall_environment/src",
    "experiments",
    "pyproject.toml",
    "uv.lock",
)
STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".toml", ".lock", ".sh"})
_PARENT_CHECKPOINT_IDENTITY_KEYS = (
    "checkpoint_content_sha256",
    "metadata_content_sha256",
    "run_manifest_content_sha256",
    "update_number",
    "global_timestep",
    "target_global_timestep",
    "config_hash",
    "run_id",
)


def _node_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _as_native(value: Any) -> Any:
    if OmegaConf.is_config(value):
        try:
            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            return OmegaConf.to_container(value, resolve=False)
    return value


def _iter_string_values(value: Any):
    value = _as_native(value)
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Path):
        yield str(value)
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_string_values(child)
        return
    if isinstance(value, list | tuple | set):
        for child in value:
            yield from _iter_string_values(child)


def infer_training_action(exp_config: Any) -> str | None:
    """Infer the dataset action name that should own training artifacts.

    Explicit ``training_action`` wins. Otherwise scan config strings for paths
    like ``datasets/<action>/...`` so action-specific configs can route their
    checkpoints and validation artifacts without duplicating paths.
    """
    exp = _node_get(exp_config, "experiment", exp_config)
    explicit = _node_get(exp, "training_action", None)
    if explicit:
        return str(explicit)

    for text in _iter_string_values(exp):
        match = _DATASET_ACTION_RE.search(text)
        if not match:
            continue
        action = match.group(1)
        if action not in _IGNORED_DATASET_ACTIONS:
            return action
    return None


def _resolve_path(path_like: str | Path, launch_dir: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = Path(launch_dir) / path
    return str(path)


def resolve_training_root(exp_config: Any, launch_dir: str | Path) -> str | None:
    """Resolve the per-action training artifact root.

    The default convention is ``datasets/<action>/training``. A configured
    ``training_root`` overrides that convention and may be relative to the
    launch directory.
    """
    exp = _node_get(exp_config, "experiment", exp_config)
    explicit_root = _node_get(exp, "training_root", None)
    if explicit_root:
        return _resolve_path(str(explicit_root), launch_dir)

    action = infer_training_action(exp)
    if not action:
        return None
    return str(Path(launch_dir) / "datasets" / action / "training")


def _is_checkpoint_complete(checkpoint_path: Path) -> bool:
    """Check if checkpoint has completed writing (Orbax finalized)."""
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        return False

    # A finalized multi-item Orbax checkpoint written by this repository has
    # both the root marker and the metadata item payload.  Checking these
    # local files is deterministic and avoids importing Orbax's async runtime
    # merely to scan a directory during auto-resume (which can block while JAX
    # backends are being initialized).  Temporary writes keep Orbax's
    # ``.orbax-checkpoint-tmp`` suffix and are rejected by the numeric directory
    # parser in ``find_latest_checkpoint``.
    return bool(
        (checkpoint_path / "_CHECKPOINT_METADATA").is_file() and (checkpoint_path / "metadata" / "metadata").is_file()
    )


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object without accepting duplicate keys or non-finite values."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _looks_like_orbax_checkpoint_leaf(path: Path) -> bool:
    """Return whether ``path`` exposes any repository Orbax leaf component."""

    return bool(
        (path / "train_state").is_dir()
        or (path / "config" / "metadata").is_file()
        or ((path / "_CHECKPOINT_METADATA").is_file() and (path / "metadata" / "metadata").is_file())
    )


def _resolve_checkpoint_validation_paths(
    checkpoint: str | Path,
) -> tuple[Path, Path, Path]:
    """Return ``(selected_path, restore_leaf, run_dir)`` for validation.

    ``selected_path`` and ``run_dir`` deliberately preserve the caller-visible
    path instead of resolving symlinks. A canonical ``checkpoint_N`` alias may
    point at a non-standard Orbax leaf in another directory, while its immutable
    run manifest still belongs to the alias's parent directory.
    """

    expanded = Path(checkpoint).expanduser()
    selected_input = Path(os.path.abspath(os.fspath(expanded)))
    if not selected_input.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint}")
    if not selected_input.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint}")

    if _CANONICAL_CHECKPOINT_RE.fullmatch(selected_input.name):
        selected_path = selected_input
        run_dir = selected_input.parent
    elif _looks_like_orbax_checkpoint_leaf(selected_input):
        selected_path = selected_input
        run_dir = selected_input.parent
    else:
        run_dir = selected_input
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            raise ValueError(f"No complete checkpoint leaf found in run directory: {run_dir}")
        selected_path = Path(latest)

    try:
        restore_leaf = selected_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint restore leaf cannot be resolved: {selected_path}") from exc
    if not restore_leaf.is_dir():
        raise ValueError(f"Checkpoint restore leaf is not a directory: {restore_leaf}")
    return selected_path, restore_leaf, run_dir


def _load_orbax_leaf_json_payloads(
    selected_path: Path,
    restore_leaf: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Strictly load and bind the JSON items from one finalized Orbax leaf."""

    if not required and not (restore_leaf / "config" / "metadata").is_file():
        return None

    marker = restore_leaf / "_CHECKPOINT_METADATA"
    train_state_dir = restore_leaf / "train_state"
    if not marker.is_file():
        raise ValueError(f"checkpoint restore leaf is not finalized: {marker}")
    if not train_state_dir.is_dir():
        raise ValueError(f"checkpoint restore leaf has no train_state item: {train_state_dir}")

    config_payload = _strict_json_object(
        restore_leaf / "config" / "metadata",
        label="checkpoint leaf config payload",
    )
    metadata_payload = _strict_json_object(
        restore_leaf / "metadata" / "metadata",
        label="checkpoint leaf metadata payload",
    )
    update_number = metadata_payload.get("update_number")
    if isinstance(update_number, bool) or not isinstance(update_number, int):
        raise ValueError("checkpoint leaf metadata requires an integer update_number")
    if update_number < 0:
        raise ValueError("checkpoint leaf metadata update_number must be non-negative")

    alias_match = _CANONICAL_CHECKPOINT_RE.fullmatch(selected_path.name)
    if alias_match is not None and update_number != int(alias_match.group(1)):
        raise ValueError(
            "checkpoint alias update does not match restore leaf metadata: "
            f"alias={alias_match.group(1)} metadata={update_number}"
        )
    return config_payload, metadata_payload


def find_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """Find latest complete checkpoint_* in directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to latest complete checkpoint directory, or None if none found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint_"):
            try:
                step = int(name.split("_")[1])
                ckpt_path = checkpoint_dir / name
                # Only include checkpoints with completed metadata
                if _is_checkpoint_complete(ckpt_path):
                    checkpoints.append((step, ckpt_path))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        return None

    return str(max(checkpoints, key=lambda x: x[0])[1])


def config_hash(config: Any, exclude: frozenset[str] | None = None) -> str:
    """Compute stable hash of experiment config (excludes volatile fields).

    Args:
        config: OmegaConf config or dict to hash.
        exclude: Additional fields to exclude from hash.

    Returns:
        12-character hex hash string.
    """
    exclude_set = _HASH_EXCLUDE_FIELDS | (exclude or frozenset())

    # Always use OmegaConf.to_container for deep conversion to native Python types
    if OmegaConf.is_config(config):
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = dict(config)

    def _remove_excluded(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in exclude_set}

    cfg_dict = _remove_excluded(cfg_dict)

    # Stable JSON serialization
    cfg_str = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:12]


def _mapping_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_parent_promotion_binding(value: Any) -> dict[str, Any]:
    native = _as_native(value)
    if not isinstance(native, dict):
        raise ValueError("parent promotion binding must be a mapping")
    expected = {
        "schema_version",
        "evidence_kind",
        "artifact_schema_version",
        "artifact_content_sha256",
        "artifact_binding_sha256",
        "checkpoint_content_sha256",
        "binding_sha256",
    }
    if set(native) != expected or native.get("schema_version") != (PARENT_PROMOTION_BINDING_SCHEMA_VERSION):
        raise ValueError("parent promotion binding schema is incompatible")
    if native.get("evidence_kind") not in {
        "verified_stage1_promotion_v2",
        "verified_stage1_peasd_promotion_v1",
    }:
        raise ValueError("parent promotion evidence kind is unsupported")
    if not isinstance(native.get("artifact_schema_version"), str) or not native["artifact_schema_version"]:
        raise ValueError("parent promotion artifact schema is missing")
    for field in (
        "artifact_content_sha256",
        "artifact_binding_sha256",
        "checkpoint_content_sha256",
    ):
        digest = native.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"parent promotion binding has invalid {field}")
    unsigned = {key: item for key, item in native.items() if key != "binding_sha256"}
    if native.get("binding_sha256") != _mapping_sha256(unsigned):
        raise ValueError("parent promotion binding hash is missing or stale")
    return dict(native)


def _build_parent_promotion_binding(
    manifest_path: str | Path,
    *,
    checkpoint_path: str | Path,
    role: str,
) -> dict[str, Any]:
    if str(role).strip() != "stage1_promoted":
        raise ValueError("a parent promotion manifest is supported only for role='stage1_promoted'")
    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("parent promotion manifest is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError("parent promotion manifest must be a JSON object")
    from musclemimic.badminton.stage1_peasd_gate import (
        PEASD_TEACHER_PROMOTION_SCHEMA_VERSION,
        validate_stage1_peasd_teacher_promotion,
    )

    if raw.get("schema_version") == PEASD_TEACHER_PROMOTION_SCHEMA_VERSION:
        artifact = validate_stage1_peasd_teacher_promotion(
            path,
            expected_checkpoint=checkpoint_path,
        )
        evidence_kind = "verified_stage1_peasd_promotion_v1"
    else:
        from musclemimic.badminton.promotion_artifact import validate_promoted_artifact

        artifact = validate_promoted_artifact(
            path,
            expected_stage="stage1",
            expected_checkpoint=checkpoint_path,
        )
        evidence_kind = "verified_stage1_promotion_v2"
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("parent promotion artifact has no checkpoint identity")
    unsigned = {
        "schema_version": PARENT_PROMOTION_BINDING_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "artifact_schema_version": str(artifact.get("schema_version", "")),
        "artifact_content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_binding_sha256": str(artifact.get("binding_sha256", "")),
        "checkpoint_content_sha256": str(checkpoint.get("checkpoint_content_sha256", "")),
    }
    return _validate_parent_promotion_binding({**unsigned, "binding_sha256": _mapping_sha256(unsigned)})


def validate_parent_checkpoint_lineage(
    lineage: Any,
    *,
    _depth: int = 0,
) -> dict[str, Any]:
    """Validate one path-independent, content-bound resume-parent chain."""

    if _depth > 16:
        raise ValueError("parent checkpoint lineage is unreasonably deep")
    native = _as_native(lineage)
    if not isinstance(native, dict):
        raise ValueError("parent checkpoint lineage must be a mapping")
    if native.get("schema_version") != PARENT_CHECKPOINT_LINEAGE_SCHEMA_VERSION:
        raise ValueError("parent checkpoint lineage schema is incompatible")
    role = native.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("parent checkpoint lineage role is missing")
    checkpoint = native.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("parent checkpoint lineage has no checkpoint identity")
    for key in _PARENT_CHECKPOINT_IDENTITY_KEYS:
        if key not in checkpoint:
            raise ValueError(f"parent checkpoint identity is missing {key}")
    for key in (
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
    ):
        value = checkpoint.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"parent checkpoint identity has invalid {key}")
    for key in ("update_number", "global_timestep", "target_global_timestep"):
        try:
            value = int(checkpoint[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parent checkpoint identity has invalid {key}") from exc
        if value < 0:
            raise ValueError(f"parent checkpoint identity has negative {key}")
    for key in ("config_hash", "run_id"):
        if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
            raise ValueError(f"parent checkpoint identity has invalid {key}")

    upstream = native.get("parent_checkpoint_lineage")
    if upstream is not None:
        validate_parent_checkpoint_lineage(upstream, _depth=_depth + 1)
    promotion = native.get("promotion")
    if promotion is not None:
        promotion = _validate_parent_promotion_binding(promotion)
        if promotion["checkpoint_content_sha256"] != checkpoint["checkpoint_content_sha256"]:
            raise ValueError("parent promotion binding points to another checkpoint")
    unsigned = {
        "schema_version": native["schema_version"],
        "role": role,
        "checkpoint": checkpoint,
        "parent_checkpoint_lineage": upstream,
    }
    if promotion is not None:
        unsigned["promotion"] = promotion
    if native.get("binding_sha256") != _mapping_sha256(unsigned):
        raise ValueError("parent checkpoint lineage binding hash is missing or stale")
    if set(native) != {*unsigned, "binding_sha256"}:
        raise ValueError("parent checkpoint lineage contains unsupported fields")
    return dict(native)


def build_parent_checkpoint_lineage(
    checkpoint_identity_payload: Any,
    *,
    role: str,
    promotion_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content identity injected before config hashing.

    Absolute paths are intentionally omitted: moving an immutable checkpoint
    does not create a new parent, while changing any checkpoint, metadata, or
    run-manifest byte does.
    """

    identity = _as_native(checkpoint_identity_payload)
    if not isinstance(identity, dict):
        raise ValueError("checkpoint identity must be a mapping")
    checkpoint: dict[str, Any] = {}
    for key in _PARENT_CHECKPOINT_IDENTITY_KEYS:
        if key not in identity:
            raise ValueError(f"checkpoint identity is missing {key}")
        checkpoint[key] = identity[key]
    upstream = identity.get("parent_checkpoint_lineage")
    if upstream is not None:
        upstream = validate_parent_checkpoint_lineage(upstream)
    unsigned = {
        "schema_version": PARENT_CHECKPOINT_LINEAGE_SCHEMA_VERSION,
        "role": str(role),
        "checkpoint": checkpoint,
        "parent_checkpoint_lineage": upstream,
    }
    if promotion_binding is not None:
        promotion = _validate_parent_promotion_binding(promotion_binding)
        if promotion["checkpoint_content_sha256"] != checkpoint["checkpoint_content_sha256"]:
            raise ValueError("parent promotion binding points to another checkpoint")
        unsigned["promotion"] = promotion
    lineage = {**unsigned, "binding_sha256": _mapping_sha256(unsigned)}
    return validate_parent_checkpoint_lineage(lineage)


def configured_parent_checkpoint_lineage(exp_config: Any) -> dict[str, Any] | None:
    """Return the runtime-resolved parent identity, if this run declares one."""

    exp = _node_get(exp_config, "experiment", exp_config)
    spec = _node_get(exp, "parent_checkpoint_lineage", None)
    if spec is None:
        return None
    identity = _node_get(spec, "identity", None)
    if identity is None:
        return None
    return validate_parent_checkpoint_lineage(identity)


def bind_explicit_parent_checkpoint(
    exp_config: Any,
    *,
    launch_dir: str | Path,
) -> dict[str, Any] | None:
    """Resolve and bind an explicitly configured fine-tuning parent.

    Runs without ``parent_checkpoint_lineage`` retain legacy resume behavior.
    Runs that opt into this contract get a recomputed content identity in the
    config before hashing, so a fixed ``run_id`` cannot resume across parents.
    """

    exp = _node_get(exp_config, "experiment", exp_config)
    spec = _node_get(exp, "parent_checkpoint_lineage", None)
    if spec is None:
        return None
    required = _node_get(spec, "required", False) is True
    resume_from = _node_get(exp, "resume_from", None)
    if resume_from in (None, ""):
        if required:
            raise ValueError(
                "this training stage requires an explicit parent checkpoint; "
                "set experiment.resume_from to the accepted upstream checkpoint"
            )
        return None
    role = _node_get(spec, "role", None)
    if not isinstance(role, str) or not role.strip():
        raise ValueError("parent_checkpoint_lineage.role must be a non-empty string")

    value = str(resume_from)
    if not value.startswith("hf://"):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path(launch_dir).expanduser().resolve() / candidate
        value = str(candidate)
    canonical = _canonicalize_resume_path(value)
    from musclemimic.badminton.promotion_artifact import checkpoint_identity

    identity = checkpoint_identity(canonical)
    promotion_manifest = _node_get(spec, "promotion_manifest", None)
    promotion_binding = None
    if promotion_manifest not in (None, ""):
        promotion_path = Path(str(promotion_manifest)).expanduser()
        if not promotion_path.is_absolute():
            promotion_path = Path(launch_dir).expanduser().resolve() / promotion_path
        promotion_path = promotion_path.resolve(strict=True)
        promotion_binding = _build_parent_promotion_binding(
            promotion_path,
            checkpoint_path=canonical,
            role=role,
        )
    lineage = build_parent_checkpoint_lineage(
        identity,
        role=role,
        promotion_binding=promotion_binding,
    )
    with open_dict(exp):
        exp.resume_from = canonical
        with open_dict(exp.parent_checkpoint_lineage):
            if promotion_manifest not in (None, ""):
                exp.parent_checkpoint_lineage.promotion_manifest = str(promotion_path)
            exp.parent_checkpoint_lineage.identity = lineage
    return lineage


def validate_explicit_parent_checkpoint(
    exp_config: Any,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Re-hash the checkpoint immediately before the initial parent restore."""

    exp = _node_get(exp_config, "experiment", exp_config)
    spec = _node_get(exp, "parent_checkpoint_lineage", None)
    expected = configured_parent_checkpoint_lineage(exp)
    if spec is None or expected is None:
        raise ValueError("explicit parent checkpoint has no resolved lineage contract")
    role = str(_node_get(spec, "role", ""))
    canonical = _canonicalize_resume_path(str(checkpoint))
    from musclemimic.badminton.promotion_artifact import checkpoint_identity

    identity = checkpoint_identity(canonical)
    promotion_manifest = _node_get(spec, "promotion_manifest", None)
    promotion_binding = None
    if promotion_manifest not in (None, ""):
        promotion_binding = _build_parent_promotion_binding(
            str(promotion_manifest),
            checkpoint_path=canonical,
            role=role,
        )
    actual = build_parent_checkpoint_lineage(
        identity,
        role=role,
        promotion_binding=promotion_binding,
    )
    if actual != expected:
        raise ValueError("explicit parent checkpoint changed after run identity resolution")
    return actual


def _get_git_sha() -> str | None:
    """Get current git SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _portable_source_mode(path: Path, *, kind: str) -> int:
    """Return the Git-relevant mode without binding a clone's umask.

    Git source identity distinguishes regular files from symlinks and tracks
    only whether a regular file is executable.  Group/other read-write bits
    are checkout policy, so hashing the full POSIX mode makes identical clean
    clones disagree when their umasks differ.
    """

    if kind == "symlink":
        return 0o120000
    if kind == "file":
        return 0o755 if path.stat().st_mode & 0o111 else 0o644
    if kind == "missing":
        return 0
    raise ValueError(f"unsupported source snapshot kind: {kind!r}")


def stage1_source_tree_snapshot() -> dict[str, Any] | None:
    """Fingerprint the exact scoped source/config worktree used by Stage1.

    HEAD alone is insufficient for a matched experiment launched from a dirty
    checkout.  This contract hashes tracked and untracked source/config bytes,
    missing tracked files, executable modes, and the scoped porcelain status.
    Runtime model/tube identities remain independently bound by their own
    contracts.
    """

    repo_root = Path(__file__).resolve().parents[2]
    try:
        head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=True,
            )
            .stdout.decode("ascii")
            .strip()
        )
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *STAGE1_SOURCE_TREE_SCOPES,
            ],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                *STAGE1_SOURCE_TREE_SCOPES,
            ],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return None

    relative_paths = sorted(
        {
            os.fsdecode(raw)
            for raw in listed.split(b"\0")
            if raw
            and (
                Path(os.fsdecode(raw)).suffix.lower() in STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES
                or Path(os.fsdecode(raw)).name in {"pyproject.toml", "uv.lock"}
            )
        }
    )
    content_digest = hashlib.sha256()
    for relative in relative_paths:
        path = repo_root / relative
        if path.is_symlink():
            kind = "symlink"
            payload = os.fsencode(os.readlink(path))
        elif path.is_file():
            kind = "file"
            payload = path.read_bytes()
        else:
            kind = "missing"
            payload = b""
        mode = _portable_source_mode(path, kind=kind)
        record = {
            "path": relative,
            "kind": kind,
            "mode": mode,
            "num_bytes": len(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
        content_digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        content_digest.update(b"\0")
    unsigned = {
        "schema_version": STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION,
        "git_sha": head,
        "scopes": list(STAGE1_SOURCE_TREE_SCOPES),
        "included_suffixes": sorted(STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES),
        "file_count": len(relative_paths),
        "worktree_dirty": bool(status),
        "worktree_status_sha256": hashlib.sha256(status).hexdigest(),
        "source_tree_fingerprint": content_digest.hexdigest(),
    }
    return {**unsigned, "binding_sha256": _mapping_sha256(unsigned)}


def write_manifest(
    checkpoint_dir: str | Path,
    config: Any,
    config_hash_value: str,
) -> None:
    """Write manifest.json in checkpoint directory on first save.

    Args:
        checkpoint_dir: Checkpoint directory path.
        config: Experiment config to save.
        config_hash_value: Pre-computed config hash.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"

    # Existing manifests are immutable run identities. This matters even when
    # no checkpoint has been written yet: a failed startup must not let a
    # subsequent invocation reuse the fixed run_id with a different parent.
    if manifest_path.exists():
        existing = _strict_json_object(
            manifest_path,
            label="existing checkpoint run manifest",
        )
        if existing.get("config_hash") != config_hash_value:
            raise ValueError("existing checkpoint run manifest has a different config hash")
        expected_parent = configured_parent_checkpoint_lineage(config)
        recorded_parent = existing.get("parent_checkpoint_lineage")
        if recorded_parent is not None:
            recorded_parent = validate_parent_checkpoint_lineage(recorded_parent)
        if recorded_parent != expected_parent:
            raise ValueError("existing checkpoint run manifest has a different parent checkpoint lineage")
        if _node_get(config, "stage1_peasd", None) is not None:
            current_source = stage1_source_tree_snapshot()
            if current_source is None or existing.get("source_tree_snapshot") != current_source:
                raise ValueError("existing Stage1 PEASD run manifest has a different source-tree snapshot")

        experiment_config = existing.get("experiment_config")
        nested_config = experiment_config if isinstance(experiment_config, dict) else {}
        for contract_key, contract_label in (
            ("muscle_control_contract", "muscle control"),
            ("body_synergy_contract", "body action"),
            ("continuity_training_contract", "continuity training"),
            ("continuity_smoke_contract", "continuity smoke"),
        ):
            expected_contract = _as_native(_node_get(config, contract_key, None))
            saved_contract = existing.get(contract_key)
            nested_contract = nested_config.get(contract_key)
            if expected_contract is None:
                if saved_contract is not None or nested_contract is not None:
                    raise ValueError(
                        f"existing checkpoint run manifest unexpectedly declares a {contract_label} contract"
                    )
                continue
            if (
                not isinstance(expected_contract, dict)
                or saved_contract != nested_contract
                or saved_contract != expected_contract
            ):
                raise ValueError(f"existing checkpoint run manifest has no consistent {contract_label} contract")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_hash": config_hash_value,
        "git_sha": _get_git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_config": OmegaConf.to_container(config, resolve=True),
    }
    if _node_get(config, "stage1_peasd", None) is not None:
        source_tree_snapshot = stage1_source_tree_snapshot()
        if source_tree_snapshot is None:
            raise ValueError("Stage1 PEASD requires a Git-backed deterministic source-tree snapshot")
        manifest["source_tree_snapshot"] = source_tree_snapshot
    parent_lineage = configured_parent_checkpoint_lineage(config)
    if parent_lineage is not None:
        manifest["parent_checkpoint_lineage"] = parent_lineage
        run_identity = {
            "schema_version": "content_bound_training_run_identity_v1",
            "experiment_id": str(_node_get(config, "run_id", None) or config_hash_value),
            "config_hash": str(config_hash_value),
            "parent_checkpoint_lineage": parent_lineage,
        }
        manifest["run_identity"] = {
            **run_identity,
            "binding_sha256": _mapping_sha256(run_identity),
        }
    action_manifest = getattr(config, "action_manifest", None)
    if action_manifest is not None:
        manifest["action_manifest"] = OmegaConf.to_container(action_manifest, resolve=True)
    body_synergy_contract = getattr(config, "body_synergy_contract", None)
    if body_synergy_contract is not None:
        # Keep the stage-portable contract at top level as well as inside the
        # resolved experiment config so checkpoint consumers do not have to
        # infer the body-action identity from a nested Hydra payload.
        manifest["body_synergy_contract"] = OmegaConf.to_container(
            body_synergy_contract,
            resolve=True,
        )
    muscle_control_contract = getattr(config, "muscle_control_contract", None)
    if muscle_control_contract is not None:
        manifest["muscle_control_contract"] = OmegaConf.to_container(
            muscle_control_contract,
            resolve=True,
        )
    continuity_training_contract = getattr(config, "continuity_training_contract", None)
    if continuity_training_contract is not None:
        manifest["continuity_training_contract"] = OmegaConf.to_container(
            continuity_training_contract,
            resolve=True,
        )
    continuity_smoke_contract = getattr(config, "continuity_smoke_contract", None)
    if continuity_smoke_contract is not None:
        manifest["continuity_smoke_contract"] = OmegaConf.to_container(
            continuity_smoke_contract,
            resolve=True,
        )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def validate_checkpoint_compatibility(
    checkpoint_dir: str | Path,
    current_hash: str,
    *,
    require_manifest: bool = False,
) -> bool:
    """Validate checkpoint config hash matches current config.

    Args:
        checkpoint_dir: Checkpoint directory containing manifest.json.
        current_hash: Current config hash to compare.

    Returns:
        True if compatible (or no manifest), False with warning if mismatch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"

    if not manifest_path.exists():
        if require_manifest:
            print(f"WARNING: Checkpoint run manifest is missing: {manifest_path}")
            return False
        return True

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        saved_hash = manifest.get("config_hash")
        if require_manifest and not saved_hash:
            print("WARNING: Checkpoint run manifest has no config_hash")
            return False
        if saved_hash and saved_hash != current_hash:
            print("WARNING: Config hash mismatch!")
            print(f"  Checkpoint: {saved_hash}")
            print(f"  Current:    {current_hash}")
            print("  Continuing anyway (override with explicit resume_from if needed)")
            return False
    except Exception as e:
        print(f"Warning: Could not read manifest: {e}")
        return not require_manifest

    return True


def validate_checkpoint_parent_lineage(
    checkpoint_dir: str | Path,
    expected_lineage: Any,
) -> bool:
    """Compare the current parent chain with an existing local run manifest."""

    manifest_path = Path(checkpoint_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read checkpoint lineage manifest: {exc}")
        return False
    try:
        expected = validate_parent_checkpoint_lineage(expected_lineage)
        actual = validate_parent_checkpoint_lineage(manifest.get("parent_checkpoint_lineage"))
    except ValueError as exc:
        print(f"WARNING: Invalid checkpoint parent lineage: {exc}")
        return False
    if actual != expected:
        print("WARNING: Parent checkpoint lineage mismatch")
        print(f"  Checkpoint parent: {actual.get('binding_sha256')}")
        print(f"  Current parent:    {expected.get('binding_sha256')}")
        return False
    return True


def validate_checkpoint_muscle_control_contract(
    checkpoint: str | Path,
    current_contract: Any,
) -> None:
    """Reject pre-v2 or drifted muscle-control checkpoints for every action ABI.

    The run manifest and the concrete Orbax leaf are independently bound. This
    prevents an older shape-compatible leaf from being moved or linked under a
    directory containing a modern manifest.
    """

    current = _as_native(current_contract)
    if not isinstance(current, dict):
        raise ValueError("current muscle control contract must be a mapping")
    selected_path, restore_leaf, run_dir = _resolve_checkpoint_validation_paths(checkpoint)
    manifest_path = run_dir / "manifest.json"
    manifest = _strict_json_object(
        manifest_path,
        label="checkpoint run manifest",
    )
    saved = manifest.get("muscle_control_contract")
    experiment = manifest.get("experiment_config")
    nested = experiment.get("muscle_control_contract") if isinstance(experiment, dict) else None
    if not isinstance(saved, dict) or saved != nested:
        raise ValueError(
            "checkpoint lacks one consistent muscle_control_contract; refusing a pre-v2 or shape-only restore"
        )
    if saved != current:
        raise ValueError("checkpoint muscle control contract differs from the verified runtime contract")

    leaf_payloads = _load_orbax_leaf_json_payloads(
        selected_path,
        restore_leaf,
        required=True,
    )
    assert leaf_payloads is not None
    leaf_config, _leaf_metadata = leaf_payloads
    leaf_experiment = leaf_config.get("experiment")
    leaf_contract = leaf_experiment.get("muscle_control_contract") if isinstance(leaf_experiment, dict) else None
    if not isinstance(leaf_contract, dict):
        raise ValueError(
            "checkpoint leaf config has no muscle_control_contract; refusing a pre-v2 or shape-only restore"
        )
    if leaf_contract != saved:
        raise ValueError("checkpoint leaf muscle control contract differs from its run manifest")


def validate_checkpoint_continuity_training_contract(
    checkpoint: str | Path,
    current_contract: Any,
) -> None:
    """Require one exact continuity release binding at every restore layer."""

    current = _as_native(current_contract)
    if not isinstance(current, dict):
        raise ValueError("current continuity training contract must be a mapping")
    selected_path, restore_leaf, run_dir = _resolve_checkpoint_validation_paths(checkpoint)
    manifest = _strict_json_object(
        run_dir / "manifest.json",
        label="checkpoint run manifest",
    )
    saved = manifest.get("continuity_training_contract")
    experiment = manifest.get("experiment_config")
    nested = experiment.get("continuity_training_contract") if isinstance(experiment, dict) else None
    if not isinstance(saved, dict) or saved != nested:
        raise ValueError("checkpoint lacks one consistent continuity_training_contract")
    if saved != current:
        raise ValueError("checkpoint continuity training contract differs from the verified runtime contract")

    leaf_config, leaf_metadata = _load_orbax_leaf_json_payloads(
        selected_path,
        restore_leaf,
        required=True,
    )
    leaf_experiment = leaf_config.get("experiment")
    leaf_contract = leaf_experiment.get("continuity_training_contract") if isinstance(leaf_experiment, dict) else None
    if not isinstance(leaf_contract, dict):
        raise ValueError("checkpoint leaf config has no continuity_training_contract")
    if leaf_contract != saved:
        raise ValueError("checkpoint leaf continuity training contract differs from its run manifest")
    metadata_contract = leaf_metadata.get("continuity_training_contract")
    if not isinstance(metadata_contract, dict):
        raise ValueError("checkpoint leaf metadata has no continuity_training_contract")
    if metadata_contract != saved:
        raise ValueError("checkpoint leaf continuity training contract differs between config and metadata")


def validate_checkpoint_body_action_contract(
    checkpoint: str | Path,
    current_contract: Any,
    *,
    compatibility: str,
    legacy_attestation: str | Path | None = None,
) -> None:
    """Validate the action decoder recorded by a checkpoint run manifest.

    ``portable`` is the intentional Stage-1 -> Stage-2/finetune hand-off: the
    ordered 354-D ABI and frozen decoder core must be identical, while the
    stage model/coverage binding may change. ``exact_runtime`` is used for a
    same-run resume and additionally requires the concrete runtime binding to
    match.  A modern run that declares a body-action contract never falls back
    to shape-only restore or to an older unbound checkpoint.
    """

    from musclemimic.synergy.multistage_contract import (
        EXACT_RUNTIME_COMPATIBILITY,
        PORTABLE_COMPATIBILITY,
        BodySynergyContractV2,
    )

    current_native = _as_native(current_contract)
    if not isinstance(current_native, dict):
        raise ValueError("current body action contract must be a mapping")
    current = BodySynergyContractV2.from_manifest(current_native)

    selected_path, restore_leaf, run_dir = _resolve_checkpoint_validation_paths(checkpoint)
    manifest_path = run_dir / "manifest.json"
    manifest = _strict_json_object(
        manifest_path,
        label="checkpoint run manifest",
    )

    saved_native = manifest.get("body_synergy_contract")
    experiment_config = manifest.get("experiment_config")
    nested_native = experiment_config.get("body_synergy_contract") if isinstance(experiment_config, dict) else None
    if saved_native is None:
        saved_native = nested_native
    elif nested_native != saved_native:
        raise ValueError("checkpoint top-level body action contract differs from experiment config")
    modern_manifest_contract = isinstance(saved_native, dict)
    if not isinstance(saved_native, dict):
        # A small number of pre-contract direct-354 checkpoints can be
        # migrated only through a separate, content-bound attestation produced
        # by a successful runtime ABI reconstruction.  Default behavior stays
        # fail-closed, and exact same-run resumes never accept this bridge.
        if legacy_attestation is None:
            raise ValueError("checkpoint has no BodySynergyContractV2; refusing a shape-only restore")
        if str(compatibility) != PORTABLE_COMPATIBILITY:
            raise ValueError("legacy body action attestation is allowed only for an explicit portable parent")
        attestation_path = Path(legacy_attestation).expanduser().resolve(strict=True)
        attestation = _strict_json_object(
            attestation_path,
            label="legacy body action attestation",
        )

        attested_contract = attestation.get("body_synergy_contract")
        attested_config = attestation.get("experiment_config")
        nested_attested_contract = (
            attested_config.get("body_synergy_contract") if isinstance(attested_config, dict) else None
        )
        if not isinstance(attested_contract, dict):
            raise ValueError("legacy body action attestation has no BodySynergyContractV2")
        if nested_attested_contract != attested_contract:
            raise ValueError("legacy body action attestation top-level and nested contracts differ")

        from musclemimic.badminton.promotion_artifact import checkpoint_identity

        attested_lineage = validate_parent_checkpoint_lineage(attestation.get("parent_checkpoint_lineage"))
        actual_identity = checkpoint_identity(restore_leaf)
        attested_identity = attested_lineage["checkpoint"]
        for key in _PARENT_CHECKPOINT_IDENTITY_KEYS:
            if actual_identity.get(key) != attested_identity.get(key):
                raise ValueError(f"legacy body action attestation parent checkpoint binding mismatch: {key}")
        saved_native = attested_contract

    leaf_payloads = _load_orbax_leaf_json_payloads(
        selected_path,
        restore_leaf,
        required=modern_manifest_contract,
    )
    if leaf_payloads is not None:
        leaf_config, _leaf_metadata = leaf_payloads
        leaf_experiment = leaf_config.get("experiment")
        leaf_contract = leaf_experiment.get("body_synergy_contract") if isinstance(leaf_experiment, dict) else None
        if modern_manifest_contract and not isinstance(leaf_contract, dict):
            raise ValueError(
                "checkpoint leaf config has no body_synergy_contract; refusing a pre-contract or shape-only restore"
            )
        if leaf_contract is not None and not isinstance(leaf_contract, dict):
            raise ValueError("checkpoint leaf body action contract must be a mapping")
        if leaf_contract is not None and leaf_contract != saved_native:
            raise ValueError("checkpoint leaf body action contract differs from its run manifest")
    saved = BodySynergyContractV2.from_manifest(saved_native)

    level = str(compatibility)
    if level == PORTABLE_COMPATIBILITY:
        current.assert_portable_compatible(saved)
    elif level == EXACT_RUNTIME_COMPATIBILITY:
        current.assert_exact_runtime_compatible(saved)
    else:
        raise ValueError(f"unsupported body action checkpoint compatibility={level!r}")


def resolve_checkpoint_dir(
    configured_ckpt_dir: str,
    launch_dir: str,
    result_dir: str,
    experiment_id: str,
    auto_resume: bool,
    checkpoint_root: str | None = None,
    training_root: str | None = None,
) -> str:
    """Resolve the checkpoint directory path based on auto_resume setting.

    Args:
        configured_ckpt_dir: Default checkpoint directory name (e.g., "checkpoints").
        launch_dir: Directory where script was launched from.
        result_dir: Hydra per-run output directory.
        experiment_id: Experiment identifier (config hash or run_id).
        auto_resume: Whether auto-resume is enabled.
        checkpoint_root: Optional explicit checkpoint root path.
        training_root: Optional action-scoped training root. When set, it owns
            checkpoint artifacts under ``<training_root>/checkpoints``.

    Returns:
        Resolved absolute checkpoint directory path.
    """
    # Determine base directory
    if training_root:
        base = os.path.join(training_root, "checkpoints")
    elif checkpoint_root:
        base = checkpoint_root if os.path.isabs(checkpoint_root) else os.path.join(launch_dir, checkpoint_root)
    elif auto_resume:
        base = (
            configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(launch_dir, configured_ckpt_dir)
        )
    else:
        base = (
            configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(result_dir, configured_ckpt_dir)
        )

    # Append experiment_id for auto_resume, otherwise use unique suffix
    if auto_resume:
        return os.path.join(base, experiment_id)
    else:
        return base  # Caller adds unique suffix


def _download_from_huggingface(repo_id: str, revision: str | None = None) -> str:
    """Download checkpoint from HuggingFace and return local path.

    Args:
        repo_id: HuggingFace repo ID

    Returns:
        Local path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    return local_dir


def _symlink_or_copy(target: Path, link: Path) -> None:
    """Create a symlink, falling back to junction/copy on Windows."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        # Windows often disallows symlinks without admin/Developer Mode (WinError 1314).
        # Prefer a directory junction as a zero-copy alias; last resort is a full copy.
        is_windows = sys.platform.startswith("win")
        winerr = getattr(e, "winerror", None)
        if not (is_windows and winerr in (1314, 5)):
            raise
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            shutil.copytree(target, link)


def _canonicalize_resume_path(path_like: str, revision: str | None = None) -> str:
    r"""Return a concrete checkpoint path to resume from.

    Accepts multiple user-friendly inputs and normalizes to an Orbax checkpoint
    directory path (.../checkpoint_<step>).

    Supported formats:
        - Local path: /path/to/checkpoint_30000
        - Local parent dir: /path/to/checkpoints/ (picks latest)
        - HuggingFace: hf://username/repo-name

    Raises:
        ValueError: If path does not exist, is a file, or contains no valid checkpoints.
    """
    # Handle HuggingFace URLs
    if path_like.startswith("hf://"):
        repo_id = path_like[5:]  # Remove "hf://" prefix
        print(f"Downloading checkpoint from HuggingFace: {repo_id}")
        local_path = _download_from_huggingface(repo_id, revision=revision)
        print(f"Downloaded to: {local_path}")
        # Recursively canonicalize the downloaded path
        return _canonicalize_resume_path(local_path)

    p = Path(path_like)
    if not p.exists():
        raise ValueError(f"Checkpoint path does not exist: {path_like}")

    if p.is_file():
        raise ValueError(f"Checkpoint path is a file, not a directory: {path_like}")

    name = p.name
    if re.match(r"^checkpoint_\d+$", name):
        return str(p)

    # Directory that may contain multiple checkpoint_* subdirs
    subdirs = [d for d in p.iterdir() if d.is_dir() and re.match(r"^checkpoint_\d+$", d.name)]
    # Filter to only complete checkpoints (have metadata)
    complete_subdirs = [d for d in subdirs if _is_checkpoint_complete(d)]
    if complete_subdirs:
        # Pick latest complete checkpoint by numeric step
        step = max(int(d.name.split("_")[-1]) for d in complete_subdirs)
        return str(p / f"checkpoint_{step}")

    # Check if it's an Orbax checkpoint directory (has train_state subdir)
    if (p / "train_state").is_dir():
        # Read step from metadata and create expected checkpoint_<step> alias
        metadata_file = p / "metadata" / "metadata"
        if metadata_file.is_symlink():
            metadata_file = metadata_file.resolve()
        with open(metadata_file) as f:
            step = json.load(f).get("update_number", 0)
        symlink_path = p.parent / f"checkpoint_{step}"
        if not symlink_path.exists():
            _symlink_or_copy(target=p, link=symlink_path)
        return str(symlink_path)

    # Not a recognizable checkpoint location
    raise ValueError(f"No valid checkpoints found in: {path_like}")


def resume_or_fresh(
    env: Any,
    agent_conf: Any,
    algorithm_cls: Any,
    config: Any,
    mh: Any,
    logging_callback,
    logging_interval: int = 1,
    val_env: Any = None,
    apply_resume_resets: bool = True,
):
    """Return a train function that resumes from checkpoint or starts fresh.

    If `experiment.resume_from` is set, validates the path and raises ValueError
    if invalid. If not set, starts fresh training.

    Raises:
        ValueError: If resume_from is set but path is invalid or contains no checkpoints.
    """
    resume_from = getattr(config.experiment, "resume_from", None)
    revision = getattr(config.experiment, "revision", None)

    # Fresh training
    if not resume_from:
        return algorithm_cls.build_train_fn(
            env,
            agent_conf,
            mh=mh,
            online_logging_callback=logging_callback,
            logging_interval=getattr(config.experiment, "online_logging_interval", logging_interval),
            val_env=val_env,
        )

    # Normalize resume path to a specific checkpoint (raises ValueError if invalid)
    canonical = _canonicalize_resume_path(resume_from, revision=revision)

    # Build resume train fn that loads on first call
    return algorithm_cls.build_resume_train_fn_from_path(
        env,
        agent_conf,
        canonical,
        mh=mh,
        online_logging_callback=logging_callback,
        logging_interval=getattr(config.experiment, "online_logging_interval", logging_interval),
        val_env=val_env,
        apply_resume_resets=apply_resume_resets,
    )
