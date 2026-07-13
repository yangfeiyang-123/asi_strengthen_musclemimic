from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

# Fields excluded from config hash (they don't affect training identity)
_HASH_EXCLUDE_FIELDS = frozenset({
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
})

_DATASET_ACTION_RE = re.compile(r"(?:^|[/\\])datasets[/\\]([^/\\]+)(?:[/\\]|$)")
_IGNORED_DATASET_ACTIONS = {"_global", "_index"}
PARENT_CHECKPOINT_LINEAGE_SCHEMA_VERSION = "resume_parent_checkpoint_lineage_v1"
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
        (checkpoint_path / "_CHECKPOINT_METADATA").is_file()
        and (checkpoint_path / "metadata" / "metadata").is_file()
    )


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
    unsigned = {
        "schema_version": native["schema_version"],
        "role": role,
        "checkpoint": checkpoint,
        "parent_checkpoint_lineage": upstream,
    }
    if native.get("binding_sha256") != _mapping_sha256(unsigned):
        raise ValueError("parent checkpoint lineage binding hash is missing or stale")
    if set(native) != {*unsigned, "binding_sha256"}:
        raise ValueError("parent checkpoint lineage contains unsupported fields")
    return dict(native)


def build_parent_checkpoint_lineage(
    checkpoint_identity_payload: Any,
    *,
    role: str,
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

    lineage = build_parent_checkpoint_lineage(
        checkpoint_identity(canonical),
        role=role,
    )
    with open_dict(exp):
        exp.resume_from = canonical
        with open_dict(exp.parent_checkpoint_lineage):
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

    actual = build_parent_checkpoint_lineage(
        checkpoint_identity(canonical),
        role=role,
    )
    if actual != expected:
        raise ValueError(
            "explicit parent checkpoint changed after run identity resolution"
        )
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
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"existing checkpoint run manifest is unreadable: {manifest_path}"
            ) from exc
        if existing.get("config_hash") != config_hash_value:
            raise ValueError(
                "existing checkpoint run manifest has a different config hash"
            )
        expected_parent = configured_parent_checkpoint_lineage(config)
        recorded_parent = existing.get("parent_checkpoint_lineage")
        if recorded_parent is not None:
            recorded_parent = validate_parent_checkpoint_lineage(recorded_parent)
        if recorded_parent != expected_parent:
            raise ValueError(
                "existing checkpoint run manifest has a different parent checkpoint lineage"
            )
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_hash": config_hash_value,
        "git_sha": _get_git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_config": OmegaConf.to_container(config, resolve=True),
    }
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
        actual = validate_parent_checkpoint_lineage(
            manifest.get("parent_checkpoint_lineage")
        )
    except ValueError as exc:
        print(f"WARNING: Invalid checkpoint parent lineage: {exc}")
        return False
    if actual != expected:
        print("WARNING: Parent checkpoint lineage mismatch")
        print(f"  Checkpoint parent: {actual.get('binding_sha256')}")
        print(f"  Current parent:    {expected.get('binding_sha256')}")
        return False
    return True


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
        base = configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(launch_dir, configured_ckpt_dir)
    else:
        base = configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(result_dir, configured_ckpt_dir)

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
                capture_output=True, text=True, check=True,
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
            logging_interval=getattr(
                config.experiment, "online_logging_interval", logging_interval
            ),
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
        logging_interval=getattr(
            config.experiment, "online_logging_interval", logging_interval
        ),
        val_env=val_env,
        apply_resume_resets=apply_resume_resets,
    )
