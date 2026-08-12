#!/usr/bin/env python3
"""Fail-closed deployment preflight for a MuscleMimic training server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import action_choices, resolve
from musclemimic.badminton.action_release import validate_action_release
from musclemimic.badminton.stage1_peasd_gate import build_verified_tube_gate
from musclemimic.runner.checkpointing import stage1_source_tree_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "musclemimic_server_training_preflight_v1"
ASSET_MANIFEST_SCHEMA_VERSION = "musclemimic_private_training_asset_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_asset_manifest(
    path: Path,
    *,
    expected_action: str | None = None,
    expected_release_binding: str | None = None,
    expected_tube_binding: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        errors.append("asset manifest schema_version mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_fingerprint"}
    if payload.get("manifest_fingerprint") != _canonical_sha256(unsigned):
        errors.append("asset manifest fingerprint mismatch")
    if expected_action is not None and payload.get("action") != expected_action:
        errors.append("asset manifest belongs to another action")
    if expected_release_binding is not None and payload.get("release_binding_sha256") != expected_release_binding:
        errors.append("asset manifest release binding is stale")
    if expected_tube_binding is not None and payload.get("tube_gate_binding_sha256") != expected_tube_binding:
        errors.append("asset manifest verified-tube binding is stale")
    if payload.get("includes_smplh") is not True:
        errors.append("asset manifest does not include SMPL-H")

    records = payload.get("files", [])
    if not isinstance(records, list) or not records:
        errors.append("asset manifest files must be a non-empty list")
        records = []
    seen: set[str] = set()
    observed_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            errors.append("asset manifest contains a non-object file record")
            continue
        relative = str(record.get("path", ""))
        if not relative or relative in seen:
            errors.append(f"asset manifest has an empty or duplicate path: {relative!r}")
            continue
        seen.add(relative)
        asset = (REPO_ROOT / relative).resolve()
        try:
            asset.relative_to(REPO_ROOT)
        except ValueError:
            errors.append(f"asset escapes repository: {relative!r}")
            continue
        if not asset.is_file():
            errors.append(f"missing asset: {relative}")
            continue
        if asset.is_symlink():
            errors.append(f"asset must not be a symlink: {relative}")
            continue
        observed_bytes += asset.stat().st_size
        if asset.stat().st_size != record.get("num_bytes") or _sha256(asset) != record.get("sha256"):
            errors.append(f"asset content mismatch: {relative}")
    if len(records) != payload.get("file_count"):
        errors.append("asset manifest file_count mismatch")
    if observed_bytes != payload.get("total_bytes"):
        errors.append("asset manifest total_bytes mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "action": payload.get("action"),
        "file_count": payload.get("file_count", 0),
        "total_bytes": payload.get("total_bytes", 0),
        "manifest_fingerprint": payload.get("manifest_fingerprint"),
    }


def _writable_directory(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="musclemimic-preflight-", dir=path):
            pass
    except OSError as exc:
        return False, str(exc)
    return True, None


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    snapshot = stage1_source_tree_snapshot()
    checks["source_tree_snapshot"] = snapshot
    if snapshot is None:
        errors.append("Git-backed source-tree snapshot is unavailable")
    else:
        if args.expected_git_sha and snapshot["git_sha"] != args.expected_git_sha:
            errors.append("Git SHA differs from --expected-git-sha")
        if (
            args.expected_source_tree_fingerprint
            and snapshot["source_tree_fingerprint"] != args.expected_source_tree_fingerprint
        ):
            errors.append("source-tree fingerprint differs from --expected-source-tree-fingerprint")
        if snapshot["worktree_dirty"] and not args.allow_dirty_source:
            errors.append("scoped source/config worktree is dirty")

    required_tools = ["git", "uv", "tmux", "wget", "bsdtar"]
    missing_tools = [name for name in required_tools if shutil.which(name) is None]
    checks["tools"] = {"required": required_tools, "missing": missing_tools}
    errors.extend(f"required tool is missing: {name}" for name in missing_tools)

    cache_root = Path(args.jax_cache_root).expanduser().resolve()
    writable, write_error = _writable_directory(cache_root)
    checks["jax_cache"] = {"path": str(cache_root), "writable": writable, "error": write_error}
    if not writable:
        errors.append(f"JAX cache root is not writable: {cache_root}: {write_error}")

    if not args.skip_gpu:
        query = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"])
        checks["gpu"] = {"returncode": query.returncode, "output": query.stdout.strip(), "error": query.stderr.strip()}
        if query.returncode != 0:
            errors.append("nvidia-smi GPU query failed")
        elif args.physical_gpu is not None:
            indices = {line.split(",", 1)[0].strip() for line in query.stdout.splitlines() if line.strip()}
            if str(args.physical_gpu) not in indices:
                errors.append(f"physical GPU {args.physical_gpu} is not present")
    else:
        checks["gpu"] = {"skipped": True}

    if not args.code_only:
        spec = resolve(args.action)
        release = validate_action_release(spec)
        checks["action_release"] = release
        if release.get("passed") is not True:
            errors.extend(f"action release: {item}" for item in release.get("errors", ()))

        smpl_root = Path(os.environ.get("MUSCLEMIMIC_SMPL_MODEL_PATH", REPO_ROOT / "smpl_models" / "smplh")).resolve()
        smpl_files = [smpl_root / "SMPLH_NEUTRAL.pkl", smpl_root / "neutral" / "model.npz"]
        checks["smplh"] = {"root": str(smpl_root), "present": [path.is_file() for path in smpl_files]}
        if not any(path.is_file() for path in smpl_files):
            errors.append(f"no usable SMPL-H neutral model found under {smpl_root}")

        tube_gate = None
        if not args.skip_tube:
            tube = args.tube or (
                REPO_ROOT
                / "artifacts"
                / "emg_human_review_v2"
                / "verified_tubes"
                / spec.emg_trial_actions[0]
                / "emg_reference_manifest.json"
            )
            try:
                tube_gate = build_verified_tube_gate(tube, action=spec.slug)
                checks["verified_tube"] = tube_gate
            except Exception as exc:
                checks["verified_tube"] = {"passed": False, "error": str(exc)}
                errors.append(f"verified PEASD tube failed: {exc}")

        if args.asset_manifest:
            asset_validation = validate_asset_manifest(
                args.asset_manifest.resolve(),
                expected_action=spec.slug,
                expected_release_binding=release["release_binding_sha256"],
                expected_tube_binding=(None if tube_gate is None else tube_gate["binding_sha256"]),
            )
            checks["asset_manifest"] = asset_validation
            errors.extend(f"asset manifest: {item}" for item in asset_validation["errors"])

    return {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "action": args.action,
        "checks": checks,
        "errors": errors,
        "passed": not errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=action_choices(), default="forehand_clear")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument(
        "--jax-cache-root", default=os.environ.get("MUSCLEMIMIC_JAX_CACHE_ROOT", "/tmp/musclemimic-jax-cache")
    )
    parser.add_argument("--tube", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--expected-source-tree-fingerprint")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-tube", action="store_true")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="diagnostics only; formal training should use a clean checkout",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_preflight(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
