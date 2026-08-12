#!/usr/bin/env python3
"""Build a content-bound private-asset transfer manifest for one action.

Git carries source and portable experiment contracts.  This command inventories
the licensed/private files that must be copied separately to another server.
It never copies data itself and never writes inside the repository by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import ActionSpec, action_choices, resolve
from musclemimic.badminton.action_release import validate_action_release
from musclemimic.badminton.stage1_peasd_gate import build_verified_tube_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "musclemimic_private_training_asset_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _resolve_declared_path(value: str, *, dataset_root: Path) -> Path | None:
    """Resolve a JSON-declared path without allowing repository escape."""

    raw = Path(value)
    if raw.is_absolute():
        candidate = raw.resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None
    for candidate in (REPO_ROOT / raw, dataset_root / raw):
        resolved = candidate.resolve()
        try:
            resolved.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _json_declared_files(payload: Any, *, dataset_root: Path) -> set[Path]:
    files: set[Path] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if isinstance(value, str) and (key == "path" or key.endswith("_path")):
            resolved = _resolve_declared_path(value, dataset_root=dataset_root)
            if resolved is not None:
                files.add(resolved)

    visit(payload)
    return files


def _add_directory_files(files: set[Path], directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files.update(path.resolve() for path in directory.rglob("*") if path.is_file())


def collect_action_assets(spec: ActionSpec) -> tuple[set[Path], dict[str, Any]]:
    report = validate_action_release(spec)
    if report.get("passed") is not True:
        raise ValueError("action release validation failed: " + "; ".join(report.get("errors", ())))
    files: set[Path] = set()
    for row in report["file_inventory"]:
        files.add((REPO_ROOT / row["source_path"]).resolve())
        files.add((REPO_ROOT / row["cache_path"]).resolve())
    release_path = Path(report["release_evidence_path"]).resolve()
    files.add(release_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    files.update(_json_declared_files(release, dataset_root=spec.dataset_root))

    visual_path = report.get("visual_qc_path")
    if visual_path:
        resolved_visual = _resolve_declared_path(str(visual_path), dataset_root=spec.dataset_root)
        if resolved_visual is not None:
            files.add(resolved_visual)
            visual = json.loads(resolved_visual.read_text(encoding="utf-8"))
            files.update(_json_declared_files(visual, dataset_root=spec.dataset_root))
    return files, report


def build_manifest(
    *,
    action: str,
    tube: Path | None,
    include_smpl: bool,
    extra_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    spec = resolve(action)
    files, release = collect_action_assets(spec)
    tube_gate = None
    if tube is not None:
        tube_gate = build_verified_tube_gate(tube, action=spec.slug)
        manifest_path = Path(tube_gate["source"]["manifest_path"]).resolve()
        _add_directory_files(files, manifest_path.parent)
    if include_smpl:
        smpl_root = (
            Path(
                os.environ.get(
                    "MUSCLEMIMIC_SMPL_MODEL_PATH",
                    REPO_ROOT / "smpl_models" / "smplh",
                )
            )
            .expanduser()
            .resolve()
        )
        try:
            smpl_root.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(
                "MUSCLEMIMIC_SMPL_MODEL_PATH must be inside the repository so "
                "the transfer manifest can preserve relative paths"
            ) from exc
        if not any(
            path.is_file()
            for path in (
                smpl_root / "SMPLH_NEUTRAL.pkl",
                smpl_root / "neutral" / "model.npz",
            )
        ):
            raise FileNotFoundError(f"no usable SMPL-H neutral model found under {smpl_root}")
        _add_directory_files(files, smpl_root)
    for path in extra_paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            _add_directory_files(files, resolved)
        elif resolved.is_file():
            files.add(resolved)
        else:
            raise FileNotFoundError(resolved)

    records = []
    for path in sorted(files):
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"asset lies outside repository root: {path}") from exc
        if path.is_symlink():
            raise ValueError(f"asset must not be a symlink: {relative}")
        records.append(
            {
                "path": relative.as_posix(),
                "num_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "release_binding_sha256": release["release_binding_sha256"],
        "tube_gate_binding_sha256": None if tube_gate is None else tube_gate["binding_sha256"],
        "includes_smplh": include_smpl,
        "file_count": len(records),
        "total_bytes": sum(record["num_bytes"] for record in records),
        "files": records,
    }
    return {**unsigned, "manifest_fingerprint": _canonical_sha256(unsigned)}


def write_manifest(manifest: Mapping[str, Any], output: Path, files_from: Path | None) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if files_from is not None:
        files_from = files_from.expanduser().resolve()
        files_from.parent.mkdir(parents=True, exist_ok=True)
        files_from.write_text("\n".join(row["path"] for row in manifest["files"]) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=action_choices(), default="forehand_clear")
    parser.add_argument("--tube", type=Path, default=None)
    parser.add_argument("--without-tube", action="store_true")
    parser.add_argument("--without-smpl", action="store_true")
    parser.add_argument("--extra-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files-from", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = resolve(args.action)
    tube = None
    if not args.without_tube:
        tube = args.tube or (
            REPO_ROOT
            / "artifacts"
            / "emg_human_review_v2"
            / "verified_tubes"
            / spec.emg_trial_actions[0]
            / "emg_reference_manifest.json"
        )
    manifest = build_manifest(
        action=spec.slug,
        tube=tube,
        include_smpl=not args.without_smpl,
        extra_paths=tuple(args.extra_path),
    )
    write_manifest(manifest, args.output, args.files_from)
    print(
        json.dumps(
            {key: manifest[key] for key in ("action", "file_count", "total_bytes", "manifest_fingerprint")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
