"""Content-addressed synergy basis artifact storage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict

BASIS_ARTIFACT_SCHEMA_VERSION = "forehand_clear_synergy_basis_v1"
REQUIRED_MANIFEST_FIELDS = {
    "signal_kind",
    "region",
    "rank",
    "normalization",
    "source_dataset_fingerprint",
    "teacher_checkpoint_fingerprint",
    "fit_seed",
    "transform",
    "split_provenance",
    "train_motion_uids",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SynergyBasisArtifact:
    basis: np.ndarray
    muscle_names: tuple[str, ...]
    manifest: dict[str, Any]
    fingerprint: str
    path: Path


def save_synergy_basis(
    path: str | Path,
    *,
    basis: np.ndarray,
    muscle_names: Sequence[str],
    manifest: Mapping[str, Any],
) -> SynergyBasisArtifact:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    matrix, names = _validate_basis(basis, muscle_names)
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"synergy basis manifest is missing {missing}")
    if int(manifest["rank"]) != matrix.shape[1]:
        raise ValueError("synergy basis manifest rank does not match basis")
    _validate_manifest_semantics(manifest, matrix=matrix, muscle_names=names)

    basis_path = root / "basis.npy"
    np.save(basis_path, matrix.astype(np.float32), allow_pickle=False)
    basis_sha256 = _file_sha256(basis_path)
    muscle_schema_sha256 = _json_sha256({"muscle_names": list(names)})
    payload = {
        **dict(manifest),
        "schema_version": BASIS_ARTIFACT_SCHEMA_VERSION,
        "basis_file": basis_path.name,
        "basis_sha256": basis_sha256,
        "basis_shape": list(matrix.shape),
        "muscle_names": list(names),
        "muscle_schema_sha256": muscle_schema_sha256,
    }
    payload["artifact_fingerprint"] = _artifact_fingerprint(payload)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return load_synergy_basis(root)


def load_synergy_basis(path: str | Path) -> SynergyBasisArtifact:
    supplied = Path(path)
    manifest_path = supplied if supplied.name == "manifest.json" else supplied / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"synergy basis manifest does not exist: {manifest_path}")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("synergy basis manifest must contain a JSON object")
    if payload.get("schema_version") != BASIS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported synergy basis artifact schema")
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(payload))
    if missing:
        raise ValueError(f"synergy basis manifest is missing {missing}")
    expected_fingerprint = _artifact_fingerprint(payload)
    if payload.get("artifact_fingerprint") != expected_fingerprint:
        raise ValueError("synergy basis artifact_fingerprint mismatch")
    basis_path = manifest_path.parent / str(payload.get("basis_file", ""))
    if _file_sha256(basis_path) != payload.get("basis_sha256"):
        raise ValueError("synergy basis file hash mismatch")
    matrix = np.load(basis_path, allow_pickle=False)
    matrix, names = _validate_basis(matrix, payload.get("muscle_names", ()))
    if list(matrix.shape) != list(payload.get("basis_shape", ())):
        raise ValueError("synergy basis shape differs from manifest")
    if int(payload["rank"]) != matrix.shape[1]:
        raise ValueError("synergy basis rank differs from manifest")
    _validate_manifest_semantics(payload, matrix=matrix, muscle_names=names)
    if _json_sha256({"muscle_names": list(names)}) != payload.get("muscle_schema_sha256"):
        raise ValueError("synergy basis muscle schema hash mismatch")
    return SynergyBasisArtifact(
        basis=matrix.astype(np.float64),
        muscle_names=names,
        manifest=payload,
        fingerprint=expected_fingerprint,
        path=manifest_path.parent,
    )


def _artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {str(key): value for key, value in payload.items() if key != "artifact_fingerprint"}
    return _json_sha256(canonical)


def _validate_manifest_semantics(
    manifest: Mapping[str, Any],
    *,
    matrix: np.ndarray,
    muscle_names: Sequence[str],
) -> None:
    for key in ("signal_kind", "region", "source_dataset_fingerprint", "teacher_checkpoint_fingerprint"):
        if not str(manifest.get(key, "")).strip():
            raise ValueError(f"synergy basis manifest {key} must be non-empty")
    if _SHA256_RE.fullmatch(str(manifest.get("teacher_checkpoint_fingerprint", ""))) is None:
        raise ValueError("synergy basis manifest teacher_checkpoint_fingerprint must be lowercase 64-hex")
    if not isinstance(manifest.get("normalization"), Mapping):
        raise ValueError("synergy basis normalization must be an object")
    if not isinstance(manifest.get("transform"), Mapping):
        raise ValueError("synergy basis transform semantics must be an object")
    if not isinstance(manifest.get("split_provenance"), Mapping):
        raise ValueError("synergy basis split_provenance must be an object")
    train_motion_uids = manifest.get("train_motion_uids")
    if not isinstance(train_motion_uids, list) or any(not isinstance(value, int) for value in train_motion_uids):
        raise ValueError("synergy basis train_motion_uids must be a list of integers")
    if int(manifest.get("rank", 0)) <= 0:
        raise ValueError("synergy basis manifest rank must be positive")
    composite_version = manifest.get("composite_schema_version")
    if composite_version is not None:
        _validate_regional_composite(
            manifest,
            matrix=np.asarray(matrix, dtype=np.float64),
            muscle_names=tuple(str(name) for name in muscle_names),
        )


def _validate_regional_composite(
    manifest: Mapping[str, Any],
    *,
    matrix: np.ndarray,
    muscle_names: tuple[str, ...],
) -> None:
    if manifest.get("composite_schema_version") != "regional_synergy_composite_v1":
        raise ValueError("unsupported regional composite synergy schema")
    if manifest.get("region") != "regional_composite":
        raise ValueError("regional composite artifact region must be regional_composite")
    regions = manifest.get("composite_regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("regional composite requires non-empty composite_regions")
    owned_rows: list[int] = []
    expected_column = 0
    for item in regions:
        if not isinstance(item, Mapping):
            raise ValueError("every composite region descriptor must be an object")
        label = str(item.get("region", "")).strip()
        rows = item.get("row_indices")
        names = item.get("muscle_names")
        start = int(item.get("column_start", -1))
        stop = int(item.get("column_stop", -1))
        rank = int(item.get("rank", 0))
        if not label or not isinstance(rows, list) or not isinstance(names, list):
            raise ValueError("composite region lacks label/row_indices/muscle_names")
        if start != expected_column or stop != start + rank or rank <= 0:
            raise ValueError("composite region column slices must be positive and contiguous")
        row_indices = [int(value) for value in rows]
        if len(row_indices) != len(set(row_indices)) or any(
            value < 0 or value >= len(muscle_names) for value in row_indices
        ):
            raise ValueError("composite region row indices are invalid")
        expected_names = [muscle_names[value] for value in row_indices]
        if [str(name) for name in names] != expected_names:
            raise ValueError("composite region muscle names/order differ from full schema rows")
        fingerprint = str(item.get("component_artifact_fingerprint", ""))
        if len(fingerprint) != 64:
            raise ValueError("composite region requires a sha256 component artifact fingerprint")
        outside = np.ones(matrix.shape[0], dtype=bool)
        outside[row_indices] = False
        if np.any(np.abs(matrix[outside, start:stop]) > 1e-12):
            raise ValueError("regional composite basis is not block diagonal")
        if np.any(np.linalg.norm(matrix[row_indices, start:stop], axis=0) <= 1e-12):
            raise ValueError("regional composite contains an empty regional component")
        owned_rows.extend(row_indices)
        expected_column = stop
    if sorted(owned_rows) != list(range(len(muscle_names))) or len(owned_rows) != len(set(owned_rows)):
        raise ValueError("regional composite muscle ownership must be unique and complete")
    if expected_column != matrix.shape[1]:
        raise ValueError("regional composite column slices do not cover the full rank")


def _validate_basis(
    basis: np.ndarray,
    muscle_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(basis, dtype=np.float64)
    names = tuple(str(name) for name in muscle_names)
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("synergy basis must be a finite non-empty matrix")
    if np.min(matrix) < -1e-10:
        raise ValueError("synergy basis must be non-negative")
    if len(names) != matrix.shape[0] or len(set(names)) != len(names):
        raise ValueError("muscle_names must uniquely match basis rows")
    if np.any(np.linalg.norm(matrix, axis=0) <= 1e-12):
        raise ValueError("synergy basis contains an empty component")
    return np.maximum(matrix, 0.0), names


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
