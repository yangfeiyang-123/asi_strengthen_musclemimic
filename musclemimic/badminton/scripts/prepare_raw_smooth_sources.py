#!/usr/bin/env python3
"""Build the immutable-input ``raw_smooth_v1`` AMASS source namespace.

The command reads canonical WHAM-derived NPZ files, verifies their pinned
SHA-256 values, applies the three reviewed repairs, and writes only to
``temp/raw_smooth_v1``.  It never mutates raw/optimized/initial inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "raw_smooth_source_recipe_v1"
PROVENANCE_SCHEMA_VERSION = "raw_smooth_source_provenance_v1"
REQUIRED_OUTPUT_SUBDIR = Path("temp/raw_smooth_v1")
REQUIRED_SOURCE_SUBDIR = Path("wham/initiall_wham")
EXPECTED_IDENTITY_COUNT = 27
EXPECTED_REPAIR_COUNT = 3
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _recipe_digest(recipe: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(recipe)).hexdigest()


def load_recipe(path: str | Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"recipe schema must be {SCHEMA_VERSION!r}, got "
            f"{recipe.get('schema_version')!r}"
        )
    if Path(str(recipe.get("source_subdir", ""))) != REQUIRED_SOURCE_SUBDIR:
        raise ValueError(f"source_subdir must be {REQUIRED_SOURCE_SUBDIR}")
    if Path(str(recipe.get("output_subdir", ""))) != REQUIRED_OUTPUT_SUBDIR:
        raise ValueError(f"output_subdir must be {REQUIRED_OUTPUT_SUBDIR}")
    identities = recipe.get("identity")
    repairs = recipe.get("repairs")
    if not isinstance(identities, list) or len(identities) != EXPECTED_IDENTITY_COUNT:
        raise ValueError(f"recipe must contain exactly {EXPECTED_IDENTITY_COUNT} identity entries")
    if not isinstance(repairs, dict) or len(repairs) != EXPECTED_REPAIR_COUNT:
        raise ValueError(f"recipe must contain exactly {EXPECTED_REPAIR_COUNT} repairs")
    motions: list[str] = []
    for item in identities:
        if not isinstance(item, dict):
            raise ValueError("each identity entry must be an object")
        motion = str(item.get("motion", ""))
        digest = str(item.get("source_sha256", ""))
        if not motion or Path(motion).name != motion or motion in {".", ".."}:
            raise ValueError(f"unsafe motion name: {motion!r}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{motion}: invalid source_sha256")
        motions.append(motion)
    if len(set(motions)) != len(motions):
        raise ValueError("identity motion names must be unique")
    if not set(repairs).issubset(motions):
        raise ValueError("every repair must reference an identity motion")
    _validate_repair_contract(repairs)
    return recipe


def _validate_repair_contract(repairs: Mapping[str, Any]) -> None:
    expected = {
        "6月2日-5": ("crop", 40, 114),
        "6月2日-6": ("crop", 0, 76),
    }
    for motion, (kind, start, stop) in expected.items():
        repair = repairs.get(motion)
        if not isinstance(repair, dict):
            raise ValueError(f"missing reviewed repair for {motion}")
        if (repair.get("type"), repair.get("start"), repair.get("stop")) != (
            kind,
            start,
            stop,
        ):
            raise ValueError(f"{motion}: crop must be [{start}:{stop}]")
    video1 = repairs.get("video1")
    if not isinstance(video1, dict) or (
        video1.get("type"),
        video1.get("field"),
        video1.get("frame"),
        video1.get("method"),
    ) != (
        "translation_continuity",
        "trans",
        29,
        "constant_offset_to_linear_prediction",
    ):
        raise ValueError("video1 must use the reviewed frame-29 translation continuity repair")


def _safe_dataset_paths(dataset_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    source_dir = (root / REQUIRED_SOURCE_SUBDIR).resolve()
    output_dir = (root / REQUIRED_OUTPUT_SUBDIR).resolve()
    if source_dir != root / REQUIRED_SOURCE_SUBDIR:
        raise ValueError(f"source path is a symlink/escape: {source_dir}")
    if output_dir != root / REQUIRED_OUTPUT_SUBDIR:
        raise ValueError(
            "refusing output symlink/escape; raw_smooth_v1 must resolve exactly to "
            f"{root / REQUIRED_OUTPUT_SUBDIR}"
        )
    protected = {
        (root / "muscle_trajectory" / name).resolve()
        for name in ("raw", "optimized", "initial")
    }
    protected.add(source_dir)
    if output_dir in protected:
        raise ValueError(f"output namespace collides with protected data: {output_dir}")
    return root, source_dir, output_dir


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _motion_frames(arrays: Mapping[str, np.ndarray], *, motion: str) -> int:
    if "poses" not in arrays:
        raise ValueError(f"{motion}: source has no poses array")
    poses = np.asarray(arrays["poses"])
    if poses.ndim < 1 or poses.shape[0] < 2:
        raise ValueError(f"{motion}: invalid poses shape {poses.shape}")
    return int(poses.shape[0])


def _validate_fps(arrays: Mapping[str, np.ndarray], *, motion: str) -> None:
    values: list[float] = []
    for field in ("mocap_framerate", "mocap_frame_rate"):
        if field not in arrays:
            raise ValueError(f"{motion}: source missing {field}")
        values.append(float(np.asarray(arrays[field]).reshape(-1)[0]))
    if any(not np.isclose(value, 60.0, rtol=0.0, atol=1e-6) for value in values):
        raise ValueError(f"{motion}: expected both source FPS fields to equal 60, got {values}")


def apply_repair(
    arrays: Mapping[str, np.ndarray],
    repair: Mapping[str, Any] | None,
    *,
    motion: str,
) -> dict[str, np.ndarray]:
    result = {name: np.array(value, copy=True) for name, value in arrays.items()}
    source_frames = _motion_frames(result, motion=motion)
    if repair is None:
        return result
    kind = repair["type"]
    if kind == "crop":
        start = int(repair["start"])
        raw_stop = repair.get("stop")
        stop = source_frames if raw_stop is None else int(raw_stop)
        if not 0 <= start < stop <= source_frames:
            raise ValueError(
                f"{motion}: invalid crop [{start}:{stop}] for {source_frames} frames"
            )
        for name, value in tuple(result.items()):
            array = np.asarray(value)
            if array.ndim > 0 and array.shape[0] == source_frames:
                result[name] = np.array(array[start:stop], copy=True)
        return result
    if kind == "translation_continuity":
        field = str(repair["field"])
        frame = int(repair["frame"])
        if field not in result:
            raise ValueError(f"{motion}: translation field {field!r} is missing")
        translation = np.asarray(result[field])
        if translation.shape != (source_frames, 3) or not 2 <= frame < source_frames:
            raise ValueError(
                f"{motion}: cannot repair frame {frame} in {field} shape {translation.shape}"
            )
        repaired = np.array(translation, copy=True)
        predicted = 2.0 * repaired[frame - 1] - repaired[frame - 2]
        offset = predicted - repaired[frame]
        repaired[frame:] += offset
        result[field] = repaired
        return result
    raise ValueError(f"{motion}: unsupported repair type {kind!r}")


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode an NPZ with fixed ordering, metadata, permissions and timestamp."""
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"unsafe NPZ field name: {name!r}")
            npy = io.BytesIO()
            np.save(npy, np.asarray(arrays[name]), allow_pickle=True)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def prepare_sources(
    dataset_root: str | Path,
    recipe_path: str | Path,
    *,
    dry_run: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    recipe = load_recipe(recipe_path)
    root, source_dir, output_dir = _safe_dataset_paths(dataset_root)
    recipe_sha256 = _recipe_digest(recipe)
    identities = recipe["identity"]
    repairs = recipe["repairs"]
    source_hashes_before: dict[str, str] = {}
    planned: list[dict[str, Any]] = []

    for identity in identities:
        motion = identity["motion"]
        source_path = source_dir / f"{motion}__original.npz"
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"{motion}: immutable source missing or symlinked: {source_path}")
        actual_hash = sha256_file(source_path)
        expected_hash = identity["source_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"{motion}: immutable source hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        source_hashes_before[motion] = actual_hash
        destination = output_dir / f"{motion}.npz"
        sidecar = output_dir / f"{motion}.source_provenance.json"
        if not dry_run and not replace and (destination.exists() or sidecar.exists()):
            raise FileExistsError(
                f"{motion}: output exists; pass --replace to replace only raw_smooth_v1"
            )
        planned.append(
            {
                "motion": motion,
                "source_path": str(source_path.relative_to(root)),
                "output_path": str(destination.relative_to(root)),
                "repair": repairs.get(motion),
            }
        )

    result: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset_root": str(root),
        "source_subdir": str(REQUIRED_SOURCE_SUBDIR),
        "output_subdir": str(REQUIRED_OUTPUT_SUBDIR),
        "recipe_path": str(Path(recipe_path).expanduser().resolve()),
        "recipe_sha256": recipe_sha256,
        "identity_count": len(identities),
        "repair_count": len(repairs),
        "dry_run": bool(dry_run),
        "motions": planned,
    }
    if dry_run:
        return result

    emitted: list[dict[str, Any]] = []
    for identity in identities:
        motion = identity["motion"]
        source_path = source_dir / f"{motion}__original.npz"
        destination = output_dir / f"{motion}.npz"
        sidecar_path = output_dir / f"{motion}.source_provenance.json"
        arrays = _load_npz(source_path)
        _validate_fps(arrays, motion=motion)
        source_frames = _motion_frames(arrays, motion=motion)
        repair = repairs.get(motion)
        repaired = apply_repair(arrays, repair, motion=motion)
        _validate_fps(repaired, motion=motion)
        output_frames = _motion_frames(repaired, motion=motion)
        payload = deterministic_npz_bytes(repaired)
        output_hash = hashlib.sha256(payload).hexdigest()
        sidecar = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "motion": motion,
            "identity_operation": "array-value-preserving-deterministic-reencode",
            "repair": repair,
            "source_path": str(source_path.relative_to(root)),
            "source_sha256": source_hashes_before[motion],
            "source_frames": source_frames,
            "source_fps": 60.0,
            "output_path": str(destination.relative_to(root)),
            "output_sha256": output_hash,
            "output_frames": output_frames,
            "output_fps": 60.0,
            "recipe_sha256": recipe_sha256,
        }
        _atomic_write(destination, payload)
        _atomic_write(sidecar_path, _canonical_json_bytes(sidecar))
        emitted.append(sidecar)

    for identity in identities:
        motion = identity["motion"]
        source_path = source_dir / f"{motion}__original.npz"
        after = sha256_file(source_path)
        if after != source_hashes_before[motion]:
            raise RuntimeError(f"{motion}: protected source changed while preparing release")

    result["dry_run"] = False
    result["motions"] = emitted
    result["source_hash_protection_passed"] = True
    _atomic_write(output_dir / "preparation_provenance.json", _canonical_json_bytes(result))
    return result


def _default_recipe() -> Path:
    return Path(__file__).with_name("raw_smooth_v1_recipe.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        default=Path("datasets/forehandClear_standard"),
    )
    parser.add_argument("--recipe", type=Path, default=_default_recipe())
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace generated files only inside temp/raw_smooth_v1",
    )
    args = parser.parse_args()
    report = prepare_sources(
        args.dataset_root,
        args.recipe,
        dry_run=args.dry_run,
        replace=args.replace,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
