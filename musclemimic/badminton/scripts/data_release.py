#!/usr/bin/env python3
"""Create or validate the deterministic ``raw_smooth_v1`` data release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from musclemimic.badminton.data_qc import TRAIN_MOTIONS, VAL_MOTIONS
from musclemimic.badminton.scripts import materialize_raw_smooth_cache_override as cache_override
from musclemimic.badminton.scripts.prepare_raw_smooth_sources import (
    PROVENANCE_SCHEMA_VERSION,
    _canonical_json_bytes,
    _recipe_digest,
    load_recipe,
    sha256_file,
)

SCHEMA_VERSION = "forehand_clear_raw_smooth_release_v3"
SOURCE_VARIANT = "raw_smooth_v1"
CACHE_VARIANT = "raw_smooth_v1"
SOURCE_FPS = 60.0
CACHE_FPS = 100.0
DAMPING = 1.0
USE_VELOCITY_LIMIT = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_recipe() -> Path:
    return Path(__file__).with_name("raw_smooth_v1_recipe.json")


def _default_ik_config() -> Path:
    return _repo_root() / "loco_mujoco" / "smpl" / "gmr_configs" / "smplh_to_myofullbody_smooth_train.json"


def _default_cache_override_recipe() -> Path:
    return cache_override.default_recipe_path()


def _safe_paths(dataset_root: str | Path) -> tuple[Path, Path, Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    source_dir = (root / "temp" / SOURCE_VARIANT).resolve()
    cache_dir = (root / "muscle_trajectory" / CACHE_VARIANT).resolve()
    manifest_dir = (root / "manifests" / CACHE_VARIANT).resolve()
    expected = (
        root / "temp" / SOURCE_VARIANT,
        root / "muscle_trajectory" / CACHE_VARIANT,
        root / "manifests" / CACHE_VARIANT,
    )
    if (source_dir, cache_dir, manifest_dir) != expected:
        raise ValueError("raw_smooth_v1 release paths must not be symlinks or escape the dataset root")
    return root, source_dir, cache_dir, manifest_dir


def _read_manifest_motions(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    motions: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        motions.append(Path(line.removesuffix(".npz")).name)
    return motions


def _split_contract(manifest_dir: Path) -> dict[str, str]:
    train_path = manifest_dir / "train_list.txt"
    val_path = manifest_dir / "val_list.txt"
    train = _read_manifest_motions(train_path)
    val = _read_manifest_motions(val_path)
    if train != list(TRAIN_MOTIONS):
        raise ValueError("raw_smooth_v1 train manifest is not the canonical ordered 22-motion split")
    if val != list(VAL_MOTIONS):
        raise ValueError("raw_smooth_v1 validation manifest is not the canonical ordered 5-motion split")
    if set(train) & set(val) or len({*train, *val}) != 27:
        raise ValueError("raw_smooth_v1 train/validation split overlaps or is incomplete")
    return dict.fromkeys(train, "train") | dict.fromkeys(val, "validation")


def _source_metadata(path: Path, *, motion: str) -> tuple[int, list[float]]:
    with np.load(path, allow_pickle=True) as source:
        if "poses" not in source:
            raise ValueError(f"{motion}: source missing poses")
        frames = int(np.asarray(source["poses"]).shape[0])
        rates = []
        for field in ("mocap_framerate", "mocap_frame_rate"):
            if field not in source:
                raise ValueError(f"{motion}: source missing {field}")
            rates.append(float(np.asarray(source[field]).reshape(-1)[0]))
    if frames < 2 or any(not np.isclose(rate, SOURCE_FPS, rtol=0.0, atol=1e-6) for rate in rates):
        raise ValueError(f"{motion}: invalid source frames/FPS: frames={frames}, fps={rates}")
    return frames, rates


def _cache_metadata(path: Path, *, motion: str) -> tuple[int, float, int, int]:
    with np.load(path, allow_pickle=True) as cache:
        for field in ("qpos", "qvel", "frequency"):
            if field not in cache:
                raise ValueError(f"{motion}: cache missing {field}")
        qpos = np.asarray(cache["qpos"])
        qvel = np.asarray(cache["qvel"])
        rate = float(np.asarray(cache["frequency"]).reshape(-1)[0])
    if (
        qpos.ndim != 2
        or qpos.shape[0] < 2
        or qpos.shape[1] != 89
        or qvel.shape != (qpos.shape[0], 88)
        or not np.isclose(rate, CACHE_FPS, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(
            f"{motion}: invalid cache contract qpos={qpos.shape}, qvel={qvel.shape}, fps={rate}"
        )
    return int(qpos.shape[0]), rate, int(qpos.shape[1]), int(qvel.shape[1])


def _load_source_sidecar(
    path: Path,
    *,
    motion: str,
    source_path: Path,
    source_hash: str,
    recipe_sha256: str,
    source_override: dict[str, Any] | None,
    override_recipe_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{motion}: missing source repair sidecar: {path}")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "motion": motion,
        "output_sha256": source_hash,
        "recipe_sha256": recipe_sha256,
    }
    for field, value in expected.items():
        if sidecar.get(field) != value:
            raise ValueError(
                f"{motion}: repair sidecar {field} mismatch: {sidecar.get(field)!r} != {value!r}"
            )
    if Path(sidecar.get("output_path", "")).name != source_path.name:
        raise ValueError(f"{motion}: repair sidecar output path mismatch")
    if source_override is None:
        if sidecar.get("release_source_override") is not None:
            raise ValueError(f"{motion}: unregistered final-release source override")
    else:
        override_expected = {
            "identity_operation": source_override["operation"],
            "source_path": source_override["protected_source_path"],
            "source_sha256": source_override["protected_source_sha256"],
            "source_crop": source_override["source_crop"],
            "output_path": source_override["output_source_path"],
            "override_recipe_sha256": override_recipe_sha256,
            "release_source_override": source_override,
            "supersedes_base_recipe_repair": source_override[
                "supersedes_base_recipe_repair"
            ],
        }
        for field, value in override_expected.items():
            if sidecar.get(field) != value:
                raise ValueError(
                    f"{motion}: final source sidecar {field} mismatch: "
                    f"{sidecar.get(field)!r} != {value!r}"
                )
    return sidecar


def _load_cache_override_sidecar(
    root: Path,
    entry: dict[str, Any],
    *,
    cache_path: Path,
    cache_hash: str,
    override_recipe_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    motion = str(entry["motion"])
    protected_path = (root / entry["source_cache_path"]).resolve()
    expected_protected_path = root / Path(entry["source_cache_path"])
    if protected_path != expected_protected_path:
        raise ValueError(f"{motion}: protected fallback path is a symlink/alias")
    if not protected_path.is_file() or protected_path.is_symlink():
        raise FileNotFoundError(f"{motion}: protected fallback cache missing/symlinked")
    protected_hash = sha256_file(protected_path)
    if protected_hash != entry["source_cache_sha256"]:
        raise ValueError(
            f"{motion}: protected fallback hash mismatch: "
            f"{protected_hash} != {entry['source_cache_sha256']}"
        )
    sidecar_path = (root / entry["provenance_sidecar_path"]).resolve()
    expected_sidecar_path = root / Path(entry["provenance_sidecar_path"])
    if sidecar_path != expected_sidecar_path or not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise FileNotFoundError(f"{motion}: cache override provenance sidecar missing/symlinked")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": cache_override.PROVENANCE_SCHEMA_VERSION,
        "motion": motion,
        "operation": entry["operation"],
        "source_cache_path": entry["source_cache_path"],
        "source_cache_sha256": entry["source_cache_sha256"],
        "source_cache_crop": entry["source_cache_crop"],
        "output_cache_path": entry["output_cache_path"],
        "output_cache_sha256": cache_hash,
        "output_analysis_path": entry["output_analysis_path"],
        "output_analysis_policy": "removed_and_must_be_absent",
        "provenance_sidecar_path": entry["provenance_sidecar_path"],
        "override_recipe_sha256": override_recipe_sha256,
        "filter": entry["filter"],
        "source_output_override": entry["source_output_override"],
        "reason": entry["reason"],
        "protected_source_hash_unchanged": True,
    }
    for field, value in expected.items():
        if sidecar.get(field) != value:
            raise ValueError(
                f"{motion}: cache override sidecar {field} mismatch: "
                f"{sidecar.get(field)!r} != {value!r}"
            )
    output_analysis = (root / entry["output_analysis_path"]).resolve()
    if output_analysis != root / Path(entry["output_analysis_path"]):
        raise ValueError(f"{motion}: override analysis path is a symlink/alias")
    if output_analysis.exists():
        raise ValueError(f"{motion}: stale smooth analysis remains after cache override")
    if cache_path != root / Path(entry["output_cache_path"]):
        raise ValueError(f"{motion}: override output path does not match release cache")
    return sidecar_path, sidecar


def _release_id(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("release_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def build_release_manifest(
    dataset_root: str | Path,
    *,
    recipe_path: str | Path | None = None,
    ik_config_path: str | Path | None = None,
    cache_override_recipe_path: str | Path | None = None,
) -> dict[str, Any]:
    root, source_dir, cache_dir, manifest_dir = _safe_paths(dataset_root)
    recipe_path = Path(recipe_path or _default_recipe()).expanduser().resolve()
    ik_config_path = Path(ik_config_path or _default_ik_config()).expanduser().resolve()
    cache_override_recipe_path = Path(
        cache_override_recipe_path or _default_cache_override_recipe()
    ).expanduser().resolve()
    recipe = load_recipe(recipe_path)
    recipe_sha256 = _recipe_digest(recipe)
    override_recipe = cache_override.load_override_recipe(cache_override_recipe_path)
    override_recipe_sha256 = cache_override.recipe_sha256(override_recipe)
    overrides_by_motion = {
        str(entry["motion"]): entry for entry in override_recipe["overrides"]
    }
    release_source_overrides = recipe.get("release_source_overrides")
    expected_source_overrides = {
        motion: entry["source_output_override"]
        for motion, entry in overrides_by_motion.items()
        if entry["source_output_override"] is not None
    }
    if release_source_overrides != expected_source_overrides:
        raise ValueError(
            "source recipe release_source_overrides do not match cache override recipe"
        )
    if not ik_config_path.is_file():
        raise FileNotFoundError(f"smooth IK config missing: {ik_config_path}")
    split_by_motion = _split_contract(manifest_dir)
    repair_by_motion = recipe["repairs"]
    records: list[dict[str, Any]] = []
    for motion in (*TRAIN_MOTIONS, *VAL_MOTIONS):
        source_path = source_dir / f"{motion}.npz"
        cache_path = cache_dir / f"{motion}.npz"
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"{motion}: source missing/symlinked: {source_path}")
        if not cache_path.is_file() or cache_path.is_symlink():
            raise FileNotFoundError(f"{motion}: cache missing/symlinked: {cache_path}")
        source_hash = sha256_file(source_path)
        cache_hash = sha256_file(cache_path)
        source_frames, source_rates = _source_metadata(source_path, motion=motion)
        cache_frames, cache_rate, qpos_dim, qvel_dim = _cache_metadata(cache_path, motion=motion)
        sidecar_path = source_dir / f"{motion}.source_provenance.json"
        override = overrides_by_motion.get(motion)
        source_override = None if override is None else override["source_output_override"]
        sidecar = _load_source_sidecar(
            sidecar_path,
            motion=motion,
            source_path=source_path,
            source_hash=source_hash,
            recipe_sha256=recipe_sha256,
            source_override=source_override,
            override_recipe_sha256=override_recipe_sha256,
        )
        if source_override is None and sidecar.get("repair") != repair_by_motion.get(motion):
            raise ValueError(f"{motion}: repair sidecar does not match release recipe")
        if source_override is not None and sidecar.get(
            "supersedes_base_recipe_repair"
        ) != repair_by_motion.get(motion):
            raise ValueError(
                f"{motion}: final source override does not identify the superseded base repair"
            )
        if override is None:
            unexpected_override_sidecar = cache_dir / f"{motion}.cache_provenance.json"
            if unexpected_override_sidecar.exists():
                raise ValueError(f"{motion}: unregistered cache override sidecar exists")
            cache_generation = {
                "mode": "default_smooth_retarget",
                "default_smooth_retarget_contract_applied": True,
            }
        else:
            override_sidecar_path, override_sidecar = _load_cache_override_sidecar(
                root,
                override,
                cache_path=cache_path,
                cache_hash=cache_hash,
                override_recipe_sha256=override_recipe_sha256,
            )
            cache_generation = {
                "mode": "filtered_protected_raw_cache_fallback",
                "default_smooth_retarget_contract_applied": False,
                "reason": override["reason"],
                "protected_input": {
                    "path": override["source_cache_path"],
                    "sha256": override["source_cache_sha256"],
                    "crop": override["source_cache_crop"],
                },
                "filter": override["filter"],
                "override_recipe_entry": override,
                "provenance_sidecar": {
                    "path": str(override_sidecar_path.relative_to(root)),
                    "sha256": sha256_file(override_sidecar_path),
                    "output_analysis_policy": override_sidecar[
                        "output_analysis_policy"
                    ],
                },
            }
        records.append(
            {
                "motion": motion,
                "split": split_by_motion[motion],
                "source": {
                    "path": str(source_path.relative_to(root)),
                    "sha256": source_hash,
                    "frames": source_frames,
                    "mocap_framerate": source_rates[0],
                    "mocap_frame_rate": source_rates[1],
                },
                "cache": {
                    "path": str(cache_path.relative_to(root)),
                    "sha256": cache_hash,
                    "frames": cache_frames,
                    "frequency": cache_rate,
                    "qpos_dim": qpos_dim,
                    "qvel_dim": qvel_dim,
                },
                "cache_generation": cache_generation,
                "source_generation": (
                    {
                        "mode": "release_source_override",
                        "contract": source_override,
                        "final_operation": sidecar["repair"],
                        "base_recipe_repair_superseded": repair_by_motion.get(motion),
                    }
                    if source_override is not None
                    else {
                        "mode": "base_source_recipe",
                        "base_recipe_repair": repair_by_motion.get(motion),
                    }
                ),
                "repair_sidecar": {
                    "path": str(sidecar_path.relative_to(root)),
                    "sha256": sha256_file(sidecar_path),
                    "identity_operation": sidecar["identity_operation"],
                    "repair": sidecar.get("repair"),
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "forehandClear_standard",
        "source_variant": SOURCE_VARIANT,
        "cache_variant": CACHE_VARIANT,
        "split": {
            "train_count": len(TRAIN_MOTIONS),
            "validation_count": len(VAL_MOTIONS),
            "overlap_count": 0,
        },
        "source_recipe": {
            "path": str(recipe_path.relative_to(_repo_root()))
            if recipe_path.is_relative_to(_repo_root())
            else str(recipe_path),
            "sha256": recipe_sha256,
            "identity_count": len(recipe["identity"]),
            "repair_count": len(recipe["repairs"]),
            "release_source_override_count": len(release_source_overrides),
            "release_source_override_motions": sorted(release_source_overrides),
        },
        "default_smooth_retarget_contract": {
            "scope": "all_non_overridden_motions",
            "motion_count": 27 - len(overrides_by_motion),
            "excluded_motions": sorted(overrides_by_motion),
            "source_fps": SOURCE_FPS,
            "gmr_target_fps": SOURCE_FPS,
            "control_cache_fps": CACHE_FPS,
            "frequency_transition": "60->100",
            "solver": "daqp",
            "damping": DAMPING,
            "use_velocity_limit": USE_VELOCITY_LIMIT,
            "ik_config_path": str(ik_config_path.relative_to(_repo_root()))
            if ik_config_path.is_relative_to(_repo_root())
            else str(ik_config_path),
            "ik_config_sha256": sha256_file(ik_config_path),
        },
        "cache_override_contract": {
            "path": str(cache_override_recipe_path.relative_to(_repo_root()))
            if cache_override_recipe_path.is_relative_to(_repo_root())
            else str(cache_override_recipe_path),
            "sha256": override_recipe_sha256,
            "motion_count": len(overrides_by_motion),
            "motions": sorted(overrides_by_motion),
            "note": (
                "Overrides are provenance-bound exceptions; the default smooth "
                "retarget contract does not apply to these motions."
            ),
        },
        "motions": records,
    }
    payload["release_sha256"] = _release_id(payload)
    return payload


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


def write_release_manifest(path: str | Path, manifest: dict[str, Any], *, replace: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not replace:
        raise FileExistsError(f"release manifest exists; pass --replace: {destination}")
    _atomic_write(destination, _canonical_json_bytes(manifest))


def validate_release_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    recipe_path: str | Path | None = None,
    ik_config_path: str | Path | None = None,
    cache_override_recipe_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    errors: list[str] = []
    try:
        persisted = json.loads(path.read_text(encoding="utf-8"))
        current = build_release_manifest(
            dataset_root,
            recipe_path=recipe_path,
            ik_config_path=ik_config_path,
            cache_override_recipe_path=cache_override_recipe_path,
        )
        if persisted != current:
            errors.append("persisted release manifest does not exactly match current files/contracts")
        if persisted.get("release_sha256") != _release_id(persisted):
            errors.append("persisted release_sha256 is invalid")
    except Exception as exc:
        errors.append(str(exc))
        persisted = {}
        current = {}
    return {
        "schema_version": "forehand_clear_raw_smooth_release_validation_v3",
        "manifest_path": str(path.expanduser().resolve()),
        "release_sha256": persisted.get("release_sha256"),
        "current_release_sha256": current.get("release_sha256"),
        "errors": errors,
        "passed": not errors,
    }


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
    parser.add_argument("--ik-config", "--ik_config", dest="ik_config", type=Path, default=_default_ik_config())
    parser.add_argument(
        "--cache-override-recipe",
        "--cache_override_recipe",
        dest="cache_override_recipe",
        type=Path,
        default=_default_cache_override_recipe(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/forehandClear_standard/manifests/raw_smooth_v1/release_manifest.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.write:
        manifest = build_release_manifest(
            args.dataset_root,
            recipe_path=args.recipe,
            ik_config_path=args.ik_config,
            cache_override_recipe_path=args.cache_override_recipe,
        )
        write_release_manifest(args.output, manifest, replace=args.replace)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0
    report = validate_release_manifest(
        args.dataset_root,
        args.output,
        recipe_path=args.recipe,
        ik_config_path=args.ik_config,
        cache_override_recipe_path=args.cache_override_recipe,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
