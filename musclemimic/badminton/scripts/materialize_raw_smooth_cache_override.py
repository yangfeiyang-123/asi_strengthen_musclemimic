#!/usr/bin/env python3
"""Materialize the six audited protected-cache fallbacks in raw_smooth_v1.

The recipe is deliberately closed: motion names, protected inputs, SHA-256
digests, cache/source crops, output paths and filter parameters are all pinned.
Only ``temp/raw_smooth_v1`` and ``muscle_trajectory/raw_smooth_v1`` are writable.
The protected raw/optimized/initial namespaces are never opened for writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from musclemimic.badminton.scripts.prepare_raw_smooth_sources import (
    PROVENANCE_SCHEMA_VERSION as SOURCE_PROVENANCE_SCHEMA_VERSION,
)
from musclemimic.badminton.scripts.prepare_raw_smooth_sources import (
    _canonical_json_bytes,
    _recipe_digest,
    deterministic_npz_bytes,
    sha256_file,
)
from musclemimic.badminton.scripts.prepare_raw_smooth_sources import (
    load_recipe as load_source_recipe,
)

RECIPE_SCHEMA_VERSION = "raw_smooth_cache_override_recipe_v2"
PROVENANCE_SCHEMA_VERSION = "raw_smooth_cache_override_provenance_v2"
MATERIALIZATION_SCHEMA_VERSION = "raw_smooth_cache_override_materialization_v2"
PINNED_OPERATION = "deterministic_filtered_protected_raw_cache_fallback"
PINNED_MOTIONS = (
    "6月2日-1",
    "6月2日-2",
    "6月2日-3",
    "6月2日-4",
    "6月2日-5",
    "6月2日-7",
)
PINNED_FILTER = {
    "name": "gaussian_lowpass_plus_velocity_limit_with_fk_recompute",
    "joint_sigma": 0.4,
    "max_joint_speed_rad_s": 100.0,
    "root_sigma": 0.0,
    "max_root_speed_m_s": 100.0,
}
_PINNED_CACHE_HASHES = {
    "6月2日-1": "ce0aa018b20823fc8c2daa2f78db5a6e98199d60233ec0a10c211b5805248a2e",
    "6月2日-2": "027625cb04768c195f3a00644ea63b1ecbd4945586c297bf32658d64f1cf47bb",
    "6月2日-3": "0963f8c67c0195cd533206815cc090791bd3009d5f7e4d761b07000fbcbaf3e7",
    "6月2日-4": "a343f47d85a7cdf20dca5ecdb44656196d882f7842f1bcd05b38fe8e54733324",
    "6月2日-5": "b1aea23b560ca7155478a05e8059de43f5dae2cfb7f1bb3905cc769563e066b2",
    "6月2日-7": "db98b78ef04a8aee82fc94ac6f61344dff35109887439fe37c6a5c1f6e28cf9f",
}
_PINNED_SOURCE_OVERRIDES: dict[str, dict[str, Any]] = {
    "6月2日-3": {
        "operation": "deterministic_source_prefix_crop",
        "protected_source_path": "wham/initiall_wham/6月2日-3__original.npz",
        "protected_source_sha256": "f60f5439ff49f3f026b77cb44072900a11a70ee0b57d9fc3b4e8be2f9ee1bf57",
        "source_crop": {"start": 9, "stop": None},
        "output_source_path": "temp/raw_smooth_v1/6月2日-3.npz",
        "source_provenance_sidecar_path": "temp/raw_smooth_v1/6月2日-3.source_provenance.json",
        "supersedes_base_recipe_repair": None,
    },
    "6月2日-5": {
        "operation": "deterministic_full_source_restore",
        "protected_source_path": "wham/initiall_wham/6月2日-5__original.npz",
        "protected_source_sha256": "12a8c12b8ba55fb9e37793e4f8468974bdaabf8f13af56c92b89fb5c99aa04e8",
        "source_crop": {"start": 0, "stop": None},
        "output_source_path": "temp/raw_smooth_v1/6月2日-5.npz",
        "source_provenance_sidecar_path": "temp/raw_smooth_v1/6月2日-5.source_provenance.json",
        "supersedes_base_recipe_repair": {
            "type": "crop",
            "start": 40,
            "stop": 114,
            "reason": "remove source discontinuities before frame 40 and at frame 114",
        },
    },
}

# Backward-compatible names for code that only queried the historical -7 entry.
PINNED_MOTION = "6月2日-7"
PINNED_SOURCE_CACHE_PATH = Path("muscle_trajectory/raw/6月2日-7.npz")
PINNED_SOURCE_CACHE_SHA256 = _PINNED_CACHE_HASHES[PINNED_MOTION]
PINNED_OUTPUT_CACHE_PATH = Path("muscle_trajectory/raw_smooth_v1/6月2日-7.npz")
PINNED_OUTPUT_ANALYSIS_PATH = Path("muscle_trajectory/raw_smooth_v1/6月2日-7_analysis.npz")
PINNED_PROVENANCE_PATH = Path("muscle_trajectory/raw_smooth_v1/6月2日-7.cache_provenance.json")


def default_recipe_path() -> Path:
    return Path(__file__).with_name("raw_smooth_v1_cache_overrides.json")


def default_source_recipe_path() -> Path:
    return Path(__file__).with_name("raw_smooth_v1_recipe.json")


def recipe_sha256(recipe: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(recipe)).hexdigest()


def _expected_entry(motion: str) -> dict[str, Any]:
    return {
        "motion": motion,
        "operation": PINNED_OPERATION,
        "source_cache_path": f"muscle_trajectory/raw/{motion}.npz",
        "source_cache_sha256": _PINNED_CACHE_HASHES[motion],
        "source_analysis_path": None,
        "source_cache_crop": {
            "start": 15 if motion == "6月2日-3" else 0,
            "stop": None,
        },
        "output_cache_path": f"muscle_trajectory/raw_smooth_v1/{motion}.npz",
        "output_analysis_path": f"muscle_trajectory/raw_smooth_v1/{motion}_analysis.npz",
        "provenance_sidecar_path": (
            f"muscle_trajectory/raw_smooth_v1/{motion}.cache_provenance.json"
        ),
        "filter": PINNED_FILTER,
        "source_output_override": _PINNED_SOURCE_OVERRIDES.get(motion),
    }


def load_override_recipe(path: str | Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise ValueError(f"override recipe schema must be {RECIPE_SCHEMA_VERSION}")
    if recipe.get("dataset") != "forehandClear_standard":
        raise ValueError("override recipe dataset must be forehandClear_standard")
    overrides = recipe.get("overrides")
    if not isinstance(overrides, list) or [row.get("motion") for row in overrides] != list(PINNED_MOTIONS):
        raise ValueError(
            "override recipe must contain exactly the six pinned motions in canonical order"
        )
    for entry in overrides:
        motion = str(entry["motion"])
        for field, value in _expected_entry(motion).items():
            if entry.get(field) != value:
                raise ValueError(
                    f"{motion}: override {field} must be pinned to {value!r}, "
                    f"got {entry.get(field)!r}"
                )
        if not str(entry.get("reason", "")).strip():
            raise ValueError(f"{motion}: override reason must be non-empty")
    return recipe


def _load_and_bind_source_recipe(path: str | Path, override_recipe: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    source_recipe = load_source_recipe(path)
    source_overrides = source_recipe.get("release_source_overrides")
    if source_overrides != _PINNED_SOURCE_OVERRIDES:
        raise ValueError(
            "raw_smooth_v1 source recipe release_source_overrides do not match the pinned final release"
        )
    cache_source_overrides = {
        str(entry["motion"]): entry["source_output_override"]
        for entry in override_recipe["overrides"]
        if entry["source_output_override"] is not None
    }
    if cache_source_overrides != source_overrides:
        raise ValueError("source and cache override recipes disagree")
    return source_recipe, _recipe_digest(source_recipe)


def _resolve_exact(root: Path, relative: str | Path, *, label: str) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe dataset-relative path: {relative}")
    lexical = root / relative
    resolved = lexical.resolve()
    if resolved != lexical:
        raise ValueError(f"{label} must not be a symlink or path alias: {resolved}")
    return resolved


def _validate_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as cache:
        for field in ("qpos", "qvel", "frequency"):
            if field not in cache:
                raise ValueError(f"protected fallback cache missing {field}")
        qpos = np.asarray(cache["qpos"])
        qvel = np.asarray(cache["qvel"])
        frequency = float(np.asarray(cache["frequency"]).reshape(-1)[0])
    if qpos.ndim != 2 or qpos.shape[1] != 89:
        raise ValueError(f"protected fallback qpos must be [T,89], got {qpos.shape}")
    if qvel.shape != (qpos.shape[0], 88):
        raise ValueError(f"protected fallback qvel must be [T,88], got {qvel.shape}")
    if not np.isclose(frequency, 100.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"protected fallback frequency must be 100, got {frequency}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("protected fallback cache contains NaN/Inf")
    return {
        "frames": int(qpos.shape[0]),
        "frequency": frequency,
        "qpos_dim": int(qpos.shape[1]),
        "qvel_dim": int(qvel.shape[1]),
    }


def _load_source_arrays(path: Path, *, motion: str) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path, allow_pickle=True) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    if "poses" not in arrays:
        raise ValueError(f"{motion}: protected source missing poses")
    frames = int(np.asarray(arrays["poses"]).shape[0])
    if frames < 2:
        raise ValueError(f"{motion}: protected source has too few frames")
    for field in ("mocap_framerate", "mocap_frame_rate"):
        if field not in arrays or not np.isclose(
            float(np.asarray(arrays[field]).reshape(-1)[0]), 60.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"{motion}: protected source {field} must equal 60")
    return arrays, frames


def _crop_bounds(crop: Mapping[str, Any], frames: int, *, motion: str, label: str) -> tuple[int, int]:
    start = int(crop["start"])
    stop = frames if crop.get("stop") is None else int(crop["stop"])
    if not 0 <= start < stop <= frames:
        raise ValueError(f"{motion}: invalid {label} [{start}:{stop}] for {frames} frames")
    return start, stop


def _crop_frame_arrays(
    arrays: Mapping[str, np.ndarray], *, frames: int, start: int, stop: int
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == frames:
            array = array[start:stop]
        result[name] = np.array(array, copy=True)
    return result


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


def _build_filter_runtime() -> dict[str, Any]:
    """Load the CPU-only MuJoCo runtime lazily, never during module import."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    import mujoco

    from musclemimic.badminton.scripts.filter_retarget_cache import (
        _compute_qvel,
        _forward_kinematics,
        _make_model,
        _model_names,
        _repo_root,
        _site_ids,
        filter_qpos,
    )

    repo_root = _repo_root()
    model, data = _make_model(repo_root, repo_root)
    return {
        "mujoco": mujoco,
        "model": model,
        "data": data,
        "compute_qvel": _compute_qvel,
        "forward_kinematics": _forward_kinematics,
        "model_names": _model_names,
        "site_ids": _site_ids,
        "filter_qpos": filter_qpos,
    }


def _filter_cache_payload(
    source_path: Path,
    *,
    entry: Mapping[str, Any],
    recipe_digest: str,
    runtime: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Crop/filter qpos and recompute qvel plus every stored FK field."""
    model = runtime["model"]
    data = runtime["data"]
    mujoco = runtime["mujoco"]
    with np.load(source_path, allow_pickle=True) as source:
        source_frames = int(np.asarray(source["qpos"]).shape[0])
        frequency = float(np.asarray(source["frequency"]).reshape(-1)[0])
        site_names = [str(name) for name in source["site_names"]]
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    start, stop = _crop_bounds(
        entry["source_cache_crop"], source_frames, motion=str(entry["motion"]), label="cache crop"
    )
    payload = _crop_frame_arrays(arrays, frames=source_frames, start=start, stop=stop)
    site_ids = runtime["site_ids"](model, site_names)
    qpos = runtime["filter_qpos"](
        np.asarray(payload["qpos"], dtype=np.float64),
        frequency=frequency,
        joint_sigma=float(entry["filter"]["joint_sigma"]),
        max_joint_speed=float(entry["filter"]["max_joint_speed_rad_s"]),
        root_sigma=float(entry["filter"]["root_sigma"]),
        max_root_speed=float(entry["filter"]["max_root_speed_m_s"]),
    )
    qvel = runtime["compute_qvel"](model, qpos, frequency)
    fk = runtime["forward_kinematics"](model, data, qpos, qvel, site_ids)
    payload.update(fk)
    payload["qpos"] = qpos.astype(np.float32)
    payload["qvel"] = qvel.astype(np.float32)
    payload["frequency"] = np.asarray(frequency, dtype=np.float64)
    payload["body_names"] = runtime["model_names"](
        model, mujoco.mjtObj.mjOBJ_BODY, model.nbody
    )
    payload["site_names"] = np.asarray(site_names)
    payload["metadata"] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_cache": entry["source_cache_path"],
        "source_cache_sha256": entry["source_cache_sha256"],
        "source_cache_crop": entry["source_cache_crop"],
        "operation": entry["operation"],
        "filter": dict(entry["filter"]),
        "override_recipe_sha256": recipe_digest,
    }
    payload["njnt"] = np.asarray(model.njnt)
    payload["jnt_type"] = np.asarray(model.jnt_type)
    payload["nbody"] = np.asarray(model.nbody)
    payload["body_rootid"] = np.asarray(model.body_rootid)
    payload["body_weldid"] = np.asarray(model.body_weldid)
    payload["body_mocapid"] = np.asarray(model.body_mocapid)
    payload["body_pos"] = np.asarray(model.body_pos, dtype=np.float32)
    payload["body_quat"] = np.asarray(model.body_quat, dtype=np.float32)
    payload["body_ipos"] = np.asarray(model.body_ipos, dtype=np.float32)
    payload["body_iquat"] = np.asarray(model.body_iquat, dtype=np.float32)
    payload["nsite"] = np.asarray(len(site_ids))
    payload["site_bodyid"] = np.asarray(model.site_bodyid[site_ids])
    payload["site_pos"] = np.asarray(model.site_pos[site_ids], dtype=np.float32)
    payload["site_quat"] = np.asarray(model.site_quat[site_ids], dtype=np.float32)
    if "split_points" in payload:
        split_points = np.asarray(payload["split_points"])
        if split_points.shape != (2,) or not np.array_equal(
            split_points, np.asarray([0, source_frames], dtype=split_points.dtype)
        ):
            raise ValueError(f"{entry['motion']}: protected cache must contain one trajectory")
        payload["split_points"] = np.asarray([0, len(qpos)], dtype=split_points.dtype)

    # Recompute optional sparse FK views when a cache stores them.
    site_body_ids = np.asarray(model.site_bodyid[site_ids], dtype=np.int32)
    if np.asarray(payload.get("cvel_parent", np.empty(0))).size:
        payload["cvel_parent"] = fk["cvel"][:, site_body_ids]
    if np.asarray(payload.get("xpos_parent", np.empty(0))).size:
        payload["xpos_parent"] = fk["xpos"][:, site_body_ids]
    if np.asarray(payload.get("xquat_parent", np.empty(0))).size:
        payload["xquat_parent"] = fk["xquat"][:, site_body_ids]
    if np.asarray(payload.get("subtree_com_root", np.empty(0))).size:
        positive_roots = np.unique(np.asarray(model.body_rootid)[np.asarray(model.body_rootid) > 0])
        if positive_roots.size != 1:
            raise ValueError(f"{entry['motion']}: cannot identify unique non-world root body")
        payload["subtree_com_root"] = fk["subtree_com"][:, int(positive_roots[0])]
    return payload


def _source_override_payload(
    root: Path,
    *,
    motion: str,
    contract: Mapping[str, Any],
    source_recipe_digest: str,
    override_recipe_digest: str,
) -> tuple[bytes, dict[str, Any], str]:
    protected = _resolve_exact(root, contract["protected_source_path"], label=f"{motion} protected source")
    arrays, source_frames = _load_source_arrays(protected, motion=motion)
    start, stop = _crop_bounds(contract["source_crop"], source_frames, motion=motion, label="source crop")
    output_arrays = _crop_frame_arrays(arrays, frames=source_frames, start=start, stop=stop)
    payload = deterministic_npz_bytes(output_arrays)
    output_hash = hashlib.sha256(payload).hexdigest()
    final_operation: dict[str, Any]
    if contract["operation"] == "deterministic_source_prefix_crop":
        final_operation = {
            "type": "crop",
            "start": start,
            "stop": contract["source_crop"]["stop"],
            "reason": "remove the reviewed invalid source prefix for final release",
        }
    else:
        final_operation = {
            "type": "restore_full_original",
            "start": 0,
            "stop": None,
            "reason": "undo the superseded base-recipe crop and restore the full immutable source",
        }
    sidecar = {
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "motion": motion,
        "identity_operation": contract["operation"],
        "repair": final_operation,
        "source_path": contract["protected_source_path"],
        "source_sha256": contract["protected_source_sha256"],
        "source_frames": source_frames,
        "source_fps": 60.0,
        "source_crop": contract["source_crop"],
        "output_path": contract["output_source_path"],
        "output_sha256": output_hash,
        "output_frames": stop - start,
        "output_fps": 60.0,
        "recipe_sha256": source_recipe_digest,
        "override_recipe_sha256": override_recipe_digest,
        "release_source_override": dict(contract),
        "supersedes_base_recipe_repair": contract["supersedes_base_recipe_repair"],
    }
    return payload, sidecar, sha256_file(protected)


def materialize_overrides(
    dataset_root: str | Path,
    recipe_path: str | Path | None = None,
    *,
    source_recipe_path: str | Path | None = None,
    dry_run: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    recipe_path = Path(recipe_path or default_recipe_path()).expanduser().resolve()
    source_recipe_path = Path(source_recipe_path or default_source_recipe_path()).expanduser().resolve()
    recipe = load_override_recipe(recipe_path)
    source_recipe, source_recipe_digest = _load_and_bind_source_recipe(source_recipe_path, recipe)
    del source_recipe
    override_recipe_digest = recipe_sha256(recipe)
    plans: list[dict[str, Any]] = []
    protected_hashes_before: dict[str, str] = {}
    existing_outputs: list[Path] = []

    # Complete all safety/hash checks before constructing or replacing any output.
    for entry in recipe["overrides"]:
        motion = str(entry["motion"])
        source = _resolve_exact(root, entry["source_cache_path"], label=f"{motion} protected cache")
        output = _resolve_exact(root, entry["output_cache_path"], label=f"{motion} output cache")
        output_analysis = _resolve_exact(root, entry["output_analysis_path"], label=f"{motion} output analysis")
        sidecar = _resolve_exact(root, entry["provenance_sidecar_path"], label=f"{motion} cache provenance")
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"{motion}: protected fallback cache missing/symlinked: {source}")
        source_analysis = root / f"muscle_trajectory/raw/{motion}_analysis.npz"
        if source_analysis.exists():
            raise ValueError(f"{motion}: protected fallback unexpectedly has an unpinned analysis file")
        source_hash = sha256_file(source)
        if source_hash != entry["source_cache_sha256"]:
            raise ValueError(
                f"{motion}: protected fallback cache hash mismatch: "
                f"expected {entry['source_cache_sha256']}, got {source_hash}"
            )
        protected_hashes_before[entry["source_cache_path"]] = source_hash
        source_metadata = _validate_cache(source)
        crop_start, crop_stop = _crop_bounds(
            entry["source_cache_crop"], source_metadata["frames"], motion=motion, label="cache crop"
        )
        source_output = None
        source_sidecar = None
        source_override = entry["source_output_override"]
        if source_override is not None:
            protected_source = _resolve_exact(
                root, source_override["protected_source_path"], label=f"{motion} protected source"
            )
            if not protected_source.is_file() or protected_source.is_symlink():
                raise FileNotFoundError(f"{motion}: protected source missing/symlinked")
            protected_source_hash = sha256_file(protected_source)
            if protected_source_hash != source_override["protected_source_sha256"]:
                raise ValueError(f"{motion}: protected source hash mismatch")
            protected_hashes_before[source_override["protected_source_path"]] = protected_source_hash
            source_output = _resolve_exact(
                root, source_override["output_source_path"], label=f"{motion} output source"
            )
            source_sidecar = _resolve_exact(
                root,
                source_override["source_provenance_sidecar_path"],
                label=f"{motion} source provenance",
            )
        for path in (output, output_analysis, sidecar, source_output, source_sidecar):
            if path is not None and path.exists():
                existing_outputs.append(path)
        plans.append(
            {
                "entry": entry,
                "source": source,
                "source_metadata": source_metadata,
                "crop_start": crop_start,
                "crop_stop": crop_stop,
                "output": output,
                "output_analysis": output_analysis,
                "sidecar": sidecar,
                "source_output": source_output,
                "source_sidecar": source_sidecar,
            }
        )
    if existing_outputs and not dry_run and not replace:
        raise FileExistsError(
            "raw_smooth_v1 override artifacts already exist; pass --replace: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    report: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "dataset_root": str(root),
        "override_recipe_path": str(recipe_path),
        "override_recipe_sha256": override_recipe_digest,
        "source_recipe_path": str(source_recipe_path),
        "source_recipe_sha256": source_recipe_digest,
        "override_count": len(plans),
        "source_override_count": len(_PINNED_SOURCE_OVERRIDES),
        "dry_run": bool(dry_run),
        "motions": [
            {
                "motion": plan["entry"]["motion"],
                "source_cache_path": plan["entry"]["source_cache_path"],
                "source_cache_sha256": plan["entry"]["source_cache_sha256"],
                "source_cache_frames": plan["source_metadata"]["frames"],
                "source_cache_crop": plan["entry"]["source_cache_crop"],
                "output_cache_path": plan["entry"]["output_cache_path"],
                "output_cache_frames": plan["crop_stop"] - plan["crop_start"],
                "source_output_override": plan["entry"]["source_output_override"],
            }
            for plan in plans
        ],
    }
    if dry_run:
        return report

    # Build every payload first. A generation failure therefore cannot leave a
    # half-regenerated release.
    source_builds: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for plan in plans:
        entry = plan["entry"]
        source_override = entry["source_output_override"]
        if source_override is None:
            continue
        payload, sidecar, protected_hash = _source_override_payload(
            root,
            motion=entry["motion"],
            contract=source_override,
            source_recipe_digest=source_recipe_digest,
            override_recipe_digest=override_recipe_digest,
        )
        if protected_hash != source_override["protected_source_sha256"]:
            raise RuntimeError(f"{entry['motion']}: protected source changed before encoding")
        source_builds[entry["motion"]] = payload, sidecar

    runtime = _build_filter_runtime()
    cache_builds: dict[str, tuple[bytes, dict[str, Any], bool]] = {}
    for plan in plans:
        entry = plan["entry"]
        filtered_payload = _filter_cache_payload(
            plan["source"], entry=entry, recipe_digest=override_recipe_digest, runtime=runtime
        )
        payload = deterministic_npz_bytes(filtered_payload)
        output_hash = hashlib.sha256(payload).hexdigest()
        stale_analysis_present = plan["output_analysis"].exists()
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "motion": entry["motion"],
            "operation": entry["operation"],
            "reason": entry["reason"],
            "source_cache_path": entry["source_cache_path"],
            "source_cache_sha256": entry["source_cache_sha256"],
            "source_cache_metadata": plan["source_metadata"],
            "source_cache_crop": entry["source_cache_crop"],
            "source_analysis_path": None,
            "output_cache_path": entry["output_cache_path"],
            "output_cache_sha256": output_hash,
            "output_cache_metadata": {
                "frames": plan["crop_stop"] - plan["crop_start"],
                "frequency": 100.0,
                "qpos_dim": 89,
                "qvel_dim": 88,
            },
            "output_analysis_path": entry["output_analysis_path"],
            "output_analysis_policy": "removed_and_must_be_absent",
            "provenance_sidecar_path": entry["provenance_sidecar_path"],
            "override_recipe_sha256": override_recipe_digest,
            "filter": dict(entry["filter"]),
            "source_output_override": entry["source_output_override"],
            "protected_source_hash_unchanged": True,
        }
        cache_builds[entry["motion"]] = payload, provenance, stale_analysis_present

    motion_reports: list[dict[str, Any]] = []
    for plan in plans:
        entry = plan["entry"]
        motion = entry["motion"]
        if motion in source_builds:
            source_payload, source_sidecar = source_builds[motion]
            _atomic_write(plan["source_output"], source_payload)
            _atomic_write(plan["source_sidecar"], _canonical_json_bytes(source_sidecar))
            if sha256_file(plan["source_output"]) != source_sidecar["output_sha256"]:
                raise RuntimeError(f"{motion}: materialized source hash mismatch")
        cache_payload, provenance, stale_analysis_present = cache_builds[motion]
        _atomic_write(plan["output"], cache_payload)
        if sha256_file(plan["output"]) != provenance["output_cache_sha256"]:
            raise RuntimeError(f"{motion}: materialized cache hash mismatch")
        if plan["output_analysis"].exists():
            if plan["output_analysis"].is_symlink():
                raise ValueError(f"{motion}: refusing to unlink symlinked stale analysis")
            plan["output_analysis"].unlink()
        output_metadata = _validate_cache(plan["output"])
        if output_metadata != provenance["output_cache_metadata"]:
            raise RuntimeError(f"{motion}: materialized cache metadata mismatch")
        _atomic_write(plan["sidecar"], _canonical_json_bytes(provenance))
        motion_report = dict(provenance)
        motion_report["removed_stale_analysis"] = stale_analysis_present
        if motion in source_builds:
            motion_report["source_output_sha256"] = source_builds[motion][1]["output_sha256"]
            motion_report["source_output_frames"] = source_builds[motion][1]["output_frames"]
        motion_reports.append(motion_report)

    protected_after = {
        relative: sha256_file(_resolve_exact(root, relative, label="protected input recheck"))
        for relative in protected_hashes_before
    }
    if protected_after != protected_hashes_before:
        raise RuntimeError("a protected raw/source input changed during materialization")
    report.update(
        {
            "dry_run": False,
            "motions": motion_reports,
            "protected_input_hashes_before": protected_hashes_before,
            "protected_input_hashes_after": protected_after,
            "protected_input_hashes_unchanged": True,
        }
    )
    return report


def materialize_override(
    dataset_root: str | Path,
    recipe_path: str | Path | None = None,
    *,
    source_recipe_path: str | Path | None = None,
    dry_run: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    """Compatibility alias; the final recipe always materializes all six."""
    return materialize_overrides(
        dataset_root,
        recipe_path,
        source_recipe_path=source_recipe_path,
        dry_run=dry_run,
        replace=replace,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        default=Path("datasets/forehandClear_standard"),
    )
    parser.add_argument("--recipe", type=Path, default=default_recipe_path())
    parser.add_argument(
        "--source-recipe", "--source_recipe", dest="source_recipe", type=Path,
        default=default_source_recipe_path(),
    )
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    report = materialize_overrides(
        args.dataset_root,
        args.recipe,
        source_recipe_path=args.source_recipe,
        dry_run=args.dry_run,
        replace=args.replace,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
