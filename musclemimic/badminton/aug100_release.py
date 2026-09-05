"""Content-bound release and leakage gates for Forehand Clear Aug100.

The Aug100 namespace contains the 27 reviewed ``raw_smooth_v1`` motions and
73 deterministic world-z yaw variants.  A file-level random split would leak
near duplicates, so this module derives the split from the augmentation
metadata and keeps every source motion and all of its yaw variants together.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from musclemimic.badminton.action_registry import resolve
from musclemimic.badminton.action_release import validate_action_release
from musclemimic.badminton.data_qc import inspect_canonical_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_ID = "forehandClear_standard"
ACTION_SLUG = "forehand_clear"
SOURCE_VARIANT = "raw_smooth_v1"
SOURCE_NAMESPACE = "muscle_trajectory/raw_smooth_v1"
CACHE_VARIANT = "raw_smooth_v1_aug100"
CACHE_NAMESPACE = f"muscle_trajectory/{CACHE_VARIANT}"
SOURCE_MODE = "verified_augmented_cache"
DATASET_MANIFEST = Path(
    "datasets/forehandClear_standard/manifests/raw_smooth_v1_aug100_list.txt"
)
TRANSFER_MANIFEST = Path("private_asset_manifests/aug100_20260813.json")
EXPECTED_TRANSFER_MANIFEST_FINGERPRINT = (
    "b9bf315bb70b3ffa3afbe7cd8769883181acbcfbda3b6d8d8a6f6fd320aa7c62"
)
EXPECTED_TRANSFER_FILE_COUNT = 309
EXPECTED_TRANSFER_TOTAL_BYTES = 535_414_566
EXPECTED_MOTION_COUNT = 100
EXPECTED_ORIGINAL_COUNT = 27
EXPECTED_AUGMENTED_COUNT = 73
EXPECTED_TRAIN_COUNT = 80
EXPECTED_VALIDATION_COUNT = 20
GROUPED_SUBSET_40_10_CONTRACT = "grouped_subset_40_train_10_validation_v1"
EXPECTED_SUBSET_TRAIN_COUNT = 40
EXPECTED_SUBSET_VALIDATION_COUNT = 10
EXPECTED_AUGMENTATION_SEED = 20_260_812
EXPECTED_AUGMENTATION_TYPE = "yaw_rotation"
EXPECTED_AUGMENTATION_SCRIPT = (
    "musclemimic/badminton/scripts/augment_yaw_rotate_dataset.py"
)
EXPECTED_DERIVED_FIELDS = "recomputed via mj_kinematics/mj_comPos/mj_comVel"
EXPECTED_FPS = 100.0
RELEASE_SCHEMA_VERSION = "musclemimic_action_release_validation_v1"

_AUGMENTED_NAME = re.compile(r"^(?P<source>.+)__aug(?P<index>\d+)_yaw(?P<yaw>\d{3})$")
_DYNAMIC_FIELDS = frozenset(
    {
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "subtree_com",
        "site_xpos",
        "site_xmat",
        "metadata",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "subtree_com",
        "site_xpos",
        "site_xmat",
        "split_points",
        "joint_names",
        "frequency",
        "body_names",
        "site_names",
        "metadata",
        "jnt_type",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any, *, ensure_ascii: bool = True) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _safe_repo_path(value: str | Path) -> Path:
    raw = str(value).strip()
    posix = PurePosixPath(raw)
    if not raw or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe repository-relative path: {raw!r}")
    path = (REPO_ROOT / Path(*posix.parts)).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {raw!r}") from exc
    return path


def _metadata(value: np.ndarray) -> Mapping[str, Any] | None:
    if value.shape != ():
        return None
    item = value.item()
    return item if isinstance(item, Mapping) else None


def _manifest_rows() -> tuple[tuple[str, Path], ...]:
    manifest_path = REPO_ROOT / DATASET_MANIFEST
    rows: list[tuple[str, Path]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        path = _safe_repo_path("datasets/" + value.removesuffix(".npz") + ".npz")
        expected_parent = (
            REPO_ROOT / "datasets" / ACTION_ID / CACHE_NAMESPACE
        ).resolve()
        if path.parent != expected_parent:
            raise ValueError(f"Aug100 manifest row points outside cache namespace: {value!r}")
        rows.append((path.stem, path))
    names = [name for name, _ in rows]
    if len(rows) != EXPECTED_MOTION_COUNT:
        raise ValueError(
            f"Aug100 manifest must contain {EXPECTED_MOTION_COUNT} motions; got {len(rows)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Aug100 manifest contains duplicate motion names")
    return tuple(rows)


def _source_group(
    motion: str,
    path: Path,
) -> tuple[str, Mapping[str, Any] | None]:
    match = _AUGMENTED_NAME.fullmatch(motion)
    if match is None:
        return motion, None
    with np.load(path, allow_pickle=True) as cache:
        metadata = _metadata(cache["metadata"]) if "metadata" in cache.files else None
    augmentation = metadata.get("augmentation") if metadata is not None else None
    if not isinstance(augmentation, Mapping):
        raise ValueError(f"{motion}: augmented cache has no metadata.augmentation")
    raw_source = str(augmentation.get("source_cache", "")).strip()
    source_path = PurePosixPath(raw_source)
    expected_parent = PurePosixPath(ACTION_ID) / SOURCE_NAMESPACE
    if source_path.parent != expected_parent or source_path.suffix != ".npz":
        raise ValueError(f"{motion}: unsafe or foreign metadata source_cache {raw_source!r}")
    source_group = source_path.stem
    if source_group != match.group("source"):
        raise ValueError(f"{motion}: filename and metadata source group differ")
    return source_group, augmentation


def expected_forehand_clear_aug100_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the ordered 80/20 split, grouped by metadata source_cache."""

    spec = resolve(ACTION_ID)
    train_groups = set(spec.train_motions)
    validation_groups = set(spec.val_motions)
    if train_groups & validation_groups:
        raise ValueError("registered Forehand Clear source split overlaps")

    train: list[str] = []
    validation: list[str] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for motion, path in _manifest_rows():
        source_group, _ = _source_group(motion, path)
        if source_group in train_groups:
            train.append(motion)
            groups_by_split["train"].add(source_group)
        elif source_group in validation_groups:
            validation.append(motion)
            groups_by_split["validation"].add(source_group)
        else:
            raise ValueError(f"{motion}: source group {source_group!r} is not released")

    overlap = groups_by_split["train"] & groups_by_split["validation"]
    if overlap:
        raise ValueError(f"Aug100 source-group train/validation leakage: {sorted(overlap)}")
    if len(train) != EXPECTED_TRAIN_COUNT or len(validation) != EXPECTED_VALIDATION_COUNT:
        raise ValueError(
            "Aug100 grouped split must be "
            f"{EXPECTED_TRAIN_COUNT}/{EXPECTED_VALIDATION_COUNT}; "
            f"got {len(train)}/{len(validation)}"
        )
    if groups_by_split["train"] != train_groups:
        raise ValueError("Aug100 train source groups differ from the reviewed 22-motion split")
    if groups_by_split["validation"] != validation_groups:
        raise ValueError("Aug100 validation source groups differ from the reviewed 5-motion split")
    return tuple(train), tuple(validation)


def motion_names_from_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Parse runtime dataset paths while rejecting foreign namespaces."""

    prefix = PurePosixPath(ACTION_ID) / CACHE_NAMESPACE
    motions: list[str] = []
    for raw in paths:
        path = PurePosixPath(str(raw).strip().removesuffix(".npz"))
        if path.parent != prefix or not path.name:
            raise ValueError(f"runtime Aug100 path points outside {prefix}: {raw!r}")
        motions.append(path.name)
    if len(motions) != len(set(motions)):
        raise ValueError("runtime Aug100 dataset paths contain duplicates")
    return tuple(motions)


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def _rotate_vectors(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, values.astype(np.float64))


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.array(
        [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)],
        dtype=np.float64,
    )


def _quat_left_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    direct = np.linalg.norm(actual64 - expected64, axis=-1)
    flipped = np.linalg.norm(actual64 + expected64, axis=-1)
    return float(np.max(np.minimum(direct, flipped)))


def _rotation_audit(
    motion: str,
    source_path: Path,
    augmented_path: Path,
    augmentation: Mapping[str, Any],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    try:
        yaw_deg = float(augmentation.get("yaw_deg"))
    except (TypeError, ValueError):
        return {}, [f"{motion}: augmentation yaw_deg is not finite"]
    if not math.isfinite(yaw_deg):
        return {}, [f"{motion}: augmentation yaw_deg is not finite"]
    match = _AUGMENTED_NAME.fullmatch(motion)
    if match is None or round(yaw_deg) % 360 != int(match.group("yaw")):
        errors.append(f"{motion}: rounded metadata yaw differs from filename")
    try:
        pivot = np.asarray(augmentation.get("pivot_xy"), dtype=np.float64)
    except (TypeError, ValueError):
        pivot = np.empty((0,), dtype=np.float64)
    if pivot.shape != (2,) or not np.all(np.isfinite(pivot)):
        errors.append(f"{motion}: pivot_xy must contain two finite values")
        pivot = np.zeros(2, dtype=np.float64)

    yaw_rad = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    yaw_quat = _yaw_quaternion(yaw_rad)
    metrics: dict[str, float] = {}
    with np.load(source_path, allow_pickle=True) as source, np.load(
        augmented_path, allow_pickle=True
    ) as augmented:
        missing = sorted(_REQUIRED_FIELDS - set(augmented.files))
        if missing:
            return metrics, [f"{motion}: missing required arrays {missing}"]
        if set(source.files) != set(augmented.files):
            errors.append(f"{motion}: augmented and source NPZ schemas differ")
        for field in sorted(set(source.files) - _DYNAMIC_FIELDS):
            if field not in augmented.files or not np.array_equal(source[field], augmented[field]):
                errors.append(f"{motion}: static field {field!r} differs from source")

        frame_count = int(augmented["qpos"].shape[0])
        shape_requirements = {
            "qpos": (frame_count, 89),
            "qvel": (frame_count, 88),
            "xpos": (frame_count, 102, 3),
            "xquat": (frame_count, 102, 4),
            "cvel": (frame_count, 102, 6),
            "subtree_com": (frame_count, 102, 3),
            "site_xpos": (frame_count, 17, 3),
            "site_xmat": (frame_count, 17, 9),
        }
        for field, shape in shape_requirements.items():
            if augmented[field].shape != shape:
                errors.append(
                    f"{motion}: {field} shape {augmented[field].shape} differs from {shape}"
                )
            elif not np.all(np.isfinite(augmented[field])):
                errors.append(f"{motion}: {field} contains non-finite values")
        if float(augmented["frequency"]) != EXPECTED_FPS:
            errors.append(f"{motion}: cache frequency is not {EXPECTED_FPS:g} Hz")

        source_qpos = source["qpos"].astype(np.float64)
        effective_pivot = pivot
        expected_root_pos = source_qpos[:, :3].copy()
        expected_root_pos[:, :2] = (
            _rotate_vectors(
                np.column_stack(
                    (
                        source_qpos[:, 0] - effective_pivot[0],
                        source_qpos[:, 1] - effective_pivot[1],
                        np.zeros(frame_count),
                    )
                ),
                rotation,
            )[:, :2]
            + effective_pivot
        )
        metrics["root_position_max_error"] = _max_abs(
            augmented["qpos"][:, :3], expected_root_pos
        )
        expected_root_quat = _quat_left_multiply(yaw_quat, source_qpos[:, 3:7])
        metrics["root_quaternion_max_error"] = _quat_error(
            augmented["qpos"][:, 3:7], expected_root_quat
        )
        metrics["nonroot_qpos_max_error"] = _max_abs(
            augmented["qpos"][:, 7:], source["qpos"][:, 7:]
        )

        expected_qvel_linear = _rotate_vectors(source["qvel"][:, :3], rotation)
        metrics["root_linear_velocity_max_error"] = _max_abs(
            augmented["qvel"][:, :3], expected_qvel_linear
        )
        metrics["remaining_qvel_max_error"] = _max_abs(
            augmented["qvel"][:, 3:], source["qvel"][:, 3:]
        )

        for field in ("xpos", "subtree_com", "site_xpos"):
            expected = source[field].astype(np.float64).copy()
            shifted = expected.copy()
            shifted[..., 0] -= effective_pivot[0]
            shifted[..., 1] -= effective_pivot[1]
            expected = _rotate_vectors(shifted, rotation)
            expected[..., 0] += effective_pivot[0]
            expected[..., 1] += effective_pivot[1]
            # MuJoCo's world body is fixed at the origin; it is not a child of
            # the rotated free root and therefore must remain unchanged.
            if field == "xpos":
                body_names = [str(value) for value in augmented["body_names"].tolist()]
                if "world" in body_names:
                    world_index = body_names.index("world")
                    expected[:, world_index] = source[field][:, world_index]
            metrics[f"{field}_max_error"] = _max_abs(augmented[field], expected)

        body_names = [str(value) for value in augmented["body_names"].tolist()]
        expected_xquat = _quat_left_multiply(yaw_quat, source["xquat"])
        if "world" in body_names:
            world_index = body_names.index("world")
            expected_xquat[:, world_index] = source["xquat"][:, world_index]
        metrics["xquat_max_error"] = _quat_error(augmented["xquat"], expected_xquat)

        source_cvel = source["cvel"].astype(np.float64)
        expected_cvel = np.concatenate(
            (
                _rotate_vectors(source_cvel[..., :3], rotation),
                _rotate_vectors(source_cvel[..., 3:], rotation),
            ),
            axis=-1,
        )
        metrics["cvel_max_error"] = _max_abs(augmented["cvel"], expected_cvel)

        source_site_xmat = source["site_xmat"].astype(np.float64).reshape(
            frame_count, -1, 3, 3
        )
        expected_site_xmat = np.einsum("ij,...jk->...ik", rotation, source_site_xmat)
        metrics["site_xmat_max_error"] = _max_abs(
            augmented["site_xmat"].reshape(frame_count, -1, 3, 3),
            expected_site_xmat,
        )

    tolerances = {
        "root_position_max_error": 2e-5,
        "root_quaternion_max_error": 2e-5,
        "nonroot_qpos_max_error": 2e-6,
        "root_linear_velocity_max_error": 2e-5,
        "remaining_qvel_max_error": 2e-6,
        "xpos_max_error": 2e-5,
        "subtree_com_max_error": 2e-5,
        "site_xpos_max_error": 2e-5,
        "xquat_max_error": 2e-5,
        "cvel_max_error": 5e-5,
        "site_xmat_max_error": 2e-5,
    }
    for field, limit in tolerances.items():
        value = metrics.get(field, math.inf)
        if not math.isfinite(value) or value > limit:
            errors.append(f"{motion}: {field}={value:.6g} exceeds {limit:.6g}")
    return metrics, errors


def _transfer_manifest_errors() -> tuple[dict[str, Any], list[str]]:
    path = REPO_ROOT / TRANSFER_MANIFEST
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Aug100 transfer manifest is unreadable: {exc}"]
    if not isinstance(payload, Mapping):
        return {}, ["Aug100 transfer manifest must be a JSON object"]
    unsigned = dict(payload)
    supplied = str(unsigned.pop("manifest_fingerprint", ""))
    computed = _fingerprint(unsigned, ensure_ascii=False)
    if supplied != computed or supplied != EXPECTED_TRANSFER_MANIFEST_FINGERPRINT:
        errors.append("Aug100 transfer manifest fingerprint differs from the synced release")
    if payload.get("schema_version") != "musclemimic_aug100_transfer_manifest_v1":
        errors.append("Aug100 transfer manifest schema differs from the synced release")
    if int(payload.get("file_count", -1)) != EXPECTED_TRANSFER_FILE_COUNT:
        errors.append("Aug100 transfer manifest file_count differs from the synced release")
    if int(payload.get("total_bytes", -1)) != EXPECTED_TRANSFER_TOTAL_BYTES:
        errors.append("Aug100 transfer manifest total_bytes differs from the synced release")

    files = payload.get("files")
    if not isinstance(files, list):
        return dict(payload), [*errors, "Aug100 transfer manifest files must be a list"]
    forehand_rows = [
        row
        for row in files
        if isinstance(row, Mapping)
        and str(row.get("path", "")).startswith(f"datasets/{ACTION_ID}/")
    ]
    if len(forehand_rows) != 103:
        errors.append(f"Forehand Clear transfer inventory must contain 103 files; got {len(forehand_rows)}")
    for row in forehand_rows:
        raw_path = str(row.get("path", ""))
        try:
            file_path = _safe_repo_path(raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not file_path.is_file() or file_path.is_symlink():
            errors.append(f"transfer-bound file is missing or a symlink: {raw_path}")
            continue
        if file_path.stat().st_size != int(row.get("num_bytes", -1)):
            errors.append(f"transfer-bound size changed: {raw_path}")
        if _sha256(file_path) != str(row.get("sha256", "")):
            errors.append(f"transfer-bound SHA-256 changed: {raw_path}")
    return dict(payload), errors


def _validate_declared_split(
    train_motions: Sequence[str] | None,
    validation_motions: Sequence[str] | None,
    *,
    split_contract: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    expected_train, expected_validation = expected_forehand_clear_aug100_split()
    declared_train = tuple(train_motions) if train_motions is not None else expected_train
    declared_validation = (
        tuple(validation_motions) if validation_motions is not None else expected_validation
    )
    errors: list[str] = []
    if split_contract is None:
        if declared_train != expected_train:
            errors.append("configured Aug100 training split differs from the metadata-grouped release")
        if declared_validation != expected_validation:
            errors.append("configured Aug100 validation split differs from the metadata-grouped release")
        return expected_train, expected_validation, errors

    if split_contract != GROUPED_SUBSET_40_10_CONTRACT:
        errors.append(f"unsupported Aug100 split contract: {split_contract!r}")
        return declared_train, declared_validation, errors
    if train_motions is None or validation_motions is None:
        errors.append("Aug100 grouped subset contract requires explicit train and validation lists")
        return declared_train, declared_validation, errors
    if len(declared_train) != EXPECTED_SUBSET_TRAIN_COUNT:
        errors.append(
            "Aug100 grouped subset training split must contain "
            f"{EXPECTED_SUBSET_TRAIN_COUNT} motions; got {len(declared_train)}"
        )
    if len(declared_validation) != EXPECTED_SUBSET_VALIDATION_COUNT:
        errors.append(
            "Aug100 grouped subset validation split must contain "
            f"{EXPECTED_SUBSET_VALIDATION_COUNT} motions; got {len(declared_validation)}"
        )
    if len(declared_train) != len(set(declared_train)):
        errors.append("Aug100 grouped subset training split contains duplicates")
    if len(declared_validation) != len(set(declared_validation)):
        errors.append("Aug100 grouped subset validation split contains duplicates")

    train_set = set(declared_train)
    validation_set = set(declared_validation)
    overlap = train_set & validation_set
    if overlap:
        errors.append(f"Aug100 grouped subset train/validation overlap: {sorted(overlap)}")

    groups: dict[str, set[str]] = defaultdict(set)
    manifest_motions: set[str] = set()
    for motion, path in _manifest_rows():
        source_group, _ = _source_group(motion, path)
        groups[source_group].add(motion)
        manifest_motions.add(motion)
    foreign = (train_set | validation_set) - manifest_motions
    if foreign:
        errors.append(f"Aug100 grouped subset contains foreign motions: {sorted(foreign)}")
    for source_group, group_motions in groups.items():
        selected_train = group_motions & train_set
        selected_validation = group_motions & validation_set
        if selected_train and selected_validation:
            errors.append(
                f"Aug100 source group {source_group!r} leaks across train and validation"
            )
        selected = selected_train or selected_validation
        if selected and selected != group_motions:
            errors.append(
                f"Aug100 source group {source_group!r} is only partially selected"
            )
    return declared_train, declared_validation, errors


def validate_forehand_clear_aug100_release(
    train_motions: Sequence[str] | None = None,
    validation_motions: Sequence[str] | None = None,
    *,
    split_contract: str | None = None,
) -> dict[str, Any]:
    """Validate all Aug100 bytes, metadata, rotations and grouped split."""

    expected_train, expected_validation, errors = _validate_declared_split(
        train_motions,
        validation_motions,
        split_contract=split_contract,
    )
    transfer, transfer_errors = _transfer_manifest_errors()
    errors.extend(transfer_errors)
    base_release = validate_action_release(ACTION_ID)
    if base_release.get("passed") is not True:
        errors.append("reviewed raw_smooth_v1 base release no longer passes")

    rows = _manifest_rows()
    inventory: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    augmented_count = 0
    original_count = 0
    cache_root = (REPO_ROOT / "datasets" / ACTION_ID / CACHE_NAMESPACE).resolve()
    directory_npz = {path.stem for path in cache_root.glob("*.npz") if path.is_file()}
    manifest_names = {motion for motion, _ in rows}
    if directory_npz != manifest_names:
        errors.append("Aug100 cache directory and 100-line dataset manifest differ")

    for motion, cache_path in rows:
        if not cache_path.is_file() or cache_path.is_symlink():
            errors.append(f"{motion}: cache file is missing or a symlink")
            continue
        try:
            source_group, augmentation = _source_group(motion, cache_path)
        except (OSError, KeyError, ValueError) as exc:
            errors.append(str(exc))
            continue
        group_counts[source_group] += 1
        source_path = (
            REPO_ROOT / "datasets" / ACTION_ID / SOURCE_NAMESPACE / f"{source_group}.npz"
        ).resolve()
        split = (
            "train"
            if motion in expected_train
            else ("validation" if motion in expected_validation else "unused")
        )
        row: dict[str, Any] = {
            "motion": motion,
            "split": split,
            "source_group": source_group,
            "source_cache_path": source_path.relative_to(REPO_ROOT).as_posix(),
            "source_cache_sha256": _sha256(source_path) if source_path.is_file() else None,
            "cache_path": cache_path.relative_to(REPO_ROOT).as_posix(),
            "cache_sha256": _sha256(cache_path),
            "is_augmented": augmentation is not None,
        }
        if not source_path.is_file() or source_path.is_symlink():
            errors.append(f"{motion}: source cache is missing or a symlink")
        elif augmentation is None:
            original_count += 1
            if _sha256(source_path) != row["cache_sha256"]:
                errors.append(f"{motion}: original copy differs from reviewed raw_smooth_v1 cache")
        else:
            augmented_count += 1
            expected_metadata = {
                "type": EXPECTED_AUGMENTATION_TYPE,
                "seed": EXPECTED_AUGMENTATION_SEED,
                "script": EXPECTED_AUGMENTATION_SCRIPT,
                "derived_fields": EXPECTED_DERIVED_FIELDS,
            }
            for field, expected in expected_metadata.items():
                if augmentation.get(field) != expected:
                    errors.append(f"{motion}: augmentation metadata {field} differs from release")
            if source_path.is_file():
                metrics, rotation_errors = _rotation_audit(
                    motion, source_path, cache_path, augmentation
                )
                errors.extend(rotation_errors)
                row["rotation_qc"] = metrics
            row["yaw_deg"] = augmentation.get("yaw_deg")
            row["augmentation_seed"] = augmentation.get("seed")
        inventory.append(row)

    if original_count != EXPECTED_ORIGINAL_COUNT or augmented_count != EXPECTED_AUGMENTED_COUNT:
        errors.append(
            "Aug100 original/augmented count differs from release: "
            f"{original_count}/{augmented_count}"
        )
    expected_groups = set(resolve(ACTION_ID).all_motions)
    if set(group_counts) != expected_groups:
        errors.append("Aug100 source groups differ from the reviewed 27-motion base release")

    transfer_path = REPO_ROOT / TRANSFER_MANIFEST
    train_source_groups = list(
        dict.fromkeys(row["source_group"] for row in inventory if row["split"] == "train")
    )
    validation_source_groups = list(
        dict.fromkeys(
            row["source_group"] for row in inventory if row["split"] == "validation"
        )
    )
    payload: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "action": ACTION_SLUG,
        "action_id": ACTION_ID,
        "data_variant": CACHE_VARIANT,
        "release_evidence_path": TRANSFER_MANIFEST.as_posix(),
        "release_evidence_sha256": _sha256(transfer_path) if transfer_path.is_file() else None,
        "dataset_manifest_path": DATASET_MANIFEST.as_posix(),
        "dataset_manifest_sha256": _sha256(REPO_ROOT / DATASET_MANIFEST),
        "transfer_manifest_fingerprint": transfer.get("manifest_fingerprint"),
        "base_release_binding_sha256": base_release.get("release_binding_sha256"),
        "review_evidence_kind": (
            "reviewed_source_release_plus_deterministic_world_z_yaw_transform_qc"
        ),
        "formal_release_manifest": True,
        "augmentation": {
            "type": EXPECTED_AUGMENTATION_TYPE,
            "seed": EXPECTED_AUGMENTATION_SEED,
            "script": EXPECTED_AUGMENTATION_SCRIPT,
            "derived_fields": EXPECTED_DERIVED_FIELDS,
            "source_group_field": "metadata.augmentation.source_cache",
        },
        "split_contract": split_contract or "reviewed_grouped_80_train_20_validation_v1",
        "train_motions": list(expected_train),
        "validation_motions": list(expected_validation),
        "train_source_groups": train_source_groups,
        "validation_source_groups": validation_source_groups,
        "source_group_counts": dict(sorted(group_counts.items())),
        "file_inventory": inventory,
        "errors": errors,
        "passed": not errors,
    }
    payload["release_binding_sha256"] = _fingerprint(payload)
    return payload


def inspect_forehand_clear_aug100_dataset(
    train_motions: Sequence[str] | None = None,
    validation_motions: Sequence[str] | None = None,
    *,
    release_report: Mapping[str, Any] | None = None,
    split_contract: str | None = None,
) -> dict[str, Any]:
    """Return the warning-free numeric contract inherited from reviewed sources."""

    report = (
        dict(release_report)
        if release_report is not None
        else validate_forehand_clear_aug100_release(
            train_motions,
            validation_motions,
            split_contract=split_contract,
        )
    )
    expected_train, expected_validation, split_errors = _validate_declared_split(
        train_motions,
        validation_motions,
        split_contract=split_contract,
    )
    base_qc = inspect_canonical_dataset(
        resolve(ACTION_ID).dataset_root,
        source_variant=resolve(ACTION_ID).source_namespace,
        cache_variant=resolve(ACTION_ID).cache_variant,
        action=ACTION_SLUG,
    )
    hard_errors = [str(value) for value in split_errors]
    if report.get("passed") is not True:
        hard_errors.extend(str(value) for value in report.get("errors", ()))
    if base_qc.get("clean_passed") is not True:
        hard_errors.append("reviewed raw_smooth_v1 base numeric QC is no longer a clean pass")
    release_inventory = report.get("file_inventory", ())
    frames = []
    for row in release_inventory if isinstance(release_inventory, list) else ():
        if not isinstance(row, Mapping):
            continue
        path = _safe_repo_path(str(row.get("cache_path", "")))
        with np.load(path, allow_pickle=False) as cache:
            frames.append(int(cache["qpos"].shape[0]))
    result = {
        "schema_version": "forehand_clear_aug100_data_qc_v1",
        "action": ACTION_ID,
        "action_slug": ACTION_SLUG,
        "source_variant": SOURCE_VARIANT,
        "cache_variant": CACHE_VARIANT,
        "resolved_source_dir": str(
            (REPO_ROOT / "datasets" / ACTION_ID / SOURCE_NAMESPACE).resolve()
        ),
        "resolved_cache_dir": str(
            (REPO_ROOT / "datasets" / ACTION_ID / CACHE_NAMESPACE).resolve()
        ),
        "expected_motion_count": EXPECTED_MOTION_COUNT,
        "train_motions": list(expected_train),
        "validation_motions": list(expected_validation),
        "split_contract": split_contract or "reviewed_grouped_80_train_20_validation_v1",
        "train_source_groups": list(report.get("train_source_groups", ())),
        "validation_source_groups": list(report.get("validation_source_groups", ())),
        "source_group_overlap": [],
        "original_count": EXPECTED_ORIGINAL_COUNT,
        "augmented_count": EXPECTED_AUGMENTED_COUNT,
        "frequency_hz": EXPECTED_FPS,
        "min_frames": min(frames) if frames else None,
        "max_frames": max(frames) if frames else None,
        "base_numeric_qc_sha256": _fingerprint(base_qc),
        "release_binding_sha256": report.get("release_binding_sha256"),
        "hard_errors": hard_errors,
        "warnings": [],
        "passed": not hard_errors,
        "clean_passed": not hard_errors,
    }
    return result
