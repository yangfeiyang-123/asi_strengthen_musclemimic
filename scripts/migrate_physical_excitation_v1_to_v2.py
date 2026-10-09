#!/usr/bin/env python3
"""Copy a legacy physical dataset into the effective-excitation v2 contract.

The source is never modified.  Migration is permitted only when raw MuJoCo
``teacher_ctrl_physical`` is present and every ordered channel can be resolved
as a scalar muscle actuator in the pinned MyoFullBody asset.  Existing v1
``muscle_excitation`` values are retained as
``legacy_muscle_excitation_ctrlrange_coordinate_v1`` for audit, while the v2
field is recomputed as MuJoCo's effective input ``clip(raw_ctrl, 0, 1)``.

This tool migrates data semantics only.  NMF bases and policy checkpoints must
be refit/retrained; they are intentionally not transformed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import musclemimic_models
import numpy as np

from musclemimic.distill.action_schema import (
    actuator_names_from_metadata,
    actuator_schema_hash,
    ordered_schema_hash,
)
from musclemimic.distill.dataset import (
    REQUIRED_FIELDS,
    PhysicalDistillDataset,
    _infer_metadata,
)
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    physical_ctrl_to_effective_muscle_excitation,
    physical_signal_metadata,
    resolve_muscle_channel_contract,
    validate_activation_valid_mask,
    validate_ordered_ctrlrange,
)
from musclemimic.environments.humanoids.myofullbody import remove_finger_dofs

LEGACY_PHYSICAL_SIGNAL_SCHEMA_VERSION = "physical_muscle_transition_v1"
LEGACY_PHYSICAL_CAPTURE_SCHEMA_VERSION = "physical_capture_spec_v1"
LEGACY_EXCITATION_TRANSFORM = "ordered_ctrlrange_affine_to_unit_interval_v1"
LEGACY_EXCITATION_FIELD = "legacy_muscle_excitation_ctrlrange_coordinate_v1"
MIGRATION_SCHEMA_VERSION = "physical_excitation_v1_to_v2_migration_v1"
SOURCE_MODEL_PROVENANCE_SCHEMA_VERSION = "myofullbody_source_model_provenance_v1"
PINNED_MODEL_PACKAGE = "musclemimic-models"
PINNED_MODEL_PACKAGE_VERSION = "1.0.5"
MODEL_ASSET_NAME = "myofullbody"
LEGACY_ROUNDOFF_ATOL = 1e-6
SUPPORTED_MODEL_WIDTHS = frozenset({354, 416})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _require_pinned_model_package() -> str:
    version = importlib.metadata.version(PINNED_MODEL_PACKAGE)
    if version != PINNED_MODEL_PACKAGE_VERSION:
        raise ValueError(
            "physical v1-to-v2 migration is bound to "
            f"{PINNED_MODEL_PACKAGE}=={PINNED_MODEL_PACKAGE_VERSION}; "
            f"installed={version!r}"
        )
    return version


def _load_pinned_spec(xml_path: Path, model_nu: int) -> mujoco.MjSpec:
    if int(model_nu) not in SUPPORTED_MODEL_WIDTHS:
        raise ValueError(
            f"automatic migration supports only exact MyoFullBody widths 354 or 416, got model_nu={int(model_nu)}"
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if int(model_nu) == 354:
        remove_finger_dofs(spec)
    return spec


def _build_pinned_models(
    model_nu: int,
) -> tuple[mujoco.MjModel, mujoco.MjModel, Path]:
    _require_pinned_model_package()
    xml_path = Path(musclemimic_models.get_xml_path(MODEL_ASSET_NAME)).resolve(strict=True)
    source_spec = _load_pinned_spec(xml_path, model_nu)
    for actuator in source_spec.actuators:
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            # Reproduce the reviewed v1 MyoFullBody runtime mutation.  The
            # package XML itself contains mixed ranges and is not the model
            # that produced legacy rollout ctrl.
            actuator.ctrlrange = [-1.0, 1.0]
            actuator.ctrllimited = True
    source_model = source_spec.compile()
    target_spec = _load_pinned_spec(xml_path, model_nu)
    for actuator in target_spec.actuators:
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            actuator.ctrlrange = [0.0, 1.0]
            actuator.ctrllimited = True
    target_model = target_spec.compile()
    for label, model in (
        ("source", source_model),
        ("target", target_model),
    ):
        if int(model.nu) != int(model_nu):
            raise ValueError(
                "pinned MyoFullBody asset dimension differs from legacy capture: "
                f"{label}={int(model.nu)} legacy={int(model_nu)}"
            )
    return source_model, target_model, xml_path


def _compiled_model_sha256(model: mujoco.MjModel) -> str:
    state = model.__getstate__()
    if not isinstance(state, bytes) or not state:
        raise ValueError("compiled MuJoCo model has no canonical byte state")
    return hashlib.sha256(state).hexdigest()


def _ordered_actuator_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        if not name:
            raise ValueError(f"pinned MyoFullBody actuator id={actuator_id} has no stable name")
        names.append(str(name))
    return names


def _source_model_provenance(
    *,
    model: mujoco.MjModel,
    model_nu: int,
    names: list[str],
    ctrlrange_schema_hash: str,
    xml_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_MODEL_PROVENANCE_SCHEMA_VERSION,
        "package": PINNED_MODEL_PACKAGE,
        "package_version": PINNED_MODEL_PACKAGE_VERSION,
        "asset_name": MODEL_ASSET_NAME,
        "disable_fingers": int(model_nu) == 354,
        "model_nu": int(model.nu),
        "model_nv": int(model.nv),
        "model_na": int(model.na),
        "model_xml_sha256": _sha256_file(xml_path),
        "compiled_model_sha256": _compiled_model_sha256(model),
        "actuator_schema_hash": actuator_schema_hash(names),
        "ctrlrange_schema_hash": str(ctrlrange_schema_hash),
    }


def _validate_legacy_signal_semantics(semantics: Any) -> None:
    if not isinstance(semantics, dict) or semantics.get("schema_version") != LEGACY_PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError(
            "source must explicitly declare physical_muscle_transition_v1; "
            "already-migrated or unversioned data is rejected"
        )
    teacher_ctrl = semantics.get("teacher_ctrl_physical")
    expected_teacher_ctrl = {
        "source": "transition_state.data.ctrl",
        "semantics": "raw_applied_mujoco_ctrl_ordered_by_actuator_names",
        "nonnegative": False,
    }
    if not isinstance(teacher_ctrl, dict) or any(
        teacher_ctrl.get(key) != value for key, value in expected_teacher_ctrl.items()
    ):
        raise ValueError("legacy teacher_ctrl_physical semantics are missing or ambiguous")
    excitation = semantics.get("muscle_excitation")
    expected_excitation = {
        "source": "teacher_ctrl_physical",
        "semantics": "unit_interval_excitation",
        "transform": LEGACY_EXCITATION_TRANSFORM,
        "nonnegative": True,
    }
    if not isinstance(excitation, dict) or any(
        excitation.get(key) != value for key, value in expected_excitation.items()
    ):
        raise ValueError("legacy muscle_excitation must explicitly declare the ordered ctrlrange affine v1 transform")
    activation = semantics.get("muscle_activation")
    expected_activation = {
        "source": MUSCLE_ACTIVATION_SOURCE,
        "semantics": MUSCLE_ACTIVATION_SEMANTICS,
        "nonnegative": True,
        "upper_bound": 1.0,
        "roundoff_policy": UNIT_INTERVAL_ROUNDOFF_POLICY,
    }
    if not isinstance(activation, dict) or any(
        activation.get(key) != value for key, value in expected_activation.items()
    ):
        raise ValueError("legacy muscle_activation semantics are missing or ambiguous")


def _validate_legacy_metadata(
    metadata: dict[str, Any],
) -> tuple[list[str], dict[str, Any], np.ndarray]:
    _validate_legacy_signal_semantics(metadata.get("physical_signal_semantics"))
    capture = metadata.get("physical_capture")
    if not isinstance(capture, dict) or capture.get("schema_version") != LEGACY_PHYSICAL_CAPTURE_SCHEMA_VERSION:
        raise ValueError("source must declare physical_capture_spec_v1 metadata")
    action_dim = int(metadata.get("action_dim", -1))
    if action_dim not in SUPPORTED_MODEL_WIDTHS:
        raise ValueError(f"legacy metadata action_dim must be an exact supported MyoFullBody width, got {action_dim}")
    names = actuator_names_from_metadata(metadata, action_dim=action_dim)
    if names is None:
        raise ValueError("legacy dataset lacks exact ordered actuator_names")
    expected_action_hash = actuator_schema_hash(names)
    if metadata.get("action_schema_hash") != expected_action_hash:
        raise ValueError("legacy dataset requires a verified action_schema_hash")
    if [str(name) for name in capture.get("actuator_names", ())] != names:
        raise ValueError("legacy physical_capture actuator order differs from dataset metadata")
    if int(capture.get("model_nu", -1)) != action_dim:
        raise ValueError("legacy physical_capture.model_nu must equal action_dim")
    activation_valid = validate_activation_valid_mask(
        capture.get("activation_valid_mask"),
        expected_width=action_dim,
    )
    if not np.all(activation_valid):
        invalid = np.flatnonzero(~activation_valid).tolist()
        raise ValueError(f"legacy activation_valid_mask must already be all true; invalid ordered indices={invalid}")

    ctrlrange = validate_ordered_ctrlrange(
        names,
        metadata.get("actuator_ctrlrange"),
    )
    ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": names,
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    if metadata.get("ctrlrange_schema_hash") != ctrlrange_hash:
        raise ValueError("legacy dataset requires a verified ctrlrange_schema_hash")
    return names, capture, ctrlrange


def _validate_source_model_binding(
    *,
    names: list[str],
    ctrlrange: np.ndarray,
    capture: dict[str, Any],
    source_model: mujoco.MjModel,
    xml_path: Path,
    acknowledge_pinned_v1_model: bool,
) -> dict[str, Any]:
    canonical_names = _ordered_actuator_names(source_model)
    if names != canonical_names:
        raise ValueError("legacy actuator names/order do not match the pinned source model")
    canonical_ctrlrange = np.asarray(
        source_model.actuator_ctrlrange,
        dtype=np.float64,
    )
    if not np.array_equal(ctrlrange, canonical_ctrlrange):
        raise ValueError("legacy actuator_ctrlrange does not match the pinned source model")
    ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": names,
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    expected = _source_model_provenance(
        model=source_model,
        model_nu=int(source_model.nu),
        names=names,
        ctrlrange_schema_hash=ctrlrange_hash,
        xml_path=xml_path,
    )
    supplied = capture.get("source_model_provenance")
    if supplied is None:
        if not acknowledge_pinned_v1_model:
            raise ValueError(
                "legacy physical_capture has no source_model_provenance; "
                "migration requires the explicit pinned-v1 model attestation"
            )
        verification_mode = (
            "explicit_user_attestation_plus_exact_v1_metadata_and_pinned_model"
        )
    else:
        if not isinstance(supplied, dict):
            raise ValueError(
                "legacy physical_capture.source_model_provenance must be an object"
            )
        mismatches = {
            key: {
                "supplied": supplied.get(key),
                "expected": value,
            }
            for key, value in expected.items()
            if supplied.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "legacy source_model_provenance does not match the pinned "
                f"MyoFullBody source model: {mismatches}"
            )
        verification_mode = "persisted_source_model_provenance_exact_match"
    for field in ("model_nv", "model_na"):
        if int(capture.get(field, -1)) != int(expected[field]):
            raise ValueError(f"legacy physical_capture.{field} differs from source model provenance")
    return {
        **expected,
        "verification_mode": verification_mode,
    }


def _validate_legacy_shard(
    arrays: dict[str, np.ndarray],
    *,
    shard_name: str,
    channel_width: int,
    qfrc_width: int,
    legacy_ctrlrange: np.ndarray,
) -> int:
    required = set(REQUIRED_FIELDS) | set(PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS)
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"legacy shard {shard_name} lacks required physical fields: {missing}")
    row_count: int | None = None
    for field, value in arrays.items():
        array = np.asarray(value)
        if array.ndim < 1:
            raise ValueError(f"legacy shard {shard_name} field {field!r} must have a row dimension")
        if row_count is None:
            row_count = int(array.shape[0])
        elif int(array.shape[0]) != row_count:
            raise ValueError(
                f"legacy shard {shard_name} field {field!r} has {int(array.shape[0])} rows, expected {row_count}"
            )
    if row_count is None or row_count <= 0:
        raise ValueError(f"legacy shard {shard_name} must contain at least one row")

    channel_fields = (
        "teacher_action",
        "teacher_ctrl_physical",
        "muscle_excitation",
        "muscle_activation",
        "muscle_force",
        "muscle_tendon_length",
        "muscle_tendon_velocity",
        "actuator_power",
    )
    for field in channel_fields:
        shape = np.asarray(arrays[field]).shape
        if shape != (row_count, int(channel_width)):
            raise ValueError(
                f"legacy shard {shard_name} field {field!r} must have shape "
                f"({row_count}, {int(channel_width)}), got {shape}"
            )
    qfrc_shape = np.asarray(arrays["qfrc_actuator"]).shape
    if qfrc_shape != (row_count, int(qfrc_width)):
        raise ValueError(
            f"legacy shard {shard_name} qfrc_actuator must have shape "
            f"({row_count}, {int(qfrc_width)}), got {qfrc_shape}"
        )
    for field in PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS:
        if not np.all(np.isfinite(np.asarray(arrays[field]))):
            raise ValueError(f"legacy shard {shard_name} field {field!r} contains non-finite values")
    activation = np.asarray(arrays["muscle_activation"], dtype=np.float64)
    if np.any(activation < 0.0) or np.any(activation > 1.0):
        raise ValueError(f"legacy shard {shard_name} muscle_activation must already lie in [0,1]")
    _validate_legacy_excitation(
        arrays["teacher_ctrl_physical"],
        arrays["muscle_excitation"],
        legacy_ctrlrange=legacy_ctrlrange,
        context=f"legacy shard {shard_name}",
    )
    return row_count


def _validate_legacy_excitation(
    raw_ctrl: Any,
    legacy_excitation: Any,
    *,
    legacy_ctrlrange: np.ndarray,
    context: str,
) -> None:
    raw = np.asarray(raw_ctrl, dtype=np.float64)
    legacy = np.asarray(legacy_excitation, dtype=np.float64)
    if raw.shape != legacy.shape:
        raise ValueError(f"{context} raw ctrl and excitation fields must have identical shapes")
    limits = np.asarray(legacy_ctrlrange, dtype=np.float64)
    if raw.ndim != 2 or limits.shape != (raw.shape[1], 2):
        raise ValueError(f"{context} raw ctrl width differs from legacy actuator_ctrlrange")
    if np.any(raw < limits[:, 0] - LEGACY_ROUNDOFF_ATOL) or np.any(raw > limits[:, 1] + LEGACY_ROUNDOFF_ATOL):
        raise ValueError(f"{context} raw ctrl lies outside its declared ctrlrange")
    expected = (raw - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    absolute_error = np.abs(legacy - expected)
    if not np.all(np.isfinite(absolute_error)) or np.any(absolute_error > LEGACY_ROUNDOFF_ATOL):
        max_error = (
            float(np.max(absolute_error))
            if absolute_error.size and np.all(np.isfinite(absolute_error))
            else float("nan")
        )
        raise ValueError(
            f"{context} muscle_excitation does not equal the declared "
            "ctrlrange affine v1 coordinate within roundoff-only tolerance; "
            f"max_abs_error={max_error}"
        )


def _migrate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    channel_contract: Any,
    legacy_ctrlrange: np.ndarray,
) -> dict[str, np.ndarray]:
    for field in ("teacher_ctrl_physical", "muscle_excitation"):
        if field not in arrays:
            raise ValueError(f"legacy shard lacks required field {field!r}")
    _validate_legacy_excitation(
        arrays["teacher_ctrl_physical"],
        arrays["muscle_excitation"],
        legacy_ctrlrange=legacy_ctrlrange,
        context="legacy shard",
    )
    raw = np.asarray(arrays["teacher_ctrl_physical"])
    legacy = np.asarray(arrays["muscle_excitation"])
    if LEGACY_EXCITATION_FIELD in arrays:
        raise ValueError(f"legacy shard already contains {LEGACY_EXCITATION_FIELD}")
    migrated = dict(arrays)
    migrated[LEGACY_EXCITATION_FIELD] = legacy
    migrated["muscle_excitation"] = physical_ctrl_to_effective_muscle_excitation(
        raw,
        channel_contract=channel_contract,
    )
    return migrated


def _validate_staging_end_to_end(
    staging: Path,
    *,
    source_shards: list[Path],
    actuator_names: list[str],
) -> None:
    expected_groups: list[tuple[str, list[str]]] = []
    for split in ("train", "val", "test"):
        names = sorted(path.name for path in source_shards if path.name.startswith(f"{split}_"))
        if names:
            expected_groups.append((split, names))
    legacy_names = sorted(path.name for path in source_shards if path.name.startswith("shard_"))
    if legacy_names:
        expected_groups.append(("__legacy_shards__", legacy_names))

    for split, expected_names in expected_groups:
        dataset = PhysicalDistillDataset(
            staging,
            split=split,
            target_actuator_names=actuator_names,
        )
        loaded_names = [path.name for path in dataset.shard_paths]
        if loaded_names != expected_names:
            raise ValueError(
                "staging end-to-end validation loaded the wrong shard set: "
                f"split={split!r} loaded={loaded_names} expected={expected_names}"
            )


def migrate_dataset(
    source: Path,
    destination: Path,
    *,
    acknowledge_pinned_v1_model: bool = False,
) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"legacy dataset source is not a directory: {source}")
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if destination.is_relative_to(source):
        raise ValueError("destination must not be inside the read-only source dataset")
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing to overwrite: {destination}")
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"legacy dataset metadata is missing: {metadata_path}")
    metadata = _load_json_object(metadata_path)
    names, capture, legacy_ctrlrange = _validate_legacy_metadata(metadata)
    model_nu = int(capture.get("model_nu", -1))
    source_model, target_model, xml_path = _build_pinned_models(model_nu)
    source_model_provenance = _validate_source_model_binding(
        names=names,
        ctrlrange=legacy_ctrlrange,
        capture=capture,
        source_model=source_model,
        xml_path=xml_path,
        acknowledge_pinned_v1_model=bool(acknowledge_pinned_v1_model),
    )
    channel_contract = resolve_muscle_channel_contract(target_model, names)

    shard_paths = sorted(
        {
            *source.glob("shard_*.npz"),
            *source.glob("train_*.npz"),
            *source.glob("val_*.npz"),
            *source.glob("test_*.npz"),
        }
    )
    if not shard_paths:
        raise FileNotFoundError(f"legacy dataset contains no NPZ shards: {source}")

    package_version = _require_pinned_model_package()
    unit_ctrlrange = np.tile([0.0, 1.0], (len(names), 1))
    migrated_metadata = dict(metadata)
    migrated_metadata["actuator_ctrlrange"] = unit_ctrlrange.tolist()
    migrated_metadata["ctrlrange_schema_hash"] = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": names,
            "ctrlrange": unit_ctrlrange.tolist(),
        },
    )
    migrated_metadata["physical_signal_semantics"] = physical_signal_metadata()
    migrated_metadata["physical_capture"] = {
        **capture,
        "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "activation_valid_mask": validate_activation_valid_mask(
            capture["activation_valid_mask"],
            expected_width=len(names),
        ).tolist(),
        "muscle_channel_contract": channel_contract.to_metadata(),
    }
    migrated_metadata["migration"] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "source_dataset": str(source),
        "source_metadata_sha256": _sha256_file(metadata_path),
        "source_physical_signal_schema_version": (LEGACY_PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "legacy_excitation_field": LEGACY_EXCITATION_FIELD,
        "new_excitation_formula": "clip(teacher_ctrl_physical,0,1)",
        "musclemimic_models_version": package_version,
        "model_xml_path": str(xml_path),
        "model_xml_sha256": _sha256_file(xml_path),
        "source_model_provenance": source_model_provenance,
        "target_compiled_model_sha256": _compiled_model_sha256(target_model),
        "requires_basis_refit": True,
        "requires_fresh_optimizer": True,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.migrating-",
            dir=destination.parent,
        )
    )
    try:
        shard_records: list[dict[str, Any]] = []
        field_tail_shapes: dict[str, tuple[int, ...]] = {}
        for source_shard in shard_paths:
            with np.load(source_shard, allow_pickle=False) as loaded:
                arrays = {str(field): np.asarray(loaded[field]) for field in loaded.files}
            _validate_legacy_shard(
                arrays,
                shard_name=source_shard.name,
                channel_width=len(names),
                qfrc_width=int(source_model.nv),
                legacy_ctrlrange=legacy_ctrlrange,
            )
            for field, array in arrays.items():
                tail_shape = tuple(int(value) for value in np.asarray(array).shape[1:])
                previous = field_tail_shapes.setdefault(field, tail_shape)
                if tail_shape != previous:
                    raise ValueError(
                        "legacy shard field width/tail shape mismatch across "
                        f"shards: field={field!r} previous={previous} "
                        f"current={tail_shape} shard={source_shard.name}"
                    )
            migrated = _migrate_arrays(
                arrays,
                channel_contract=channel_contract,
                legacy_ctrlrange=legacy_ctrlrange,
            )
            output_shard = staging / source_shard.name
            np.savez_compressed(output_shard, **migrated)
            shard_records.append(
                {
                    "name": source_shard.name,
                    "source_sha256": _sha256_file(source_shard),
                    "output_sha256": _sha256_file(output_shard),
                }
            )
        migrated_metadata["migration"]["shards"] = shard_records
        migrated_metadata = _infer_metadata(staging, migrated_metadata)
        (staging / "metadata.json").write_text(
            json.dumps(
                migrated_metadata,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_staging_end_to_end(
            staging,
            source_shards=shard_paths,
            actuator_names=names,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a physical_muscle_transition_v1 dataset to the verified MuJoCo effective-excitation v2 contract."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--acknowledge-pinned-v1-model",
        action="store_true",
        help=(
            "Required only when the historical v1 metadata predates "
            "source_model_provenance. Attest that it was collected with the "
            "reviewed repository v1 MyoFullBody mutation and the pinned "
            "musclemimic-models==1.0.5 asset; exact names, signed ctrlranges, "
            "dimensions and v1 signal equations are still verified."
        ),
    )
    args = parser.parse_args()
    output = migrate_dataset(
        args.source,
        args.destination,
        acknowledge_pinned_v1_model=args.acknowledge_pinned_v1_model,
    )
    print(output)


if __name__ == "__main__":
    main()
