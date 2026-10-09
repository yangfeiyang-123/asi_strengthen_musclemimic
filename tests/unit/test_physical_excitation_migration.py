from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from musclemimic.distill.action_schema import (
    actuator_schema_hash,
    ordered_schema_hash,
)
from musclemimic.distill.dataset import PhysicalDistillDataset
from musclemimic.distill.physical import MuscleChannelContract
from scripts import migrate_physical_excitation_v1_to_v2 as migration


def _contract() -> MuscleChannelContract:
    return MuscleChannelContract(
        actuator_names=("m0", "m1"),
        actuator_ids=(0, 1),
        actuator_dyntype=("muscle", "muscle"),
        actuator_actnum=(1, 1),
        actuator_actadr=(0, 1),
        model_na=2,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_signal_semantics() -> dict[str, Any]:
    return {
        "schema_version": migration.LEGACY_PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "teacher_ctrl_physical": {
            "source": "transition_state.data.ctrl",
            "semantics": ("raw_applied_mujoco_ctrl_ordered_by_actuator_names"),
            "nonnegative": False,
        },
        "muscle_excitation": {
            "source": "teacher_ctrl_physical",
            "semantics": "unit_interval_excitation",
            "transform": migration.LEGACY_EXCITATION_TRANSFORM,
            "nonnegative": True,
        },
        "muscle_activation": {
            "source": ("transition_state.data.act via model.actuator_actadr"),
            "semantics": "mujoco_unit_interval_activation_state",
            "nonnegative": True,
            "upper_bound": 1.0,
            "roundoff_policy": ("fail_outside_unit_interval_then_clamp_within_tolerance_only"),
            "validity_mask": "physical_capture.activation_valid_mask",
        },
    }


@pytest.fixture(scope="module")
def pinned_354() -> dict[str, Any]:
    source_model, _, xml_path = migration._build_pinned_models(354)
    names = migration._ordered_actuator_names(source_model)
    ctrlrange = np.asarray(
        source_model.actuator_ctrlrange,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        ctrlrange,
        np.tile([-1.0, 1.0], (int(source_model.nu), 1)),
    )
    ctrlrange_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": names,
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    provenance = migration._source_model_provenance(
        model=source_model,
        model_nu=354,
        names=names,
        ctrlrange_schema_hash=ctrlrange_hash,
        xml_path=xml_path,
    )
    return {
        "model": source_model,
        "names": names,
        "ctrlrange": ctrlrange,
        "ctrlrange_hash": ctrlrange_hash,
        "provenance": provenance,
    }


def _legacy_metadata(binding: dict[str, Any]) -> dict[str, Any]:
    model = binding["model"]
    names = binding["names"]
    return {
        "schema_version": "distill_v1",
        "action_dim": len(names),
        "actuator_names": names,
        "action_schema_hash": actuator_schema_hash(names),
        "actuator_ctrlrange": binding["ctrlrange"].tolist(),
        "ctrlrange_schema_hash": binding["ctrlrange_hash"],
        "physical_signal_semantics": _legacy_signal_semantics(),
        "physical_capture": {
            "schema_version": (migration.LEGACY_PHYSICAL_CAPTURE_SCHEMA_VERSION),
            "actuator_names": names,
            "model_nu": int(model.nu),
            "model_nv": int(model.nv),
            "model_na": int(model.na),
            "activation_valid_mask": [True] * len(names),
            "source_model_provenance": copy.deepcopy(binding["provenance"]),
        },
        # Deliberately stale values must be replaced by _infer_metadata.
        "num_samples": 999,
        "fields": ["stale"],
        "shards": ["stale.npz"],
    }


def _legacy_arrays(
    binding: dict[str, Any],
    *,
    rows: int = 2,
) -> dict[str, np.ndarray]:
    model = binding["model"]
    ctrlrange = np.asarray(binding["ctrlrange"], dtype=np.float64)
    fractions = np.linspace(0.2, 0.8, num=rows, dtype=np.float64)[:, None]
    raw = ctrlrange[:, 0] + fractions * (ctrlrange[:, 1] - ctrlrange[:, 0])
    excitation = (raw - ctrlrange[:, 0]) / (ctrlrange[:, 1] - ctrlrange[:, 0])
    width = int(model.nu)
    return {
        "student_obs": np.zeros((rows, 5), dtype=np.float32),
        "teacher_action": np.zeros((rows, width), dtype=np.float32),
        "teacher_ctrl_physical": raw.astype(np.float32),
        "muscle_excitation": excitation.astype(np.float32),
        "muscle_activation": np.broadcast_to(
            np.linspace(0.1, 0.9, num=width, dtype=np.float32),
            (rows, width),
        ).copy(),
        "muscle_force": np.zeros((rows, width), dtype=np.float32),
        "muscle_tendon_length": np.ones(
            (rows, width),
            dtype=np.float32,
        ),
        "muscle_tendon_velocity": np.zeros(
            (rows, width),
            dtype=np.float32,
        ),
        "actuator_power": np.zeros((rows, width), dtype=np.float32),
        "qfrc_actuator": np.zeros(
            (rows, int(model.nv)),
            dtype=np.float32,
        ),
    }


def _write_legacy_dataset(
    root: Path,
    binding: dict[str, Any],
    *,
    metadata_mutator=None,
    arrays_mutator=None,
    shard_names: tuple[str, ...] = ("train_000000.npz",),
) -> tuple[Path, dict[str, np.ndarray]]:
    source = root / "legacy"
    source.mkdir()
    metadata = _legacy_metadata(binding)
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    (source / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    arrays = _legacy_arrays(binding)
    if arrays_mutator is not None:
        arrays_mutator(arrays)
    for shard_name in shard_names:
        np.savez_compressed(source / shard_name, **arrays)
    return source, arrays


def _assert_atomic_rejection(
    source: Path,
    destination: Path,
    *,
    match: str,
) -> None:
    before = {path.name: _file_sha256(path) for path in sorted(source.iterdir()) if path.is_file()}
    with pytest.raises(ValueError, match=match):
        migration.migrate_dataset(source, destination)
    assert not destination.exists()
    after = {path.name: _file_sha256(path) for path in sorted(source.iterdir()) if path.is_file()}
    assert after == before


def test_migration_retains_legacy_coordinate_and_recomputes_effective_excitation():
    raw = np.asarray([[-0.5, 0.25], [0.75, 1.0]], dtype=np.float32)
    ctrlrange = np.asarray([[-1.0, 1.0], [0.0, 1.0]])
    legacy = np.asarray([[0.25, 0.25], [0.875, 1.0]], dtype=np.float32)

    migrated = migration._migrate_arrays(
        {
            "teacher_ctrl_physical": raw,
            "muscle_excitation": legacy,
            "student_obs": np.zeros((2, 1), dtype=np.float32),
        },
        channel_contract=_contract(),
        legacy_ctrlrange=ctrlrange,
    )

    np.testing.assert_array_equal(
        migrated[migration.LEGACY_EXCITATION_FIELD],
        legacy,
    )
    np.testing.assert_allclose(
        migrated["muscle_excitation"],
        np.asarray([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        migrated["teacher_ctrl_physical"],
        raw,
    )


def test_migration_requires_complete_physical_fields_and_is_atomic(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def remove_force(arrays):
        arrays.pop("muscle_force")

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        arrays_mutator=remove_force,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="lacks required physical fields.*muscle_force",
    )


def test_migration_rejects_original_false_activation_mask(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def invalidate_mask(metadata):
        metadata["physical_capture"]["activation_valid_mask"][17] = False

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        metadata_mutator=invalidate_mask,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="activation_valid_mask must already be all true",
    )


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_migration_rejects_invalid_original_activation(
    tmp_path: Path,
    pinned_354: dict[str, Any],
    invalid_value: float,
):
    def invalidate_activation(arrays):
        arrays["muscle_activation"][0, 0] = invalid_value

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        arrays_mutator=invalidate_activation,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match=r"muscle_activation must already lie in \[0,1\]",
    )


def test_migration_rejects_model_package_version_mismatch(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def change_version(metadata):
        metadata["physical_capture"]["source_model_provenance"]["package_version"] = "1.0.4"

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        metadata_mutator=change_version,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="source_model_provenance does not match",
    )


def test_migration_requires_exact_installed_package_version(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        migration.importlib.metadata,
        "version",
        lambda package: "1.0.6",
    )
    with pytest.raises(
        ValueError,
        match=r"musclemimic-models==1\.0\.5",
    ):
        migration._require_pinned_model_package()


def test_migration_rejects_source_model_provenance_mismatch(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def corrupt_provenance(metadata):
        metadata["physical_capture"]["source_model_provenance"]["compiled_model_sha256"] = "0" * 64

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        metadata_mutator=corrupt_provenance,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="source_model_provenance does not match",
    )


def test_migration_rejects_missing_source_model_provenance(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def remove_provenance(metadata):
        metadata["physical_capture"].pop("source_model_provenance")

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        metadata_mutator=remove_provenance,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="explicit pinned-v1 model attestation",
    )

    destination = tmp_path / "attested_migration"
    assert (
        migration.migrate_dataset(
            source,
            destination,
            acknowledge_pinned_v1_model=True,
        )
        == destination
    )
    migrated = json.loads(
        (destination / "metadata.json").read_text(encoding="utf-8")
    )
    assert migrated["migration"]["source_model_provenance"][
        "verification_mode"
    ] == (
        "explicit_user_attestation_plus_exact_v1_metadata_and_pinned_model"
    )


def test_migration_rejects_legacy_excitation_not_matching_affine_contract(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    def corrupt_excitation(arrays):
        arrays["muscle_excitation"][0, 0] += 1e-3

    source, _ = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        arrays_mutator=corrupt_excitation,
    )
    _assert_atomic_rejection(
        source,
        tmp_path / "migrated",
        match="does not equal the declared ctrlrange affine",
    )


def test_complete_migration_reinfers_metadata_and_loads_all_shard_kinds(
    tmp_path: Path,
    pinned_354: dict[str, Any],
):
    shard_names = (
        "train_000000.npz",
        "val_000000.npz",
        "test_000000.npz",
        "shard_000000.npz",
    )
    source, source_arrays = _write_legacy_dataset(
        tmp_path,
        pinned_354,
        shard_names=shard_names,
    )
    source_hashes = {path.name: _file_sha256(path) for path in sorted(source.iterdir()) if path.is_file()}
    destination = tmp_path / "migrated"

    assert migration.migrate_dataset(source, destination) == destination

    assert {path.name: _file_sha256(path) for path in sorted(source.iterdir()) if path.is_file()} == source_hashes
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_samples"] == len(shard_names) * 2
    assert metadata["shards"] == sorted(shard_names)
    assert migration.LEGACY_EXCITATION_FIELD in metadata["fields"]
    assert metadata["physical_capture"]["activation_valid_mask"] == ([True] * 354)
    assert metadata["migration"]["musclemimic_models_version"] == "1.0.5"
    assert metadata["migration"]["requires_basis_refit"] is True
    assert metadata["migration"]["requires_fresh_optimizer"] is True

    with np.load(
        destination / "train_000000.npz",
        allow_pickle=False,
    ) as output:
        np.testing.assert_array_equal(
            output[migration.LEGACY_EXCITATION_FIELD],
            source_arrays["muscle_excitation"],
        )
        np.testing.assert_array_equal(
            output["muscle_activation"],
            source_arrays["muscle_activation"],
        )
        np.testing.assert_allclose(
            output["muscle_excitation"],
            np.clip(source_arrays["teacher_ctrl_physical"], 0.0, 1.0),
        )

    for split in ("train", "val", "test"):
        dataset = PhysicalDistillDataset(
            destination,
            split=split,
            target_actuator_names=pinned_354["names"],
        )
        assert dataset.num_samples == 2
    legacy = PhysicalDistillDataset(
        destination,
        split="__legacy_shards__",
        target_actuator_names=pinned_354["names"],
    )
    assert legacy.num_samples == 2
