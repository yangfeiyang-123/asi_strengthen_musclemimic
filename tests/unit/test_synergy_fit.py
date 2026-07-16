import hashlib
import json

import numpy as np
import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.fit import (
    SynergyFitConfig,
    fit_synergy_dataset,
    load_synergy_split,
    primitive_task_phase_balanced_weights,
    synergy_phase_weight_fingerprint,
    synergy_preprocessing_fingerprint,
)
from musclemimic.synergy.grouping import load_grouping_json
from musclemimic.synergy.primitive_manifest import (
    save_primitive_source_manifest,
    save_primitive_source_manifest_from_splits,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash

TEACHER_SHA256 = "a" * 64


def test_primitive_task_phase_weights_balance_cells_and_apply_quality():
    tasks = np.asarray(["squat", "squat", "jump", "jump", "jump", "jump"])
    trials = np.asarray(["s1", "s2", "j1", "j1", "j2", "j2"])
    phases = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32)
    quality = np.asarray([0.5, 1.0, 1.0, 1.0, 0.5, 1.0])
    weights = primitive_task_phase_balanced_weights(
        tasks,
        phases,
        trial_ids=trials,
        quality_weights=quality,
        phase_weights={0: 1.0, 1: 2.0},
    )

    assert np.mean(weights) == pytest.approx(1.0)
    # Every primitive task receives the same total despite different phases.
    assert np.sum(weights[:2]) == pytest.approx(np.sum(weights[2:]))
    # A rollout-level low-quality trial receives less total cell weight.
    assert weights[0] == pytest.approx(0.5 * weights[1])
    # QC quality still downweights the lower-quality sample in its cell.
    assert weights[4] == pytest.approx(0.5 * weights[5])


def _json_fingerprint(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _teacher_content():
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": "fixture",
        "resolved_path": "/fixture/checkpoint",
        "sha256": TEACHER_SHA256,
        "num_files": 1,
        "num_bytes": 1,
        "files": [{"path": "params", "sha256": "b" * 64, "num_bytes": 1}],
    }


def _primitive_checkpoint_content(sha256: str, task: str) -> dict:
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": f"fixtures/{task}",
        "resolved_path": f"/fixtures/{task}",
        "sha256": sha256,
        "num_files": 1,
        "num_bytes": 1,
        "files": [
            {"path": "params", "sha256": "b" * 64, "num_bytes": 1}
        ],
    }


def _unit_signals(samples, *, seed):
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(0.05, 1.0, size=(samples, 2))
    basis = np.asarray(
        [
            [1.0, 0.05],
            [0.7, 0.15],
            [0.05, 1.0],
            [0.15, 0.7],
        ]
    )
    excitation = (coefficients @ basis.T) / 1.2
    assert np.max(excitation) <= 1.0
    return excitation.astype(np.float32), (0.7 * excitation).astype(np.float32)


def _write_dataset(root):
    names = ["left_hip", "right_hip", "trunk", "right_wrist"]
    ctrlrange = np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float32), (len(names), 1))
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": names, "ctrlrange": ctrlrange.astype(float).tolist()},
        ),
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": {
            "schema_version": "physical_capture_spec_v1",
            "actuator_names": names,
            "activation_valid_mask": [True] * len(names),
        },
        "teacher_checkpoint_fingerprint": TEACHER_SHA256,
        "teacher_checkpoint_content": _teacher_content(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, samples, seed in (("train", 72, 3), ("val", 48, 7)):
        excitation, activation = _unit_signals(samples, seed=seed)
        phase_id = np.resize(np.arange(6, dtype=np.int32), samples)
        motion_start = 100 if split == "train" else 200
        motion_uid = np.repeat(
            np.arange(motion_start, motion_start + (samples // 24), dtype=np.int64),
            24,
        )
        np.savez(
            root / f"{split}_000000.npz",
            teacher_ctrl_physical=2.0 * excitation - 1.0,
            muscle_excitation=excitation,
            muscle_activation=activation,
            phase_id=phase_id,
            motion_uid=motion_uid,
        )
    return names


def _write_primitive_dataset(root):
    names = _write_dataset(root)
    checkpoints = {"squat": "1" * 64, "jump": "2" * 64}
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_checkpoint_fingerprints"] = checkpoints
    metadata["source_checkpoint_contents"] = {
        task: _primitive_checkpoint_content(checkpoint, task)
        for task, checkpoint in checkpoints.items()
    }
    metadata["primitive_required_phase_ids"] = {
        "squat": [0, 2, 4],
        "jump": [1, 3, 5],
    }
    metadata["primitive_phase_schema_fingerprints"] = {
        "squat": "5" * 64,
        "jump": "6" * 64,
    }
    metadata["model_hash"] = "4" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        count = arrays["phase_id"].shape[0]
        task_id = np.resize(np.asarray(["squat", "jump"]), count)
        sample_index = np.arange(count)
        if path.name.startswith("train"):
            trial_suffix = (sample_index // 2) % 2
            trial_id = np.asarray(
                [
                    f"{task}-train-{motion}-{suffix}"
                    for task, motion, suffix in zip(
                        task_id,
                        arrays["motion_uid"],
                        trial_suffix,
                        strict=True,
                    )
                ]
            )
        else:
            trial_id = np.asarray(
                [
                    f"{task}-val-{motion}"
                    for task, motion in zip(
                        task_id,
                        arrays["motion_uid"],
                        strict=True,
                    )
                ]
            )
        arrays.update(
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "source_kind": np.full(count, "primitive"),
                "success": np.ones(count, dtype=np.int8),
                "quality_weight": np.linspace(0.8, 1.0, count, dtype=np.float32),
            }
        )
        np.savez(path, **arrays)
    return names, checkpoints


def test_fit_cli_core_builds_global_regional_and_composite_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    names = _write_dataset(dataset)
    grouping = tmp_path / "groups.json"
    grouping.write_text(
        json.dumps(
            {
                "regions": {
                    "lower_body": ["left_hip", "right_hip"],
                    "upper_body": ["trunk", "right_wrist"],
                }
            }
        ),
        encoding="utf-8",
    )
    config = SynergyFitConfig(
        ranks=(1, 2),
        seeds=(0, 1),
        max_iter=300,
        tol=1e-7,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_val_global_vaf=0.94,
        min_val_local_vaf_quantile=0.70,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
    )
    report = fit_synergy_dataset(
        dataset,
        dataset,
        output_dir=tmp_path / "fit",
        signal_kinds=("excitation", "activation"),
        mode="both",
        grouping_json=grouping,
        config=config,
    )

    assert report["schema_version"] == "forehand_clear_synergy_fit_report_v1"
    assert set(report["preferred_decoder_artifacts"]) == {
        "physical_excitation_unit",
        "muscle_activation",
    }
    excitation_preferred = report["preferred_decoder_artifacts"]["physical_excitation_unit"]
    composite = load_synergy_basis(excitation_preferred["artifact_path"])
    assert composite.muscle_names == tuple(names)
    assert composite.manifest["composite_schema_version"] == "regional_synergy_composite_v1"
    assert composite.manifest["region"] == "regional_composite"
    descriptors = composite.manifest["composite_regions"]
    assert [item["region"] for item in descriptors] == ["lower_body", "upper_body"]
    for item in descriptors:
        rows = np.asarray(item["row_indices"], dtype=np.int32)
        outside = np.ones(len(names), dtype=bool)
        outside[rows] = False
        np.testing.assert_allclose(
            composite.basis[outside, item["column_start"] : item["column_stop"]],
            0.0,
        )
    assert all(
        artifact["selected_metrics"]["validation"]["global_vaf"] > 0.85
        for artifact in report["artifacts"]
        if artifact["artifact_role"] != "primary_regional_composite"
    )
    assert (tmp_path / "fit" / "fit_report.json").is_file()
    promotion = json.loads((tmp_path / "fit" / "promotion_metrics.json").read_text(encoding="utf-8"))
    assert promotion["heldout_sample_count"] == 48
    assert promotion["explained_variance"] > 0.85
    assert promotion["artifact_binding_verified"] == 1.0
    assert promotion["basis_artifact_fingerprint"] == composite.fingerprint


def test_primitive_fit_binds_sample_inventory_and_source_manifest(tmp_path):
    dataset = tmp_path / "primitive_dataset"
    names, checkpoints = _write_primitive_dataset(dataset)

    train = load_synergy_split(dataset, split="train")
    validation = load_synergy_split(dataset, split="val")
    source_dataset_fingerprint = _json_fingerprint(
        {"train": train.provenance(), "validation": validation.provenance()}
    )
    config = SynergyFitConfig(
        ranks=(1, 2),
        seeds=(0, 1),
        max_iter=200,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
    )
    source = save_primitive_source_manifest_from_splits(
        tmp_path / "primitive_source",
        train_source=dataset,
        validation_source=dataset,
        target_skill_id="ChinaJump",
        excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
        source_checkpoint_fingerprints=checkpoints,
        fit_config=config,
    )

    report = fit_synergy_dataset(
        dataset,
        dataset,
        output_dir=tmp_path / "primitive_fit",
        signal_kinds=("excitation",),
        mode="global",
        primitive_source_manifest=source.path,
        config=config,
    )
    basis = load_synergy_basis(
        report["preferred_decoder_artifacts"]["physical_excitation_unit"][
            "artifact_path"
        ]
    )
    binding = basis.manifest["primitive_source_binding"]
    assert binding["manifest_fingerprint"] == source.fingerprint
    assert binding["source_dataset_fingerprint"] == source_dataset_fingerprint
    assert binding["primitive_only"] is True
    assert binding["contains_target_skill_rollouts"] is False
    assert basis.manifest["phase_balancing"]["sample_balancing"]["kind"] == (
        "primitive_task_phase_trial_balanced"
    )


def test_primitive_manifest_builder_rejects_missing_required_task_phase(tmp_path):
    dataset = tmp_path / "primitive_missing_phase"
    _, checkpoints = _write_primitive_dataset(dataset)
    shard_path = dataset / "val_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    missing_mask = (arrays["task_id"] == "squat") & (arrays["phase_id"] == 4)
    assert np.any(missing_mask)
    arrays["phase_id"] = arrays["phase_id"].copy()
    arrays["phase_id"][missing_mask] = 0
    np.savez(shard_path, **arrays)

    with pytest.raises(ValueError, match="missing required phase_ids: \\[4\\]"):
        save_primitive_source_manifest_from_splits(
            tmp_path / "source",
            train_source=dataset,
            validation_source=dataset,
            target_skill_id="ChinaJump",
            excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
            source_checkpoint_fingerprints=checkpoints,
            fit_config=SynergyFitConfig(seeds=(0, 1)),
        )


def test_primitive_fit_rechecks_required_phases_against_loaded_rows(tmp_path):
    dataset = tmp_path / "primitive_fit_missing_phase"
    _, checkpoints = _write_primitive_dataset(dataset)
    config = SynergyFitConfig(seeds=(0, 1))
    source = save_primitive_source_manifest_from_splits(
        tmp_path / "original_source",
        train_source=dataset,
        validation_source=dataset,
        target_skill_id="ChinaJump",
        excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
        source_checkpoint_fingerprints=checkpoints,
        fit_config=config,
    )
    shard_path = dataset / "val_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    missing_mask = (arrays["task_id"] == "squat") & (arrays["phase_id"] == 4)
    arrays["phase_id"] = arrays["phase_id"].copy()
    arrays["phase_id"][missing_mask] = 0
    np.savez(shard_path, **arrays)

    train = load_synergy_split(dataset, split="train")
    validation = load_synergy_split(dataset, split="val")
    source_dataset_fingerprint = _json_fingerprint(
        {"train": train.provenance(), "validation": validation.provenance()}
    )
    manifest = source.manifest
    rebound = save_primitive_source_manifest(
        tmp_path / "rebound_source",
        target_skill_id=manifest["target_skill_id"],
        excluded_target_motion_paths=[
            item["path"] for item in manifest["excluded_target_motions"]
        ],
        primitive_task_ids=manifest["primitive_task_ids"],
        primitive_source_kinds=manifest["primitive_source_kinds"],
        primitive_trial_ids=manifest["primitive_trial_ids"],
        train_motion_uids=manifest["train_motion_uids"],
        validation_motion_uids=manifest["validation_motion_uids"],
        source_checkpoint_fingerprints=manifest[
            "source_checkpoint_fingerprints"
        ],
        source_checkpoint_contents=manifest["source_checkpoint_contents"],
        primitive_required_phase_ids=manifest["primitive_required_phase_ids"],
        primitive_phase_schema_fingerprints=manifest[
            "primitive_phase_schema_fingerprints"
        ],
        source_dataset_fingerprint=source_dataset_fingerprint,
        model_hash=manifest["model_hash"],
        actuator_schema_hash=manifest["actuator_schema_hash"],
        control_range_hash=manifest["control_range_hash"],
        transform_ctrlrange_schema_hash=manifest[
            "transform_ctrlrange_schema_hash"
        ],
        preprocessing_fingerprint=manifest["preprocessing_fingerprint"],
        phase_weight_fingerprint=manifest["phase_weight_fingerprint"],
        nmf_seeds=manifest["NMF_seeds"],
    )

    with pytest.raises(ValueError, match="missing required phase_ids: \\[4\\]"):
        fit_synergy_dataset(
            dataset,
            dataset,
            output_dir=tmp_path / "fit",
            signal_kinds=("excitation",),
            mode="global",
            primitive_source_manifest=rebound.path,
            config=config,
        )


def test_primitive_manifest_builder_requires_checkpoint_content_audit(tmp_path):
    dataset = tmp_path / "primitive_missing_content"
    _, checkpoints = _write_primitive_dataset(dataset)
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("source_checkpoint_contents")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="require source_checkpoint_contents"):
        save_primitive_source_manifest_from_splits(
            tmp_path / "source",
            train_source=dataset,
            validation_source=dataset,
            target_skill_id="ChinaJump",
            excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
            source_checkpoint_fingerprints=checkpoints,
            fit_config=SynergyFitConfig(seeds=(0, 1)),
        )


def test_primitive_manifest_builder_rejects_trial_spanning_motion_uids(tmp_path):
    dataset = tmp_path / "primitive_cross_motion_trial"
    _, checkpoints = _write_primitive_dataset(dataset)
    shard_path = dataset / "train_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    squat_indices = np.flatnonzero(arrays["task_id"] == "squat")
    first = int(squat_indices[0])
    second = next(
        int(index)
        for index in squat_indices
        if arrays["motion_uid"][index] != arrays["motion_uid"][first]
    )
    arrays["trial_id"] = arrays["trial_id"].copy()
    arrays["trial_id"][[first, second]] = "squat-cross-motion"
    np.savez(shard_path, **arrays)

    with pytest.raises(ValueError, match="trial_id must bind exactly one motion_uid"):
        save_primitive_source_manifest_from_splits(
            tmp_path / "source",
            train_source=dataset,
            validation_source=dataset,
            target_skill_id="ChinaJump",
            excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
            source_checkpoint_fingerprints=checkpoints,
            fit_config=SynergyFitConfig(seeds=(0, 1)),
        )


def test_primitive_manifest_builder_rejects_split_phase_schema_mismatch(tmp_path):
    train_dataset = tmp_path / "primitive_train"
    validation_dataset = tmp_path / "primitive_validation"
    _, checkpoints = _write_primitive_dataset(train_dataset)
    _write_primitive_dataset(validation_dataset)
    metadata_path = validation_dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["primitive_phase_schema_fingerprints"]["jump"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="train/validation metadata primitive_phase_schema_fingerprints differ",
    ):
        save_primitive_source_manifest_from_splits(
            tmp_path / "source",
            train_source=train_dataset,
            validation_source=validation_dataset,
            target_skill_id="ChinaJump",
            excluded_target_motion_paths=["ChinaJump/forehandJump-1"],
            source_checkpoint_fingerprints=checkpoints,
            fit_config=SynergyFitConfig(seeds=(0, 1)),
        )


def test_fit_source_recomputes_excitation_and_rejects_tampering(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    shard_path = dataset / "train_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        payload = {key: np.asarray(shard[key]) for key in shard.files}
    payload["muscle_excitation"] = payload["muscle_excitation"].copy()
    payload["muscle_excitation"][0, 0] += 0.05
    np.savez(shard_path, **payload)

    split = load_synergy_split(dataset, split="train")
    with pytest.raises(ValueError, match="differs from explicit raw ctrlrange transform"):
        split.signal("excitation")


def test_fit_source_rejects_activation_outside_unit_contract(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    shard_path = dataset / "train_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        payload = {key: np.asarray(shard[key]) for key in shard.files}
    payload["muscle_activation"] = payload["muscle_activation"].copy()
    payload["muscle_activation"][0, 0] = 1.01
    np.savez(shard_path, **payload)

    split = load_synergy_split(dataset, split="train")
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        split.signal("activation")


def test_grouping_json_rejects_duplicate_keys(tmp_path):
    grouping = tmp_path / "duplicate.json"
    grouping.write_text(
        '{"regions":{"first":["a"],"first":["b"]}}',
        encoding="utf-8",
    )
    with pytest.raises(DuplicateJsonKeyError):
        load_grouping_json(grouping, muscle_names=("a", "b"))
