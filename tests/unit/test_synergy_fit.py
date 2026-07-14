import json

import numpy as np
import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError
from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.fit import (
    SynergyFitConfig,
    fit_synergy_dataset,
    load_synergy_split,
)
from musclemimic.synergy.grouping import load_grouping_json

TEACHER_SHA256 = "a" * 64


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
