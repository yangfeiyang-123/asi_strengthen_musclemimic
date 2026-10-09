from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    physical_signal_metadata,
)
from musclemimic.synergy.action_interface import (
    _validate_residual_fit_contract,
    load_structured_residual_basis,
    save_coefficient_statistics,
    save_structured_residual_basis,
)
from musclemimic.synergy.basis_artifact import save_synergy_basis
from musclemimic.synergy.fit import SynergyFitConfig, load_synergy_split
from musclemimic.synergy.primitive_manifest import (
    save_primitive_source_manifest_from_splits,
)
from musclemimic.synergy.residual_fit import (
    StructuredResidualFitConfig,
    _residual_metrics,
    fit_structured_residual_basis,
)
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND, ctrlrange_schema_hash


def _checkpoint_content(task: str, fingerprint: str) -> dict:
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": f"fixtures/{task}",
        "resolved_path": f"/fixtures/{task}",
        "sha256": fingerprint,
        "num_files": 1,
        "num_bytes": 1,
        "files": [{"path": "params", "sha256": "f" * 64, "num_bytes": 1}],
    }


def _muscle_contract(names) -> dict:
    width = len(names)
    return {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(names),
        "actuator_ids": list(range(width)),
        "actuator_dyntype": ["muscle"] * width,
        "actuator_actnum": [1] * width,
        "actuator_actadr": list(range(width)),
        "model_na": width,
    }


def _write_primitive_dataset(root):
    names = [f"muscle_{index}" for index in range(6)]
    ctrlrange = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (len(names), 1))
    checkpoints = {"jump": "1" * 64, "squat": "2" * 64}
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": names, "ctrlrange": ctrlrange.tolist()},
        ),
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": {
            "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
            "actuator_names": names,
            "activation_valid_mask": [True] * len(names),
            "muscle_channel_contract": _muscle_contract(names),
        },
        "model_hash": "3" * 64,
        "source_checkpoint_fingerprints": checkpoints,
        "source_checkpoint_contents": {
            task: _checkpoint_content(task, fingerprint) for task, fingerprint in checkpoints.items()
        },
        "primitive_required_phase_ids": {"jump": [0, 1], "squat": [0, 1]},
        "primitive_phase_schema_fingerprints": {
            "jump": "4" * 64,
            "squat": "5" * 64,
        },
    }
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    primary_basis = np.asarray(
        [
            [0.55, 0.05],
            [0.10, 0.50],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
        ],
        dtype=np.float64,
    )
    residual_row = {
        ("jump", 0): 4,
        ("jump", 1): 5,
        ("squat", 0): 2,
        ("squat", 1): 3,
    }
    for split, trials_per_task, motion_start in (("train", 2, 10), ("val", 1, 30)):
        arrays: dict[str, list] = {
            "teacher_ctrl_physical": [],
            "muscle_excitation": [],
            "muscle_activation": [],
            "phase_id": [],
            "motion_uid": [],
            "task_id": [],
            "trial_id": [],
            "source_kind": [],
            "success": [],
            "quality_weight": [],
        }
        motion = motion_start
        for task in ("jump", "squat"):
            for trial_index in range(trials_per_task):
                trial = f"{task}-{split}-{trial_index}"
                for repeat in range(4):
                    for phase in (0, 1):
                        coefficient = np.asarray([0.18 + 0.01 * repeat, 0.22 - 0.01 * repeat])
                        excitation = primary_basis @ coefficient
                        own_row = residual_row[(task, phase)]
                        cross_row = 2 + ((own_row - 2 + 1) % 4)
                        excitation[own_row] += 0.012 + 0.001 * repeat
                        excitation[cross_row] += 0.008
                        arrays["teacher_ctrl_physical"].append(excitation.copy())
                        arrays["muscle_excitation"].append(excitation)
                        arrays["muscle_activation"].append(0.7 * excitation)
                        arrays["phase_id"].append(phase)
                        arrays["motion_uid"].append(motion)
                        arrays["task_id"].append(task)
                        arrays["trial_id"].append(trial)
                        arrays["source_kind"].append("primitive")
                        arrays["success"].append(1)
                        arrays["quality_weight"].append(1.0)
                motion += 1
        np.savez(
            root / f"{split}_000000.npz",
            teacher_ctrl_physical=np.asarray(arrays["teacher_ctrl_physical"], dtype=np.float32),
            muscle_excitation=np.asarray(arrays["muscle_excitation"], dtype=np.float32),
            muscle_activation=np.asarray(arrays["muscle_activation"], dtype=np.float32),
            phase_id=np.asarray(arrays["phase_id"], dtype=np.int32),
            motion_uid=np.asarray(arrays["motion_uid"], dtype=np.int64),
            task_id=np.asarray(arrays["task_id"]),
            trial_id=np.asarray(arrays["trial_id"]),
            source_kind=np.asarray(arrays["source_kind"]),
            success=np.asarray(arrays["success"], dtype=np.int8),
            quality_weight=np.asarray(arrays["quality_weight"], dtype=np.float32),
        )
    return names, ctrlrange, checkpoints, primary_basis


def _artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    names, ctrlrange, checkpoints, matrix = _write_primitive_dataset(dataset)
    fit_config = SynergyFitConfig(seeds=(0, 1))
    source = save_primitive_source_manifest_from_splits(
        tmp_path / "source",
        train_source=dataset,
        validation_source=dataset,
        target_skill_id="ChinaJump",
        excluded_target_motion_paths=["ChinaJump/target/train", "ChinaJump/target/val"],
        source_checkpoint_fingerprints=checkpoints,
        fit_config=fit_config,
    )
    train = load_synergy_split(dataset, split="train")
    validation = load_synergy_split(dataset, split="val")
    binding = {
        "schema_version": source.manifest["schema_version"],
        "manifest_fingerprint": source.fingerprint,
        "source_dataset_fingerprint": source.manifest["source_dataset_fingerprint"],
        "primitive_only": True,
        "contains_target_skill_rollouts": False,
        "target_skill_id": "ChinaJump",
        "excluded_target_motions": source.manifest["excluded_target_motions"],
        "primitive_task_ids": source.manifest["primitive_task_ids"],
        "model_hash": "3" * 64,
        "transform_ctrlrange_schema_hash": ctrlrange_schema_hash(names, ctrlrange),
    }
    basis = save_synergy_basis(
        tmp_path / "basis",
        basis=matrix,
        muscle_names=names,
        manifest={
            "signal_kind": EXCITATION_SIGNAL_KIND,
            "region": "whole_body",
            "rank": 2,
            "normalization": {"kind": "none"},
            "source_dataset_fingerprint": source.manifest["source_dataset_fingerprint"],
            "teacher_checkpoint_fingerprint": "6" * 64,
            "fit_seed": 0,
            "transform": {
                "kind": UNIT_EXCITATION_TRANSFORM,
                "raw_signal_kind": "applied_ctrl",
                "formula": MUSCLE_EXCITATION_FORMULA,
                "ctrlrange": ctrlrange.tolist(),
                "actuator_names": names,
                "ctrlrange_schema_hash": ctrlrange_schema_hash(names, ctrlrange),
                "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
                "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
                "muscle_channel_contract": _muscle_contract(names),
            },
            "split_provenance": {
                "train": train.provenance(),
                "validation": validation.provenance(),
            },
            "train_motion_uids": sorted(np.unique(train.motion_ids).tolist()),
            "primitive_source_binding": binding,
        },
    )
    stats = save_coefficient_statistics(
        basis.path / "coefficient_stats.npz",
        np.asarray([[0.15, 0.20], [0.25, 0.30], [0.20, 0.25]], dtype=np.float64),
        basis_fingerprint=basis.fingerprint,
    )
    mask = {
        "schema_version": "early_synergy_residual_mask_v1",
        "actuator_names": names,
        "groups": [
            {
                "name": "jump_takeoff",
                "task_phase_selectors": {"jump": [0]},
                "allowed_muscle_names": ["muscle_4"],
                "rank": 1,
            },
            {
                "name": "jump_landing",
                "task_phase_selectors": {"jump": [1]},
                "allowed_muscle_names": ["muscle_5"],
                "rank": 1,
            },
            {
                "name": "squat_lowering",
                "task_phase_selectors": {"squat": [0]},
                "allowed_muscle_names": ["muscle_2"],
                "rank": 1,
            },
            {
                "name": "squat_rising",
                "task_phase_selectors": {"squat": [1]},
                "allowed_muscle_names": ["muscle_3"],
                "rank": 1,
            },
        ],
    }
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    return dataset, source, basis, stats, mask_path


def test_fit_structured_residual_builds_bound_train_only_artifact(tmp_path):
    dataset, source, basis, stats, mask = _artifacts(tmp_path)
    report = fit_structured_residual_basis(
        dataset,
        dataset,
        primary_basis_path=basis.path,
        coefficient_statistics_path=stats["path"],
        primitive_source_manifest_path=source.path,
        expected_primitive_source_manifest_fingerprint=source.fingerprint,
        residual_mask_path=mask,
        output_path=tmp_path / "residual",
        config=StructuredResidualFitConfig(
            min_validation_residual_energy_reduction=0.5,
            min_group_validation_residual_energy_reduction=0.5,
        ),
    )

    residual = load_structured_residual_basis(
        report["artifact_path"],
        expected_actuator_names=basis.muscle_names,
        expected_source_basis_fingerprint=basis.fingerprint,
    )
    assert residual.dimension == 4
    assert residual.fit_contract is not None
    assert residual.fit_contract["passed"] is True
    assert residual.fit_contract["fit_scope"] == "train_only_validation_held_out"
    assert residual.fit_contract["projection_solver_parameters"] == {
        "energy_epsilon": 1e-12,
        "solver_max_iterations": 500,
        "solver_tolerance": 1e-10,
    }
    assert np.count_nonzero(residual.allowed_muscle_mask) == 4
    assert (tmp_path / "residual" / "fit_report.json").is_file()

    forged = np.asarray(residual.basis, dtype=np.float64).copy()
    forged[:, 0] *= -1.0
    with pytest.raises(ValueError, match="matrix fingerprint differs"):
        save_structured_residual_basis(
            tmp_path / "forged_residual",
            basis=forged,
            actuator_names=basis.muscle_names,
            source_basis_fingerprint=basis.fingerprint,
            source_description="attempt to reuse passed evidence with another R",
            allowed_muscle_mask=residual.allowed_muscle_mask,
            fit_contract=report["fit_contract"],
        )

    saturated_contract = json.loads(json.dumps(report["fit_contract"]))
    first_group = next(iter(saturated_contract["metrics"]["per_validation_group"].values()))
    first_group["coordinate_saturation_fraction"] = 1.0
    with pytest.raises(ValueError, match="pass flag differs from metrics"):
        _validate_residual_fit_contract(
            saturated_contract,
            source_basis_fingerprint=basis.fingerprint,
            actuator_names=basis.muscle_names,
            allowed_muscle_mask=residual.allowed_muscle_mask,
            basis=residual.basis,
        )


def test_fit_structured_residual_writes_nothing_when_group_gate_fails(tmp_path):
    dataset, source, basis, stats, mask = _artifacts(tmp_path)
    output = tmp_path / "rejected_residual"
    with pytest.raises(ValueError, match="held-out gates failed"):
        fit_structured_residual_basis(
            dataset,
            dataset,
            primary_basis_path=basis.path,
            coefficient_statistics_path=stats["path"],
            primitive_source_manifest_path=source.path,
            expected_primitive_source_manifest_fingerprint=source.fingerprint,
            residual_mask_path=mask,
            output_path=output,
            config=StructuredResidualFitConfig(
                min_group_validation_residual_energy_reduction=0.95,
            ),
        )
    assert not output.exists()


def test_residual_metrics_match_runtime_clip_after_primary_plus_correction():
    metrics = _residual_metrics(
        np.asarray([[0.9]], dtype=np.float64),
        np.asarray([[1.2]], dtype=np.float64),
        np.asarray([[1.0]], dtype=np.float64),
        alpha=0.3,
        config=StructuredResidualFitConfig(alpha=0.3),
    )

    assert metrics["primary_residual_energy"] == pytest.approx(0.01)
    assert metrics["augmented_residual_energy"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["residual_energy_reduction"] == pytest.approx(1.0)
    # Correcting 1.2 down to 0.9 needs the full -alpha coordinate.  Solving
    # against clip(1.2)=1.0 would incorrectly report the same improvement with
    # only -0.1, even though runtime clips only after Wc + R rho.
    assert metrics["coordinate_saturation_fraction"] == pytest.approx(1.0)


def test_residual_metrics_do_not_treat_zero_demand_as_perfect_improvement():
    metrics = _residual_metrics(
        np.zeros((2, 1), dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        alpha=0.03,
        config=StructuredResidualFitConfig(),
    )

    assert metrics["primary_residual_energy"] == 0.0
    assert metrics["augmented_residual_energy"] == 0.0
    assert metrics["residual_energy_reduction"] == 0.0
