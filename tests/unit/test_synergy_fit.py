import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.synergy.action_interface import (
    build_early_synergy_action_interface,
    save_coefficient_statistics,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis, save_synergy_basis
from musclemimic.synergy.fit import (
    BasisNotEligibleForEarlyControl,
    SynergyFitConfig,
    build_parser,
    fit_synergy_dataset,
    load_synergy_split,
    primitive_task_phase_balanced_weights,
    synergy_phase_weight_fingerprint,
    synergy_preprocessing_fingerprint,
)
from musclemimic.synergy.grouping import load_grouping_json
from musclemimic.synergy.hybrid_basis import HybridBasisConfig
from musclemimic.synergy.primitive_manifest import (
    save_primitive_source_manifest,
    save_primitive_source_manifest_from_splits,
)
from musclemimic.synergy.rank_selection import (
    DYNAMIC_COVERAGE_EVIDENCE_KIND,
    DYNAMIC_COVERAGE_SCHEMA_VERSION,
    dynamic_coverage_artifact_fingerprint,
    dynamic_coverage_requirement,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash

TEACHER_SHA256 = "a" * 64


def _dynamic_report(*, signal_kind, region, rank, candidate_fingerprint):
    report = {
        "schema_version": DYNAMIC_COVERAGE_SCHEMA_VERSION,
        "evidence_kind": DYNAMIC_COVERAGE_EVIDENCE_KIND,
        "signal_kind": signal_kind,
        "region": region,
        "rank": int(rank),
        "candidate_basis_fingerprint": candidate_fingerprint,
        "rollout_manifest_fingerprint": hashlib.sha256(b"rollout").hexdigest(),
        "environment_fingerprint": hashlib.sha256(b"environment").hexdigest(),
        "metrics": {
            "mean_dynamic_gap": 0.10,
            "max_key_phase_dynamic_gap": 0.20,
            "rollout_count": 8,
            "key_phase_count": 3,
            "horizon_steps": 32,
        },
        "thresholds": {
            "max_mean_dynamic_gap": 0.15,
            "max_key_phase_dynamic_gap": 0.25,
        },
        "checks": {
            "mean_dynamic_gap": True,
            "key_phase_dynamic_gap": True,
            "nonempty_rollout_evidence": True,
        },
        "passed": True,
    }
    report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(report)
    return report


def test_hybrid_fit_and_cli_threshold_defaults_match_the_formal_builder():
    hybrid = HybridBasisConfig()
    fit = SynergyFitConfig().validated()
    assert fit.hybrid_novelty_residual_ratio == hybrid.novelty_residual_ratio
    assert fit.hybrid_duplicate_cosine_similarity == hybrid.duplicate_cosine_similarity
    assert (
        fit.hybrid_min_heldout_global_vaf_marginal_gain
        == hybrid.min_heldout_global_vaf_marginal_gain
    )
    assert fit.hybrid_max_total_rank == hybrid.max_total_rank
    assert fit.hybrid_min_heldout_global_vaf == hybrid.min_heldout_global_vaf
    assert fit.hybrid_local_vaf_quantile == hybrid.local_vaf_quantile
    assert (
        fit.hybrid_min_heldout_local_vaf_quantile
        == hybrid.min_heldout_local_vaf_quantile
    )
    assert fit.hybrid_max_basis_condition_number == hybrid.max_basis_condition_number
    assert fit.hybrid_min_effective_rank_fraction == hybrid.min_effective_rank_fraction
    assert (
        fit.hybrid_effective_rank_relative_tolerance
        == hybrid.effective_rank_relative_tolerance
    )
    args = build_parser().parse_args(
        ["--train", "train", "--val", "val", "--output-dir", "output"]
    )
    assert args.hybrid_min_heldout_global_vaf_marginal_gain == hybrid.min_heldout_global_vaf_marginal_gain
    assert args.hybrid_max_total_rank == hybrid.max_total_rank
    assert args.hybrid_max_basis_condition_number == hybrid.max_basis_condition_number


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
    hybrid = load_synergy_basis(excitation_preferred["artifact_path"])
    assert hybrid.manifest["region"] == "hybrid_global_regional"
    assert hybrid.manifest["artifact_role"] == "primary_hybrid_global_regional"
    assert hybrid.manifest["hybrid_construction"]["heldout_evaluation"]["all_passed"] is True
    source_components = hybrid.manifest["source_components"]
    composite = load_synergy_basis(source_components["regional"]["artifact_path"])
    global_source = load_synergy_basis(source_components["global"]["artifact_path"])
    assert source_components["regional"]["artifact_fingerprint"] == composite.fingerprint
    assert source_components["global"]["artifact_fingerprint"] == global_source.fingerprint
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
    regional_rank = composite.basis.shape[1]
    np.testing.assert_array_equal(hybrid.basis[:, :regional_rank], composite.basis)
    selected_global = hybrid.manifest["hybrid_construction"][
        "retained_global_column_indices_in_output_order"
    ]
    np.testing.assert_array_equal(
        hybrid.basis[:, regional_rank:],
        global_source.basis[:, selected_global],
    )
    hybrid_report = next(
        item for item in report["artifacts"] if item["artifact_role"] == "primary_hybrid_global_regional"
    )
    assert Path(hybrid_report["coefficient_statistics_path"]).is_file()
    action_config = {
        "schema_version": "early_synergy_action_v1",
        "mode": "fixed_synergy",
        "basis_path": str(hybrid.path),
        "expected_basis_fingerprint": hybrid.fingerprint,
        "expected_underlying_action_dim": len(names),
        "expected_actuator_schema_hash": actuator_schema_hash(names),
        "expected_basis_region": "hybrid_global_regional",
        "required_hybrid_thresholds": {
            "novelty_residual_ratio_strictly_greater_than": 0.15,
            "duplicate_cosine_similarity_reject_at_or_above": 0.95,
            "heldout_global_vaf_marginal_gain_retain_strictly_greater_than": 1e-6,
            "max_total_rank": 64,
            "min_heldout_global_vaf": 0.90,
            "local_vaf_quantile": 0.10,
            "min_heldout_local_vaf_quantile": 0.70,
            "max_basis_condition_number": 100.0,
            "min_effective_rank_fraction": 0.80,
            "effective_rank_relative_tolerance": 1e-8,
        },
        "require_all_basis_gates": True,
        "forbid_fallback_selected_basis": True,
        "require_coverage_gate": False,
        "coefficient_transform": {
            "kind": "bounded_sigmoid",
            "stats_path": hybrid_report["coefficient_statistics_path"],
            "expected_stats_fingerprint": hybrid_report[
                "coefficient_statistics_fingerprint"
            ],
            "max_source": "train_q99_times_1p2",
            "center_source": "train_q50",
            "temperature": 1.0,
        },
        "tonic_baseline": {"kind": "zero", "learned_full_dimensional": False},
        "residual": {"enabled": False, "alpha": 0.0},
    }
    interface = build_early_synergy_action_interface(
        action_config,
        expected_actuator_names=names,
    )
    assert interface.synergy_dim == hybrid.basis.shape[1]

    wrong_region_config = copy.deepcopy(action_config)
    wrong_region_config["expected_basis_region"] = "regional_composite"
    with pytest.raises(ValueError, match="basis region differs"):
        build_early_synergy_action_interface(
            wrong_region_config,
            expected_actuator_names=names,
        )

    dynamic_required_config = copy.deepcopy(action_config)
    dynamic_required_config["require_hybrid_dynamic_coverage"] = True
    dynamic_required_config["required_hybrid_dynamic_thresholds"] = {
        "max_mean_dynamic_gap": 0.15,
        "max_key_phase_dynamic_gap": 0.25,
    }
    with pytest.raises(ValueError, match="requires hybrid dynamic coverage"):
        build_early_synergy_action_interface(
            dynamic_required_config,
            expected_actuator_names=names,
        )

    missing_dynamic_manifest = dict(hybrid.manifest)
    missing_dynamic_manifest["hybrid_dynamic_coverage"] = {
        **missing_dynamic_manifest["hybrid_dynamic_coverage"],
        "requirement": dynamic_coverage_requirement(
            required=True,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            expected_environment_fingerprint="8" * 64,
            expected_rollout_manifest_fingerprint="9" * 64,
        ),
        "evidence": None,
    }
    missing_dynamic = save_synergy_basis(
        tmp_path / "missing_hybrid_dynamic",
        basis=hybrid.basis,
        muscle_names=hybrid.muscle_names,
        manifest=missing_dynamic_manifest,
    )
    missing_stats = save_coefficient_statistics(
        missing_dynamic.path / "coefficient_stats.npz",
        np.asarray([[0.1] * missing_dynamic.basis.shape[1], [0.2] * missing_dynamic.basis.shape[1]]),
        basis_fingerprint=missing_dynamic.fingerprint,
    )
    missing_config = copy.deepcopy(action_config)
    missing_config["basis_path"] = str(missing_dynamic.path)
    missing_config["expected_basis_fingerprint"] = missing_dynamic.fingerprint
    missing_config["coefficient_transform"]["stats_path"] = missing_stats["path"]
    missing_config["coefficient_transform"]["expected_stats_fingerprint"] = missing_stats[
        "stats_fingerprint"
    ]
    with pytest.raises(ValueError, match="lacks required exact rollout coverage"):
        build_early_synergy_action_interface(
            missing_config,
            expected_actuator_names=names,
        )

    wrong_source_manifest = copy.deepcopy(hybrid.manifest)
    wrong_source_manifest["source_components"]["global"]["artifact_path"] = str(
        composite.path.resolve()
    )
    wrong_source = save_synergy_basis(
        tmp_path / "wrong_hybrid_source",
        basis=hybrid.basis,
        muscle_names=hybrid.muscle_names,
        manifest=wrong_source_manifest,
    )
    wrong_stats = save_coefficient_statistics(
        wrong_source.path / "coefficient_stats.npz",
        np.asarray([[0.1] * wrong_source.basis.shape[1], [0.2] * wrong_source.basis.shape[1]]),
        basis_fingerprint=wrong_source.fingerprint,
    )
    wrong_config = copy.deepcopy(action_config)
    wrong_config["basis_path"] = str(wrong_source.path)
    wrong_config["expected_basis_fingerprint"] = wrong_source.fingerprint
    wrong_config["coefficient_transform"]["stats_path"] = wrong_stats["path"]
    wrong_config["coefficient_transform"]["expected_stats_fingerprint"] = wrong_stats[
        "stats_fingerprint"
    ]
    with pytest.raises(ValueError, match="global source fingerprint mismatch"):
        build_early_synergy_action_interface(
            wrong_config,
            expected_actuator_names=names,
        )
    assert all(
        artifact["selected_metrics"]["validation"]["global_vaf"] > 0.85
        for artifact in report["artifacts"]
        if artifact["artifact_role"] in {"global_comparator", "regional_component"}
    )
    assert (tmp_path / "fit" / "fit_report.json").is_file()
    promotion = json.loads((tmp_path / "fit" / "promotion_metrics.json").read_text(encoding="utf-8"))
    assert promotion["heldout_sample_count"] == 48
    assert promotion["explained_variance"] > 0.85
    assert promotion["artifact_binding_verified"] == 1.0
    assert promotion["basis_artifact_fingerprint"] == hybrid.fingerprint


def test_both_mode_requires_an_exact_hybrid_dynamic_rollout_gate(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
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
    output = tmp_path / "fit"
    environment_fingerprint = hashlib.sha256(b"environment").hexdigest()
    rollout_fingerprint = hashlib.sha256(b"rollout").hexdigest()
    config = SynergyFitConfig(
        ranks=(1, 2),
        seeds=(0, 1),
        max_iter=200,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_val_global_vaf=0.90,
        min_val_local_vaf_quantile=0.60,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
        require_dynamic_coverage=True,
        expected_environment_fingerprint=environment_fingerprint,
        expected_rollout_manifest_fingerprint=rollout_fingerprint,
    )
    fit_kwargs = {
        "output_dir": output,
        "signal_kinds": ("activation",),
        "mode": "both",
        "grouping_json": grouping,
        "config": config,
    }

    with pytest.raises(BasisNotEligibleForEarlyControl, match="dynamic coverage requires"):
        fit_synergy_dataset(dataset, dataset, **fit_kwargs)

    signal_kind = "muscle_activation"
    source_reports: dict[str, dict[str, dict[str, dict]]] = {signal_kind: {}}
    for region in ("whole_body", "lower_body", "upper_body"):
        inventory = json.loads(
            (output / signal_kind / region / "candidate_inventory.json").read_text(encoding="utf-8")
        )
        reports_for_region = {}
        for candidate in inventory["candidates"]:
            if candidate["offline_eligible"]:
                rank = candidate["rank"]
                reports_for_region[str(rank)] = _dynamic_report(
                    signal_kind=signal_kind,
                    region=region,
                    rank=rank,
                    candidate_fingerprint=candidate["candidate_basis_fingerprint"],
                )
        assert reports_for_region
        source_reports[signal_kind][region] = reports_for_region

    with pytest.raises(BasisNotEligibleForEarlyControl, match="dynamic coverage requires"):
        fit_synergy_dataset(
            dataset,
            dataset,
            dynamic_coverage_reports=source_reports,
            **fit_kwargs,
        )

    hybrid_root = output / signal_kind / "hybrid_global_regional"
    hybrid_inventory = json.loads((hybrid_root / "candidate_inventory.json").read_text(encoding="utf-8"))
    assert hybrid_inventory["region"] == "hybrid_global_regional"
    assert len(hybrid_inventory["candidates"]) == 1
    hybrid_candidate = hybrid_inventory["candidates"][0]
    candidate_artifact = load_synergy_basis(
        hybrid_root / hybrid_candidate["candidate_artifact_path"]
    )
    assert candidate_artifact.manifest["artifact_role"] == "dynamic_coverage_rollout_candidate"
    assert candidate_artifact.manifest["hybrid_dynamic_coverage"]["evidence"] is None
    assert not (hybrid_root / "coefficient_stats.npz").exists()

    hybrid_rank = hybrid_candidate["rank"]
    source_reports[signal_kind]["hybrid_global_regional"] = {
        str(hybrid_rank): _dynamic_report(
            signal_kind=signal_kind,
            region="hybrid_global_regional",
            rank=hybrid_rank,
            candidate_fingerprint=hybrid_candidate["candidate_basis_fingerprint"],
        )
    }
    report = fit_synergy_dataset(
        dataset,
        dataset,
        dynamic_coverage_reports=source_reports,
        **fit_kwargs,
    )
    preferred = report["preferred_decoder_artifacts"][signal_kind]
    hybrid = load_synergy_basis(preferred["artifact_path"])
    assert hybrid.manifest["artifact_role"] == "primary_hybrid_global_regional"
    assert hybrid.manifest["hybrid_dynamic_coverage"]["evidence"]["passed"] is True
    assert (hybrid.path / "coefficient_stats.npz").is_file()

    reports_without_hybrid = copy.deepcopy(source_reports)
    reports_without_hybrid[signal_kind].pop("hybrid_global_regional")
    with pytest.raises(BasisNotEligibleForEarlyControl, match="dynamic coverage requires"):
        fit_synergy_dataset(
            dataset,
            dataset,
            dynamic_coverage_reports=reports_without_hybrid,
            **fit_kwargs,
        )
    assert not (hybrid_root / "manifest.json").exists()
    assert not (hybrid_root / "basis.npy").exists()
    assert not (hybrid_root / "coefficient_stats.npz").exists()
    assert not (output / "promotion_metrics.json").exists()
    pending_report = json.loads((output / "fit_report.json").read_text(encoding="utf-8"))
    assert pending_report["status"] == "dynamic_coverage_evidence_required"
    assert "preferred_decoder_artifacts" not in pending_report


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
