from __future__ import annotations

import numpy as np
import pytest


def test_effective_dimension_reports_participation_ratio_and_jacobian_rank():
    from analysis.latent_synergy.effective_dimension import effective_dimension_report

    z = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    jacobian = np.repeat(np.eye(2)[None, :, :], len(z), axis=0)
    report = effective_dimension_report(z, decoder_jacobians=jacobian)
    assert report["configured_dimension"] == 2
    assert report["participation_ratio_dimension"] == pytest.approx(2.0)
    assert report["jacobian_effective_rank_mean"] == pytest.approx(2.0)


def test_jacobian_alignment_is_one_for_matching_span_and_phase_conditioned():
    from analysis.latent_synergy.jacobian_alignment import jacobian_alignment_report

    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    jacobians = np.repeat(basis[None, :, :], 6, axis=0)
    report = jacobian_alignment_report(jacobians, basis, phase_ids=np.arange(6), require_all_phases=True)
    assert report["projection_score_mean"] == pytest.approx(1.0)
    assert set(report["by_phase"]) == {
        "ready",
        "backswing",
        "acceleration",
        "impact",
        "followthrough",
        "recovery",
    }


def test_representation_report_recovers_linear_latent_to_synergy_mapping():
    from analysis.latent_synergy.representation_similarity import representation_report

    rng = np.random.default_rng(4)
    z = rng.normal(size=(80, 3))
    coefficients = z @ np.array([[1.0, -0.5], [0.25, 0.75], [-0.5, 0.2]]) + 0.001 * rng.normal(size=(80, 2))
    train_mask = np.zeros(80, dtype=bool)
    train_mask[:60] = True
    report = representation_report(z, coefficients, train_mask=train_mask)
    assert report["ridge"]["r2"] > 0.99
    assert report["mean_canonical_correlation"] > 0.99
    assert report["linear_cka"] > 0.5


def test_intervention_builder_and_summary_are_shape_safe():
    from analysis.latent_synergy.intervention import (
        build_intervention_latents,
        summarize_intervention_effects,
    )

    z = np.zeros((5, 2))
    directions = np.eye(2)
    perturbed_z = build_intervention_latents(z, directions, epsilons=(-1.0, 1.0))
    assert perturbed_z.shape == (5, 2, 2, 2)
    baseline = np.zeros((5, 3))
    perturbed = np.ones((5, 2, 2, 3))
    report = summarize_intervention_effects(
        {"physical_excitation": baseline},
        {"physical_excitation": perturbed},
        epsilons=(-1.0, 1.0),
        require_metrics=("physical_excitation",),
    )
    assert report["metrics"]["physical_excitation"]["direction_0"]["1.0"]["delta_rms_mean"] == pytest.approx(1.0)


def test_cross_seed_similarity_is_rotation_invariant():
    from analysis.latent_synergy.cross_seed import cross_seed_report

    rng = np.random.default_rng(8)
    x = rng.normal(size=(50, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    report = cross_seed_report({"0": x, "1": x @ q})
    assert report["linear_cka_mean"] == pytest.approx(1.0, abs=1e-10)
    assert report["procrustes_similarity_mean"] == pytest.approx(1.0, abs=1e-10)


def test_dimension_sweep_requires_and_propagates_basis_fingerprint(tmp_path):
    from analysis.latent_synergy.dimension_sweep import build_sweep_specs

    with pytest.raises(ValueError, match="expected_fingerprint"):
        build_sweep_specs(
            base_config="base.yaml",
            output_root=tmp_path,
            dimensions=(2,),
            decoder_types=("fixed_synergy",),
            seeds=(0,),
            synergy_basis_path="basis",
            heldout_motion_paths=("m0", "m1", "m2", "m3", "m4"),
        )
    specs = build_sweep_specs(
        base_config="base.yaml",
        output_root=tmp_path,
        dimensions=(2,),
        decoder_types=("fixed_synergy",),
        seeds=(0,),
        dataset_dir="train",
        val_dataset_dir="val",
        teacher_ckpt="teacher",
        teacher_promotion_manifest="teacher_promotion.json",
        direct_bc_metrics="bc.json",
        direct_rollout_metrics="rollout.json",
        direct_promotion_evidence="direct_promotion.json",
        heldout_motion_paths=("m0", "m1", "m2", "m3", "m4"),
        synergy_basis_path="basis",
        synergy_basis_expected_fingerprint="a" * 64,
        frozen_body_decoder_path="frozen_decoder",
        frozen_body_decoder_expected_fingerprint="b" * 64,
        body_synergy_contract_expected_fingerprint="c" * 64,
        body_synergy_portable_core_expected_fingerprint="d" * 64,
        closed_loop_correction_root=tmp_path / "corrections",
        require_causal_interventions=True,
    )
    assert "--frozen_body_decoder_expected_fingerprint" in specs[0]["command"]
    assert specs[0]["synergy_basis_expected_fingerprint"] == "a" * 64
    assert specs[0]["frozen_body_decoder_fingerprint"] == "b" * 64
    assert "--teacher_ckpt" in specs[0]["training_command"]
    assert "--teacher_promotion_manifest" in specs[0]["training_command"]
    assert "--direct_bc_metrics" in specs[0]["training_command"]
    assert "--motion_path" in specs[0]["closed_loop_command"]
    assert specs[0]["closed_loop_command"][
        specs[0]["closed_loop_command"].index("--motion_path") + 1 : specs[0]["closed_loop_command"].index(
            "--motion_path"
        )
        + 6
    ] == ["m0", "m1", "m2", "m3", "m4"]
    assert (
        specs[0]["analysis_export_command"][specs[0]["analysis_export_command"].index("--val-dataset-dir") + 1] == "val"
    )
    assert "--require-causal-interventions" not in specs[0]["analysis_export_command"]
    assert "--causal-interventions-npz" not in specs[0]["analysis_export_command"]
    assert "--require-causal-interventions" in specs[0]["causal_finalize_command"]
    assert "--causal-interventions-npz" in specs[0]["causal_finalize_command"]
    assert "--closed-loop-correction-dataset-dir" in specs[0]["training_command"]
    assert specs[0]["closed_loop_correction_dataset_dir"].endswith("corrections/d2_fixed_synergy_seed0/dataset")


def test_explicit_execute_runs_registered_full_lifecycle(monkeypatch, tmp_path):
    import argparse

    import musclemimic.badminton.scripts.latent_synergy_sweep as sweep

    checkpoint = tmp_path / "run" / "latent_checkpoint"
    job = {
        "run_name": "d2_direct_seed0",
        "checkpoint_dir": str(checkpoint),
        "training_command": ["python", "train"],
        "closed_loop_command": ["python", "closed-loop"],
        "analysis_export_command": ["python", "export"],
    }
    monkeypatch.setattr(
        sweep,
        "_load_and_validate_plan",
        lambda _output_dir: {"jobs": [job]},
    )
    calls = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = argparse.Namespace(
        output_dir=tmp_path,
        run_name=[],
        stage="full",
    )
    assert sweep._execute(args) == 0
    assert calls == [
        job["training_command"],
        job["closed_loop_command"],
        job["analysis_export_command"],
    ]


def test_causal_evaluate_runs_driver_then_artifact_with_plan_bound_adapter_config(
    monkeypatch,
    tmp_path,
):
    import argparse
    import hashlib
    import json

    import musclemimic.badminton.scripts.latent_synergy_sweep as sweep

    run_dir = tmp_path / "runs" / "d2_direct_seed0"
    checkpoint = run_dir / "latent_checkpoint"
    checkpoint.mkdir(parents=True)
    analysis_inputs = run_dir / "analysis_inputs.npz"
    analysis_inputs.write_bytes(b"bootstrap")
    analysis_manifest = run_dir / "analysis_inputs.json"
    bootstrap_sidecar = {
        "schema_version": "latent_synergy_analysis_inputs_v2",
        "npz_sha256": hashlib.sha256(b"bootstrap").hexdigest(),
        "checkpoint_dir": str(checkpoint.resolve()),
        "causal_evidence": {"causal_rollout_verified": False},
    }
    bootstrap_sidecar["manifest_fingerprint"] = sweep._canonical_json_sha256(bootstrap_sidecar)
    analysis_manifest.write_text(
        json.dumps(bootstrap_sidecar) + "\n",
        encoding="utf-8",
    )
    job = {
        "run_name": "d2_direct_seed0",
        "output_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint),
        "causal_interventions_npz": str(run_dir / "causal_interventions.npz"),
        "causal_interventions_manifest": str(run_dir / "causal_interventions.json"),
    }
    plan = {
        "plan_fingerprint": "b" * 64,
        "analysis_contract": {"causal_interventions_required": True},
        "lifecycle_inputs": {
            "teacher_checkpoint": str(tmp_path / "teacher"),
            "train_dataset_dir": str(tmp_path / "train"),
            "validation_dataset_dir": str(tmp_path / "val"),
        },
        "jobs": [job],
    }
    monkeypatch.setattr(sweep, "_load_and_validate_plan", lambda _output_dir: plan)
    shared_path = tmp_path / "adapter.json"
    shared_path.write_text(
        json.dumps(
            {
                "schema_version": "latent_causal_adapter_shared_config_v1",
                "adapter_config": {"rollout_horizon_steps": 120},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command)
        if any("causal_rollout_driver" in item for item in command):
            rollout_dir = run_dir / "causal_rollouts"
            rollout_dir.mkdir()
            (rollout_dir / "baseline_records.npz").write_bytes(b"baseline")
            (rollout_dir / "perturbed_records.npz").write_bytes(b"perturbed")
            source = {"schema_version": "paired", "evidence_kind": "environment_rollout"}
            source["manifest_fingerprint"] = sweep._canonical_json_sha256(source)
            (rollout_dir / "paired_rollout_manifest.json").write_text(
                json.dumps(source) + "\n",
                encoding="utf-8",
            )
        else:
            (run_dir / "causal_interventions.npz").write_bytes(b"sealed")
            (run_dir / "causal_interventions.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)

    def fake_validate(npz_path, manifest_path):
        assert npz_path == run_dir / "causal_interventions.npz"
        assert manifest_path == run_dir / "causal_interventions.json"
        source = json.loads((run_dir / "causal_rollouts" / "paired_rollout_manifest.json").read_text(encoding="utf-8"))
        return {
            "analysis_inputs_sha256": hashlib.sha256(b"bootstrap").hexdigest(),
            "analysis_manifest_fingerprint": bootstrap_sidecar["manifest_fingerprint"],
            "paired_rollout_source_manifest_fingerprint": source["manifest_fingerprint"],
            "baseline_records_sha256": hashlib.sha256(b"baseline").hexdigest(),
            "perturbed_records_sha256": hashlib.sha256(b"perturbed").hexdigest(),
            "manifest_fingerprint": "c" * 64,
            "checkpoint_fingerprint": "d" * 64,
            "environment_fingerprint": "e" * 64,
            "policy_abi_hash": "f" * 64,
            "outcome_availability": {
                "muscle_excitation": True,
                "muscle_activation": True,
                "joint_position": True,
                "joint_velocity": True,
                "trunk_state": True,
                "racket_state": True,
                "impact_outcome": False,
                "landing_outcome": False,
            },
        }

    monkeypatch.setattr(sweep, "validate_causal_rollout_artifact", fake_validate)
    args = argparse.Namespace(
        output_dir=tmp_path / "runs",
        shared_config=shared_path,
        run_name=[],
    )
    assert sweep._causal_evaluate(args) == 0
    assert len(calls) == 2
    assert "musclemimic.latent_muscle.causal_rollout_driver" in calls[0]
    assert "musclemimic.latent_muscle.causal_rollout_artifact" in calls[1]
    generated = json.loads((run_dir / "causal_rollout_job.json").read_text(encoding="utf-8"))
    config = generated["adapter_config"]
    assert generated["adapter_import"] == ("musclemimic.latent_muscle.stage2_causal_adapter:create_adapter")
    assert config["latent_checkpoint"] == str(checkpoint.resolve())
    assert config["analysis_inputs"] == str(analysis_inputs.resolve())
    assert config["teacher_ckpt"] == str((tmp_path / "teacher").resolve())
    assert config["rollout_horizon_steps"] == 120
    batch_manifest = json.loads((tmp_path / "runs" / "causal_evaluation_manifest.json").read_text(encoding="utf-8"))
    assert batch_manifest["runs"][0]["stage2_diagnostic_outcomes_complete"] is True
    assert batch_manifest["runs"][0]["task_outcomes_complete"] is False


def test_causal_shared_config_rejects_replay_and_plan_bound_overrides(tmp_path):
    import json

    import musclemimic.badminton.scripts.latent_synergy_sweep as sweep

    path = tmp_path / "shared.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "latent_causal_adapter_shared_config_v1",
                "adapter_import": "replay-record",
                "adapter_config": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="never replay-record"):
        sweep._load_causal_adapter_shared_config(path)
    path.write_text(
        json.dumps(
            {
                "schema_version": "latent_causal_adapter_shared_config_v1",
                "adapter_config": {"latent_checkpoint": "wrong"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="plan-bound"):
        sweep._load_causal_adapter_shared_config(path)


def test_plan_parser_has_no_implicit_execution_flag():
    from musclemimic.badminton.scripts.latent_synergy_sweep import build_parser

    parser = build_parser()
    plan_parser = next(
        action.choices["plan"]
        for action in parser._actions
        if getattr(action, "choices", None) and "plan" in action.choices
    )
    assert "execute" not in {action.dest for action in plan_parser._actions}


def test_plan_accepts_primary_hybrid_with_matching_frozen_contract(
    monkeypatch,
    tmp_path,
):
    import json

    from musclemimic.badminton.scripts import latent_synergy_sweep as sweep
    from musclemimic.distill.action_schema import actuator_schema_hash
    from musclemimic.synergy.basis_artifact import save_synergy_basis
    from musclemimic.synergy.frozen_decoder import (
        FrozenBodyDecoder,
        build_frozen_body_decoder_execution_binding,
    )
    from musclemimic.synergy.hybrid_basis import HYBRID_BASIS_SCHEMA_VERSION
    from musclemimic.synergy.multistage_contract import BodySynergyContractV2
    from musclemimic.synergy.schema import ctrlrange_schema_hash

    names = ("m0", "m1", "m2")
    basis = np.asarray(
        [[0.45, 0.05], [0.15, 0.40], [0.20, 0.25]],
        dtype=np.float32,
    )
    basis_artifact = save_synergy_basis(
        tmp_path / "hybrid",
        basis=basis,
        muscle_names=names,
        manifest={
            "signal_kind": "physical_excitation_unit",
            "region": "hybrid_global_regional",
            "rank": 2,
            "normalization": {"kind": "none"},
            "source_dataset_fingerprint": "hybrid-source",
            "teacher_checkpoint_fingerprint": "1" * 64,
            "fit_seed": 0,
            "transform": {"kind": "ctrlrange_affine_to_unit"},
            "split_provenance": {"train": {}, "validation": {}},
            "train_motion_uids": [1],
            "artifact_role": "primary_hybrid_global_regional",
            "hybrid_schema_version": HYBRID_BASIS_SCHEMA_VERSION,
        },
    )
    bounds = np.asarray([[0.0, 1.0]] * len(names), dtype=np.float32)
    coefficient_maximum = np.asarray([0.5, 0.7], dtype=np.float32)
    coefficient_center = np.asarray([0.1, 0.2], dtype=np.float32)
    coefficient_temperature = np.asarray([0.8, 1.2], dtype=np.float32)
    tonic_baseline = np.asarray([0.02, 0.03, 0.01], dtype=np.float32)
    residual_basis = np.zeros((len(names), 0), dtype=np.float32)
    control_hash = ctrlrange_schema_hash(names, bounds)
    execution_binding = build_frozen_body_decoder_execution_binding(
        mode="fixed_synergy",
        actuator_names=names,
        residual_alpha=0.0,
        basis=basis,
        excitation_bounds=bounds,
        coefficient_maximum=coefficient_maximum,
        coefficient_center=coefficient_center,
        coefficient_temperature=coefficient_temperature,
        tonic_baseline=tonic_baseline,
        residual_basis=residual_basis,
        basis_fingerprint=basis_artifact.fingerprint,
        runtime_basis_fingerprint=basis_artifact.fingerprint,
        coefficient_transform_fingerprint="4" * 64,
        coefficient_statistics_fingerprint="5" * 64,
        tonic_baseline_fingerprint="6" * 64,
        residual_basis_fingerprint=None,
        residual_fit_contract_fingerprint=None,
        residual_allowed_muscle_mask_fingerprint=None,
    )
    contract = BodySynergyContractV2(
        mode="fixed_synergy",
        body_action_dim=len(names),
        policy_action_dim=2,
        actuator_names=names,
        actuator_schema_hash=actuator_schema_hash(names),
        control_range_hash=control_hash,
        runtime_control_range_hash=control_hash,
        runtime_model_hash="2" * 64,
        physical_action_interface_hash="3" * 64,
        basis_fingerprint=basis_artifact.fingerprint,
        runtime_basis_fingerprint=basis_artifact.fingerprint,
        basis_rank=2,
        coefficient_transform_fingerprint="4" * 64,
        coefficient_statistics_fingerprint="5" * 64,
        tonic_baseline_fingerprint="6" * 64,
        source_binding_json=json.dumps(
            {"frozen_body_decoder_execution_binding": execution_binding},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    frozen = FrozenBodyDecoder(
        body_synergy_contract=contract,
        basis=basis,
        excitation_bounds=bounds,
        coefficient_maximum=coefficient_maximum,
        coefficient_center=coefficient_center,
        coefficient_temperature=coefficient_temperature,
        tonic_baseline=tonic_baseline,
        residual_basis=residual_basis,
    )
    frozen_path = frozen.save(tmp_path / "frozen")
    monkeypatch.setattr(
        sweep,
        "validate_dataset_manifest",
        lambda _path: {
            "collections": [
                {
                    "contract": {
                        "split": "val",
                        "motion_paths": [f"heldout_{index}" for index in range(5)],
                    }
                }
            ]
        },
    )
    args = sweep.build_parser().parse_args(
        [
            "plan",
            "--dataset-dir",
            str(tmp_path / "train"),
            "--val-dataset-dir",
            str(tmp_path / "val"),
            "--teacher-ckpt",
            str(tmp_path / "teacher"),
            "--teacher-promotion-manifest",
            str(tmp_path / "teacher_promotion.json"),
            "--direct-bc-metrics",
            str(tmp_path / "bc.json"),
            "--direct-rollout-metrics",
            str(tmp_path / "rollout.json"),
            "--direct-promotion-evidence",
            str(tmp_path / "promotion.json"),
            "--synergy-basis",
            str(basis_artifact.path),
            "--synergy-basis-fingerprint",
            basis_artifact.fingerprint,
            "--frozen-body-decoder",
            str(frozen_path),
            "--frozen-body-decoder-fingerprint",
            frozen.artifact_fingerprint,
            "--body-synergy-contract-fingerprint",
            contract.contract_fingerprint,
            "--body-synergy-portable-core-fingerprint",
            contract.portable_decoder_core_fingerprint,
            "--output-dir",
            str(tmp_path / "sweep"),
            "--base-config",
            str(tmp_path / "latent.yaml"),
            "--dimensions",
            "2",
            "--seeds",
            "0",
            "--decoder-types",
            "fixed_synergy",
        ]
    )

    assert sweep._plan(args) == 0
    plan = json.loads(
        (tmp_path / "sweep" / "sweep_plan.json").read_text(encoding="utf-8")
    )
    assert plan["synergy_basis_fingerprint"] == basis_artifact.fingerprint
    assert plan["frozen_body_decoder_fingerprint"] == frozen.artifact_fingerprint
    assert plan["body_synergy_contract_fingerprint"] == contract.contract_fingerprint
    assert plan["body_synergy_portable_core_fingerprint"] == (
        contract.portable_decoder_core_fingerprint
    )


def test_plan_still_rejects_one_regional_component(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from musclemimic.badminton.scripts import latent_synergy_sweep as sweep

    monkeypatch.setattr(
        sweep,
        "load_synergy_basis",
        lambda _path: SimpleNamespace(
            manifest={
                "signal_kind": "physical_excitation_unit",
                "region": "upper_limb",
            },
            fingerprint="a" * 64,
        ),
    )
    args = SimpleNamespace(synergy_basis=tmp_path / "single")

    with pytest.raises(ValueError, match="never one regional component"):
        sweep._plan(args)


def test_analysis_functions_fail_closed_on_empty_inputs():
    from analysis.latent_synergy.effective_dimension import effective_dimension_report
    from analysis.latent_synergy.intervention import summarize_intervention_effects

    with pytest.raises(ValueError, match="rank-2"):
        effective_dimension_report(np.empty((0, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        summarize_intervention_effects({}, {}, epsilons=(1.0,))


def test_closed_loop_usage_and_phase_interfaces_fail_closed():
    from musclemimic.latent_muscle.closed_loop_eval import (
        _decoder_usage_observation,
        _finalize_decoder_usage,
        _phase_id_from_info,
    )
    from musclemimic.latent_muscle.synergy_decoder import SynergyDecoderOutput

    components = SynergyDecoderOutput(
        action=np.zeros((1, 2)),
        physical_excitation=np.ones((1, 2)),
        synergy_coefficients=np.ones((1, 1)),
        baseline_excitation=np.zeros((1, 2)),
        residual_excitation=np.array([[0.5, 0.0]]),
    )
    observation = _decoder_usage_observation(components, decoder_type="synergy_residual")
    report = _finalize_decoder_usage(observation)
    assert report["residual_energy_ratio"] == pytest.approx(0.125)
    assert _phase_id_from_info({"phase_id": 3}, "phase_id") == 3
    with pytest.raises(ValueError, match="missing required phase"):
        _phase_id_from_info({}, "phase_id")


def test_analysis_inputs_v2_runs_without_optional_causal_and_can_require_it(tmp_path):
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _analyze_run_inputs,
    )

    rng = np.random.default_rng(21)
    sample_count = 60
    latent_dim = 3
    action_dim = 4
    synergy_dim = 2
    direction_count = 2
    epsilons = np.array([-0.5, 0.5], dtype=np.float32)
    basis = np.abs(rng.normal(size=(action_dim, synergy_dim)))
    coefficients = np.abs(rng.normal(size=(sample_count, synergy_dim)))
    baseline = rng.uniform(size=(sample_count, action_dim))
    path = tmp_path / "analysis_inputs.npz"
    np.savez_compressed(
        path,
        latents=rng.normal(size=(sample_count, latent_dim)),
        synergy_coefficients=coefficients,
        target_synergy_coefficients=coefficients,
        decoder_jacobians=rng.normal(size=(sample_count, action_dim, latent_dim)),
        phase_ids=np.tile(np.arange(6), 10),
        train_mask=np.arange(sample_count) < 42,
        sample_uids=np.asarray([f"uid-{index:03d}" for index in range(sample_count)]),
        teacher_physical_excitation=rng.uniform(size=(sample_count, action_dim)),
        baseline_physical_excitation=baseline,
        perturbed_physical_excitation=(
            baseline[:, None, None, :]
            + rng.normal(
                scale=0.01,
                size=(sample_count, direction_count, len(epsilons), action_dim),
            )
        ),
        intervention_epsilons=epsilons,
        intervention_directions=np.eye(latent_dim)[:direction_count],
        intervention_direction_names=np.asarray(["pc0", "pc1"]),
    )
    report, cross = _analyze_run_inputs(
        path,
        basis=basis,
        checkpoint_fingerprint="c" * 64,
        synergy_basis_fingerprint="b" * 64,
        max_samples=sample_count,
        require_all_phases=True,
        require_causal=False,
    )
    assert report["causal_evidence_status"] == "not_provided_optional"
    assert report["offline_intervention_verified"] is True
    assert report["causal_rollout_verified"] is False
    assert cross["causal_effects"] is None
    with pytest.raises(ValueError, match="causal_effects"):
        _analyze_run_inputs(
            path,
            basis=basis,
            checkpoint_fingerprint="c" * 64,
            synergy_basis_fingerprint="b" * 64,
            max_samples=sample_count,
            require_all_phases=True,
            require_causal=True,
        )


def _closed_loop_residual_stub(checkpoint_fingerprint, success, residual=0.0):
    phase = {"residual_energy_ratio": residual}
    by_lambda = {
        key: {
            "residual_energy_ratio": residual,
            "by_phase": {"ready": dict(phase), "recovery": dict(phase)},
        }
        for key in ("lambda_0p000", "lambda_0p250", "lambda_0p500")
    }
    return {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "prior_mean_no_fall_rate": success,
        "promotion": {"passed": True},
        "by_lambda": by_lambda,
    }


def test_promotion_uses_complete_group_aggregate_and_fixed_deployment_seed():
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _select_promotion_model,
    )

    basis_fingerprint = "b" * 64
    jobs = [
        {
            "latent_dim": 4,
            "decoder_type": "fixed_synergy",
            "seed": seed,
        }
        for seed in (0, 1, 2)
    ]
    plan = {
        "jobs": jobs,
        "synergy_basis_fingerprint": basis_fingerprint,
    }
    records = []
    for seed, success in enumerate((0.91, 0.99, 0.95)):
        checkpoint_fingerprint = str(seed) * 64
        records.append(
            {
                "run_name": f"d4_fixed_synergy_seed{seed}",
                "latent_dim": 4,
                "decoder_type": "fixed_synergy",
                "seed": seed,
                "checkpoint_dir": f"/checkpoint/{seed}",
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "dataset_fingerprint": "d" * 64,
                "validation_dataset_fingerprint": "v" * 64,
                "motion_split_fingerprint": "s" * 64,
                "runtime_synergy_basis_fingerprint": "r" * 64,
                "runtime_synergy_basis_source_fingerprint": basis_fingerprint,
                "synergy_basis_expected_fingerprint": basis_fingerprint,
                "basis_binding_verified": True,
                "analysis_complete": True,
                "metrics": {
                    "num_eval_samples": 100,
                    "physical_excitation_mse": 0.01 + 0.001 * seed,
                    "residual_energy_ratio": 0.0,
                    "residual_energy_ratio_ready": 0.0,
                    "residual_energy_ratio_recovery": 0.0,
                },
                "closed_loop": _closed_loop_residual_stub(checkpoint_fingerprint, success),
                "analysis": {
                    "jacobian_alignment": {"projection_score_mean": 0.8},
                    "intervention": {"num_samples": 100},
                },
            }
        )
    cross = [
        {
            "latent_dim": 4,
            "decoder_type": "fixed_synergy",
            "seed_set": ["0", "1", "2"],
            "report": {
                "num_pairs": 3,
                "linear_cka_mean": 0.7,
                "jacobian_projection_score_mean": 0.8,
            },
        }
    ]
    promotion = _select_promotion_model(
        records,
        plan,
        cross_seed_analysis=cross,
        failures=[],
    )
    assert promotion["schema_version"] == "latent_synergy_promotion_metrics_v2"
    assert promotion["selected_model"]["seed"] == 0
    assert promotion["selected_group"]["closed_loop_success_rate_mean"] == pytest.approx(0.95)
    assert promotion["full_matrix_complete"] == 1.0
    assert promotion["offline_intervention_verified"] == 1.0
    assert promotion["causal_rollout_verified"] == 0.0
    assert promotion["intervention_evidence_verified"] == 0.0
    with pytest.raises(ValueError, match="causal environment rollouts"):
        _select_promotion_model(
            records,
            plan,
            cross_seed_analysis=cross,
            failures=[],
            require_causal_interventions=True,
        )
    with pytest.raises(ValueError, match="exact complete"):
        _select_promotion_model(
            records[:-1],
            plan,
            cross_seed_analysis=cross,
            failures=[],
        )


def test_promotion_selects_direct_and_synergy_families_independently():
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _select_promotion_model,
    )

    basis = "b" * 64
    jobs = [
        {"latent_dim": 4, "decoder_type": decoder, "seed": seed}
        for decoder in ("direct", "fixed_synergy")
        for seed in (0, 1)
    ]
    records = []
    for decoder in ("direct", "fixed_synergy"):
        for seed in (0, 1):
            checkpoint = ("d" if decoder == "direct" else "s") + str(seed) * 63
            records.append(
                {
                    "run_name": f"d4_{decoder}_seed{seed}",
                    "latent_dim": 4,
                    "decoder_type": decoder,
                    "seed": seed,
                    "checkpoint_dir": f"/{decoder}/{seed}",
                    "checkpoint_fingerprint": checkpoint,
                    "dataset_fingerprint": "x" * 64,
                    "validation_dataset_fingerprint": "v" * 64,
                    "motion_split_fingerprint": "m" * 64,
                    "runtime_synergy_basis_fingerprint": (None if decoder == "direct" else "r" * 64),
                    "runtime_synergy_basis_source_fingerprint": (None if decoder == "direct" else basis),
                    "synergy_basis_expected_fingerprint": basis,
                    "basis_binding_verified": True,
                    "analysis_complete": True,
                    "metrics": {
                        "num_eval_samples": 20,
                        "physical_excitation_mse": (0.02 if decoder == "direct" else 0.01),
                        "residual_energy_ratio": 0.0,
                        "residual_energy_ratio_ready": 0.0,
                        "residual_energy_ratio_recovery": 0.0,
                    },
                    "closed_loop": _closed_loop_residual_stub(
                        checkpoint,
                        0.90 if decoder == "direct" else 0.95,
                    ),
                    "analysis": {
                        "jacobian_alignment": {"projection_score_mean": 0.75},
                        "intervention": {"num_samples": 20},
                        "offline_intervention_verified": True,
                        "causal_rollout_verified": False,
                    },
                }
            )
    cross = [
        {
            "latent_dim": 4,
            "decoder_type": decoder,
            "seed_set": ["0", "1"],
            "offline_intervention_verified": True,
            "causal_rollout_verified": False,
            "report": {
                "num_pairs": 1,
                "linear_cka_mean": 0.8,
                "jacobian_projection_score_mean": 0.7,
            },
        }
        for decoder in ("direct", "fixed_synergy")
    ]
    result = _select_promotion_model(
        records,
        {"jobs": jobs, "synergy_basis_fingerprint": basis},
        cross_seed_analysis=cross,
        failures=[],
    )
    assert set(result["selected_models"]) == {"best_direct", "best_synergy"}
    assert result["selected_models"]["best_direct"]["decoder_type"] == "direct"
    assert result["selected_models"]["best_synergy"]["decoder_type"] == "fixed_synergy"
    assert result["selected_model"] == result["selected_models"]["best_synergy"]


def test_explicit_validation_split_is_seed_independent(monkeypatch, tmp_path):
    import importlib

    module = importlib.import_module("musclemimic.latent_muscle.train_latent")
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig

    class FakeDataset:
        def __init__(self, path, split, **_kwargs):
            is_validation = split == "val"
            motion_ids = np.repeat(np.arange(100, 105), 2) if is_validation else np.arange(22)
            self.arrays = {"motion_uid": motion_ids}
            self.num_samples = len(motion_ids)
            self.student_obs_dim = 3
            self.reference_features_dim = 2
            self.action_dim = 4
            self.actuator_names = ["a", "b", "c", "d"]
            self.action_schema_hash = "schema"
            self.actuator_ctrlrange = np.tile([[-1.0, 1.0]], (4, 1))
            self.metadata = {
                "student_state_schema_hash": "state",
                "body_obs_schema_hash": "body",
                "student_obs_filter": {"kind": "body"},
                "physical_signal_semantics": {"kind": "unit"},
            }
            self.split = split

    monkeypatch.setattr(module, "SequenceDistillDataset", FakeDataset)
    manifests = (
        {"manifest_fingerprint": "t" * 64},
        {"manifest_fingerprint": "v" * 64},
    )
    split_fingerprints = []
    for seed in (0, 99):
        config = LatentTrainConfig(
            dataset_dir=str(tmp_path / "train"),
            val_dataset_dir=str(tmp_path / "val"),
            expected_val_motion_count=5,
            output_dir=str(tmp_path / f"out-{seed}"),
            val_fraction=0.0,
            seed=seed,
            strict_motion_identity=True,
        )
        _train, _validation, split = module._load_latent_train_validation_datasets(
            config,
            motion_field="motion_uid",
            target_body_names=None,
            dataset_manifest=manifests[0],
            validation_dataset_manifest=manifests[1],
        )
        assert split["split_seed"] is None
        assert split["val_motion_ids"] == [100, 101, 102, 103, 104]
        split_fingerprints.append(split["split_fingerprint"])
    assert split_fingerprints[0] == split_fingerprints[1]


def test_latent_closed_loop_correction_contract_requires_latent_student_rollout():
    from copy import deepcopy

    from musclemimic.latent_muscle.train_latent import (
        _validate_closed_loop_correction_manifest,
    )

    teacher = "a" * 64
    student = "c" * 64
    manifest = {
        "collections": [
            {
                "num_samples": 32,
                "contract": {
                    "schema_version": "distill_collection_contract_v2",
                    "collector": "dagger_student_rollout_teacher_relabel",
                    "split": "train",
                    "dagger_iteration": 1,
                    "teacher_checkpoint_sha256": teacher,
                    "student_checkpoint_sha256": student,
                    "student_checkpoint": {"sha256": student},
                    "request": {
                        "student_policy_kind": "latent_checkpoint_prior_mean_lab",
                        "teacher_relabel_target": "normalized_body_action",
                        "closed_loop_state_source": "environment_student_visited_state",
                    },
                },
            }
        ]
    }
    _validate_closed_loop_correction_manifest(manifest, expected_teacher_sha256=teacher)
    direct = deepcopy(manifest)
    direct["collections"][0]["contract"]["request"]["student_policy_kind"] = "direct_student"
    with pytest.raises(ValueError, match="student rollouts"):
        _validate_closed_loop_correction_manifest(direct, expected_teacher_sha256=teacher)


def test_runtime_basis_binding_covers_direct_and_synergy_sources():
    from types import SimpleNamespace

    from musclemimic.latent_muscle.analysis_export import (
        validate_runtime_basis_binding,
    )

    formal = "f" * 64
    direct = SimpleNamespace(
        config={"synergy_basis_expected_fingerprint": formal},
        decoder_type="direct",
        synergy_basis=None,
        control_manifest={},
    )
    assert validate_runtime_basis_binding(direct, formal_basis_fingerprint=formal)["verified"]
    runtime_fingerprint = "r" * 64
    synergy = SimpleNamespace(
        config={
            "synergy_basis_expected_fingerprint": formal,
            "synergy_basis_fingerprint": runtime_fingerprint,
        },
        decoder_type="fixed_synergy",
        synergy_basis=SimpleNamespace(
            fingerprint=runtime_fingerprint,
            manifest={"source_fingerprint": formal},
        ),
        control_manifest={"synergy_basis_fingerprint": runtime_fingerprint},
    )
    binding = validate_runtime_basis_binding(synergy, formal_basis_fingerprint=formal)
    assert binding["runtime_synergy_basis_source_fingerprint"] == formal
    synergy.control_manifest["synergy_basis_fingerprint"] = "x" * 64
    with pytest.raises(ValueError, match="mutually bound"):
        validate_runtime_basis_binding(synergy, formal_basis_fingerprint=formal)


def test_causal_evidence_rejects_non_rollout_artifact(tmp_path):
    import json

    from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
    from musclemimic.latent_muscle.analysis_export import (
        CAUSAL_EVIDENCE_SCHEMA_VERSION,
        load_optional_causal_evidence,
    )

    sample_uids = np.asarray(["a", "b"])
    directions = np.eye(2)
    epsilons = np.asarray([-0.5, 0.5])
    path = tmp_path / "causal.npz"
    np.savez_compressed(
        path,
        causal_effects=np.ones((2, 2, 2, 3)),
        sample_uids=sample_uids,
        intervention_directions=directions,
        intervention_epsilons=epsilons,
    )
    manifest_path = path.with_suffix(".json")
    manifest = {
        "schema_version": CAUSAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "offline_decoder",
        "npz_sha256": file_sha256(path),
        "checkpoint_fingerprint": "c" * 64,
        "synergy_basis_fingerprint": "b" * 64,
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="environment_rollout"):
        load_optional_causal_evidence(
            path,
            manifest_path,
            checkpoint_fingerprint="c" * 64,
            synergy_basis_fingerprint="b" * 64,
            sample_uids=sample_uids,
            directions=directions,
            epsilons=epsilons,
        )


def test_selected_checkpoint_artifact_is_atomic_bound_pointer(tmp_path):
    import json

    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _canonical_json_sha256,
        _materialize_selected_artifact,
        validate_selected_artifact,
    )
    from musclemimic.latent_muscle.checkpoint import latent_checkpoint_fingerprint

    direct = tmp_path / "d4_direct_seed0" / "latent_checkpoint"
    synergy = tmp_path / "d4_fixed_synergy_seed0" / "latent_checkpoint"
    for checkpoint, suffix in ((direct, b"direct"), (synergy, b"synergy")):
        checkpoint.mkdir(parents=True)
        (checkpoint / "prior.msgpack").write_bytes(b"prior-" + suffix)
        (checkpoint / "decoder.msgpack").write_bytes(b"decoder-" + suffix)
    direct_fingerprint = latent_checkpoint_fingerprint(direct)
    synergy_fingerprint = latent_checkpoint_fingerprint(synergy)
    common = {
        "formal_synergy_basis_fingerprint": "b" * 64,
        "dataset_fingerprint": "d" * 64,
        "validation_dataset_fingerprint": "v" * 64,
        "motion_split_fingerprint": "s" * 64,
        "latent_dim": 4,
        "seed": 0,
    }
    promotion = {
        "selection_rule": {"deployment_seed": "smallest"},
        "selected_models": {
            "best_direct": {
                **common,
                "run_name": "d4_direct_seed0",
                "checkpoint_dir": str(direct),
                "checkpoint_fingerprint": direct_fingerprint,
                "runtime_synergy_basis_fingerprint": None,
                "runtime_synergy_basis_source_fingerprint": None,
                "decoder_type": "direct",
            },
            "best_synergy": {
                **common,
                "run_name": "d4_fixed_synergy_seed0",
                "checkpoint_dir": str(synergy),
                "checkpoint_fingerprint": synergy_fingerprint,
                "runtime_synergy_basis_fingerprint": "r" * 64,
                "runtime_synergy_basis_source_fingerprint": "b" * 64,
                "decoder_type": "fixed_synergy",
            },
        },
    }
    promotion["selected_model"] = promotion["selected_models"]["best_synergy"]
    promotion["promotion_metrics_fingerprint"] = _canonical_json_sha256(promotion)
    artifact = _materialize_selected_artifact(
        tmp_path,
        promotion_metrics=promotion,
        plan={
            "plan_fingerprint": "q" * 64,
            "jobs": [
                {"decoder_type": "direct"},
                {"decoder_type": "fixed_synergy"},
            ],
        },
    )
    stable = tmp_path / "selected" / "latent_checkpoint"
    assert stable.is_symlink()
    assert latent_checkpoint_fingerprint(stable) == synergy_fingerprint
    assert (tmp_path / "selected" / "best_direct").is_symlink()
    assert (tmp_path / "selected" / "best_synergy").is_symlink()
    manifest = json.loads((tmp_path / "selected" / "selection_manifest.json").read_text())
    assert manifest["schema_version"] == "latent_synergy_selected_checkpoints_v2"
    assert manifest["checkpoints"]["best_direct"]["checkpoint_fingerprint"] == direct_fingerprint
    assert manifest["checkpoints"]["best_synergy"]["checkpoint_fingerprint"] == synergy_fingerprint
    assert manifest["compatibility_alias"]["target_family"] == "best_synergy"
    assert (
        validate_selected_artifact(tmp_path / "selected" / "selection_manifest.json")["selection_manifest_fingerprint"]
        == manifest["selection_manifest_fingerprint"]
    )
    assert artifact["selection_manifest_fingerprint"] == manifest["selection_manifest_fingerprint"]
