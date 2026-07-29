from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    physical_signal_metadata,
)
from musclemimic.synergy.action_interface import save_coefficient_statistics
from musclemimic.synergy.basis_artifact import save_synergy_basis
from musclemimic.synergy.coverage_proxy import (
    build_coverage_proxy,
    write_target_control_qc,
    write_target_control_source_manifest,
)
from musclemimic.synergy.fit import EXCITATION_SIGNAL_KIND
from musclemimic.synergy.oracle_coverage import load_static_proxy_phase_schema
from musclemimic.synergy.schema import ctrlrange_schema_hash
from musclemimic.synergy.stage1_pipeline import (
    PipelineInputError,
    PipelineRequest,
    apply_stage1_pipeline,
    plan_stage1_pipeline,
    preflight_stage1_release,
    write_shell_bindings,
)

NAMES = ("muscle_a", "muscle_b")
CTRLRANGE = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
MODEL_HASH = "1" * 64
CHECKPOINT_HASH = "2" * 64
PHASE_HASH = "3" * 64
SOURCE_MANIFEST_HASH = "4" * 64


def _muscle_contract() -> dict:
    width = len(NAMES)
    return {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(NAMES),
        "actuator_ids": list(range(width)),
        "actuator_dyntype": ["muscle"] * width,
        "actuator_actnum": [1] * width,
        "actuator_actadr": list(range(width)),
        "model_na": width,
    }


def _muscle_contract_arrays() -> dict[str, np.ndarray]:
    width = len(NAMES)
    return {
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_excitation_transform": np.asarray(UNIT_EXCITATION_TRANSFORM),
        "muscle_channel_contract_schema_version": np.asarray(
            MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION
        ),
        "actuator_ids": np.arange(width, dtype=np.int32),
        "actuator_dyntype": np.asarray(["muscle"] * width),
        "actuator_actnum": np.ones(width, dtype=np.int32),
        "actuator_actadr": np.arange(width, dtype=np.int32),
        "model_na": np.asarray(width, dtype=np.int32),
    }


def _basis_transform() -> dict:
    return {
        "kind": UNIT_EXCITATION_TRANSFORM,
        "raw_signal_kind": "applied_ctrl",
        "formula": MUSCLE_EXCITATION_FORMULA,
        "ctrlrange": CTRLRANGE.tolist(),
        "actuator_names": list(NAMES),
        "ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, CTRLRANGE),
        "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "muscle_channel_contract": _muscle_contract(),
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_fixture(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "primitive"
    source.mkdir()
    train_ctrl = np.asarray([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=np.float32)
    val_ctrl = np.asarray([[0.3, 0.3], [0.6, 0.6]], dtype=np.float32)
    for split, ctrl, phases, motion in (
        ("train", train_ctrl, [0, 1, 1], [10, 10, 11]),
        ("val", val_ctrl, [0, 1], [20, 21]),
    ):
        excitation = (ctrl - CTRLRANGE[:, 0]) / (CTRLRANGE[:, 1] - CTRLRANGE[:, 0])
        np.savez_compressed(
            source / f"{split}_000.npz",
            teacher_ctrl_physical=ctrl,
            muscle_excitation=excitation.astype(np.float32),
            phase_id=np.asarray(phases, dtype=np.int32),
            motion_uid=np.asarray(motion, dtype=np.int64),
        )
    metadata = {
        "actuator_names": list(NAMES),
        "actuator_ctrlrange": CTRLRANGE.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": list(NAMES), "ctrlrange": CTRLRANGE.tolist()},
        ),
        "transform_ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, CTRLRANGE),
        "model_hash": MODEL_HASH,
        "source_checkpoint_fingerprints": {"P01": CHECKPOINT_HASH},
        "source_checkpoint_contents": {"P01": {"sha256": CHECKPOINT_HASH}},
        "primitive_required_phase_ids": {"P01": [0, 1]},
        "primitive_phase_schema_fingerprints": {"P01": PHASE_HASH},
        "physical_signal_semantics": physical_signal_metadata(),
    }
    _write_json(source / "metadata.json", metadata)
    grouping = _write_json(
        tmp_path / "groups.json",
        {"regions": {"all": list(NAMES)}},
    )
    phase_schema = _write_json(
        tmp_path / "phase_schema.json",
        {
            "schema_version": "chinajump_coverage_phase_schema_v1",
            "target_skill_id": "ChinaJump",
            "phase_field": "phase_id",
            "producer_contract": "fixture_contact_events_v1",
            "phases": [
                {"id": 0, "name": "prepare", "definition": "Prepare."},
                {"id": 1, "name": "execute", "definition": "Execute."},
            ],
        },
    )
    return {
        "source": source,
        "grouping": grouping,
        "phase_schema": phase_schema,
        "phase_fingerprint": load_static_proxy_phase_schema(phase_schema)["phase_schema_fingerprint"],
    }


def _request(fixture: dict[str, Path | str], output: Path, *, readiness: str) -> PipelineRequest:
    return PipelineRequest(
        train=str(fixture["source"]),
        val=str(fixture["source"]),
        primitive_catalog=None,
        grouping_json=str(fixture["grouping"]),
        coverage_proxy_artifact=None,
        phase_schema=str(fixture["phase_schema"]),
        residual_mask=None,
        output_root=str(output),
        target_skill_id="ChinaJump",
        env_prefix="MUSCLEMIMIC_TEST",
        readiness_mode=readiness,
        with_residual=False,
        formal_config_name="fixture_formal",
        residual_config_name="fixture_residual",
        bootstrap_config_name="fixture_bootstrap",
        ranks=(1,),
        seeds=(0, 1),
        normalization="none",
        near_zero_threshold=1e-8,
        phase_weights_json=None,
    )


def _patch_contract(
    monkeypatch,
    fixture: dict[str, Path | str],
    *,
    required_coverage_phase_ids: tuple[int, ...] = (),
) -> None:
    import musclemimic.synergy.stage1_pipeline as pipeline

    def contract(config_name, *, env_prefix, readiness_mode):
        del env_prefix
        return {
            "config_name": config_name,
            "readiness_mode": readiness_mode,
            "expected_target_skill_id": "ChinaJump",
            "expected_underlying_action_dim": len(NAMES),
            "expected_actuator_schema_hash": actuator_schema_hash(NAMES),
            "expected_excluded_target_motion_paths": ["ChinaJump/target.npz"],
            "selection_thresholds": {},
            "coverage_thresholds": {
                "required_phase_ids": list(required_coverage_phase_ids),
            },
            "phase_schema_fingerprint": fixture["phase_fingerprint"],
            "max_policy_action_dim": 4,
            "config_status": {
                "readiness": "bootstrap_only" if readiness_mode == "bootstrap" else "formal",
            },
        }

    monkeypatch.setattr(pipeline, "_pipeline_config_contract", contract)


def _patch_apply_builders(monkeypatch) -> dict[str, list]:
    import musclemimic.synergy.stage1_pipeline as pipeline

    calls: dict[str, list] = {
        "fit": [],
        "fit_kwargs": [],
        "loaded_basis": [],
        "preflight": [],
    }

    def save_manifest(path, **kwargs):
        del kwargs
        output = _write_json(Path(path), {"fixture": True})
        return SimpleNamespace(
            path=output,
            fingerprint=SOURCE_MANIFEST_HASH,
            manifest={"source_dataset_fingerprint": "fixture-dataset"},
        )

    def fit_dataset(train_source, validation_source, *, output_dir, **kwargs):
        del train_source, validation_source
        calls["fit"].append(str(output_dir))
        calls["fit_kwargs"].append(kwargs)
        output = Path(output_dir)
        basis = save_synergy_basis(
            output / "selected-preferred",
            basis=np.asarray([[1.0], [0.5]], dtype=np.float64),
            muscle_names=NAMES,
            manifest={
                "signal_kind": EXCITATION_SIGNAL_KIND,
                "region": "whole_body",
                "rank": 1,
                "normalization": {"kind": "none"},
                "source_dataset_fingerprint": "fixture-dataset",
                "teacher_checkpoint_fingerprint": CHECKPOINT_HASH,
                "fit_seed": 0,
                "transform": _basis_transform(),
                "split_provenance": {"train": {}, "validation": {}},
                "train_motion_uids": [10, 11],
            },
        )
        save_coefficient_statistics(
            basis.path / "coefficient_stats.npz",
            np.asarray([[0.2], [0.5], [0.8]], dtype=np.float64),
            basis_fingerprint=basis.fingerprint,
        )
        _write_json(output / "fit_report.json", {"fixture": True})
        return {
            "preferred_decoder_artifacts": {
                EXCITATION_SIGNAL_KIND: {
                    "artifact_path": str(basis.path),
                    "artifact_fingerprint": basis.fingerprint,
                }
            }
        }

    real_load = pipeline.load_synergy_basis

    def load_basis(path):
        calls["loaded_basis"].append(str(path))
        return real_load(path)

    def offline_preflight(
        *,
        config_name,
        readiness_mode,
        bindings,
        runtime_contract,
        frozen_decoder_output_path=None,
        expected_frozen_decoder=None,
    ):
        del bindings, runtime_contract
        calls["preflight"].append(config_name)
        if expected_frozen_decoder is not None:
            frozen_descriptor = dict(expected_frozen_decoder)
        else:
            frozen_root = Path(frozen_decoder_output_path).resolve()
            frozen_root.mkdir(parents=True, exist_ok=True)
            frozen_descriptor = {
                "path": str(frozen_root),
                "fingerprint": "a" * 64,
                "body_synergy_contract_path": str(
                    (frozen_root / "body_synergy_contract.json").resolve()
                ),
                "body_synergy_contract_fingerprint": "b" * 64,
                "portable_decoder_core_fingerprint": "c" * 64,
                "decoder_core_fingerprint": "d" * 64,
            }
        return {
            "status": "passed",
            "config_name": config_name,
            "readiness_mode": readiness_mode,
            "policy_action_dim": 1,
            "body_action_dim": len(NAMES),
            "frozen_body_decoder": frozen_descriptor,
        }

    monkeypatch.setattr(pipeline, "save_primitive_source_manifest_from_splits", save_manifest)
    monkeypatch.setattr(pipeline, "fit_synergy_dataset", fit_dataset)
    monkeypatch.setattr(pipeline, "load_synergy_basis", load_basis)
    monkeypatch.setattr(pipeline, "_offline_action_preflight", offline_preflight)
    return calls


def test_plan_is_read_only_and_formal_without_proxy_is_explicit(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    output = tmp_path / "never-created-by-plan"

    plan = plan_stage1_pipeline(_request(fixture, output, readiness="formal"))

    assert plan["can_apply"] is True
    assert plan["writes_performed"] is False
    assert not output.exists()
    assert any("cannot emit formal training bindings" in item for item in plan["warnings"])


def test_rank_and_dynamic_contract_reaches_plan_fit_config_and_formal_apply(
    tmp_path,
    monkeypatch,
):
    import musclemimic.synergy.stage1_pipeline as pipeline

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    calls = _patch_apply_builders(monkeypatch)
    reports = {
        EXCITATION_SIGNAL_KIND: {
            "all": {"1": {"external_fixture_report": True}},
        }
    }
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")
    request = PipelineRequest(
        **{
            **request.__dict__,
            "region_ranks": {"all": (2, 1, 2)},
            "total_rank_budget": 3,
            "require_dynamic_coverage": True,
            "max_mean_dynamic_gap": 0.11,
            "max_key_phase_dynamic_gap": 0.22,
            "max_basis_condition_number": 1234.0,
            "min_effective_rank_fraction": 0.8,
            "expected_environment_fingerprint": "a" * 64,
            "expected_rollout_manifest_fingerprint": "b" * 64,
            "dynamic_coverage_reports": reports,
        }
    )

    plan = plan_stage1_pipeline(request)
    fit_identity = plan["request_identity"]["fit"]
    assert fit_identity["region_ranks"] == {"all": [1, 2]}
    assert fit_identity["total_rank_budget"] == 3
    assert fit_identity["require_dynamic_coverage"] is True
    assert fit_identity["dynamic_coverage_reports"] == reports
    changed = PipelineRequest(
        **{**request.__dict__, "max_mean_dynamic_gap": 0.12}
    )
    assert plan_stage1_pipeline(changed)["input_fingerprint"] != plan["input_fingerprint"]

    fit_config = pipeline._fit_config_for_request(request)
    assert fit_config.region_ranks == {"all": (1, 2)}
    assert fit_config.total_rank_budget == 3
    assert fit_config.require_dynamic_coverage is True
    assert fit_config.expected_environment_fingerprint == "a" * 64
    assert fit_config.expected_rollout_manifest_fingerprint == "b" * 64
    assert fit_config.max_basis_condition_number == 1234.0
    assert fit_config.min_effective_rank_fraction == 0.8

    release = apply_stage1_pipeline(request)
    assert release["readiness"] == "basis_ready"
    assert calls["fit_kwargs"][0]["dynamic_coverage_reports"] == reports
    applied_config = calls["fit_kwargs"][0]["config"]
    assert applied_config.region_ranks == {"all": (1, 2)}
    assert applied_config.total_rank_budget == 3


def test_stage1_cli_resolves_rank_and_dynamic_json_contracts(tmp_path):
    import musclemimic.synergy.stage1_pipeline as pipeline

    region_ranks = _write_json(tmp_path / "region_ranks.json", {"arm": [3, 1]})
    dynamic_reports = _write_json(
        tmp_path / "dynamic_reports.json",
        {EXCITATION_SIGNAL_KIND: {"arm": {"1": {"fixture": True}}}},
    )
    args = pipeline.build_parser().parse_args(
        [
            "plan",
            "--train",
            "train",
            "--val",
            "val",
            "--region-ranks-json",
            str(region_ranks),
            "--total-rank-budget",
            "7",
            "--require-dynamic-coverage",
            "--expected-environment-fingerprint",
            "a" * 64,
            "--expected-rollout-manifest-fingerprint",
            "b" * 64,
            "--dynamic-coverage-reports-json",
            str(dynamic_reports),
        ]
    )
    request = pipeline._request_from_args(args)
    assert request.region_ranks == {"arm": (1, 3)}
    assert request.total_rank_budget == 7
    assert request.require_dynamic_coverage is True
    assert request.dynamic_coverage_reports == {
        EXCITATION_SIGNAL_KIND: {"arm": {"1": {"fixture": True}}}
    }


def test_required_dynamic_coverage_without_reports_publishes_candidates_but_not_basis(
    tmp_path,
    monkeypatch,
):
    import musclemimic.synergy.stage1_pipeline as pipeline

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")
    request = PipelineRequest(
        **{
            **request.__dict__,
            "require_dynamic_coverage": True,
            "expected_environment_fingerprint": "a" * 64,
            "expected_rollout_manifest_fingerprint": "b" * 64,
        }
    )
    plan = plan_stage1_pipeline(request)
    assert plan["can_apply"] is True
    assert any("second-stage" in warning for warning in plan["warnings"])

    def awaiting_dynamic_evidence(*args, output_dir, **kwargs):
        del args, kwargs
        payload = {
            "schema_version": "synergy_dynamic_coverage_dataset_candidate_inventory_v1",
            "status": "dynamic_coverage_evidence_required",
            "regions": [],
        }
        payload["inventory_fingerprint"] = pipeline._json_sha256(payload)
        _write_json(
            Path(output_dir) / "dynamic_coverage_candidate_inventory.json",
            payload,
        )
        raise pipeline.BasisNotEligibleForEarlyControl(
            "dynamic coverage requires second-stage environment-rollout evidence"
        )

    monkeypatch.setattr(pipeline, "fit_synergy_dataset", awaiting_dynamic_evidence)
    release = apply_stage1_pipeline(request)
    assert release["readiness"] == "source_validated"
    assert release["ready_for_training"] is False
    assert "basis" not in release["artifacts"]
    assert release["artifacts"]["dynamic_coverage_candidates"]["status"] == (
        "dynamic_coverage_evidence_required"
    )
    assert "second-stage" in release["failures"][0]["message"]


def test_formal_without_proxy_publishes_basis_only_and_is_idempotent(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    calls = _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")

    release = apply_stage1_pipeline(request)

    assert release["readiness"] == "basis_ready"
    assert release["ready_for_training"] is False
    assert release["formal_target_coverage"] is False
    assert release["training_bindings"] is None
    assert Path(release["ready_path"]).is_file()
    assert calls["preflight"] == []
    assert calls["loaded_basis"][-1].endswith("selected-preferred")
    reused = apply_stage1_pipeline(request)
    assert reused["idempotent_reuse"] is True


def test_bootstrap_release_is_training_ready_but_never_claims_target_coverage(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    calls = _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="bootstrap")

    release = apply_stage1_pipeline(request)

    assert release["readiness"] == "training_ready_s"
    assert release["ready_for_training"] is True
    assert release["formal_target_coverage"] is False
    assert release["evidence_limitations"] == ["no_independent_chinajump_target_control_coverage"]
    descriptor = release["training_bindings"]
    assert descriptor["config_name"] == "fixture_bootstrap"
    bindings = json.loads(Path(descriptor["json_path"]).read_text(encoding="utf-8"))
    assert not any("COVERAGE" in key or "PROXY" in key for key in bindings["variables"])
    assert calls["preflight"] == ["fixture_bootstrap"]

    report = preflight_stage1_release(release["release_pointer_path"])
    assert report["passed"] is True
    assert report["real_environment_smoke"]["status"] == "not_run"
    assert report["training_started"] is False
    assert report["wandb_started"] is False
    assert calls["preflight"] == ["fixture_bootstrap", "fixture_bootstrap"]


def test_valid_v2_proxy_reaches_formal_training_ready_release(tmp_path, monkeypatch):
    from musclemimic.synergy.oracle_coverage import load_static_proxy_coverage_gate

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture, required_coverage_phase_ids=(0, 1))
    calls = _patch_apply_builders(monkeypatch)

    phase_id = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coefficients = np.asarray([0.25, 0.40, 0.55, 0.70], dtype=np.float64)
    excitation = coefficients[:, None] * np.asarray([[1.0, 0.5]], dtype=np.float64)
    controls = CTRLRANGE[:, 0] + excitation * (CTRLRANGE[:, 1] - CTRLRANGE[:, 0])
    target = tmp_path / "formal_target_controls.npz"
    np.savez_compressed(
        target,
        teacher_ctrl_physical=controls,
        physical_excitation=excitation,
        phase_id=phase_id,
        actuator_names=np.asarray(NAMES),
        actuator_ctrlrange=CTRLRANGE,
        motion_uid=np.full(phase_id.shape, 99, dtype=np.int64),
        subtraj_step_no=np.arange(phase_id.size, dtype=np.int32),
        **_muscle_contract_arrays(),
    )
    checkpoint = tmp_path / "independent_full_action_teacher.bin"
    checkpoint.write_bytes(b"independent full-action target controller")
    source_path = tmp_path / "formal_target_source.json"
    source = write_target_control_source_manifest(
        source_path,
        target,
        source_kind="full_action_teacher",
        target_skill_id="ChinaJump",
        action_dim=len(NAMES),
        model_hash=MODEL_HASH,
        actuator_schema_fingerprint=actuator_schema_hash(NAMES),
        ctrlrange_schema_fingerprint=ctrlrange_schema_hash(NAMES, CTRLRANGE),
        reference_fingerprint="7" * 64,
        phase_annotation_fingerprint="8" * 64,
        checkpoint_artifact_path=checkpoint,
    )
    qc_path = tmp_path / "formal_target_qc.json"
    write_target_control_qc(
        qc_path,
        source,
        phase_id=phase_id,
        tracking_qc_passed=True,
        forward_replay_verified=True,
        complete_trajectory_coverage=True,
        early_termination_rate=0.0,
        frame_coverage=1.0,
        episode_success_rate=1.0,
        excitation_upper_saturation_fraction=0.0,
        max_early_termination_rate=0.05,
        min_frame_coverage=0.95,
        min_episode_success_rate=0.95,
        max_excitation_upper_saturation_fraction=0.05,
    )
    proxy = build_coverage_proxy(
        target,
        source_manifest_path=source_path,
        source_qc_path=qc_path,
        phase_schema_path=fixture["phase_schema"],
        output_dir=tmp_path / "formal_proxy",
        expected_action_dim=len(NAMES),
        required_phase_ids=(0, 1),
        min_phase_samples=2,
    )
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")
    request = PipelineRequest(
        **{
            **request.__dict__,
            "coverage_proxy_artifact": str(proxy.manifest_path),
        }
    )

    release = apply_stage1_pipeline(request)

    assert release["readiness"] == "training_ready_s"
    assert release["ready_for_training"] is True
    assert release["formal_target_coverage"] is True
    assert release["evidence_limitations"] == []
    assert release["training_bindings"]["config_name"] == "fixture_formal"
    gate = load_static_proxy_coverage_gate(release["artifacts"]["coverage"]["path"])
    assert gate["schema_version"] == "chinajump_static_proxy_coverage_gate_v4"
    assert gate["proxy_binding"]["producer_binding"]["producer_manifest_fingerprint"] == (proxy.manifest_fingerprint)
    assert calls["preflight"] == ["fixture_formal"]
    report = preflight_stage1_release(release["release_pointer_path"])
    assert report["passed"] is True
    assert report["release_mode"] == "formal"


def test_existing_incomplete_object_is_quarantined_then_rebuilt(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")
    plan = plan_stage1_pipeline(request)
    incomplete = Path(plan["object_dir"])
    incomplete.mkdir(parents=True)
    (incomplete / "diagnostic-marker.txt").write_text("preserve me", encoding="utf-8")

    release = apply_stage1_pipeline(request)

    assert release["readiness"] == "basis_ready"
    assert Path(release["ready_path"]).is_file()
    quarantined = list((Path(request.output_root) / ".failed").glob(f"{plan['input_fingerprint']}.*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "diagnostic-marker.txt").read_text(encoding="utf-8") == "preserve me"


def test_ready_commit_without_pointer_is_validated_and_pointer_is_republished(
    tmp_path,
    monkeypatch,
):
    import musclemimic.synergy.stage1_pipeline as pipeline

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    calls = _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="bootstrap")
    real_atomic_write = pipeline._atomic_write_json
    injected = {"raised": False}

    def fail_once_at_pointer(path, payload):
        destination = Path(path)
        if destination.parent.name == "releases" and not injected["raised"]:
            injected["raised"] = True
            raise RuntimeError("simulated crash after READY before pointer")
        real_atomic_write(destination, payload)

    monkeypatch.setattr(pipeline, "_atomic_write_json", fail_once_at_pointer)
    with pytest.raises(RuntimeError, match="after READY"):
        apply_stage1_pipeline(request)

    plan = plan_stage1_pipeline(request)
    object_dir = Path(plan["object_dir"])
    assert (object_dir / "release.json").is_file()
    assert (object_dir / "READY.json").is_file()
    assert not (Path(request.output_root) / "releases").exists()

    recovered = apply_stage1_pipeline(request)

    assert recovered["idempotent_reuse"] is True
    assert Path(recovered["release_pointer_path"]).is_file()
    assert calls["fit"] == [str(object_dir / "fit")]
    loaded, loaded_path = pipeline.load_stage1_release(recovered["release_pointer_path"])
    assert loaded["release_fingerprint"] == recovered["release_fingerprint"]
    assert loaded_path == object_dir / "release.json"


def test_shell_binding_tamper_fails_preflight_and_ready_reuse(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="bootstrap")
    release = apply_stage1_pipeline(request)
    shell_path = Path(release["training_bindings"]["shell_path"])
    shell_path.write_text(
        shell_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineInputError, match="shell binding SHA256 mismatch"):
        preflight_stage1_release(release["release_pointer_path"])
    with pytest.raises(PipelineInputError, match="shell binding SHA256 mismatch"):
        apply_stage1_pipeline(request)


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("input_fingerprint", "f" * 64),
        ("release_mode", "formal"),
        ("readiness", "basis_ready"),
        ("ready_for_training", False),
        ("config_name", "different_config"),
        ("hydra_overrides", []),
        ("evidence_limitations", []),
    ],
)
def test_rehashed_ready_cross_contract_tamper_is_rejected(
    tmp_path,
    monkeypatch,
    field,
    tampered,
):
    import musclemimic.synergy.stage1_pipeline as pipeline

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="bootstrap")
    release = apply_stage1_pipeline(request)
    ready_path = Path(release["ready_path"])
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready[field] = tampered
    unsigned = {key: value for key, value in ready.items() if key != "ready_fingerprint"}
    ready["ready_fingerprint"] = pipeline._json_sha256(unsigned)
    _write_json(ready_path, ready)

    with pytest.raises(PipelineInputError, match=f"READY {field} differs"):
        preflight_stage1_release(release["release_path"])


def test_rehashed_binding_semantic_tamper_still_fails_commit_validation(
    tmp_path,
    monkeypatch,
):
    import musclemimic.synergy.stage1_pipeline as pipeline

    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    _patch_apply_builders(monkeypatch)
    request = _request(fixture, tmp_path / "artifacts", readiness="bootstrap")
    release = apply_stage1_pipeline(request)
    descriptor = release["training_bindings"]
    bindings_path = Path(descriptor["json_path"])
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings["release_mode"] = "formal"
    unsigned_bindings = {key: value for key, value in bindings.items() if key != "bindings_fingerprint"}
    bindings["bindings_fingerprint"] = pipeline._json_sha256(unsigned_bindings)
    _write_json(bindings_path, bindings)

    ready_path = Path(release["ready_path"])
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["training_bindings"]["fingerprint"] = bindings["bindings_fingerprint"]
    unsigned_ready = {key: value for key, value in ready.items() if key != "ready_fingerprint"}
    ready["ready_fingerprint"] = pipeline._json_sha256(unsigned_ready)
    _write_json(ready_path, ready)

    with pytest.raises(PipelineInputError, match="bindings release_mode differs"):
        preflight_stage1_release(release["release_path"])


def test_formal_bare_npz_and_unsafe_shell_values_fail_closed(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _patch_contract(monkeypatch, fixture)
    proxy = tmp_path / "proxy.npz"
    np.savez(proxy, physical_excitation=np.zeros((1, 2)))
    request = _request(fixture, tmp_path / "artifacts", readiness="formal")
    request = PipelineRequest(**{**request.__dict__, "coverage_proxy_artifact": str(proxy)})
    with pytest.raises(PipelineInputError, match="never a bare NPZ"):
        plan_stage1_pipeline(request)

    with pytest.raises(PipelineInputError, match="forbidden control"):
        write_shell_bindings(
            tmp_path / "bindings.env",
            variables={"SAFE_NAME": "bad\nvalue"},
            bindings_fingerprint="5" * 64,
            release_fingerprint="6" * 64,
        )


def test_public_proxy_loader_exposes_the_formal_producer_binding(tmp_path):
    import musclemimic.synergy.stage1_pipeline as pipeline

    phase_id = np.asarray([0, 0, 1, 1], dtype=np.int32)
    excitation = np.asarray(
        [[0.2, 0.1], [0.3, 0.2], [0.6, 0.7], [0.7, 0.8]],
        dtype=np.float32,
    )
    controls = CTRLRANGE[:, 0] + excitation * (CTRLRANGE[:, 1] - CTRLRANGE[:, 0])
    target = tmp_path / "target_controls.npz"
    np.savez_compressed(
        target,
        teacher_ctrl_physical=controls,
        physical_excitation=excitation,
        phase_id=phase_id,
        actuator_names=np.asarray(NAMES),
        actuator_ctrlrange=CTRLRANGE,
        motion_uid=np.arange(4, dtype=np.int64),
        subtraj_step_no=np.arange(4, dtype=np.int32),
        **_muscle_contract_arrays(),
    )
    source_path = tmp_path / "target_source.json"
    checkpoint_path = tmp_path / "teacher-checkpoint.bin"
    checkpoint_path.write_bytes(b"sealed full-action teacher")
    source = write_target_control_source_manifest(
        source_path,
        target,
        source_kind="full_action_teacher",
        target_skill_id="ChinaJump",
        action_dim=len(NAMES),
        model_hash=MODEL_HASH,
        actuator_schema_fingerprint=actuator_schema_hash(NAMES),
        ctrlrange_schema_fingerprint=ctrlrange_schema_hash(NAMES, CTRLRANGE),
        reference_fingerprint="7" * 64,
        phase_annotation_fingerprint="8" * 64,
        checkpoint_artifact_path=checkpoint_path,
    )
    qc_path = tmp_path / "target_qc.json"
    write_target_control_qc(
        qc_path,
        source,
        phase_id=phase_id,
        tracking_qc_passed=True,
        forward_replay_verified=True,
        complete_trajectory_coverage=True,
        early_termination_rate=0.0,
        frame_coverage=1.0,
        episode_success_rate=1.0,
        excitation_upper_saturation_fraction=0.0,
        max_early_termination_rate=0.05,
        min_frame_coverage=0.95,
        min_episode_success_rate=0.95,
        max_excitation_upper_saturation_fraction=0.05,
    )
    phase_schema_path = _write_json(
        tmp_path / "target_phase_schema.json",
        {
            "schema_version": "chinajump_coverage_phase_schema_v1",
            "target_skill_id": "ChinaJump",
            "phase_field": "phase_id",
            "producer_contract": "explicit_contact_event_frame_annotation_v1",
            "phases": [
                {"id": 0, "name": "prepare", "definition": "Prepare."},
                {"id": 1, "name": "execute", "definition": "Execute."},
            ],
        },
    )
    artifact = build_coverage_proxy(
        target,
        source_manifest_path=source_path,
        source_qc_path=qc_path,
        phase_schema_path=phase_schema_path,
        output_dir=tmp_path / "proxy",
        expected_action_dim=len(NAMES),
        required_phase_ids=(0, 1),
        min_phase_samples=2,
    )

    view = pipeline._load_coverage_proxy_artifact(
        artifact.manifest_path,
        expected_target_skill_id="ChinaJump",
        expected_runtime_contract={
            "model_hash": MODEL_HASH,
            "actuator_names": list(NAMES),
            "actuator_ctrlrange": CTRLRANGE.tolist(),
            "actuator_schema_hash": actuator_schema_hash(NAMES),
            "ctrlrange_schema_hash": ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": list(NAMES), "ctrlrange": CTRLRANGE.tolist()},
            ),
            "transform_ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, CTRLRANGE),
        },
        expected_phase_schema_fingerprint=artifact.manifest["phase_binding"]["phase_schema_fingerprint"],
    )

    assert view.fingerprint == artifact.manifest_fingerprint
    assert view.producer_binding == artifact.oracle_binding
    assert view.source_kind == "full_action_teacher"
    assert view.required_phase_ids == (0, 1)
    assert view.min_phase_samples == 2
    assert view.per_phase_sample_counts == {0: 2, 1: 2}
    pipeline._validate_formal_proxy_phase_contract(
        view,
        {"coverage_thresholds": {"required_phase_ids": [0, 1]}},
    )
    with pytest.raises(PipelineInputError, match="differ from the formal"):
        pipeline._validate_formal_proxy_phase_contract(
            view,
            {"coverage_thresholds": {"required_phase_ids": [0]}},
        )

    under_sampled = pipeline.CoverageProxyView(
        **{
            **view.__dict__,
            "per_phase_sample_counts": {0: 2, 1: 1},
        }
    )
    with pytest.raises(PipelineInputError, match="min_phase_samples"):
        pipeline._validate_formal_proxy_phase_contract(
            under_sampled,
            {"coverage_thresholds": {"required_phase_ids": [0, 1]}},
        )


def test_offline_preflight_reads_the_public_body_action_dimension(monkeypatch):
    import musclemimic.synergy.stage1_pipeline as pipeline

    cfg = SimpleNamespace(
        config_status={"readiness": "bootstrap_only"},
        experiment=SimpleNamespace(
            action_representation={
                "bootstrap_without_target_coverage": True,
                "require_coverage_gate": False,
            }
        ),
    )
    interface = SimpleNamespace(
        policy_action_dim=3,
        body_action_dim=2,
        synergy_dim=3,
        residual_dim=0,
        action_manifest={"physical_action_interface_hash": "9" * 64},
        frozen_decoder=SimpleNamespace(
            artifact_fingerprint="a" * 64,
            decoder_core_fingerprint="b" * 64,
        ),
        body_synergy_contract=SimpleNamespace(
            contract_fingerprint="c" * 64,
            portable_decoder_core_fingerprint="d" * 64,
            to_manifest=lambda: {"schema_version": "body_synergy_contract_v2"},
        ),
    )
    monkeypatch.setattr(pipeline, "_compose_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        pipeline,
        "build_early_synergy_action_interface",
        lambda *_args, **_kwargs: interface,
    )

    result = pipeline._offline_action_preflight(
        config_name="fixture_bootstrap",
        readiness_mode="bootstrap",
        bindings={"MUSCLEMIMIC_TEST_BASIS": "/tmp/unused"},
        runtime_contract={
            "actuator_names": list(NAMES),
            "actuator_ctrlrange": CTRLRANGE.tolist(),
            "model_hash": MODEL_HASH,
        },
    )

    assert result["policy_action_dim"] == 3
    assert result["body_action_dim"] == 2
