from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.synergy.basis_artifact import save_synergy_basis
from musclemimic.synergy.coverage_proxy import (
    COVERAGE_PROXY_ARTIFACT_KIND,
    build_coverage_proxy,
    build_target_control_qc,
    build_target_control_source_manifest,
    load_coverage_proxy_artifact,
    write_target_control_qc,
    write_target_control_source_manifest,
)
from musclemimic.synergy.oracle_coverage import (
    FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
    StaticProxyCoverageThresholds,
    evaluate_static_proxy_coverage,
    load_static_proxy_coverage_gate,
    write_static_proxy_coverage_gate,
)
from musclemimic.synergy.schema import (
    EXCITATION_SIGNAL_KIND,
    ctrlrange_schema_hash,
)

NAMES = ("muscle_a", "muscle_b", "muscle_c")
CTRLRANGE = np.asarray([[0.0, 1.0], [-1.0, 1.0], [0.2, 1.2]], dtype=np.float64)
PHASE_ID = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
EXCITATION = np.asarray(
    [
        [0.10, 0.20, 0.30],
        [0.15, 0.25, 0.35],
        [0.20, 0.30, 0.40],
        [0.25, 0.35, 0.45],
        [0.30, 0.40, 0.50],
        [0.35, 0.45, 0.55],
        [0.20, 0.25, 0.30],
        [0.15, 0.20, 0.25],
    ],
    dtype=np.float32,
)
MODEL_HASH = "1" * 64
REFERENCE_FINGERPRINT = "2" * 64
PHASE_ANNOTATION_FINGERPRINT = "5" * 64
QC_THRESHOLDS = {
    "max_early_termination_rate": 0.05,
    "min_frame_coverage": 0.95,
    "min_episode_success_rate": 0.95,
    "max_excitation_upper_saturation_fraction": 0.05,
}


def _phase_schema() -> dict:
    return {
        "schema_version": "chinajump_coverage_phase_schema_v1",
        "target_skill_id": "ChinaJump",
        "phase_field": "phase_id",
        "producer_contract": "explicit_contact_event_frame_annotation_v1",
        "phases": [
            {"id": 1, "name": "takeoff", "definition": "Audited takeoff propulsion."},
            {"id": 2, "name": "flight", "definition": "Audited rotation during flight."},
            {"id": 3, "name": "landing", "definition": "Audited landing absorption."},
            {"id": 4, "name": "recovery", "definition": "Audited post-landing balance."},
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _raw_ctrl(excitation: np.ndarray = EXCITATION) -> np.ndarray:
    return CTRLRANGE[:, 0] + np.asarray(excitation) * (CTRLRANGE[:, 1] - CTRLRANGE[:, 0])


def _qc_kwargs(*, saturation: float, tracking_qc_passed: bool = True) -> dict:
    return {
        "tracking_qc_passed": tracking_qc_passed,
        "forward_replay_verified": True,
        "complete_trajectory_coverage": True,
        "early_termination_rate": 0.0,
        "frame_coverage": 1.0,
        "episode_success_rate": 1.0,
        "excitation_upper_saturation_fraction": saturation,
        **QC_THRESHOLDS,
    }


def _source_fixture(
    tmp_path: Path,
    *,
    phase_id: np.ndarray = PHASE_ID,
    source_kind: str = "full_action_teacher",
    declared_excitation: np.ndarray | None = EXCITATION,
    raw_controls: bool = True,
) -> dict:
    input_path = tmp_path / "target_controls.npz"
    arrays: dict[str, np.ndarray] = {
        "actuator_names": np.asarray(NAMES),
        "actuator_ctrlrange": CTRLRANGE,
        "phase_id": np.asarray(phase_id),
        "motion_uid": np.arange(len(phase_id), dtype=np.int64) + 100,
        "subtraj_step_no": np.arange(len(phase_id), dtype=np.int32),
    }
    if raw_controls:
        arrays["teacher_ctrl_physical" if source_kind == "full_action_teacher" else "applied_ctrl"] = _raw_ctrl(
            EXCITATION[: len(phase_id)]
        )
    else:
        arrays["qpos"] = np.zeros((len(phase_id), 7), dtype=np.float32)
        arrays["qvel"] = np.zeros((len(phase_id), 6), dtype=np.float32)
    if declared_excitation is not None:
        arrays["physical_excitation"] = np.asarray(declared_excitation[: len(phase_id)])
    np.savez_compressed(input_path, **arrays)
    if source_kind == "full_action_teacher":
        control_artifact = tmp_path / "checkpoint_1"
        control_artifact.mkdir()
        (control_artifact / "weights.bin").write_bytes(b"full-action-teacher")
        control_artifact_kwargs = {"checkpoint_artifact_path": control_artifact}
    else:
        control_artifact = tmp_path / "trajectory_optimizer.bin"
        control_artifact.write_bytes(b"full-action-trajectory-optimizer")
        control_artifact_kwargs = {"optimizer_artifact_path": control_artifact}
    source_path = tmp_path / "source_manifest.json"
    source = write_target_control_source_manifest(
        source_path,
        input_path,
        source_kind=source_kind,
        target_skill_id="ChinaJump",
        action_dim=len(NAMES),
        model_hash=MODEL_HASH,
        actuator_schema_fingerprint=actuator_schema_hash(NAMES),
        ctrlrange_schema_fingerprint=ctrlrange_schema_hash(NAMES, CTRLRANGE),
        reference_fingerprint=REFERENCE_FINGERPRINT,
        phase_annotation_fingerprint=PHASE_ANNOTATION_FINGERPRINT,
        **control_artifact_kwargs,
    )
    saturation = float(np.mean(EXCITATION[: len(phase_id)] >= 1.0 - 1e-6))
    qc_path = tmp_path / "source_qc.json"
    write_target_control_qc(
        qc_path,
        source,
        phase_id=phase_id,
        **_qc_kwargs(saturation=saturation),
    )
    phase_schema_path = _write_json(tmp_path / "phase_schema.json", _phase_schema())
    return {
        "input": input_path,
        "source_path": source_path,
        "source": source,
        "qc_path": qc_path,
        "phase_schema_path": phase_schema_path,
        "control_artifact": control_artifact,
    }


def _build(tmp_path: Path, fixture: dict, **kwargs):
    return build_coverage_proxy(
        fixture["input"],
        source_manifest_path=fixture["source_path"],
        source_qc_path=fixture["qc_path"],
        phase_schema_path=fixture["phase_schema_path"],
        output_dir=tmp_path / "proxy",
        expected_action_dim=len(NAMES),
        min_phase_samples=2,
        **kwargs,
    )


@pytest.mark.parametrize("source_kind", ["full_action_teacher", "trajectory_optimizer"])
def test_coverage_proxy_recomputes_excitation_and_seals_provenance(tmp_path, source_kind):
    fixture = _source_fixture(tmp_path, source_kind=source_kind)
    artifact = _build(tmp_path, fixture)
    loaded = load_coverage_proxy_artifact(
        artifact.manifest_path,
        expected_manifest_fingerprint=artifact.manifest_fingerprint,
        expected_content_fingerprint=artifact.content_fingerprint,
    )
    assert load_coverage_proxy_artifact(artifact.manifest_path.parent).manifest_fingerprint == (
        artifact.manifest_fingerprint
    )
    assert load_coverage_proxy_artifact(artifact.npz_path).content_fingerprint == artifact.content_fingerprint

    assert loaded.manifest["artifact_kind"] == COVERAGE_PROXY_ARTIFACT_KIND
    assert loaded.source_kind == source_kind
    assert loaded.source_manifest_fingerprint == fixture["source"]["manifest_fingerprint"]
    assert loaded.oracle_binding["producer_manifest_fingerprint"] == loaded.manifest_fingerprint
    with np.load(loaded.npz_path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["physical_excitation"], EXCITATION, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(data["phase_id"], PHASE_ID)
        assert tuple(data["actuator_names"].astype(str).tolist()) == NAMES


def test_formal_static_gate_binds_proxy_producer_manifest(tmp_path):
    fixture = _source_fixture(tmp_path)
    proxy_artifact = _build(tmp_path, fixture)
    basis = save_synergy_basis(
        tmp_path / "basis",
        basis=np.eye(len(NAMES), dtype=np.float64),
        muscle_names=NAMES,
        manifest={
            "signal_kind": EXCITATION_SIGNAL_KIND,
            "region": "whole_body",
            "rank": len(NAMES),
            "normalization": {"kind": "none", "basis_space": "unit_physical_excitation"},
            "source_dataset_fingerprint": "primitive-fixture",
            "teacher_checkpoint_fingerprint": "9" * 64,
            "fit_seed": 1,
            "transform": {"kind": "ctrlrange_affine_to_unit", "formula": "(ctrl-low)/(high-low)"},
            "split_provenance": {"train": {"motion_uids": [1]}, "validation": {"motion_uids": [2]}},
            "train_motion_uids": [1],
        },
    )
    report = evaluate_static_proxy_coverage(
        basis,
        EXCITATION,
        phase_id=PHASE_ID,
        phase_schema=_phase_schema(),
        coefficient_upper_bounds=1.0,
        thresholds=StaticProxyCoverageThresholds(
            max_decoded_saturation_fraction=1.0,
            required_phase_ids=(1, 2, 3, 4),
        ),
        proxy_muscle_names=NAMES,
        proxy_producer_binding=proxy_artifact.oracle_binding,
    )
    gate = tmp_path / "formal_gate.json"
    write_static_proxy_coverage_gate(gate, report)
    loaded = load_static_proxy_coverage_gate(gate, expected_basis_fingerprint=basis.fingerprint)

    assert loaded["schema_version"] == FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION
    producer = loaded["proxy_binding"]["producer_binding"]
    assert producer["producer_manifest_fingerprint"] == proxy_artifact.manifest_fingerprint
    assert producer["source_manifest_fingerprint"] == fixture["source"]["manifest_fingerprint"]
    assert producer["required_phase_ids"] == [1, 2, 3, 4]
    assert producer["min_phase_samples"] == 2
    assert producer["per_phase_sample_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2}

    forged_binding = dict(proxy_artifact.oracle_binding)
    forged_binding["min_phase_samples"] = 3
    with pytest.raises(ValueError, match="sample floor"):
        evaluate_static_proxy_coverage(
            basis,
            EXCITATION,
            phase_id=PHASE_ID,
            phase_schema=_phase_schema(),
            thresholds=StaticProxyCoverageThresholds(required_phase_ids=(1, 2, 3, 4)),
            proxy_muscle_names=NAMES,
            proxy_producer_binding=forged_binding,
        )


def test_qpos_only_input_is_never_interpreted_as_excitation(tmp_path):
    fixture = _source_fixture(tmp_path, raw_controls=False, declared_excitation=None)
    with pytest.raises(ValueError, match="kinematic qpos/qvel"):
        _build(tmp_path, fixture)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("primitive_source", True, "primitive-only"),
        ("early_synergy_action_representation", True, "circular"),
        ("action_space_kind", "synergy_coefficients", "full-dimensional"),
    ],
)
def test_circular_or_non_full_action_sources_are_rejected(tmp_path, field, value, message):
    fixture = _source_fixture(tmp_path)
    source = dict(fixture["source"])
    source[field] = value
    source["manifest_fingerprint"] = _manifest_fingerprint(source, "manifest_fingerprint")
    _write_json(fixture["source_path"], source)

    with pytest.raises(ValueError, match=message):
        _build(tmp_path, fixture)


def test_failed_qc_and_missing_required_phase_fail_closed(tmp_path):
    fixture = _source_fixture(tmp_path)
    failed_qc = build_target_control_qc(
        fixture["source"],
        phase_id=PHASE_ID,
        **_qc_kwargs(saturation=0.0, tracking_qc_passed=False),
    )
    _write_json(fixture["qc_path"], failed_qc)
    with pytest.raises(ValueError, match="QC did not pass"):
        _build(tmp_path, fixture)

    missing_phase = np.asarray([1, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    other_dir = tmp_path / "missing_phase"
    other_dir.mkdir()
    fixture = _source_fixture(other_dir, phase_id=missing_phase)
    with pytest.raises(ValueError, match="missing or undersampled"):
        _build(other_dir, fixture)


def test_declared_excitation_abi_and_source_content_drift_are_rejected(tmp_path):
    mismatched = EXCITATION.copy()
    mismatched[0, 0] += 0.1
    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    fixture = _source_fixture(mismatch_dir, declared_excitation=mismatched)
    with pytest.raises(ValueError, match="differs from exact applied-ctrl transform"):
        _build(mismatch_dir, fixture)

    abi_dir = tmp_path / "abi"
    abi_dir.mkdir()
    fixture = _source_fixture(abi_dir)
    bad_source = build_target_control_source_manifest(
        fixture["input"],
        source_kind="full_action_teacher",
        target_skill_id="ChinaJump",
        action_dim=len(NAMES),
        model_hash=MODEL_HASH,
        actuator_schema_fingerprint="f" * 64,
        ctrlrange_schema_fingerprint=ctrlrange_schema_hash(NAMES, CTRLRANGE),
        reference_fingerprint=REFERENCE_FINGERPRINT,
        phase_annotation_fingerprint=PHASE_ANNOTATION_FINGERPRINT,
        checkpoint_artifact_path=fixture["control_artifact"],
    )
    _write_json(fixture["source_path"], bad_source)
    bad_qc = build_target_control_qc(
        bad_source,
        phase_id=PHASE_ID,
        **_qc_kwargs(saturation=0.0),
    )
    _write_json(fixture["qc_path"], bad_qc)
    with pytest.raises(ValueError, match="actuator names/order drift"):
        _build(abi_dir, fixture)

    drift_dir = tmp_path / "content_drift"
    drift_dir.mkdir()
    fixture = _source_fixture(drift_dir)
    with np.load(fixture["input"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["teacher_ctrl_physical"] = arrays["teacher_ctrl_physical"].copy()
    arrays["teacher_ctrl_physical"][0, 0] += 0.01
    np.savez_compressed(fixture["input"], **arrays)
    with pytest.raises(ValueError, match="content hash"):
        _build(drift_dir, fixture)


def test_proxy_loader_rejects_npz_tampering(tmp_path):
    fixture = _source_fixture(tmp_path)
    artifact = _build(tmp_path, fixture)
    with np.load(artifact.npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["physical_excitation"] = arrays["physical_excitation"].copy()
    arrays["physical_excitation"][0, 0] += 0.01
    np.savez_compressed(artifact.npz_path, **arrays)

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_coverage_proxy_artifact(artifact.manifest_path)


def test_proxy_embeds_canonical_source_qc_and_seals_phase_floor(tmp_path):
    fixture = _source_fixture(tmp_path)
    artifact = _build(tmp_path, fixture)
    source_copy = artifact.source_manifest_path
    qc_copy = artifact.source_qc_path

    assert source_copy.is_file()
    assert qc_copy.is_file()
    assert artifact.required_phase_ids == (1, 2, 3, 4)
    assert artifact.min_phase_samples == 2
    assert artifact.per_phase_sample_counts == {1: 2, 2: 2, 3: 2, 4: 2}
    assert artifact.oracle_binding["required_phase_ids"] == [1, 2, 3, 4]
    assert artifact.oracle_binding["min_phase_samples"] == 2
    source_binding = artifact.manifest["source_binding"]
    assert source_binding["source_manifest_file_sha256"] == _file_hash(source_copy)
    assert source_binding["source_qc_file_sha256"] == _file_hash(qc_copy)
    audit = json.loads(source_copy.read_text(encoding="utf-8"))["control_artifact_audit"]
    assert audit["files"]
    assert audit["sha256"] == source_binding["checkpoint_fingerprint"]


@pytest.mark.parametrize("embedded_name", ["source_manifest.json", "source_qc.json"])
def test_proxy_loader_rejects_embedded_source_or_qc_byte_tampering(tmp_path, embedded_name):
    fixture = _source_fixture(tmp_path)
    artifact = _build(tmp_path, fixture)
    embedded = artifact.manifest_path.parent / embedded_name
    embedded.write_text(embedded.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="embedded source.*content hash mismatch"):
        load_coverage_proxy_artifact(artifact.manifest_path)


def test_forged_self_consistent_qc_json_is_semantically_rejected(tmp_path):
    fixture = _source_fixture(tmp_path)
    artifact = _build(tmp_path, fixture)
    qc_path = artifact.manifest_path.parent / "source_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["tracking_qc_passed"] = False
    # Keep the old passing check and passed flag, then recompute every outer
    # fingerprint/hash.  Fingerprint self-consistency must not bypass semantics.
    qc["qc_fingerprint"] = _manifest_fingerprint(qc, "qc_fingerprint")
    _write_json(qc_path, qc)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["source_binding"]["source_qc_fingerprint"] = qc["qc_fingerprint"]
    manifest["source_binding"]["source_qc_file_sha256"] = _file_hash(qc_path)
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest, "manifest_fingerprint")
    _write_json(artifact.manifest_path, manifest)

    with pytest.raises(ValueError, match="checks are stale or inconsistent"):
        load_coverage_proxy_artifact(artifact.manifest_path)


@pytest.mark.parametrize("source_kind", ["full_action_teacher", "trajectory_optimizer"])
def test_proxy_loader_rehashes_live_control_artifact(tmp_path, source_kind):
    fixture = _source_fixture(tmp_path, source_kind=source_kind)
    artifact = _build(tmp_path, fixture)
    control_artifact = fixture["control_artifact"]
    changed = control_artifact / "weights.bin" if control_artifact.is_dir() else control_artifact
    changed.write_bytes(changed.read_bytes() + b"-drift")

    with pytest.raises(ValueError, match="live artifact content drifted"):
        load_coverage_proxy_artifact(artifact.manifest_path)


def test_source_and_qc_builders_require_live_and_explicit_evidence(tmp_path):
    fixture = _source_fixture(tmp_path)
    with pytest.raises(TypeError, match="required keyword-only argument"):
        build_target_control_qc(
            fixture["source"],
            phase_id=PHASE_ID,
            excitation_upper_saturation_fraction=0.0,
        )
    with pytest.raises(TypeError, match="checkpoint_fingerprint"):
        build_target_control_source_manifest(
            fixture["input"],
            source_kind="full_action_teacher",
            target_skill_id="ChinaJump",
            action_dim=len(NAMES),
            model_hash=MODEL_HASH,
            actuator_schema_fingerprint=actuator_schema_hash(NAMES),
            ctrlrange_schema_fingerprint=ctrlrange_schema_hash(NAMES, CTRLRANGE),
            reference_fingerprint=REFERENCE_FINGERPRINT,
            phase_annotation_fingerprint=PHASE_ANNOTATION_FINGERPRINT,
            checkpoint_fingerprint="f" * 64,
        )


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_fingerprint(payload: dict, field: str) -> str:
    import hashlib

    canonical = {key: value for key, value in payload.items() if key != field}
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
