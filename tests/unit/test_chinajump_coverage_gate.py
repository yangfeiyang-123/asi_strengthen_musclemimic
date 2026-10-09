from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.synergy.action_interface import save_coefficient_statistics
from musclemimic.synergy.basis_artifact import save_synergy_basis
from musclemimic.synergy.oracle_coverage import (
    STATIC_PROXY_EVIDENCE_KIND,
    StaticProxyCoverageThresholds,
    canonicalize_static_proxy_phase_schema,
    evaluate_static_proxy_coverage,
    load_static_proxy_coverage_gate,
    main,
    proxy_content_fingerprint,
    write_static_proxy_coverage_gate,
)
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND


def _formal_basis(tmp_path, basis: np.ndarray):
    names = tuple(f"muscle_{index}" for index in range(basis.shape[0]))
    manifest = {
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "region": "whole_body",
        "rank": int(basis.shape[1]),
        "normalization": {"kind": "none", "basis_space": "unit_physical_excitation"},
        "source_dataset_fingerprint": "primitive-bank-fixture",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "fit_seed": 7,
        "transform": {
            "kind": "ctrlrange_affine_to_unit",
            "formula": "(ctrl-low)/(high-low)",
        },
        "split_provenance": {"train": {"motion_uids": [1]}, "validation": {"motion_uids": [2]}},
        "train_motion_uids": [1],
    }
    return save_synergy_basis(
        tmp_path / "formal_basis",
        basis=np.asarray(basis, dtype=np.float64),
        muscle_names=names,
        manifest=manifest,
    )


def _phase_schema(*phase_ids: int) -> dict:
    return {
        "schema_version": "chinajump_coverage_phase_schema_v1",
        "target_skill_id": "test_skill",
        "phase_field": "phase_id",
        "producer_contract": "unit_test_explicit_labels_v1",
        "phases": [
            {
                "id": phase_id,
                "name": f"phase_{phase_id}",
                "definition": f"Unit-test semantic definition for phase {phase_id}.",
            }
            for phase_id in sorted(phase_ids)
        ],
    }


def test_static_proxy_gate_passes_and_strict_loader_binds_formal_basis(tmp_path):
    basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    artifact = _formal_basis(tmp_path, basis)
    coefficients = np.asarray([[0.10, 0.20], [0.35, 0.15], [0.20, 0.50], [0.45, 0.25]])
    proxy = coefficients @ basis.T
    phase_id = np.asarray([0, 0, 1, 1], dtype=np.int32)

    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        phase_id=phase_id,
        phase_schema=_phase_schema(0, 1),
        coefficient_upper_bounds=np.asarray([0.75, 0.75]),
        thresholds=StaticProxyCoverageThresholds(required_phase_ids=(0, 1)),
        proxy_muscle_names=artifact.muscle_names,
    )
    gate_path = tmp_path / "static_proxy_gate.json"
    write_static_proxy_coverage_gate(gate_path, report)
    loaded = load_static_proxy_coverage_gate(
        gate_path,
        expected_basis_fingerprint=artifact.fingerprint,
    )

    assert loaded["passed"] is True
    assert loaded["evidence_kind"] == STATIC_PROXY_EVIDENCE_KIND
    assert loaded["limitations"][0] == "static_proxy_only"
    assert loaded["formal_basis_binding"]["artifact_fingerprint"] == artifact.fingerprint
    assert loaded["proxy_binding"]["observed_phase_ids"] == [0, 1]
    assert loaded["metrics"]["global"]["relative_l2_nrmse"] < 1e-8
    assert loaded["metrics"]["basis_effective_rank_fraction"] == 1.0


def test_bounded_coefficients_can_fail_static_proxy_gate(tmp_path):
    basis = np.eye(2, dtype=np.float64)
    artifact = _formal_basis(tmp_path, basis)
    proxy = np.asarray([[0.8, 0.7], [0.7, 0.8]], dtype=np.float64)
    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        coefficient_upper_bounds=0.2,
        thresholds=StaticProxyCoverageThresholds(
            max_global_relative_l2_nrmse=0.10,
            max_decoded_saturation_fraction=1.0,
        ),
    )
    gate_path = tmp_path / "failed_gate.json"
    write_static_proxy_coverage_gate(gate_path, report)

    assert report["passed"] is False
    assert report["checks"]["global_relative_l2_nrmse"] is False
    with pytest.raises(ValueError, match="did not pass"):
        load_static_proxy_coverage_gate(gate_path, expected_basis_fingerprint=artifact.fingerprint)
    assert load_static_proxy_coverage_gate(gate_path, require_passed=False)["passed"] is False


def test_zero_or_inactive_proxy_cannot_pass_coverage(tmp_path):
    artifact = _formal_basis(tmp_path, np.eye(2, dtype=np.float64))
    report = evaluate_static_proxy_coverage(
        artifact,
        np.zeros((4, 2), dtype=np.float64),
        coefficient_upper_bounds=1.0,
        thresholds=StaticProxyCoverageThresholds(
            max_decoded_saturation_fraction=1.0,
        ),
    )

    assert report["checks"]["proxy_target_rms"] is False
    assert report["checks"]["proxy_active_muscle_fraction"] is False
    assert report["passed"] is False


def test_static_proxy_gate_rejects_fingerprint_tampering(tmp_path):
    basis = np.eye(2, dtype=np.float64)
    artifact = _formal_basis(tmp_path, basis)
    proxy = np.asarray([[0.2, 0.4], [0.3, 0.1]], dtype=np.float64)
    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        coefficient_upper_bounds=1.0,
        thresholds=StaticProxyCoverageThresholds(max_decoded_saturation_fraction=1.0),
    )
    gate_path = tmp_path / "tampered_gate.json"
    write_static_proxy_coverage_gate(gate_path, report)
    untampered_path = tmp_path / "untampered_gate.json"
    write_static_proxy_coverage_gate(untampered_path, report)
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["proxy_binding"]["content_fingerprint"] = "f" * 64
    gate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="proxy_fingerprint|artifact_fingerprint mismatch"):
        load_static_proxy_coverage_gate(gate_path, require_passed=False)
    with pytest.raises(ValueError, match="differs from expected formal basis"):
        load_static_proxy_coverage_gate(
            untampered_path,
            expected_basis_fingerprint="b" * 64,
        )


def test_key_phase_failure_is_not_hidden_by_good_global_coverage(tmp_path):
    basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    artifact = _formal_basis(tmp_path, basis)
    easy = np.tile(np.asarray([[0.4, 0.3, 0.0]]), (40, 1))
    hard_landing = np.asarray([[0.0, 0.0, 0.5]])
    proxy = np.concatenate([easy, hard_landing], axis=0)
    phase_id = np.concatenate([np.zeros(easy.shape[0], dtype=np.int32), np.asarray([4], dtype=np.int32)])
    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        phase_id=phase_id,
        phase_schema=_phase_schema(0, 4),
        coefficient_upper_bounds=1.0,
        thresholds=StaticProxyCoverageThresholds(
            max_global_relative_l2_nrmse=0.20,
            max_phase_relative_l2_nrmse=0.25,
            max_decoded_saturation_fraction=1.0,
            required_phase_ids=(0, 4),
        ),
    )

    assert report["checks"]["global_relative_l2_nrmse"] is True
    assert report["checks"]["per_phase_relative_l2_nrmse"] == {"0": True, "4": False}
    assert report["checks"]["all_observed_phases_relative_l2_nrmse"] is False
    assert report["passed"] is False

    changed_phase_fingerprint = proxy_content_fingerprint(
        proxy,
        muscle_names=artifact.muscle_names,
        phase_id=np.zeros_like(phase_id),
        phase_schema=_phase_schema(0, 4),
    )
    assert changed_phase_fingerprint != report["proxy_binding"]["content_fingerprint"]


def test_present_but_zero_required_phase_cannot_satisfy_coverage(tmp_path):
    artifact = _formal_basis(tmp_path, np.eye(2, dtype=np.float64))
    proxy = np.asarray([[0.3, 0.2], [0.4, 0.1], [0.0, 0.0]], dtype=np.float64)
    report = evaluate_static_proxy_coverage(
        artifact,
        proxy,
        phase_id=np.asarray([1, 1, 4], dtype=np.int32),
        phase_schema=_phase_schema(1, 4),
        coefficient_upper_bounds=1.0,
        thresholds=StaticProxyCoverageThresholds(
            max_decoded_saturation_fraction=1.0,
            required_phase_ids=(1, 4),
        ),
    )

    assert report["checks"]["required_phase_presence"] is True
    assert report["checks"]["required_phase_target_rms"] == {
        "1": True,
        "4": False,
    }
    assert report["passed"] is False


def test_phase_semantics_are_required_and_bound_into_proxy_fingerprint(tmp_path):
    artifact = _formal_basis(tmp_path, np.eye(2, dtype=np.float64))
    proxy = np.asarray([[0.2, 0.3], [0.4, 0.1]], dtype=np.float64)
    phase_id = np.asarray([0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="semantic phase_schema"):
        evaluate_static_proxy_coverage(
            artifact,
            proxy,
            phase_id=phase_id,
            coefficient_upper_bounds=1.0,
        )

    original = _phase_schema(0, 1)
    swapped = _phase_schema(0, 1)
    swapped["phases"][0]["name"], swapped["phases"][1]["name"] = (
        swapped["phases"][1]["name"],
        swapped["phases"][0]["name"],
    )
    original_contract = canonicalize_static_proxy_phase_schema(original)
    swapped_contract = canonicalize_static_proxy_phase_schema(swapped)
    assert original_contract["phase_schema_fingerprint"] != swapped_contract["phase_schema_fingerprint"]
    assert proxy_content_fingerprint(
        proxy,
        muscle_names=artifact.muscle_names,
        phase_id=phase_id,
        phase_schema=original,
    ) != proxy_content_fingerprint(
        proxy,
        muscle_names=artifact.muscle_names,
        phase_id=phase_id,
        phase_schema=swapped,
    )


def test_static_proxy_cli_evaluates_and_writes_gate_report(tmp_path):
    basis = np.eye(2, dtype=np.float64)
    artifact = _formal_basis(tmp_path, basis)
    proxy_path = tmp_path / "chinajump_proxy.npz"
    np.savez_compressed(
        proxy_path,
        physical_excitation=np.asarray([[0.2, 0.4], [0.3, 0.1]], dtype=np.float32),
        phase_id=np.asarray([0, 1], dtype=np.int32),
        actuator_names=np.asarray(artifact.muscle_names),
    )
    output = tmp_path / "cli_gate.json"
    phase_schema_path = tmp_path / "phase_schema.json"
    phase_schema_path.write_text(
        json.dumps(_phase_schema(0, 1)),
        encoding="utf-8",
    )
    stats = save_coefficient_statistics(
        tmp_path / "coefficient_stats.npz",
        np.asarray([[0.25, 0.45], [0.35, 0.50], [0.40, 0.55]]),
        basis_fingerprint=artifact.fingerprint,
    )

    return_code = main(
        [
            "--basis-artifact",
            str(artifact.path),
            "--proxy-npz",
            str(proxy_path),
            "--output",
            str(output),
            "--coefficient-stats",
            stats["path"],
            "--phase-schema",
            str(phase_schema_path),
        ]
    )

    assert return_code == 0
    loaded = load_static_proxy_coverage_gate(
        output,
        expected_basis_fingerprint=artifact.fingerprint,
    )
    assert loaded["passed"]
    np.testing.assert_allclose(
        loaded["solver"]["coefficient_upper_bounds"],
        1.2 * np.asarray(stats["coefficient_q99"]),
    )
