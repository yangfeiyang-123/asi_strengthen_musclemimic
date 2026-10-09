from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.hybrid_basis import (
    HYBRID_BASIS_SCHEMA_VERSION,
    HybridBasisConfig,
    HybridBasisGateError,
    HybridBasisResult,
    build_hybrid_basis,
    save_hybrid_basis_artifact,
    validate_hybrid_basis_result,
)
from musclemimic.synergy.rank_selection import dynamic_coverage_requirement

REGIONAL_FINGERPRINT = "1" * 64
GLOBAL_FINGERPRINT = "2" * 64


def _formal_hybrid_metadata(tmp_path):
    return {
        "primitive_source_binding": None,
        "source_components": {
            "regional": {
                "region": "regional_composite",
                "artifact_path": str((tmp_path / "regional").resolve()),
                "artifact_fingerprint": REGIONAL_FINGERPRINT,
            },
            "global": {
                "region": "whole_body",
                "artifact_path": str((tmp_path / "global").resolve()),
                "artifact_fingerprint": GLOBAL_FINGERPRINT,
            },
        },
        "artifact_role": "primary_hybrid_global_regional",
        "dynamic_coverage_requirement": dynamic_coverage_requirement(
            required=False,
            max_mean_dynamic_gap=0.15,
            max_key_phase_dynamic_gap=0.25,
            expected_environment_fingerprint=None,
            expected_rollout_manifest_fingerprint=None,
        ),
        "dynamic_coverage": None,
        "candidate_basis_fingerprint": "4" * 64,
    }


def _selection_inputs():
    regional = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    global_basis = np.asarray(
        [
            [2**-0.5, 0.0, 0.0, 0.0],
            [2**-0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    heldout = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    names = ("m0", "m1", "m2", "m3")
    return regional, global_basis, heldout, names


def _build_selection_result(*, config: HybridBasisConfig | None = None):
    regional, global_basis, heldout, names = _selection_inputs()
    return build_hybrid_basis(
        regional,
        global_basis,
        regional_muscle_names=names,
        global_muscle_names=names,
        heldout_values=heldout,
        regional_source_fingerprint=REGIONAL_FINGERPRINT,
        global_source_fingerprint=GLOBAL_FINGERPRINT,
        config=config,
    )


def test_hybrid_selection_is_deterministic_deduplicated_and_ordered_by_marginal_vaf():
    first = _build_selection_result()
    second = _build_selection_result()

    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(first.basis, expected)
    np.testing.assert_array_equal(second.basis, expected)
    assert first.manifest == second.manifest
    assert first.manifest["retained_global_column_indices_in_output_order"] == [3, 1]
    assert first.manifest["construction_mode"] == "regional_plus_retained_original_global_columns"
    assert first.manifest["heldout_evaluation"]["all_passed"] is True

    decisions = first.manifest["candidate_decisions"]
    assert decisions[0]["decision_reason"] == "represented_by_regional_nonnegative_cone"
    assert decisions[1]["status"] == "retained"
    assert decisions[1]["selection_order"] == 1
    assert decisions[2]["decision_reason"] == "cosine_duplicate_of_retained_global_column"
    assert decisions[2]["duplicate_of_global_column_index"] == 1
    assert decisions[3]["selection_order"] == 0
    assert decisions[3]["heldout_global_vaf_marginal_gain"] > decisions[1]["heldout_global_vaf_marginal_gain"]
    validated = validate_hybrid_basis_result(
        first,
        regional_basis=_selection_inputs()[0],
        global_basis=_selection_inputs()[1],
    )
    np.testing.assert_array_equal(validated.basis, first.basis)
    json.dumps(first.manifest, allow_nan=False)


def test_hybrid_appends_original_global_column_never_its_signed_projection_residual():
    regional = np.asarray([[1.0], [0.0], [0.0]])
    original_global_column = np.asarray([[1.0], [0.0], [1.0]])
    heldout = np.asarray([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    names = ("a", "silent", "c")

    result = build_hybrid_basis(
        regional,
        original_global_column,
        regional_muscle_names=names,
        global_muscle_names=names,
        heldout_values=heldout,
        regional_source_fingerprint=REGIONAL_FINGERPRINT,
        global_source_fingerprint=GLOBAL_FINGERPRINT,
    )

    np.testing.assert_array_equal(result.basis[:, 1:], original_global_column)
    assert not np.array_equal(result.basis[:, 1], np.asarray([0.0, 0.0, 1.0]))
    assert result.manifest["column_policy"].endswith("never_signed_projection_residuals")
    assert result.manifest["candidate_decisions"][0]["regional_cone_projection_residual_ratio"] == pytest.approx(
        2**-0.5
    )


def test_total_rank_budget_fails_closed_without_truncating_selected_columns():
    with pytest.raises(HybridBasisGateError, match="rank_budget") as error:
        _build_selection_result(config=HybridBasisConfig(max_total_rank=3))

    report = error.value.gate_report
    assert report["failures"] == ["rank_budget"]
    construction = report["construction_manifest"]
    assert construction["total_rank"] == 4
    assert construction["retained_global_column_indices_in_output_order"] == [3, 1]


def test_no_novel_global_columns_is_an_explicit_regional_only_result():
    regional = np.eye(2)
    global_basis = np.asarray([[0.5], [0.5]])
    heldout = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    names = ("a", "b")

    result = build_hybrid_basis(
        regional,
        global_basis,
        regional_muscle_names=names,
        global_muscle_names=names,
        heldout_values=heldout,
        regional_source_fingerprint=REGIONAL_FINGERPRINT,
        global_source_fingerprint=GLOBAL_FINGERPRINT,
    )

    np.testing.assert_array_equal(result.basis, regional)
    assert result.manifest["construction_mode"] == "regional_only_no_novel_global_columns"
    assert result.manifest["retained_global_rank"] == 0
    assert result.manifest["candidate_decisions"][0]["status"] == "rejected"


def test_novel_but_zero_heldout_gain_column_is_rejected_instead_of_inflating_rank():
    regional = np.asarray([[1.0], [0.0]])
    global_basis = np.asarray([[0.0], [1.0]])
    heldout = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    names = ("active", "unseen")

    result = build_hybrid_basis(
        regional,
        global_basis,
        regional_muscle_names=names,
        global_muscle_names=names,
        heldout_values=heldout,
        regional_source_fingerprint=REGIONAL_FINGERPRINT,
        global_source_fingerprint=GLOBAL_FINGERPRINT,
    )

    np.testing.assert_array_equal(result.basis, regional)
    decision = result.manifest["candidate_decisions"][0]
    assert decision["regional_cone_projection_residual_ratio"] == pytest.approx(1.0)
    assert decision["decision_reason"] == "insufficient_heldout_global_vaf_marginal_gain"
    assert decision["heldout_global_vaf_marginal_gain"] == pytest.approx(0.0)
    assert result.manifest["thresholds"][
        "heldout_global_vaf_marginal_gain_retain_strictly_greater_than"
    ] == pytest.approx(1e-6)
    assert result.manifest["construction_mode"] == "regional_only_no_novel_global_columns"


def test_marginal_gain_threshold_is_validated():
    with pytest.raises(ValueError, match="min_heldout_global_vaf_marginal_gain"):
        _build_selection_result(config=HybridBasisConfig(min_heldout_global_vaf_marginal_gain=-1e-6))


@pytest.mark.parametrize(
    ("regional", "global_basis", "heldout", "config", "expected_failure"),
    [
        (
            np.asarray([[1.0], [0.0]]),
            np.asarray([[1.0], [0.0]]),
            np.asarray([[1.0, 1.0]]),
            HybridBasisConfig(min_heldout_global_vaf=0.75, min_heldout_local_vaf_quantile=0.0),
            "heldout_global_vaf",
        ),
        (
            np.asarray([[1.0], [0.0]]),
            np.asarray([[1.0], [0.0]]),
            np.asarray([[10.0, 1.0]]),
            HybridBasisConfig(
                min_heldout_global_vaf=0.90,
                local_vaf_quantile=0.0,
                min_heldout_local_vaf_quantile=0.50,
            ),
            "heldout_local_vaf_quantile",
        ),
        (
            np.asarray([[1.0, 1.0], [0.0, 0.01]]),
            np.asarray([[1.0], [0.0]]),
            np.asarray([[1.0, 0.0], [1.0, 0.01]]),
            HybridBasisConfig(max_basis_condition_number=10.0),
            "basis_condition_number",
        ),
        (
            np.asarray([[1.0, 1.0], [0.0, 0.0]]),
            np.asarray([[1.0], [0.0]]),
            np.asarray([[1.0, 0.0]]),
            HybridBasisConfig(min_effective_rank_fraction=0.80),
            "effective_rank_fraction",
        ),
    ],
)
def test_every_configured_quality_gate_fails_closed(regional, global_basis, heldout, config, expected_failure):
    names = ("a", "b")
    with pytest.raises(HybridBasisGateError) as error:
        build_hybrid_basis(
            regional,
            global_basis,
            regional_muscle_names=names,
            global_muscle_names=names,
            heldout_values=heldout,
            regional_source_fingerprint=REGIONAL_FINGERPRINT,
            global_source_fingerprint=GLOBAL_FINGERPRINT,
            config=config,
        )

    assert expected_failure in error.value.gate_report["failures"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("names", "names/order"),
        ("negative", "finite and non-negative"),
        ("fingerprint", "lowercase 64-hex"),
    ],
)
def test_source_schema_nonnegativity_and_fingerprints_are_fail_closed(mutation, message):
    regional, global_basis, heldout, names = _selection_inputs()
    global_names = names
    fingerprint = GLOBAL_FINGERPRINT
    if mutation == "names":
        global_names = ("m1", "m0", "m2", "m3")
    elif mutation == "negative":
        global_basis[0, 0] = -1e-12
    else:
        fingerprint = "not-a-sha256"

    with pytest.raises(ValueError, match=message):
        build_hybrid_basis(
            regional,
            global_basis,
            regional_muscle_names=names,
            global_muscle_names=global_names,
            heldout_values=heldout,
            regional_source_fingerprint=REGIONAL_FINGERPRINT,
            global_source_fingerprint=fingerprint,
        )


def test_formal_artifact_helper_binds_sources_decisions_and_matrix_hash(tmp_path):
    result = _build_selection_result()
    artifact = save_hybrid_basis_artifact(
        tmp_path / "hybrid",
        result,
        signal_kind="physical_excitation_unit",
        source_dataset_fingerprint="heldout-dataset-fingerprint",
        teacher_checkpoint_fingerprint="3" * 64,
        normalization={"kind": "none", "fit_scope": "source_artifacts"},
        fit_seed=7,
        transform={"kind": "identity_nonnegative_excitation"},
        split_provenance={"train": {"motion_uids": [10]}, "validation": {"motion_uids": [20]}},
        train_motion_uids=[10],
        **_formal_hybrid_metadata(tmp_path),
    )
    loaded = load_synergy_basis(artifact.path)

    np.testing.assert_array_equal(loaded.basis, result.basis.astype(np.float32))
    assert loaded.manifest["hybrid_schema_version"] == HYBRID_BASIS_SCHEMA_VERSION
    assert loaded.manifest["hybrid_construction"] == result.manifest
    assert loaded.manifest["source_basis_fingerprints"] == {
        "regional": REGIONAL_FINGERPRINT,
        "global": GLOBAL_FINGERPRINT,
    }
    assert "composite_schema_version" not in loaded.manifest
    assert loaded.manifest["hybrid_matrix_content_sha256"] == result.manifest["matrix_content_sha256"]


def test_artifact_helper_rejects_tampered_result_before_creating_output(tmp_path):
    result = _build_selection_result()
    tampered_basis = result.basis.copy()
    tampered_basis[0, 0] += 0.25
    tampered = HybridBasisResult(
        basis=tampered_basis,
        muscle_names=result.muscle_names,
        manifest=copy.deepcopy(result.manifest),
    )
    output = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="matrix content hash differs"):
        save_hybrid_basis_artifact(
            output,
            tampered,
            signal_kind="physical_excitation_unit",
            source_dataset_fingerprint="dataset",
            teacher_checkpoint_fingerprint="3" * 64,
            normalization={"kind": "none"},
            fit_seed=0,
            transform={"kind": "identity"},
            split_provenance={"train": {}, "validation": {}},
            train_motion_uids=[1],
            **_formal_hybrid_metadata(tmp_path),
        )
    assert not output.exists()

    tampered_manifest = copy.deepcopy(result.manifest)
    tampered_manifest["candidate_decisions"][0]["decision_reason"] = "forged"
    forged = HybridBasisResult(result.basis.copy(), result.muscle_names, tampered_manifest)
    with pytest.raises(ValueError, match="construction_fingerprint mismatch"):
        save_hybrid_basis_artifact(
            output,
            forged,
            signal_kind="physical_excitation_unit",
            source_dataset_fingerprint="dataset",
            teacher_checkpoint_fingerprint="3" * 64,
            normalization={"kind": "none"},
            fit_seed=0,
            transform={"kind": "identity"},
            split_provenance={"train": {}, "validation": {}},
            train_motion_uids=[1],
            **_formal_hybrid_metadata(tmp_path),
        )
    assert not output.exists()
