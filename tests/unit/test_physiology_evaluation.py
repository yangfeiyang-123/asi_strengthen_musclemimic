"""CPU-only tests for simulation physiology metrics."""

from __future__ import annotations

import dataclasses
import importlib.metadata

import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    MUSCLE_EXCITATION_SEMANTICS,
    MUSCLE_EXCITATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
)
from musclemimic.evaluation.physiology import (
    build_physiology_report,
    kinetic_chain_metrics,
    main,
    muscle_timing_metrics,
    synergy_residual_metrics,
    validate_physiology_lineage,
    validate_physiology_signal_contract,
)
from musclemimic.physiology.anatomical_groups import (
    ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
    AnatomicalTaxonomy,
    taxonomy_muscle_channel_core_fingerprint,
)
from musclemimic.physiology.continuity_groups import (
    FASCICLE_CONTINUITY_SCHEMA_VERSION,
    continuity_graph_fingerprint,
    validate_fascicle_continuity_graph,
)
from musclemimic.physiology.synergy_binding import ordered_muscle_schema_sha256


def _physical_signal_fields(
    names: tuple[str, ...],
    excitation: np.ndarray,
) -> dict[str, np.ndarray]:
    width = len(names)
    return {
        "teacher_ctrl_physical": np.asarray(excitation, dtype=np.float64).copy(),
        "actuator_ctrlrange": np.asarray([[0.0, 1.0]] * width, dtype=np.float64),
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_excitation_source": np.asarray(MUSCLE_EXCITATION_SOURCE),
        "muscle_excitation_semantics": np.asarray(MUSCLE_EXCITATION_SEMANTICS),
        "muscle_excitation_transform": np.asarray(UNIT_EXCITATION_TRANSFORM),
        "muscle_excitation_formula": np.asarray(MUSCLE_EXCITATION_FORMULA),
        "muscle_excitation_roundoff_policy": np.asarray(MUSCLE_EXCITATION_ROUNDOFF_POLICY),
        "muscle_activation_source": np.asarray(MUSCLE_ACTIVATION_SOURCE),
        "muscle_activation_semantics": np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
        "muscle_activation_roundoff_policy": np.asarray(UNIT_INTERVAL_ROUNDOFF_POLICY),
        "activation_valid_mask": np.ones((width,), dtype=bool),
        "muscle_channel_contract_schema_version": np.asarray(MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION),
        "actuator_ids": np.arange(width, dtype=np.int32),
        "actuator_dyntype": np.asarray(["muscle"] * width),
        "actuator_actnum": np.ones(width, dtype=np.int32),
        "actuator_actadr": np.arange(width, dtype=np.int32),
        "model_na": np.asarray(width, dtype=np.int32),
    }


def _diagnostic_taxonomy(names: tuple[str, ...]) -> AnatomicalTaxonomy:
    rows = tuple(
        {
            "ordered_index": index,
            "actuator_id": index,
            "name": name,
            "side": "right",
            "dyntype": "mjDYN_MUSCLE",
            "dyntype_id": 4,
            "actnum": 1,
            "actadr": index,
            "ctrlrange": [0.0, 1.0],
            "target": {
                "transmission": "mjTRN_TENDON",
                "transmission_id": 3,
                "object_type": "tendon",
                "object_id": index,
                "name": f"{name}_tendon",
            },
            "dynprm": [0.0] * 10,
            "gainprm": [0.0] * 10,
            "biasprm": [0.0] * 10,
        }
        for index, name in enumerate(names)
    )
    hard_group = {
        "group_id": "fixture_verified_lines",
        "members": list(names),
        "member_weights": [1.0] * len(names),
        "group_weight": 1.0,
        "deadband": 0.1,
        "activity_off": 0.0,
        "activity_on": 0.1,
        "training_enabled": False,
    }
    return AnatomicalTaxonomy(
        schema_version=ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
        taxonomy_id="fixture_taxonomy",
        model_binding={
            "package": "musclemimic-models",
            "version": importlib.metadata.version("musclemimic-models"),
            "actuator_schema_hash": actuator_schema_hash(names),
            "runtime_model_hash": "f" * 64,
        },
        signal_contract={},
        ordered_actuators=rows,
        hard_line_groups=(hard_group,),
        soft_compartment_groups=(),
        observation_aggregates=(),
        functional_synergy_regions=(),
        fingerprint="e" * 64,
        notes="test-only",
    )


def _diagnostic_continuity_graph(taxonomy: AnatomicalTaxonomy):
    core = taxonomy_muscle_channel_core_fingerprint(taxonomy)
    payload = {
        "schema_version": FASCICLE_CONTINUITY_SCHEMA_VERSION,
        "graph_id": "fixture_continuity",
        "taxonomy_binding": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "ordered_muscle_schema_sha256": ordered_muscle_schema_sha256(taxonomy.actuator_names),
            "actuator_schema_hash": taxonomy.stable_model_binding["actuator_schema_hash"],
            "muscle_channel_core_fingerprint": core,
            "runtime_compatibility": "exact_runtime_model",
        },
        "default_behavior": "diagnostics_only_no_reward",
        "chains": [
            {
                "chain_id": "fixture_chain",
                "side": "right",
                "anatomical_structure": "fixture",
                "members": list(taxonomy.actuator_names),
                "edges": [
                    [taxonomy.actuator_names[index], taxonomy.actuator_names[index + 1]]
                    for index in range(len(taxonomy.actuator_names) - 1)
                ],
                "edge_weights": [1.0] * (len(taxonomy.actuator_names) - 1),
                "deadband": 0.1,
                "chain_weight": 1.0,
                "activity_off": 0.0,
                "activity_on": 0.1,
                "review_status": "provisional",
                "training_enabled": False,
                "provenance": [],
            }
        ],
        "notes": "test",
    }
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    return validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)


def test_muscle_timing_is_reported_relative_to_impact():
    signal = np.zeros((2, 12, 2), dtype=np.float64)
    signal[:, 3, 0] = 1.0
    signal[:, 7, 1] = 2.0

    result = muscle_timing_metrics(
        signal,
        muscle_names=["proximal", "distal"],
        impact_frames=np.asarray([5, 5], dtype=np.int32),
        sampling_rate_hz=10.0,
    )

    assert result["proximal"]["peak_time_from_impact_s"]["mean"] == pytest.approx(-0.2)
    assert result["distal"]["peak_time_from_impact_s"]["mean"] == pytest.approx(0.2)
    assert result["distal"]["peak_value"]["mean"] == pytest.approx(2.0)


def test_residual_report_detects_energy_outside_distal_mask():
    target = np.ones((1, 4, 3), dtype=np.float64)
    reconstruction = target.copy()
    residual = np.zeros_like(target)
    residual[:, :, 0] = 0.5
    residual[:, :, 2] = 1.0

    result = synergy_residual_metrics(
        target,
        reconstruction,
        residual=residual,
        allowed_residual_mask=np.asarray([False, False, True]),
        phase_id=np.asarray([[0, 0, 1, 1]], dtype=np.int32),
    )

    assert result["allowed_channel_count"] == 1
    assert result["outside_allowed_mask_energy_ratio"] == pytest.approx(0.2)
    assert set(result["per_phase_residual_energy_ratio"]) == {"0", "1"}


def test_kinetic_chain_reports_order_and_post_impact_deceleration():
    velocity = np.zeros((2, 20, 3), dtype=np.float64)
    velocity[:, 5, 0] = 2.0
    velocity[:, 7, 1] = 3.0
    velocity[:, 9, 2] = 4.0
    # Give every post-impact segment a finite deceleration curve.
    velocity[:, 10:15, :] += np.linspace(1.0, 0.0, 5)[None, :, None]

    result = kinetic_chain_metrics(
        velocity,
        joint_names=["pelvis", "shoulder", "wrist"],
        ordered_segments=[
            {"name": "proximal", "joints": ["pelvis"]},
            {"name": "middle", "joints": ["shoulder"]},
            {"name": "distal", "joints": ["wrist"]},
        ],
        impact_frames=np.asarray([10, 10], dtype=np.int32),
        sampling_rate_hz=10.0,
        post_impact_horizon_s=0.4,
    )

    assert result["proximal_to_distal_order_agreement"] == pytest.approx(1.0)
    assert result["ordered_segments"] == ["proximal", "middle", "distal"]
    assert result["post_impact_deceleration_per_s"]["distal"]["mean"] > 0.0


def test_build_report_covers_phase_synergy_joint_load_and_cocontraction():
    activation = np.asarray(
        [[[0.1, 0.2], [0.4, 0.1], [0.8, 0.2], [0.2, 0.7], [0.1, 0.2], [0.0, 0.1]]],
        dtype=np.float64,
    )
    arrays = {
        "muscle_excitation": activation,
        "muscle_activation": activation,
        "actuator_names": np.asarray(["agonist", "antagonist"]),
        "sampling_rate_hz": np.asarray(10.0),
        "impact_frame": np.asarray([3], dtype=np.int32),
        "phase_id": np.asarray([[0, 0, 1, 1, 2, 2]], dtype=np.int32),
        "synergy_reconstruction": 0.9 * activation,
        "synergy_residual": 0.1 * activation,
        "joint_torque": np.ones((1, 6, 2), dtype=np.float64),
        "joint_angular_velocity": np.ones((1, 6, 2), dtype=np.float64),
        "joint_names": np.asarray(["shoulder", "wrist"]),
        **_physical_signal_fields(("agonist", "antagonist"), activation),
    }

    taxonomy = _diagnostic_taxonomy(("agonist", "antagonist"))
    result = build_physiology_report(
        arrays,
        co_contraction_pairs=[["agonist", "antagonist"]],
        allowed_residual_mask=np.asarray([False, True]),
        anatomical_taxonomy=taxonomy,
        fascicle_continuity_graph=_diagnostic_continuity_graph(taxonomy),
    )

    assert result["schema_version"] == "simulation_physiology_v2"
    assert len(result["physical_signal_semantics_fingerprint"]) == 64
    assert result["physical_signal_contract"]["activation_valid_mask"] == [
        True,
        True,
    ]
    assert set(result["phase_activation"]) == {"0", "1", "2"}
    assert "agonist__antagonist" in result["co_contraction"]
    assert "synergy_residual" in result
    assert "joint_load" in result
    intra = result["intra_muscle_diagnostics"]
    assert intra["default_behavior"] == "diagnostics_only_no_reward"
    hard = intra["relationships"]["hard_line_groups"]["activation"]
    assert hard["aggregate"]["group_count"] == 1
    assert set(hard["per_phase"]) == {"0", "1", "2"}
    assert hard["aggregate"]["exact_exo"]["loss_mean"] > 0.0
    group = hard["aggregate"]["per_group"]["fixture_verified_lines"]
    assert group["rms_deviation"] > 0.0
    assert group["p95_abs_deviation"] >= group["mean_abs_deviation"]
    assert intra["relationships"]["soft_compartment_groups"]["activation"]["aggregate"]["group_count"] == 0
    assert intra["coverage"]["intra_muscle_measured"] is True
    assert intra["coverage"]["measured_group_counts"] == {
        "hard_line_groups": 1,
        "soft_compartment_groups": 0,
    }
    assert intra["coverage"]["total_measured_group_count"] == 1
    binding = result["synergy_residual"]["taxonomy_binding"]
    assert binding["actuator_count"] == 2
    # Compare against a hash computed from the channel names in this NPZ, not
    # against another field derived from the same taxonomy object -- the point is
    # that the reported binding describes the exported channels.
    assert binding["ordered_muscle_schema_sha256"] == ordered_muscle_schema_sha256(("agonist", "antagonist"))
    assert binding["ordered_muscle_schema_sha256"] == intra["ordered_muscle_schema_sha256"]
    continuity = result["fascicle_continuity"]
    assert continuity["coverage"] == {
        "declared_chain_count": 1,
        "measured_chain_count": 1,
        "training_enabled_chain_count": 0,
        "measured_edge_count": 1,
        "continuity_measured": True,
        "zero_loss_interpretation": "loss_reflects_measured_adjacency_dispersion",
    }
    assert set(continuity["activation"]["per_phase"]) == {"0", "1", "2"}
    activation_continuity = continuity["activation"]["aggregate"]
    assert activation_continuity["chain_count"] == 1
    assert activation_continuity["edge_count"] == 1
    assert activation_continuity["edge_absolute_difference"]["p95"] > 0.0
    assert "fixture_chain" in continuity["activation"]["per_chain"]
    assert continuity["excitation"]["aggregate"]["loss"]["n"] == 6


def test_empty_taxonomy_reports_zero_loss_as_uncovered_not_as_consistent():
    """A zero IMR loss must never be readable as evidence of consistency.

    The checked-in MyoFullBody taxonomy has no legal group in any class, so every
    IMR loss it produces is 0.0.  Coverage is the only field that distinguishes
    that from a perfectly consistent model.
    """

    activation = np.asarray(
        [[[0.1, 0.9], [0.4, 0.1], [0.8, 0.2], [0.2, 0.7]]],
        dtype=np.float64,
    )
    names = ("agonist", "antagonist")
    taxonomy = _diagnostic_taxonomy(names)
    empty = dataclasses.replace(taxonomy, hard_line_groups=())
    arrays = {
        "muscle_excitation": activation,
        "muscle_activation": activation,
        "actuator_names": np.asarray(names),
        "sampling_rate_hz": np.asarray(10.0),
        "impact_frame": np.asarray([2], dtype=np.int32),
        **_physical_signal_fields(names, activation),
    }

    result = build_physiology_report(arrays, anatomical_taxonomy=empty)
    intra = result["intra_muscle_diagnostics"]
    hard = intra["relationships"]["hard_line_groups"]["activation"]["aggregate"]

    assert hard["group_count"] == 0
    assert hard["exact_exo"]["loss_mean"] == 0.0
    assert intra["coverage"]["intra_muscle_measured"] is False
    assert intra["coverage"]["total_measured_group_count"] == 0
    assert intra["coverage"]["zero_loss_interpretation"] == "no_group_measured_zero_loss_is_not_evidence_of_consistency"


def test_synergy_reconstruction_hash_must_match_the_taxonomy_channel_schema():
    activation = np.asarray([[[0.1, 0.9], [0.4, 0.1]]], dtype=np.float64)
    names = ("agonist", "antagonist")
    arrays = {
        "muscle_excitation": activation,
        "muscle_activation": activation,
        "actuator_names": np.asarray(names),
        "sampling_rate_hz": np.asarray(10.0),
        "impact_frame": np.asarray([1], dtype=np.int32),
        "synergy_reconstruction": 0.9 * activation,
        "muscle_schema_sha256": np.asarray("f" * 64),
        **_physical_signal_fields(names, activation),
    }

    with pytest.raises(ValueError, match="does not match the taxonomy"):
        build_physiology_report(
            arrays,
            anatomical_taxonomy=_diagnostic_taxonomy(names),
        )


def test_synergy_binding_records_when_no_artifact_hash_was_declared():
    activation = np.asarray([[[0.1, 0.9], [0.4, 0.1]]], dtype=np.float64)
    names = ("agonist", "antagonist")
    arrays = {
        "muscle_excitation": activation,
        "muscle_activation": activation,
        "actuator_names": np.asarray(names),
        "sampling_rate_hz": np.asarray(10.0),
        "impact_frame": np.asarray([1], dtype=np.int32),
        "synergy_reconstruction": 0.9 * activation,
        **_physical_signal_fields(names, activation),
    }

    result = build_physiology_report(
        arrays,
        anatomical_taxonomy=_diagnostic_taxonomy(names),
    )
    binding = result["synergy_residual"]["taxonomy_binding"]

    assert binding["verified_synergy_hash_fields"] == []
    assert "unverified_synergy_lineage" in binding


def test_exported_ordered_muscle_hash_is_verified_against_the_taxonomy():
    """Mirror what stage3_signal_export now writes, so the check is not inert.

    The exporter publishes ``ordered_muscle_schema_sha256`` alongside
    ``actuator_names``; with that producer in place the report must actually verify
    the field rather than fall through to ``unverified_synergy_lineage``.
    """

    activation = np.asarray([[[0.1, 0.9], [0.4, 0.1]]], dtype=np.float64)
    names = ("agonist", "antagonist")
    arrays = {
        "muscle_excitation": activation,
        "muscle_activation": activation,
        "actuator_names": np.asarray(names),
        "ordered_muscle_schema_sha256": np.asarray(ordered_muscle_schema_sha256(names)),
        "sampling_rate_hz": np.asarray(10.0),
        "impact_frame": np.asarray([1], dtype=np.int32),
        "synergy_reconstruction": 0.9 * activation,
        **_physical_signal_fields(names, activation),
    }

    result = build_physiology_report(
        arrays,
        anatomical_taxonomy=_diagnostic_taxonomy(names),
    )
    binding = result["synergy_residual"]["taxonomy_binding"]

    assert binding["verified_synergy_hash_fields"] == ["ordered_muscle_schema_sha256"]
    assert "unverified_synergy_lineage" not in binding


def test_physiology_lineage_binds_policy_event_session_and_decoder():
    checkpoint = "a" * 64
    promotion = "b" * 64
    formal = "c" * 64
    runtime = "d" * 64
    arrays = {
        "muscle_excitation": np.full((1, 4, 2), 0.2, dtype=np.float64),
        "muscle_activation": np.full((1, 4, 2), 0.1, dtype=np.float64),
        "actuator_names": np.asarray(["bic", "tri"]),
        "policy_decoder_type": np.asarray("synergy_residual"),
        "policy_checkpoint_fingerprint": np.asarray(checkpoint),
        "policy_promotion_fingerprint": np.asarray(promotion),
        "formal_synergy_basis_fingerprint": np.asarray(formal),
        "analysis_synergy_basis_fingerprint": np.asarray(formal),
        "runtime_synergy_basis_fingerprint": np.asarray(runtime),
        "runtime_synergy_basis_source_fingerprint": np.asarray(formal),
        "event_reference_fingerprint": np.asarray("e" * 64),
        "session_uid": np.asarray("session-heldout-01"),
        **_physical_signal_fields(
            ("bic", "tri"),
            np.full((1, 4, 2), 0.2, dtype=np.float64),
        ),
    }
    result = validate_physiology_lineage(
        arrays,
        expected_policy_checkpoint_fingerprint=checkpoint,
        expected_policy_promotion_fingerprint=promotion,
        expected_formal_synergy_basis_fingerprint=formal,
        expected_event_reference_fingerprint="e" * 64,
        expected_session_uid="session-heldout-01",
        expected_policy_decoder_type="synergy_residual",
    )
    assert result["binding_verified"] == 1.0
    assert len(result["lineage_fingerprint"]) == 64
    assert len(result["physical_signal_semantics_fingerprint"]) == 64

    arrays["session_uid"] = np.asarray("wrong-session")
    with pytest.raises(ValueError, match="session identity"):
        validate_physiology_lineage(
            arrays,
            expected_policy_checkpoint_fingerprint=checkpoint,
            expected_policy_promotion_fingerprint=promotion,
            expected_formal_synergy_basis_fingerprint=formal,
            expected_event_reference_fingerprint="e" * 64,
            expected_session_uid="session-heldout-01",
            expected_policy_decoder_type="synergy_residual",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "muscle_excitation_source",
            np.asarray("teacher_action_normalized"),
            "excitation semantics",
        ),
        (
            "muscle_excitation_semantics",
            np.asarray("raw_signed_ctrl"),
            "excitation semantics",
        ),
        (
            "muscle_activation_source",
            np.asarray("muscle_excitation"),
            "activation semantics",
        ),
        (
            "muscle_activation_roundoff_policy",
            np.asarray("silently_clip"),
            "activation semantics",
        ),
    ],
)
def test_physiology_signal_contract_rejects_semantic_relabeling(
    field: str,
    value: np.ndarray,
    message: str,
):
    excitation = np.full((1, 3, 2), 0.3)
    arrays = {
        "muscle_excitation": excitation,
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(("bic", "tri"), excitation),
    }
    arrays[field] = value

    with pytest.raises(ValueError, match=message):
        validate_physiology_signal_contract(arrays)


@pytest.mark.parametrize(
    ("signal", "bad_value", "message"),
    [
        ("muscle_excitation", -0.1, "signed normalized action/raw ctrl"),
        ("muscle_excitation", 1.1, "outside \\[0,1\\]"),
        ("muscle_activation", -0.1, "must not be relabeled as activation"),
        ("muscle_activation", 1.1, "outside \\[0,1\\]"),
    ],
)
def test_physiology_signal_contract_rejects_non_unit_physical_signals(
    signal: str,
    bad_value: float,
    message: str,
):
    excitation = np.full((1, 3, 2), 0.3)
    arrays = {
        "muscle_excitation": excitation,
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(("bic", "tri"), excitation),
    }
    arrays[signal][0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_physiology_signal_contract(arrays)


def test_physiology_signal_contract_requires_boolean_all_valid_mask():
    excitation = np.full((1, 3, 2), 0.3)
    arrays = {
        "muscle_excitation": excitation,
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(("bic", "tri"), excitation),
    }
    arrays["activation_valid_mask"] = np.asarray([1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="must be boolean"):
        validate_physiology_signal_contract(arrays)

    arrays["activation_valid_mask"] = np.asarray([True, False])
    with pytest.raises(ValueError, match="without a scalar MuJoCo activation state"):
        validate_physiology_signal_contract(arrays)


def test_physiology_dry_run_lists_the_physical_signal_contract(capsys):
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out

    assert '"schema_version": "simulation_physiology_v3"' in output
    assert '"physical_signal_schema_version"' in output
    assert '"muscle_excitation_transform"' in output
    assert '"muscle_activation_roundoff_policy"' in output
    assert '"activation_valid_mask"' in output
