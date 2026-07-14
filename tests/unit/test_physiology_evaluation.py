"""CPU-only tests for simulation physiology metrics."""

from __future__ import annotations

import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
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


def _physical_signal_fields(width: int) -> dict[str, np.ndarray]:
    return {
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_excitation_source": np.asarray(MUSCLE_EXCITATION_SOURCE),
        "muscle_excitation_semantics": np.asarray(MUSCLE_EXCITATION_SEMANTICS),
        "muscle_excitation_transform": np.asarray(UNIT_EXCITATION_TRANSFORM),
        "muscle_activation_source": np.asarray(MUSCLE_ACTIVATION_SOURCE),
        "muscle_activation_semantics": np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
        "muscle_activation_roundoff_policy": np.asarray(UNIT_INTERVAL_ROUNDOFF_POLICY),
        "activation_valid_mask": np.ones((width,), dtype=bool),
    }


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
        **_physical_signal_fields(2),
    }

    result = build_physiology_report(
        arrays,
        co_contraction_pairs=[["agonist", "antagonist"]],
        allowed_residual_mask=np.asarray([False, True]),
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
        **_physical_signal_fields(2),
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
    arrays = {
        "muscle_excitation": np.full((1, 3, 2), 0.3),
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(2),
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
    arrays = {
        "muscle_excitation": np.full((1, 3, 2), 0.3),
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(2),
    }
    arrays[signal][0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_physiology_signal_contract(arrays)


def test_physiology_signal_contract_requires_boolean_all_valid_mask():
    arrays = {
        "muscle_excitation": np.full((1, 3, 2), 0.3),
        "muscle_activation": np.full((1, 3, 2), 0.2),
        "actuator_names": np.asarray(["bic", "tri"]),
        **_physical_signal_fields(2),
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
