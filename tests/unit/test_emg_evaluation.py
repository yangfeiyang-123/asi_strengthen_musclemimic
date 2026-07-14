"""CPU-only tests for the local sEMG validation framework."""

from __future__ import annotations

import json

import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
)
from musclemimic.evaluation.emg_eval import (
    EmgFilterConfig,
    _summary_with_clustered_bootstrap,
    evaluate_emg_validation,
    impact_aligned_resample,
    match_synergy_bases,
    preprocess_emg,
    validate_emg_mapping,
    validate_simulation_policy_evidence,
)

POLICY_CHECKPOINT = "1" * 64
POLICY_PROMOTION = "2" * 64
FORMAL_BASIS = "3" * 64
RUNTIME_BASIS = "4" * 64


def _mapping() -> dict:
    return {
        "schema_version": "emg_local_mapping_v1",
        "validation_scope": "right_upper_limb_local",
        "normalization": "per_trial_peak",
        "channels": [
            {
                "emg_channel": "biceps",
                "simulation_actuators": ["bic_long", "bic_short"],
                "weights": [0.6, 0.4],
                "mapping_uncertainty": "two model heads share one surface electrode",
            },
            {
                "emg_channel": "triceps",
                "simulation_actuators": ["tri"],
                "weights": [1.0],
                "mapping_uncertainty": "single model compartment approximation",
            },
        ],
        "cocontraction_pairs": [["biceps", "triceps"]],
    }


def _activation_contract_fields(width: int) -> dict[str, np.ndarray]:
    return {
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_activation_source": np.asarray(MUSCLE_ACTIVATION_SOURCE),
        "muscle_activation_semantics": np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
        "muscle_activation_roundoff_policy": np.asarray(
            UNIT_INTERVAL_ROUNDOFF_POLICY
        ),
        "activation_valid_mask": np.ones((width,), dtype=bool),
        **_policy_evidence_fields(),
    }


def _policy_evidence_fields(*, decoder_type="direct") -> dict[str, np.ndarray]:
    fields = {
        "policy_decoder_type": np.asarray(decoder_type),
        "policy_checkpoint_fingerprint": np.asarray(POLICY_CHECKPOINT),
        "policy_promotion_fingerprint": np.asarray(POLICY_PROMOTION),
        "formal_synergy_basis_fingerprint": np.asarray(FORMAL_BASIS),
        "analysis_synergy_basis_fingerprint": np.asarray(FORMAL_BASIS),
    }
    if decoder_type != "direct":
        fields.update(
            {
                "runtime_synergy_basis_fingerprint": np.asarray(RUNTIME_BASIS),
                "runtime_synergy_basis_source_fingerprint": np.asarray(FORMAL_BASIS),
            }
        )
    return fields


def _expected_policy_binding() -> dict[str, str]:
    return {
        "expected_policy_checkpoint_fingerprint": POLICY_CHECKPOINT,
        "expected_policy_promotion_fingerprint": POLICY_PROMOTION,
        "expected_formal_synergy_basis_fingerprint": FORMAL_BASIS,
    }


def test_mapping_requires_local_scope_names_and_uncertainty():
    contract = validate_emg_mapping(
        _mapping(),
        emg_channel_names=["biceps", "triceps"],
        actuator_names=["bic_long", "bic_short", "tri"],
    )

    assert contract["validation_scope"] == "right_upper_limb_local"
    np.testing.assert_allclose(contract["channels"][0]["weights"], [0.6, 0.4])

    invalid = _mapping()
    invalid["channels"][0].pop("mapping_uncertainty")
    with pytest.raises(ValueError, match="mapping_uncertainty"):
        validate_emg_mapping(invalid)


def test_preprocess_emg_is_nonnegative_and_impact_alignment_is_physical_time():
    fs = 1000.0
    time = np.arange(2000) / fs
    envelope = 0.2 + np.exp(-0.5 * ((time - 1.0) / 0.12) ** 2)
    raw = (envelope * np.sin(2.0 * np.pi * 90.0 * time))[None, :, None]

    processed = preprocess_emg(
        raw,
        sampling_rate_hz=fs,
        config=EmgFilterConfig(bandpass_high_hz=300.0),
    )
    aligned, relative_time = impact_aligned_resample(
        processed,
        np.asarray([1000], dtype=np.int32),
        sampling_rate_hz=fs,
        pre_impact_s=0.5,
        post_impact_s=0.5,
        output_samples=101,
    )

    assert processed.shape == raw.shape
    assert np.min(processed) >= 0.0
    assert aligned.shape == (1, 101, 1)
    assert relative_time[50] == pytest.approx(0.0)


def test_synergy_matching_is_permutation_invariant():
    first = np.asarray([[1.0, 0.1], [0.2, 1.0], [0.1, 0.3]])
    second = first[:, [1, 0]]
    coefficients = np.asarray([[0.1, 0.8], [0.5, 0.2], [0.9, 0.1]])

    result = match_synergy_bases(
        first,
        second,
        first_coefficients=coefficients,
        second_coefficients=coefficients[:, [1, 0]],
    )

    assert result["first_to_second_assignment"] == [1, 0]
    assert result["mean_weight_cosine_similarity"] == pytest.approx(1.0)
    assert result["mean_coefficient_correlation"] == pytest.approx(1.0)


def test_policy_evidence_supports_direct_analysis_basis_and_binds_synergy_source():
    direct = validate_simulation_policy_evidence(
        _policy_evidence_fields(),
        **_expected_policy_binding(),
    )
    assert direct["binding_verified"] == 1.0
    assert direct["formal_basis_role"] == "formal_analysis_basis_only"
    assert direct["runtime_synergy_basis_fingerprint"] is None

    synergy = _policy_evidence_fields(decoder_type="fixed_synergy")
    synergy["runtime_synergy_basis_source_fingerprint"] = np.asarray("5" * 64)
    with pytest.raises(ValueError, match="source fingerprint differs"):
        validate_simulation_policy_evidence(
            synergy,
            **_expected_policy_binding(),
        )


def test_cluster_bootstrap_keeps_channels_within_trial_together():
    summary = _summary_with_clustered_bootstrap(
        np.asarray([[0.0, 10.0], [0.0, 10.0]]),
        subject_ids=["subject-a", "subject-b"],
        bootstrap_samples=100,
        seed=3,
    )

    assert summary["mean"] == pytest.approx(5.0)
    assert summary["std"] == pytest.approx(0.0)
    np.testing.assert_allclose(summary["ci95"], [5.0, 5.0])


def test_end_to_end_emg_report_is_impact_aligned_and_scope_limited(tmp_path):
    sim_fs = 100.0
    emg_fs = 1000.0
    sim_time = np.arange(240) / sim_fs
    emg_time = np.arange(2400) / emg_fs
    sim_impact = np.asarray([100, 105], dtype=np.int32)
    emg_impact = np.asarray([1000, 1050], dtype=np.int32)
    simulation_trials = []
    emg_trials = []
    for trial in range(2):
        impact_s = sim_impact[trial] / sim_fs
        bic = np.exp(-0.5 * ((sim_time - (impact_s - 0.08)) / 0.13) ** 2)
        tri = np.exp(-0.5 * ((sim_time - (impact_s + 0.07)) / 0.16) ** 2)
        simulation_trials.append(np.stack([bic, 0.8 * bic, tri], axis=1))
        bic_emg = np.interp(emg_time, sim_time, bic)
        tri_emg = np.interp(emg_time, sim_time, tri)
        carriers = np.stack(
            [
                bic_emg * np.sin(2.0 * np.pi * 85.0 * emg_time),
                tri_emg * np.sin(2.0 * np.pi * 105.0 * emg_time),
            ],
            axis=1,
        )
        emg_trials.append(carriers)
    simulation_path = tmp_path / "simulation.npz"
    emg_path = tmp_path / "emg.npz"
    mapping_path = tmp_path / "mapping.json"
    np.savez_compressed(
        simulation_path,
        muscle_activation=np.asarray(simulation_trials),
        actuator_names=np.asarray(["bic_long", "bic_short", "tri"]),
        sampling_rate_hz=np.asarray(sim_fs),
        impact_frame=sim_impact,
        trial_uid=np.asarray(["trial-a", "trial-b"]),
        subject_uid=np.asarray(["subject-a", "subject-a"]),
        session_uid=np.asarray(["heldout-session", "heldout-session"]),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray(["train-session"]),
        phase_id=np.broadcast_to(
            np.where(sim_time[None, :] < sim_impact[:, None] / sim_fs, 2, 4),
            (2, sim_time.size),
        ).astype(np.int32),
        **_activation_contract_fields(3),
    )
    np.savez_compressed(
        emg_path,
        # Reverse the physical row order: pairing must use trial_uid, not row index.
        emg=np.asarray(emg_trials)[::-1],
        channel_names=np.asarray(["biceps", "triceps"]),
        sampling_rate_hz=np.asarray(emg_fs),
        impact_frame=emg_impact[::-1],
        trial_uid=np.asarray(["trial-b", "trial-a"]),
        subject_uid=np.asarray(["subject-a", "subject-a"]),
        session_uid=np.asarray(["heldout-session", "heldout-session"]),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray(["train-session"]),
    )
    mapping_path.write_text(json.dumps(_mapping()), encoding="utf-8")

    report = evaluate_emg_validation(
        simulation_npz=simulation_path,
        emg_npz=emg_path,
        mapping_json=mapping_path,
        **_expected_policy_binding(),
        pre_impact_s=0.4,
        post_impact_s=0.6,
        output_samples=81,
        synergy_rank=2,
        filter_config=EmgFilterConfig(bandpass_high_hz=300.0),
        bootstrap_samples=50,
        seed=7,
    )

    assert report["claim_scope"] == "right_upper_limb_local"
    assert report["paired_trials"] == 2
    assert report["trial_binding"]["binding_verified"] == 1.0
    assert report["trial_binding"]["emg_reordered_to_simulation"] is True
    assert report["trial_binding"]["trial_uids"] == ["trial-a", "trial-b"]
    assert report["policy_evidence"]["binding_verified"] == 1.0
    assert report["policy_evidence"]["formal_basis_role"] == "formal_analysis_basis_only"
    assert report["mapped_channels"] == ["biceps", "triceps"]
    assert len(report["claim_limitations"]) >= 3
    assert report["synergy"]["rank"] == 2
    assert set(report["phase_activation"]) == {"2", "4"}
    assert report["envelope_metrics"]["bootstrap_design"] == {
        "method": "hierarchical_subject_trial_cluster_bootstrap_v1",
        "subject_count": 1,
        "trial_count": 2,
        "bootstrap_samples": 50,
        "channels_within_trial_resampled_together": True,
    }
    assert set(report["input_fingerprints"]) == {
        "simulation_npz_sha256",
        "emg_npz_sha256",
        "mapping_json_sha256",
    }


def test_emg_report_rejects_trial_identity_mismatch(tmp_path):
    simulation_path = tmp_path / "simulation.npz"
    emg_path = tmp_path / "emg.npz"
    mapping_path = tmp_path / "mapping.json"
    common_identity = {
        "subject_uid": np.asarray(["subject-a"]),
        "session_uid": np.asarray(["heldout-session"]),
        "dataset_split": np.asarray("heldout"),
        "training_session_uid": np.asarray(["train-session"]),
    }
    np.savez_compressed(
        simulation_path,
        muscle_activation=np.ones((1, 200, 3)),
        actuator_names=np.asarray(["bic_long", "bic_short", "tri"]),
        sampling_rate_hz=np.asarray(100.0),
        impact_frame=np.asarray([100], dtype=np.int32),
        trial_uid=np.asarray(["simulation-trial"]),
        **_activation_contract_fields(3),
        **common_identity,
    )
    np.savez_compressed(
        emg_path,
        emg=np.ones((1, 2000, 2)),
        channel_names=np.asarray(["biceps", "triceps"]),
        sampling_rate_hz=np.asarray(1000.0),
        impact_frame=np.asarray([1000], dtype=np.int32),
        trial_uid=np.asarray(["different-emg-trial"]),
        **common_identity,
    )
    mapping_path.write_text(json.dumps(_mapping()), encoding="utf-8")

    with pytest.raises(ValueError, match="trial_uid sets differ"):
        evaluate_emg_validation(
            simulation_npz=simulation_path,
            emg_npz=emg_path,
            mapping_json=mapping_path,
            **_expected_policy_binding(),
            pre_impact_s=0.4,
            post_impact_s=0.4,
            output_samples=41,
            filter_config=EmgFilterConfig(bandpass_high_hz=300.0),
            bootstrap_samples=10,
        )


def test_emg_report_rejects_excitation_relabelled_as_activation(tmp_path):
    simulation_path = tmp_path / "simulation.npz"
    emg_path = tmp_path / "emg.npz"
    mapping_path = tmp_path / "mapping.json"
    identity = {
        "trial_uid": np.asarray(["trial"]),
        "subject_uid": np.asarray(["subject"]),
        "session_uid": np.asarray(["heldout-session"]),
        "dataset_split": np.asarray("heldout"),
        "training_session_uid": np.asarray(["train-session"]),
    }
    contract = _activation_contract_fields(3)
    contract["muscle_activation_source"] = np.asarray("teacher_ctrl_physical")
    np.savez_compressed(
        simulation_path,
        muscle_activation=np.ones((1, 200, 3)),
        actuator_names=np.asarray(["bic_long", "bic_short", "tri"]),
        sampling_rate_hz=np.asarray(100.0),
        impact_frame=np.asarray([100], dtype=np.int32),
        **contract,
        **identity,
    )
    np.savez_compressed(
        emg_path,
        emg=np.ones((1, 2000, 2)),
        channel_names=np.asarray(["biceps", "triceps"]),
        sampling_rate_hz=np.asarray(1000.0),
        impact_frame=np.asarray([1000], dtype=np.int32),
        **identity,
    )
    mapping_path.write_text(json.dumps(_mapping()), encoding="utf-8")

    with pytest.raises(ValueError, match="must come from transition_state.data.act"):
        evaluate_emg_validation(
            simulation_npz=simulation_path,
            emg_npz=emg_path,
            mapping_json=mapping_path,
            **_expected_policy_binding(),
            pre_impact_s=0.4,
            post_impact_s=0.4,
            output_samples=41,
            filter_config=EmgFilterConfig(bandpass_high_hz=300.0),
            bootstrap_samples=10,
        )
