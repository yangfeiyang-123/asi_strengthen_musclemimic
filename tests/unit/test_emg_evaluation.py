"""CPU-only tests for the local sEMG validation framework."""

from __future__ import annotations

import json

import numpy as np
import pytest

import musclemimic.evaluation.emg_eval as emg_eval_module
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
PROFILE_FINGERPRINT = "5" * 64
TAXONOMY_FINGERPRINT = "6" * 64
RUNTIME_MODEL_HASH = "7" * 64
ACTUATOR_SCHEMA_HASH = "8" * 64
PROCESSING_MANIFEST_HASH = "9" * 64
SOURCE_PROVENANCE_HASH = "a" * 64
REFERENCE_A = "b" * 64
REFERENCE_B = "c" * 64

V2_CHANNEL_IDENTITIES = [
    ("right", "upper_trapezius"),
    ("right", "anterior_deltoid"),
    ("right", "posterior_deltoid"),
    ("right", "pectoralis_major_clavicular"),
    ("right", "latissimus_dorsi"),
    ("right", "triceps_lateral"),
    ("right", "pronator_teres"),
    ("right", "extensor_carpi_radialis"),
    ("right", "external_oblique"),
    ("left", "external_oblique"),
    ("right", "vastus_lateralis"),
    ("left", "vastus_lateralis"),
    ("right", "biceps_femoris_long_head"),
    ("left", "biceps_femoris_long_head"),
    ("right", "gastrocnemius_medialis"),
    ("left", "gastrocnemius_medialis"),
]


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


def _mapping_v2(*, review_status: str = "provisional") -> dict:
    channels = []
    for sensor_id, (side, muscle_slug) in enumerate(V2_CHANNEL_IDENTITIES, start=1):
        identity = {
            "sensor_id": sensor_id,
            "emg_channel": f"S{sensor_id} {side}:{muscle_slug}",
            "side": side,
            "muscle_slug": muscle_slug,
        }
        if sensor_id == 1:
            identity.update(
                {
                    "mapping_status": "excluded_no_verified_model_homolog",
                    "simulation_actuators": [],
                    "weights": [],
                    "exclusion_reason": "no verified upper-trapezius homolog in this taxonomy",
                    "mapping_uncertainty": "excluded rather than approximated",
                }
            )
        else:
            identity.update(
                {
                    "mapping_status": "mapped",
                    "simulation_actuators": [f"actuator_{sensor_id:02d}"],
                    "weights": [1.0],
                    "mapping_confidence": ("low" if review_status == "verified" else "provisional"),
                    "mapping_uncertainty": "synthetic unit-test mapping",
                }
            )
        channels.append(identity)
    return {
        "schema_version": "emg_observation_mapping_v2",
        "mapping_id": "badminton_synergy_16_v2_test_mapping",
        "validation_scope": "whole_body_surface_emg_15_of_16",
        "review_status": review_status,
        "review_evidence": ["unit-test-review"] if review_status == "verified" else [],
        "profile_binding": {
            "profile_id": "badminton_synergy_16_v2",
            "profile_version": 2,
            "profile_sha256": PROFILE_FINGERPRINT,
            "intended_handedness": "right",
            "acquired_channel_count": 16,
            "comparable_channel_count": 15,
        },
        "model_binding": {
            "taxonomy_id": "myofullbody_unsuffixed_right_v1",
            "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
            "runtime_model_hash": RUNTIME_MODEL_HASH,
            "actuator_schema_hash": ACTUATOR_SCHEMA_HASH,
        },
        "normalization": "mvc",
        "channels": channels,
        "cocontraction_pairs": [],
    }


def _v2_actuator_names() -> list[str]:
    return [f"actuator_{sensor_id:02d}" for sensor_id in range(2, 17)]


def _v2_channel_names() -> list[str]:
    return [
        f"S{sensor_id} {side}:{muscle_slug}"
        for sensor_id, (side, muscle_slug) in enumerate(V2_CHANNEL_IDENTITIES, start=1)
    ]


def _activation_contract_fields(width: int) -> dict[str, np.ndarray]:
    return {
        "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        "muscle_activation_source": np.asarray(MUSCLE_ACTIVATION_SOURCE),
        "muscle_activation_semantics": np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
        "muscle_activation_roundoff_policy": np.asarray(UNIT_INTERVAL_ROUNDOFF_POLICY),
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


def test_v2_mapping_is_exact_and_provisional_use_requires_explicit_opt_in():
    mapping = _mapping_v2()
    with pytest.raises(ValueError, match="exploratory-only"):
        validate_emg_mapping(
            mapping,
            emg_channel_names=_v2_channel_names(),
            actuator_names=_v2_actuator_names(),
        )

    contract = validate_emg_mapping(
        mapping,
        emg_channel_names=_v2_channel_names(),
        actuator_names=_v2_actuator_names(),
        allow_provisional_mapping=True,
    )
    assert contract["exploratory_only"] is True
    assert contract["profile_binding"]["acquired_channel_count"] == 16
    assert contract["profile_binding"]["comparable_channel_count"] == 15
    assert contract["profile_binding"]["excluded_sensor_ids"] == [1]
    assert contract["channels"][0]["mapping_status"] == ("excluded_no_verified_model_homolog")

    invalid = json.loads(json.dumps(mapping))
    invalid["channels"][0]["mapping_status"] = "mapped"
    with pytest.raises(ValueError, match="sensor 1 must be explicitly excluded"):
        validate_emg_mapping(invalid, allow_provisional_mapping=True)

    invalid = json.loads(json.dumps(mapping))
    invalid["channels"][1]["weights"] = [0.5]
    with pytest.raises(ValueError, match="sum to one"):
        validate_emg_mapping(invalid, allow_provisional_mapping=True)

    verified = validate_emg_mapping(_mapping_v2(review_status="verified"))
    assert verified["exploratory_only"] is False


def test_preprocessed_envelope_contract_requires_complete_provenance():
    payload = {
        "emg_signal_kind": np.asarray("preprocessed_normalized_envelope_v1"),
        "emg": np.ones((1, 20, 16)),
        "processing_manifest_schema_version": np.asarray("semg_processing_v2"),
        "processing_manifest_sha256": np.asarray(PROCESSING_MANIFEST_HASH),
        "channel_profile_id": np.asarray("badminton_synergy_16_v2"),
        "channel_profile_version": np.asarray(2, dtype=np.int32),
        "channel_profile_sha256": np.asarray(PROFILE_FINGERPRINT),
        "handedness": np.asarray("right"),
        "normalization_method": np.asarray("mvc"),
        "processing_fallback_method": np.asarray("none"),
    }

    with pytest.raises(ValueError, match="source_provenance_sha256"):
        emg_eval_module._validate_preprocessed_emg_contract(
            payload,
            mapping={"normalization": "mvc"},
        )


def test_paired_binding_fails_closed_for_unpaired_design_and_reference_mismatch():
    identity = {
        "trial_uid": np.asarray(["trial-a"]),
        "subject_uid": np.asarray(["subject-a"]),
        "session_uid": np.asarray(["heldout-session"]),
        "dataset_split": np.asarray("heldout"),
        "training_session_uid": np.asarray(["train-session"]),
        "action_id": np.asarray("forehand_high_clear"),
        "comparison_set_uid": np.asarray("paired-forehand-high-clear-v1"),
    }
    simulation = {
        **identity,
        "comparison_design": np.asarray("unpaired_action_cohort_v1"),
        "reference_trial_fingerprint": np.asarray([REFERENCE_A]),
    }
    emg = dict(simulation)
    with pytest.raises(ValueError, match="only supports comparison_design"):
        emg_eval_module._bind_paired_trials(
            simulation,
            emg,
            require_reference_evidence=True,
        )

    simulation["comparison_design"] = np.asarray("paired_same_reference_v1")
    emg["comparison_design"] = np.asarray("paired_same_reference_v1")
    emg["action_id"] = np.asarray("forehand_smash")
    with pytest.raises(ValueError, match="paired action_id values differ"):
        emg_eval_module._bind_paired_trials(
            simulation,
            emg,
            require_reference_evidence=True,
        )
    emg["action_id"] = np.asarray("forehand_high_clear")
    emg["comparison_set_uid"] = np.asarray("different-paired-set")
    with pytest.raises(ValueError, match="paired comparison_set_uid values differ"):
        emg_eval_module._bind_paired_trials(
            simulation,
            emg,
            require_reference_evidence=True,
        )
    emg["comparison_set_uid"] = np.asarray("paired-forehand-high-clear-v1")
    emg["reference_trial_fingerprint"] = np.asarray([REFERENCE_B])
    with pytest.raises(ValueError, match="fingerprint values differ"):
        emg_eval_module._bind_paired_trials(
            simulation,
            emg,
            require_reference_evidence=True,
        )


def test_paired_binding_fails_closed_without_shared_reference_fingerprint():
    # Legacy v1 inputs (no comparison_design, no reference_trial_fingerprint) must
    # not be accepted as paired evidence just because they share trial_uid strings.
    identity = {
        "trial_uid": np.asarray(["trial-a"]),
        "subject_uid": np.asarray(["subject-a"]),
        "session_uid": np.asarray(["heldout-session"]),
        "dataset_split": np.asarray("heldout"),
        "training_session_uid": np.asarray(["train-session"]),
    }
    simulation = dict(identity)
    emg = dict(identity)
    with pytest.raises(ValueError, match="requires reference_trial_fingerprint"):
        emg_eval_module._bind_paired_trials(simulation, emg)

    # Supplying the shared reference fingerprint on both sides unlocks pairing.
    simulation["reference_trial_fingerprint"] = np.asarray([REFERENCE_A])
    emg["reference_trial_fingerprint"] = np.asarray([REFERENCE_A])
    order, binding = emg_eval_module._bind_paired_trials(simulation, emg)
    assert list(order) == [0]
    assert binding["reference_evidence_verified"] == 1.0


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
        reference_trial_fingerprint=np.asarray([REFERENCE_A, REFERENCE_B]),
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
        # Reversed to match the reversed trial_uid order: pairing must line the
        # shared reference fingerprint up per trial_uid, not per physical row.
        reference_trial_fingerprint=np.asarray([REFERENCE_B, REFERENCE_A]),
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
    assert report["trial_binding"]["reference_evidence_verified"] == 1.0
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


def test_v2_preprocessed_envelope_is_projected_without_refilter_or_renormalize(
    tmp_path,
    monkeypatch,
):
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
        early = np.exp(-0.5 * ((sim_time - (impact_s - 0.11 - 0.01 * trial)) / 0.12) ** 2)
        late = np.exp(-0.5 * ((sim_time - (impact_s + 0.10 + 0.01 * trial)) / 0.15) ** 2)
        channels = []
        for channel in range(15):
            early_weight = 0.15 + 0.7 * ((channel % 5) / 4.0)
            signal = early_weight * early + (1.0 - early_weight) * late
            channels.append(signal / np.max(signal))
        simulation_trial = np.stack(channels, axis=1)
        simulation_trials.append(simulation_trial)
        mapped_emg = np.stack(
            [0.02 + 0.9 * np.interp(emg_time, sim_time, simulation_trial[:, channel]) for channel in range(15)],
            axis=1,
        )
        excluded_upper_trapezius = (0.03 + 0.5 * np.interp(emg_time, sim_time, early))[:, None]
        emg_trials.append(np.concatenate([excluded_upper_trapezius, mapped_emg], axis=1))

    simulation_path = tmp_path / "simulation_v2.npz"
    emg_path = tmp_path / "emg_v2.npz"
    mapping_path = tmp_path / "mapping_v2.json"
    np.savez_compressed(
        simulation_path,
        muscle_activation=np.asarray(simulation_trials),
        actuator_names=np.asarray(_v2_actuator_names()),
        sampling_rate_hz=np.asarray(sim_fs),
        impact_frame=sim_impact,
        trial_uid=np.asarray(["trial-a", "trial-b"]),
        subject_uid=np.asarray(["subject-a", "subject-a"]),
        session_uid=np.asarray(["heldout-session", "heldout-session"]),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray(["train-session"]),
        comparison_design=np.asarray("paired_same_reference_v1"),
        comparison_set_uid=np.asarray("paired-forehand-high-clear-v1"),
        action_id=np.asarray("forehand_high_clear"),
        reference_trial_fingerprint=np.asarray([REFERENCE_A, REFERENCE_B]),
        model_taxonomy_id=np.asarray("myofullbody_unsuffixed_right_v1"),
        model_taxonomy_fingerprint=np.asarray(TAXONOMY_FINGERPRINT),
        runtime_model_hash=np.asarray(RUNTIME_MODEL_HASH),
        actuator_schema_hash=np.asarray(ACTUATOR_SCHEMA_HASH),
        handedness=np.asarray("right"),
        **_activation_contract_fields(15),
    )
    sides = [side for side, _ in V2_CHANNEL_IDENTITIES]
    muscle_slugs = [muscle for _, muscle in V2_CHANNEL_IDENTITIES]
    np.savez_compressed(
        emg_path,
        # Reverse physical rows; every row-bound fingerprint must follow it.
        emg=np.asarray(emg_trials)[::-1],
        channel_names=np.asarray(_v2_channel_names()),
        sampling_rate_hz=np.asarray(emg_fs),
        impact_frame=emg_impact[::-1],
        trial_uid=np.asarray(["trial-b", "trial-a"]),
        subject_uid=np.asarray(["subject-a", "subject-a"]),
        session_uid=np.asarray(["heldout-session", "heldout-session"]),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray(["train-session"]),
        comparison_design=np.asarray("paired_same_reference_v1"),
        comparison_set_uid=np.asarray("paired-forehand-high-clear-v1"),
        action_id=np.asarray("forehand_high_clear"),
        reference_trial_fingerprint=np.asarray([REFERENCE_B, REFERENCE_A]),
        emg_signal_kind=np.asarray("preprocessed_normalized_envelope_v1"),
        processing_manifest_schema_version=np.asarray("semg_processing_v2"),
        processing_manifest_sha256=np.asarray(PROCESSING_MANIFEST_HASH),
        source_provenance_sha256=np.asarray(SOURCE_PROVENANCE_HASH),
        channel_profile_id=np.asarray("badminton_synergy_16_v2"),
        channel_profile_version=np.asarray(2, dtype=np.int32),
        channel_profile_sha256=np.asarray(PROFILE_FINGERPRINT),
        handedness=np.asarray("right"),
        normalization_method=np.asarray("mvc"),
        processing_fallback_method=np.asarray("none"),
        stream_channel_ids=np.arange(1, 17, dtype=np.int16),
        sides=np.asarray(sides),
        muscle_slugs=np.asarray(muscle_slugs),
    )
    mapping_path.write_text(json.dumps(_mapping_v2()), encoding="utf-8")

    def fail_if_raw_preprocessing_is_called(*args, **kwargs):
        raise AssertionError("preprocessed envelope was sent through the raw EMG filter")

    original_normalize = emg_eval_module.normalize_envelopes
    normalization_calls = []

    def tracked_normalize(values, *, method, mvc_values=None):
        normalization_calls.append(method)
        return original_normalize(values, method=method, mvc_values=mvc_values)

    monkeypatch.setattr(emg_eval_module, "preprocess_emg", fail_if_raw_preprocessing_is_called)
    monkeypatch.setattr(emg_eval_module, "normalize_envelopes", tracked_normalize)

    report = evaluate_emg_validation(
        simulation_npz=simulation_path,
        emg_npz=emg_path,
        mapping_json=mapping_path,
        **_expected_policy_binding(),
        pre_impact_s=0.4,
        post_impact_s=0.6,
        output_samples=61,
        synergy_rank=2,
        bootstrap_samples=20,
        seed=11,
        allow_provisional_mapping=True,
    )

    # The only normalization call is the simulation's timing-scale normalization.
    assert normalization_calls == ["per_trial_peak"]
    assert report["claim_scope"] == "whole_body_surface_emg_15_of_16"
    assert report["exploratory_only"] is True
    assert report["mapped_channels"] == _v2_channel_names()[1:]
    assert report["trial_binding"]["reference_evidence_verified"] == 1.0
    assert report["trial_binding"]["emg_reordered_to_simulation"] is True
    assert report["trial_binding"]["action_id"] == "forehand_high_clear"
    assert report["trial_binding"]["comparison_set_uid"] == ("paired-forehand-high-clear-v1")
    assert report["mapping_runtime_binding"]["profile_binding_verified"] == 1.0
    assert report["mapping_runtime_binding"]["model_binding_verified"] == 1.0
    assert report["preprocessing"]["emg_filter"] is None
    assert report["preprocessing"]["evaluator_filter_applied"] is False
    assert report["preprocessing"]["evaluator_normalization_applied"] is False
    assert (
        report["preprocessing"]["measurement_processing_contract"]["processing_manifest_sha256"]
        == PROCESSING_MANIFEST_HASH
    )


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
