"""CPU-only tests for explicitly unpaired sEMG cohort evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import musclemimic.evaluation.emg_eval as paired_emg_module
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
)
from musclemimic.evaluation.emg_cohort_eval import (
    UNPAIRED_COMPARISON_DESIGN,
    evaluate_emg_cohort_validation,
)
from musclemimic.evaluation.jidian_emg_import import import_jidian_emg
from tests.unit.test_jidian_emg_import import _make_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = (
    REPOSITORY_ROOT / "configs" / "physiology" / "emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)
TAXONOMY_PATH = REPOSITORY_ROOT / "configs" / "physiology" / "myofullbody_354_muscle_taxonomy_audit_v1.json"

POLICY_CHECKPOINT = "1" * 64
POLICY_PROMOTION = "2" * 64
FORMAL_BASIS = "3" * 64
PROCESSING_MANIFEST = "4" * 64
SOURCE_PROVENANCE = "5" * 64
SELECTION_MANIFEST = "6" * 64
COMPARISON_SET_UID = "forehand-high-clear-unpaired-heldout-v1"
ACTION_ID = "forehand_high_clear"


def _expected_policy_binding() -> dict[str, str]:
    return {
        "expected_policy_checkpoint_fingerprint": POLICY_CHECKPOINT,
        "expected_policy_promotion_fingerprint": POLICY_PROMOTION,
        "expected_formal_synergy_basis_fingerprint": FORMAL_BASIS,
    }


def _write_unpaired_inputs(
    tmp_path: Path,
    *,
    simulation_design: str = UNPAIRED_COMPARISON_DESIGN,
    emg_design: str = UNPAIRED_COMPARISON_DESIGN,
    simulation_action_id: str = ACTION_ID,
    emg_action_id: str = ACTION_ID,
    simulation_comparison_set_uid: str = COMPARISON_SET_UID,
    emg_comparison_set_uid: str = COMPARISON_SET_UID,
    training_session_uid: str = "policy-training-session",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    actuator_names = [row["name"] for row in taxonomy["ordered_actuators"]]
    actuator_index = {name: index for index, name in enumerate(actuator_names)}

    sim_fs = 100.0
    emg_fs = 1000.0
    sim_time = np.arange(240) / sim_fs
    emg_time = np.arange(2400) / emg_fs
    sim_impact = np.asarray([100, 106], dtype=np.int32)
    emg_impact = np.asarray([1000, 1040, 1080], dtype=np.int32)

    simulation = np.zeros((2, sim_time.size, len(actuator_names)), dtype=np.float64)
    for trial, impact_frame in enumerate(sim_impact.tolist()):
        impact_s = impact_frame / sim_fs
        early = np.exp(-0.5 * ((sim_time - (impact_s - 0.12)) / 0.13) ** 2)
        late = np.exp(-0.5 * ((sim_time - (impact_s + 0.11)) / 0.16) ** 2)
        for channel_offset, channel in enumerate(mapping["channels"][1:]):
            early_weight = 0.15 + 0.7 * ((channel_offset % 5) / 4.0)
            signal = early_weight * early + (1.0 - early_weight) * late
            signal = signal / np.max(signal)
            for actuator in channel["simulation_actuators"]:
                simulation[trial, :, actuator_index[actuator]] = signal

    emg = np.empty((3, emg_time.size, 16), dtype=np.float64)
    for trial, impact_frame in enumerate(emg_impact.tolist()):
        impact_s = impact_frame / emg_fs
        early = np.exp(-0.5 * ((emg_time - (impact_s - 0.10)) / 0.14) ** 2)
        late = np.exp(-0.5 * ((emg_time - (impact_s + 0.13)) / 0.17) ** 2)
        emg[trial, :, 0] = 0.03 + 0.45 * early
        for channel_offset in range(15):
            early_weight = 0.18 + 0.64 * ((channel_offset % 5) / 4.0)
            signal = early_weight * early + (1.0 - early_weight) * late
            emg[trial, :, channel_offset + 1] = 0.02 + 0.9 * signal / np.max(signal)

    simulation_path = tmp_path / "simulation_unpaired.npz"
    emg_path = tmp_path / "emg_unpaired.npz"
    np.savez_compressed(
        simulation_path,
        muscle_activation=simulation,
        actuator_names=np.asarray(actuator_names),
        physical_signal_schema_version=np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
        muscle_activation_source=np.asarray(MUSCLE_ACTIVATION_SOURCE),
        muscle_activation_semantics=np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
        muscle_activation_roundoff_policy=np.asarray(UNIT_INTERVAL_ROUNDOFF_POLICY),
        activation_valid_mask=np.ones((len(actuator_names),), dtype=bool),
        sampling_rate_hz=np.asarray(sim_fs),
        impact_frame=sim_impact,
        policy_decoder_type=np.asarray("direct"),
        policy_checkpoint_fingerprint=np.asarray(POLICY_CHECKPOINT),
        policy_promotion_fingerprint=np.asarray(POLICY_PROMOTION),
        formal_synergy_basis_fingerprint=np.asarray(FORMAL_BASIS),
        analysis_synergy_basis_fingerprint=np.asarray(FORMAL_BASIS),
        model_taxonomy_id=np.asarray(mapping["model_binding"]["taxonomy_id"]),
        model_taxonomy_fingerprint=np.asarray(mapping["model_binding"]["taxonomy_fingerprint"]),
        runtime_model_hash=np.asarray(mapping["model_binding"]["runtime_model_hash"]),
        actuator_schema_hash=np.asarray(mapping["model_binding"]["actuator_schema_hash"]),
        comparison_design=np.asarray(simulation_design),
        comparison_set_uid=np.asarray(simulation_comparison_set_uid),
        action_id=np.asarray(simulation_action_id),
        handedness=np.asarray("right"),
        trial_uid=np.asarray(["simulation-trial-a", "simulation-trial-b"]),
        subject_uid=np.asarray(["simulation-policy", "simulation-policy"]),
        session_uid=np.asarray(["simulation-heldout", "simulation-heldout"]),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray([training_session_uid]),
    )
    sides = [channel["side"] for channel in mapping["channels"]]
    muscle_slugs = [channel["muscle_slug"] for channel in mapping["channels"]]
    channel_names = [channel["emg_channel"] for channel in mapping["channels"]]
    np.savez_compressed(
        emg_path,
        import_schema_version=np.asarray("jidian_emg_import_v1"),
        emg=emg,
        emg_signal_kind=np.asarray("preprocessed_normalized_envelope_v1"),
        channel_names=np.asarray(channel_names),
        stream_channel_ids=np.arange(1, 17, dtype=np.int16),
        sides=np.asarray(sides),
        muscle_slugs=np.asarray(muscle_slugs),
        sampling_rate_hz=np.asarray(emg_fs),
        impact_frame=emg_impact,
        processing_manifest_schema_version=np.asarray("jidian_semg_processing_manifest_v2"),
        processing_manifest_sha256=np.asarray(PROCESSING_MANIFEST),
        source_provenance_sha256=np.asarray(SOURCE_PROVENANCE),
        selection_manifest_sha256=np.asarray(SELECTION_MANIFEST),
        channel_profile_id=np.asarray(mapping["profile_binding"]["profile_id"]),
        channel_profile_version=np.asarray(mapping["profile_binding"]["profile_version"], dtype=np.int64),
        channel_profile_sha256=np.asarray(mapping["profile_binding"]["profile_sha256"]),
        normalization_method=np.asarray("mvc"),
        processing_fallback_method=np.asarray("none"),
        acquired_channel_count=np.asarray(16, dtype=np.int64),
        comparable_channel_count=np.asarray(15, dtype=np.int64),
        excluded_sensor_ids=np.asarray([1], dtype=np.int16),
        comparison_design=np.asarray(emg_design),
        comparison_set_uid=np.asarray(emg_comparison_set_uid),
        action_id=np.asarray(emg_action_id),
        handedness=np.asarray("right"),
        trial_uid=np.asarray(["emg-trial-a", "emg-trial-b", "emg-trial-c"]),
        subject_uid=np.asarray(["human-subject"] * 3),
        session_uid=np.asarray(["human-heldout-session"] * 3),
        dataset_split=np.asarray("heldout"),
        training_session_uid=np.asarray([training_session_uid]),
    )
    return simulation_path, emg_path


def _evaluate(
    simulation_path: Path,
    emg_path: Path,
    *,
    mapping_path: Path = MAPPING_PATH,
    allow_provisional=True,
    synergy_rank: int = 2,
    pre_impact_s: float = 0.4,
    post_impact_s: float = 0.6,
    output_samples: int = 61,
):
    return evaluate_emg_cohort_validation(
        simulation_npz=simulation_path,
        emg_npz=emg_path,
        mapping_json=mapping_path,
        **_expected_policy_binding(),
        synergy_rank=synergy_rank,
        initialization_seeds=[7, 3, 11],
        pre_impact_s=pre_impact_s,
        post_impact_s=post_impact_s,
        output_samples=output_samples,
        max_iter=300,
        tol=1e-7,
        allow_provisional_mapping=allow_provisional,
    )


def test_unpaired_cohorts_allow_different_trials_and_are_deterministic(
    tmp_path,
    monkeypatch,
):
    simulation_path, emg_path = _write_unpaired_inputs(tmp_path)

    def fail_if_paired_preprocessing_is_called(*args, **kwargs):
        raise AssertionError("unpaired preprocessed EMG was filtered or normalized again")

    monkeypatch.setattr(
        paired_emg_module,
        "preprocess_emg",
        fail_if_paired_preprocessing_is_called,
    )
    monkeypatch.setattr(
        paired_emg_module,
        "normalize_envelopes",
        fail_if_paired_preprocessing_is_called,
    )
    first = _evaluate(simulation_path, emg_path)
    second = _evaluate(simulation_path, emg_path)

    assert first["cohort_contract"]["simulation_trial_count"] == 2
    assert first["cohort_contract"]["emg_trial_count"] == 3
    assert first["cohort_contract"]["pairing_performed"] is False
    assert first["cohort_contract"]["trial_uid_used_for_pairing"] is False
    assert first["rank_contract"]["initialization_seeds"] == [3, 7, 11]
    assert first["nmf"] == second["nmf"]
    assert first["nmf"]["simulation"]["global_vaf"] > 0.95
    assert first["nmf"]["emg"]["global_vaf"] > 0.95
    assert len(first["nmf"]["simulation"]["per_channel_vaf"]) == 15
    assert len(first["nmf"]["emg"]["per_channel_vaf"]) == 15
    assert first["nmf"]["simulation"]["basis_condition_number"] is not None
    assert first["nmf"]["emg"]["basis_condition_number"] is not None
    assert first["nmf"]["hungarian_basis_similarity"]["mean_weight_cosine_similarity"] > 0.95
    assert first["preprocessing"]["evaluator_filter_applied"] is False
    assert first["preprocessing"]["evaluator_emg_normalization_applied"] is False
    assert first["uncertainty"]["single_measured_subject_limitation"] is True
    assert first["uncertainty"]["confidence_intervals_computed"] is False


def test_strict_jidian_import_output_flows_directly_into_unpaired_evaluator(tmp_path):
    fixture = _make_fixture(tmp_path / "jidian_source")
    imported_emg_path = tmp_path / "strict_unpaired_import.npz"
    import_jidian_emg(fixture["selection"], imported_emg_path)
    simulation_path, _ = _write_unpaired_inputs(
        tmp_path / "simulation_source",
        simulation_action_id="forehand_smash",
        emg_action_id="forehand_smash",
        simulation_comparison_set_uid="synthetic-cohort-v1",
        emg_comparison_set_uid="synthetic-cohort-v1",
        training_session_uid="session-registry-training-a",
    )
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    with np.load(imported_emg_path, allow_pickle=False) as imported:
        mapping["profile_binding"]["profile_sha256"] = str(imported["channel_profile_sha256"])
    integration_mapping_path = tmp_path / "integration_mapping.json"
    integration_mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    report = _evaluate(
        simulation_path,
        imported_emg_path,
        mapping_path=integration_mapping_path,
        synergy_rank=1,
        pre_impact_s=0.2,
        post_impact_s=0.3,
        output_samples=51,
    )

    assert report["mapping_runtime_binding"]["strict_import_verified"] == 1.0
    assert report["cohort_contract"]["comparison_set_uid"] == "synthetic-cohort-v1"
    assert report["cohort_contract"]["trial_uid_used_for_pairing"] is False
    assert report["nmf"]["simulation"]["rank"] == 1
    assert report["nmf"]["emg"]["rank"] == 1


def test_unpaired_report_explicitly_omits_all_paired_metrics(tmp_path):
    simulation_path, emg_path = _write_unpaired_inputs(tmp_path)
    report = _evaluate(simulation_path, emg_path)

    expected_unavailable = {
        "envelope_correlation",
        "normalized_dtw",
        "onset_error_s",
        "peak_timing_error_s",
        "nmf_coefficient_h_correlation",
        "shared_phase_comparison",
    }
    assert set(report["metric_availability"]) == expected_unavailable
    assert all(
        item["status"] == "unavailable" and item["available"] is False and item["reason"]
        for item in report["metric_availability"].values()
    )
    assert "envelope_metrics" not in report
    assert "phase_activation" not in report
    assert "coefficient_correlation" not in json.dumps(report, sort_keys=True)


def test_unpaired_cohort_rejects_provisional_mapping_without_opt_in(tmp_path):
    simulation_path, emg_path = _write_unpaired_inputs(tmp_path)

    with pytest.raises(ValueError, match="provisional.*exploratory-only"):
        _evaluate(simulation_path, emg_path, allow_provisional=False)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"emg_design": "paired_same_reference_v1"}, "both cohort inputs must declare"),
        ({"emg_action_id": "forehand_smash"}, "action_id values differ"),
        (
            {"emg_comparison_set_uid": "different-comparison-set"},
            "comparison_set_uid values differ",
        ),
    ],
)
def test_unpaired_cohort_rejects_cross_cohort_contract_mismatch(
    tmp_path,
    overrides,
    error,
):
    simulation_path, emg_path = _write_unpaired_inputs(tmp_path, **overrides)

    with pytest.raises(ValueError, match=error):
        _evaluate(simulation_path, emg_path)
