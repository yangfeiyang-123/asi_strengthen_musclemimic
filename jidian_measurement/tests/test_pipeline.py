from pathlib import Path

import numpy as np
import pytest

from emg.acquisition import collect_action
from emg.dataset import build_synergy_dataset
from emg.event_annotation import annotate_event
from emg.models import SessionMetadata
from emg.mvc import collect_mvc
from emg.offline import preprocess_session
from emg.profiles import BADMINTON_SYNERGY_16_V2, LEGACY_HIGH_CLEAR_6CH
from emg.storage import atomic_save_npz, atomic_write_json, read_json
from emg.synergy import channel_scale, fit_nmf_best, match_synergies, vaf_metrics


def test_dry_run_creates_complete_trial_bundle_and_resumes(tmp_path):
    saved = collect_action(
        tmp_path,
        "TEST001",
        "DRYRUN",
        BADMINTON_SYNERGY_16_V2.profile_id,
        "badminton_primitive_protocol_v1",
        "forehand_high_clear",
        dry_run=True,
        interactive=False,
        target_valid_trials=1,
        action_duration_s=0.2,
        show_preview=False,
        save_legacy_csv=False,
    )
    assert len(saved) == 1
    required = {"raw_emg.npz", "metadata.json", "events.csv", "qc.json", "preview.png"}
    assert required <= {path.name for path in saved[0].iterdir()}
    with np.load(saved[0] / "raw_emg.npz") as raw:
        assert raw["emg_mV"].shape[1] == 16
        np.testing.assert_array_equal(raw["stream_channel_ids"], np.arange(1, 17))
    reviewed_metadata = read_json(saved[0] / "metadata.json")
    assert reviewed_metadata["valid_for_analysis"] is True
    assert reviewed_metadata["error_labels"] == ["correct"]
    assert reviewed_metadata["post_trial_rest"]["reason"] == "target_reached"
    assert reviewed_metadata["post_trial_rest"]["planned_seconds"] == 0.0
    assert reviewed_metadata["post_trial_rest"]["actual_seconds"] == 0.0
    assert reviewed_metadata["post_trial_rest"]["skipped"] is False
    assert collect_action(
        tmp_path, "TEST001", "DRYRUN", BADMINTON_SYNERGY_16_V2.profile_id,
        "badminton_primitive_protocol_v1", "forehand_high_clear", dry_run=True,
        interactive=False, target_valid_trials=1, action_duration_s=0.2,
        save_legacy_csv=False,
    ) == []


def test_new_mvc_rep_persists_hard_qc_and_rest_metadata(tmp_path):
    profile = BADMINTON_SYNERGY_16_V2
    session_metadata = SessionMetadata(
        participant_id="TEST001",
        session_id="MVC_DRYRUN",
        channel_profile_id=profile.profile_id,
        protocol_id="badminton_primitive_protocol_v1",
        handedness="right",
        dominant_leg="right",
    ).to_dict()
    collect_mvc(
        tmp_path,
        "TEST001",
        "MVC_DRYRUN",
        profile.profile_id,
        "badminton_primitive_protocol_v1",
        session_metadata,
        repetitions=1,
        contraction_s=0.2,
        rest_s=60.0,
        window_s=0.05,
        muscle_slugs=["right_upper_trapezius"],
        dry_run=True,
        interactive=False,
    )
    rep = read_json(
        tmp_path
        / "TEST001"
        / "MVC_DRYRUN"
        / "mvc"
        / "right_upper_trapezius"
        / "rep_001"
        / "metadata.json"
    )
    assert rep["valid"] is True
    assert rep["qc"]["hard_qc_pass"] is True
    assert rep["qc"]["hard_failures"] == []
    assert rep["post_repetition_rest"]["reason"] == "target_reached"
    assert rep["post_repetition_rest"]["planned_seconds"] == 0.0
    assert rep["post_repetition_rest"]["actual_seconds"] == 0.0
    assert rep["post_repetition_rest"]["skipped"] is False


def _write_processed_trial(
    root,
    participant,
    session,
    profile,
    trial_index,
    *,
    write_preprocessing_qc=True,
    normalization_method="none",
    normalization_reference_values=None,
    manual_movement_event=True,
):
    trial = root / participant / session / "trials" / "forehand_high_clear" / f"trial_{trial_index:03d}"
    trial.mkdir(parents=True)
    metadata = {
        "participant_id": participant,
        "session_id": session,
        "trial_id": f"trial_{trial_index:03d}",
        "trial_index": trial_index,
        "action_id": "forehand_high_clear",
        "protocol_id": "badminton_primitive_protocol_v1",
        "channel_profile_id": profile.profile_id,
        "channel_profile_snapshot": profile.to_dict(),
        "valid_for_analysis": True,
    }
    atomic_write_json(trial / "metadata.json", metadata)
    atomic_write_json(
        trial / "processing.json",
        {
            "processing_config": {
                "sample_rate_hz": 2000,
                "normalization": normalization_method,
            },
            "normalization_method": normalization_method,
            "fallback_method": None,
            "normalization_reference_scope": (
                "participant_raw_mvc_across_sessions"
                if normalization_method == "mvc"
                else "none"
            ),
            "normalization_reference_values_mV": normalization_reference_values,
            "mvc_reference_manifest": (
                f"{participant}/{session}/preprocessing_mvc_reference.json"
                if normalization_method == "mvc"
                else None
            ),
            "channel_profile_snapshot": profile.to_dict(),
        },
    )
    if write_preprocessing_qc:
        atomic_write_json(trial / "preprocessing_qc.json", {"analysis_ready": True})
    atomic_save_npz(
        trial / "processed_emg.npz",
        normalized_envelope=np.ones((100, len(profile.channels)), dtype=np.float32),
        fs_hz=np.asarray(2000.0),
    )
    atomic_save_npz(
        trial / "raw_emg.npz",
        emg_mV=np.zeros((100, len(profile.channels)), dtype=np.float32),
        time_s=np.arange(100, dtype=np.float64) / 2000.0,
        sample_index=np.arange(100, dtype=np.int64),
        stream_channel_ids=np.asarray(profile.channel_ids, dtype=np.int16),
        fs_hz=np.asarray(2000.0),
    )
    rows = [
        "movement_cue,10,0.005,1,x,software,0.5,x",
    ]
    if manual_movement_event:
        rows.append(
            "movement_start_manual,,,,,unannotated,0,pending"
        )
    rows.append("recording_stop,90,0.045,2,x,software,0.5,x")
    (trial / "events.csv").write_text(
        "event_name,sample_index,emg_time_s,monotonic_time_ns,wall_clock_iso,source,confidence,notes\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    if manual_movement_event:
        annotate_event(
            trial,
            "movement_start_manual",
            sample_index=20,
            source="manual_video",
            confidence=0.9,
            annotator="DEMO_OPERATOR",
            evidence_reference="DEMO_VIDEO#frame=1",
            evidence_sha256="b" * 64,
        )


def test_different_profiles_cannot_be_merged(tmp_path):
    _write_processed_trial(tmp_path, "P1", "S1", BADMINTON_SYNERGY_16_V2, 1)
    _write_processed_trial(tmp_path, "P2", "S2", LEGACY_HIGH_CLEAR_6CH, 1)
    with pytest.raises(ValueError, match="different or reordered"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
        )


def test_only_valid_excludes_preprocessing_qc_failures(tmp_path):
    _write_processed_trial(tmp_path, "P1", "S1", BADMINTON_SYNERGY_16_V2, 1)
    trial = tmp_path / "P1" / "S1" / "trials" / "forehand_high_clear" / "trial_001"
    atomic_write_json(trial / "preprocessing_qc.json", {"analysis_ready": False})
    with pytest.raises(ValueError, match="No trials matched"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
            only_valid=True,
        )


def test_only_valid_rejects_missing_preprocessing_qc(tmp_path):
    _write_processed_trial(
        tmp_path,
        "P1",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
        write_preprocessing_qc=False,
    )
    with pytest.raises(ValueError, match="requires preprocessing_qc"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
            only_valid=True,
        )


def test_different_participants_may_use_distinct_mvc_denominators(tmp_path):
    _write_processed_trial(
        tmp_path,
        "P1",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
        normalization_method="mvc",
        normalization_reference_values=[1.0] * 16,
    )
    _write_processed_trial(
        tmp_path,
        "P2",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
        normalization_method="mvc",
        normalization_reference_values=[2.0] * 16,
    )
    output = build_synergy_dataset(
        tmp_path,
        tmp_path / "out.npz",
        BADMINTON_SYNERGY_16_V2.profile_id,
        "badminton_primitive_protocol_v1",
        only_valid=True,
    )
    metadata = read_json(output.with_suffix(".json"))
    assert sorted(metadata["normalization_references"]) == ["participant:P1", "participant:P2"]
    assert metadata["normalization_references"]["participant:P1"][
        "normalization_reference_values_mV"
    ] == [1.0] * 16
    assert metadata["normalization_references"]["participant:P2"][
        "normalization_reference_values_mV"
    ] == [2.0] * 16
    assert {trial["normalization_reference_key"] for trial in metadata["trials"]} == {
        "participant:P1",
        "participant:P2",
    }


def test_default_crop_requires_evidence_backed_manual_movement_start(tmp_path):
    _write_processed_trial(
        tmp_path,
        "P1",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
        manual_movement_event=False,
    )
    with pytest.raises(ValueError, match="movement_start_manual"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
        )


def test_default_crop_rejects_manual_csv_edits_without_committed_audit(tmp_path):
    _write_processed_trial(
        tmp_path,
        "P1",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
    )
    trial = tmp_path / "P1" / "S1" / "trials" / "forehand_high_clear" / "trial_001"
    (trial / "events.annotation.audit.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing committed annotation audit"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
        )


def test_software_cue_crop_is_explicitly_exploratory_and_never_falls_back(tmp_path):
    _write_processed_trial(
        tmp_path,
        "P1",
        "S1",
        BADMINTON_SYNERGY_16_V2,
        1,
        manual_movement_event=False,
    )
    output = build_synergy_dataset(
        tmp_path,
        tmp_path / "out.npz",
        BADMINTON_SYNERGY_16_V2.profile_id,
        "badminton_primitive_protocol_v1",
        crop_mode="software_cue_exploratory",
    )
    metadata = read_json(output.with_suffix(".json"))
    assert metadata["exploratory"] is True
    assert metadata["trials"][0]["crop_is_exploratory"] is True
    assert metadata["trials"][0]["source_start_sample"] == 10

    events_path = (
        tmp_path / "P1" / "S1" / "trials" / "forehand_high_clear" / "trial_001" / "events.csv"
    )
    events_path.write_text(
        "event_name,sample_index,emg_time_s,monotonic_time_ns,wall_clock_iso,source,confidence,notes\n"
        "recording_stop,90,0.045,2,x,software,0.5,x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="movement_cue"):
        build_synergy_dataset(
            tmp_path,
            tmp_path / "out2.npz",
            BADMINTON_SYNERGY_16_V2.profile_id,
            "badminton_primitive_protocol_v1",
            crop_mode="software_cue_exploratory",
        )


def test_nmf_recovers_low_rank_synthetic_data():
    rng = np.random.default_rng(42)
    true_W = rng.uniform(0.1, 1.0, size=(16, 3))
    true_H = rng.uniform(0.0, 1.0, size=(3, 500))
    V = true_W @ true_H
    W, H, metrics = fit_nmf_best(V, 3, n_init=8, seed=12)
    assert metrics["global_vaf"] > 0.995
    assert (W >= 0).all() and (H >= 0).all()
    assert match_synergies(W, W)["mean_cosine_similarity"] == pytest.approx(1.0)


def test_channel_scaling_keeps_quiet_muscles_from_being_abandoned():
    """A few loud electrodes must not decide the whole factorisation.

    Surface-EMG amplitude spans more than an order of magnitude across sites, and
    NMF minimises an unweighted sum of squares, so an unbalanced fit buys global
    VAF by reproducing the loudest channels and leaves the quiet ones
    unexplained.  Equalising the channels first is what makes the recovered
    components describe co-activation across the whole set.
    """
    rng = np.random.default_rng(7)
    samples = np.arange(900)
    # Temporally separated bursts, the shape real synergy activations take.  Flat
    # random coefficients would be near-collinear and any K would fit them.
    true_H = np.stack([np.exp(-0.5 * ((samples - centre) / 38.0) ** 2) for centre in np.linspace(70, 830, 6)])
    true_W = np.zeros((16, 6))
    for index in range(3):
        true_W[index, index] = 1.0  # three loud electrodes, each on its own component
    true_W[3:, 3:] = rng.uniform(0.3, 1.0, size=(13, 3))  # thirteen quiet muscles share three
    gains = np.ones(16)
    gains[:3] = 80.0
    V = np.maximum(gains[:, None] * (true_W @ true_H) + rng.normal(0.0, 0.004, size=(16, 900)), 0.0)

    unbalanced_W, unbalanced_H, _ = fit_nmf_best(V, 3, n_init=10, seed=3)
    unbalanced_global, unbalanced_local, _ = vaf_metrics(V, unbalanced_W @ unbalanced_H)

    scale = channel_scale(V, "unit_max")
    balanced_W, balanced_H, _ = fit_nmf_best(V / scale[:, None], 3, n_init=10, seed=3)
    balanced_global, balanced_local, _ = vaf_metrics(V / scale[:, None], balanced_W @ balanced_H)

    quiet = np.arange(3, 16)
    # The headline failure: a near-perfect global VAF while most muscles are not
    # described at all.  Global VAF alone cannot detect this.
    assert unbalanced_global > 0.99
    assert unbalanced_local[quiet].mean() < 0.05
    # Equalising the channels first spends the same three components on the
    # structure shared by the thirteen quiet muscles.
    assert balanced_global > 0.85
    assert balanced_local[quiet].min() > 0.85


def test_channel_scaling_rejects_a_silent_channel_instead_of_amplifying_noise():
    V = np.abs(np.random.default_rng(1).normal(size=(4, 50)))
    V[2] = 0.0
    assert channel_scale(V, "none").tolist() == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="numerically silent"):
        channel_scale(V, "unit_max")


def test_dynamic_p95_uses_one_session_reference_not_per_trial(tmp_path):
    session = tmp_path / "P1" / "S1"
    atomic_write_json(session / "channel_profile.json", BADMINTON_SYNERGY_16_V2.to_dict())
    fs = 2000.0
    t = np.arange(4000) / fs
    base = np.sin(2 * np.pi * 100 * t)[:, None] * np.ones((1, 16))
    for index, gain in enumerate((1.0, 5.0), start=1):
        trial = session / "trials" / "forehand_high_clear" / f"trial_{index:03d}"
        trial.mkdir(parents=True)
        atomic_save_npz(
            trial / "raw_emg.npz",
            emg_mV=(gain * base).astype(np.float32),
            fs_hz=np.asarray(fs),
            sample_index=np.arange(len(t)),
            time_s=t,
        )
    outputs = preprocess_session(session, BADMINTON_SYNERGY_16_V2.profile_id, normalization="dynamic_p95")
    configs = [read_json(path.parent / "processing.json") for path in outputs]
    assert configs[0]["normalization_reference_scope"] == "session_all_trials"
    assert configs[0]["normalization_reference_values_mV"] == configs[1]["normalization_reference_values_mV"]
    with np.load(outputs[0]) as low, np.load(outputs[1]) as high:
        low_level = np.median(low["normalized_envelope"])
        high_level = np.median(high["normalized_envelope"])
    assert high_level / low_level == pytest.approx(5.0, rel=0.05)
