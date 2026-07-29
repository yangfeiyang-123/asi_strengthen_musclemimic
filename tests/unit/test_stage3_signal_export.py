"""CPU/source-only tests for final Stage-3 physical-signal export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from musclemimic.distill.physical import MuscleChannelContract
from musclemimic.evaluation.emg_eval import (
    validate_simulation_activation_contract,
    validate_simulation_policy_evidence,
)
from musclemimic.evaluation.physiology import (
    validate_physiology_lineage,
    validate_physiology_signal_contract,
)
from musclemimic.evaluation.stage3_signal_export import (
    Stage3PolicyEvidence,
    Stage3SignalCollector,
    Stage3SignalLayout,
    load_trial_identity_manifest,
    write_stage3_signal_export,
)

POLICY = "1" * 64
PROMOTION = "2" * 64
FORMAL = "3" * 64
EVENT = "4" * 64
PAIRED = "5" * 64
FEED_MANIFEST = "6" * 64
EVALUATION_BINDING = "7" * 64
STAGE3_CHECKPOINT = "8" * 64


class _DirectRuntime:
    checkpoint_fingerprint = POLICY
    decoder_type = "direct"
    synergy_basis = None


class _SynergyBasis:
    fingerprint = "a" * 64

    def __init__(self):
        self.manifest = {"source_fingerprint": FORMAL}


class _SynergyRuntime:
    checkpoint_fingerprint = POLICY
    decoder_type = "synergy_residual"
    synergy_basis = _SynergyBasis()


def _write_identities(tmp_path: Path, *, leak: bool = False) -> tuple[Path, tuple[str, str]]:
    fingerprints = ("a" * 64, "b" * 64)
    heldout_session = "session-heldout"
    payload = {
        "schema_version": "stage3_signal_trial_identity_v1",
        "dataset_split": "heldout",
        "training_session_uids": [heldout_session if leak else "session-training"],
        "trials": [
            {
                "feed_index": index,
                "feed_fingerprint": fingerprint,
                "trial_uid": f"trial-{index}",
                "subject_uid": "subject-01",
                "session_uid": heldout_session,
            }
            for index, fingerprint in enumerate(fingerprints)
        ],
    }
    path = tmp_path / ("identity-leak.json" if leak else "identity.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, fingerprints


def _layout() -> Stage3SignalLayout:
    return Stage3SignalLayout(
        actuator_names=("bic", "tri"),
        actuator_ids=np.asarray([0, 1], dtype=np.int32),
        activation_addresses=np.asarray([0, 1], dtype=np.int32),
        actuator_ctrlrange=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        activation_valid_mask=np.ones((2,), dtype=bool),
        muscle_channel_contract=MuscleChannelContract(
            actuator_names=("bic", "tri"),
            actuator_ids=(0, 1),
            actuator_dyntype=("muscle", "muscle"),
            actuator_actnum=(1, 1),
            actuator_actadr=(0, 1),
            model_na=2,
        ),
        joint_names=("elbow",),
        joint_dof_addresses=np.asarray([0], dtype=np.int32),
        scene_runtime_model_hash="e" * 64,
    )


def _taxonomy_layout() -> Stage3SignalLayout:
    taxonomy_path = (
        Path(__file__).resolve().parents[2] / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v1.json"
    )
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    rows = payload["ordered_actuators"]
    names = tuple(str(row["name"]) for row in rows)
    ids = tuple(int(row["actuator_id"]) for row in rows)
    actadr = tuple(int(row["actadr"]) for row in rows)
    actnum = tuple(int(row["actnum"]) for row in rows)
    return Stage3SignalLayout(
        actuator_names=names,
        actuator_ids=np.asarray(ids, dtype=np.int32),
        activation_addresses=np.asarray(actadr, dtype=np.int32),
        actuator_ctrlrange=np.asarray(
            [row["ctrlrange"] for row in rows],
            dtype=np.float64,
        ),
        activation_valid_mask=np.ones((len(rows),), dtype=bool),
        muscle_channel_contract=MuscleChannelContract(
            actuator_names=names,
            actuator_ids=ids,
            actuator_dyntype=("muscle",) * len(rows),
            actuator_actnum=actnum,
            actuator_actadr=actadr,
            model_na=len(rows),
        ),
        joint_names=("elbow",),
        joint_dof_addresses=np.asarray([0], dtype=np.int32),
        scene_runtime_model_hash="e" * 64,
    )


def _write_v2_identity(tmp_path: Path, *, paired: bool) -> tuple[Path, str]:
    taxonomy_path = (
        Path(__file__).resolve().parents[2] / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v1.json"
    )
    feed_fingerprint = "a" * 64
    row = {
        "feed_index": 0,
        "feed_fingerprint": feed_fingerprint,
        "trial_uid": "trial-v2-0",
        "subject_uid": "subject-simulation-01",
        "session_uid": "session-simulation-heldout",
    }
    if paired:
        row["reference_trial_fingerprint"] = "9" * 64
    payload = {
        "schema_version": "stage3_signal_trial_identity_v2",
        "dataset_split": "heldout",
        "action_id": "forehand_high_clear",
        "handedness": "right",
        "comparison_design": ("paired_same_reference_v1" if paired else "unpaired_action_cohort_v1"),
        "comparison_set_uid": "comparison-set-v2",
        "model_taxonomy_path": str(taxonomy_path),
        "training_session_uids": ["session-training"],
        "trials": [row],
    }
    path = tmp_path / ("identity-v2-paired.json" if paired else "identity-v2.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, feed_fingerprint


def _evidence(tmp_path: Path, *, decoder_type: str = "direct") -> Stage3PolicyEvidence:
    source = tmp_path / "paired.json"
    source.write_text("{}\n", encoding="utf-8")
    return Stage3PolicyEvidence(
        family="best_direct" if decoder_type == "direct" else "best_synergy",
        decoder_type=decoder_type,
        policy_checkpoint_fingerprint=POLICY,
        policy_promotion_fingerprint=PROMOTION,
        formal_synergy_basis_fingerprint=FORMAL,
        event_reference_fingerprint=EVENT,
        stage3_checkpoint_payload_sha256=STAGE3_CHECKPOINT,
        paired_comparison_fingerprint=PAIRED,
        source_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def _transition(step: int, *, hit: bool = False, fall: bool = False) -> dict:
    phase = (0.0, 0.2, 0.55, 0.8, 1.0, 1.0)[step - 1]
    return {
        "teacher_ctrl_physical": np.asarray([0.2 + 0.05 * step, 0.7 - 0.03 * step]),
        "muscle_excitation": np.asarray([0.2 + 0.05 * step, 0.7 - 0.03 * step]),
        "muscle_activation": np.asarray([0.1 + 0.04 * step, 0.5 - 0.02 * step]),
        "joint_torque": np.asarray([0.5 * step]),
        "joint_angular_velocity": np.asarray([0.25 * step]),
        "step_index": step,
        "elapsed_time_s": 0.1 * step,
        "swing_phase": phase,
        "event_state_code": 1 if hit else (3 if step > 3 else 0),
        "hit_this_step": hit,
        "body_fall": fall,
        "recovery_complete": step == 6,
    }


def _collector(tmp_path: Path) -> tuple[Stage3SignalCollector, tuple[str, str]]:
    identity_path, fingerprints = _write_identities(tmp_path)
    collector = Stage3SignalCollector(
        layout=_layout(),
        identities=load_trial_identity_manifest(identity_path),
        policy_evidence=_evidence(tmp_path),
        control_dt_s=0.1,
        pre_impact_s=0.1,
        post_impact_s=0.2,
        expected_episode_count=2,
        runtime=_DirectRuntime(),
        event_reference_fingerprint=EVENT,
        stage3_checkpoint_payload_sha256=STAGE3_CHECKPOINT,
        evaluation_feed_manifest_fingerprint=FEED_MANIFEST,
        evaluation_seed=123,
    )
    return collector, fingerprints


def test_export_is_dense_real_impact_aligned_and_directly_consumable(tmp_path):
    collector, fingerprints = _collector(tmp_path)
    for episode, fingerprint in enumerate(fingerprints):
        collector.begin_episode(
            episode_index=episode,
            feed_index=episode,
            feed_fingerprint=fingerprint,
        )
        for step in range(1, 7):
            collector.record_transition(_transition(step, hit=step == 3))
        collector.end_episode()

    arrays = collector.finalize_arrays(evaluation_binding_sha256=EVALUATION_BINDING)

    assert arrays["muscle_activation"].shape == (2, 4, 2)
    assert arrays["joint_torque"].shape == (2, 4, 1)
    assert arrays["impact_frame"].tolist() == [1, 1]
    assert np.flatnonzero(arrays["impact_flag"][0]).tolist() == [1]
    assert arrays["trial_uid"].tolist() == ["trial-0", "trial-1"]
    assert arrays["session_uid"].tolist() == ["session-heldout", "session-heldout"]
    assert "runtime_synergy_basis_fingerprint" not in arrays

    physiology = validate_physiology_signal_contract(arrays)
    assert physiology["unit_interval_verified"] is True
    mapping = {
        "schema_version": "emg_local_mapping_v1",
        "validation_scope": "right_upper_limb_local",
        "normalization": "per_trial_peak",
        "channels": [
            {
                "emg_channel": "biceps",
                "simulation_actuators": ["bic"],
                "weights": [1.0],
                "mapping_uncertainty": "one model compartment approximation",
            }
        ],
        "cocontraction_pairs": [],
    }
    activation = validate_simulation_activation_contract(
        arrays,
        actuator_names=["bic", "tri"],
        mapping=mapping,
    )
    assert activation["mapped_activation_valid"] == 1.0
    policy = validate_simulation_policy_evidence(
        arrays,
        expected_policy_checkpoint_fingerprint=POLICY,
        expected_policy_promotion_fingerprint=PROMOTION,
        expected_formal_synergy_basis_fingerprint=FORMAL,
    )
    assert policy["formal_basis_role"] == "formal_analysis_basis_only"
    lineage = validate_physiology_lineage(
        arrays,
        expected_policy_checkpoint_fingerprint=POLICY,
        expected_policy_promotion_fingerprint=PROMOTION,
        expected_formal_synergy_basis_fingerprint=FORMAL,
        expected_event_reference_fingerprint=EVENT,
        expected_session_uid="session-heldout",
        expected_policy_decoder_type="direct",
    )
    assert lineage["session_uid"] == "session-heldout"

    output = tmp_path / "simulation_signals.npz"
    manifest = write_stage3_signal_export(output, arrays, collector=collector)
    assert output.is_file()
    assert output.with_suffix(".manifest.json").is_file()
    assert len(manifest["npz_sha256"]) == 64
    assert manifest["npz_shape"] == {
        "trials": 2,
        "time_steps": 4,
        "muscle_channels": 2,
        "joint_channels": 1,
    }


def test_export_fails_on_identity_leak_miss_fall_and_incomplete_window(tmp_path):
    leak_path, _ = _write_identities(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leaks a held-out session"):
        load_trial_identity_manifest(leak_path)

    collector, fingerprints = _collector(tmp_path)
    collector.begin_episode(episode_index=0, feed_index=0, feed_fingerprint=fingerprints[0])
    for step in range(1, 7):
        collector.record_transition(_transition(step))
    with pytest.raises(ValueError, match="no real hit_this_step"):
        collector.end_episode()

    collector, fingerprints = _collector(tmp_path)
    collector.begin_episode(episode_index=0, feed_index=0, feed_fingerprint=fingerprints[0])
    for step in range(1, 7):
        collector.record_transition(_transition(step, hit=step == 3, fall=step == 5))
    with pytest.raises(ValueError, match="body fall"):
        collector.end_episode()

    collector, fingerprints = _collector(tmp_path)
    collector.begin_episode(episode_index=0, feed_index=0, feed_fingerprint=fingerprints[0])
    collector.record_transition(_transition(1, hit=True))
    collector.record_transition(_transition(2))
    with pytest.raises(ValueError, match="lacks the complete requested"):
        collector.end_episode()


def test_synergy_export_binds_runtime_basis_to_formal_source(tmp_path):
    identity_path, fingerprints = _write_identities(tmp_path)
    collector = Stage3SignalCollector(
        layout=_layout(),
        identities=load_trial_identity_manifest(identity_path),
        policy_evidence=_evidence(tmp_path, decoder_type="synergy_residual"),
        control_dt_s=0.1,
        pre_impact_s=0.1,
        post_impact_s=0.2,
        expected_episode_count=1,
        runtime=_SynergyRuntime(),
        event_reference_fingerprint=EVENT,
        stage3_checkpoint_payload_sha256=STAGE3_CHECKPOINT,
        evaluation_feed_manifest_fingerprint=FEED_MANIFEST,
        evaluation_seed=123,
    )
    collector.begin_episode(episode_index=0, feed_index=0, feed_fingerprint=fingerprints[0])
    for step in range(1, 7):
        collector.record_transition(_transition(step, hit=step == 3))
    collector.end_episode()
    arrays = collector.finalize_arrays(evaluation_binding_sha256=EVALUATION_BINDING)

    assert str(arrays["runtime_synergy_basis_fingerprint"]) == "a" * 64
    assert str(arrays["runtime_synergy_basis_source_fingerprint"]) == FORMAL
    policy = validate_simulation_policy_evidence(
        arrays,
        expected_policy_checkpoint_fingerprint=POLICY,
        expected_policy_promotion_fingerprint=PROMOTION,
        expected_formal_synergy_basis_fingerprint=FORMAL,
    )
    assert policy["formal_basis_role"] == "formal_runtime_source_and_analysis_basis"


@pytest.mark.parametrize("paired", [False, True])
def test_v2_identity_binds_comparison_and_taxonomy_into_export(tmp_path, paired):
    identity_path, feed_fingerprint = _write_v2_identity(tmp_path, paired=paired)
    identities = load_trial_identity_manifest(identity_path)
    layout = _taxonomy_layout()
    collector = Stage3SignalCollector(
        layout=layout,
        identities=identities,
        policy_evidence=_evidence(tmp_path),
        control_dt_s=0.1,
        pre_impact_s=0.1,
        post_impact_s=0.2,
        expected_episode_count=1,
        runtime=_DirectRuntime(),
        event_reference_fingerprint=EVENT,
        stage3_checkpoint_payload_sha256=STAGE3_CHECKPOINT,
        evaluation_feed_manifest_fingerprint=FEED_MANIFEST,
        evaluation_seed=123,
    )
    collector.begin_episode(
        episode_index=0,
        feed_index=0,
        feed_fingerprint=feed_fingerprint,
    )
    for step in range(1, 7):
        transition = _transition(step, hit=step == 3)
        width = len(layout.actuator_names)
        for field in (
            "teacher_ctrl_physical",
            "muscle_excitation",
            "muscle_activation",
        ):
            transition[field] = np.full((width,), 0.25, dtype=np.float32)
        collector.record_transition(transition)
    collector.end_episode()
    arrays = collector.finalize_arrays(evaluation_binding_sha256=EVALUATION_BINDING)

    assert str(arrays["action_id"]) == "forehand_high_clear"
    assert str(arrays["comparison_set_uid"]) == "comparison-set-v2"
    assert str(arrays["model_taxonomy_id"]) == "myofullbody_354_muscle_taxonomy_audit_v1"
    assert len(str(arrays["model_taxonomy_fingerprint"])) == 64
    assert str(arrays["scene_runtime_model_hash"]) == "e" * 64
    if paired:
        assert arrays["reference_trial_fingerprint"].tolist() == ["9" * 64]
    else:
        assert "reference_trial_fingerprint" not in arrays
    output = tmp_path / ("signals-paired.npz" if paired else "signals-unpaired.npz")
    sidecar = write_stage3_signal_export(output, arrays, collector=collector)
    assert sidecar["identity"]["comparison_set_uid"] == "comparison-set-v2"
    assert sidecar["identity"]["comparison_design"] == str(arrays["comparison_design"])


def test_v2_identity_rejects_reference_evidence_in_wrong_design(tmp_path):
    path, _ = _write_v2_identity(tmp_path, paired=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparison_design"] = "unpaired_action_cohort_v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not claim reference_trial_fingerprint"):
        load_trial_identity_manifest(path)


def test_v2_identity_file_is_immutable_during_collection(tmp_path):
    path, _ = _write_v2_identity(tmp_path, paired=False)
    identities = load_trial_identity_manifest(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity manifest changed"):
        Stage3SignalCollector(
            layout=_taxonomy_layout(),
            identities=identities,
            policy_evidence=_evidence(tmp_path),
            control_dt_s=0.1,
            pre_impact_s=0.1,
            post_impact_s=0.2,
            expected_episode_count=1,
            runtime=_DirectRuntime(),
            event_reference_fingerprint=EVENT,
            stage3_checkpoint_payload_sha256=STAGE3_CHECKPOINT,
            evaluation_feed_manifest_fingerprint=FEED_MANIFEST,
            evaluation_seed=123,
        )


@pytest.mark.parametrize(
    ("filename", "design"),
    [
        ("stage3_signal_trial_identity_template.json", "unpaired_action_cohort_v1"),
        ("stage3_signal_trial_identity_paired_template.json", "paired_same_reference_v1"),
    ],
)
def test_checked_in_v2_identity_templates_are_loadable(filename, design):
    path = Path(__file__).resolve().parents[2] / "configs/public" / filename
    manifest = load_trial_identity_manifest(path)
    assert manifest.comparison_design == design
    assert manifest.model_taxonomy_id == "myofullbody_354_muscle_taxonomy_audit_v1"
