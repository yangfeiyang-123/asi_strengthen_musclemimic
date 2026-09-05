"""Contracts for the guided human EMG review workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.review_emg_for_training import (
    ATTESTATION,
    finalize_reviews,
    prepare_review_packet,
    run_wizard,
    validate_review_outputs,
)

ACTION = "forehand_high_clear"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, flagged_trial: bool = False) -> tuple[Path, Path]:
    session = tmp_path / "session"
    channels = [
        {
            "sensor_id": index,
            "side": "right" if index != 10 else "left",
            "muscle_slug": f"muscle_{index}",
        }
        for index in range(1, 17)
    ]
    _write_json(
        session / "channel_profile.json",
        {"profile_id": "test_profile", "channels": channels},
    )
    _write_json(
        session / "preprocessing_mvc_reference.json",
        {
            "scope": "participant",
            "algorithm": "test MVC",
            "channels": [
                {
                    "sensor_id": index,
                    "selected_peak_mV": 0.1,
                    "valid_repetitions": [
                        {"path": f"mvc_s{index}.npz", "envelope_peak_mV": 0.1}
                    ],
                }
                for index in range(1, 17)
            ],
        },
    )
    mapping_channels = []
    for channel in channels:
        name = f"S{channel['sensor_id']} {channel['side']}:{channel['muscle_slug']}"
        if channel["sensor_id"] == 1:
            mapping_channels.append(
                {
                    **channel,
                    "emg_channel": name,
                    "mapping_status": "excluded_no_verified_model_homolog",
                    "exclusion_reason": "test model has no verified homolog",
                }
            )
        else:
            mapping_channels.append(
                {
                    **channel,
                    "emg_channel": name,
                    "mapping_status": "mapped",
                    "simulation_actuators": [f"actuator_{channel['sensor_id']}"],
                    "weights": [1.0],
                    "mapping_confidence": "provisional",
                    "mapping_uncertainty": "synthetic review fixture",
                }
            )
    mapping_path = tmp_path / "mapping.json"
    _write_json(
        mapping_path,
        {
            "schema_version": "emg_observation_mapping_v2",
            "mapping_id": "test_mapping",
            "review_status": "provisional",
            "review_evidence": [],
            "profile_binding": {
                "acquired_channel_count": 16,
                "comparable_channel_count": 15,
            },
            "model_binding": {"actuator_schema_hash": "a" * 64},
            "channels": mapping_channels,
            "training_enabled": False,
        },
    )

    channel_names = np.asarray(
        [f"S{item['sensor_id']} {item['side']}:{item['muscle_slug']}" for item in channels]
    )
    for index in range(1, 4):
        trial = session / "trials" / ACTION / f"trial_{index:03d}"
        trial.mkdir(parents=True)
        values = np.full((200, 16), 0.2, dtype=np.float32)
        values[:, 1] = 2.5  # Above MVC, but deliberately not a signal-QC failure.
        values[:, 8] = 0.01 * index
        np.savez_compressed(
            trial / "mvc_normalized_emg.npz",
            normalized_envelope=values,
            channel_names=channel_names,
            fs_hz=np.asarray(2000.0),
        )
        s9_critical = flagged_trial and index == 1
        _write_json(
            trial / "preprocessing_qc.json",
            {
                "qc_pass": not s9_critical,
                "analysis_ready": not s9_critical,
                "channels": [
                    {
                        "sensor_id": sensor_id,
                        "filtered_rms_mV": 0.001 * sensor_id,
                        "warnings": [],
                        "critical_failures": (
                            ["filtered_signal_near_flatline"]
                            if s9_critical and sensor_id == 9
                            else []
                        ),
                    }
                    for sensor_id in range(1, 17)
                ],
            },
        )
        _write_json(
            trial / "metadata.json",
            {
                "trial_id": f"{ACTION}_trial_{index:03d}",
                "start_time": f"2026-01-01T00:00:0{index}+00:00",
                "valid_for_analysis": True,
            },
        )
    return session, mapping_path


def _complete_answers(path: Path, *, add_flag_resolution: bool = True) -> dict:
    answers = json.loads(path.read_text(encoding="utf-8"))
    answers["reviewer"] = {
        "reviewer_id": "reviewer-1",
        "role": "PI",
        "evidence_statement": "I inspected the acquisition record and generated plots.",
    }
    for entry in answers["mapping_decisions"]:
        entry["decision"] = "accept_exclusion" if entry["sensor_id"] == 1 else "medium"
        entry["reason"] = "Compared the recorded placement with the model inventory."
    for action in answers["actions"]:
        for entry in action["trial_decisions"]:
            entry["decision"] = "include"
            entry["reason"] = "Waveform and acquisition record were manually inspected."
            if add_flag_resolution and entry["trial_id"] == "trial_001":
                entry["qc_resolution_evidence"] = [
                    "S9 raw/QC review concluded task-specific low activation"
                ]
        for entry in action["channel_decisions"]:
            entry["decision"] = "include_after_review"
            entry["reason"] = "Retained after reviewing all action trials."
        action["risk_decisions"][0] = {
            "risk_id": "s9_progressive_near_flatline",
            "decision": "accepted_after_review",
            "reason": "Reviewed session chronology and trial plots.",
            "evidence": ["session_s9_chronology.png", "acquisition notebook entry"],
        }
    answers["attestation"] = ATTESTATION
    path.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    return answers


def test_prepare_reports_super_mvc_without_automatic_exclusion(tmp_path):
    session, mapping = _fixture(tmp_path)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    trial = packet["actions"][0]["trials"][0]
    s2 = trial["channels"][1]
    assert s2["p99_over_mvc"] == pytest.approx(2.5)
    assert s2["mvc_quality"] == "invalid_for_absolute_amplitude"
    assert s2["mvc_exceedance_is_exclusion_reason"] is False
    assert trial["machine_recommendation"] == "include_after_visual_confirmation"
    assert packet["session_s9_context"]["record_count"] == 3
    assert answers_path.is_file()


def test_complete_review_emits_builder_compatible_mapping_and_action_review(tmp_path):
    session, mapping = _fixture(tmp_path, flagged_trial=True)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )
    _complete_answers(answers_path)

    outputs = finalize_reviews(
        packet_path=packet_path,
        answers_path=answers_path,
        output_dir=tmp_path / "final",
    )

    reviewed_mapping = json.loads(outputs["mapping"].read_text(encoding="utf-8"))
    action_review = json.loads(outputs[ACTION].read_text(encoding="utf-8"))
    validation = json.loads(outputs["validation"].read_text(encoding="utf-8"))
    assert reviewed_mapping["review_status"] == "verified"
    assert reviewed_mapping["training_enabled"] is True
    assert reviewed_mapping["channels"][1]["mapping_confidence"] == "medium"
    assert action_review["review_status"] == "verified"
    assert len(action_review["trial_decisions"]) == 3
    assert validation["passed"] is True
    assert validate_review_outputs(
        review_dir=tmp_path / "final",
        packet_path=packet_path,
    ).is_file()


def test_finalize_refuses_pending_answers(tmp_path):
    session, mapping = _fixture(tmp_path)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )

    with pytest.raises(ValueError, match=r"reviewer\.reviewer_id"):
        finalize_reviews(
            packet_path=packet_path,
            answers_path=answers_path,
            output_dir=tmp_path / "final",
        )


def test_flagged_included_trial_requires_resolution_evidence(tmp_path):
    session, mapping = _fixture(tmp_path, flagged_trial=True)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )
    _complete_answers(answers_path, add_flag_resolution=False)

    with pytest.raises(ValueError, match="QC-resolution evidence"):
        finalize_reviews(
            packet_path=packet_path,
            answers_path=answers_path,
            output_dir=tmp_path / "final",
        )


def test_packet_detects_source_tampering_and_prepare_is_resumable(tmp_path):
    session, mapping = _fixture(tmp_path)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    answers["reviewer"]["reviewer_id"] = "saved-progress"
    answers_path.write_text(json.dumps(answers) + "\n", encoding="utf-8")

    repeated_packet, repeated_answers = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )
    assert repeated_packet == packet_path
    assert repeated_answers == answers_path
    assert json.loads(answers_path.read_text())["reviewer"]["reviewer_id"] == "saved-progress"

    trial = session / "trials" / ACTION / "trial_001" / "preprocessing_qc.json"
    trial.write_text(trial.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        prepare_review_packet(
            session_root=session,
            mapping_path=mapping,
            output_dir=tmp_path / "packet",
            actions=(ACTION,),
            make_plots=False,
        )


def test_wizard_walks_every_question_and_finalizes(tmp_path):
    session, mapping = _fixture(tmp_path)
    packet_path, answers_path = prepare_review_packet(
        session_root=session,
        mapping_path=mapping,
        output_dir=tmp_path / "packet",
        actions=(ACTION,),
        make_plots=False,
    )
    responses = ["reviewer-1", "PI", "placement record and plots"]
    for sensor_id in range(1, 17):
        responses.extend(
            [
                "a" if sensor_id == 1 else "m",
                "reviewed mapping evidence",
            ]
        )
    for _trial in range(3):
        responses.extend(["i", "reviewed trial waveform"])
    for _channel in range(15):
        responses.extend(["i", "reviewed channel across trials"])
    responses.extend(
        [
            "a",
            "reviewed S9 session chronology",
            "session_s9_chronology.png;acquisition notes",
            ATTESTATION,
        ]
    )
    iterator = iter(responses)

    outputs = run_wizard(
        packet_path=packet_path,
        answers_path=answers_path,
        final_output_dir=tmp_path / "final",
        input_fn=lambda _prompt: next(iterator),
    )

    assert outputs is not None
    assert outputs["mapping"].is_file()
    assert outputs[ACTION].is_file()
    assert outputs["validation"].is_file()
