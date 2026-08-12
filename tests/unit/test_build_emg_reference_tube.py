"""Focused contracts for the PEASD EMG reference-tube builder."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_emg_reference_tube import (
    DEFAULT_FOREHAND_OUTPUT,
    EMG_OBSERVATION_MAPPING_FILENAME,
    EMG_TRIAL_QC_REVIEW_FILENAME,
    EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION,
    LEGACY_FOREHAND_OUTPUT,
    _bundle_mapping,
    _bundle_trial_qc_review,
    _default_output_root,
    _load_verified_trial_qc_review,
    _mapping_binding,
    _phase_input_trials,
    _require_verified_mapping,
    _software_cue_time_normalized_trial,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = (
    REPO_ROOT
    / "configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)


def _reviewed_mapping() -> dict:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping["review_status"] = "verified"
    mapping["training_enabled"] = True
    mapping["review_evidence"] = ["docs/jidian_emg_integration.md#mapping-review"]
    for channel in mapping["channels"]:
        if channel["mapping_status"] == "mapped":
            channel["mapping_confidence"] = "medium"
    return mapping


def _write_trial_qc_review(
    path: Path,
    *,
    action: str,
    mapping_sha256: str,
    trials: list[tuple[Path, np.ndarray]],
    channel_names: list[str],
) -> dict:
    payload = {
        "schema_version": EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION,
        "action": action,
        "review_status": "verified",
        "training_enabled": True,
        "mapping_sha256": mapping_sha256,
        "reviewer_id": "domain-expert-fixture",
        "reviewed_at": "2026-08-10T00:00:00Z",
        "review_evidence": ["controlled-review-record"],
        "trial_decisions": [
            {
                "trial_id": trial_dir.name,
                "decision": "include",
                "reason": "reviewed synthetic fixture",
                "mvc_normalized_emg_sha256": hashlib.sha256(
                    (trial_dir / "mvc_normalized_emg.npz").read_bytes()
                ).hexdigest(),
                "preprocessing_qc_sha256": hashlib.sha256(
                    (trial_dir / "preprocessing_qc.json").read_bytes()
                ).hexdigest(),
            }
            for trial_dir, _values in trials
        ],
        "channel_decisions": [
            {
                "emg_channel": name,
                "decision": "include_after_review",
                "reason": "reviewed synthetic fixture",
            }
            for name in channel_names
        ],
        "risk_decisions": [
            {
                "risk_id": risk_id,
                "decision": "mitigated",
                "reason": "reviewed synthetic fixture",
                "evidence": ["controlled-review-record"],
            }
            for risk_id in ("s9_progressive_near_flatline", "super_mvc")
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _synthetic_trials(root: Path, *, count: int = 3) -> list[tuple[Path, np.ndarray]]:
    trials = []
    for index in range(1, count + 1):
        trial_dir = root / f"trial_{index:03d}"
        trial_dir.mkdir(parents=True)
        np.savez_compressed(
            trial_dir / "mvc_normalized_emg.npz",
            normalized_envelope=np.full((40, 16), 0.2 + index * 0.01),
        )
        (trial_dir / "preprocessing_qc.json").write_text(
            json.dumps({"trial": trial_dir.name}), encoding="utf-8"
        )
        trials.append((trial_dir, np.full((40, 16), 0.2 + index * 0.01)))
    return trials


def test_default_output_root_is_action_specific_and_preserves_v1_as_read_only():
    assert Path(LEGACY_FOREHAND_OUTPUT) != Path(DEFAULT_FOREHAND_OUTPUT)
    assert _default_output_root("forehand_high_clear") == Path(DEFAULT_FOREHAND_OUTPUT)
    assert _default_output_root("forehand_lift_footwork") == Path(
        "artifacts/forehand_lift_peasd_v1/data/emg_reference_v2"
    )
    assert _default_output_root("shadow_forehand_lift") == Path(
        "artifacts/forehand_lift_peasd_v1/data/emg_reference_v2"
    )
    assert _default_output_root("china_jump_high_clear") == Path(
        "artifacts/chinajump_peasd_v1/data/emg_reference_v2"
    )


def test_mapping_binding_uses_exact_mapping_file_hash_not_profile_hash():
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping_sha = hashlib.sha256(MAPPING_PATH.read_bytes()).hexdigest()

    binding = _mapping_binding(mapping, mapping_sha256=mapping_sha)

    assert binding["mapping_sha256"] == mapping_sha
    assert binding["mapping_sha256"] != mapping["profile_binding"]["profile_sha256"]


def test_mapping_bundle_preserves_exact_bytes_and_hash(tmp_path):
    mapping_sha = hashlib.sha256(MAPPING_PATH.read_bytes()).hexdigest()

    destination = _bundle_mapping(MAPPING_PATH, tmp_path, expected_sha256=mapping_sha)

    assert destination.name == EMG_OBSERVATION_MAPPING_FILENAME
    assert destination.read_bytes() == MAPPING_PATH.read_bytes()


def test_verified_trial_qc_review_binds_exact_trials_channels_and_known_risks(
    tmp_path,
):
    trials = _synthetic_trials(tmp_path / "trials")
    mapping_sha256 = "a" * 64
    channel_names = [f"S{index}" for index in range(2, 17)]
    review_path = tmp_path / "review.json"
    _write_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )

    included, binding = _load_verified_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )

    assert [path.name for path, _values in included] == [
        "trial_001",
        "trial_002",
        "trial_003",
    ]
    assert binding["review_sha256"] == hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    assert {entry["risk_id"] for entry in binding["risk_decisions"]} >= {
        "s9_progressive_near_flatline",
    }


def test_verified_trial_qc_review_filters_excluded_trials_before_fit(tmp_path):
    trials = _synthetic_trials(tmp_path / "trials")
    mapping_sha256 = "a" * 64
    channel_names = [f"S{index}" for index in range(2, 17)]
    review_path = tmp_path / "review.json"
    payload = _write_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )
    payload["trial_decisions"][1].update(
        decision="exclude", reason="human review rejected this trial"
    )
    review_path.write_text(json.dumps(payload), encoding="utf-8")

    included, binding = _load_verified_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )

    assert [path.name for path, _values in included] == ["trial_001", "trial_003"]
    assert [
        entry["trial_id"]
        for entry in binding["trial_decisions"]
        if entry["decision"] == "exclude"
    ] == ["trial_002"]


def test_verified_trial_qc_review_rejects_unresolved_known_risk(tmp_path):
    trials = _synthetic_trials(tmp_path / "trials")
    mapping_sha256 = "a" * 64
    channel_names = [f"S{index}" for index in range(2, 17)]
    review_path = tmp_path / "review.json"
    payload = _write_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )
    payload["risk_decisions"] = [
        entry
        for entry in payload["risk_decisions"]
        if entry["risk_id"] != "s9_progressive_near_flatline"
    ]
    review_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"does not resolve known risks"):
        _load_verified_trial_qc_review(
            review_path,
            action="forehand_lift_footwork",
            mapping_sha256=mapping_sha256,
            trials=trials,
            channel_names=channel_names,
        )


def test_verified_trial_qc_review_does_not_require_super_mvc_waiver(tmp_path):
    trials = _synthetic_trials(tmp_path / "trials")
    mapping_sha256 = "a" * 64
    channel_names = [f"S{index}" for index in range(2, 17)]
    review_path = tmp_path / "review.json"
    payload = _write_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )
    payload["risk_decisions"] = [
        entry
        for entry in payload["risk_decisions"]
        if entry["risk_id"] != "super_mvc"
    ]
    review_path.write_text(json.dumps(payload), encoding="utf-8")

    included, binding = _load_verified_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )

    assert len(included) == len(trials)
    assert {item["risk_id"] for item in binding["risk_decisions"]} == {
        "s9_progressive_near_flatline"
    }


def test_verified_trial_qc_review_rejects_silent_channel_drop(tmp_path):
    trials = _synthetic_trials(tmp_path / "trials")
    mapping_sha256 = "a" * 64
    channel_names = [f"S{index}" for index in range(2, 17)]
    review_path = tmp_path / "review.json"
    payload = _write_trial_qc_review(
        review_path,
        action="forehand_lift_footwork",
        mapping_sha256=mapping_sha256,
        trials=trials,
        channel_names=channel_names,
    )
    payload["channel_decisions"][7].update(
        decision="exclude", reason="known S9 failure"
    )
    review_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"changes the EMG observation ABI"):
        _load_verified_trial_qc_review(
            review_path,
            action="forehand_lift_footwork",
            mapping_sha256=mapping_sha256,
            trials=trials,
            channel_names=channel_names,
        )


def test_trial_qc_review_bundle_preserves_exact_bytes_and_hash(tmp_path):
    review_path = tmp_path / "source-review.json"
    review_path.write_bytes(b'{"review":"fixture"}\n')
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()

    destination = _bundle_trial_qc_review(
        review_path,
        tmp_path / "bundle",
        expected_sha256=review_sha256,
    )

    assert destination.name == EMG_TRIAL_QC_REVIEW_FILENAME
    assert destination.read_bytes() == review_path.read_bytes()


def test_verified_builder_gate_accepts_only_complete_review_contract():
    mapping = _reviewed_mapping()

    assert _require_verified_mapping(mapping) == mapping["review_evidence"]

    cases: list[tuple[str, object]] = [
        ("review_status", "provisional"),
        ("training_enabled", False),
        ("review_evidence", []),
    ]
    for field, value in cases:
        invalid = copy.deepcopy(mapping)
        invalid[field] = value
        with pytest.raises(ValueError, match=r"--verified requires"):
            _require_verified_mapping(invalid)

    provisional_channel = copy.deepcopy(mapping)
    provisional_channel["channels"][1]["mapping_confidence"] = "provisional"
    with pytest.raises(ValueError, match=r"every comparable channel"):
        _require_verified_mapping(provisional_channel)


def test_repository_mapping_cannot_be_self_promoted_with_verified_flag():
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match=r"review_status=verified"):
        _require_verified_mapping(mapping)


def test_clear_uses_explicit_software_cue_crop_and_101_sample_axis(tmp_path):
    signal = np.arange(120 * 16, dtype=np.float64).reshape(120, 16)
    (tmp_path / "events.csv").write_text(
        "event_name,sample_index,source\n"
        "movement_cue,20,software_audio_visual\n"
        "recording_stop,100,software\n",
        encoding="utf-8",
    )

    normalized = _software_cue_time_normalized_trial(tmp_path, signal)
    selected = _phase_input_trials(
        "forehand_high_clear",
        [(tmp_path, signal)],
    )

    assert normalized.shape == (101, 16)
    assert np.all(normalized >= 0.0)
    np.testing.assert_allclose(selected[0], normalized)


def test_nonclear_phase_input_remains_full_trial_duration_normalized(tmp_path):
    signal = np.ones((77, 16), dtype=np.float64)

    selected = _phase_input_trials(
        "china_jump_high_clear",
        [(tmp_path, signal)],
    )

    assert selected[0] is signal
