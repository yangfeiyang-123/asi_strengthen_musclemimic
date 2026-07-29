"""Checked-in Jidian 16-channel / MyoFullBody 15-channel binding tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jidian_measurement.emg.profiles import BADMINTON_SYNERGY_16_V2
from musclemimic.evaluation.emg_eval import validate_emg_mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = (
    REPOSITORY_ROOT / "configs" / "physiology" / "emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)
TAXONOMY_PATH = REPOSITORY_ROOT / "configs" / "physiology" / "myofullbody_354_muscle_taxonomy_audit_v1.json"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_checked_in_jidian_mapping_is_exact_16_acquired_15_comparable() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    profile = BADMINTON_SYNERGY_16_V2
    channel_names = [f"S{channel.sensor_id} {channel.side}:{channel.muscle_slug}" for channel in profile.channels]
    contract = validate_emg_mapping(
        mapping,
        emg_channel_names=channel_names,
        actuator_names=[row["name"] for row in taxonomy["ordered_actuators"]],
        allow_provisional_mapping=True,
    )

    assert contract["profile_binding"]["profile_sha256"] == _canonical_sha256(profile.to_dict())
    assert contract["model_binding"]["taxonomy_fingerprint"] == taxonomy["taxonomy_fingerprint"]
    assert contract["model_binding"]["runtime_model_hash"] == taxonomy["model_binding"]["runtime_model_hash"]
    assert contract["model_binding"]["actuator_schema_hash"] == taxonomy["model_binding"]["actuator_schema_hash"]
    assert len(contract["channels"]) == 16
    assert sum(channel["mapping_status"] == "mapped" for channel in contract["channels"]) == 15
    assert contract["channels"][0]["muscle_slug"] == "upper_trapezius"
    assert contract["channels"][0]["mapping_status"] == "excluded_no_verified_model_homolog"
    assert contract["channels"][0]["simulation_actuators"] == []


def test_checked_in_jidian_mapping_remains_exploratory_until_reviewed() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="provisional.*exploratory-only"):
        validate_emg_mapping(mapping)
