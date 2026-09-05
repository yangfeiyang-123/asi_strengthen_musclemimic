from __future__ import annotations

import json

import pytest

from environment.overall_environment.src.racket_attachment import (
    load_racket_attachment_contract,
)
from musclemimic.badminton.racket_grip_preset import (
    DEFAULT_RACKET_GRIP_PRESET_PATH,
    RACKET_GRIP_PRESET_SCHEMA,
    RACKET_GRIP_PRESET_SCOPE,
    RIGHT_HAND_GRIP_JOINT_NAMES,
    build_racket_grip_preset_document,
    grip_preset_fingerprint,
    load_racket_grip_preset,
    write_racket_grip_preset,
)


def _angles() -> dict[str, float]:
    return {
        name: 0.01 * index
        for index, name in enumerate(RIGHT_HAND_GRIP_JOINT_NAMES)
    }


def test_default_preset_is_the_promoted_v2_export() -> None:
    preset = load_racket_grip_preset(DEFAULT_RACKET_GRIP_PRESET_PATH)
    assert preset.preset_id == "forehand_clear_grip_v2_custom"
    assert preset.attachment_contract_path.name == "forehand_clear_rigid_v4_custom.json"
    assert preset.attachment_contract_fingerprint == (
        "sha256:7d1819f8bd04bae2951168c737247bba3e4d1ed02911bd93ec89c55bda271d73"
    )


def test_global_grip_preset_round_trip_binds_attachment(tmp_path) -> None:
    contract = load_racket_attachment_contract()
    output = tmp_path / "global_grip.json"

    preset = write_racket_grip_preset(
        output,
        preset_id="test_global_grip",
        attachment_contract=contract,
        finger_joint_angles_rad=_angles(),
    )

    assert preset.schema == RACKET_GRIP_PRESET_SCHEMA
    assert preset.scope == RACKET_GRIP_PRESET_SCOPE
    assert preset.attachment_contract_fingerprint == contract.fingerprint
    assert preset.finger_angles_by_name == pytest.approx(_angles())
    assert load_racket_grip_preset(output) == preset


def test_global_grip_document_has_canonical_fingerprint() -> None:
    document = build_racket_grip_preset_document(
        preset_id="canonical",
        attachment_contract=load_racket_attachment_contract(),
        finger_joint_angles_rad=_angles(),
    )
    assert document["fingerprint"] == grip_preset_fingerprint(document)


def test_global_grip_preset_rejects_missing_joint() -> None:
    angles = _angles()
    angles.pop(RIGHT_HAND_GRIP_JOINT_NAMES[-1])
    with pytest.raises(ValueError, match="keys mismatch"):
        build_racket_grip_preset_document(
            preset_id="missing_joint",
            attachment_contract=load_racket_attachment_contract(),
            finger_joint_angles_rad=angles,
        )


def test_global_grip_preset_rejects_tampering(tmp_path) -> None:
    output = tmp_path / "tampered.json"
    write_racket_grip_preset(
        output,
        preset_id="tampered",
        attachment_contract=load_racket_attachment_contract(),
        finger_joint_angles_rad=_angles(),
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["finger_joint_angles_rad"][RIGHT_HAND_GRIP_JOINT_NAMES[0]] += 0.5
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_racket_grip_preset(output)
