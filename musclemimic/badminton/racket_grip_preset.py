"""Versioned global hand-racket grip presets.

A preset binds one exact-child racket attachment contract to the twenty
right-hand finger joint targets used at reset and by the racket imitation
reward.  It is deliberately trajectory-independent: one preset applies to all
training, validation, and viewer trajectories loaded by an environment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environment.overall_environment.src.racket_attachment import (
    REPO_ROOT,
    RacketAttachmentContract,
    canonical_json_fingerprint,
    load_racket_attachment_contract,
)

RACKET_GRIP_PRESET_SCHEMA = "musclemimic.racket_grip_preset.v1"
RACKET_GRIP_PRESET_SCOPE = "all_trajectories"
DEFAULT_RACKET_GRIP_PRESET_PATH = (
    REPO_ROOT / "configs" / "racket_grip" / "forehand_clear_grip_v2_custom.json"
)

RIGHT_HAND_GRIP_JOINT_NAMES = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field} keys mismatch: missing={missing}, extra={extra}")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def grip_preset_fingerprint(document: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 fingerprint excluding the declaration."""

    payload = dict(document)
    payload.pop("fingerprint", None)
    return canonical_json_fingerprint(payload)


def _display_contract_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_contract_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class RacketGripPreset:
    """Strictly validated all-trajectory grip preset."""

    schema: str
    preset_id: str
    scope: str
    attachment_contract_path: Path
    attachment_contract_fingerprint: str
    finger_joint_angles_rad: tuple[tuple[str, float], ...]
    fingerprint: str
    source_path: Path

    @property
    def finger_angles_by_name(self) -> dict[str, float]:
        return dict(self.finger_joint_angles_rad)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "preset_id": self.preset_id,
            "scope": self.scope,
            "attachment_contract": {
                "path": _display_contract_path(self.attachment_contract_path),
                "fingerprint": self.attachment_contract_fingerprint,
            },
            "finger_joint_angles_rad": dict(self.finger_joint_angles_rad),
            "fingerprint": self.fingerprint,
        }


def build_racket_grip_preset_document(
    *,
    preset_id: str,
    attachment_contract: RacketAttachmentContract,
    finger_joint_angles_rad: Mapping[str, float],
) -> dict[str, Any]:
    """Build a canonical global preset document."""

    identifier = _require_string(preset_id, field="preset_id")
    actual_names = set(finger_joint_angles_rad)
    expected_names = set(RIGHT_HAND_GRIP_JOINT_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "finger_joint_angles_rad keys mismatch: "
            f"missing={missing}, extra={extra}"
        )
    angles: dict[str, float] = {}
    for name in RIGHT_HAND_GRIP_JOINT_NAMES:
        value = float(finger_joint_angles_rad[name])
        if not math.isfinite(value):
            raise ValueError(f"finger_joint_angles_rad.{name} must be finite")
        clean = round(value, 9)
        angles[name] = 0.0 if clean == -0.0 else clean
    document = {
        "schema": RACKET_GRIP_PRESET_SCHEMA,
        "preset_id": identifier,
        "scope": RACKET_GRIP_PRESET_SCOPE,
        "attachment_contract": {
            "path": _display_contract_path(attachment_contract.source_path),
            "fingerprint": attachment_contract.fingerprint,
        },
        "finger_joint_angles_rad": angles,
    }
    document["fingerprint"] = grip_preset_fingerprint(document)
    return document


def load_racket_grip_preset(path: str | Path) -> RacketGripPreset:
    """Load a preset and verify its fingerprint and attachment binding."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"racket grip preset not found: {source_path}")
    document = json.loads(
        source_path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    root = _require_exact_keys(
        document,
        {
            "schema",
            "preset_id",
            "scope",
            "attachment_contract",
            "finger_joint_angles_rad",
            "fingerprint",
        },
        field="racket grip preset",
    )
    schema = _require_string(root["schema"], field="schema")
    if schema != RACKET_GRIP_PRESET_SCHEMA:
        raise ValueError(f"unsupported racket grip preset schema: {schema!r}")
    scope = _require_string(root["scope"], field="scope")
    if scope != RACKET_GRIP_PRESET_SCOPE:
        raise ValueError(f"unsupported racket grip preset scope: {scope!r}")
    attachment = _require_exact_keys(
        root["attachment_contract"],
        {"path", "fingerprint"},
        field="attachment_contract",
    )
    attachment_path = _resolve_contract_path(
        _require_string(attachment["path"], field="attachment_contract.path")
    )
    attachment_fingerprint = _require_string(
        attachment["fingerprint"],
        field="attachment_contract.fingerprint",
    )
    angles_raw = _require_exact_keys(
        root["finger_joint_angles_rad"],
        set(RIGHT_HAND_GRIP_JOINT_NAMES),
        field="finger_joint_angles_rad",
    )
    angles = []
    for name in RIGHT_HAND_GRIP_JOINT_NAMES:
        value = angles_raw[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"finger_joint_angles_rad.{name} must be a finite number")
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError(f"finger_joint_angles_rad.{name} must be finite")
        angles.append((name, angle))
    fingerprint = _require_string(root["fingerprint"], field="fingerprint")
    expected_fingerprint = grip_preset_fingerprint(root)
    if fingerprint != expected_fingerprint:
        raise ValueError(
            "racket grip preset fingerprint mismatch: "
            f"declared={fingerprint}, computed={expected_fingerprint}"
        )
    contract = load_racket_attachment_contract(attachment_path)
    if contract.fingerprint != attachment_fingerprint:
        raise ValueError(
            "racket grip preset attachment fingerprint mismatch: "
            f"declared={attachment_fingerprint}, actual={contract.fingerprint}"
        )
    return RacketGripPreset(
        schema=schema,
        preset_id=_require_string(root["preset_id"], field="preset_id"),
        scope=scope,
        attachment_contract_path=attachment_path,
        attachment_contract_fingerprint=attachment_fingerprint,
        finger_joint_angles_rad=tuple(angles),
        fingerprint=fingerprint,
        source_path=source_path,
    )


def write_racket_grip_preset(
    output_path: str | Path,
    *,
    preset_id: str,
    attachment_contract: RacketAttachmentContract,
    finger_joint_angles_rad: Mapping[str, float],
) -> RacketGripPreset:
    """Atomically write and strictly reload a global grip preset."""

    output = Path(output_path).expanduser().resolve()
    document = build_racket_grip_preset_document(
        preset_id=preset_id,
        attachment_contract=attachment_contract,
        finger_joint_angles_rad=finger_joint_angles_rad,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return load_racket_grip_preset(output)
