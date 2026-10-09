"""Action-specific phase identities shared by latent training and analysis."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASE_CONTRACT_SCHEMA_VERSION = "latent_phase_contract_v1"
FOREHAND_PHASE_NAMES = (
    "ready",
    "backswing",
    "acceleration",
    "impact",
    "followthrough",
    "recovery",
)
_PHASE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def legacy_forehand_phase_contract(
    *,
    phase_field: str = "phase_id",
    require_all_phases: bool = True,
) -> dict[str, Any]:
    """Return the historical six-phase contract without changing old callers."""

    return {
        "schema_version": PHASE_CONTRACT_SCHEMA_VERSION,
        "phase_field": str(phase_field),
        "phases": [
            {"id": phase_id, "name": name}
            for phase_id, name in enumerate(FOREHAND_PHASE_NAMES)
        ],
        "require_all_phases": bool(require_all_phases),
    }


def normalize_phase_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize an ordered, action-specific phase contract."""

    if not isinstance(value, Mapping):
        raise ValueError("phase_contract must be an object")
    schema = value.get("schema_version", PHASE_CONTRACT_SCHEMA_VERSION)
    if schema != PHASE_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported phase_contract schema: {schema!r}")
    raw_phase_field = value.get("phase_field")
    phase_field = None if raw_phase_field is None else str(raw_phase_field).strip()
    require_all = value.get("require_all_phases")
    if not isinstance(require_all, bool):
        raise ValueError("phase_contract.require_all_phases must be boolean")
    raw_phases = value.get("phases")
    if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
        raise ValueError("phase_contract.phases must be an ordered list")
    if not raw_phases:
        if phase_field is not None or require_all:
            raise ValueError(
                "a disabled phase_contract requires phase_field=null, phases=[], "
                "and require_all_phases=false"
            )
        return {
            "schema_version": PHASE_CONTRACT_SCHEMA_VERSION,
            "phase_field": None,
            "phases": [],
            "require_all_phases": False,
        }
    if not phase_field:
        raise ValueError("phase_contract.phase_field must be non-empty when phases are declared")
    phases: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_phases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"phase_contract.phases[{index}] must be an object")
        raw_id = raw.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise ValueError(f"phase_contract.phases[{index}].id must be a non-negative integer")
        name = str(raw.get("name", "")).strip().lower()
        if not _PHASE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"phase_contract.phases[{index}].name must be a lowercase snake_case identifier"
            )
        phases.append({"id": int(raw_id), "name": name})
    ids = [item["id"] for item in phases]
    names = [item["name"] for item in phases]
    if len(set(ids)) != len(ids):
        raise ValueError("phase_contract phase IDs must be unique")
    if len(set(names)) != len(names):
        raise ValueError("phase_contract phase names must be unique")
    return {
        "schema_version": PHASE_CONTRACT_SCHEMA_VERSION,
        "phase_field": phase_field,
        "phases": phases,
        "require_all_phases": require_all,
    }


def load_phase_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return normalize_phase_contract(payload)


def phase_items(value: Mapping[str, Any] | None) -> tuple[tuple[int, str], ...]:
    contract = (
        legacy_forehand_phase_contract()
        if value is None
        else normalize_phase_contract(value)
    )
    return tuple((int(item["id"]), str(item["name"])) for item in contract["phases"])


def canonical_phase_contract_sha256(value: Mapping[str, Any]) -> str:
    contract = normalize_phase_contract(value)
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
