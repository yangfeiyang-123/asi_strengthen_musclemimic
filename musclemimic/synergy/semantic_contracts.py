"""Versioned semantic attestations required by primitive synergy sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

PRIMITIVE_SEMANTIC_ATTESTATION_SCHEMA_VERSION = "primitive_synergy_semantic_attestation_v1"
P12_RECOVERY_SEMANTIC_CONTRACT_VERSION = "p12_post_landing_recovery_com_v2"
P12_REQUIRED_SEMANTIC_GATES = (
    "bilateral_contact_entire_primitive",
    "post_impact_initial_vertical_speed",
    "posture_restore_com_rise",
    "ready_hold_min_frames",
    "ready_hold_bilateral",
    "ready_hold_vertical_speed",
    "ready_hold_recovered_height",
)
P12_MIN_COM_VERTICAL_EXCURSION = 0.03
P12_MIN_READY_HOLD_FRAMES = 10
P12_MAX_POST_IMPACT_COM_VERTICAL_SPEED = 0.20
P12_MAX_READY_HOLD_COM_VERTICAL_SPEED = 0.15

_P12_CONTRACT_PAYLOAD = {
    "schema_version": P12_RECOVERY_SEMANTIC_CONTRACT_VERSION,
    "phase_order": [0, 1, 2],
    "required_gates": list(P12_REQUIRED_SEMANTIC_GATES),
    "required_actual_evidence_kind": "actual_rollout_exact_contact",
    "reports": ["selected_target", "actual_rollout"],
    "height_baseline": "landing_stabilization_terminal_com_height",
    "ready_height_policy": "minimum_over_complete_ready_hold",
    "recompute_from_hash_bound_qc_arrays": True,
    "threshold_policy": {
        "min_com_vertical_excursion": {"comparison": ">=", "value": P12_MIN_COM_VERTICAL_EXCURSION},
        "min_ready_hold_frames": {"comparison": ">=", "value": P12_MIN_READY_HOLD_FRAMES},
        "max_post_impact_com_vertical_speed": {
            "comparison": "<=",
            "value": P12_MAX_POST_IMPACT_COM_VERTICAL_SPEED,
        },
        "max_ready_hold_com_vertical_speed": {
            "comparison": "<=",
            "value": P12_MAX_READY_HOLD_COM_VERTICAL_SPEED,
        },
    },
}
P12_RECOVERY_SEMANTIC_CONTRACT_FINGERPRINT = hashlib.sha256(
    json.dumps(
        _P12_CONTRACT_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def primitive_semantic_contracts(task_ids: Sequence[str]) -> dict[str, str]:
    """Return current semantic contract versions for task families that require one."""

    return {
        str(task_id): P12_RECOVERY_SEMANTIC_CONTRACT_FINGERPRINT
        for task_id in task_ids
        if str(task_id).split("_", 1)[0] == "P12"
    }


def validate_primitive_semantic_contracts(
    task_ids: Sequence[str],
    payload: Any,
    *,
    label: str,
) -> dict[str, str]:
    """Fail closed when a versioned task is backed by legacy semantic evidence."""

    required = primitive_semantic_contracts(task_ids)
    if not required and payload is None:
        return {}
    if not isinstance(payload, Mapping):
        if required:
            raise ValueError(f"{label} P12 semantic contract is missing or stale")
        raise ValueError(f"{label} primitive semantic contracts must contain an object")
    declared = {str(key): str(value) for key, value in payload.items()}
    for task_id, version in required.items():
        if declared.get(task_id) != version:
            raise ValueError(f"{label} P12 semantic contract is missing or stale for {task_id!r}: expected {version!r}")
    return declared
