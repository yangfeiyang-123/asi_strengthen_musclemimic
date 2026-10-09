"""Portable-v2 taxonomy identity and local runtime-audit contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.physiology.anatomical_groups import (
    ANATOMICAL_TAXONOMY_SCHEMA_VERSION,
    ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
    EXACT_RUNTIME_MODEL_COMPATIBILITY,
    PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    load_anatomical_taxonomy,
    taxonomy_fingerprint,
    validate_anatomical_taxonomy,
    validate_taxonomy_against_model,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT_V1 = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v1.json"
AUDIT_V2 = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v2.json"


def _reseal(payload: dict) -> dict:
    payload["taxonomy_fingerprint"] = taxonomy_fingerprint(payload)
    return payload


@pytest.fixture(scope="module")
def runtime_model():
    return MyoFullBody(disable_fingers=True)._model


def test_exported_taxonomy_stable_binding_matches_checked_in_asset():
    from scripts.export_myofullbody_muscle_taxonomy import build_taxonomy_manifest

    exported = validate_anatomical_taxonomy(build_taxonomy_manifest())
    checked_in = load_anatomical_taxonomy(AUDIT_V2)
    assert checked_in.schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION
    assert exported.stable_model_binding == checked_in.stable_model_binding
    assert exported.ordered_actuators == checked_in.ordered_actuators
    assert checked_in.release_eligible is True
    assert load_anatomical_taxonomy(AUDIT_V1).schema_version == ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION
    assert load_anatomical_taxonomy(AUDIT_V1).release_eligible is False


def test_portable_binding_ignores_full_runtime_model_hash_only(runtime_model):
    taxonomy = load_anatomical_taxonomy(AUDIT_V2)
    payload = taxonomy.to_manifest()
    actual = payload["model_binding"]["compiled_runtime_audit"]["runtime_model_hash"]
    payload["model_binding"]["compiled_runtime_audit"]["runtime_model_hash"] = (
        "f" * 64 if actual != "f" * 64 else "e" * 64
    )
    drifted_audit = validate_anatomical_taxonomy(_reseal(payload))
    validate_taxonomy_against_model(
        drifted_audit,
        runtime_model,
        compatibility=PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    )

    core_drift = taxonomy.to_manifest()
    core_drift["model_binding"]["stable_model_binding"]["muscle_channel_core_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="muscle_channel_core_fingerprint is stale"):
        validate_anatomical_taxonomy(_reseal(core_drift))


def test_exact_runtime_binding_rejects_full_model_hash_drift(runtime_model):
    payload = load_anatomical_taxonomy(AUDIT_V2).to_manifest()
    actual = payload["model_binding"]["compiled_runtime_audit"]["runtime_model_hash"]
    payload["model_binding"]["compiled_runtime_audit"]["runtime_model_hash"] = (
        "f" * 64 if actual != "f" * 64 else "e" * 64
    )
    drifted = validate_anatomical_taxonomy(_reseal(copy.deepcopy(payload)))
    with pytest.raises(ValueError, match="runtime MuJoCo model hash differs"):
        validate_taxonomy_against_model(
            drifted,
            runtime_model,
            compatibility=EXACT_RUNTIME_MODEL_COMPATIBILITY,
        )
