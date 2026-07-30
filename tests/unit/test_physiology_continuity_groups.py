"""Source-only contracts for curated anatomy and continuity assets."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    build_fascicle_continuity_spec,
    continuity_graph_fingerprint,
    load_fascicle_continuity_graph,
    resolve_fascicle_continuity_reward_gate,
    validate_fascicle_continuity_graph,
)
from scripts.build_myofullbody_curated_taxonomy import (
    EXPECTED_AUDIT_FINGERPRINT,
    build_curated_taxonomy,
)
from scripts.build_myofullbody_fascicle_continuity import (
    EXPECTED_TAXONOMY_FINGERPRINT,
    build_continuity_graph,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v1.json"
CURATED_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v1.json"
GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v1.json"


def _assets():
    taxonomy = load_anatomical_taxonomy(CURATED_PATH)
    graph = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    return taxonomy, graph


def _reseal(payload: dict) -> dict:
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    return payload


def test_curated_taxonomy_is_deterministic_nonhard_and_diagnostics_only():
    audit = load_anatomical_taxonomy(AUDIT_PATH)
    curated = load_anatomical_taxonomy(CURATED_PATH)
    assert audit.fingerprint == EXPECTED_AUDIT_FINGERPRINT
    assert build_curated_taxonomy(audit) == curated.to_manifest()
    assert curated.generation["parent_taxonomy_fingerprint"] == audit.fingerprint
    assert curated.actuator_names == audit.actuator_names
    assert len(curated.actuator_names) == 354
    assert curated.hard_line_groups == ()
    assert len(curated.soft_compartment_groups) == 24
    assert all(group["training_enabled"] is False for group in curated.soft_compartment_groups)
    hard_members = {member for group in curated.hard_line_groups for member in group["members"]}
    assert not hard_members & {
        "DELT1",
        "DELT2",
        "DELT3",
        "PECM1",
        "PECM2",
        "PECM3",
        "LAT1",
        "LAT2",
        "LAT3",
    }
    assert all(not name.startswith(("FDS", "FDP", "EDC")) for name in curated.actuator_names)


def test_checked_in_graph_is_deterministic_nonempty_and_diagnostics_only():
    taxonomy, graph = _assets()
    assert taxonomy.fingerprint == EXPECTED_TAXONOMY_FINGERPRINT
    assert (
        build_continuity_graph(
            taxonomy,
            expected_taxonomy_fingerprint=EXPECTED_TAXONOMY_FINGERPRINT,
        )
        == graph.to_manifest()
    )
    assert len(graph.chains) == 28
    assert graph.edge_count == 140
    assert graph.training_enabled_chain_count == 0
    assert all(chain["review_status"] == "provisional" for chain in graph.chains)
    assert all(chain["training_enabled"] is False for chain in graph.chains)
    spec = build_fascicle_continuity_spec(graph, taxonomy)
    assert spec.edge_indices.shape == (28, 11, 2)
    assert int(spec.edge_mask.sum()) == 140
    assert spec.activation_addresses.shape == (354,)
    assert resolve_fascicle_continuity_reward_gate(graph, enabled=False)[0] is False
    with pytest.raises(ValueError, match="no verified training-enabled chains"):
        resolve_fascicle_continuity_reward_gate(graph, enabled=True)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["chains"][0].update(
                members=["EO1_r", "unknown"],
                edges=[["EO1_r", "unknown"]],
                edge_weights=[1.0],
            ),
            "unknown members",
        ),
        (
            lambda payload: payload["chains"][0].update(
                members=["EO1_r", "EO1_l"],
                edges=[["EO1_r", "EO1_l"]],
                edge_weights=[1.0],
            ),
            "mixes or misdeclares sides",
        ),
        (
            lambda payload: payload["chains"][0].update(
                members=["EO1_r", "EO2_r"],
                edges=[["EO1_r", "EO1_r"]],
                edge_weights=[1.0],
            ),
            "self-edge",
        ),
        (
            lambda payload: payload["chains"][0].update(
                members=["EO1_r", "EO2_r"],
                edges=[["EO1_r", "EO2_r"], ["EO2_r", "EO1_r"]],
                edge_weights=[1.0, 1.0],
            ),
            "repeats an undirected edge",
        ),
        (
            lambda payload: payload["chains"][0].update(training_enabled=True),
            "requires verified review and provenance",
        ),
        (
            lambda payload: payload["chains"][0].update(review_status="unknown"),
            "review_status must be provisional or verified",
        ),
    ],
)
def test_graph_rejects_invalid_members_edges_and_training_promotion(mutator, message):
    taxonomy, graph = _assets()
    payload = copy.deepcopy(graph.to_manifest())
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_fascicle_continuity_graph(_reseal(payload), taxonomy=taxonomy)


def test_graph_rejects_stale_or_wrong_taxonomy_binding():
    taxonomy, graph = _assets()
    stale = graph.to_manifest()
    stale["notes"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_fascicle_continuity_graph(stale, taxonomy=taxonomy)

    wrong = copy.deepcopy(graph.to_manifest())
    wrong["taxonomy_binding"]["ordered_muscle_schema_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="ordered_muscle_schema_sha256"):
        validate_fascicle_continuity_graph(_reseal(wrong), taxonomy=taxonomy)


def test_training_chain_requires_nonempty_verified_provenance():
    taxonomy, graph = _assets()
    payload = copy.deepcopy(graph.to_manifest())
    chain = payload["chains"][0]
    chain["review_status"] = "verified"
    chain["training_enabled"] = True
    chain["provenance"] = []
    with pytest.raises(ValueError, match="requires verified review and provenance"):
        validate_fascicle_continuity_graph(_reseal(payload), taxonomy=taxonomy)
