"""Exact canonical continuity loss-spec identity contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    CONTINUITY_LOSS_METHOD,
    build_continuity_loss_spec,
    continuity_graph_fingerprint,
    continuity_loss_spec_fingerprint,
    load_fascicle_continuity_graph,
    validate_continuity_loss_spec_identity,
    validate_fascicle_continuity_graph,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"


def _assets():
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    return taxonomy, graph


def _candidate_graph():
    taxonomy, topology = _assets()
    payload = topology.to_manifest()
    chain = payload["chains"][0]
    chain["review_status"] = "verified"
    chain["training_enabled"] = True
    chain["provenance"] = [
        {
            "kind": "independent_topology_review",
            "reference": "fixture-review-v2",
        }
    ]
    payload["graph_id"] = "fixture_candidate_graph"
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    return taxonomy, validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)


def _build(graph, taxonomy, *, training_enabled_only: bool, scale: float = 0.05):
    return build_continuity_loss_spec(
        graph,
        taxonomy,
        training_enabled_only=training_enabled_only,
        signal="activation",
        method=CONTINUITY_LOSS_METHOD,
        scale=scale,
        huber_delta=1.0,
    )


def test_diagnostic_loss_spec_fingerprints_every_chain_and_array_semantic():
    taxonomy, graph = _assets()
    spec, identity = _build(graph, taxonomy, training_enabled_only=False)
    assert identity.chain_count == 28
    assert identity.edge_count == 140
    assert identity.chain_ids == graph.chain_ids
    assert spec.chain_ids == identity.chain_ids
    assert int(np.asarray(spec.edge_mask).sum()) == identity.edge_count
    first = identity.chains[0]
    np.testing.assert_array_equal(
        np.asarray(spec.member_indices[0, : len(first["member_indices"])]),
        first["member_indices"],
    )
    np.testing.assert_array_equal(
        np.asarray(spec.edge_indices[0, : len(first["edge_indices"])]),
        first["edge_indices"],
    )
    _, repeated = _build(graph, taxonomy, training_enabled_only=False)
    assert repeated.to_manifest() == identity.to_manifest()
    _, changed_scale = _build(
        graph,
        taxonomy,
        training_enabled_only=False,
        scale=0.06,
    )
    assert changed_scale.loss_spec_fingerprint != identity.loss_spec_fingerprint


def test_training_loss_spec_is_nonempty_verified_and_exactly_filtered():
    taxonomy, topology = _assets()
    with pytest.raises(ValueError, match="cannot be empty"):
        _build(topology, taxonomy, training_enabled_only=True)

    taxonomy, candidate = _candidate_graph()
    spec, identity = _build(candidate, taxonomy, training_enabled_only=True)
    assert identity.chain_ids == ("right_external_oblique_continuity",)
    assert identity.chain_count == 1
    assert identity.edge_count == 5
    assert spec.chain_ids == identity.chain_ids


def test_loss_spec_manifest_rejects_stale_or_internally_inconsistent_values():
    taxonomy, candidate = _candidate_graph()
    _, identity = _build(candidate, taxonomy, training_enabled_only=True)
    stale = identity.to_manifest()
    stale["scale"] = 0.06
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_continuity_loss_spec_identity(stale)

    inconsistent = identity.to_manifest()
    inconsistent["chains"][0]["edge_indices"][0][0] += 1
    inconsistent["loss_spec_fingerprint"] = continuity_loss_spec_fingerprint(inconsistent)
    with pytest.raises(ValueError, match="names and indices disagree"):
        validate_continuity_loss_spec_identity(inconsistent)


def test_loss_spec_fingerprint_changes_with_final_graph_numerics():
    taxonomy, candidate = _candidate_graph()
    _, original = _build(candidate, taxonomy, training_enabled_only=True)
    payload = copy.deepcopy(candidate.to_manifest())
    payload["chains"][0]["deadband"] = 0.16
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    changed_graph = validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    _, changed = _build(changed_graph, taxonomy, training_enabled_only=True)
    assert changed.loss_spec_fingerprint != original.loss_spec_fingerprint
