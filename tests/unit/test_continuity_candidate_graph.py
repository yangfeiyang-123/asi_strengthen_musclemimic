"""Independent topology-review and candidate-graph contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from analysis.physiology_synergy.build_candidate_continuity_graph import (
    build_candidate_continuity_graph,
)
from analysis.physiology_synergy.review_continuity_topology import (
    TOPOLOGY_REVIEW_SCHEMA_VERSION,
    seal_topology_review,
    topology_review_fingerprint,
    validate_topology_review,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    CONTINUITY_LOSS_METHOD,
    build_continuity_loss_spec,
    continuity_graph_fingerprint,
    load_fascicle_continuity_graph,
    validate_candidate_continuity_graph,
    validate_fascicle_continuity_graph,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
TOPOLOGY_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"
BATCH_A = {
    "right_external_oblique_continuity",
    "right_internal_oblique_continuity",
    "left_external_oblique_continuity",
    "left_internal_oblique_continuity",
}


def _assets():
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    topology = load_fascicle_continuity_graph(TOPOLOGY_PATH, taxonomy=taxonomy)
    return taxonomy, topology


def _review_draft(*, independent: bool = True):
    taxonomy, topology = _assets()
    chains = []
    for chain in topology.chains:
        approved = chain["chain_id"] in BATCH_A
        chains.append(
            {
                "chain_id": chain["chain_id"],
                "approve_as_training_candidate": approved,
                "checks": {
                    "exact_asset_topology_reviewed": True,
                    "same_side_verified": True,
                    "adjacent_level_definition_reviewed": True,
                    "not_hard_line_equivalence": True,
                    "baseline_activation_distribution_reviewed": approved,
                    "deadband_data_supported": approved,
                },
                "approved_deadband": (
                    0.12 if chain["chain_id"] == "right_external_oblique_continuity" else chain["deadband"]
                ),
                "approved_edge_weights": list(chain["edge_weights"]),
                "approved_chain_weight": chain["chain_weight"],
                "approved_activity_off": chain["activity_off"],
                "approved_activity_on": chain["activity_on"],
                "provenance": (
                    [
                        {
                            "kind": "independent_anatomical_review",
                            "reference": f"fixture:{chain['chain_id']}",
                        }
                    ]
                    if approved
                    else []
                ),
            }
        )
    return {
        "schema_version": TOPOLOGY_REVIEW_SCHEMA_VERSION,
        "source_graph_fingerprint": topology.graph_fingerprint,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "reviewer": {
            "name": "Independent Fixture Reviewer",
            "affiliation_or_role": "biomechanics reviewer",
            "reviewed_at_utc": "2026-07-30T00:00:00Z",
            "independent_of_code_author": independent,
        },
        "chains": chains,
    }


def _sealed_review():
    taxonomy, topology = _assets()
    return seal_topology_review(
        _review_draft(),
        source_graph=topology,
        taxonomy=taxonomy,
    )


def test_topology_review_is_independent_complete_and_calibration_free():
    taxonomy, topology = _assets()
    review = _sealed_review()
    assert review["schema_version"] == TOPOLOGY_REVIEW_SCHEMA_VERSION
    assert "calibration_fingerprint" not in review
    assert [item["chain_id"] for item in review["chains"] if item["approve_as_training_candidate"]] == [
        chain_id for chain_id in topology.chain_ids if chain_id in BATCH_A
    ]
    assert (
        validate_topology_review(
            review,
            source_graph=topology,
            taxonomy=taxonomy,
        )
        == review
    )


def test_topology_review_rejects_nonindependence_omissions_and_unsupported_data():
    taxonomy, topology = _assets()
    with pytest.raises(ValueError, match="independent review"):
        seal_topology_review(
            _review_draft(independent=False),
            source_graph=topology,
            taxonomy=taxonomy,
        )

    missing = _review_draft()
    missing["chains"].pop()
    with pytest.raises(ValueError, match="cover every source chain"):
        seal_topology_review(
            missing,
            source_graph=topology,
            taxonomy=taxonomy,
        )

    incomplete = _sealed_review()
    incomplete["chains"][0]["checks"]["deadband_data_supported"] = False
    incomplete["review_fingerprint"] = topology_review_fingerprint(incomplete)
    with pytest.raises(ValueError, match="incomplete candidate review"):
        validate_topology_review(
            incomplete,
            source_graph=topology,
            taxonomy=taxonomy,
        )


def test_candidate_graph_freezes_reviewed_parameters_before_calibration():
    taxonomy, topology = _assets()
    review = _sealed_review()
    manifest = build_candidate_continuity_graph(
        taxonomy_path=TAXONOMY_PATH,
        source_graph_path=TOPOLOGY_PATH,
        topology_review=review,
    )
    candidate = validate_fascicle_continuity_graph(manifest, taxonomy=taxonomy)
    validate_candidate_continuity_graph(
        candidate,
        taxonomy,
        expected_review_fingerprint=review["review_fingerprint"],
        source_graph=topology,
    )
    assert topology.training_enabled_chain_count == 0
    assert candidate.training_enabled_chain_count == 4
    enabled = [chain for chain in candidate.chains if chain["training_enabled"]]
    assert all(chain["review_status"] == "verified_candidate" for chain in enabled)
    assert enabled[0]["deadband"] == pytest.approx(0.12)

    _, identity = build_continuity_loss_spec(
        candidate,
        taxonomy,
        training_enabled_only=True,
        signal="activation",
        method=CONTINUITY_LOSS_METHOD,
        scale=0.05,
        huber_delta=1.0,
    )
    assert identity.chain_count == 4
    assert identity.edge_count == 20


def test_candidate_validator_rejects_manual_training_graph_without_review_lineage():
    taxonomy, topology = _assets()
    payload = copy.deepcopy(topology.to_manifest())
    payload["chains"][0]["review_status"] = "verified_candidate"
    payload["chains"][0]["training_enabled"] = True
    payload["chains"][0]["provenance"] = [{"kind": "manual_edit", "reference": "not-a-review-artifact"}]
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    manual = validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    with pytest.raises(ValueError, match="not a reviewed candidate graph"):
        validate_candidate_continuity_graph(manual, taxonomy)
