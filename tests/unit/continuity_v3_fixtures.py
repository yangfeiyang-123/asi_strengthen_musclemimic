"""Deterministic source-only fixtures for continuity baseline/calibration v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.physiology_synergy.build_candidate_continuity_graph import (
    build_candidate_continuity_graph,
)
from analysis.physiology_synergy.calibrate_continuity_reward import (
    build_baseline_rollout_evidence,
)
from analysis.physiology_synergy.collect_continuity_baseline import (
    _TRAJECTORY_ARRAY_FIELDS,
    CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
    CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
    _trajectory_manifest_digest,
    build_rollout_identity,
)
from analysis.physiology_synergy.review_continuity_topology import (
    TOPOLOGY_REVIEW_SCHEMA_VERSION,
    seal_topology_review,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    CONTINUITY_LOSS_METHOD,
    build_continuity_loss_spec,
    load_fascicle_continuity_graph,
    validate_fascicle_continuity_graph,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
DIAGNOSTIC_GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"
TARGET_CHAIN_IDS = {
    "right_external_oblique_continuity",
    "right_internal_oblique_continuity",
    "left_external_oblique_continuity",
    "left_internal_oblique_continuity",
}


def candidate_assets():
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    diagnostic = load_fascicle_continuity_graph(
        DIAGNOSTIC_GRAPH_PATH,
        taxonomy=taxonomy,
    )
    draft = {
        "schema_version": TOPOLOGY_REVIEW_SCHEMA_VERSION,
        "source_graph_fingerprint": diagnostic.graph_fingerprint,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "reviewer": {
            "name": "Independent Fixture Reviewer",
            "affiliation_or_role": "source-only test fixture",
            "reviewed_at_utc": "2026-07-30T00:00:00Z",
            "independent_of_code_author": True,
        },
        "chains": [],
    }
    for chain in diagnostic.chains:
        approved = chain["chain_id"] in TARGET_CHAIN_IDS
        draft["chains"].append(
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
                "approved_deadband": chain["deadband"],
                "approved_edge_weights": list(chain["edge_weights"]),
                "approved_chain_weight": chain["chain_weight"],
                "approved_activity_off": chain["activity_off"],
                "approved_activity_on": chain["activity_on"],
                "provenance": (
                    [{"kind": "fixture_review", "reference": f"fixture:{chain['chain_id']}"}] if approved else []
                ),
            }
        )
    review = seal_topology_review(
        draft,
        source_graph=diagnostic,
        taxonomy=taxonomy,
    )
    candidate_payload = build_candidate_continuity_graph(
        taxonomy_path=TAXONOMY_PATH,
        source_graph_path=DIAGNOSTIC_GRAPH_PATH,
        topology_review=review,
    )
    candidate = validate_fascicle_continuity_graph(
        candidate_payload,
        taxonomy=taxonomy,
    )
    _, loss_identity = build_continuity_loss_spec(
        candidate,
        taxonomy,
        training_enabled_only=True,
        signal="activation",
        method=CONTINUITY_LOSS_METHOD,
        scale=0.05,
        huber_delta=1.0,
    )
    return taxonomy, diagnostic, review, candidate, loss_identity


def rollout_identity_and_manifest():
    taxonomy, diagnostic, _review, candidate, loss_identity = candidate_assets()
    environment = {
        "schema_version": CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
        "resolved_env_params": {"disable_fingers": True},
        "resolved_task_factory": {"name": "fixture"},
        "resolved_action_representation": {"enabled": False},
        "runtime_model_hash": "1" * 64,
        "muscle_channel_core_fingerprint": "2" * 64,
        "ordered_actuator_schema_hash": "3" * 64,
        "ordered_action_dim": 354,
        "control_dt": 0.01,
        "horizon": 32,
    }
    environment["environment_fingerprint"] = json_fingerprint(environment)
    arrays = {
        field: {
            "dtype": "<f4",
            "shape": [1],
            "sha256": hashlib.sha256(field.encode("utf-8")).hexdigest(),
        }
        for field in _TRAJECTORY_ARRAY_FIELDS
    }
    heldout = {
        "schema_version": CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
        "ordered_motion_paths": ["heldout/motion-0", "heldout/motion-1"],
        "trajectory_count": 2,
        "trajectory_lengths": [32, 32],
        "frequency_hz": 100.0,
        "array_manifest": arrays,
        "trajectory_content_sha256": _trajectory_manifest_digest(arrays),
    }
    heldout["heldout_split_fingerprint"] = json_fingerprint(heldout)
    identity, manifest = build_rollout_identity(
        config={
            "experiment": {
                "run_id": "continuity-v3-fixture",
                "action_representation": {"enabled": False},
            }
        },
        checkpoint_identity={
            "run_id": "continuity-v3-fixture",
            "config_hash": "fixture-config-hash",
            "checkpoint_content_sha256": "a" * 64,
        },
        promoted_artifact={
            "binding_sha256": "b" * 64,
            "checkpoint": {"checkpoint_content_sha256": "a" * 64},
        },
        taxonomy=taxonomy,
        diagnostic_graph=diagnostic,
        candidate_graph=candidate,
        target_loss_identity=loss_identity,
        environment_manifest=environment,
        heldout_split_manifest=heldout,
        backend="mjx",
        deterministic=True,
        eval_seed=0,
        num_envs=2,
    )
    return identity, manifest, loss_identity


def raw_samples(*, target_all_zero: bool = False) -> dict:
    trajectory_index = np.repeat(np.arange(2), 32)
    trajectory_step = np.tile(np.arange(32), 2)
    target = np.zeros(64) if target_all_zero else np.linspace(0.02, 0.12, 64)
    return {
        "trajectory_index": trajectory_index.tolist(),
        "trajectory_step": trajectory_step.tolist(),
        "reward_imitation_total": np.linspace(0.5, 1.5, 64).tolist(),
        "continuity_global_loss": np.linspace(0.01, 0.20, 64).tolist(),
        "continuity_target_loss": target.tolist(),
        "continuity_global_chain_count": [28] * 64,
        "continuity_global_edge_count": [140] * 64,
        "continuity_target_chain_count": [4] * 64,
        "continuity_target_edge_count": [20] * 64,
    }


def baseline_evidence(*, target_all_zero: bool = False):
    identity, manifest, loss_identity = rollout_identity_and_manifest()
    baseline = build_baseline_rollout_evidence(
        identity,
        raw_samples(target_all_zero=target_all_zero),
        expected_trajectory_count=2,
        expected_global_chain_count=28,
        expected_global_edge_count=140,
        expected_target_chain_count=4,
        expected_target_edge_count=20,
    )
    return baseline, manifest, loss_identity


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
